//! Bind the `extern "C"` bytes helpers to WasmGC operations past the
//! representation/ABI boundary.
//!
//! `__tribute_bytes_len`, `__tribute_bytes_range_equal`,
//! `__tribute_bytes_concat`, and `__tribute_bytes_slice_or_panic` become
//! struct/array operations on the bytes layout. The bytes element read intrinsic is lowered inside the boundary by
//! `wasm/bytes.rs`.

use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::core;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::TypeDataBuilder;

use trunk_ir_wasm_backend::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX};

use super::bytes::{
    DATA_FIELD as BYTES_DATA_FIELD, LEN_FIELD as BYTES_LEN_FIELD,
    OFFSET_FIELD as BYTES_OFFSET_FIELD,
};

/// Extracted Bytes struct fields: (data, offset, len) values.
struct BytesFields {
    data: ValueRef,
    offset: ValueRef,
    len: ValueRef,
}

/// Extract (data, offset, len) fields from a Bytes struct value.
///
/// Returns the extracted field values and the operations that produced them.
fn extract_bytes_fields(
    ctx: &mut IrContext,
    location: trunk_ir::types::Location,
    bytes_value: ValueRef,
) -> (BytesFields, Vec<OpRef>) {
    let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    let array_ref_ty = super::bytes::bytes_data_type(ctx);

    let get_data = wasm_dialect::StructGet::operands(bytes_value)
        .type_idx(BYTES_STRUCT_IDX)
        .field_idx(BYTES_DATA_FIELD)
        .results(array_ref_ty)
        .build(ctx, location);

    let get_offset = wasm_dialect::StructGet::operands(bytes_value)
        .type_idx(BYTES_STRUCT_IDX)
        .field_idx(BYTES_OFFSET_FIELD)
        .results(i32_ty)
        .build(ctx, location);

    let get_len = wasm_dialect::StructGet::operands(bytes_value)
        .type_idx(BYTES_STRUCT_IDX)
        .field_idx(BYTES_LEN_FIELD)
        .results(i32_ty)
        .build(ctx, location);

    let fields = BytesFields {
        data: get_data.result(ctx),
        offset: get_offset.result(ctx),
        len: get_len.result(ctx),
    };

    let ops = vec![get_data.op_ref(), get_offset.op_ref(), get_len.op_ref()];

    (fields, ops)
}

/// C link name of the bytes length helper.
pub const BYTES_LEN: &str = "__tribute_bytes_len";
/// C link name of the bytes concatenation helper.
pub const BYTES_CONCAT: &str = "__tribute_bytes_concat";
/// C link name of the bytes range comparison helper.
pub const BYTES_RANGE_EQUAL: &str = "__tribute_bytes_range_equal";
/// C link name of the bounds-checked bytes slicing helper.
pub const BYTES_SLICE_OR_PANIC: &str = "__tribute_bytes_slice_or_panic";

/// Bind calls to the `extern "C"` bytes helpers to WasmGC operations.
pub fn lower(ctx: &mut IrContext, module: Module) {
    let symbols = Rc::new(SymbolTable::collect(ctx, module));
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(BytesLenPattern(Rc::clone(&symbols)))
        .add_pattern(BytesRangeEqualPattern(Rc::clone(&symbols)))
        .add_pattern(BytesConcatPattern(Rc::clone(&symbols)))
        .add_pattern(BytesSliceOrPanicPattern(symbols));

    applicator.apply_partial(ctx, module);
}

// =============================================================================
// Bytes helper patterns
// =============================================================================

/// Whether `op` calls the C helper `helper`: its callee resolves to a
/// bodyless `abi = "C"` declaration with that link name.
fn calls_c_helper(ctx: &IrContext, symbols: &SymbolTable, op: OpRef, helper: &'static str) -> bool {
    wasm_dialect::Call::from_op(ctx, op).is_ok_and(|call| {
        super::runtime_bindings::c_helper(ctx, symbols, call.callee(ctx))
            .is_some_and(|name| name == Symbol::new(helper))
    })
}

/// Pattern for `__tribute_bytes_len(bytes)` -> `struct.get $bytes 2`
///
/// Returns i32 directly since Nat is mapped to i32.
struct BytesLenPattern(Rc<SymbolTable>);

impl RewritePattern for BytesLenPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !calls_c_helper(ctx, &self.0, op, BYTES_LEN) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        let Some(bytes_ref) = operands.first().copied() else {
            return false;
        };

        let location = ctx.op(op).location;
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());

        // struct.get to get len field (field 2)
        let get_len = wasm_dialect::StructGet::operands(bytes_ref)
            .type_idx(BYTES_STRUCT_IDX)
            .field_idx(BYTES_LEN_FIELD)
            .results(i32_ty)
            .build(ctx, location);

        rewriter.replace_op(get_len.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesLenPattern"
    }
}

/// Pattern for comparing equal-length ranges in two Bytes values.
///
/// The generated Wasm loop compares bytes directly in the two backing arrays
/// and exits at the first mismatch. String equality invokes this once per pair
/// of contiguous rope-leaf spans rather than once per logical byte.
struct BytesRangeEqualPattern(Rc<SymbolTable>);

impl RewritePattern for BytesRangeEqualPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !calls_c_helper(ctx, &self.0, op, BYTES_RANGE_EQUAL) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() != 5 {
            return false;
        }
        let location = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let nil_ty = core::nil(ctx).as_type_ref();
        let (left, left_ops) = extract_bytes_fields(ctx, location, operands[0]);
        let (right, right_ops) = extract_bytes_fields(ctx, location, operands[2]);
        let left_start =
            wasm_dialect::I32Add::operands(left.offset, operands[1]).build(ctx, location);
        let right_start =
            wasm_dialect::I32Add::operands(right.offset, operands[3]).build(ctx, location);
        let len = operands[4];
        let zero = wasm_dialect::I32Const::operands()
            .value(0)
            .results(i32_ty)
            .build(ctx, location);

        let loop_block = ctx.create_block(BlockData {
            location,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let index = ctx.block_arg(loop_block, 0);

        let done = wasm_dialect::I32GeU::operands(index, len)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(loop_block, done.op_ref());
        let done_then = value_break_region(ctx, location, i32_ty, 1);
        let done_else = empty_region(ctx, location);
        let break_when_done = wasm_dialect::If::operands(done.result(ctx))
            .results([nil_ty])
            .regions(done_then, done_else)
            .build(ctx, location);
        ctx.push_op(loop_block, break_when_done.op_ref());

        let left_index =
            wasm_dialect::I32Add::operands(left_start.result(ctx), index).build(ctx, location);
        ctx.push_op(loop_block, left_index.op_ref());
        let left_byte = wasm_dialect::ArrayGetU::operands(left.data, left_index.result(ctx))
            .type_idx(BYTES_ARRAY_IDX)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(loop_block, left_byte.op_ref());
        let right_index =
            wasm_dialect::I32Add::operands(right_start.result(ctx), index).build(ctx, location);
        ctx.push_op(loop_block, right_index.op_ref());
        let right_byte = wasm_dialect::ArrayGetU::operands(right.data, right_index.result(ctx))
            .type_idx(BYTES_ARRAY_IDX)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(loop_block, right_byte.op_ref());
        let mismatch = wasm_dialect::I32Ne::operands(left_byte.result(ctx), right_byte.result(ctx))
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(loop_block, mismatch.op_ref());
        let mismatch_then = value_break_region(ctx, location, i32_ty, 0);
        let mismatch_else = empty_region(ctx, location);
        let break_on_mismatch = wasm_dialect::If::operands(mismatch.result(ctx))
            .results([nil_ty])
            .regions(mismatch_then, mismatch_else)
            .build(ctx, location);
        ctx.push_op(loop_block, break_on_mismatch.op_ref());

        let one = wasm_dialect::I32Const::operands()
            .value(1)
            .results(i32_ty)
            .build(ctx, location);
        ctx.push_op(loop_block, one.op_ref());
        let next = wasm_dialect::I32Add::operands(index, one.result(ctx)).build(ctx, location);
        ctx.push_op(loop_block, next.op_ref());
        let yield_next = wasm_dialect::Yield::operands(next.result(ctx)).build(ctx, location);
        ctx.push_op(loop_block, yield_next.op_ref());
        let continue_loop = wasm_dialect::Br::operands().target(0).build(ctx, location);
        ctx.push_op(loop_block, continue_loop.op_ref());

        let loop_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![loop_block],
            parent_op: None,
        });
        let compare_loop = wasm_dialect::Loop::operands([zero.result(ctx)])
            .results([result_ty])
            .regions(loop_region)
            .build(ctx, location);
        let outer_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![compare_loop.op_ref()],
            parent_region: None,
        });
        let outer_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![outer_block],
            parent_op: None,
        });
        let compare = wasm_dialect::Block::operands()
            .results([result_ty])
            .regions(outer_region)
            .build(ctx, location);

        for field_op in left_ops.into_iter().chain(right_ops) {
            rewriter.insert_op(field_op);
        }
        rewriter.insert_op(left_start.op_ref());
        rewriter.insert_op(right_start.op_ref());
        rewriter.insert_op(zero.op_ref());
        rewriter.replace_op(compare.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesRangeEqualPattern"
    }
}

fn empty_region(ctx: &mut IrContext, location: trunk_ir::types::Location) -> trunk_ir::RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

fn value_break_region(
    ctx: &mut IrContext,
    location: trunk_ir::types::Location,
    i32_ty: trunk_ir::TypeRef,
    value: i32,
) -> trunk_ir::RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let value = wasm_dialect::I32Const::operands()
        .value(value)
        .results(i32_ty)
        .build(ctx, location);
    ctx.push_op(block, value.op_ref());
    let yield_value = wasm_dialect::Yield::operands(value.result(ctx)).build(ctx, location);
    ctx.push_op(block, yield_value.op_ref());
    let break_outer = wasm_dialect::Br::operands().target(2).build(ctx, location);
    ctx.push_op(block, break_outer.op_ref());
    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

/// Pattern for `__tribute_bytes_concat(left, right)` -> allocate a new array and copy both
struct BytesConcatPattern(Rc<SymbolTable>);

impl RewritePattern for BytesConcatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !calls_c_helper(ctx, &self.0, op, BYTES_CONCAT) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 2 {
            return false;
        }
        let left = operands[0];
        let right = operands[1];

        let location = ctx.op(op).location;
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let bytes_ty = core::bytes(ctx).as_type_ref();
        let array_ref_ty = super::bytes::bytes_data_type(ctx);

        // Extract fields from left and right Bytes structs
        let (left_fields, left_ops) = extract_bytes_fields(ctx, location, left);
        let (right_fields, right_ops) = extract_bytes_fields(ctx, location, right);

        // Calculate total_len = left.len + right.len
        let total_len =
            wasm_dialect::I32Add::operands(left_fields.len, right_fields.len).build(ctx, location);

        // Allocate new array: array_new_default(total_len)
        let new_array = wasm_dialect::ArrayNewDefault::operands(total_len.result(ctx))
            .type_idx(BYTES_ARRAY_IDX)
            .results(array_ref_ty)
            .build(ctx, location);

        // Copy left bytes: array_copy(new_arr, 0, left.data, left.offset, left.len)
        let zero = wasm_dialect::I32Const::operands()
            .value(0)
            .results(i32_ty)
            .build(ctx, location);

        let copy_left = wasm_dialect::ArrayCopy::operands(
            new_array.result(ctx),
            zero.result(ctx),
            left_fields.data,
            left_fields.offset,
            left_fields.len,
        )
        .dst_type_idx(BYTES_ARRAY_IDX)
        .src_type_idx(BYTES_ARRAY_IDX)
        .build(ctx, location);

        // Copy right bytes: array_copy(new_arr, left.len, right.data, right.offset, right.len)
        let copy_right = wasm_dialect::ArrayCopy::operands(
            new_array.result(ctx),
            left_fields.len,
            right_fields.data,
            right_fields.offset,
            right_fields.len,
        )
        .dst_type_idx(BYTES_ARRAY_IDX)
        .src_type_idx(BYTES_ARRAY_IDX)
        .build(ctx, location);

        // Create new Bytes struct: struct_new(new_arr, 0, total_len)
        let struct_new = wasm_dialect::StructNew::operands(vec![
            new_array.result(ctx),
            zero.result(ctx),
            total_len.result(ctx),
        ])
        .type_idx(BYTES_STRUCT_IDX)
        .results(bytes_ty)
        .build(ctx, location);

        // Combine all operations in order
        let mut ops = Vec::with_capacity(left_ops.len() + right_ops.len() + 6);
        ops.extend(left_ops);
        ops.extend(right_ops);
        ops.push(total_len.op_ref());
        ops.push(new_array.op_ref());
        ops.push(zero.op_ref());
        ops.push(copy_left.op_ref());
        ops.push(copy_right.op_ref());
        ops.push(struct_new.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }

    fn name(&self) -> &'static str {
        "BytesConcatPattern"
    }
}

/// Pattern for `__tribute_bytes_slice_or_panic(bytes, start, end)`: trap
/// unless `start <= end <= bytes.len`, then share the backing array in a new
/// Bytes struct.
struct BytesSliceOrPanicPattern(Rc<SymbolTable>);

impl RewritePattern for BytesSliceOrPanicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !calls_c_helper(ctx, &self.0, op, BYTES_SLICE_OR_PANIC) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        let [bytes, start, end] = operands[..] else {
            return false;
        };
        let location = ctx.op(op).location;
        let bytes_ty = core::bytes(ctx).as_type_ref();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let nil_ty = core::nil(ctx).as_type_ref();

        let (fields, field_ops) = extract_bytes_fields(ctx, location, bytes);
        let reversed = wasm_dialect::I32GtU::operands(start, end)
            .results(i32_ty)
            .build(ctx, location);
        let past_len = wasm_dialect::I32GtU::operands(end, fields.len)
            .results(i32_ty)
            .build(ctx, location);
        let out_of_range =
            wasm_dialect::I32Or::operands(reversed.result(ctx), past_len.result(ctx))
                .results(i32_ty)
                .build(ctx, location);
        let trap = trap_region(ctx, location);
        let ok = empty_region(ctx, location);
        let check = wasm_dialect::If::operands(out_of_range.result(ctx))
            .results([nil_ty])
            .regions(trap, ok)
            .build(ctx, location);
        let offset = wasm_dialect::I32Add::operands(fields.offset, start).build(ctx, location);
        let len = wasm_dialect::I32Sub::operands(end, start)
            .results(i32_ty)
            .build(ctx, location);
        let slice = wasm_dialect::StructNew::operands(vec![
            fields.data,
            offset.result(ctx),
            len.result(ctx),
        ])
        .type_idx(BYTES_STRUCT_IDX)
        .results(bytes_ty)
        .build(ctx, location);

        for field_op in field_ops {
            rewriter.insert_op(field_op);
        }
        for checked in [reversed.op_ref(), past_len.op_ref(), out_of_range.op_ref()] {
            rewriter.insert_op(checked);
        }
        rewriter.insert_op(check.op_ref());
        rewriter.insert_op(offset.op_ref());
        rewriter.insert_op(len.op_ref());
        rewriter.replace_op(slice.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesSliceOrPanicPattern"
    }
}

fn trap_region(ctx: &mut IrContext, location: trunk_ir::types::Location) -> trunk_ir::RegionRef {
    let region = empty_region(ctx, location);
    let block = ctx.region(region).blocks[0];
    let unreachable = wasm_dialect::Unreachable::operands().build(ctx, location);
    ctx.push_op(block, unreachable.op_ref());
    region
}
