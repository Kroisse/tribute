//! Lower intrinsic calls to WASM operations.
//!
//! This pass transforms high-level intrinsic calls to low-level WASM instructions:
//! - `__bytes_len`, `__bytes_get_or_panic`, etc. -> WasmGC struct/array operations

use tribute_ir::ModulePathExt;
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
use trunk_ir::types::TypeDataBuilder;

use trunk_ir_wasm_backend::gc_types::{BYTES_ARRAY_IDX, BYTES_STRUCT_IDX};

// Bytes struct field indices (must match gc_types layout)
const BYTES_DATA_FIELD: u32 = 0; // ref (array i8)
const BYTES_OFFSET_FIELD: u32 = 1; // i32
const BYTES_LEN_FIELD: u32 = 2; // i32

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
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let i8_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());
    let array_ty = core::array(ctx, i8_ty).as_type_ref();
    let array_ref_ty = core::r#ref(ctx, array_ty, false).as_type_ref();

    let get_data = wasm_dialect::struct_get(
        ctx,
        location,
        bytes_value,
        array_ref_ty,
        BYTES_STRUCT_IDX,
        BYTES_DATA_FIELD,
    );

    let get_offset = wasm_dialect::struct_get(
        ctx,
        location,
        bytes_value,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_OFFSET_FIELD,
    );

    let get_len = wasm_dialect::struct_get(
        ctx,
        location,
        bytes_value,
        i32_ty,
        BYTES_STRUCT_IDX,
        BYTES_LEN_FIELD,
    );

    let fields = BytesFields {
        data: get_data.result(ctx),
        offset: get_offset.result(ctx),
        len: get_len.result(ctx),
    };

    let ops = vec![get_data.op_ref(), get_offset.op_ref(), get_len.op_ref()];

    (fields, ops)
}

/// Lower target-independent Bytes intrinsic calls.
pub fn lower(ctx: &mut IrContext, module: Module) {
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(BytesLenPattern)
        .add_pattern(BytesGetOrPanicPattern)
        .add_pattern(BytesRangeEqualPattern)
        .add_pattern(BytesConcatPattern);

    applicator.apply_partial(ctx, module);
}

// =============================================================================
// Bytes intrinsic patterns
// =============================================================================

/// Check whether an operation calls one of the target-lowered Bytes helpers.
fn is_bytes_intrinsic_call(ctx: &IrContext, op: OpRef, intrinsic_names: &[&str]) -> bool {
    let Ok(call) = wasm_dialect::Call::from_op(ctx, op) else {
        return false;
    };
    let callee = call.callee(ctx).last_segment();
    intrinsic_names.iter().any(|name| callee == *name)
}

/// Pattern for `__bytes_len(bytes)` -> `struct.get $bytes 2`
///
/// Returns i32 directly since Nat is mapped to i32.
struct BytesLenPattern;

impl RewritePattern for BytesLenPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, &["__bytes_len", "__tribute_bytes_len"]) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        let Some(bytes_ref) = operands.first().copied() else {
            return false;
        };

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        // struct.get to get len field (field 2)
        let get_len = wasm_dialect::struct_get(
            ctx,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_LEN_FIELD,
        );

        rewriter.replace_op(get_len.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesLenPattern"
    }
}

/// Pattern for `Bytes::get_or_panic(bytes, index)` -> array access with offset
///
/// Index is i32 (Nat), returns i32 (Nat, byte value 0-255).
struct BytesGetOrPanicPattern;

impl RewritePattern for BytesGetOrPanicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, &["__bytes_get_or_panic"]) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 2 {
            return false;
        }
        let bytes_ref = operands[0];
        let index = operands[1]; // i32 (Nat)

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let i8_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());

        // Get data array ref (field 0)
        let array_ty = core::array(ctx, i8_ty).as_type_ref();
        let array_ref_ty = core::r#ref(ctx, array_ty, false).as_type_ref();
        let get_data = wasm_dialect::struct_get(
            ctx,
            location,
            bytes_ref,
            array_ref_ty,
            BYTES_STRUCT_IDX,
            BYTES_DATA_FIELD,
        );

        // Get offset (field 1)
        let get_offset = wasm_dialect::struct_get(
            ctx,
            location,
            bytes_ref,
            i32_ty,
            BYTES_STRUCT_IDX,
            BYTES_OFFSET_FIELD,
        );

        // Add offset to index: actual_index = offset + index
        let add_offset =
            wasm_dialect::i32_add(ctx, location, get_offset.result(ctx), index, i32_ty);

        // array.get_u (unsigned extend to i32, for byte values 0-255)
        let array_get = wasm_dialect::array_get_u(
            ctx,
            location,
            get_data.result(ctx),
            add_offset.result(ctx),
            i32_ty,
            BYTES_ARRAY_IDX,
        );

        rewriter.insert_op(get_data.op_ref());
        rewriter.insert_op(get_offset.op_ref());
        rewriter.insert_op(add_offset.op_ref());
        rewriter.replace_op(array_get.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesGetOrPanicPattern"
    }
}

/// Pattern for comparing equal-length ranges in two Bytes values.
///
/// The generated Wasm loop compares bytes directly in the two backing arrays
/// and exits at the first mismatch. String equality invokes this once per pair
/// of contiguous rope-leaf spans rather than once per logical byte.
struct BytesRangeEqualPattern;

impl RewritePattern for BytesRangeEqualPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, &["__tribute_bytes_range_equal"]) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() != 5 {
            return false;
        }
        let location = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let nil_ty = core::nil(ctx).as_type_ref();
        let (left, left_ops) = extract_bytes_fields(ctx, location, operands[0]);
        let (right, right_ops) = extract_bytes_fields(ctx, location, operands[2]);
        let left_start = wasm_dialect::i32_add(ctx, location, left.offset, operands[1], i32_ty);
        let right_start = wasm_dialect::i32_add(ctx, location, right.offset, operands[3], i32_ty);
        let len = operands[4];
        let zero = wasm_dialect::i32_const(ctx, location, i32_ty, 0);

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

        let done = wasm_dialect::i32_ge_u(ctx, location, index, len, i32_ty);
        ctx.push_op(loop_block, done.op_ref());
        let done_then = value_break_region(ctx, location, i32_ty, 1);
        let done_else = empty_region(ctx, location);
        let break_when_done = wasm_dialect::r#if(
            ctx,
            location,
            done.result(ctx),
            nil_ty,
            done_then,
            done_else,
        );
        ctx.push_op(loop_block, break_when_done.op_ref());

        let left_index =
            wasm_dialect::i32_add(ctx, location, left_start.result(ctx), index, i32_ty);
        ctx.push_op(loop_block, left_index.op_ref());
        let left_byte = wasm_dialect::array_get_u(
            ctx,
            location,
            left.data,
            left_index.result(ctx),
            i32_ty,
            BYTES_ARRAY_IDX,
        );
        ctx.push_op(loop_block, left_byte.op_ref());
        let right_index =
            wasm_dialect::i32_add(ctx, location, right_start.result(ctx), index, i32_ty);
        ctx.push_op(loop_block, right_index.op_ref());
        let right_byte = wasm_dialect::array_get_u(
            ctx,
            location,
            right.data,
            right_index.result(ctx),
            i32_ty,
            BYTES_ARRAY_IDX,
        );
        ctx.push_op(loop_block, right_byte.op_ref());
        let mismatch = wasm_dialect::i32_ne(
            ctx,
            location,
            left_byte.result(ctx),
            right_byte.result(ctx),
            i32_ty,
        );
        ctx.push_op(loop_block, mismatch.op_ref());
        let mismatch_then = value_break_region(ctx, location, i32_ty, 0);
        let mismatch_else = empty_region(ctx, location);
        let break_on_mismatch = wasm_dialect::r#if(
            ctx,
            location,
            mismatch.result(ctx),
            nil_ty,
            mismatch_then,
            mismatch_else,
        );
        ctx.push_op(loop_block, break_on_mismatch.op_ref());

        let one = wasm_dialect::i32_const(ctx, location, i32_ty, 1);
        ctx.push_op(loop_block, one.op_ref());
        let next = wasm_dialect::i32_add(ctx, location, index, one.result(ctx), i32_ty);
        ctx.push_op(loop_block, next.op_ref());
        let yield_next = wasm_dialect::r#yield(ctx, location, next.result(ctx));
        ctx.push_op(loop_block, yield_next.op_ref());
        let continue_loop = wasm_dialect::br(ctx, location, 0);
        ctx.push_op(loop_block, continue_loop.op_ref());

        let loop_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![loop_block],
            parent_op: None,
        });
        let compare_loop =
            wasm_dialect::r#loop(ctx, location, [zero.result(ctx)], result_ty, loop_region);
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
        let compare = wasm_dialect::block(ctx, location, result_ty, outer_region);

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
    let value = wasm_dialect::i32_const(ctx, location, i32_ty, value);
    ctx.push_op(block, value.op_ref());
    let yield_value = wasm_dialect::r#yield(ctx, location, value.result(ctx));
    ctx.push_op(block, yield_value.op_ref());
    let break_outer = wasm_dialect::br(ctx, location, 2);
    ctx.push_op(block, break_outer.op_ref());
    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

/// Pattern for `Bytes::concat(left, right)` -> allocate new array and copy both
struct BytesConcatPattern;

impl RewritePattern for BytesConcatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !is_bytes_intrinsic_call(ctx, op, &["__bytes_concat", "__tribute_bytes_concat"]) {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 2 {
            return false;
        }
        let left = operands[0];
        let right = operands[1];

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
        let i8_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build());
        let bytes_ty = core::bytes(ctx).as_type_ref();
        let array_ty = core::array(ctx, i8_ty).as_type_ref();
        let array_ref_ty = core::r#ref(ctx, array_ty, false).as_type_ref();

        // Extract fields from left and right Bytes structs
        let (left_fields, left_ops) = extract_bytes_fields(ctx, location, left);
        let (right_fields, right_ops) = extract_bytes_fields(ctx, location, right);

        // Calculate total_len = left.len + right.len
        let total_len =
            wasm_dialect::i32_add(ctx, location, left_fields.len, right_fields.len, i32_ty);

        // Allocate new array: array_new_default(total_len)
        let new_array = wasm_dialect::array_new_default(
            ctx,
            location,
            total_len.result(ctx),
            array_ref_ty,
            BYTES_ARRAY_IDX,
        );

        // Copy left bytes: array_copy(new_arr, 0, left.data, left.offset, left.len)
        let zero = wasm_dialect::i32_const(ctx, location, i32_ty, 0);

        let copy_left = wasm_dialect::array_copy(
            ctx,
            location,
            new_array.result(ctx),
            zero.result(ctx),
            left_fields.data,
            left_fields.offset,
            left_fields.len,
            BYTES_ARRAY_IDX,
            BYTES_ARRAY_IDX,
        );

        // Copy right bytes: array_copy(new_arr, left.len, right.data, right.offset, right.len)
        let copy_right = wasm_dialect::array_copy(
            ctx,
            location,
            new_array.result(ctx),
            left_fields.len,
            right_fields.data,
            right_fields.offset,
            right_fields.len,
            BYTES_ARRAY_IDX,
            BYTES_ARRAY_IDX,
        );

        // Create new Bytes struct: struct_new(new_arr, 0, total_len)
        let struct_new = wasm_dialect::struct_new(
            ctx,
            location,
            vec![
                new_array.result(ctx),
                zero.result(ctx),
                total_len.result(ctx),
            ],
            bytes_ty,
            BYTES_STRUCT_IDX,
        );

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
