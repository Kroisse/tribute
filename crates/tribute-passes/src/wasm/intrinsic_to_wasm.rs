//! Bind the `extern "C"` byte array helpers to WasmGC operations past the
//! representation/ABI boundary.
//!
//! `__tribute_byte_array_range_equal` and `__tribute_byte_array_concat`
//! become operations on the byte array's GC array, which a byte array keeps
//! in values and signatures (`design/wasm-backend.md` § "Structural GC 타입과
//! descriptor 필드"). A range that is not inside its array traps: an
//! `array.get_u` or `array.copy` past the end traps, and a range whose end
//! wraps around `i32` traps before any read.

use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::core;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::wasm_gc;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, OpRewritePattern, PatternApplicator, PatternRewriter, TypeConverter,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::Location;

/// C link name of the byte array concatenation helper.
pub const BYTE_ARRAY_CONCAT: &str = "__tribute_byte_array_concat";
/// C link name of the byte array range comparison helper.
pub const BYTE_ARRAY_RANGE_EQUAL: &str = "__tribute_byte_array_range_equal";

/// Bind calls to the `extern "C"` byte array helpers to WasmGC operations.
pub(crate) fn lower(ctx: &mut IrContext, module: Module) {
    let symbols = SymbolTable::collect(ctx, module);
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(RangeEqualPattern(&symbols))
        .add_pattern(ConcatPattern(&symbols));

    applicator.apply_partial(ctx, module);
}

/// Whether `call` calls the C helper `helper`: its callee resolves to a
/// bodyless `abi = "C"` declaration with that link name.
fn calls_c_helper(
    ctx: &IrContext,
    symbols: &SymbolTable,
    call: wasm_dialect::Call,
    helper: &'static str,
) -> bool {
    super::runtime_bindings::c_helper(ctx, symbols, call.op_ref(), call.callee(ctx))
        .is_some_and(|name| name == helper)
}

/// The GC array of the byte array.
fn byte_array_type(ctx: &mut IrContext) -> TypeRef {
    let array = tribute_ir::runtime_layout::byte_array(ctx);
    super::type_converter::gc_array_type(ctx, array)
}

/// Trap if `start + len` wraps around `i32`. Returns the operations, in order.
fn trap_if_range_wraps(
    ctx: &mut IrContext,
    location: Location,
    start: ValueRef,
    len: ValueRef,
) -> Vec<OpRef> {
    let i32_ty = core::I32::type_ref(ctx);
    let nil_ty = core::Nil::params().intern(ctx).as_type_ref();
    let end = wasm_dialect::I32Add::operands(start, len).build(ctx, location);
    let wrapped = wasm_dialect::I32LtU::operands(end.result(ctx), start)
        .results(i32_ty)
        .build(ctx, location);
    let trap = trap_region(ctx, location);
    let ok = empty_region(ctx, location);
    let check = wasm_dialect::If::operands(wrapped.result(ctx))
        .results([nil_ty])
        .regions(trap, ok)
        .build(ctx, location);
    vec![end.op_ref(), wrapped.op_ref(), check.op_ref()]
}

/// Trap unless `[start, start + len)` lies within `array`, as the native
/// helpers check before reading (`design/rc.md` § "Bytes"). Returns the
/// operations, in order.
fn trap_if_out_of_range(
    ctx: &mut IrContext,
    location: Location,
    array: ValueRef,
    start: ValueRef,
    len: ValueRef,
) -> Vec<OpRef> {
    let i32_ty = core::I32::type_ref(ctx);
    let nil_ty = core::Nil::params().intern(ctx).as_type_ref();
    let mut ops = trap_if_range_wraps(ctx, location, start, len);
    let end = ctx.op_result(ops[0], 0);
    let array_len = wasm_dialect::ArrayLen::operands(array).build(ctx, location);
    let past_end = wasm_dialect::I32GtU::operands(end, array_len.result(ctx))
        .results(i32_ty)
        .build(ctx, location);
    let trap = trap_region(ctx, location);
    let ok = empty_region(ctx, location);
    let check = wasm_dialect::If::operands(past_end.result(ctx))
        .results([nil_ty])
        .regions(trap, ok)
        .build(ctx, location);
    ops.extend([array_len.op_ref(), past_end.op_ref(), check.op_ref()]);
    ops
}

/// Pattern for comparing equal-length ranges of two byte arrays.
///
/// The generated Wasm loop compares bytes directly in the two arrays and
/// exits at the first mismatch. String equality invokes this once per pair of
/// contiguous rope-leaf spans rather than once per logical byte.
struct RangeEqualPattern<'a>(&'a SymbolTable);

impl OpRewritePattern for RangeEqualPattern<'_> {
    type Op = wasm_dialect::Call;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        call: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = call.op_ref();
        if !calls_c_helper(ctx, self.0, call, BYTE_ARRAY_RANGE_EQUAL) {
            return None;
        }

        let operands = ctx.op_operands(op).to_vec();
        let &[left, left_start, right, right_start, len] = operands.as_array()?;
        let location = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let i32_ty = core::I32::type_ref(ctx);
        let nil_ty = core::Nil::params().intern(ctx).as_type_ref();
        let array_ty = byte_array_type(ctx);
        // Both whole ranges are checked first, since the loop may stop at
        // the first mismatch before reading past an array.
        let mut checks = trap_if_out_of_range(ctx, location, left, left_start, len);
        checks.extend(trap_if_out_of_range(ctx, location, right, right_start, len));
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

        let read = |ctx: &mut IrContext, array: ValueRef, start: ValueRef| {
            let at = wasm_dialect::I32Add::operands(start, index).build(ctx, location);
            ctx.push_op(loop_block, at.op_ref());
            let byte = wasm_gc::ArrayGetU::operands(array, at.result(ctx))
                .r#type(array_ty)
                .results(i32_ty)
                .build(ctx, location);
            ctx.push_op(loop_block, byte.op_ref());
            byte.result(ctx)
        };
        let left_byte = read(ctx, left, left_start);
        let right_byte = read(ctx, right, right_start);
        let mismatch = wasm_dialect::I32Ne::operands(left_byte, right_byte)
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
        let yield_next = wasm_dialect::Yield::operands([next.result(ctx)]).build(ctx, location);
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

        for check in checks {
            rewriter.insert_op(check);
        }
        rewriter.insert_op(zero.op_ref());
        rewriter.replace_op(compare.op_ref());
        Some(())
    }

    fn name(&self) -> &'static str {
        "ByteArrayRangeEqualPattern"
    }
}

pub(super) fn empty_region(ctx: &mut IrContext, location: Location) -> trunk_ir::RegionRef {
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
    location: Location,
    i32_ty: TypeRef,
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
    let yield_value = wasm_dialect::Yield::operands([value.result(ctx)]).build(ctx, location);
    ctx.push_op(block, yield_value.op_ref());
    let break_outer = wasm_dialect::Br::operands().target(2).build(ctx, location);
    ctx.push_op(block, break_outer.op_ref());
    ctx.create_region(RegionData {
        location,
        blocks: smallvec![block],
        parent_op: None,
    })
}

/// Pattern for `__tribute_byte_array_concat(left, ls, ll, right, rs, rl)`:
/// allocate a new byte array and copy both ranges into it. `array.copy`
/// traps on a range outside its source.
struct ConcatPattern<'a>(&'a SymbolTable);

impl OpRewritePattern for ConcatPattern<'_> {
    type Op = wasm_dialect::Call;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        call: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = call.op_ref();
        if !calls_c_helper(ctx, self.0, call, BYTE_ARRAY_CONCAT) {
            return None;
        }

        let operands = ctx.op_operands(op).to_vec();
        let &[left, left_start, left_len, right, right_start, right_len] = operands.as_array()?;
        let location = ctx.op(op).location;
        let i32_ty = core::I32::type_ref(ctx);
        let array_ty = byte_array_type(ctx);

        // A total that wraps would allocate too few bytes; trap instead.
        let mut ops = trap_if_range_wraps(ctx, location, left_len, right_len);
        let total = wasm_dialect::I32Add::operands(left_len, right_len).build(ctx, location);
        let joined = wasm_gc::ArrayNewDefault::operands(total.result(ctx))
            .r#type(array_ty)
            .build(ctx, location);
        let zero = wasm_dialect::I32Const::operands()
            .value(0)
            .results(i32_ty)
            .build(ctx, location);
        let copy = |ctx: &mut IrContext, at: ValueRef, src: ValueRef, start, len| {
            wasm_gc::ArrayCopy::operands(joined.result(ctx), at, src, start, len)
                .dst_type(array_ty)
                .src_type(array_ty)
                .build(ctx, location)
                .op_ref()
        };
        let copy_left = copy(ctx, zero.result(ctx), left, left_start, left_len);
        let copy_right = copy(ctx, left_len, right, right_start, right_len);
        ops.extend([
            total.op_ref(),
            joined.op_ref(),
            zero.op_ref(),
            copy_left,
            copy_right,
        ]);
        for inserted in ops {
            rewriter.insert_op(inserted);
        }
        rewriter.erase_op(vec![joined.result(ctx)]);
        Some(())
    }

    fn name(&self) -> &'static str {
        "ByteArrayConcatPattern"
    }
}

pub(super) fn trap_region(ctx: &mut IrContext, location: Location) -> trunk_ir::RegionRef {
    let region = empty_region(ctx, location);
    let block = ctx.region(region).blocks[0];
    let unreachable = wasm_dialect::Unreachable::operands().build(ctx, location);
    ctx.push_op(block, unreachable.op_ref());
    region
}
