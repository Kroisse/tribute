//! What an operation's instruction leaves on the Wasm stack.
//!
//! The order in which an instruction takes its operands is the `wasm`
//! dialect's [`StackOperandOrder`](trunk_ir::dialect::wasm::StackOperandOrder).
//! Which results it leaves on the stack is this backend's representation
//! rule: a call's `core.nil` result slots, a `wasm.nop` of a type with no Wasm
//! null, and structured control whose result is `core.nil` or whose then
//! region passes no value leave nothing (`design/wasm-backend.md`, "명시적
//! 지역 변수와 stack 계약"). `ExplicitLocals`, the stack contract check, and
//! the emitter all read it here.

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::smallvec::SmallVec;
use wasm_encoder::AbstractHeapType;

use crate::emit::helpers::is_nil_type;

/// The results `op` leaves on the stack, in the order it pushes them.
pub(crate) fn pushed_results(ctx: &IrContext, op: OpRef) -> SmallVec<[ValueRef; 4]> {
    let results = ctx.op_results(op);
    if wasm::Nop::matches(ctx, op) {
        return results
            .iter()
            .copied()
            .filter(|&result| nop_null_heap_type(ctx, ctx.value_ty(result)).is_some())
            .collect();
    }
    if wasm::Call::matches(ctx, op) || wasm::CallIndirect::matches(ctx, op) {
        return results
            .iter()
            .copied()
            .filter(|&result| !is_nil_type(ctx, ctx.value_ty(result)))
            .collect();
    }
    if is_structured(ctx, op) {
        return structured_result_type(ctx, op)
            .map(|_| ctx.op_result(op, 0))
            .into_iter()
            .collect();
    }
    results.iter().copied().collect()
}

/// Whether `op` is a `wasm.block`, `wasm.loop`, or `wasm.if`.
pub(crate) fn is_structured(ctx: &IrContext, op: OpRef) -> bool {
    wasm::Block::matches(ctx, op) || wasm::Loop::matches(ctx, op) || wasm::If::matches(ctx, op)
}

/// The result type of a structured operation's Wasm block type, or `None`
/// when the block type is empty.
///
/// A `wasm.if` has a result only when its then region passes one.
pub(crate) fn structured_result_type(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    let ty = ctx
        .op_result_types(op)
        .first()
        .copied()
        .filter(|&ty| !is_nil_type(ctx, ty))?;
    if wasm::If::matches(ctx, op) {
        let then_region = ctx.op_region(op, 0)?;
        region_result_value(ctx, then_region)?;
    }
    Some(ty)
}

/// The value a region passes to its parent: the operand of its last
/// operation when that is a `wasm.yield`, or else the first result of its
/// last operation.
pub(crate) fn region_result_value(ctx: &IrContext, region: RegionRef) -> Option<ValueRef> {
    let &block = ctx.region(region).blocks.last()?;
    let &op = ctx.block(block).ops.last()?;
    if let Ok(yield_op) = wasm::Yield::from_op(ctx, op) {
        return Some(yield_op.value(ctx));
    }
    ctx.op_results(op).first().copied()
}

/// The abstract heap type of the null a `wasm.nop` of type `ty` pushes, or
/// `None` when it pushes nothing.
pub(crate) fn nop_null_heap_type(ctx: &IrContext, ty: TypeRef) -> Option<AbstractHeapType> {
    if wasm::FuncSig::matches(ctx, ty) || wasm::Funcref::matches(ctx, ty) {
        Some(AbstractHeapType::Func)
    } else if wasm::Anyref::matches(ctx, ty) || wasm::Structref::matches(ctx, ty) {
        Some(AbstractHeapType::Any)
    } else if is_nil_type(ctx, ty) {
        Some(AbstractHeapType::None)
    } else {
        None
    }
}

/// The stack operands of `op` in the order its instruction takes them, each
/// with its operand position. A value an operation uses twice appears once
/// for each position.
pub(crate) fn stack_operand_slots(ctx: &IrContext, op: OpRef) -> SmallVec<[(ValueRef, u32); 4]> {
    let operands = ctx.op_operands(op);
    let mut taken = SmallVec::<[bool; 4]>::from_elem(false, operands.len());
    wasm::DynStackOperandOrder::of(ctx, op)
        .into_iter()
        .map(|value| {
            let slot = operands
                .iter()
                .enumerate()
                .position(|(slot, &operand)| operand == value && !taken[slot])
                .expect("a stack operand is one of the operation's operands");
            taken[slot] = true;
            (value, slot as u32)
        })
        .collect()
}

/// Where a `wasm.yield` passes its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum YieldTransfer {
    /// On the stack, to the region's parent or the label of the `wasm.br`
    /// after it.
    Stack,
    /// To the only body block argument of the loop the `wasm.br` after it
    /// continues.
    LoopArgument,
    /// To each of the other number of body block arguments of the loop the
    /// `wasm.br` after it continues.
    LoopArguments(usize),
    /// Nowhere: the parent takes no value and no branch follows.
    None,
}

/// Where the `wasm.yield` at `position` of `ops` passes its value. `loop_args`
/// holds, for each enclosing structured operation, innermost last, the
/// number of body block arguments of a loop or `None`; `passes_result` tells
/// whether the region's parent takes its value.
pub(crate) fn yield_transfer(
    ctx: &IrContext,
    ops: &[OpRef],
    position: usize,
    loop_args: &[Option<usize>],
    passes_result: bool,
) -> YieldTransfer {
    let branch_depth = ops
        .get(position + 1)
        .and_then(|&next| wasm::Br::from_op(ctx, next).ok())
        .map(|br| br.target(ctx) as usize);
    match branch_depth {
        Some(depth) => match loop_args
            .len()
            .checked_sub(depth + 1)
            .and_then(|label| loop_args[label])
        {
            Some(1) => YieldTransfer::LoopArgument,
            Some(count) => YieldTransfer::LoopArguments(count),
            None => YieldTransfer::Stack,
        },
        None if passes_result && position + 1 == ops.len() => YieldTransfer::Stack,
        None => YieldTransfer::None,
    }
}

/// Whether nothing after `op` in its block runs: an unconditional branch,
/// a return, a tail call, or `wasm.unreachable`. Values left on the stack
/// before it are discarded.
pub(crate) fn ends_block(ctx: &IrContext, op: OpRef) -> bool {
    wasm::Br::matches(ctx, op)
        || wasm::BrTable::matches(ctx, op)
        || wasm::Return::matches(ctx, op)
        || wasm::ReturnCall::matches(ctx, op)
        || wasm::ReturnCallIndirect::matches(ctx, op)
        || wasm::Unreachable::matches(ctx, op)
}
