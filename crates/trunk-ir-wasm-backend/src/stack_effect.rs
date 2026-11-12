//! What an operation's instruction leaves on the Wasm stack.
//!
//! The order in which an instruction takes its operands is the `wasm`
//! dialect's [`StackOperandOrder`](trunk_ir::dialect::wasm::StackOperandOrder).
//! Which results it leaves on the stack is this backend's representation
//! rule: a call's `core.nil` result slots, a `wasm.nop` of a type with no Wasm
//! null, and structured control whose result is `core.nil` or whose regions
//! yield no value leave nothing (`design/wasm-backend.md`, "명시적 지역
//! 변수와 stack 계약"). Where a `wasm.br` or `wasm.yield` passes its values
//! follows from that (`design/wasm-backend.md`, "분기와 region 끝의 값").
//! `ExplicitLocals`, the stack contract check, and the emitter all read it
//! here.

use trunk_ir::IrContext;
use trunk_ir::dialect::wasm;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef, ValueDef, ValueRef};
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
/// A `wasm.if` has a result only when one of its regions passes one. The
/// other may end the function instead, as an arm ending in a trap does.
pub(crate) fn structured_result_type(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    let ty = ctx
        .op_result_types(op)
        .first()
        .copied()
        .filter(|&ty| !is_nil_type(ctx, ty))?;
    if wasm::If::matches(ctx, op)
        && !ctx
            .op_regions(op)
            .any(|region| region_result_value(ctx, region).is_some())
    {
        return None;
    }
    Some(ty)
}

/// The value a region passes to its parent: the first value of the
/// `wasm.yield` that ends it.
pub(crate) fn region_result_value(ctx: &IrContext, region: RegionRef) -> Option<ValueRef> {
    let &block = ctx.region(region).blocks.last()?;
    let &op = ctx.block(block).ops.last()?;
    let yield_op = wasm::Yield::from_op(ctx, op).ok()?;
    yield_op.values(ctx).first().copied()
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

/// The local of a function parameter: `value`'s position when it is an
/// argument of a `wasm.func` body's entry block.
pub(crate) fn parameter_local(ctx: &IrContext, value: ValueRef) -> Option<u32> {
    let ValueDef::BlockArg(block, index) = ctx.value_def(value) else {
        return None;
    };
    let region = ctx.block(block).parent_region?;
    let func = ctx.region(region).parent_op?;
    (wasm::Func::matches(ctx, func) && ctx.region(region).blocks.first() == Some(&block))
        .then_some(index)
}

/// The function parameters `op` reads right before it: the last stack
/// operands that are parameters, in the order its instruction takes them.
pub(crate) fn trailing_parameters(ctx: &IrContext, op: OpRef) -> SmallVec<[u32; 4]> {
    let operands = wasm::DynStackOperandOrder::of(ctx, op);
    let mut locals: SmallVec<[u32; 4]> = operands
        .iter()
        .rev()
        .map_while(|&value| parameter_local(ctx, value))
        .collect();
    locals.reverse();
    locals
}

/// Where a `wasm.br` or `wasm.yield` passes its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ValueTransfer {
    /// On the stack, as the result its label's operation leaves there.
    Stack,
    /// To the body block arguments of the loop a branch continues.
    LoopArguments(wasm::Loop),
    /// Nowhere: the label's operation leaves no result on the stack.
    None,
}

/// Where `op` passes its values, when it is a `wasm.br` or a `wasm.yield`
/// inside the structured operation it names (`design/wasm-backend.md`,
/// "분기와 region 끝의 값").
pub(crate) fn value_transfer(ctx: &IrContext, op: OpRef) -> Option<ValueTransfer> {
    let label = if let Ok(br) = wasm::Br::from_op(ctx, op) {
        (0..=br.target(ctx)).try_fold(op, |inner, _| enclosing_structured(ctx, inner))?
    } else if wasm::Yield::matches(ctx, op) {
        enclosing_structured(ctx, op)?
    } else {
        return None;
    };
    if wasm::Br::matches(ctx, op)
        && let Ok(loop_op) = wasm::Loop::from_op(ctx, label)
    {
        return Some(ValueTransfer::LoopArguments(loop_op));
    }
    Some(match structured_result_type(ctx, label) {
        Some(_) => ValueTransfer::Stack,
        None => ValueTransfer::None,
    })
}

/// The `wasm.block`, `wasm.loop`, or `wasm.if` whose region holds `op`.
fn enclosing_structured(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    let parent = ctx.region(ctx.block(block).parent_region?).parent_op?;
    is_structured(ctx, parent).then_some(parent)
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
