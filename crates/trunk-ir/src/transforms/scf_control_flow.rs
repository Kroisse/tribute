//! Conservative terminal-region queries shared by SCF target lowerings.

use crate::context::IrContext;
use crate::dialect::scf;
use crate::op_interface::{CallableExitOps, RegionBranchOps, RegionBranchPoint, RegionSuccessor};
use crate::ops::DialectOp;
use crate::refs::{OpRef, RegionRef};
use crate::symbol::Symbol;

/// Whether a one-block region has only terminal structured control or a
/// registered callable exit, so lowering it cannot need a continuation block.
pub fn is_terminal_region(ctx: &IrContext, region: RegionRef) -> bool {
    let [branch] = ctx.region(region).blocks.as_slice() else {
        return false;
    };
    let Some(&terminator) = ctx.block(*branch).ops.last() else {
        return false;
    };
    if scf::If::matches(ctx, terminator) {
        has_terminal_unused_never_result(ctx, terminator)
    } else if scf::Switch::matches(ctx, terminator) {
        ctx.op_results(terminator).is_empty()
            && has_only_terminal_region_successors(ctx, terminator)
    } else {
        !scf::Loop::matches(ctx, terminator)
            && CallableExitOps::exits_callable(ctx, terminator).is_ok()
    }
}

/// Whether every semantic entry successor of a structured operation is a
/// terminal region. Missing, incomplete, or parent-returning mappings are
/// nonterminal by construction.
pub fn has_only_terminal_region_successors(ctx: &IrContext, op: OpRef) -> bool {
    let Some(interface) = RegionBranchOps::get(ctx, op) else {
        return false;
    };
    let Ok(successors) = interface.successors(ctx, op, RegionBranchPoint::Parent) else {
        return false;
    };
    !successors.as_slice().is_empty()
        && successors.as_slice().iter().all(|successor| {
            matches!(successor, RegionSuccessor::Region(region) if is_terminal_region(ctx, *region))
        })
}

/// Whether structured control has exactly one unused Never result and no
/// continuation: it ends its block and all entry successors are terminal.
pub fn has_terminal_unused_never_result(ctx: &IrContext, scf_op: OpRef) -> bool {
    let Some(block) = ctx.op(scf_op).parent_block else {
        return false;
    };
    let [result] = ctx.op_results(scf_op) else {
        return false;
    };
    let result_ty = ctx.types.get(ctx.value_ty(*result));
    if result_ty.dialect != Symbol::new("core")
        || result_ty.name != Symbol::new("never")
        || ctx.has_uses(*result)
        || ctx.block(block).ops.last() != Some(&scf_op)
    {
        return false;
    }

    has_only_terminal_region_successors(ctx, scf_op)
}
