//! Arena-based cf dialect.

#[trunk_ir::dialect]
mod cf {
    fn br(#[rest] args: ()) {
        #[successor(dest)]
        {}
    }

    fn cond_br(cond: ()) {
        #[successor(then_dest)]
        {}
        #[successor(else_dest)]
        {}
    }
}

use crate::IrContext;
use crate::op_interface::{
    BranchOps, BranchSuccessor, BranchSuccessors, ControlFlowInterfaceError,
};
use crate::ops::DialectOp;
use crate::refs::OpRef;

fn br_successors(
    ctx: &IrContext,
    op: OpRef,
) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
    if ctx.op(op).successors.len() != 1 {
        return Err(ControlFlowInterfaceError::new(
            "cf.br requires exactly one successor",
        ));
    }
    let branch = Br::from_op(ctx, op)
        .map_err(|error| ControlFlowInterfaceError::new(format!("malformed cf.br: {error:?}")))?;
    Ok(BranchSuccessors::new([BranchSuccessor::new(
        branch.dest(ctx),
        branch.args(ctx).iter().copied(),
    )]))
}

fn cond_br_successors(
    ctx: &IrContext,
    op: OpRef,
) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
    if ctx.op_operands(op).len() != 1 || ctx.op(op).successors.len() != 2 {
        return Err(ControlFlowInterfaceError::new(
            "cf.cond_br requires one condition and exactly two successors",
        ));
    }
    let branch = CondBr::from_op(ctx, op).map_err(|error| {
        ControlFlowInterfaceError::new(format!("malformed cf.cond_br: {error:?}"))
    })?;
    Ok(BranchSuccessors::new([
        BranchSuccessor::new(branch.then_dest(ctx), []),
        BranchSuccessor::new(branch.else_dest(ctx), []),
    ]))
}

inventory::submit! {
    BranchOps::register("cf", "br", br_successors)
}

inventory::submit! {
    BranchOps::register("cf", "cond_br", cond_br_successors)
}
