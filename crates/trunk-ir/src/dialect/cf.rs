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
    BranchModel, BranchOps, BranchSuccessor, BranchSuccessors, ControlFlowInterfaceError,
};

impl BranchModel for Br {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        if ctx.op(self.op_ref()).successors.len() != 1 {
            return Err(ControlFlowInterfaceError::new(
                "cf.br requires exactly one successor",
            ));
        }
        Ok(BranchSuccessors::new([BranchSuccessor::new(
            self.dest(ctx),
            self.args(ctx).iter().copied(),
        )]))
    }
}

impl BranchModel for CondBr {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        let op = self.op_ref();
        if ctx.op_operands(op).len() != 1 || ctx.op(op).successors.len() != 2 {
            return Err(ControlFlowInterfaceError::new(
                "cf.cond_br requires one condition and exactly two successors",
            ));
        }
        Ok(BranchSuccessors::new([
            BranchSuccessor::new(self.then_dest(ctx), []),
            BranchSuccessor::new(self.else_dest(ctx), []),
        ]))
    }
}

inventory::submit! {
    BranchOps::register::<Br>()
}

inventory::submit! {
    BranchOps::register::<CondBr>()
}
