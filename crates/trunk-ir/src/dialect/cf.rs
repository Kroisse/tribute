//! Arena-based cf dialect.

#[trunk_ir::dialect]
mod cf {
    fn br(args: Variadic<_>) {
        #[successor(dest)]
        {}
    }

    fn cond_br(cond: Value<_>) {
        #[successor(then_dest)]
        {}
        #[successor(else_dest)]
        {}
    }

    /// A multi-way branch on an integer discriminant.
    ///
    /// Control transfers to the `targets` successor at the position where
    /// `cases` holds the discriminant's value, or to `default` when no case
    /// matches. A case is a value of the discriminant's type. Successors
    /// receive no arguments.
    #[verify]
    fn switch<T: IntegerLike>(cases: Attr<[i64]>, discriminant: Value<T>) {
        #[successor(default)]
        {}
        #[successors(targets)]
        {}
    }
}

use crate::IrContext;
use crate::dialect::core::IntegerLike;
use crate::op_interface::{
    BranchModel, BranchOps, BranchSuccessor, BranchSuccessors, ControlFlowInterfaceError,
};

impl BranchModel for Br {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        if ctx.op_successor_count(self.op_ref()) != 1 {
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
        if ctx.op_operands(op).len() != 1 || ctx.op_successor_count(op) != 2 {
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

impl crate::ops::Verify for Switch {
    /// Every case has one target, and no case value repeats.
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let cases = self.cases(ctx).count();
        let targets = self.targets(ctx).count();
        if cases != targets {
            return Err(format!("{cases} case(s) but {targets} target(s)"));
        }
        let mut seen = rustc_hash::FxHashSet::default();
        match self.cases(ctx).find(|&case| !seen.insert(case)) {
            Some(case) => Err(format!("duplicate case {case}")),
            None => Ok(()),
        }
    }
}

impl BranchModel for Switch {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        let op = self.op_ref();
        if ctx.op_operands(op).len() != 1 || !ctx.op_has_successors(op) {
            return Err(ControlFlowInterfaceError::new(
                "cf.switch requires one discriminant and a default successor",
            ));
        }
        Ok(BranchSuccessors::new(
            ctx.op_successors(op)
                .map(|block| BranchSuccessor::new(block, [])),
        ))
    }
}

inventory::submit! {
    BranchOps::register::<Switch>()
}
