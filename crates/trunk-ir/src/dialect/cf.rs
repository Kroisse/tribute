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

impl Switch {
    /// Check that `cases` are distinct values of an integer type `width` bits
    /// wide. A case may be written as a signed or an unsigned value of that
    /// width; two spellings of one bit pattern are the same value.
    pub fn check_cases(width: u32, cases: &[i64]) -> Result<(), String> {
        let mut seen = rustc_hash::FxHashSet::default();
        for &case in cases {
            let pattern = case_pattern(width, case)
                .ok_or_else(|| format!("case {case} is not a value of a {width}-bit integer"))?;
            if !seen.insert(pattern) {
                return Err(format!("duplicate case {case}"));
            }
        }
        Ok(())
    }
}

/// The bit pattern of `case` as an integer `width` bits wide, if it is a
/// signed or an unsigned value of that width.
fn case_pattern(width: u32, case: i64) -> Option<i128> {
    if width >= 64 {
        return Some(i128::from(case));
    }
    let unsigned = 0..1i64 << width;
    let signed = -(1i64 << (width - 1))..0;
    (unsigned.contains(&case) || signed.contains(&case))
        .then(|| i128::from(case & ((1i64 << width) - 1)))
}

impl crate::ops::Verify for Switch {
    /// Every case has one target and is a distinct value of the
    /// discriminant's type.
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let cases: Vec<i64> = self.cases(ctx).collect();
        let targets = self.targets(ctx).count();
        if cases.len() != targets {
            return Err(format!("{} case(s) but {targets} target(s)", cases.len()));
        }
        let width = IntegerLike::width(ctx, ctx.value_ty(self.discriminant(ctx)))
            .expect("schema-verified integer discriminant");
        Self::check_cases(width, &cases)
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
