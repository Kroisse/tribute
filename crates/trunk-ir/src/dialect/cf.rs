//! Arena-based cf dialect.

#[trunk_ir::dialect]
mod cf {
    #[implements(Branch)]
    #[asm_format(args "[" successor(dest) "]" attr_dict_bare)]
    fn br(args: Variadic<_>) {
        #[successor(dest)]
        {}
    }

    #[implements(Branch)]
    #[asm_format(cond "[" successor(then_dest) "," successor(else_dest) "]" attr_dict_bare)]
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
    ///
    /// ```text
    /// cf.switch %flag : core.i32, [default: ^bb1, 42: ^bb2, -1: ^bb3]
    /// ```
    #[verify]
    #[implements(Branch)]
    #[asm_format(discriminant ":" type(discriminant) "," "[" custom(SwitchCases(default, cases, targets)) "]" attr_dict)]
    fn switch(cases: Attr<[i64]>, discriminant: Value<impl IntegerLike>) {
        #[successor(default)]
        {}
        #[successors(targets)]
        {}
    }
}

use crate::dialect::core::IntegerLike;
use crate::op_interface::{Branch, BranchSuccessor, BranchSuccessors, ControlFlowInterfaceError};
use crate::{Attribute, BlockRef, IrContext};

impl Branch for Br {
    fn successors(&self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
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

impl Branch for CondBr {
    fn successors(&self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
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
    fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
        let cases: Vec<i64> = self.cases(ctx).collect();
        let targets = self.targets(ctx).count();
        if cases.len() != targets {
            return Err(format!("{} case(s) but {targets} target(s)", cases.len()).into());
        }
        let width = IntegerLike::width(ctx, ctx.value_ty(self.discriminant(ctx)))
            .expect("schema-verified integer discriminant");
        Ok(Self::check_cases(width, &cases)?)
    }
}

/// `default: ^d, 0: ^a, 5: ^b`: the default successor, then each case with
/// its target.
pub struct SwitchCases;

impl crate::asm_format::OpAsmDirective for SwitchCases {
    type Parts<'a> = (BlockRef, Vec<(i64, BlockRef)>);

    fn decompose<'a>(
        _ctx: &'a IrContext,
        entities: &'a [crate::asm_format::CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        use crate::asm_format::CustomEntity;
        let [
            CustomEntity::Successors(default),
            CustomEntity::Attribute(Some(Attribute::List(cases))),
            CustomEntity::Successors(targets),
        ] = entities
        else {
            return None;
        };
        let [default] = default.as_slice() else {
            return None;
        };
        if cases.len() != targets.len() {
            return None;
        }
        let arms = cases
            .iter()
            .zip(targets)
            .map(|(case, &target)| match *case {
                Attribute::Int(case) => Some((i64::try_from(case).ok()?, target)),
                _ => None,
            })
            .collect::<Option<_>>()?;
        Some((*default, arms))
    }

    fn print(
        p: &mut crate::printer::OpAsmPrinter<'_>,
        (default, arms): Self::Parts<'_>,
    ) -> std::fmt::Result {
        use std::fmt::Write as _;

        let s = p.state();
        write!(p, "default: {}", s.block(default))?;
        for (case, target) in arms {
            write!(p, ", {case}: {}", s.block(target))?;
        }
        Ok(())
    }

    fn parse(
        p: &mut crate::parser::OpAsmParser<'_>,
    ) -> crate::parser::ParseResult<Vec<crate::asm_format::ParsedEntity>> {
        use crate::asm_format::ParsedEntity;

        p.parse_keyword("default")?;
        p.parse_punct(":")?;
        let default = p.parse_successor()?;
        let mut cases = Vec::new();
        let mut targets = Vec::new();
        while p.parse_optional_punct(",") {
            cases.push(Attribute::Int(p.parse_integer()?));
            p.parse_punct(":")?;
            targets.push(p.parse_successor()?);
        }
        Ok(vec![
            ParsedEntity::Successors(vec![default]),
            ParsedEntity::Attribute(Some(Attribute::List(cases))),
            ParsedEntity::Successors(targets),
        ])
    }
}

impl Branch for Switch {
    fn successors(&self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
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

#[cfg(test)]
mod tests {
    use crate::IrContext;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;

    #[test]
    fn switch_writes_its_type_and_default_first_as_mlir_does() {
        let text = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.nil {
      cf.switch %0 : core.i32, [default: ^bb1, 42: ^bb2, -1: ^bb3]
    ^bb1:
      func.return
    ^bb2:
      func.return
    ^bb3:
      cf.switch %0 : core.i32, [default: ^bb1]
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(print_module(&ctx, module.op()), text);
    }
}
