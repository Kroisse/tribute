//! Arena-based scf dialect.

#[trunk_ir::dialect]
mod scf {
    #[asm_format(cond ("->" "(" type(result)^ ")")? attr_dict region(then_region) "else" region(else_region))]
    #[implements(RegionBranch, Fold)]
    #[flags(SelectsOneRegion)]
    fn r#if(cond: Value<_>) -> Option<Value<_>> {
        #[region(then_region)]
        {}
        #[region(else_region)]
        {}
    }

    /// A structured multi-way branch on an integer discriminant, as MLIR's
    /// `scf.index_switch`.
    ///
    /// ```text
    /// %r = scf.index_switch %x : (core.i32) -> core.i32
    /// case 0 {
    ///     scf.yield %a
    /// }
    /// default {
    ///     scf.yield %b
    /// }
    /// ```
    ///
    /// - **Shape:** a fixed-width integer discriminant, distinct `cases`, the
    ///   required `default_region`, one `case_regions` entry per case in the
    ///   same order, and any number of results. Unlike MLIR's, the
    ///   discriminant is not `index`.
    /// - **Meaning:** runs the case region whose value equals the
    ///   discriminant, or `default_region` when none does. Each region ends
    ///   in `scf.yield`, whose values become the results, or in its own
    ///   terminator.
    /// - **Verification:** one region per case, and no case twice.
    ///
    /// [`IndexSwitch::build_with_arms`] builds it from each case with its
    /// region.
    #[verify]
    #[asm_format(discriminant attr_dict ":" functional_type(discriminant, results) custom(SwitchArms(cases, case_regions, default_region)))]
    #[implements(RegionBranch, Fold)]
    #[flags(SelectsOneRegion)]
    fn index_switch(cases: Attr<[i64]>, discriminant: Value<impl IntegerLike>) -> Variadic<_> {
        #[region(default_region)]
        {}
        #[regions(case_regions)]
        {}
    }

    #[implements(RegionBranchTerminator)]
    #[asm_format(values attr_dict_bare)]
    fn r#yield(values: Variadic<_>) {}

    /// A loop whose body receives `init` as its entry block arguments on
    /// the first iteration and the values of `scf.continue` on the next.
    /// `scf.break` leaves it with the optional result.
    ///
    /// ```text
    /// %r = scf.loop(%init) -> (core.i32) {
    ///   ^body(%acc: core.i32):
    ///     scf.break %acc
    /// }
    /// ```
    #[asm_format("(" init ")" ("->" "(" type(result)^ ")")? attr_dict region(body))]
    #[implements(RegionBranch)]
    fn r#loop(init: Variadic<_>) -> Option<Value<_>> {
        #[region(body)]
        {}
    }

    #[implements(RegionBranchTerminator)]
    #[asm_format(values attr_dict_bare)]
    fn r#continue(values: Variadic<_>) {}

    #[implements(RegionBranchTerminator)]
    #[asm_format(value attr_dict_bare)]
    fn r#break(value: Value<_>) {}
}

// =========================================================================
// Control-flow interfaces
// =========================================================================

use crate::op_flag::SelectsOneRegion;
use crate::op_interface::{
    ControlFlowInterfaceError, ForwardedValues, RegionBranch, RegionBranchPoint,
    RegionBranchTerminator, RegionSuccessor, RegionSuccessors,
};

fn op_parent_region(ctx: &IrContext, op: OpRef) -> Option<crate::refs::RegionRef> {
    ctx.op(op)
        .parent_block
        .and_then(|block| ctx.block(block).parent_region)
}

impl Loop {
    fn enclosing(ctx: &IrContext, op: OpRef) -> Option<Self> {
        let mut region = op_parent_region(ctx, op);
        while let Some(current) = region {
            let owner = ctx.region(current).parent_op?;
            if let Ok(loop_op) = Self::from_op(ctx, owner) {
                return Some(loop_op);
            }
            region = op_parent_region(ctx, owner);
        }
        None
    }
}

fn immediate_region_owner(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    op_parent_region(ctx, op).and_then(|region| ctx.region(region).parent_op)
}

impl If {
    fn validate_region_shape(self, ctx: &IrContext) -> Result<(), ControlFlowInterfaceError> {
        let op = self.op_ref();
        if ctx.op_operands(op).len() != 1 || ctx.op_region_count(op) != 2 {
            return Err(ControlFlowInterfaceError::new(
                "scf.if requires one condition and exactly two regions",
            ));
        }
        Ok(())
    }
}

impl RegionBranch for If {
    fn successors(
        &self,
        ctx: &IrContext,
        point: RegionBranchPoint,
    ) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
        self.validate_region_shape(ctx)?;
        match point {
            RegionBranchPoint::Parent => Ok(RegionSuccessors::new([
                RegionSuccessor::Region(self.then_region(ctx)),
                RegionSuccessor::Region(self.else_region(ctx)),
            ])),
            RegionBranchPoint::Terminator(terminator)
                if Yield::matches(ctx, terminator)
                    && immediate_region_owner(ctx, terminator) == Some(self.op_ref()) =>
            {
                Ok(RegionSuccessors::new([RegionSuccessor::Parent]))
            }
            RegionBranchPoint::Terminator(terminator) => {
                Err(ControlFlowInterfaceError::not_applicable(format!(
                    "{terminator} is not an scf.if region yield"
                )))
            }
        }
    }

    fn entry_successor_operands(
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        self.validate_region_shape(ctx)?;
        match successor {
            RegionSuccessor::Region(region)
                if region == self.then_region(ctx) || region == self.else_region(ctx) =>
            {
                Ok(ForwardedValues::default())
            }
            _ => Err(ControlFlowInterfaceError::new(
                "successor is not an scf.if entry region",
            )),
        }
    }
}

impl Loop {
    fn validate_region_shape(self, ctx: &IrContext) -> Result<(), ControlFlowInterfaceError> {
        if ctx.op_region_count(self.op_ref()) != 1 {
            return Err(ControlFlowInterfaceError::new(
                "scf.loop requires exactly one body region",
            ));
        }
        Ok(())
    }
}

impl RegionBranch for Loop {
    fn successors(
        &self,
        ctx: &IrContext,
        point: RegionBranchPoint,
    ) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
        self.validate_region_shape(ctx)?;
        match point {
            RegionBranchPoint::Parent => Ok(RegionSuccessors::new([RegionSuccessor::Region(
                self.body(ctx),
            )])),
            RegionBranchPoint::Terminator(terminator)
                if Self::enclosing(ctx, terminator) == Some(*self)
                    && Continue::matches(ctx, terminator) =>
            {
                Ok(RegionSuccessors::new([RegionSuccessor::Region(
                    self.body(ctx),
                )]))
            }
            RegionBranchPoint::Terminator(terminator)
                if Self::enclosing(ctx, terminator) == Some(*self)
                    && Break::matches(ctx, terminator) =>
            {
                Ok(RegionSuccessors::new([RegionSuccessor::Parent]))
            }
            RegionBranchPoint::Terminator(terminator) => {
                Err(ControlFlowInterfaceError::not_applicable(format!(
                    "{terminator} is not a continue or break for this scf.loop"
                )))
            }
        }
    }

    fn entry_successor_operands(
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        self.validate_region_shape(ctx)?;
        match successor {
            RegionSuccessor::Region(region) if region == self.body(ctx) => {
                Ok(ForwardedValues::new(self.init(ctx).iter().copied()))
            }
            _ => Err(ControlFlowInterfaceError::new(
                "successor is not the scf.loop body",
            )),
        }
    }
}

impl IndexSwitch {
    /// Build a switch with `results` from each case value with its region,
    /// in order, and the default region, so a case cannot drift from its
    /// region.
    pub fn build_with_arms(
        ctx: &mut IrContext,
        location: crate::types::Location,
        discriminant: ValueRef,
        results: impl IntoIterator<Item = crate::refs::TypeRef>,
        arms: impl IntoIterator<Item = (i64, crate::refs::RegionRef)>,
        default_region: crate::refs::RegionRef,
    ) -> Self {
        let (cases, case_regions): (Vec<_>, Vec<_>) = arms.into_iter().unzip();
        Self::operands(discriminant)
            .cases(cases)
            .results(results)
            .regions(default_region, case_regions)
            .build(ctx, location)
    }

    /// The case regions in order, then the default region.
    pub fn arm_regions(self, ctx: &IrContext) -> crate::RegionList {
        let mut regions: crate::RegionList = self.case_regions(ctx).collect();
        regions.push(self.default_region(ctx));
        regions
    }
}

impl crate::ops::Verify for IndexSwitch {
    /// One region per case, and no case twice.
    fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
        let cases: Vec<i64> = self.cases(ctx).collect();
        let regions = self.case_regions(ctx).count();
        if cases.len() != regions {
            return Err(format!("{} case(s) but {regions} case region(s)", cases.len()).into());
        }
        let mut seen = rustc_hash::FxHashSet::default();
        if let Some(case) = cases.iter().find(|case| !seen.insert(**case)) {
            return Err(format!("case {case} appears twice").into());
        }
        Ok(())
    }
}

impl RegionBranch for IndexSwitch {
    fn successors(
        &self,
        ctx: &IrContext,
        point: RegionBranchPoint,
    ) -> Result<RegionSuccessors, ControlFlowInterfaceError> {
        let regions = self.arm_regions(ctx);
        match point {
            RegionBranchPoint::Parent => Ok(RegionSuccessors::new(
                regions.iter().copied().map(RegionSuccessor::Region),
            )),
            RegionBranchPoint::Terminator(terminator)
                if Yield::matches(ctx, terminator)
                    && op_parent_region(ctx, terminator)
                        .is_some_and(|region| regions.contains(&region)) =>
            {
                Ok(RegionSuccessors::new([RegionSuccessor::Parent]))
            }
            RegionBranchPoint::Terminator(terminator) => {
                Err(ControlFlowInterfaceError::not_applicable(format!(
                    "{terminator} is not a yield from this scf.index_switch"
                )))
            }
        }
    }

    fn entry_successor_operands(
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        match successor {
            RegionSuccessor::Region(region) if self.arm_regions(ctx).contains(&region) => {
                Ok(ForwardedValues::default())
            }
            _ => Err(ControlFlowInterfaceError::new(
                "successor is not an scf.index_switch region",
            )),
        }
    }
}

/// `case 0 {..}` on its own line for each case, then `default {..}`, as
/// MLIR's `scf.index_switch` writes them.
pub struct SwitchArms;

impl crate::asm_format::OpAsmDirective for SwitchArms {
    type Parts<'a> = (Vec<(i64, crate::refs::RegionRef)>, crate::refs::RegionRef);

    const SPACED: bool = false;

    fn decompose<'a>(
        _ctx: &'a IrContext,
        entities: &'a [crate::asm_format::CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>> {
        use crate::asm_format::CustomEntity;
        let [
            CustomEntity::Attribute(Some(Attribute::List(cases))),
            CustomEntity::Regions(regions),
            CustomEntity::Regions(default),
        ] = entities
        else {
            return None;
        };
        let [default] = default.as_slice() else {
            return None;
        };
        if cases.len() != regions.len() {
            return None;
        }
        let arms = cases
            .iter()
            .zip(regions)
            .map(|(case, &region)| match *case {
                Attribute::Int(case) => Some((i64::try_from(case).ok()?, region)),
                _ => None,
            })
            .collect::<Option<_>>()?;
        Some((arms, *default))
    }

    fn print(
        p: &mut crate::printer::OpAsmPrinter<'_>,
        (arms, default): Self::Parts<'_>,
    ) -> std::fmt::Result {
        use std::fmt::Write as _;

        for (case, region) in arms {
            p.newline()?;
            write!(p, "case {case} ")?;
            p.print_region(region)?;
        }
        p.newline()?;
        p.write_str("default ")?;
        p.print_region(default)
    }

    fn parse(
        p: &mut crate::parser::OpAsmParser<'_>,
    ) -> crate::parser::ParseResult<Vec<crate::asm_format::ParsedEntity>> {
        use crate::asm_format::ParsedEntity;

        let mut cases = Vec::new();
        let mut regions = Vec::new();
        while p.parse_optional_keyword("case") {
            cases.push(Attribute::Int(p.parse_integer()?));
            regions.push(p.parse_region()?);
        }
        p.parse_keyword("default")?;
        let default = p.parse_region()?;
        Ok(vec![
            ParsedEntity::Attribute(Some(Attribute::List(cases))),
            ParsedEntity::Regions(regions),
            ParsedEntity::Regions(vec![default]),
        ])
    }
}

impl RegionBranchTerminator for Yield {
    fn successor_operands(
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        match successor {
            RegionSuccessor::Parent => Ok(ForwardedValues::new(self.values(ctx).iter().copied())),
            RegionSuccessor::Region(_) => Err(ControlFlowInterfaceError::new(
                "scf.yield can only return to a parent",
            )),
        }
    }
}

impl RegionBranchTerminator for Continue {
    fn successor_operands(
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        match successor {
            RegionSuccessor::Region(region) => {
                let owner = Loop::enclosing(ctx, self.op_ref()).ok_or_else(|| {
                    ControlFlowInterfaceError::new("scf.continue has no enclosing scf.loop")
                })?;
                if region != owner.body(ctx) {
                    return Err(ControlFlowInterfaceError::new(
                        "successor is not the enclosing scf.loop body",
                    ));
                }
                Ok(ForwardedValues::new(self.values(ctx).iter().copied()))
            }
            RegionSuccessor::Parent => Err(ControlFlowInterfaceError::new(
                "scf.continue cannot exit its loop",
            )),
        }
    }
}

impl RegionBranchTerminator for Break {
    fn successor_operands(
        &self,
        ctx: &IrContext,
        successor: RegionSuccessor,
    ) -> Result<ForwardedValues, ControlFlowInterfaceError> {
        let op = self.op_ref();
        if ctx.op_operands(op).len() != 1 {
            return Err(ControlFlowInterfaceError::new(
                "scf.break requires exactly one value",
            ));
        }
        match successor {
            RegionSuccessor::Parent => {
                let owner = Loop::enclosing(ctx, op).ok_or_else(|| {
                    ControlFlowInterfaceError::new("scf.break has no enclosing scf.loop")
                })?;
                if ctx.op_results(owner.op_ref()).is_empty() {
                    Ok(ForwardedValues::default())
                } else {
                    Ok(ForwardedValues::new([self.value(ctx)]))
                }
            }
            RegionSuccessor::Region(_) => Err(ControlFlowInterfaceError::new(
                "scf.break cannot enter a region",
            )),
        }
    }
}

// =========================================================================
// Canonicalization folds
// =========================================================================

use crate::context::IrContext;
use crate::dialect::arith::const_int_value;
use crate::dialect::core::{BoolLike, IntegerLike};
use crate::ops::DialectOp;
use crate::refs::{OpRef, ValueRef};
use crate::transforms::canonicalize::{Fold, FoldResult};
use crate::types::Attribute;

/// `scf.if(arith.constant Int(c) : core.i1)` → splice the chosen region's
/// body into the parent block.
///
/// When the condition is a compile-time constant, exactly one branch
/// runs; the other is dead. The chosen region's `scf.yield <values>`
/// supplies the if op's results, so the fold names the body ops to
/// keep and the values that should take over the if op's result slots
/// — the canonicalize dispatcher does the splice + cleanup of the
/// dead branch and the chosen region's yield via [`FoldResult::Splice`].
///
/// Multi-block regions are left alone — they only appear
/// post-`scf_to_cf`, where this pass doesn't run anyway. Bails out on
/// any structural mismatch (yield arity, non-`i1` const, malformed
/// regions).
impl Fold for If {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
        let if_op = self;
        let cond = if_op.cond(ctx);
        // Guard against malformed IR: only fold when the cond's type is
        // `core.i1`. Without this, an `arith.constant value=2 : core.i32`
        // wired into the cond slot would be treated as truthy here,
        // burning the dead branch before the validator gets to reject it.
        if !BoolLike::matches(ctx, ctx.value_ty(cond)) {
            return None;
        }
        let cond_value = const_int_value(ctx, cond)?;
        // We've gated on cond's type being `core.i1`; the only valid
        // payloads are 0 and 1. Reject anything else (e.g.
        // `arith.constant value=2 : core.i1`) so the validator surfaces the
        // malformed const instead of the fold silently picking a branch.
        let active_region = match cond_value {
            0 => if_op.else_region(ctx),
            1 => if_op.then_region(ctx),
            _ => return None,
        };

        // Active region must be a single block whose terminator is `scf.yield`.
        let blocks = &ctx.region(active_region).blocks;
        let [active_block] = blocks.as_array()?;
        let region_ops = &ctx.block(*active_block).ops;
        let (yield_op, body_ops) = region_ops.split_last()?;
        if !Yield::matches(ctx, *yield_op) {
            return None;
        }

        // Yield arity must match the if op's result count.
        let yield_operands: Vec<ValueRef> = ctx.op_operands(*yield_op).to_vec();
        if yield_operands.len() != ctx.op_results(op).len() {
            return None;
        }

        Some(FoldResult::Splice {
            body: body_ops.to_vec(),
            results: yield_operands,
        })
    }
}

/// `scf.index_switch(arith.constant Int(c))` → splice the selected region's
/// body into the parent block.
///
/// The region is the case region whose value equals the selector, or the
/// default region when no case matches. A region either ends in `scf.yield`,
/// whose values replace the results, or in its own terminator (kept, which
/// requires a resultless switch to end its block so nothing is left after
/// the terminator).
///
/// Bails on a multi-block region. Falling back to the default is declined
/// when the selector or any case value is negative: a sign-extended constant
/// may alias an unsigned case value at the discriminant's width.
impl Fold for IndexSwitch {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
        let selector = const_int_value(ctx, self.discriminant(ctx))?;
        let mut any_negative = selector < 0;
        let mut matched = None;
        for (case, region) in self.cases(ctx).zip(self.case_regions(ctx)) {
            let case = i128::from(case);
            any_negative |= case < 0;
            if case == selector && matched.is_none() {
                matched = Some(region);
            }
        }
        let active_region = match matched {
            Some(region) => region,
            None if any_negative => return None,
            None => self.default_region(ctx),
        };

        let [active_block] = ctx.region(active_region).blocks.as_array()?;
        let arm_ops = &ctx.block(*active_block).ops;
        let (last, rest) = arm_ops.split_last()?;
        if let Ok(yield_op) = Yield::from_op(ctx, *last) {
            let results = yield_op.values(ctx).to_vec();
            if results.len() != ctx.op_results(op).len() {
                return None;
            }
            return Some(FoldResult::Splice {
                body: rest.to_vec(),
                results,
            });
        }
        let parent = ctx.op(op).parent_block?;
        if !ctx.op_results(op).is_empty() || ctx.block(parent).ops.last() != Some(&op) {
            return None;
        }
        Some(FoldResult::Splice {
            body: arm_ops.to_vec(),
            results: Vec::new(),
        })
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::parser::parse_test_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};

    use crate::transforms::canonicalize::{FoldDispatchPattern, fold_ops_for_dialect};
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    /// Run only this dialect's folds on `module` via a single
    /// [`FoldDispatchPattern`] — mirrors `arith.rs::run_arith_patterns`
    /// to keep per-dialect tests isolated from other dialects' folds.
    fn run_scf_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let dispatcher = FoldDispatchPattern::from_ops(fold_ops_for_dialect("scf"));
        PatternApplicator::new(TypeConverter::new())
            .add_pattern_box(Box::new(dispatcher))
            .apply_partial(ctx, module)
    }

    fn count_ops(ctx: &IrContext, module: Module, name: crate::OpName) -> usize {
        let mut count = 0usize;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            if data.name == name {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn loop_writes_its_init_and_optional_result() {
        let text = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.nil {
      %1 = scf.loop(%0) -> (core.i32) {
        ^bb1(%2: core.i32):
          scf.break %2
      }
      scf.loop() {
          scf.continue
      }
      func.return
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
    }

    #[test]
    fn index_switch_writes_each_region_on_its_own_line() {
        let text = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.nil {
      scf.index_switch %0 attributes {note = 1} : (core.i32) -> ()
      case 0 {
          scf.yield
      }
      case -7 {
          func.return
      }
      default {
          scf.yield
      }
      scf.index_switch %0 : (core.i32) -> ()
      default {
          scf.yield
      }
      func.return
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
        let one_line = "core.module @test {\n  func.func @f(%0: core.i32) {\n    scf.index_switch %0 : (core.i32) -> () case 1 { scf.yield } default { scf.yield }\n    func.return\n  }\n}";
        let mut ctx = IrContext::new();
        parse_test_module(&mut ctx, one_line);
    }

    #[test]
    fn index_switch_builds_from_case_region_pairs() {
        let text = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.nil {
      scf.index_switch %0 : (core.i32) -> ()
      case 3 {
          scf.yield
      }
      case 1 {
          func.return
      }
      default {
          scf.yield
      }
      func.return
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        let function = crate::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let entry = ctx.region(function.body(&ctx)).blocks[0];
        let parsed = IndexSwitch::from_op(&ctx, ctx.block(entry).ops[0]).unwrap();
        let regions = parsed.arm_regions(&ctx);
        let location = ctx.op(parsed.op_ref()).location;
        let discriminant = parsed.discriminant(&ctx);
        for &region in &regions {
            ctx.detach_region(region);
        }
        let built = IndexSwitch::build_with_arms(
            &mut ctx,
            location,
            discriminant,
            [],
            [(3, regions[0]), (1, regions[1])],
            regions[2],
        );
        ctx.insert_op_before(entry, parsed.op_ref(), built.op_ref());
        ctx.remove_op_from_block(entry, parsed.op_ref());
        assert_eq!(module.op().display(&ctx).to_string(), text);
    }

    #[test]
    fn if_const_true_splices_then_region() {
        // `scf.if(const true) { %a = arith.addi %x, %x; yield %a } { yield %x }`
        // → splice the addi into the parent block, replace the if's
        // result with %a (the yield's operand).
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %t = arith.constant 1 : core.i1
    %r = scf.if %t -> (core.i32) {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    } else {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, crate::dialect::scf::If::NAME), 0);
        // The then-region's `arith.addi` is now at the parent block.
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
        insta::assert_snapshot!(module.op().display(&ctx).to_string());
    }

    #[test]
    fn if_const_false_splices_else_region() {
        // Mirror of the previous test but with const false: the else
        // region is the active one.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %f = arith.constant 0 : core.i1
    %r = scf.if %f -> (core.i32) {
      scf.yield %x
    } else {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, crate::dialect::scf::If::NAME), 0);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
    }

    #[test]
    fn if_const_does_not_match_non_const_cond() {
        // The condition is a block argument — the pattern can't decide
        // which branch runs at compile time, so the if must stay.
        let input = r#"core.module @test {
  func.func @f(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond -> (core.i32) {
      scf.yield %x
    } else {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, crate::dialect::scf::If::NAME), 1);
    }

    #[test]
    fn if_non_i1_const_cond_is_not_folded() {
        // Malformed IR: the cond slot is wired to a `core.i32` constant
        // instead of `core.i1`. The fold must bail and leave the op for
        // validation rather than picking a branch by truthiness.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %bad = arith.constant 2 : core.i32
    %r = scf.if %bad -> (core.i32) {
      scf.yield %x
    } else {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, crate::dialect::scf::If::NAME), 1);
    }

    #[test]
    fn if_malformed_i1_payload_is_not_folded() {
        // Cond is i1-typed but carries a payload outside {0, 1}.
        // The fold must bail so validation flags the bad const rather
        // than the rewrite picking the then-branch by truthiness.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %bad = arith.constant 2 : core.i1
    %r = scf.if %bad -> (core.i32) {
      scf.yield %x
    } else {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, crate::dialect::scf::If::NAME), 1);
    }

    #[test]
    fn if_const_with_empty_active_region_just_forwards_yield() {
        // No body ops in the active region — the rewrite still fires:
        // the if is erased and its result becomes the yield's operand.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %t = arith.constant 1 : core.i1
    %r = scf.if %t -> (core.i32) {
      scf.yield %x
    } else {
      %a = arith.addi %x, %x : core.i32
      scf.yield %a
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, crate::dialect::scf::If::NAME), 0);
        // The else region's addi was dropped along with the if op (it
        // was unreachable). Only the parent's `func.return` remains for
        // `arith` ops, plus the const cond.
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
    }

    const SWITCH_FALLTHROUGH: &str = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %sel = arith.constant SELECTOR : core.i32
    scf.index_switch %sel : (core.i32) -> ()
    case 0 {
      %a = arith.addi %x, %x : core.i32
      scf.yield
    }
    case 1 {
      %b = arith.muli %x, %x : core.i32
      scf.yield
    }
    default {
      scf.yield
    }
    func.return %x
  }
}"#;

    fn switch_input(selector: i64) -> String {
        SWITCH_FALLTHROUGH.replace("SELECTOR", &selector.to_string())
    }

    #[test]
    fn index_switch_yields_its_results() {
        let text = r#"core.module @test {
  func.func @f(%0: core.i32, %1: core.i64) -> core.i64 {
      %2 = scf.index_switch %0 : (core.i32) -> core.i64
      case 0 {
          scf.yield %1
      }
      default {
          %3 = arith.addi %1, %1 : core.i64
          scf.yield %3
      }
      func.return %2
  }
}
"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        assert_eq!(module.op().display(&ctx).to_string(), text);
        let result = crate::validation::validate_operation_verifiers(&ctx, module);
        assert!(result.is_ok(), "{result}");

        let folded = text.replace(
            "      %2 = scf.index_switch %0",
            "      %4 = arith.constant 5 : core.i32\n      %2 = scf.index_switch %4",
        );
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &folded);
        run_scf_patterns(&mut ctx, module);
        assert_eq!(count_ops(&ctx, module, IndexSwitch::NAME), 0);
        let printed = module.op().display(&ctx).to_string();
        assert!(
            printed.contains("%3 = arith.addi %1, %1 : core.i64\n      func.return %3"),
            "{printed}"
        );
    }

    #[test]
    fn switch_const_selector_splices_matching_case() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &switch_input(1));

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::scf::IndexSwitch::NAME),
            0
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Muli::NAME),
            1
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
        insta::assert_snapshot!(module.op().display(&ctx).to_string());
    }

    #[test]
    fn switch_const_selector_without_match_splices_default() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &switch_input(7));

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::scf::IndexSwitch::NAME),
            0
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Muli::NAME),
            0
        );
    }

    #[test]
    fn switch_terminal_arm_is_spliced_when_switch_ends_block() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %sel = arith.constant 0 : core.i32
    scf.index_switch %sel : (core.i32) -> ()
    case 0 {
      func.return %x
    }
    default {
      func.unreachable
    }
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::scf::IndexSwitch::NAME),
            0
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::func::Return::NAME),
            1
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::func::Unreachable::NAME),
            0
        );
    }

    #[test]
    fn switch_terminal_arm_is_kept_when_ops_follow_the_switch() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %sel = arith.constant 0 : core.i32
    scf.index_switch %sel : (core.i32) -> ()
    case 0 {
      func.return %x
    }
    default {
      func.return %x
    }
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::scf::IndexSwitch::NAME),
            1
        );
    }

    #[test]
    fn switch_non_const_selector_is_not_folded() {
        let input = r#"core.module @test {
  func.func @f(%tag: core.i32, %x: core.i32) -> core.i32 {
    scf.index_switch %tag : (core.i32) -> ()
    case 0 {
      scf.yield
    }
    default {
      scf.yield
    }
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::scf::IndexSwitch::NAME),
            1
        );
    }

    #[test]
    fn switch_negative_selector_does_not_fall_back_to_default() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &switch_input(-1));

        let result = run_scf_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::scf::IndexSwitch::NAME),
            1
        );
    }
}
