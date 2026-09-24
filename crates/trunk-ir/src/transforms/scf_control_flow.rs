//! Conservative structured-control facts shared by target lowerings.

use std::collections::HashSet;
use std::ops::ControlFlow;

use crate::analysis::{Analysis, AnalysisContext, AnalysisError};
use crate::context::IrContext;
use crate::dialect::scf;
use crate::op_interface::{CallableExitOps, RegionBranchOps, RegionBranchPoint, RegionSuccessor};
use crate::ops::DialectOp;
use crate::refs::{OpRef, RegionRef};
use crate::walk::{WalkAction, walk_op};

/// Terminal proofs for one operation subtree at the time of computation.
///
/// A false query means the conservative proof did not succeed, including for
/// malformed interfaces, multi-block regions, and loop cycles. It is not an
/// analysis failure or proof that execution returns. Target acceptance and
/// conversion diagnostics belong to consumers, not this analysis.
///
/// Cache lookups recompute this analysis after the IR changes. A lowering may
/// retain decisions derived from it for the original operations, but must not
/// use those decisions as facts about the rewritten IR.
#[derive(Default)]
pub struct StructuredControlAnalysis {
    terminal_regions: HashSet<RegionRef>,
    terminal_successors: HashSet<OpRef>,
    unused_never_controls: HashSet<OpRef>,
    terminal_switches: HashSet<OpRef>,
}

impl Analysis for StructuredControlAnalysis {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let ir = ctx.ir();
        let mut order = Vec::new();
        let _ = walk_op::<()>(ir, target, &mut |op| {
            order.push(op);
            ControlFlow::Continue(WalkAction::Advance)
        });
        let mut analysis = Self::default();
        // Reverse preorder puts every descendant before its owner. Region
        // proofs then read already-computed child facts, never recurse.
        for op in order.into_iter().rev() {
            for &region in &ir.op(op).regions {
                if analysis.region_is_terminal(ir, region) {
                    analysis.terminal_regions.insert(region);
                }
            }
            if !(scf::If::matches(ir, op)
                || scf::Loop::matches(ir, op)
                || scf::Switch::matches(ir, op))
            {
                continue;
            }
            let Some(interface) = RegionBranchOps::get(ir, op) else {
                continue;
            };
            let Ok(successors) = interface.successors(ir, op, RegionBranchPoint::Parent) else {
                continue;
            };
            if successors.as_slice().is_empty()
                || !successors.as_slice().iter().all(|successor| {
                    matches!(successor, RegionSuccessor::Region(region) if analysis.is_terminal_region(*region))
                })
            {
                continue;
            }
            analysis.terminal_successors.insert(op);
            if !ir
                .op(op)
                .parent_block
                .is_some_and(|block| ir.block(block).ops.last() == Some(&op))
            {
                continue;
            }
            let results = ir.op_results(op);
            if let [result] = results {
                let ty = ir.get_type(ir.value_ty(*result));
                if ty.dialect == "core" && ty.name == "never" && !ir.has_uses(*result) {
                    analysis.unused_never_controls.insert(op);
                }
            }
            if scf::Switch::matches(ir, op) && results.is_empty() {
                analysis.terminal_switches.insert(op);
            }
        }
        Ok(analysis)
    }
}

impl StructuredControlAnalysis {
    /// Whether a single-block region is proven to exit its callable.
    pub fn is_terminal_region(&self, region: RegionRef) -> bool {
        self.terminal_regions.contains(&region)
    }

    /// Whether every semantic entry successor is a proven terminal region.
    pub fn has_only_terminal_region_successors(&self, op: OpRef) -> bool {
        self.terminal_successors.contains(&op)
    }

    /// Whether final structured control has exactly one unused Never result
    /// and only terminal entry successors.
    pub fn has_terminal_unused_never_result(&self, op: OpRef) -> bool {
        self.unused_never_controls.contains(&op)
    }

    /// Whether a final resultless switch has only terminal entry successors.
    pub fn is_terminal_resultless_switch(&self, op: OpRef) -> bool {
        self.terminal_switches.contains(&op)
    }

    fn region_is_terminal(&self, ctx: &IrContext, region: RegionRef) -> bool {
        let [block] = ctx.region(region).blocks.as_slice() else {
            return false;
        };
        let Some(&last) = ctx.block(*block).ops.last() else {
            return false;
        };
        if scf::If::matches(ctx, last) || scf::Loop::matches(ctx, last) {
            self.has_terminal_unused_never_result(last)
                || (ctx.op_results(last).is_empty()
                    && self.has_only_terminal_region_successors(last))
        } else if scf::Switch::matches(ctx, last) {
            self.is_terminal_resultless_switch(last)
        } else {
            CallableExitOps::exits_callable(ctx, last).is_ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::analysis::AnalysisCache;
    use crate::dialect::func;
    use crate::parser::parse_test_module;

    const NESTED_CONTROL: &str = r#"core.module @test {
  func.func @first(%cond: core.i1, %tag: core.i32) -> core.nil {
    %never = scf.if %cond : core.never {
      scf.switch %tag {
        scf.case {value = 0} { func.return }
        scf.default { func.unreachable }
      }
    } { func.unreachable }
  }
  func.func @second() -> core.nil { func.return }
}"#;

    #[test]
    fn cached_facts_cover_nested_control_and_stay_within_target() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED_CONTROL);
        let first = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let second = func::Func::from_op(&ctx, module.ops(&ctx)[1]).unwrap();
        let mut cache = AnalysisCache::new();
        let all = cache
            .get::<StructuredControlAnalysis>(&ctx, module.op())
            .unwrap();
        let again = cache
            .get::<StructuredControlAnalysis>(&ctx, module.op())
            .unwrap();
        assert!(Arc::ptr_eq(&all, &again));
        let scoped = cache
            .get::<StructuredControlAnalysis>(&ctx, first.op_ref())
            .unwrap();
        assert!(all.is_terminal_region(first.body(&ctx)));
        assert!(all.is_terminal_region(second.body(&ctx)));
        assert!(scoped.is_terminal_region(first.body(&ctx)));
        assert!(!scoped.is_terminal_region(second.body(&ctx)));
        let _ = walk_op::<()>(&ctx, first.op_ref(), &mut |op| {
            if scf::If::matches(&ctx, op) {
                assert!(scoped.has_terminal_unused_never_result(op));
            } else if scf::Switch::matches(&ctx, op) {
                assert!(scoped.is_terminal_resultless_switch(op));
                assert!(scoped.has_only_terminal_region_successors(op));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }

    #[test]
    fn ir_revision_recomputes_use_position_and_nested_exit_facts() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, NESTED_CONTROL);
        let first = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let body = first.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        let control = ctx.block(entry).ops[0];
        let never = ctx.op_results(control)[0];
        let mut cache = AnalysisCache::new();
        let original = cache
            .get::<StructuredControlAnalysis>(&ctx, module.op())
            .unwrap();
        assert!(original.has_terminal_unused_never_result(control));

        // A synthetic use before the control isolates liveness from final position.
        let loc = ctx.op(control).location;
        let user = scf::Yield::operands([never]).build(&mut ctx, loc);
        ctx.insert_op_before(entry, control, user.op_ref());
        assert!(
            cache
                .get_cached::<StructuredControlAnalysis>(&ctx, module.op())
                .is_none()
        );
        let used = cache
            .get::<StructuredControlAnalysis>(&ctx, module.op())
            .unwrap();
        assert!(!Arc::ptr_eq(&original, &used));
        assert!(used.has_only_terminal_region_successors(control));
        assert!(!used.has_terminal_unused_never_result(control));
        assert!(!used.is_terminal_region(body));

        // Remove the use and append an exit: only the final-position fact changes.
        ctx.detach_op(user.op_ref());
        ctx.remove_op(user.op_ref());
        let trailing = func::unreachable(&mut ctx, loc);
        ctx.push_op(entry, trailing.op_ref());
        assert!(
            cache
                .get_cached::<StructuredControlAnalysis>(&ctx, module.op())
                .is_none()
        );
        let followed = cache
            .get::<StructuredControlAnalysis>(&ctx, module.op())
            .unwrap();
        assert!(followed.has_only_terminal_region_successors(control));
        assert!(!followed.has_terminal_unused_never_result(control));
        assert!(followed.is_terminal_region(body));

        // Restoring the outer position cannot hide a changed nested exit.
        ctx.detach_op(trailing.op_ref());
        ctx.remove_op(trailing.op_ref());
        let mut nested_exit = None;
        let _ = walk_op::<()>(&ctx, control, &mut |op| {
            if func::Return::matches(&ctx, op) {
                nested_exit = Some(op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        let exit = nested_exit.unwrap();
        let arm = ctx.op(exit).parent_block.unwrap();
        ctx.detach_op(exit);
        ctx.remove_op(exit);
        let yielding = scf::Yield::operands([]).build(&mut ctx, loc);
        ctx.push_op(arm, yielding.op_ref());
        assert!(
            cache
                .get_cached::<StructuredControlAnalysis>(&ctx, module.op())
                .is_none()
        );
        let changed = cache
            .get::<StructuredControlAnalysis>(&ctx, module.op())
            .unwrap();
        assert!(!changed.has_only_terminal_region_successors(control));
        assert!(!changed.has_terminal_unused_never_result(control));
        assert!(!changed.is_terminal_region(body));
        assert!(crate::validation::validate_use_chains(&ctx, module).is_ok());
    }
}
