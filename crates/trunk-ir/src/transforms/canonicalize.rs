//! Canonicalization pass for TrunkIR.
//!
//! Greedy fixed-point pass that folds operations into a canonical form.
//! Each dialect contributes per-op folds by implementing the [`Fold`]
//! operation interface; the pass discovers them through the interface's
//! registry rather than referencing each dialect by name. As a result
//! this module knows nothing about which dialects exist — adding a new
//! dialect's folds requires no edit here.
//!
//! The driver dispatches each visited op to its fold (if any) through
//! [`DynFold`], which looks the op's name up in that registry. A fold
//! returns one of:
//!
//! - [`FoldResult::Forward`] — RAUW the op's result with an existing
//!   value.
//! - [`FoldResult::ArithConst`] — replace with a fresh `arith.const`
//!   of the op's result type.
//! - [`FoldResult::Splice`] — splice a body of ops into the parent
//!   block before the matched op and RAUW its results with the
//!   supplied values; the dispatcher additionally drops anything
//!   still attached to the matched op's regions (e.g. yields and the
//!   dead branch in `scf.if(const)`).
//!
//! Currently registered (across `arith`, `core`, `scf`, `adt`, `tribute_rt`):
//!
//! - `arith.addi`/`subi`/`muli` — integer identity (`x+0`, `x-0`,
//!   `x*0`, `x*1`) and constant folding at the result type's bit-width.
//! - `core.unrealized_conversion_cast` — identity (same source/target
//!   type) and round-trip elimination (`A → B → A`).
//! - `scf.if` / `scf.switch` with a constant condition or selector — splice
//!   the chosen arm.
//! - `adt.struct_get` of an `adt.struct_new`, and identity `adt.ref_cast`.
//! - `tribute_rt.unbox_*` of the matching `box_*`.
//!
//! Float and div/rem folds are deferred until each one's edge cases
//! (NaN/-0.0, division-by-zero) are pinned down.

use rustc_hash::FxHashSet as HashSet;
use trunk_ir_macros::op_interface;

use crate::OpName;
use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::{arith, func};
use crate::ops::DialectOp;
use crate::pass::{Pass, PassRunResult};
use crate::refs::{OpRef, ValueRef};
use crate::rewrite::{
    PatternApplicator, PatternRewriter, RewritePattern, RewriteScope, TypeConverter,
};
use crate::types::Attribute;

// =========================================================================
// Fold API
// =========================================================================

/// Result of trying to fold an operation.
///
/// Folds are *strict-progress* rewrites — applying one must remove the op
/// from the IR. A fold function that produces another op of the same
/// (dialect, op_name) would loop the canonicalize pass; this is a bug in
/// the fold, not a property the driver tries to detect.
#[derive(Debug, Clone)]
pub enum FoldResult {
    /// RAUW the op's single result with this existing value.
    Forward(ValueRef),
    /// Replace with a freshly-built `arith.const` of the op's result type,
    /// carrying this attribute. The driver materializes the const op.
    ///
    /// Specifically named `ArithConst` (not `Const`) so that future
    /// const-producing dialects require an explicit new variant — the
    /// driver currently knows only how to build `arith.const`.
    ArithConst(Attribute),
    /// Splice a sequence of body ops into the matched op's parent block
    /// (in order, immediately before the matched op), then erase the
    /// matched op while RAUW'ing its results to `results`.
    ///
    /// Used by region-bearing ops whose canonicalization is "pick one
    /// region's body and inline it" — `scf.if(const)` being the seed
    /// case. The dispatcher additionally drops any ops still attached
    /// to the matched op's regions (typically the chosen region's
    /// yield + the dead branch's body); fold authors only have to name
    /// the ops they want to *keep* (`body`) and the values that should
    /// take over the matched op's result slots (`results`).
    Splice {
        /// Ops to detach from their current parent and insert in the
        /// matched op's parent block, in order, immediately before the
        /// matched op.
        body: Vec<OpRef>,
        /// Values to RAUW the matched op's results with. Length must
        /// match the matched op's result count.
        results: Vec<ValueRef>,
    },
}

/// An operation that folds into a canonical form.
///
/// The context is read-only; the canonicalize driver applies the returned
/// [`FoldResult`] after the fold returns.
#[op_interface]
pub trait Fold: DialectOp {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult>;
}

/// One `RewritePattern` that dispatches to per-op folds through
/// [`DynFold`].
///
/// Replaces N individual self-filtering patterns (one per op kind) with
/// a single O(1) dispatcher: instead of every pattern testing
/// whether it is `arith.addi`, the dispatcher looks the op's name up in
/// the [`Fold`] registry and calls the fold directly. Duplicate
/// registrations for the same operation panic when the registry is first
/// queried.
pub struct FoldDispatchPattern {
    /// The operations whose folds this dispatcher applies; `None` applies
    /// every registered fold.
    ops: Option<HashSet<OpName>>,
}

impl FoldDispatchPattern {
    /// Build a dispatcher that applies only the folds of `ops`.
    pub fn from_ops(ops: impl IntoIterator<Item = OpName>) -> Self {
        Self {
            ops: Some(ops.into_iter().collect()),
        }
    }

    /// Build a dispatcher from every [`Fold`] implementation registered
    /// across the workspace.
    pub fn from_inventory() -> Self {
        Self { ops: None }
    }
}

impl RewritePattern for FoldDispatchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if let Some(ops) = &self.ops
            && !ops.contains(&ctx.op(op).name)
        {
            return false;
        }
        let Some(result) = DynFold::get(ctx, op).and_then(|fold| fold.fold(ctx)) else {
            return false;
        };
        match result {
            FoldResult::Forward(v) => {
                rewriter.erase_op(vec![v]);
                true
            }
            FoldResult::ArithConst(attr) => {
                let loc = ctx.op(op).location;
                let result_ty = match ctx.op_result_types(op) {
                    [t] => *t,
                    _ => return false,
                };
                let new_const = arith::Const::operands()
                    .value(attr)
                    .results(result_ty)
                    .build(ctx, loc);
                rewriter.replace_op(new_const.op_ref());
                true
            }
            FoldResult::Splice { body, results } => apply_splice(ctx, rewriter, op, body, results),
        }
    }

    fn name(&self) -> &'static str {
        "FoldDispatch"
    }
}

/// Apply a [`FoldResult::Splice`]: detach the listed body ops from
/// their current parents, insert them before `op` in `op`'s parent
/// block, drop everything still attached to `op`'s regions (typically
/// terminators and the dead branch), then ask the rewriter to erase
/// `op` while mapping its results to `results`.
fn apply_splice(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    op: OpRef,
    body: Vec<OpRef>,
    results: Vec<ValueRef>,
) -> bool {
    // Precondition: `results` must cover every result slot of the
    // matched op. Check before any mutation so a buggy fold can't leave
    // body ops spliced out and the matched op's regions emptied while
    // the final RAUW fails arity checks deep in `rewriter.erase_op`.
    if ctx.op_results(op).len() != results.len() {
        return false;
    }
    let Some(parent_block) = ctx.op(op).parent_block else {
        return false;
    };
    for body_op in body {
        ctx.detach_op(body_op);
        ctx.insert_op_before(parent_block, op, body_op);
    }
    // Sweep up anything left in the matched op's regions — chosen
    // region's terminator (we already kept the body it preceded) plus
    // the dead branch entirely.
    // Erasing ops mutates these child lists, so iterate snapshots. Cloning
    // keeps them inline for the usual few children.
    let regions = ctx.op_regions(op).collect::<crate::RegionList>();
    for region in regions {
        let blocks = ctx.region(region).blocks.clone();
        for block in blocks {
            // Reverse so a block-internal op is erased before any op that
            // depends on its results, satisfying `remove_op`'s "no remaining
            // uses" precondition.
            let remaining = ctx.block(block).ops.clone();
            for orphan in remaining.into_iter().rev() {
                crate::rewrite::erase_op(ctx, orphan);
            }
        }
    }
    rewriter.erase_op(results);
    true
}

/// Every operation in `dialect` that implements [`Fold`]. Used by
/// per-dialect test helpers to keep their assertions isolated from
/// other dialects' folds (folds dispatch by op key, so cross-dialect
/// interference is unlikely, but filtering keeps tests honest as more
/// dialects add folds).
#[cfg(test)]
pub(crate) fn fold_ops_for_dialect(dialect: &'static str) -> impl Iterator<Item = OpName> {
    DynFold::implementors()
        .filter(move |name| name.def().is_some_and(|def| def.name.dialect() == dialect))
}

// =========================================================================
// Pass entry
// =========================================================================

/// Outcome of one `canonicalize` invocation.
///
/// `#[must_use]` because silently dropping the result hides the case
/// where the pass exhausted its iteration budget without reaching a
/// fixed point — a real bug we want callers to acknowledge (log, fail,
/// or explicitly `let _ = ...` to opt out).
#[must_use]
#[derive(Debug, Clone, Copy)]
struct CanonicalizeResult {
    pub iterations: usize,
    pub total_changes: usize,
    pub reached_fixpoint: bool,
}

fn canonicalize_applicator() -> PatternApplicator<'static> {
    PatternApplicator::new(TypeConverter::new())
        .add_pattern_box(Box::new(FoldDispatchPattern::from_inventory()))
}

fn result_from_apply(result: crate::rewrite::ApplyResult) -> CanonicalizeResult {
    CanonicalizeResult {
        iterations: result.iterations,
        total_changes: result.total_changes,
        reached_fixpoint: result.reached_fixpoint,
    }
}

/// Run canonicalization to a fixed point within one rewrite scope.
fn canonicalize<S: RewriteScope>(ctx: &mut IrContext, scope: S) -> CanonicalizeResult {
    result_from_apply(canonicalize_applicator().apply_partial(ctx, scope))
}

/// Function-anchored canonicalization to a fixed point.
pub struct Canonicalize;

impl Pass for Canonicalize {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "canonicalize-func"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: func::Func,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let result = canonicalize(ctx, target);
        if !result.reached_fixpoint {
            tracing::warn!(
                "canonicalize-func did not reach a fixed point: {} iterations, {} changes",
                result.iterations,
                result.total_changes
            );
        }
        Ok(())
    }
}

// =========================================================================
// Pass-level tests
//
// Tests that exercise the pass *shell* — region walk, fixed-point
// iteration, and the collaboration between folds from one or more
// dialects. Per-fold positive/negative tests live alongside the fold
// itself in its dialect module.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::DialectOp;
    use crate::parser::parse_test_module;
    use crate::rewrite::Module;
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

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

    fn count_ops_under(ctx: &IrContext, root: OpRef, name: crate::OpName) -> usize {
        let mut count = 0usize;
        let _ = walk_op::<()>(ctx, root, &mut |op| {
            let data = ctx.op(op);
            if data.name == name {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn canonicalize_walks_into_nested_regions() {
        // The fold target lives inside an `scf.if` then-region. The
        // applicator must descend into nested regions for the rewrite
        // to fire.
        let input = r#"core.module @test {
  func.func @f(%cond: core.i1, %x: core.i32) -> core.i32 {
    %r = scf.if %cond : core.i32 {
      %z = arith.const {value = 0} : core.i32
      %s = arith.addi %x, %z : core.i32
      scf.yield %s
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
    }

    #[test]
    fn canonicalize_reaches_fixpoint_on_chained_identity() {
        // `((x + 0) + 0)` must collapse to `x` in a single canonicalize
        // call. The applicator iterates until no pattern fires.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %t = arith.addi %x, %z : core.i32
    %r = arith.addi %t, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.reached_fixpoint);
        assert!(result.total_changes >= 2);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
    }

    #[test]
    fn canonicalize_rewrites_only_selected_function_scope() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.addi %x, %z : core.i32
    func.return %r
  }
  func.func @g(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.addi %x, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let funcs: Vec<func::Func> = module
            .ops(&ctx)
            .iter()
            .copied()
            .map(|op| func::Func::from_op(&ctx, op).expect("test op must be func.func"))
            .collect();

        let result = canonicalize(&mut ctx, funcs[0]);

        assert!(result.reached_fixpoint);
        assert_eq!(
            count_ops_under(&ctx, funcs[0].op_ref(), crate::dialect::arith::Addi::NAME),
            0
        );
        assert_eq!(
            count_ops_under(&ctx, funcs[1].op_ref(), crate::dialect::arith::Addi::NAME),
            1
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
    }

    #[test]
    fn fold_and_identity_collaborate_to_fixpoint() {
        // `((2 + 3) * x) + 0` should collapse to `5 * x` after one
        // canonicalize call: the `arith.addi` fold const-folds `2 + 3` to
        // `5`, then the same fold (on the outer addi) erases `_ + 0`.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %a = arith.const {value = 2} : core.i32
    %b = arith.const {value = 3} : core.i32
    %s = arith.addi %a, %b : core.i32
    %m = arith.muli %s, %x : core.i32
    %z = arith.const {value = 0} : core.i32
    %r = arith.addi %m, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.reached_fixpoint);
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
        // muli %s, %x stays — only one operand is const.
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Muli::NAME),
            1
        );
    }

    #[test]
    fn if_const_collaborates_with_arith_folds() {
        // `scf.if(true) { yield (x + 0) } { yield x }` should collapse to
        // `func.return %x` in one canonicalize call, exercising both the
        // pattern-based escape hatch (`IfConstFold` splices the then
        // region) and the per-op folds (the `arith.addi` fold erases the
        // `_ + 0`). The dead else region's `arith.muli` is dropped along
        // with the erased `scf.if` op.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %t = arith.const {value = 1} : core.i1
    %r = scf.if %t : core.i32 {
      %z = arith.const {value = 0} : core.i32
      %a = arith.addi %x, %z : core.i32
      scf.yield %a
    } {
      %two = arith.const {value = 2} : core.i32
      %m = arith.muli %x, %two : core.i32
      scf.yield %m
    }
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.reached_fixpoint);
        // scf.if is gone; the then-region's body was spliced out.
        assert_eq!(count_ops(&ctx, module, crate::dialect::scf::If::NAME), 0);
        // arith.addi was folded away (`x + 0` → `x`).
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
        // The else region's `arith.muli` was orphaned with the erased
        // `scf.if`; it should not appear in the walked module.
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Muli::NAME),
            0
        );
    }

    #[test]
    fn canonicalize_reports_no_changes_on_already_canonical_module() {
        // A module with nothing to fold must converge in a single
        // sweep that records zero changes — the result fields the
        // pipeline now relies on (iterations / total_changes /
        // reached_fixpoint) must reflect that exactly.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.reached_fixpoint);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.total_changes, 0);
    }

    #[test]
    fn applicator_reports_non_fixpoint_when_iteration_budget_exhausted() {
        // `((x + 0) + 0)` collapses fully in one greedy sweep, but
        // confirming the fixed point still requires a follow-up
        // iteration with zero changes. Capping the budget at 1
        // forces the run to terminate as non-fixpoint with at
        // least one fold applied — the exact case
        // `Canonicalize` logs about.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %t = arith.addi %x, %z : core.i32
    %r = arith.addi %t, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let applicator = PatternApplicator::new(TypeConverter::new())
            .with_max_iterations(1)
            .add_pattern_box(Box::new(FoldDispatchPattern::from_inventory()));
        let result = applicator.apply_partial(&mut ctx, module);

        assert!(!result.reached_fixpoint);
        assert_eq!(result.iterations, 1);
        assert!(result.total_changes >= 1);
    }
}
