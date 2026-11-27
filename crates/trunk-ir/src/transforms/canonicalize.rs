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
//! - [`FoldResult::Constant`] — replace the op with a constant of its
//!   result type, which the op's dialect builds through its
//!   [`ConstantMaterializer`].
//! - [`FoldResult::Splice`] — splice a body of ops into the parent
//!   block before the matched op and RAUW its results with the
//!   supplied values; the dispatcher additionally drops anything
//!   still attached to the matched op's regions (e.g. yields and the
//!   dead branch in `scf.if(const)`).
//!
//! A fold reads its operands' constants through [`ConstantLike`], with
//! [`constant_value`] and [`const_int_value`], without naming the dialect
//! that defines them. Folds are owned by their dialects: `arith`, `clif`,
//! `core`, `scf`, `adt`, and `tribute_rt` register them today.

use trunk_ir_macros::{dialect_interface, op_interface};

use crate::OpName;
use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::func;
use crate::ops::{Dialect, DialectOp};
use crate::pass::{Pass, PassRunResult};
use crate::refs::{OpRef, TypeRef, ValueDef, ValueRef};
use crate::rewrite::{
    PatternApplicator, PatternRewriter, RewritePattern, RewriteScope, TypeConverter,
};
use crate::types::{Attribute, Location};

// =========================================================================
// Fold API
// =========================================================================

/// Result of trying to fold an operation, the counterpart of MLIR's
/// `OpFoldResult` (an attribute or an existing value) plus [`Splice`] for
/// inlining one of the operation's regions.
///
/// [`Splice`]: FoldResult::Splice
#[derive(Debug, Clone)]
pub enum FoldResult {
    /// RAUW the op's single result with this existing value.
    Forward(ValueRef),
    /// Replace the op with a constant of its single result's type carrying
    /// this attribute. The driver asks the op's dialect to materialize it
    /// and leaves the op unchanged when the dialect has no materializer or
    /// declines the attribute.
    Constant(Attribute),
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

/// An operation that folds into a canonical form, the counterpart of
/// MLIR's `fold` hook.
///
/// The context is read-only; the canonicalize driver applies the returned
/// [`FoldResult`] after the fold returns. A fold follows three rules:
///
/// - **Strict progress.** Applying a result removes the operation from the
///   IR, so a fold returns `None` unless it simplifies something. It never
///   produces another operation of its own kind, and a [`ConstantLike`]
///   operation never folds to its own value, or the driver would rebuild
///   it forever.
/// - **Materialization.** A [`FoldResult::Constant`] names only the value.
///   The driver builds the constant with the materializer of the folded
///   operation's dialect, so a fold returns an attribute that dialect's
///   constants can hold at the result type. When the materializer declines
///   it, the operation stays as it was.
/// - **Operand constants.** A fold reads its operands' constant values
///   with [`constant_value`] or [`const_int_value`], which accept a
///   constant of any dialect through [`ConstantLike`]. It does not match
///   another dialect's constant operation by name.
#[op_interface]
pub trait Fold: DialectOp {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult>;
}

/// An operation whose single result is a compile-time constant, the
/// counterpart of MLIR's `ConstantLike`.
///
/// Folds read their operands' values through it, so a fold recognizes a
/// constant of any dialect. The attribute uses the representation
/// [`FoldResult::Constant`] does: [`Attribute::Int`] holds an integer of
/// any width sign-extended to `i128`.
#[op_interface]
pub trait ConstantLike: DialectOp {
    fn constant_value(&self, ctx: &IrContext) -> Attribute;
}

/// The constant value of `value`, if a [`ConstantLike`] operation defines
/// it.
pub fn constant_value(ctx: &IrContext, value: ValueRef) -> Option<Attribute> {
    let ValueDef::OpResult(producer, _) = ctx.value_def(value) else {
        return None;
    };
    Some(DynConstantLike::get(ctx, producer)?.constant_value(ctx))
}

/// The integer constant of `value`, if a [`ConstantLike`] operation defines
/// it with an [`Attribute::Int`].
pub fn const_int_value(ctx: &IrContext, value: ValueRef) -> Option<i128> {
    match constant_value(ctx, value)? {
        Attribute::Int(v) => Some(v),
        _ => None,
    }
}

// =========================================================================
// Constant materialization
// =========================================================================

/// A dialect that builds its own constants, the counterpart of MLIR's
/// `Dialect::materializeConstant`.
///
/// The canonicalize driver builds a [`FoldResult::Constant`] with the
/// materializer of the folded operation's dialect, so a fold's constant
/// stays at that dialect's level of the pipeline. A dialect implements it on
/// the `Dialect` marker `#[dialect]` generates and registers it with
/// [`register_interfaces!`](crate::register_interfaces).
#[dialect_interface]
pub trait ConstantMaterializer: Dialect {
    /// Build a detached constant operation of the dialect whose single
    /// result has type `ty` and holds `value`, at `location`. Return `None`
    /// without changing the IR when the dialect has no constant for that
    /// value and type.
    fn materialize_constant(
        &self,
        ctx: &mut IrContext,
        value: Attribute,
        ty: TypeRef,
        location: Location,
    ) -> Option<OpRef>;
}

/// One `RewritePattern` that dispatches to per-op folds through
/// [`DynFold`].
///
/// Replaces N individual self-filtering patterns (one per op kind) with
/// a single O(1) dispatcher: instead of every pattern testing
/// whether the op is its own, the dispatcher looks the op's name up in
/// the [`Fold`] registry and calls the fold directly. Duplicate
/// registrations for the same operation panic when the registry is first
/// queried.
pub struct FoldDispatchPattern {
    /// The operations whose folds this dispatcher applies.
    ops: Vec<OpName>,
}

impl FoldDispatchPattern {
    /// Build a dispatcher that applies only the folds of `ops`.
    pub fn from_ops(ops: impl IntoIterator<Item = OpName>) -> Self {
        Self {
            ops: ops.into_iter().collect(),
        }
    }

    /// Build a dispatcher from every [`Fold`] implementation registered
    /// across the workspace.
    pub fn from_inventory() -> Self {
        Self {
            ops: DynFold::implementors().collect(),
        }
    }
}

impl RewritePattern for FoldDispatchPattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&self.ops)
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let result = DynFold::get(ctx, op).and_then(|fold| fold.fold(ctx))?;
        match result {
            FoldResult::Forward(v) => {
                rewriter.erase_op(vec![v]);
                Some(())
            }
            FoldResult::Constant(value) => {
                let &[result_ty] = ctx.op_result_types(op) else {
                    return rewriter.no_match("a constant fold needs exactly one result");
                };
                let data = ctx.op(op);
                let location = data.location;
                let materializer = rewriter.require(
                    DynConstantMaterializer::get(data.name.dialect(ctx)),
                    "the dialect has no constant materializer",
                )?;
                let constant = rewriter.require(
                    materializer.materialize_constant(ctx, value, result_ty, location),
                    "the dialect declined to materialize the constant",
                )?;
                rewriter.replace_op(constant);
                Some(())
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
) -> Option<()> {
    // Precondition: `results` must cover every result slot of the
    // matched op. Check before any mutation so a buggy fold can't leave
    // body ops spliced out and the matched op's regions emptied while
    // the final RAUW fails arity checks deep in `rewriter.erase_op`.
    if ctx.op_results(op).len() != results.len() {
        return None;
    }
    let parent_block = ctx.op(op).parent_block?;
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
    Some(())
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
    %r = scf.if %cond -> (core.i32) {
      %z = arith.constant 0 : core.i32
      %s = arith.addi %x, %z : core.i32
      scf.yield %s
    } else {
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
    %z = arith.constant 0 : core.i32
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
    %z = arith.constant 0 : core.i32
    %r = arith.addi %x, %z : core.i32
    func.return %r
  }
  func.func @g(%x: core.i32) -> core.i32 {
    %z = arith.constant 0 : core.i32
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
    %a = arith.constant 2 : core.i32
    %b = arith.constant 3 : core.i32
    %s = arith.addi %a, %b : core.i32
    %m = arith.muli %s, %x : core.i32
    %z = arith.constant 0 : core.i32
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
    %t = arith.constant 1 : core.i1
    %r = scf.if %t -> (core.i32) {
      %z = arith.constant 0 : core.i32
      %a = arith.addi %x, %z : core.i32
      scf.yield %a
    } else {
      %two = arith.constant 2 : core.i32
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
    %z = arith.constant 0 : core.i32
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
