//! Canonicalization pass for TrunkIR.
//!
//! Greedy fixed-point pass that folds operations into a canonical form.
//! Each dialect contributes per-op `fold` functions (one fold per op kind)
//! that return either a value to forward or an attribute that the driver
//! materializes as `arith.const`. The pass aggregates all dialect folds
//! into a single dispatch pattern that O(1)-looks up the fold for each
//! visited op.
//!
//! - `arith::folds()` — `addi`/`subi`/`muli` integer identities and
//!   constant folding at the result type's bit-width.
//! - `core::folds()` — `unrealized_conversion_cast` identity (same
//!   source/target type) and round-trip elimination (`A → B → A`).
//!
//! Adding a fold for an existing op = edit one function in that dialect.
//! Adding a brand-new dialect's folds = a one-line `chain` here. Truly
//! local rewrites that don't fit the fold shape (e.g. region splice for
//! `scf.if(const)`) still go through `RewritePattern`; tracked in #690.
//!
//! Float and div/rem folds are deferred until each one's edge cases
//! (NaN/-0.0, division-by-zero) are pinned down.

use std::collections::HashMap;

use crate::context::IrContext;
use crate::dialect::{arith, core as core_dialect};
use crate::refs::{OpRef, ValueRef};
use crate::rewrite::{Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter};
use crate::symbol::Symbol;
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
}

/// Per-op fold function. `&IrContext` is read-only; mutation happens in
/// the driver after the fold returns.
pub type FoldFn = fn(&IrContext, OpRef) -> Option<FoldResult>;

/// One `RewritePattern` that dispatches to per-op fold functions via
/// hashmap lookup.
///
/// Replaces N individual self-filtering patterns (one per op kind) with
/// a single O(1) dispatcher: instead of every pattern testing
/// `op.dialect == "arith" && op.name == "addi"`, the dispatcher does one
/// `HashMap::get((dialect, name))` and calls the fold directly.
///
/// The hashmap is built once at pass-construction time. Duplicate
/// registrations for the same (dialect, op_name) are caught by
/// `debug_assert!`.
pub struct FoldDispatchPattern {
    table: HashMap<(Symbol, Symbol), FoldFn>,
}

impl FoldDispatchPattern {
    /// Build a dispatcher from an iterator of `(dialect, op_name, fold)`
    /// triples. Panics in debug builds on duplicate keys.
    pub fn from_folds(folds: impl IntoIterator<Item = (Symbol, Symbol, FoldFn)>) -> Self {
        let mut table = HashMap::new();
        for (dialect, op_name, fold) in folds {
            debug_assert!(
                table.insert((dialect, op_name), fold).is_none(),
                "duplicate canonicalize fold for {dialect}.{op_name}",
            );
        }
        Self { table }
    }

    /// Convenience for tests: build a single-entry dispatcher. Lets a
    /// test compose one specific fold with other patterns in the same
    /// applicator without pulling in every registered fold.
    #[cfg(test)]
    pub(crate) fn single(dialect: &'static str, op_name: &'static str, fold: FoldFn) -> Self {
        Self::from_folds([(Symbol::new(dialect), Symbol::new(op_name), fold)])
    }
}

impl RewritePattern for FoldDispatchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let key = {
            let data = ctx.op(op);
            (data.dialect, data.name)
        };
        let Some(fold) = self.table.get(&key).copied() else {
            return false;
        };
        let Some(result) = fold(ctx, op) else {
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
                let new_const = arith::r#const(ctx, loc, result_ty, attr);
                rewriter.replace_op(new_const.op_ref());
                true
            }
        }
    }

    fn name(&self) -> &'static str {
        "FoldDispatch"
    }
}

// =========================================================================
// Pass entry
// =========================================================================

/// Outcome of one `canonicalize` invocation.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalizeResult {
    pub iterations: usize,
    pub total_changes: usize,
    pub reached_fixpoint: bool,
}

/// Run canonicalization to a fixed point on `module`.
pub fn canonicalize(ctx: &mut IrContext, module: Module) -> CanonicalizeResult {
    let folds = arith::folds().into_iter().chain(core_dialect::folds());
    let dispatcher = FoldDispatchPattern::from_folds(folds);
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern_box(Box::new(dispatcher));
    let result = applicator.apply_partial(ctx, module);
    CanonicalizeResult {
        iterations: result.iterations,
        total_changes: result.total_changes,
        reached_fixpoint: result.reached_fixpoint,
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
    use crate::parser::parse_test_module;
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    fn count_ops(ctx: &IrContext, module: Module, dialect: &str, name: &str) -> usize {
        let dialect_sym = Symbol::from_dynamic(dialect);
        let name_sym = Symbol::from_dynamic(name);
        let mut count = 0usize;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            if data.dialect == dialect_sym && data.name == name_sym {
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
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
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
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
    }

    #[test]
    fn fold_and_identity_collaborate_to_fixpoint() {
        // `((2 + 3) * x) + 0` should collapse to `5 * x` after one
        // canonicalize call: `fold_addi` const-folds `2 + 3` to `5`, then
        // the same fold (on the outer addi) erases `_ + 0`.
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
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        // muli %s, %x stays — only one operand is const.
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 1);
    }
}
