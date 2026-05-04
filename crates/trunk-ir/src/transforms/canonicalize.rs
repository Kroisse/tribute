//! Canonicalization pass for TrunkIR.
//!
//! Greedy fixed-point pass that folds operations into a canonical form.
//! Each dialect contributes per-op `fold` functions registered via
//! [`crate::register_canonicalize_fold!`]; the pass discovers them at
//! startup time through `inventory` rather than referencing each dialect
//! by name. As a result this module knows nothing about which dialects
//! exist — adding a new dialect's folds requires no edit here.
//!
//! The driver dispatches each visited op to its fold (if any) by
//! `(dialect, op_name)` HashMap lookup. A fold returns either a value
//! to forward (RAUW the op's result) or an `Attribute` that the driver
//! materializes as `arith.const` of the op's result type.
//!
//! Multi-op rewrites that don't fit the fold shape (e.g. region splice
//! for `scf.if(const)`) register as full [`RewritePattern`]s via
//! [`crate::register_canonicalize_pattern!`] instead.
//!
//! Currently registered (across `arith`, `core`):
//!
//! - `arith.addi`/`subi`/`muli` — integer identity (`x+0`, `x-0`,
//!   `x*0`, `x*1`) and constant folding at the result type's bit-width.
//! - `core.unrealized_conversion_cast` — identity (same source/target
//!   type) and round-trip elimination (`A → B → A`).
//!
//! Float and div/rem folds are deferred until each one's edge cases
//! (NaN/-0.0, division-by-zero) are pinned down.

use std::collections::HashMap;

use crate::context::IrContext;
use crate::dialect::arith;
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
/// registrations for the same (dialect, op_name) panic via `assert!`
/// in all build profiles — silently overwriting one fold with another
/// is always a bug.
pub struct FoldDispatchPattern {
    table: HashMap<(Symbol, Symbol), FoldFn>,
}

impl FoldDispatchPattern {
    /// Build a dispatcher from an iterator of `(dialect, op_name, fold)`
    /// triples. Panics on duplicate keys (in release builds too — see
    /// type-level docs).
    pub fn from_folds(folds: impl IntoIterator<Item = (Symbol, Symbol, FoldFn)>) -> Self {
        let mut table = HashMap::new();
        for (dialect, op_name, fold) in folds {
            assert!(
                table.insert((dialect, op_name), fold).is_none(),
                "duplicate canonicalize fold for {dialect}.{op_name}",
            );
        }
        Self { table }
    }

    /// Build a dispatcher from every fold registered via
    /// [`crate::register_canonicalize_fold!`] across the workspace.
    pub fn from_inventory() -> Self {
        Self::from_folds(folds_from_inventory())
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
// Inventory registration
// =========================================================================

/// One inventory entry registering a per-op fold function. Submitted via
/// [`crate::register_canonicalize_fold!`] — never constructed directly
/// from user code.
pub struct CanonicalizeFold {
    pub dialect: &'static str,
    pub op_name: &'static str,
    pub fold: FoldFn,
}
inventory::collect!(CanonicalizeFold);

/// One inventory entry registering a full `RewritePattern` for the
/// canonicalize pass — used as an escape hatch for rewrites that don't
/// fit the fold shape (e.g. multi-op region splicing). Submitted via
/// [`crate::register_canonicalize_pattern!`].
///
/// `make` is a `fn` (not a closure) so the entry meets `inventory`'s
/// `'static + Sync` bound. The function builds a fresh boxed pattern
/// every time the pass runs; any state should be in the pattern itself,
/// not captured.
///
/// **Determinism warning**: `inventory::iter` order is unspecified
/// (linker-dependent). With a single registered pattern this is
/// harmless; the moment a *second* `CanonicalizePattern` is registered,
/// add explicit ordering (e.g. `priority: i32` + `name: &'static str`
/// fields and sort by `(priority, name)` in `canonicalize()`) so the
/// first-match-wins behavior of `PatternApplicator` doesn't drift
/// across builds. Folds are dispatched by op key in
/// `FoldDispatchPattern`, so the ordering issue applies only here.
pub struct CanonicalizePattern {
    pub make: fn() -> Box<dyn RewritePattern>,
}
inventory::collect!(CanonicalizePattern);

/// Iterate every fold registered via inventory, keyed by interned
/// `(dialect, op_name)` symbols ready for [`FoldDispatchPattern::from_folds`].
fn folds_from_inventory() -> impl Iterator<Item = (Symbol, Symbol, FoldFn)> {
    inventory::iter::<CanonicalizeFold>.into_iter().map(|reg| {
        (
            Symbol::from_dynamic(reg.dialect),
            Symbol::from_dynamic(reg.op_name),
            reg.fold,
        )
    })
}

/// Iterate inventory folds whose dialect matches `dialect`. Used by
/// per-dialect test helpers to keep their assertions isolated from
/// other dialects' folds (folds dispatch by op key, so cross-dialect
/// interference is unlikely, but filtering keeps tests honest as more
/// dialects add folds).
#[cfg(test)]
pub(crate) fn folds_for_dialect(
    dialect: &'static str,
) -> impl Iterator<Item = (Symbol, Symbol, FoldFn)> {
    inventory::iter::<CanonicalizeFold>
        .into_iter()
        .filter(move |reg| reg.dialect == dialect)
        .map(|reg| {
            (
                Symbol::from_dynamic(reg.dialect),
                Symbol::from_dynamic(reg.op_name),
                reg.fold,
            )
        })
}

/// Register a per-op fold for the canonicalize pass.
///
/// # Example
/// ```text
/// register_canonicalize_fold!(arith.addi => fold_addi);
/// register_canonicalize_fold!(core.unrealized_conversion_cast => fold_uncc);
/// ```
///
/// The dialect and op-name idents are stringified (handling raw
/// identifiers like `r#const` correctly via `raw_ident_str!`). `$fold`
/// is a path to a function with signature `fn(&IrContext, OpRef) ->
/// Option<FoldResult>`.
#[macro_export]
macro_rules! register_canonicalize_fold {
    ($dialect:ident . $op_name:ident => $fold:path) => {
        ::inventory::submit! {
            $crate::transforms::canonicalize::CanonicalizeFold {
                dialect: $crate::raw_ident_str!($dialect),
                op_name: $crate::raw_ident_str!($op_name),
                fold: $fold,
            }
        }
    };
}

/// Register a non-fold `RewritePattern` for the canonicalize pass.
/// Use when the rewrite needs more than the fold shape allows
/// (e.g. modifying multiple ops or splicing regions).
///
/// # Example
/// ```text
/// fn make_if_const_fold() -> Box<dyn RewritePattern> { Box::new(IfConstFold) }
/// register_canonicalize_pattern!(make_if_const_fold);
/// ```
#[macro_export]
macro_rules! register_canonicalize_pattern {
    ($make:path) => {
        ::inventory::submit! {
            $crate::transforms::canonicalize::CanonicalizePattern { make: $make }
        }
    };
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
pub struct CanonicalizeResult {
    pub iterations: usize,
    pub total_changes: usize,
    pub reached_fixpoint: bool,
}

/// Run canonicalization to a fixed point on `module`.
pub fn canonicalize(ctx: &mut IrContext, module: Module) -> CanonicalizeResult {
    let mut applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern_box(Box::new(FoldDispatchPattern::from_inventory()));
    for entry in inventory::iter::<CanonicalizePattern> {
        applicator = applicator.add_pattern_box((entry.make)());
    }
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

    #[test]
    fn if_const_collaborates_with_arith_folds() {
        // `scf.if(true) { yield (x + 0) } { yield x }` should collapse to
        // `func.return %x` in one canonicalize call, exercising both the
        // pattern-based escape hatch (`IfConstFold` splices the then
        // region) and the per-op folds (`fold_addi` erases the `_ + 0`).
        // The dead else region's `arith.muli` is dropped along with the
        // erased `scf.if` op.
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
        assert_eq!(count_ops(&ctx, module, "scf", "if"), 0);
        // arith.addi was folded away (`x + 0` → `x`).
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        // The else region's `arith.muli` was orphaned with the erased
        // `scf.if`; it should not appear in the walked module.
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 0);
    }
}
