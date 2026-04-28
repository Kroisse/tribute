//! Canonicalization pass for TrunkIR.
//!
//! Greedy fixed-point pass that folds operations into a canonical form via
//! local rewrite patterns. Patterns are owned by the dialect that defines
//! the ops they operate on, and aggregated here at pass-construction time.
//!
//! - `arith::canonicalization_patterns()` — integer identities (`x+0`,
//!   `x*1`, `x*0`, `x-0`) and constant folding for `addi`/`subi`/`muli` at
//!   the result type's bit-width.
//! - `core::canonicalization_patterns()` — `unrealized_conversion_cast`
//!   identity (same source/target type) and round-trip elimination
//!   (`A → B → A`).
//!
//! Adding a new pattern to an existing dialect requires editing only that
//! dialect's module. Adding a brand-new dialect's patterns is a one-line
//! `chain` here. Tracked in #690.
//!
//! Float, div/rem, and `scf` patterns live in follow-up PRs once each
//! one's semantics are pinned down (NaN/-0.0, division-by-zero, region
//! splice).

use crate::context::IrContext;
use crate::dialect::{arith, core as core_dialect};
use crate::rewrite::{Module, PatternApplicator, TypeConverter};

/// Outcome of one `canonicalize` invocation.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalizeResult {
    pub iterations: usize,
    pub total_changes: usize,
    pub reached_fixpoint: bool,
}

/// Run canonicalization to a fixed point on `module`.
pub fn canonicalize(ctx: &mut IrContext, module: Module) -> CanonicalizeResult {
    let applicator = arith::canonicalization_patterns()
        .into_iter()
        .chain(core_dialect::canonicalization_patterns())
        .fold(
            PatternApplicator::new(TypeConverter::new()),
            PatternApplicator::add_pattern_box,
        );
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
// iteration, and the collaboration between patterns from one or more
// dialects. Per-pattern positive/negative tests live alongside the
// patterns themselves in their respective dialect modules.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_test_module;
    use crate::symbol::Symbol;
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
        // canonicalize call: const fold turns `2 + 3` into `5`, then
        // AddZeroFold erases `_ + 0`.
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
