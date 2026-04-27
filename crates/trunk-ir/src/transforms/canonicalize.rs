//! Canonicalization pass for TrunkIR.
//!
//! Greedy fixed-point pass that folds operations into a canonical form via
//! local rewrite patterns. The first iteration ships a small set of integer
//! identity rewrites that the inliner produces in bulk:
//!
//! - `arith.addi %x, 0` / `arith.addi 0, %x` → `%x`
//! - `arith.muli %x, 1` / `arith.muli 1, %x` → `%x`
//! - `arith.muli %x, 0` / `arith.muli 0, %x` → `arith.const 0`
//!
//! Float / div-rem / scf / `unrealized_conversion_cast` patterns live in
//! follow-up PRs once each one's semantics are pinned down.
//!
//! Patterns are language-agnostic: this module sits in `trunk-ir`, not
//! `tribute-passes`. Each pattern self-filters by dialect/op-name inside
//! `match_and_rewrite` (the applicator does no central op-kind dispatch).

use crate::context::IrContext;
use crate::dialect::arith;
use crate::refs::{OpRef, ValueDef, ValueRef};
use crate::rewrite::{Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter};
use crate::symbol::Symbol;
use crate::types::Attribute;

/// Outcome of one `canonicalize` invocation.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalizeResult {
    pub iterations: usize,
    pub total_changes: usize,
    pub reached_fixpoint: bool,
}

/// Run canonicalization to a fixed point on `module`.
pub fn canonicalize(ctx: &mut IrContext, module: Module) -> CanonicalizeResult {
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(AddZeroFold)
        .add_pattern(MulOneFold)
        .add_pattern(MulZeroFold);
    let result = applicator.apply_partial(ctx, module);
    CanonicalizeResult {
        iterations: result.iterations,
        total_changes: result.total_changes,
        reached_fixpoint: result.reached_fixpoint,
    }
}

// =========================================================================
// Patterns
// =========================================================================

/// `arith.addi %x, 0` / `arith.addi 0, %x` → `%x`.
pub struct AddZeroFold;

impl RewritePattern for AddZeroFold {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !op_is(ctx, op, ARITH, ADDI) {
            return false;
        }
        let Some((other, _)) = find_const_int_operand(ctx, op, 0) else {
            return false;
        };
        rewriter.erase_op(vec![other]);
        true
    }

    fn name(&self) -> &'static str {
        "AddZeroFold"
    }
}

/// `arith.muli %x, 1` / `arith.muli 1, %x` → `%x`.
pub struct MulOneFold;

impl RewritePattern for MulOneFold {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !op_is(ctx, op, ARITH, MULI) {
            return false;
        }
        let Some((other, _)) = find_const_int_operand(ctx, op, 1) else {
            return false;
        };
        rewriter.erase_op(vec![other]);
        true
    }

    fn name(&self) -> &'static str {
        "MulOneFold"
    }
}

/// `arith.muli %x, 0` / `arith.muli 0, %x` → `arith.const {value = 0}`.
pub struct MulZeroFold;

impl RewritePattern for MulZeroFold {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !op_is(ctx, op, ARITH, MULI) {
            return false;
        }
        if find_const_int_operand(ctx, op, 0).is_none() {
            return false;
        }
        let loc = ctx.op(op).location;
        let result_types = ctx.op_result_types(op).to_vec();
        let result_ty = match result_types.as_slice() {
            [t] => *t,
            _ => return false,
        };
        let zero = arith::r#const(ctx, loc, result_ty, Attribute::Int(0));
        rewriter.replace_op(zero.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "MulZeroFold"
    }
}

// =========================================================================
// Helpers
// =========================================================================

const ARITH: &str = "arith";
const ADDI: &str = "addi";
const MULI: &str = "muli";
const CONST: &str = "const";
const VALUE: &str = "value";

fn op_is(ctx: &IrContext, op: OpRef, dialect: &'static str, name: &'static str) -> bool {
    let data = ctx.op(op);
    data.dialect == Symbol::new(dialect) && data.name == Symbol::new(name)
}

/// If `op` has exactly two operands and one of them is produced by an
/// `arith.const {value = Int(target)}`, return `(other_operand, const_op)`.
/// `other_operand` is the operand *not* matched against the constant.
fn find_const_int_operand(ctx: &IrContext, op: OpRef, target: i128) -> Option<(ValueRef, OpRef)> {
    let operands = ctx.op_operands(op);
    if operands.len() != 2 {
        return None;
    }
    let lhs = operands[0];
    let rhs = operands[1];

    if let Some(c) = const_int_producer(ctx, rhs, target) {
        return Some((lhs, c));
    }
    if let Some(c) = const_int_producer(ctx, lhs, target) {
        return Some((rhs, c));
    }
    None
}

fn const_int_producer(ctx: &IrContext, value: ValueRef, target: i128) -> Option<OpRef> {
    let producer = match ctx.value_def(value) {
        ValueDef::OpResult(op, _) => op,
        ValueDef::BlockArg(_, _) => return None,
    };
    if !op_is(ctx, producer, ARITH, CONST) {
        return None;
    }
    let value_sym: Symbol = Symbol::new(VALUE);
    match ctx.op(producer).attributes.get(&value_sym) {
        Some(Attribute::Int(v)) if *v == target => Some(producer),
        _ => None,
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::IrContext;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
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
    fn add_zero_fold_rhs_zero() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.addi %x, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn add_zero_fold_lhs_zero() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.addi %z, %x : core.i32
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
    fn add_zero_fold_does_not_match_when_neither_operand_is_const_zero() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32, %y: core.i32) -> core.i32 {
    %r = arith.addi %x, %y : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
    }

    #[test]
    fn mul_one_fold_rhs_one() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %one = arith.const {value = 1} : core.i32
    %r = arith.muli %x, %one : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 0);
    }

    #[test]
    fn mul_zero_fold_replaces_with_const_zero() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.muli %x, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        // muli is gone, two const ops remain: the original `%z` plus the
        // freshly-built zero replacing the multiplication result.
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 0);
        assert_eq!(count_ops(&ctx, module, "arith", "const"), 2);
        insta::assert_snapshot!(print_module(&ctx, module.op()));
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
}
