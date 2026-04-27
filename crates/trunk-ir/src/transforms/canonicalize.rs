//! Canonicalization pass for TrunkIR.
//!
//! Greedy fixed-point pass that folds operations into a canonical form via
//! local rewrite patterns. The pattern set so far covers integer identities
//! and constant folding for `addi`/`subi`/`muli`:
//!
//! - `arith.addi %x, 0` / `arith.addi 0, %x` → `%x`
//! - `arith.subi %x, 0` → `%x`
//!   (`arith.subi 0, %x` is `arith.negi`, handled separately.)
//! - `arith.muli %x, 1` / `arith.muli 1, %x` → `%x`
//! - `arith.muli %x, 0` / `arith.muli 0, %x` → `arith.const 0`
//! - `arith.addi const(a), const(b)` / `subi` / `muli` → `arith.const`
//!   (i128 wrapping arithmetic — the bit-width is enforced by the result
//!   type at codegen, so wrap is the conservative semantic preservation.)
//!
//! Float, div/rem, scf, and `unrealized_conversion_cast` patterns live in
//! follow-up PRs once each one's semantics are pinned down (NaN/-0.0,
//! division-by-zero, region splice, materialization vs identity).
//!
//! Patterns are language-agnostic: this module sits in `trunk-ir`, not
//! `tribute-passes`. Each pattern self-filters by dialect/op-name inside
//! `match_and_rewrite` (the applicator does no central op-kind dispatch).
//! Per-dialect pattern registries are tracked separately in #690.

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
        .add_pattern(SubZeroFold)
        .add_pattern(MulOneFold)
        .add_pattern(MulZeroFold)
        .add_pattern(IntConstFold);
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

/// `arith.subi %x, 0` → `%x`.
///
/// `arith.subi 0, %x` is the `negi` semantic and is left for a separate
/// pattern to keep the rewrite direction unambiguous.
pub struct SubZeroFold;

impl RewritePattern for SubZeroFold {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !op_is(ctx, op, ARITH, SUBI) {
            return false;
        }
        let operands = ctx.op_operands(op);
        if operands.len() != 2 {
            return false;
        }
        let lhs = operands[0];
        let rhs = operands[1];
        if const_int_producer(ctx, rhs, 0).is_none() {
            return false;
        }
        rewriter.erase_op(vec![lhs]);
        true
    }

    fn name(&self) -> &'static str {
        "SubZeroFold"
    }
}

/// Constant-fold integer binary ops with two `arith.const` operands.
///
/// Folds `addi`/`subi`/`muli` using i128 wrapping arithmetic. The result
/// type is preserved; bit-width semantics are enforced at codegen via type
/// information, so a wrap inside i128 is conservative wrt every concrete
/// width.
///
/// `divsi` / `divui` / `remsi` / `remui` are intentionally excluded — they
/// require a division-by-zero guard that is its own design decision.
pub struct IntConstFold;

impl RewritePattern for IntConstFold {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let fold = if op_is(ctx, op, ARITH, ADDI) {
            i128::wrapping_add
        } else if op_is(ctx, op, ARITH, SUBI) {
            i128::wrapping_sub
        } else if op_is(ctx, op, ARITH, MULI) {
            i128::wrapping_mul
        } else {
            return false;
        };

        let operands = ctx.op_operands(op);
        if operands.len() != 2 {
            return false;
        }
        let lhs = operands[0];
        let rhs = operands[1];
        let lhs_val = const_int_value(ctx, lhs);
        let rhs_val = const_int_value(ctx, rhs);
        let (Some(a), Some(b)) = (lhs_val, rhs_val) else {
            return false;
        };

        let result_types = ctx.op_result_types(op).to_vec();
        let result_ty = match result_types.as_slice() {
            [t] => *t,
            _ => return false,
        };
        let loc = ctx.op(op).location;
        let folded = fold(a, b);
        let new_const = arith::r#const(ctx, loc, result_ty, Attribute::Int(folded));
        rewriter.replace_op(new_const.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "IntConstFold"
    }
}

// =========================================================================
// Helpers
// =========================================================================

const ARITH: &str = "arith";
const ADDI: &str = "addi";
const SUBI: &str = "subi";
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
    let (producer, v) = const_int_def(ctx, value)?;
    if v == target { Some(producer) } else { None }
}

/// If `value` is the result of an `arith.const {value = Int(_)}`, return
/// its raw integer attribute.
fn const_int_value(ctx: &IrContext, value: ValueRef) -> Option<i128> {
    const_int_def(ctx, value).map(|(_, v)| v)
}

fn const_int_def(ctx: &IrContext, value: ValueRef) -> Option<(OpRef, i128)> {
    let producer = match ctx.value_def(value) {
        ValueDef::OpResult(op, _) => op,
        ValueDef::BlockArg(_, _) => return None,
    };
    if !op_is(ctx, producer, ARITH, CONST) {
        return None;
    }
    let value_sym: Symbol = Symbol::new(VALUE);
    match ctx.op(producer).attributes.get(&value_sym) {
        Some(Attribute::Int(v)) => Some((producer, *v)),
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

    #[test]
    fn sub_zero_fold_rhs_zero() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.subi %x, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "subi"), 0);
    }

    #[test]
    fn sub_zero_fold_does_not_match_lhs_zero() {
        // `arith.subi 0, %x` is the negi semantic — it must not be folded
        // by SubZeroFold. (Const fold also can't fire because %x is a
        // block argument.)
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.subi %z, %x : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "subi"), 1);
    }

    #[test]
    fn int_const_fold_addi() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 2} : core.i32
    %b = arith.const {value = 3} : core.i32
    %r = arith.addi %a, %b : core.i32
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
    fn int_const_fold_subi_and_muli() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 10} : core.i32
    %b = arith.const {value = 4} : core.i32
    %s = arith.subi %a, %b : core.i32
    %c = arith.const {value = 3} : core.i32
    %r = arith.muli %s, %c : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.reached_fixpoint);
        assert_eq!(count_ops(&ctx, module, "arith", "subi"), 0);
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 0);
    }

    #[test]
    fn int_const_fold_wraps_on_overflow() {
        // i128::MAX + 1 wraps to i128::MIN — this is the conservative
        // semantic preservation choice (codegen carries the actual
        // bit-width). The pattern must produce a fresh const without
        // panicking.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 170141183460469231731687303715884105727} : core.i32
    %b = arith.const {value = 1} : core.i32
    %r = arith.addi %a, %b : core.i32
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
    fn int_const_fold_does_not_match_when_only_one_operand_is_const() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %a = arith.const {value = 7} : core.i32
    %r = arith.addi %x, %a : core.i32
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
        // The outer `addi _, 0` is erased; the inner `2 + 3` is folded
        // into a single const.
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        // muli %s, %x stays — only one operand is const.
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 1);
    }
}
