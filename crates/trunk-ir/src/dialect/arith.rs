//! Arena-based arith dialect.
//!
//! Operations are split by type category following MLIR/Cranelift conventions:
//! - Integer arithmetic: `addi`, `subi`, `muli`, `divsi`/`divui`, `remsi`/`remui`, `negi`
//! - Float arithmetic: `addf`, `subf`, `mulf`, `divf`, `negf`
//! - Integer comparison: `cmpi` with predicate attribute (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
//! - Float comparison: `cmpf` with predicate attribute (oeq, one, olt, ole, ogt, oge)

// === Pure operation registrations ===
crate::register_pure_op!(arith.r#const);

// Integer arithmetic
crate::register_pure_op!(arith.addi);
crate::register_pure_op!(arith.subi);
crate::register_pure_op!(arith.muli);
crate::register_pure_op!(arith.divsi);
crate::register_pure_op!(arith.divui);
crate::register_pure_op!(arith.remsi);
crate::register_pure_op!(arith.remui);
crate::register_pure_op!(arith.negi);

// Float arithmetic
crate::register_pure_op!(arith.addf);
crate::register_pure_op!(arith.subf);
crate::register_pure_op!(arith.mulf);
crate::register_pure_op!(arith.divf);
crate::register_pure_op!(arith.negf);

// Comparisons
crate::register_pure_op!(arith.cmpi);
crate::register_pure_op!(arith.cmpf);

// Bitwise (integer-only, unchanged)
crate::register_pure_op!(arith.and);
crate::register_pure_op!(arith.or);
crate::register_pure_op!(arith.xor);
crate::register_pure_op!(arith.shl);
crate::register_pure_op!(arith.shr);
crate::register_pure_op!(arith.shru);

// Conversions (unchanged)
crate::register_pure_op!(arith.cast);
crate::register_pure_op!(arith.trunc);
crate::register_pure_op!(arith.extend);
crate::register_pure_op!(arith.convert);

#[crate::dialect(crate = crate)]
mod arith {
    #[attr(value: any)]
    fn r#const() -> result {}

    // Integer arithmetic
    fn addi(lhs: (), rhs: ()) -> result {}
    fn subi(lhs: (), rhs: ()) -> result {}
    fn muli(lhs: (), rhs: ()) -> result {}
    fn divsi(lhs: (), rhs: ()) -> result {}
    fn divui(lhs: (), rhs: ()) -> result {}
    fn remsi(lhs: (), rhs: ()) -> result {}
    fn remui(lhs: (), rhs: ()) -> result {}
    fn negi(operand: ()) -> result {}

    // Float arithmetic
    fn addf(lhs: (), rhs: ()) -> result {}
    fn subf(lhs: (), rhs: ()) -> result {}
    fn mulf(lhs: (), rhs: ()) -> result {}
    fn divf(lhs: (), rhs: ()) -> result {}
    fn negf(operand: ()) -> result {}

    // Comparisons
    #[attr(predicate: Symbol)]
    fn cmpi(lhs: (), rhs: ()) -> result {}

    #[attr(predicate: Symbol)]
    fn cmpf(lhs: (), rhs: ()) -> result {}

    // Bitwise (integer-only)
    fn and(lhs: (), rhs: ()) -> result {}
    fn or(lhs: (), rhs: ()) -> result {}
    fn xor(lhs: (), rhs: ()) -> result {}
    fn shl(value: (), amount: ()) -> result {}
    fn shr(value: (), amount: ()) -> result {}
    fn shru(value: (), amount: ()) -> result {}

    // Conversions
    fn cast(operand: ()) -> result {}
    fn trunc(operand: ()) -> result {}
    fn extend(operand: ()) -> result {}
    fn convert(operand: ()) -> result {}
}

// =========================================================================
// Canonicalization folds
//
// Owned by this dialect and aggregated by `transforms::canonicalize` via
// [`folds`]. Each fold returns a `FoldResult` describing how the
// pass should rewrite the op (or `None` to leave it alone). The driver
// dispatches by (dialect, op_name) so folds don't self-filter.
// =========================================================================

use crate::context::IrContext;
use crate::ops::DialectOp;
use crate::refs::{OpRef, TypeRef, ValueDef, ValueRef};
use crate::symbol::Symbol;
use crate::transforms::canonicalize::FoldResult;
use crate::types::Attribute;

// Folds this dialect contributes to `transforms::canonicalize`, registered
// via `inventory`. The pass discovers them at startup; no manual aggregation
// in `canonicalize.rs` is needed.
crate::register_canonicalize_fold!(arith.addi => fold_addi);
crate::register_canonicalize_fold!(arith.subi => fold_subi);
crate::register_canonicalize_fold!(arith.muli => fold_muli);

/// `arith.addi` folds:
/// - `x + 0` / `0 + x` → `x`
/// - `const(a) + const(b)` → `const(wrap(a+b))` at the result width
pub(crate) fn fold_addi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) {
        return Some(FoldResult::Forward(lhs));
    }
    if const_int_value(ctx, lhs) == Some(0) {
        return Some(FoldResult::Forward(rhs));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = core_int_width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_add(b), width),
    )))
}

/// `arith.subi` folds:
/// - `x - 0` → `x`. (`0 - x` is the `negi` semantic and is left for a
///   separate fold so the rewrite direction stays unambiguous.)
/// - `const(a) - const(b)` → `const(wrap(a-b))` at the result width.
pub(crate) fn fold_subi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) {
        return Some(FoldResult::Forward(lhs));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = core_int_width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_sub(b), width),
    )))
}

/// `arith.muli` folds:
/// - `x * 0` / `0 * x` → `const 0` (checked before x*1 to short-circuit).
/// - `x * 1` / `1 * x` → `x`.
/// - `const(a) * const(b)` → `const(wrap(a*b))` at the result width.
pub(crate) fn fold_muli(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) || const_int_value(ctx, lhs) == Some(0) {
        return Some(FoldResult::ArithConst(Attribute::Int(0)));
    }
    if const_int_value(ctx, rhs) == Some(1) {
        return Some(FoldResult::Forward(lhs));
    }
    if const_int_value(ctx, lhs) == Some(1) {
        return Some(FoldResult::Forward(rhs));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = core_int_width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_mul(b), width),
    )))
}

// =========================================================================
// Fold helpers (private to this module)
// =========================================================================

/// Extract `(lhs, rhs)` from a binary op. `None` if the op doesn't have
/// exactly two operands.
fn two_operands(ctx: &IrContext, op: OpRef) -> Option<(ValueRef, ValueRef)> {
    match ctx.op_operands(op) {
        [lhs, rhs] => Some((*lhs, *rhs)),
        _ => None,
    }
}

/// Extract the op's single result type. `None` if the op has zero or
/// multiple results.
fn single_result_type(ctx: &IrContext, op: OpRef) -> Option<TypeRef> {
    match ctx.op_result_types(op) {
        [t] => Some(*t),
        _ => None,
    }
}

/// If `value` is the result of an `arith.const {value = Int(_)}`, return
/// its raw integer attribute.
///
/// Uses the dialect-generated `Const` typed wrapper so the match is
/// rename-safe and tied to the schema rather than literal `"arith"` /
/// `"const"` / `"value"` strings.
pub(crate) fn const_int_value(ctx: &IrContext, value: ValueRef) -> Option<i128> {
    let producer = match ctx.value_def(value) {
        ValueDef::OpResult(op, _) => op,
        ValueDef::BlockArg(_, _) => return None,
    };
    let const_op = Const::from_op(ctx, producer).ok()?;
    match const_op.value(ctx) {
        Attribute::Int(v) => Some(v),
        _ => None,
    }
}

/// If `ty` is `core.i{N}` for some `1 <= N <= 128`, return `N`.
///
/// Returns `None` for other dialects, parameterized types, types
/// carrying attributes, names that don't follow the `i{N}` shape, or
/// widths outside `[1, 128]` (the upper bound is what
/// `wrap_signed_to_width` can represent in i128).
fn core_int_width(ctx: &IrContext, ty: TypeRef) -> Option<u32> {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("core") || !data.params.is_empty() || !data.attrs.is_empty() {
        return None;
    }
    data.name.with_str(|s| {
        let digits = s.strip_prefix('i')?;
        // u32::from_str rejects empty input and any sign character, so
        // `i`, `i+32`, `i-1` all fail here.
        let width: u32 = digits.parse().ok()?;
        (1..=128).contains(&width).then_some(width)
    })
}

/// Truncate `value` to `width` bits, sign-extended back to i128.
///
/// `width` must satisfy `1 <= width <= 128`. The implementation uses
/// arithmetic shift, which sign-extends from the top bit of the kept
/// portion — matching the two's-complement wrap semantics of every
/// concrete integer width supported here.
fn wrap_signed_to_width(value: i128, width: u32) -> i128 {
    debug_assert!((1..=128).contains(&width));
    if width == 128 {
        return value;
    }
    let shift = 128 - width;
    (value << shift) >> shift
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};
    use crate::transforms::canonicalize::{FoldDispatchPattern, folds_for_dialect};
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    /// Run only this dialect's folds on `module` via a single
    /// [`FoldDispatchPattern`]. Filters the inventory by dialect so
    /// per-fold tests stay isolated from other dialects' folds even
    /// though the production `canonicalize` pass aggregates everyone.
    fn run_arith_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let dispatcher = FoldDispatchPattern::from_folds(folds_for_dialect("arith"));
        PatternApplicator::new(TypeConverter::new())
            .add_pattern_box(Box::new(dispatcher))
            .apply_partial(ctx, module)
    }

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

    /// Walk `module` and return the `Attribute::Int` carried by the
    /// `arith.const` that the first `func.return` returns. Returns
    /// `None` if no return is found, or if its operand is not produced
    /// by a const-int op. Lets value-shaped tests assert the folded
    /// constant directly without going through textual snapshots.
    fn return_value_int_const(ctx: &IrContext, module: Module) -> Option<i128> {
        let func_return_dialect = Symbol::new("func");
        let return_name = Symbol::new("return");
        let mut found = None;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if found.is_some() {
                return ControlFlow::Continue(WalkAction::Advance);
            }
            let data = ctx.op(op);
            if data.dialect == func_return_dialect
                && data.name == return_name
                && let Some(&v) = ctx.op_operands(op).first()
            {
                found = const_int_value(ctx, v);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found
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

        let result = run_arith_patterns(&mut ctx, module);
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

        let result = run_arith_patterns(&mut ctx, module);
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

        let result = run_arith_patterns(&mut ctx, module);
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

        let result = run_arith_patterns(&mut ctx, module);
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

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 0);
        assert_eq!(count_ops(&ctx, module, "arith", "const"), 2);
        insta::assert_snapshot!(print_module(&ctx, module.op()));
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

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "subi"), 0);
    }

    #[test]
    fn sub_zero_fold_does_not_match_lhs_zero() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.subi %z, %x : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
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

        let result = run_arith_patterns(&mut ctx, module);
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

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.reached_fixpoint);
        assert_eq!(count_ops(&ctx, module, "arith", "subi"), 0);
        assert_eq!(count_ops(&ctx, module, "arith", "muli"), 0);
    }

    #[test]
    fn int_const_fold_wraps_at_i32_width() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 2147483647} : core.i32
    %b = arith.const {value = 1} : core.i32
    %r = arith.addi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        assert_eq!(return_value_int_const(&ctx, module), Some(i32::MIN as i128));
    }

    #[test]
    fn int_const_fold_wraps_at_i64_width() {
        let input = r#"core.module @test {
  func.func @f() -> core.i64 {
    %a = arith.const {value = 9223372036854775807} : core.i64
    %b = arith.const {value = 1} : core.i64
    %r = arith.addi %a, %b : core.i64
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        assert_eq!(return_value_int_const(&ctx, module), Some(i64::MIN as i128));
    }

    #[test]
    fn int_const_fold_accepts_nonstandard_width() {
        let input = r#"core.module @test {
  func.func @f() -> core.i7 {
    %a = arith.const {value = 60} : core.i7
    %b = arith.const {value = 10} : core.i7
    %r = arith.addi %a, %b : core.i7
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        // 60 + 10 = 70, which exceeds i7::MAX (63), wraps to -58.
        assert_eq!(return_value_int_const(&ctx, module), Some(-58));
    }

    #[test]
    fn int_const_fold_skips_widths_above_128() {
        let input = r#"core.module @test {
  func.func @f() -> core.i129 {
    %a = arith.const {value = 1} : core.i129
    %b = arith.const {value = 2} : core.i129
    %r = arith.addi %a, %b : core.i129
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
    }

    #[test]
    fn int_const_fold_skips_non_i_prefixed_type() {
        let input = r#"core.module @test {
  func.func @f() -> core.foo {
    %a = arith.const {value = 1} : core.foo
    %b = arith.const {value = 2} : core.foo
    %r = arith.addi %a, %b : core.foo
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
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

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
    }

    #[test]
    fn wrap_signed_to_width_matches_two_complement_semantics() {
        assert_eq!(
            wrap_signed_to_width(i32::MAX as i128 + 1, 32),
            i32::MIN as i128
        );
        assert_eq!(
            wrap_signed_to_width(i32::MIN as i128 - 1, 32),
            i32::MAX as i128
        );
        assert_eq!(
            wrap_signed_to_width(i64::MAX as i128 + 1, 64),
            i64::MIN as i128
        );
        assert_eq!(wrap_signed_to_width(0, 32), 0);
        assert_eq!(wrap_signed_to_width(-1, 32), -1);
        assert_eq!(wrap_signed_to_width(0, 1), 0);
        assert_eq!(wrap_signed_to_width(1, 1), -1);
        assert_eq!(wrap_signed_to_width(-1, 1), -1);
        assert_eq!(wrap_signed_to_width(2, 1), 0);
    }
}
