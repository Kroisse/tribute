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
// Canonicalization patterns
//
// Owned by this dialect and aggregated by `transforms::canonicalize` via
// [`canonicalization_patterns`]. Each pattern self-filters by op name in
// `match_and_rewrite` (the applicator does no central op-kind dispatch),
// so adding a new arith pattern only requires touching this file.
// =========================================================================

use crate::context::IrContext;
use crate::refs::{OpRef, TypeRef, ValueDef, ValueRef};
use crate::rewrite::{PatternRewriter, RewritePattern};
use crate::symbol::Symbol;
use crate::types::Attribute;

/// Patterns this dialect contributes to `transforms::canonicalize`.
pub fn canonicalization_patterns() -> Vec<Box<dyn RewritePattern>> {
    vec![
        Box::new(AddZeroFold),
        Box::new(SubZeroFold),
        Box::new(MulOneFold),
        Box::new(MulZeroFold),
        Box::new(IntConstFold),
    ]
}

/// `arith.addi %x, 0` / `arith.addi 0, %x` → `%x`.
pub struct AddZeroFold;

impl RewritePattern for AddZeroFold {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !op_is(ctx, op, "arith", "addi") {
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
        if !op_is(ctx, op, "arith", "muli") {
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
        if !op_is(ctx, op, "arith", "muli") {
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
        let zero = r#const(ctx, loc, result_ty, Attribute::Int(0));
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
        if !op_is(ctx, op, "arith", "subi") {
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
/// Folds `addi`/`subi`/`muli` at the result type's bit-width. The raw
/// operation is performed in i128 (wide enough for any width up to i64
/// without intermediate overflow), then sign-truncated to fit the result
/// type — so e.g. `arith.addi const(i32::MAX), const(1) : core.i32`
/// folds to `arith.const i32::MIN`, matching codegen semantics.
///
/// Result types must be `core.i{N}` with `1 <= N <= 128`. Other
/// dialects and exotic integer types fall through unchanged.
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
        let fold = if op_is(ctx, op, "arith", "addi") {
            i128::wrapping_add
        } else if op_is(ctx, op, "arith", "subi") {
            i128::wrapping_sub
        } else if op_is(ctx, op, "arith", "muli") {
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
        let Some(width) = core_int_width(ctx, result_ty) else {
            return false;
        };
        let folded = wrap_signed_to_width(fold(a, b), width);
        let loc = ctx.op(op).location;
        let new_const = r#const(ctx, loc, result_ty, Attribute::Int(folded));
        rewriter.replace_op(new_const.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "IntConstFold"
    }
}

// =========================================================================
// Pattern helpers (private to this module)
// =========================================================================

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
    if !op_is(ctx, producer, "arith", "const") {
        return None;
    }
    let value_sym: Symbol = Symbol::new("value");
    match ctx.op(producer).attributes.get(&value_sym) {
        Some(Attribute::Int(v)) => Some((producer, *v)),
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
    use crate::rewrite::{Module, PatternApplicator, TypeConverter};
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    /// Run only this dialect's patterns on `module`. Returns the
    /// applicator's partial-application result for assertions.
    fn run_arith_patterns(ctx: &mut IrContext, module: Module) -> crate::rewrite::ApplyResult {
        let applicator = canonicalization_patterns().into_iter().fold(
            PatternApplicator::new(TypeConverter::new()),
            PatternApplicator::add_pattern_box,
        );
        applicator.apply_partial(ctx, module)
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
