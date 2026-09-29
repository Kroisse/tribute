//! Arena-based arith dialect.
//!
//! Operations are split by type category following MLIR/Cranelift conventions:
//! - Integer arithmetic: `addi`, `subi`, `muli`, `divsi`/`divui`, `remsi`/`remui`, `negi`
//! - Float arithmetic: `addf`, `subf`, `mulf`, `divf`, `negf`
//! - Integer comparison: `cmpi` with predicate attribute (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
//! - Float comparison: `cmpf` with predicate attribute (oeq, une, olt, ole, ogt, oge)

// === Pure operation registrations ===
crate::register_pure_op!(Const);

// Integer arithmetic
crate::register_pure_op!(Addi);
crate::register_pure_op!(Subi);
crate::register_pure_op!(Muli);
crate::register_pure_op!(Divsi);
crate::register_pure_op!(Divui);
crate::register_pure_op!(Remsi);
crate::register_pure_op!(Remui);
crate::register_pure_op!(Negi);

// Float arithmetic
crate::register_pure_op!(Addf);
crate::register_pure_op!(Subf);
crate::register_pure_op!(Mulf);
crate::register_pure_op!(Divf);
crate::register_pure_op!(Negf);

// Comparisons
crate::register_pure_op!(Cmpi);
crate::register_pure_op!(Cmpf);

// Bitwise (integer-only, unchanged)
crate::register_pure_op!(And);
crate::register_pure_op!(Or);
crate::register_pure_op!(Xor);
crate::register_pure_op!(Shl);
crate::register_pure_op!(Shr);
crate::register_pure_op!(Shru);

// Conversions
crate::register_pure_op!(Extsi);
crate::register_pure_op!(Extui);
crate::register_pure_op!(Trunci);
crate::register_pure_op!(Sitofp);
crate::register_pure_op!(Uitofp);
crate::register_pure_op!(Extf);
crate::register_pure_op!(Truncf);
// fptosi/fptoui trap on out-of-range values and NaN, so they are not pure.

#[trunk_ir::dialect]
mod arith {
    #[verify]
    fn r#const(value: Attr<_>) -> Value<impl NumericLike> {}

    // Integer arithmetic
    fn addi<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn subi<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn muli<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn divsi<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn divui<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn remsi<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn remui<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn negi<T: IntegerLike>(operand: Value<T>) -> Value<T> {}

    // Float arithmetic
    fn addf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn subf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn mulf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn divf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn negf<T: FloatLike>(operand: Value<T>) -> Value<T> {}

    // Comparisons
    fn cmpi<T: IntegerLike>(predicate: Attr<Symbol>, lhs: Value<T>, rhs: Value<T>) -> Value<I1> {}

    #[verify]
    fn cmpf<T: FloatLike>(predicate: Attr<Symbol>, lhs: Value<T>, rhs: Value<T>) -> Value<I1> {}

    // Bitwise (integer-only)
    fn and<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn or<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn xor<T: IntegerLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    fn shl<T: IntegerLike>(value: Value<T>, amount: Value<T>) -> Value<T> {}
    fn shr<T: IntegerLike>(value: Value<T>, amount: Value<T>) -> Value<T> {}
    fn shru<T: IntegerLike>(value: Value<T>, amount: Value<T>) -> Value<T> {}

    // Conversions. Integers are signless; each sign-dependent conversion is
    // an explicit signed/unsigned pair.
    #[verify]
    fn extsi(operand: Value<impl IntegerLike>) -> Value<impl IntegerLike> {}
    #[verify]
    fn extui(operand: Value<impl IntegerLike>) -> Value<impl IntegerLike> {}
    #[verify]
    fn trunci(operand: Value<impl IntegerLike>) -> Value<impl IntegerLike> {}
    fn sitofp(operand: Value<impl IntegerLike>) -> Value<impl FloatLike> {}
    fn uitofp(operand: Value<impl IntegerLike>) -> Value<impl FloatLike> {}
    /// Traps on out-of-range values and NaN.
    fn fptosi(operand: Value<impl FloatLike>) -> Value<impl IntegerLike> {}
    /// Traps on out-of-range values and NaN.
    fn fptoui(operand: Value<impl FloatLike>) -> Value<impl IntegerLike> {}
    #[verify]
    fn extf(operand: Value<impl FloatLike>) -> Value<impl FloatLike> {}
    #[verify]
    fn truncf(operand: Value<impl FloatLike>) -> Value<impl FloatLike> {}
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
use crate::dialect::core::{FloatLike, I1, IntegerLike, NumericLike};
use crate::ops::DialectOp;
use crate::refs::{OpRef, TypeRef, ValueDef, ValueRef};
use crate::transforms::canonicalize::FoldResult;
use crate::types::Attribute;
use itertools::Itertools;

/// `arith.cmpf` predicates that every backend lowers.
const SUPPORTED_CMPF_PREDICATES: [&str; 6] = ["oeq", "une", "olt", "ole", "ogt", "oge"];

impl crate::ops::Verify for Const {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let ty = ctx.value_ty(self.result(ctx));
        let accepted = match self.value(ctx) {
            Attribute::Int(_) => IntegerLike::matches(ctx, ty),
            Attribute::Bool(_) => I1::matches(ctx, ty),
            Attribute::FloatBits(_) => FloatLike::matches(ctx, ty),
            _ => false,
        };
        if accepted {
            Ok(())
        } else {
            Err(format!(
                "value {:?} does not fit result type {}",
                self.value(ctx),
                crate::printer::print_type(ctx, ty)
            ))
        }
    }
}

impl crate::ops::Verify for Cmpf {
    fn verify(self, ctx: &IrContext) -> Result<(), String> {
        let predicate = self.predicate(ctx);
        if predicate.with_str(|name| SUPPORTED_CMPF_PREDICATES.contains(&name)) {
            return Ok(());
        }
        Err(format!(
            "unsupported predicate '{predicate}'; supported predicates are {}",
            SUPPORTED_CMPF_PREDICATES.iter().format(", "),
        ))
    }
}

/// Check that a conversion changes width in the declared direction.
fn verify_width(
    ctx: &IrContext,
    op: OpRef,
    width: fn(&IrContext, TypeRef) -> Option<u32>,
    widens: bool,
) -> Result<(), String> {
    let (Some(&operand), Some(&result)) = (ctx.op_operands(op).first(), ctx.op_results(op).first())
    else {
        return Err("conversion requires one operand and one result".into());
    };
    let (Some(from), Some(to)) = (
        width(ctx, ctx.value_ty(operand)),
        width(ctx, ctx.value_ty(result)),
    ) else {
        return Err("conversion operand and result must have a width".into());
    };
    if (widens && to > from) || (!widens && to < from) {
        Ok(())
    } else {
        let direction = if widens { "wider" } else { "narrower" };
        Err(format!(
            "result width {to} must be {direction} than operand width {from}"
        ))
    }
}

macro_rules! width_verifier {
    ($op:ident, $width:path, $widens:literal) => {
        impl crate::ops::Verify for $op {
            fn verify(self, ctx: &IrContext) -> Result<(), String> {
                verify_width(ctx, self.op_ref(), $width, $widens)
            }
        }
    };
}

width_verifier!(Extsi, IntegerLike::width, true);
width_verifier!(Extui, IntegerLike::width, true);
width_verifier!(Trunci, IntegerLike::width, false);
width_verifier!(Extf, FloatLike::width, true);
width_verifier!(Truncf, FloatLike::width, false);

// Folds this dialect contributes to `transforms::canonicalize`. Each
// `#[trunk_ir::canonicalize_fold(...)]` attribute below registers the
// function via `inventory`; the pass discovers them at startup.

/// `arith.addi` folds:
/// - `x + 0` / `0 + x` → `x`
/// - `const(a) + const(b)` → `const(wrap(a+b))` at the result width
#[trunk_ir::canonicalize_fold(Addi)]
pub(crate) fn fold_addi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) {
        return Some(FoldResult::Forward(lhs));
    }
    if const_int_value(ctx, lhs) == Some(0) {
        return Some(FoldResult::Forward(rhs));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_add(b), width),
    )))
}

/// `arith.subi` folds:
/// - `x - 0` → `x`. (`0 - x` is the `negi` semantic and is left for a
///   separate fold so the rewrite direction stays unambiguous.)
/// - `const(a) - const(b)` → `const(wrap(a-b))` at the result width.
#[trunk_ir::canonicalize_fold(Subi)]
pub(crate) fn fold_subi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) {
        return Some(FoldResult::Forward(lhs));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_sub(b), width),
    )))
}

/// `arith.muli` folds:
/// - `x * 0` / `0 * x` → `const 0` (checked before x*1 to short-circuit).
/// - `x * 1` / `1 * x` → `x`.
/// - `const(a) * const(b)` → `const(wrap(a*b))` at the result width.
#[trunk_ir::canonicalize_fold(Muli)]
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
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_mul(b), width),
    )))
}

/// `arith.divsi const(a), const(b)` → `arith.const(a/b)` at the result width.
///
/// Conservative bailouts that leave the op intact:
/// - `b == 0`: backend trap (Cranelift `sdiv`, WASM `i32.div_s`) defines
///   runtime behavior; the fold doesn't synthesize a result.
/// - `a == INT_MIN && b == -1` at the result width: backend traps on
///   signed-overflow. Tribute's integer-overflow policy isn't pinned
///   down (`new-plans/types.md` doesn't specify wrap vs trap), so the
///   conservative choice is to leave the op alone — that way IR
///   semantics match whatever the backend does.
#[trunk_ir::canonicalize_fold(Divsi)]
pub(crate) fn fold_divsi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    if b == 0 {
        return None;
    }
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    if is_signed_overflow_at_width(a, b, width) {
        return None;
    }
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_div(b), width),
    )))
}

/// `arith.divui const(a), const(b)` → `arith.const(a/b)` at the result
/// width, interpreting both operands as unsigned `N`-bit values.
/// Bails when the unsigned divisor is zero.
#[trunk_ir::canonicalize_fold(Divui)]
pub(crate) fn fold_divui(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    let mask = width_mask_u128(width);
    let (a_u, b_u) = ((a as u128) & mask, (b as u128) & mask);
    if b_u == 0 {
        return None;
    }
    let result = a_u.wrapping_div(b_u);
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(result as i128, width),
    )))
}

/// `arith.remsi const(a), const(b)` → `arith.const(a%b)` at the result width.
///
/// Mirrors [`fold_divsi`]: bails on `b == 0` and on `INT_MIN % -1` (which
/// also traps on Cranelift `srem` and WASM `i32.rem_s`).
#[trunk_ir::canonicalize_fold(Remsi)]
pub(crate) fn fold_remsi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    if b == 0 {
        return None;
    }
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    if is_signed_overflow_at_width(a, b, width) {
        return None;
    }
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a.wrapping_rem(b), width),
    )))
}

/// `arith.remui const(a), const(b)` → `arith.const(a%b)` at the result
/// width, interpreting both operands as unsigned `N`-bit values.
/// Bails when the unsigned divisor is zero.
#[trunk_ir::canonicalize_fold(Remui)]
pub(crate) fn fold_remui(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    let mask = width_mask_u128(width);
    let (a_u, b_u) = ((a as u128) & mask, (b as u128) & mask);
    if b_u == 0 {
        return None;
    }
    let result = a_u.wrapping_rem(b_u);
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(result as i128, width),
    )))
}

/// `arith.and` folds:
/// - `x & 0` / `0 & x` → `const 0` (short-circuit before width-extraction
///   so the fold still applies even if the result type is non-standard).
/// - `x & -1` / `-1 & x` → `x`. `-1` is the all-ones bit pattern in any
///   width, since `Attribute::Int` values are stored sign-extended in
///   `i128`.
/// - `const(a) & const(b)` → `const(wrap(a&b))` at the result width.
#[trunk_ir::canonicalize_fold(And)]
pub(crate) fn fold_and(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) || const_int_value(ctx, lhs) == Some(0) {
        return Some(FoldResult::ArithConst(Attribute::Int(0)));
    }
    if const_int_value(ctx, rhs) == Some(-1) {
        return Some(FoldResult::Forward(lhs));
    }
    if const_int_value(ctx, lhs) == Some(-1) {
        return Some(FoldResult::Forward(rhs));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a & b, width),
    )))
}

/// `arith.or` folds:
/// - `x | 0` / `0 | x` → `x`.
/// - `x | -1` / `-1 | x` → `const -1` (all-ones; same value in any
///   width when interpreted as a sign-extended `i128`).
/// - `const(a) | const(b)` → `const(wrap(a|b))` at the result width.
#[trunk_ir::canonicalize_fold(Or)]
pub(crate) fn fold_or(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) {
        return Some(FoldResult::Forward(lhs));
    }
    if const_int_value(ctx, lhs) == Some(0) {
        return Some(FoldResult::Forward(rhs));
    }
    if const_int_value(ctx, rhs) == Some(-1) || const_int_value(ctx, lhs) == Some(-1) {
        return Some(FoldResult::ArithConst(Attribute::Int(-1)));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a | b, width),
    )))
}

/// `arith.xor` folds:
/// - `x ^ 0` / `0 ^ x` → `x`.
/// - `const(a) ^ const(b)` → `const(wrap(a^b))` at the result width.
///
/// `x ^ x → 0` would require same-operand detection and is left for a
/// later pass that handles same-operand peepholes uniformly across
/// the dialect (see `subi`, which doesn't fold `x - x → 0` either).
#[trunk_ir::canonicalize_fold(Xor)]
pub(crate) fn fold_xor(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let (lhs, rhs) = two_operands(ctx, op)?;
    if const_int_value(ctx, rhs) == Some(0) {
        return Some(FoldResult::Forward(lhs));
    }
    if const_int_value(ctx, lhs) == Some(0) {
        return Some(FoldResult::Forward(rhs));
    }
    let (a, b) = (const_int_value(ctx, lhs)?, const_int_value(ctx, rhs)?);
    let width = IntegerLike::width(ctx, single_result_type(ctx, op)?)?;
    Some(FoldResult::ArithConst(Attribute::Int(
        wrap_signed_to_width(a ^ b, width),
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

/// Bottom `N` bits all set, the rest zero — used to truncate an i128 to
/// its unsigned `N`-bit interpretation in `divui`/`remui` folds.
fn width_mask_u128(width: u32) -> u128 {
    debug_assert!((1..=128).contains(&width));
    if width == 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// `true` iff `(a, b)` is the signed-overflow corner: `b == -1` and `a`
/// equals the smallest representable signed value at `width` bits. Both
/// signed division (`a / b`) and remainder (`a % b`) trap on this case
/// in Cranelift `sdiv`/`srem` and WASM `i32.div_s`/`i32.rem_s`, so
/// `divsi`/`remsi` folds bail out to leave the runtime behavior to the
/// backend.
fn is_signed_overflow_at_width(a: i128, b: i128, width: u32) -> bool {
    if b != -1 {
        return false;
    }
    let min_at_width = if width == 128 {
        i128::MIN
    } else {
        -(1i128 << (width - 1))
    };
    a == min_at_width
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::Symbol;
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

    // ---------------------------------------------------------------------
    // div/rem folds
    // ---------------------------------------------------------------------

    #[test]
    fn int_const_fold_divsi() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 10} : core.i32
    %b = arith.const {value = 4} : core.i32
    %r = arith.divsi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "divsi"), 0);
        // Truncation toward zero: 10 / 4 == 2.
        assert_eq!(return_value_int_const(&ctx, module), Some(2));
    }

    #[test]
    fn int_const_fold_divsi_negative_lhs() {
        // Rust signed integer division truncates toward zero: -10 / 4 == -2.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = -10} : core.i32
    %b = arith.const {value = 4} : core.i32
    %r = arith.divsi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(return_value_int_const(&ctx, module), Some(-2));
    }

    #[test]
    fn int_const_fold_divsi_div_by_zero_left_alone() {
        // `1 / 0` must NOT fold — backend trap defines runtime behavior.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 1} : core.i32
    %b = arith.const {value = 0} : core.i32
    %r = arith.divsi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "divsi"), 1);
    }

    #[test]
    fn int_const_fold_divsi_int_min_div_neg_one_left_alone() {
        // i32::MIN / -1 traps on Cranelift `sdiv` and WASM `i32.div_s`;
        // the fold must leave the op alone so IR semantics match.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = -2147483648} : core.i32
    %b = arith.const {value = -1} : core.i32
    %r = arith.divsi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "divsi"), 1);
    }

    #[test]
    fn int_const_fold_divui_treats_operands_as_unsigned() {
        // i32 stored value -1 == u32 0xFFFFFFFF == 4294967295 unsigned.
        // 4294967295 / 2 == 2147483647 (== i32::MAX, top bit clear).
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = -1} : core.i32
    %b = arith.const {value = 2} : core.i32
    %r = arith.divui %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(return_value_int_const(&ctx, module), Some(i32::MAX as i128));
    }

    #[test]
    fn int_const_fold_divui_div_by_zero_left_alone() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 5} : core.i32
    %b = arith.const {value = 0} : core.i32
    %r = arith.divui %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "divui"), 1);
    }

    #[test]
    fn int_const_fold_remsi_negative_lhs() {
        // Rust signed remainder truncates toward zero: -7 % 2 == -1.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = -7} : core.i32
    %b = arith.const {value = 2} : core.i32
    %r = arith.remsi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(return_value_int_const(&ctx, module), Some(-1));
    }

    #[test]
    fn int_const_fold_remsi_rem_by_zero_left_alone() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 5} : core.i32
    %b = arith.const {value = 0} : core.i32
    %r = arith.remsi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "remsi"), 1);
    }

    #[test]
    fn int_const_fold_remsi_int_min_rem_neg_one_left_alone() {
        // INT_MIN % -1 also traps on Cranelift `srem` / WASM `i32.rem_s`.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = -2147483648} : core.i32
    %b = arith.const {value = -1} : core.i32
    %r = arith.remsi %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "remsi"), 1);
    }

    #[test]
    fn int_const_fold_remui_treats_operands_as_unsigned() {
        // i32 stored value -1 == u32 4294967295. 4294967295 % 3 == 0.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = -1} : core.i32
    %b = arith.const {value = 3} : core.i32
    %r = arith.remui %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(return_value_int_const(&ctx, module), Some(0));
    }

    #[test]
    fn int_const_fold_remui_rem_by_zero_left_alone() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 5} : core.i32
    %b = arith.const {value = 0} : core.i32
    %r = arith.remui %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "remui"), 1);
    }

    #[test]
    fn is_signed_overflow_at_width_corners() {
        // i32::MIN / -1 — overflow.
        assert!(is_signed_overflow_at_width(i32::MIN as i128, -1, 32));
        // i64::MIN / -1 — overflow.
        assert!(is_signed_overflow_at_width(i64::MIN as i128, -1, 64));
        // i128::MIN / -1 — overflow at width 128.
        assert!(is_signed_overflow_at_width(i128::MIN, -1, 128));
        // Just above MIN: not overflow.
        assert!(!is_signed_overflow_at_width(i32::MIN as i128 + 1, -1, 32));
        // Divisor not -1: never overflow.
        assert!(!is_signed_overflow_at_width(i32::MIN as i128, 1, 32));
        assert!(!is_signed_overflow_at_width(0, -1, 32));
    }

    // ---------- arith.and ----------

    #[test]
    fn and_zero_fold_replaces_with_const_zero() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.and %x, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "and"), 0);
        assert_eq!(return_value_int_const(&ctx, module), Some(0));
    }

    #[test]
    fn and_all_ones_fold_forwards_x() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %m = arith.const {value = -1} : core.i32
    %r = arith.and %x, %m : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "and"), 0);
    }

    #[test]
    fn int_const_fold_and() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 12} : core.i32
    %b = arith.const {value = 10} : core.i32
    %r = arith.and %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        // 0b1100 & 0b1010 == 0b1000 == 8.
        assert_eq!(return_value_int_const(&ctx, module), Some(8));
    }

    // ---------- arith.or ----------

    #[test]
    fn or_zero_fold_forwards_x() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.or %x, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "or"), 0);
    }

    #[test]
    fn or_all_ones_fold_replaces_with_const_neg_one() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %m = arith.const {value = -1} : core.i32
    %r = arith.or %x, %m : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "or"), 0);
        assert_eq!(return_value_int_const(&ctx, module), Some(-1));
    }

    #[test]
    fn int_const_fold_or() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 12} : core.i32
    %b = arith.const {value = 10} : core.i32
    %r = arith.or %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        // 0b1100 | 0b1010 == 0b1110 == 14.
        assert_eq!(return_value_int_const(&ctx, module), Some(14));
    }

    // ---------- arith.xor ----------

    #[test]
    fn xor_zero_fold_forwards_x() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %z = arith.const {value = 0} : core.i32
    %r = arith.xor %x, %z : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "xor"), 0);
    }

    #[test]
    fn int_const_fold_xor() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = 12} : core.i32
    %b = arith.const {value = 10} : core.i32
    %r = arith.xor %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        // 0b1100 ^ 0b1010 == 0b0110 == 6.
        assert_eq!(return_value_int_const(&ctx, module), Some(6));
    }

    #[test]
    fn int_const_fold_xor_negative_signed() {
        // -1 ^ 0xFF = -256 at i32 width: -1 is 0xFFFFFFFF (all ones),
        // 0xFF is 255, XOR is 0xFFFFFF00 which is -256 as i32.
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %a = arith.const {value = -1} : core.i32
    %b = arith.const {value = 255} : core.i32
    %r = arith.xor %a, %b : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(return_value_int_const(&ctx, module), Some(-256));
    }

    #[test]
    fn bitwise_fold_does_not_match_when_operands_are_not_const() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32, %y: core.i32) -> core.i32 {
    %a = arith.and %x, %y : core.i32
    %b = arith.or %a, %y : core.i32
    %r = arith.xor %b, %x : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_arith_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "and"), 1);
        assert_eq!(count_ops(&ctx, module, "arith", "or"), 1);
        assert_eq!(count_ops(&ctx, module, "arith", "xor"), 1);
    }
}

#[cfg(test)]
mod schema_tests {
    use crate::parser::parse_test_module;

    #[test]
    fn conversions_check_categories_and_width_direction() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%n: core.i32, %w: core.i64, %x: core.f32, %y: core.f64, %p: core.ptr) {
    %ok_extsi = arith.extsi %n : core.i64
    %ok_extui = arith.extui %n : core.i64
    %ok_trunci = arith.trunci %w : core.i32
    %ok_sitofp = arith.sitofp %n : core.f64
    %ok_fptoui = arith.fptoui %y : core.i64
    %ok_extf = arith.extf %x : core.f64
    %ok_truncf = arith.truncf %y : core.f32
    %bad_extsi = arith.extsi %w : core.i32
    %bad_trunci = arith.trunci %n : core.i64
    %bad_extf = arith.extf %y : core.f32
    %bad_ptr = arith.extui %p : core.i64
    %bad_float = arith.sitofp %x : core.f64
    func.return
  }
}"#,
        );
        let result = crate::validation::validate_operation_verifiers(&ctx, module);
        let messages = result.to_string();
        assert_eq!(result.errors.len(), 5, "{messages}");
        assert!(messages.contains("must be wider"), "{messages}");
        assert!(messages.contains("must be narrower"), "{messages}");
    }

    #[test]
    fn constants_are_numeric_and_match_their_value_kind() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() {
    %ok_int = arith.const {value = 7} : core.i32
    %ok_bool = arith.const {value = true} : core.i1
    %ok_float = arith.const {value = 1.5} : core.f64
    %nil = core.nil_value : core.nil
    %bad_nil = arith.const {value = unit} : core.nil
    %bad_ptr = arith.const {value = 0} : core.ptr
    %bad_bool = arith.const {value = true} : core.i32
    %bad_float = arith.const {value = 1.5} : core.i32
    %bad_int = arith.const {value = 1} : core.f32
    func.return
  }
}"#,
        );
        let result = crate::validation::validate_operation_verifiers(&ctx, module);
        let messages = result.to_string();
        assert_eq!(result.errors.len(), 5, "{messages}");
        assert!(messages.contains("expected NumericLike"), "{messages}");
        assert!(messages.contains("does not fit result type"), "{messages}");
    }

    #[test]
    fn arithmetic_operations_require_one_operand_and_result_type() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%a: core.i32, %b: core.i64, %x: core.f64) {
    %sub = arith.subi %a, %b : core.i32
    %shl = arith.shl %a, %b : core.i32
    %mul = arith.mulf %x, %a : core.f64
    %neg = arith.negi %x : core.f64
    %and = arith.and %a, %a : core.i64
    %ok = arith.divsi %a, %a : core.i32
    func.return
  }
}"#,
        );
        let result = crate::validation::validate_operation_verifiers(&ctx, module);
        let messages = result.to_string();
        for op in [
            "arith.subi",
            "arith.shl",
            "arith.mulf",
            "arith.negi",
            "arith.and",
        ] {
            assert!(messages.contains(op), "missing {op}: {messages}");
        }
        assert!(!messages.contains("arith.divsi"), "{messages}");
        // `negi` reports both its operand and its result.
        assert_eq!(result.errors.len(), 6, "{messages}");
    }
}
