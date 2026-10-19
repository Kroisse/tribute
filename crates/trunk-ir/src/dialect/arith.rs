//! Arena-based arith dialect.
//!
//! Operations are split by type category following MLIR/Cranelift conventions:
//! - Integer arithmetic: `addi`, `subi`, `muli`, `divsi`/`divui`, `remsi`/`remui`, `negi`
//! - Float arithmetic: `addf`, `subf`, `mulf`, `divf`, `negf`
//! - Integer comparison: `cmpi` with predicate attribute (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
//! - Float comparison: `cmpf` with predicate attribute (oeq, une, olt, ole, ogt, oge)

#[trunk_ir::dialect]
mod arith {
    #[verify]
    #[flags(Pure)]
    fn r#const(value: Attr<_>) -> Value<impl NumericLike> {}

    // Integer arithmetic
    #[implements(Fold)]
    #[flags(Pure)]
    fn addi<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn subi<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn muli<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn divsi<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn divui<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn remsi<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn remui<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn negi<T: IntegerOrIndex>(operand: Value<T>) -> Value<T> {}

    // Float arithmetic
    #[flags(Pure)]
    fn addf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn subf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn mulf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn divf<T: FloatLike>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn negf<T: FloatLike>(operand: Value<T>) -> Value<T> {}

    // Comparisons
    #[flags(Pure)]
    fn cmpi<T: IntegerOrIndex>(predicate: Attr<String>, lhs: Value<T>, rhs: Value<T>) -> Value<I1> {
    }

    #[verify]
    #[flags(Pure)]
    fn cmpf<T: FloatLike>(predicate: Attr<String>, lhs: Value<T>, rhs: Value<T>) -> Value<I1> {}

    // Bitwise (integer-only)
    #[implements(Fold)]
    #[flags(Pure)]
    fn and<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn or<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[implements(Fold)]
    #[flags(Pure)]
    fn xor<T: IntegerOrIndex>(lhs: Value<T>, rhs: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn shl<T: IntegerOrIndex>(value: Value<T>, amount: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn shr<T: IntegerOrIndex>(value: Value<T>, amount: Value<T>) -> Value<T> {}
    #[flags(Pure)]
    fn shru<T: IntegerOrIndex>(value: Value<T>, amount: Value<T>) -> Value<T> {}

    // Conversions. Integers are signless; each sign-dependent conversion is
    // an explicit signed/unsigned pair.
    #[verify]
    #[flags(Pure)]
    fn extsi(operand: Value<impl IntegerLike>) -> Value<impl IntegerLike> {}
    #[verify]
    #[flags(Pure)]
    fn extui(operand: Value<impl IntegerLike>) -> Value<impl IntegerLike> {}
    #[verify]
    #[flags(Pure)]
    fn trunci(operand: Value<impl IntegerLike>) -> Value<impl IntegerLike> {}
    #[flags(Pure)]
    fn sitofp(operand: Value<impl IntegerLike>) -> Value<impl FloatLike> {}
    #[flags(Pure)]
    fn uitofp(operand: Value<impl IntegerLike>) -> Value<impl FloatLike> {}
    /// Traps on out-of-range values and NaN.
    fn fptosi(operand: Value<impl FloatLike>) -> Value<impl IntegerLike> {}
    /// Traps on out-of-range values and NaN.
    fn fptoui(operand: Value<impl FloatLike>) -> Value<impl IntegerLike> {}
    /// Convert between a fixed-width integer and `core.index`, in either
    /// direction, extending the sign when the result is wider.
    #[verify]
    #[implements(Fold)]
    #[flags(Pure)]
    fn index_castsi(operand: Value<impl IntegerOrIndex>) -> Value<impl IntegerOrIndex> {}
    /// Convert between a fixed-width integer and `core.index`, in either
    /// direction, extending with zeros when the result is wider.
    #[verify]
    #[implements(Fold)]
    #[flags(Pure)]
    fn index_castui(operand: Value<impl IntegerOrIndex>) -> Value<impl IntegerOrIndex> {}
    #[verify]
    #[flags(Pure)]
    fn extf(operand: Value<impl FloatLike>) -> Value<impl FloatLike> {}
    #[verify]
    #[flags(Pure)]
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
use crate::dialect::core::{FloatLike, I1, Index, IntegerLike, IntegerOrIndex, NumericLike};
use crate::op_flag::Pure;
use crate::ops::DialectOp;
use crate::refs::{OpRef, TypeRef, ValueDef, ValueRef};
use crate::transforms::canonicalize::{Fold, FoldResult};
use crate::types::Attribute;
use itertools::Itertools;

/// `arith.cmpf` predicates that every backend lowers.
const SUPPORTED_CMPF_PREDICATES: [&str; 6] = ["oeq", "une", "olt", "ole", "ogt", "oge"];

impl crate::ops::Verify for Const {
    fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
        let ty = ctx.value_ty(self.result(ctx));
        let accepted = match self.value(ctx) {
            Attribute::Int(_) => IntegerOrIndex::matches(ctx, ty),
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
            )
            .into())
        }
    }
}

impl crate::ops::Verify for Cmpf {
    fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
        let predicate = self.predicate(ctx);
        if SUPPORTED_CMPF_PREDICATES.contains(&predicate) {
            return Ok(());
        }
        Err(format!(
            "unsupported predicate '{predicate}'; supported predicates are {}",
            SUPPORTED_CMPF_PREDICATES.iter().format(", "),
        )
        .into())
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
            fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
                Ok(verify_width(ctx, self.op_ref(), $width, $widens)?)
            }
        }
    };
}

width_verifier!(Extsi, IntegerLike::width, true);
width_verifier!(Extui, IntegerLike::width, true);
width_verifier!(Trunci, IntegerLike::width, false);
width_verifier!(Extf, FloatLike::width, true);
width_verifier!(Truncf, FloatLike::width, false);

/// Check that an index cast converts between `core.index` and a fixed-width
/// integer.
fn verify_index_cast(ctx: &IrContext, op: OpRef) -> Result<(), String> {
    let (Some(&operand), Some(&result)) = (ctx.op_operands(op).first(), ctx.op_results(op).first())
    else {
        return Err("index cast requires one operand and one result".into());
    };
    let (from, to) = (ctx.value_ty(operand), ctx.value_ty(result));
    let fixed_to_index = IntegerLike::matches(ctx, from) && Index::matches(ctx, to);
    let index_to_fixed = Index::matches(ctx, from) && IntegerLike::matches(ctx, to);
    if fixed_to_index || index_to_fixed {
        Ok(())
    } else {
        Err(format!(
            "index cast converts between core.index and a fixed-width integer, not {} to {}",
            crate::printer::print_type(ctx, from),
            crate::printer::print_type(ctx, to)
        ))
    }
}

impl crate::ops::Verify for IndexCastsi {
    fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
        Ok(verify_index_cast(ctx, self.op_ref())?)
    }
}

impl crate::ops::Verify for IndexCastui {
    fn verify(&self, ctx: &IrContext) -> Result<(), crate::ops::VerificationError> {
        Ok(verify_index_cast(ctx, self.op_ref())?)
    }
}

// Folds this dialect contributes to `transforms::canonicalize`. Each
// operation below declares `#[implements(Fold)]` in the dialect.

/// `arith.addi` folds:
/// - `x + 0` / `0 + x` → `x`
/// - `const(a) + const(b)` → `const(wrap(a+b))` at the result width
impl Fold for Addi {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.subi` folds:
/// - `x - 0` → `x`. (`0 - x` is the `negi` semantic and is left for a
///   separate fold so the rewrite direction stays unambiguous.)
/// - `const(a) - const(b)` → `const(wrap(a-b))` at the result width.
impl Fold for Subi {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.muli` folds:
/// - `x * 0` / `0 * x` → `const 0` (checked before x*1 to short-circuit).
/// - `x * 1` / `1 * x` → `x`.
/// - `const(a) * const(b)` → `const(wrap(a*b))` at the result width.
impl Fold for Muli {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
impl Fold for Divsi {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.divui const(a), const(b)` → `arith.const(a/b)` at the result
/// width, interpreting both operands as unsigned `N`-bit values.
/// Bails when the unsigned divisor is zero.
impl Fold for Divui {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.remsi const(a), const(b)` → `arith.const(a%b)` at the result width.
///
/// Mirrors the `arith.divsi` fold: bails on `b == 0` and on `INT_MIN % -1` (which
/// also traps on Cranelift `srem` and WASM `i32.rem_s`).
impl Fold for Remsi {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.remui const(a), const(b)` → `arith.const(a%b)` at the result
/// width, interpreting both operands as unsigned `N`-bit values.
/// Bails when the unsigned divisor is zero.
impl Fold for Remui {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.and` folds:
/// - `x & 0` / `0 & x` → `const 0` (short-circuit before width-extraction
///   so the fold still applies even if the result type is non-standard).
/// - `x & -1` / `-1 & x` → `x`. `-1` is the all-ones bit pattern in any
///   width, since `Attribute::Int` values are stored sign-extended in
///   `i128`.
/// - `const(a) & const(b)` → `const(wrap(a&b))` at the result width.
impl Fold for And {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.or` folds:
/// - `x | 0` / `0 | x` → `x`.
/// - `x | -1` / `-1 | x` → `const -1` (all-ones; same value in any
///   width when interpreted as a sign-extended `i128`).
/// - `const(a) | const(b)` → `const(wrap(a|b))` at the result width.
impl Fold for Or {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

/// `arith.xor` folds:
/// - `x ^ 0` / `0 ^ x` → `x`.
/// - `const(a) ^ const(b)` → `const(wrap(a^b))` at the result width.
///
/// `x ^ x → 0` would require same-operand detection and is left for a
/// later pass that handles same-operand peepholes uniformly across
/// the dialect (see `subi`, which doesn't fold `x - x → 0` either).
impl Fold for Xor {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        let op = self.op_ref();
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
}

impl Fold for IndexCastsi {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        fold_index_cast(ctx, self.op_ref(), true)
    }
}

impl Fold for IndexCastui {
    fn fold(&self, ctx: &IrContext) -> Option<FoldResult> {
        fold_index_cast(ctx, self.op_ref(), false)
    }
}

/// Folds of `arith.index_castsi` (`signed`) and `arith.index_castui`
/// that hold at every index width of at least [`Index::MIN_WIDTH`] bits:
///
/// - A constant fixed-width operand becomes the index constant of its
///   extended value, which every index width truncates or extends alike.
/// - A constant index operand becomes an `iN` constant when `N` is at most
///   the minimum index width, or when the value already fits that width.
/// - `iN → index → iN` forwards the original value when `N` is at most the
///   minimum index width, so the index held it without truncation.
fn fold_index_cast(ctx: &IrContext, op: OpRef, signed: bool) -> Option<FoldResult> {
    let &operand = ctx.op_operands(op).first()?;
    let result_ty = single_result_type(ctx, op)?;
    if let Some(value) = const_int_value(ctx, operand) {
        return index_cast_const(ctx, value, ctx.value_ty(operand), result_ty, signed)
            .map(|value| FoldResult::ArithConst(Attribute::Int(value)));
    }
    let width = IntegerLike::width(ctx, result_ty)?;
    if let ValueDef::OpResult(producer, _) = ctx.value_def(operand)
        && (IndexCastsi::matches(ctx, producer) || IndexCastui::matches(ctx, producer))
        && let Some(&original) = ctx.op_operands(producer).first()
        && ctx.value_ty(original) == result_ty
        && width <= Index::MIN_WIDTH
    {
        return Some(FoldResult::Forward(original));
    }
    None
}

/// The constant `value` of type `from` cast to `to`, if it does not depend
/// on the index width.
fn index_cast_const(
    ctx: &IrContext,
    value: i128,
    from: TypeRef,
    to: TypeRef,
    signed: bool,
) -> Option<i128> {
    if let Some(width) = IntegerLike::width(ctx, from) {
        // `Int` values are stored sign-extended to `i128`.
        return Some(if signed {
            value
        } else {
            ((value as u128) & width_mask_u128(width)) as i128
        });
    }
    let width = IntegerLike::width(ctx, to)?;
    let min = Index::MIN_WIDTH;
    let fits = if signed {
        (-(1i128 << (min - 1))..(1i128 << (min - 1))).contains(&value)
    } else {
        (0..(1i128 << min)).contains(&value)
    };
    (width <= min || fits).then(|| wrap_signed_to_width(value, width))
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

    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};
    use crate::transforms::canonicalize::{FoldDispatchPattern, fold_ops_for_dialect};
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    /// Run only this dialect's folds on `module` via a single
    /// [`FoldDispatchPattern`]. Filters the folds by dialect so
    /// per-fold tests stay isolated from other dialects' folds even
    /// though the production `canonicalize` pass aggregates everyone.
    fn run_arith_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let dispatcher = FoldDispatchPattern::from_ops(fold_ops_for_dialect("arith"));
        PatternApplicator::new(TypeConverter::new())
            .add_pattern_box(Box::new(dispatcher))
            .apply_partial(ctx, module)
    }

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

    /// Walk `module` and return the `Attribute::Int` carried by the
    /// `arith.const` that the first `func.return` returns. Returns
    /// `None` if no return is found, or if its operand is not produced
    /// by a const-int op. Lets value-shaped tests assert the folded
    /// constant directly without going through textual snapshots.
    fn return_value_int_const(ctx: &IrContext, module: Module) -> Option<i128> {
        let mut found = None;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if found.is_some() {
                return ControlFlow::Continue(WalkAction::Advance);
            }
            if crate::dialect::func::Return::matches(ctx, op)
                && let Some(&v) = ctx.op_operands(op).first()
            {
                found = const_int_value(ctx, v);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found
    }

    /// Fold `body`, a function returning `ret_ty`, and return the folded
    /// module and the constant it returns, if any.
    fn fold_function(params: &str, ret_ty: &str, body: &str) -> (IrContext, Module, Option<i128>) {
        let input = format!(
            "core.module @test {{\n  func.func @f({params}) -> {ret_ty} {{\n{body}\n  }}\n}}"
        );
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, &input);
        run_arith_patterns(&mut ctx, module);
        let value = return_value_int_const(&ctx, module);
        (ctx, module, value)
    }

    #[test]
    fn index_cast_of_a_fixed_width_constant_folds_to_its_extended_value() {
        for (op, expected) in [("index_castsi", -1), ("index_castui", 0xffff_ffff)] {
            let (_, _, value) = fold_function(
                "",
                "core.index",
                &format!(
                    "    %c = arith.const {{value = -1}} : core.i32\n    %r = arith.{op} %c : core.index\n    func.return %r"
                ),
            );
            assert_eq!(value, Some(expected), "{op}");
        }
    }

    #[test]
    fn index_cast_of_an_index_constant_folds_only_when_the_index_width_cannot_matter() {
        let fold = |op: &str, value: i128, ty: &str| {
            fold_function(
                "",
                ty,
                &format!(
                    "    %c = arith.const {{value = {value}}} : core.index\n    %r = arith.{op} %c : {ty}\n    func.return %r"
                ),
            )
            .2
        };
        assert_eq!(fold("index_castsi", -5, "core.i64"), Some(-5));
        assert_eq!(
            fold("index_castui", 0xffff_ffff, "core.i64"),
            Some(0xffff_ffff)
        );
        // Truncation to at most the minimum index width ignores the higher bits.
        assert_eq!(fold("index_castsi", 1 << 40, "core.i32"), Some(0));
        assert_eq!(fold("index_castui", -1, "core.i8"), Some(-1));
        // A 32-bit index would have dropped these bits before extending.
        assert_eq!(fold("index_castsi", 1 << 40, "core.i64"), None);
        assert_eq!(fold("index_castui", -1, "core.i64"), None);
    }

    #[test]
    fn index_cast_round_trip_forwards_values_no_wider_than_the_minimum_index_width() {
        let round_trip = |ty: &str| {
            let (ctx, module, _) = fold_function(
                &format!("%x: {ty}"),
                ty,
                &format!(
                    "    %i = arith.index_castui %x : core.index\n    %r = arith.index_castsi %i : {ty}\n    func.return %r"
                ),
            );
            count_ops(&ctx, module, crate::dialect::arith::IndexCastsi::NAME)
        };
        assert_eq!(round_trip("core.i32"), 0);
        assert_eq!(round_trip("core.i64"), 1);
    }

    #[test]
    fn index_arithmetic_folds_identities_but_not_width_dependent_constants() {
        let (ctx, module, _) = fold_function(
            "%x: core.index",
            "core.index",
            "    %z = arith.const {value = 0} : core.index\n    %r = arith.addi %x, %z : core.index\n    func.return %r",
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
        let (ctx, module, value) = fold_function(
            "",
            "core.index",
            "    %a = arith.const {value = 2147483647} : core.index\n    %b = arith.const {value = 1} : core.index\n    %r = arith.addi %a, %b : core.index\n    func.return %r",
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
        assert_eq!(value, None);
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Muli::NAME),
            0
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Muli::NAME),
            0
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Const::NAME),
            2
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Subi::NAME),
            0
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Subi::NAME),
            1
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            0
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Subi::NAME),
            0
        );
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Muli::NAME),
            0
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Addi::NAME),
            1
        );
    }

    // ---------------------------------------------------------------------
    // div/rem folds
    // ---------------------------------------------------------------------

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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Divsi::NAME),
            1
        );
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
        assert_eq!(
            count_ops(&ctx, module, crate::dialect::arith::Divsi::NAME),
            1
        );
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
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::And::NAME), 0);
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
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::And::NAME), 0);
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
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::Or::NAME), 0);
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
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::Or::NAME), 0);
        assert_eq!(return_value_int_const(&ctx, module), Some(-1));
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
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::Xor::NAME), 0);
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
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::And::NAME), 1);
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::Or::NAME), 1);
        assert_eq!(count_ops(&ctx, module, crate::dialect::arith::Xor::NAME), 1);
    }

    // ---------------------------------------------------------------------
    // Constant-fold properties
    // ---------------------------------------------------------------------

    mod const_fold_props {
        use super::*;
        use proptest::prelude::*;

        /// Binary integer operations with a constant fold.
        const BINARY_OPS: [&str; 10] = [
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.and",
            "arith.or",
            "arith.xor",
        ];

        /// Native Rust integer widths, so the expected value comes straight
        /// from the matching `iN`/`uN` operation.
        const NATIVE_WIDTHS: [u32; 5] = [8, 16, 32, 64, 128];

        /// Fold `op const(a), const(b)` at `core.i{width}`. Returns the folded
        /// constant, or `None` when the operation was left unchanged.
        fn fold_binary(
            op: &str,
            width: u32,
            a: i128,
            b: i128,
        ) -> Result<Option<i128>, TestCaseError> {
            let input = format!(
                "core.module @test {{
  func.func @f() -> core.i{width} {{
    %a = arith.const {{value = {a}}} : core.i{width}
    %b = arith.const {{value = {b}}} : core.i{width}
    %r = {op} %a, %b : core.i{width}
    func.return %r
  }}
}}"
            );
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, &input);
            let result = run_arith_patterns(&mut ctx, module);
            let def = crate::OpDef::lookup(op).expect("registered arith op");
            let remaining = count_ops(&ctx, module, crate::OpName::registered(def));
            if result.total_changes == 0 {
                prop_assert_eq!(remaining, 1);
                return Ok(None);
            }
            prop_assert_eq!(remaining, 0);
            let folded = return_value_int_const(&ctx, module);
            prop_assert!(folded.is_some(), "{op} folded to a non-constant");
            Ok(folded)
        }

        /// The fold's contract at a native width: two's-complement wrapping
        /// for `+ - * & | ^`, and `checked_*` for division and remainder, whose
        /// `None` (zero divisor, signed `MIN / -1`) means "left unchanged".
        fn expected_native(op: &str, width: u32, a: i128, b: i128) -> Option<i128> {
            macro_rules! at {
                ($s:ty, $u:ty) => {{
                    let (a, b) = (a as $s, b as $s);
                    let (ua, ub) = (a as $u, b as $u);
                    match op {
                        "arith.addi" => Some(a.wrapping_add(b) as i128),
                        "arith.subi" => Some(a.wrapping_sub(b) as i128),
                        "arith.muli" => Some(a.wrapping_mul(b) as i128),
                        "arith.divsi" => a.checked_div(b).map(|r| r as i128),
                        "arith.remsi" => a.checked_rem(b).map(|r| r as i128),
                        "arith.divui" => ua.checked_div(ub).map(|r| r as $s as i128),
                        "arith.remui" => ua.checked_rem(ub).map(|r| r as $s as i128),
                        "arith.and" => Some((a & b) as i128),
                        "arith.or" => Some((a | b) as i128),
                        "arith.xor" => Some((a ^ b) as i128),
                        _ => unreachable!("unknown op {op}"),
                    }
                }};
            }
            match width {
                8 => at!(i8, u8),
                16 => at!(i16, u16),
                32 => at!(i32, u32),
                64 => at!(i64, u64),
                128 => at!(i128, u128),
                _ => unreachable!("not a native width: {width}"),
            }
        }

        /// Reference two's-complement wrap: keep the low `width` bits and
        /// fill the high bits with the sign bit.
        fn reference_wrap(value: i128, width: u32) -> i128 {
            if width == 128 {
                return value;
            }
            let mask = (1u128 << width) - 1;
            let low = (value as u128) & mask;
            if low >> (width - 1) & 1 == 1 {
                (low | !mask) as i128
            } else {
                low as i128
            }
        }

        /// Raw values biased toward the corners where wrapping and the
        /// division bailouts happen.
        fn raw_value() -> impl Strategy<Value = i128> {
            prop_oneof![
                3 => any::<i128>(),
                2 => -3i128..=3,
                1 => prop_oneof![
                    Just(i128::MIN),
                    Just(i128::MAX),
                    Just(i64::MIN as i128),
                    Just(i64::MAX as i128),
                    Just(i32::MIN as i128),
                    Just(i32::MAX as i128),
                    Just(i16::MIN as i128),
                    Just(i8::MIN as i128),
                    Just(i8::MAX as i128),
                ],
            ]
        }

        proptest! {
            /// Folding a constant binary operation at a native width yields
            /// exactly Rust's result for that width, and the trapping
            /// division/remainder cases are left unchanged.
            #[test]
            fn binary_fold_matches_rust_at_native_widths(
                op in prop::sample::select(&BINARY_OPS[..]),
                width in prop::sample::select(&NATIVE_WIDTHS[..]),
                a in raw_value(),
                b in raw_value(),
            ) {
                // Constants are stored sign-extended at their width.
                let (a, b) = (reference_wrap(a, width), reference_wrap(b, width));
                let folded = fold_binary(op, width, a, b)?;
                prop_assert_eq!(folded, expected_native(op, width, a, b));
            }

            /// At any width in `1..=128`, including `i1` and non-native
            /// widths, the fold agrees with the i128 operation wrapped back
            /// to the width.
            #[test]
            fn binary_fold_wraps_at_any_width(
                op in prop::sample::select(&BINARY_OPS[..]),
                width in 1u32..=128,
                a in raw_value(),
                b in raw_value(),
            ) {
                let (a, b) = (reference_wrap(a, width), reference_wrap(b, width));
                let unsigned = |v: i128| {
                    if width == 128 { v as u128 } else { (v as u128) & ((1u128 << width) - 1) }
                };
                let min = reference_wrap(1i128 << (width - 1), width);
                let signed_trap = b == 0 || (a == min && b == -1);
                let expected = match op {
                    "arith.addi" => Some(a.wrapping_add(b)),
                    "arith.subi" => Some(a.wrapping_sub(b)),
                    "arith.muli" => Some(a.wrapping_mul(b)),
                    "arith.divsi" => (!signed_trap).then(|| a.wrapping_div(b)),
                    "arith.remsi" => (!signed_trap).then(|| a.wrapping_rem(b)),
                    "arith.divui" => (unsigned(b) != 0).then(|| (unsigned(a) / unsigned(b)) as i128),
                    "arith.remui" => (unsigned(b) != 0).then(|| (unsigned(a) % unsigned(b)) as i128),
                    "arith.and" => Some(a & b),
                    "arith.or" => Some(a | b),
                    "arith.xor" => Some(a ^ b),
                    _ => unreachable!("unknown op {op}"),
                }
                .map(|r| reference_wrap(r, width));
                prop_assert_eq!(fold_binary(op, width, a, b)?, expected);
            }

            /// `MIN / -1` and `MIN % -1` are never folded at any width, while
            /// the neighbouring `(MIN + 1) / -1` is.
            #[test]
            fn signed_div_rem_overflow_corner_is_left_alone(
                op in prop::sample::select(&["arith.divsi", "arith.remsi"][..]),
                width in 1u32..=128,
            ) {
                let min = reference_wrap(1i128 << (width - 1), width);
                prop_assert_eq!(fold_binary(op, width, min, -1)?, None);
                prop_assert!(is_signed_overflow_at_width(min, -1, width));
                prop_assert!(fold_binary(op, width, min + 1, -1)?.is_some());
                prop_assert!(!is_signed_overflow_at_width(min + 1, -1, width));
            }

            /// `wrap_signed_to_width` agrees with the mask-based reference wrap.
            #[test]
            fn wrap_signed_to_width_matches_reference(
                value in raw_value(),
                width in 1u32..=128,
            ) {
                prop_assert_eq!(wrap_signed_to_width(value, width), reference_wrap(value, width));
            }
        }
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

    #[test]
    fn index_takes_integer_arithmetic_and_converts_only_through_index_casts() {
        let mut ctx = crate::IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%i: core.index, %n: core.i32, %w: core.i64, %x: core.f64) {
    %ok_const = arith.const {value = 4} : core.index
    %ok_add = arith.addi %i, %ok_const : core.index
    %ok_cmp = arith.cmpi %i, %ok_add {predicate = "ult"} : core.i1
    %ok_si = arith.index_castsi %n : core.index
    %ok_ui = arith.index_castui %i : core.i64
    %bad_mixed = arith.addi %i, %n : core.index
    %bad_extsi = arith.extsi %i : core.i64
    %bad_float = arith.sitofp %i : core.f64
    %bad_fixed = arith.index_castsi %n : core.i64
    %bad_same = arith.index_castui %i : core.index
    %bad_cast_float = arith.index_castsi %x : core.index
    func.return
  }
}"#,
        );
        let result = crate::validation::validate_operation_verifiers(&ctx, module);
        let messages = result.to_string();
        for op in [
            "arith.addi",
            "arith.extsi",
            "arith.sitofp",
            "arith.index_castsi",
            "arith.index_castui",
        ] {
            assert!(messages.contains(op), "missing {op}: {messages}");
        }
        assert!(
            messages.contains("index cast converts between core.index"),
            "{messages}"
        );
        assert!(!messages.contains("arith.cmpi"), "{messages}");
        assert!(!messages.contains("arith.const"), "{messages}");
    }
}
