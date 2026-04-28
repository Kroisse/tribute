//! Canonicalization pass for TrunkIR.
//!
//! Greedy fixed-point pass that folds operations into a canonical form via
//! local rewrite patterns. The pattern set so far covers integer identities,
//! constant folding for `addi`/`subi`/`muli`, and `unrealized_conversion_cast`
//! cleanup:
//!
//! - `arith.addi %x, 0` / `arith.addi 0, %x` → `%x`
//! - `arith.subi %x, 0` → `%x`
//!   (`arith.subi 0, %x` is `arith.negi`, handled separately.)
//! - `arith.muli %x, 1` / `arith.muli 1, %x` → `%x`
//! - `arith.muli %x, 0` / `arith.muli 0, %x` → `arith.const 0`
//! - `arith.addi const(a), const(b)` / `subi` / `muli` → `arith.const`
//!   (signed wrapping at the result type's bit-width — `i32::MAX + 1`
//!   becomes `i32::MIN`, matching what codegen would produce.)
//! - `core.unrealized_conversion_cast %x : T → T` → `%x`
//!   (`UnrealizedCastIdentity` — same source/target type.)
//! - `core.unrealized_conversion_cast(core.unrealized_conversion_cast(x : A → B) : B → A)`
//!   → `x` (`UnrealizedCastRoundTrip` — types collapse back to the input.)
//!
//! Float, div/rem, and scf patterns live in follow-up PRs once each one's
//! semantics are pinned down (NaN/-0.0, division-by-zero, region splice).
//!
//! Patterns are language-agnostic: this module sits in `trunk-ir`, not
//! `tribute-passes`. Each pattern self-filters by dialect/op-name inside
//! `match_and_rewrite` (the applicator does no central op-kind dispatch).
//! Per-dialect pattern registries are tracked separately in #690.

use crate::context::IrContext;
use crate::dialect::{arith, core as arena_core};
use crate::ops::DialectOp;
use crate::refs::{OpRef, TypeRef, ValueDef, ValueRef};
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
        .add_pattern(IntConstFold)
        .add_pattern(UnrealizedCastIdentity)
        .add_pattern(UnrealizedCastRoundTrip);
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
        let Some(width) = core_int_width(ctx, result_ty) else {
            return false;
        };
        let folded = wrap_signed_to_width(fold(a, b), width);
        let loc = ctx.op(op).location;
        let new_const = arith::r#const(ctx, loc, result_ty, Attribute::Int(folded));
        rewriter.replace_op(new_const.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "IntConstFold"
    }
}

/// `core.unrealized_conversion_cast %x : T → T` → `%x`.
///
/// A no-op cast left over from a type conversion that ended up matching the
/// input type (e.g. after surrounding ops were rewritten). The op carries no
/// useful work — drop it and forward the operand.
pub struct UnrealizedCastIdentity;

impl RewritePattern for UnrealizedCastIdentity {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_core::UnrealizedConversionCast::matches(ctx, op) {
            return false;
        }
        let operands = ctx.op_operands(op);
        let result_types = ctx.op_result_types(op);
        if operands.len() != 1 || result_types.len() != 1 {
            return false;
        }
        let input = operands[0];
        let result_ty = result_types[0];
        if ctx.value_ty(input) != result_ty {
            return false;
        }
        rewriter.erase_op(vec![input]);
        true
    }

    fn name(&self) -> &'static str {
        "UnrealizedCastIdentity"
    }
}

/// `cast<A → B>(cast<B → A>(%x))` → `%x`.
///
/// A pair of casts that collapse back to the input type — common when one
/// rewrite materializes `B` and a later rewrite expects `A` again. We forward
/// the inner cast's input. The inner cast is left in place; if it had no
/// other users it becomes dead and is collected by DCE.
pub struct UnrealizedCastRoundTrip;

impl RewritePattern for UnrealizedCastRoundTrip {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_core::UnrealizedConversionCast::matches(ctx, op) {
            return false;
        }
        let operands = ctx.op_operands(op);
        let result_types = ctx.op_result_types(op);
        if operands.len() != 1 || result_types.len() != 1 {
            return false;
        }
        let outer_input = operands[0];
        let outer_result_ty = result_types[0];
        let producer = match ctx.value_def(outer_input) {
            ValueDef::OpResult(p, _) => p,
            ValueDef::BlockArg(_, _) => return false,
        };
        if !arena_core::UnrealizedConversionCast::matches(ctx, producer) {
            return false;
        }
        let inner_operands = ctx.op_operands(producer);
        let Some(&inner_input) = inner_operands.first() else {
            return false;
        };
        // Round-trip iff the outer's result type matches the inner cast's
        // input type, i.e. types form `A → B → A`.
        if ctx.value_ty(inner_input) != outer_result_ty {
            return false;
        }
        rewriter.erase_op(vec![inner_input]);
        true
    }

    fn name(&self) -> &'static str {
        "UnrealizedCastRoundTrip"
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
    fn int_const_fold_wraps_at_i32_width() {
        // i32::MAX + 1 wraps to i32::MIN. Asserting the folded value
        // directly catches a non-type-aware regression that would
        // leave 2147483648 in an i32-typed const.
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

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        assert_eq!(return_value_int_const(&ctx, module), Some(i32::MIN as i128));
    }

    #[test]
    fn int_const_fold_wraps_at_i64_width() {
        // i64::MAX + 1 → i64::MIN. Same shape as the i32 case but at
        // a different width — guards against the wrap being hard-coded
        // to one width.
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

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        assert_eq!(return_value_int_const(&ctx, module), Some(i64::MIN as i128));
    }

    #[test]
    fn int_const_fold_accepts_nonstandard_width() {
        // Any `core.i{N}` with `1 <= N <= 128` is foldable — including
        // widths the backends don't necessarily support natively. The
        // pattern's job is semantic preservation at the declared width;
        // backend support is a separate concern.
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

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 0);
        // 60 + 10 = 70, which exceeds i7::MAX (63), wraps to -58.
        assert_eq!(return_value_int_const(&ctx, module), Some(-58));
    }

    #[test]
    fn int_const_fold_skips_widths_above_128() {
        // `core.i129` exceeds the i128 envelope of `wrap_signed_to_width`.
        // The fold must bail out rather than silently producing a value
        // it can't represent.
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

        let result = canonicalize(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
    }

    #[test]
    fn int_const_fold_skips_non_i_prefixed_type() {
        // `core.foo` doesn't follow the `i{N}` shape — fold must skip.
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

        let result = canonicalize(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(count_ops(&ctx, module, "arith", "addi"), 1);
    }

    #[test]
    fn wrap_signed_to_width_matches_two_complement_semantics() {
        // Direct unit checks for the helper. These pin the algorithm
        // independently of `parse_test_module` / pattern dispatch.
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
        // i1 behaves like a signed 1-bit number: 0 stays 0, 1 wraps to -1.
        assert_eq!(wrap_signed_to_width(0, 1), 0);
        assert_eq!(wrap_signed_to_width(1, 1), -1);
        assert_eq!(wrap_signed_to_width(-1, 1), -1);
        assert_eq!(wrap_signed_to_width(2, 1), 0);
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
    fn unrealized_cast_identity_drops_same_type_cast() {
        // A cast whose result type matches the operand type carries no
        // useful work and should fold away to its operand.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %r = core.unrealized_conversion_cast %x : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            0
        );
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn unrealized_cast_identity_does_not_match_when_types_differ() {
        // i32 → i64 is a real type change; the pattern must leave it for
        // resolve_unrealized_casts (or a backend's materializer).
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %r = core.unrealized_conversion_cast %x : core.i64
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            1
        );
    }

    #[test]
    fn unrealized_cast_round_trip_collapses_pair() {
        // `cast<i32→i64>(cast<i64→i32>(%x))` collapses to `%x`.
        // The outer cast is erased; the inner cast is left in place
        // (pending DCE) since canonicalize only handles local rewrites.
        let input = r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i32
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert!(result.total_changes >= 1);
        // Inner cast remains (now dead); outer is gone.
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            1
        );
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn unrealized_cast_round_trip_does_not_match_three_step_chain() {
        // `cast<i64→i32>(cast<i32→i16>(...))` is not a round-trip — the
        // outer's result type (`i64`) is not the inner's input type
        // (`i32`). Both casts must remain.
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i16
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = canonicalize(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            2
        );
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
