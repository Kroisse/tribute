//! Lower intrinsic operator calls to arith dialect operations.
//!
//! This pass transforms `func.call @"Int::(+)"(a, b)` → `arith.addi(a, b)`, etc.
//! It runs in the shared pipeline before backend-specific lowering, handling
//! arithmetic and comparison intrinsics declared in the prelude for Int, Nat, and Float.

use std::collections::HashMap;

use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, Location};

/// Lower intrinsic arithmetic/comparison calls to arith dialect operations.
///
/// Also removes `func.func` declarations for intrinsic operators, since they
/// have no real body and would cause undefined symbol errors at link time.
pub fn lower_intrinsic_to_arith(ctx: &mut IrContext, module: Module) {
    let pattern = ArithIntrinsicPattern::new();
    let intrinsic_names: HashSet<Symbol> = pattern.map.keys().copied().collect();

    let mut applicator = PatternApplicator::new(TypeConverter::new());
    applicator = applicator
        .add_pattern(pattern)
        .add_pattern(ArithIntrinsicFuncDeclPattern { intrinsic_names });
    applicator.apply_partial(ctx, module);
}

/// What kind of arith operation to emit.
enum ArithMapping {
    /// Binary arithmetic: addi, addf, subi, etc.
    BinaryOp(fn(&mut IrContext, Location, ValueRef, ValueRef, TypeRef) -> OpRef),
    /// Integer comparison with predicate.
    CmpI(&'static str),
    /// Float comparison with predicate.
    CmpF(&'static str),
}

/// Pattern that matches `func.call` to known arithmetic intrinsics and
/// rewrites them to the corresponding `arith.*` dialect operations.
struct ArithIntrinsicPattern {
    map: HashMap<Symbol, ArithMapping>,
}

impl ArithIntrinsicPattern {
    fn new() -> Self {
        let mut map = HashMap::new();

        macro_rules! binary {
            ($name:expr, $op_fn:expr) => {
                map.insert(Symbol::from_dynamic($name), ArithMapping::BinaryOp($op_fn));
            };
        }
        macro_rules! cmpi {
            ($name:expr, $pred:expr) => {
                map.insert(Symbol::from_dynamic($name), ArithMapping::CmpI($pred));
            };
        }
        macro_rules! cmpf {
            ($name:expr, $pred:expr) => {
                map.insert(Symbol::from_dynamic($name), ArithMapping::CmpF($pred));
            };
        }

        // --- Int (signed) ---
        binary!("Int::+", |ctx, loc, l, r, ty| arith::addi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::-", |ctx, loc, l, r, ty| arith::subi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::*", |ctx, loc, l, r, ty| arith::muli(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::/", |ctx, loc, l, r, ty| arith::divsi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Int::%", |ctx, loc, l, r, ty| arith::remsi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        cmpi!("Int::==", "eq");
        cmpi!("Int::!=", "ne");
        cmpi!("Int::<", "slt");
        cmpi!("Int::<=", "sle");
        cmpi!("Int::>", "sgt");
        cmpi!("Int::>=", "sge");

        // --- Nat (unsigned) ---
        binary!("Nat::+", |ctx, loc, l, r, ty| arith::addi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::-", |ctx, loc, l, r, ty| arith::subi(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::*", |ctx, loc, l, r, ty| arith::muli(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::/", |ctx, loc, l, r, ty| arith::divui(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Nat::%", |ctx, loc, l, r, ty| arith::remui(
            ctx, loc, l, r, ty
        )
        .op_ref());
        cmpi!("Nat::==", "eq");
        cmpi!("Nat::!=", "ne");
        cmpi!("Nat::<", "ult");
        cmpi!("Nat::<=", "ule");
        cmpi!("Nat::>", "ugt");
        cmpi!("Nat::>=", "uge");

        // --- Float ---
        binary!("Float::+", |ctx, loc, l, r, ty| arith::addf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Float::-", |ctx, loc, l, r, ty| arith::subf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Float::*", |ctx, loc, l, r, ty| arith::mulf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        binary!("Float::/", |ctx, loc, l, r, ty| arith::divf(
            ctx, loc, l, r, ty
        )
        .op_ref());
        cmpf!("Float::==", "oeq");
        cmpf!("Float::!=", "une");
        cmpf!("Float::<", "olt");
        cmpf!("Float::<=", "ole");
        cmpf!("Float::>", "ogt");
        cmpf!("Float::>=", "oge");

        Self { map }
    }
}

impl RewritePattern for ArithIntrinsicPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call_op) = arena_func::Call::from_op(ctx, op) else {
            return false;
        };
        let callee = call_op.callee(ctx);

        let Some(mapping) = self.map.get(&callee) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let operands = ctx.op_operands(op).to_vec();
        let lhs = operands[0];
        let rhs = operands[1];

        match mapping {
            ArithMapping::BinaryOp(op_fn) => {
                let new_op = op_fn(ctx, loc, lhs, rhs, result_ty);
                rewriter.replace_op(new_op);
            }
            ArithMapping::CmpI(predicate) => {
                let cmp = arith::cmpi(ctx, loc, lhs, rhs, result_ty, Symbol::new(predicate));
                rewriter.replace_op(cmp.op_ref());
            }
            ArithMapping::CmpF(predicate) => {
                let cmp = arith::cmpf(ctx, loc, lhs, rhs, result_ty, Symbol::new(predicate));
                rewriter.replace_op(cmp.op_ref());
            }
        }

        true
    }
}

/// Pattern that removes `func.func` declarations for intrinsic operators.
///
/// These declarations have `abi = "intrinsic"` and `func.unreachable` body,
/// so they must not reach the backend or linker.
struct ArithIntrinsicFuncDeclPattern {
    intrinsic_names: HashSet<Symbol>,
}

impl RewritePattern for ArithIntrinsicFuncDeclPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        // Check if this function has abi = "intrinsic"
        let attrs = &ctx.op(op).attributes;
        let is_intrinsic = matches!(
            attrs.get(&Symbol::new("abi")),
            Some(Attribute::String(s)) if s == "intrinsic"
        );
        if !is_intrinsic {
            return false;
        }

        // Check if this is one of our known arithmetic intrinsics
        let sym_name = func_op.sym_name(ctx);
        if !self.intrinsic_names.contains(&sym_name) {
            return false;
        }

        // Erase the function declaration (no results to map)
        rewriter.erase_op(vec![]);
        true
    }
}
