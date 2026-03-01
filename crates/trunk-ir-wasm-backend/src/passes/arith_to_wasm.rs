//! Lower arith dialect operations to wasm dialect (arena IR).
//!
//! This pass converts arithmetic operations to their wasm equivalents:
//! - `arith.const` -> `wasm.{i32,i64,f32,f64}_const`
//! - `arith.{add,sub,mul,div,rem}` -> `wasm.{type}_{op}`
//! - `arith.cmp_*` -> `wasm.{type}_{cmp}`
//! - `arith.neg` -> `wasm.{f32,f64}_neg` or 0 - x for integers
//! - `arith.{and,or,xor,shl,shr,shru}` -> `wasm.i{32,64}_{op}`
//! - `arith.{cast,trunc,extend,convert}` -> appropriate wasm conversion ops

use tracing::warn;

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::arith as arena_arith;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::Attribute as ArenaAttribute;
use trunk_ir::ir::Symbol;

/// Lower arith dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(ArithConstPattern)
        .add_pattern(ArithBinOpPattern)
        .add_pattern(ArithCmpPattern)
        .add_pattern(ArithNegPattern)
        .add_pattern(ArithBitwisePattern)
        .add_pattern(ArithConversionPattern);
    applicator.apply_partial(ctx, module);
}

/// Pattern for `arith.const` -> `wasm.{type}_const`
struct ArithConstPattern;

impl ArenaRewritePattern for ArithConstPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_const_op) = arena_arith::Const::from_op(ctx, op) else {
            return false;
        };

        let result_types = ctx.op_result_types(op);
        let Some(&result_ty) = result_types.first() else {
            return false;
        };

        // Handle nil type constants specially
        let type_name = type_suffix(ctx, Some(result_ty));
        if type_name == "nil" {
            let loc = ctx.op(op).location;
            let nop = arena_wasm::nop(ctx, loc, result_ty);
            rewriter.replace_op(nop.op_ref());
            return true;
        }

        let loc = ctx.op(op).location;
        let value = _const_op.value(ctx);

        let new_op_ref = match type_name {
            "i32" => {
                let v = match value {
                    ArenaAttribute::IntBits(v) => v as i32,
                    _ => 0,
                };
                arena_wasm::i32_const(ctx, loc, result_ty, v).op_ref()
            }
            "i64" => {
                let v = match value {
                    ArenaAttribute::IntBits(v) => v as i64,
                    _ => 0,
                };
                arena_wasm::i64_const(ctx, loc, result_ty, v).op_ref()
            }
            "f32" => {
                let v = match value {
                    ArenaAttribute::FloatBits(v) => f32::from_bits(v as u32),
                    _ => 0.0,
                };
                arena_wasm::f32_const(ctx, loc, result_ty, v).op_ref()
            }
            "f64" => {
                let v = match value {
                    ArenaAttribute::FloatBits(v) => f64::from_bits(v),
                    _ => 0.0,
                };
                arena_wasm::f64_const(ctx, loc, result_ty, v).op_ref()
            }
            _ => {
                let v = match value {
                    ArenaAttribute::IntBits(v) => v as i32,
                    _ => 0,
                };
                arena_wasm::i32_const(ctx, loc, result_ty, v).op_ref()
            }
        };

        rewriter.replace_op(new_op_ref);
        true
    }
}

/// Pattern for `arith.{add,sub,mul,div,rem}` -> `wasm.{type}_{op}`
struct ArithBinOpPattern;

impl ArenaRewritePattern for ArithBinOpPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let data = ctx.op(op);
        if data.dialect != Symbol::new("arith") {
            return false;
        }

        let name = data.name;
        let is_binop = name == Symbol::new("add")
            || name == Symbol::new("sub")
            || name == Symbol::new("mul")
            || name == Symbol::new("div")
            || name == Symbol::new("rem");

        if !is_binop {
            return false;
        }

        let result_types = ctx.op_result_types(op);
        let Some(&result_ty) = result_types.first() else {
            return false;
        };
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let suffix = type_suffix(ctx, Some(result_ty));
        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("add") {
            match suffix {
                "i32" => arena_wasm::i32_add(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "i64" => arena_wasm::i64_add(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f32" => arena_wasm::f32_add(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f64" => arena_wasm::f64_add(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_add(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("sub") {
            match suffix {
                "i32" => arena_wasm::i32_sub(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "i64" => arena_wasm::i64_sub(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f32" => arena_wasm::f32_sub(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f64" => arena_wasm::f64_sub(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_sub(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("mul") {
            match suffix {
                "i32" => arena_wasm::i32_mul(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "i64" => arena_wasm::i64_mul(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f32" => arena_wasm::f32_mul(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f64" => arena_wasm::f64_mul(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_mul(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("div") {
            match suffix {
                "i32" => arena_wasm::i32_div_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "i64" => arena_wasm::i64_div_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f32" => arena_wasm::f32_div(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f64" => arena_wasm::f64_div(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_div_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("rem") {
            match suffix {
                "i32" => arena_wasm::i32_rem_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "i64" => arena_wasm::i64_rem_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_rem_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `arith.cmp_*` -> `wasm.{type}_{cmp}`
struct ArithCmpPattern;

impl ArenaRewritePattern for ArithCmpPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let data = ctx.op(op);
        if data.dialect != Symbol::new("arith") {
            return false;
        }

        let name = data.name;
        let is_cmp = name == Symbol::new("cmp_eq")
            || name == Symbol::new("cmp_ne")
            || name == Symbol::new("cmp_lt")
            || name == Symbol::new("cmp_le")
            || name == Symbol::new("cmp_gt")
            || name == Symbol::new("cmp_ge");

        if !is_cmp {
            return false;
        }

        // Get operand type from first operand
        let operands = ctx.op_operands(op).to_vec();
        let operand_ty = operands.first().map(|&v| ctx.value_ty(v));
        let suffix = type_suffix_opt(ctx, operand_ty);
        let is_integer = matches!(suffix, "i32" | "i64");

        let result_types = ctx.op_result_types(op);
        let result_ty = result_types[0];
        let loc = ctx.op(op).location;
        let lhs = operands[0];
        let rhs = operands[1];

        let new_op = if name == Symbol::new("cmp_eq") {
            match suffix {
                "i32" => arena_wasm::i32_eq(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "i64" => arena_wasm::i64_eq(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f32" => arena_wasm::f32_eq(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f64" => arena_wasm::f64_eq(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_eq(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("cmp_ne") {
            match suffix {
                "i32" => arena_wasm::i32_ne(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "i64" => arena_wasm::i64_ne(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f32" => arena_wasm::f32_ne(ctx, loc, lhs, rhs, result_ty).op_ref(),
                "f64" => arena_wasm::f64_ne(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_ne(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("cmp_lt") {
            match (suffix, is_integer) {
                ("i32", true) => arena_wasm::i32_lt_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("i64", true) => arena_wasm::i64_lt_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f32", false) => arena_wasm::f32_lt(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f64", false) => arena_wasm::f64_lt(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_lt_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("cmp_le") {
            match (suffix, is_integer) {
                ("i32", true) => arena_wasm::i32_le_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("i64", true) => arena_wasm::i64_le_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f32", false) => arena_wasm::f32_le(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f64", false) => arena_wasm::f64_le(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_le_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("cmp_gt") {
            match (suffix, is_integer) {
                ("i32", true) => arena_wasm::i32_gt_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("i64", true) => arena_wasm::i64_gt_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f32", false) => arena_wasm::f32_gt(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f64", false) => arena_wasm::f64_gt(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_gt_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("cmp_ge") {
            match (suffix, is_integer) {
                ("i32", true) => arena_wasm::i32_ge_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("i64", true) => arena_wasm::i64_ge_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f32", false) => arena_wasm::f32_ge(ctx, loc, lhs, rhs, result_ty).op_ref(),
                ("f64", false) => arena_wasm::f64_ge(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_ge_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `arith.neg` -> `wasm.{f32,f64}_neg` or 0 - x for integers
struct ArithNegPattern;

impl ArenaRewritePattern for ArithNegPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(neg_op) = arena_arith::Neg::from_op(ctx, op) else {
            return false;
        };

        let result_types = ctx.op_result_types(op);
        let result_ty = result_types.first().copied();
        let suffix = type_suffix(ctx, result_ty);
        let loc = ctx.op(op).location;
        let operand = neg_op.operand(ctx);

        match suffix {
            "f32" => {
                let f32_ty = result_ty.unwrap_or_else(|| intern_f32_type(ctx));
                let new_op = arena_wasm::f32_neg(ctx, loc, operand, f32_ty);
                rewriter.replace_op(new_op.op_ref());
                true
            }
            "f64" => {
                let f64_ty = result_ty.unwrap_or_else(|| intern_f64_type(ctx));
                let new_op = arena_wasm::f64_neg(ctx, loc, operand, f64_ty);
                rewriter.replace_op(new_op.op_ref());
                true
            }
            "i64" => {
                let i64_ty = intern_i64_type(ctx);
                let zero = arena_wasm::i64_const(ctx, loc, i64_ty, 0);
                let sub = arena_wasm::i64_sub(ctx, loc, zero.result(ctx), operand, i64_ty);
                rewriter.insert_op(zero.op_ref());
                rewriter.replace_op(sub.op_ref());
                true
            }
            _ => {
                // Default to i32: 0 - x
                let i32_ty = intern_i32_type(ctx);
                let zero = arena_wasm::i32_const(ctx, loc, i32_ty, 0);
                let sub = arena_wasm::i32_sub(ctx, loc, zero.result(ctx), operand, i32_ty);
                rewriter.insert_op(zero.op_ref());
                rewriter.replace_op(sub.op_ref());
                true
            }
        }
    }
}

/// Pattern for `arith.{and,or,xor,shl,shr,shru}` -> `wasm.i{32,64}_{op}`
struct ArithBitwisePattern;

impl ArenaRewritePattern for ArithBitwisePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let data = ctx.op(op);
        if data.dialect != Symbol::new("arith") {
            return false;
        }

        let name = data.name;
        let is_bitwise = name == Symbol::new("and")
            || name == Symbol::new("or")
            || name == Symbol::new("xor")
            || name == Symbol::new("shl")
            || name == Symbol::new("shr")
            || name == Symbol::new("shru");

        if !is_bitwise {
            return false;
        }

        let result_types = ctx.op_result_types(op);
        let Some(&result_ty) = result_types.first() else {
            return false;
        };
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let suffix = type_suffix(ctx, Some(result_ty));
        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("and") {
            match suffix {
                "i64" => arena_wasm::i64_and(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_and(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("or") {
            match suffix {
                "i64" => arena_wasm::i64_or(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_or(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("xor") {
            match suffix {
                "i64" => arena_wasm::i64_xor(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_xor(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("shl") {
            match suffix {
                "i64" => arena_wasm::i64_shl(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_shl(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("shr") {
            match suffix {
                "i64" => arena_wasm::i64_shr_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_shr_s(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("shru") {
            match suffix {
                "i64" => arena_wasm::i64_shr_u(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_wasm::i32_shr_u(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern for `arith.{cast,trunc,extend,convert}` -> wasm conversion ops
struct ArithConversionPattern;

impl ArenaRewritePattern for ArithConversionPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let data = ctx.op(op);
        if data.dialect != Symbol::new("arith") {
            return false;
        }

        let name = data.name;
        let is_conv = name == Symbol::new("cast")
            || name == Symbol::new("trunc")
            || name == Symbol::new("extend")
            || name == Symbol::new("convert");

        if !is_conv {
            return false;
        }

        // Get source type from operand
        let operands = ctx.op_operands(op).to_vec();
        let operand = operands[0];
        let src_ty = ctx.value_ty(operand);
        let src_suffix = type_suffix(ctx, Some(src_ty));

        // Get destination type from result
        let result_types = ctx.op_result_types(op);
        let dst_ty = result_types[0];
        let dst_suffix = type_suffix(ctx, Some(dst_ty));

        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("cast") {
            match (src_suffix, dst_suffix) {
                ("i64", "i32") => arena_wasm::i32_wrap_i64(ctx, loc, operand, dst_ty).op_ref(),
                ("i32", "i64") => arena_wasm::i64_extend_i32_s(ctx, loc, operand, dst_ty).op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("trunc") {
            match (src_suffix, dst_suffix) {
                ("f32", "i32") => arena_wasm::i32_trunc_f32_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f64", "i32") => arena_wasm::i32_trunc_f64_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f32", "i64") => arena_wasm::i64_trunc_f32_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f64", "i64") => arena_wasm::i64_trunc_f64_s(ctx, loc, operand, dst_ty).op_ref(),
                ("i64", "i32") => arena_wasm::i32_wrap_i64(ctx, loc, operand, dst_ty).op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("extend") {
            match (src_suffix, dst_suffix) {
                ("i32", "i64") => arena_wasm::i64_extend_i32_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f32", "f64") => arena_wasm::f64_promote_f32(ctx, loc, operand, dst_ty).op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("convert") {
            match (src_suffix, dst_suffix) {
                ("i32", "f32") => arena_wasm::f32_convert_i32_s(ctx, loc, operand, dst_ty).op_ref(),
                ("i32", "f64") => arena_wasm::f64_convert_i32_s(ctx, loc, operand, dst_ty).op_ref(),
                ("i64", "f32") => arena_wasm::f32_convert_i64_s(ctx, loc, operand, dst_ty).op_ref(),
                ("i64", "f64") => arena_wasm::f64_convert_i64_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f32", "i32") => arena_wasm::i32_trunc_f32_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f64", "i32") => arena_wasm::i32_trunc_f64_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f32", "i64") => arena_wasm::i64_trunc_f32_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f64", "i64") => arena_wasm::i64_trunc_f64_s(ctx, loc, operand, dst_ty).op_ref(),
                ("f32", "f64") => arena_wasm::f64_promote_f32(ctx, loc, operand, dst_ty).op_ref(),
                ("f64", "f32") => arena_wasm::f32_demote_f64(ctx, loc, operand, dst_ty).op_ref(),
                _ => return false,
            }
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Get the wasm type suffix from a TypeRef.
pub(crate) fn type_suffix(ctx: &IrContext, ty: Option<TypeRef>) -> &'static str {
    match ty {
        Some(t) => type_suffix_opt(ctx, Some(t)),
        None => {
            #[cfg(debug_assertions)]
            warn!("No type in arith_to_wasm, defaulting to i32");
            "i32"
        }
    }
}

fn type_suffix_opt(ctx: &IrContext, ty: Option<TypeRef>) -> &'static str {
    match ty {
        Some(t) => {
            let data = ctx.types.get(t);
            let name = data.name;
            if name == Symbol::new("i32") {
                "i32"
            } else if name == Symbol::new("i64") {
                "i64"
            } else if name == Symbol::new("f32") {
                "f32"
            } else if name == Symbol::new("f64") {
                "f64"
            } else if name == Symbol::new("i1") {
                "i32"
            } else if name == Symbol::new("int")
                || name == Symbol::new("nat")
                || name == Symbol::new("bool")
            {
                "i32"
            } else if name == Symbol::new("nil") {
                "nil"
            } else {
                #[cfg(debug_assertions)]
                warn!(
                    "Unknown type '{}' in arith_to_wasm, defaulting to i32",
                    name
                );
                "i32"
            }
        }
        None => {
            #[cfg(debug_assertions)]
            warn!("No type in arith_to_wasm, defaulting to i32");
            "i32"
        }
    }
}

/// Intern a core.i32 type.
pub(crate) fn intern_i32_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Intern a core.i64 type.
pub(crate) fn intern_i64_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build())
}

/// Intern a core.f32 type.
pub(crate) fn intern_f32_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f32")).build())
}

/// Intern a core.f64 type.
pub(crate) fn intern_f64_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build())
}
