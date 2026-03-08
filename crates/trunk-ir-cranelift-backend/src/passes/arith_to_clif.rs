//! Lower arith dialect operations to clif dialect.
//!
//! This pass converts arithmetic operations to their Cranelift equivalents:
//! - `arith.const` -> `clif.iconst` / `clif.f32const` / `clif.f64const`
//! - `arith.{add,sub,mul,div,rem}` -> `clif.{iadd,isub,imul,sdiv,srem}` / `clif.{fadd,fsub,fmul,fdiv}`
//! - `arith.cmp_*` -> `clif.icmp` / `clif.fcmp` with `cond` attribute
//! - `arith.neg` -> `clif.ineg` / `clif.fneg` (native support, no expansion needed)
//! - `arith.{and,or,xor,shl,shr,shru}` -> `clif.{band,bor,bxor,ishl,sshr,ushr}`
//! - `arith.{cast,trunc,extend,convert}` -> `clif.{ireduce,sextend,fpromote,fdemote,fcvt_*}`

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::clif as arena_clif;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Attribute;

/// Lower arith dialect to clif dialect.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
    use trunk_ir::rewrite::ConversionTarget;

    let mut target = ConversionTarget::new();
    target.add_legal_dialect("clif");
    target.add_illegal_dialect("arith");

    let applicator = PatternApplicator::new(type_converter)
        .with_target(target)
        .add_pattern(ArithConstPattern)
        .add_pattern(ArithBinOpPattern)
        .add_pattern(ArithCmpPattern)
        .add_pattern(ArithNegPattern)
        .add_pattern(ArithBitwisePattern)
        .add_pattern(ArithConversionPattern);
    applicator.apply_partial(ctx, module);
}

/// Classify arena type into integer vs float category (for clif lowering).
fn type_category(ctx: &IrContext, ty: Option<TypeRef>) -> &'static str {
    match ty {
        Some(t) => {
            let name = ctx.types.get(t).name;
            if name == Symbol::new("i1")
                || name == Symbol::new("i8")
                || name == Symbol::new("i16")
                || name == Symbol::new("i32")
                || name == Symbol::new("i64")
                || name == Symbol::new("int")
                || name == Symbol::new("nat")
                || name == Symbol::new("bool")
            {
                "int"
            } else if name == Symbol::new("f32") {
                "f32"
            } else if name == Symbol::new("f64") {
                "f64"
            } else if name == Symbol::new("nil") {
                "nil"
            } else {
                "int"
            }
        }
        None => "int",
    }
}

fn is_unsigned_int(ctx: &IrContext, ty: Option<TypeRef>) -> bool {
    match ty {
        Some(t) => {
            let name = ctx.types.get(t).name;
            name == Symbol::new("nat") || name == Symbol::new("bool")
        }
        None => false,
    }
}

fn is_wider_int(ctx: &IrContext, dst: TypeRef, src: Option<TypeRef>) -> bool {
    let width = |t: TypeRef| -> u8 {
        let name = ctx.types.get(t).name;
        if name == Symbol::new("i64") {
            64
        } else if name == Symbol::new("i32")
            || name == Symbol::new("int")
            || name == Symbol::new("nat")
        {
            32
        } else if name == Symbol::new("i16") {
            16
        } else if name == Symbol::new("i8")
            || name == Symbol::new("bool")
            || name == Symbol::new("i1")
        {
            8
        } else {
            32
        }
    };
    match src {
        Some(s) => width(dst) > width(s),
        None => true,
    }
}

struct ArithConstPattern;

impl RewritePattern for ArithConstPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(const_op) = arith::Const::from_op(ctx, op) else {
            return false;
        };

        let result_types = ctx.op_result_types(op);
        let Some(&raw_result_ty) = result_types.first() else {
            return false;
        };

        let result_ty = rewriter
            .type_converter()
            .convert_type(ctx, raw_result_ty)
            .unwrap_or(raw_result_ty);

        let category = type_category(ctx, Some(result_ty));
        let loc = ctx.op(op).location;
        let value = const_op.value(ctx);

        if category == "nil" {
            let new_op = arena_clif::iconst(ctx, loc, result_ty, 0);
            rewriter.replace_op(new_op.op_ref());
            return true;
        }

        let new_op_ref = match category {
            "f32" => {
                if let Attribute::FloatBits(v) = value {
                    arena_clif::f32const(ctx, loc, result_ty, f32::from_bits(v as u32)).op_ref()
                } else {
                    arena_clif::f32const(ctx, loc, result_ty, 0.0).op_ref()
                }
            }
            "f64" => {
                if let Attribute::FloatBits(v) = value {
                    arena_clif::f64const(ctx, loc, result_ty, f64::from_bits(v)).op_ref()
                } else {
                    arena_clif::f64const(ctx, loc, result_ty, 0.0).op_ref()
                }
            }
            _ => {
                if let Attribute::Int(v) = value {
                    arena_clif::iconst(ctx, loc, result_ty, v as i64).op_ref()
                } else {
                    arena_clif::iconst(ctx, loc, result_ty, 0).op_ref()
                }
            }
        };

        rewriter.replace_op(new_op_ref);
        true
    }
}

struct ArithBinOpPattern;

impl RewritePattern for ArithBinOpPattern {
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

        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        // Use raw (pre-conversion) result type for signedness check, since
        // the type converter maps nat/bool → core.i32/i8, losing unsigned info.
        let raw_result_ty = ctx.op_result_types(op).first().copied();
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let category = type_category(ctx, Some(result_ty));
        let loc = ctx.op(op).location;
        let is_unsigned = is_unsigned_int(ctx, raw_result_ty);

        let new_op = if name == Symbol::new("add") {
            match category {
                "f32" | "f64" => arena_clif::fadd(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_clif::iadd(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("sub") {
            match category {
                "f32" | "f64" => arena_clif::fsub(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_clif::isub(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("mul") {
            match category {
                "f32" | "f64" => arena_clif::fmul(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_clif::imul(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("div") {
            match category {
                "f32" | "f64" => arena_clif::fdiv(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ if is_unsigned => arena_clif::udiv(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_clif::sdiv(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else if name == Symbol::new("rem") {
            match category {
                "f32" | "f64" => return false,
                _ if is_unsigned => arena_clif::urem(ctx, loc, lhs, rhs, result_ty).op_ref(),
                _ => arena_clif::srem(ctx, loc, lhs, rhs, result_ty).op_ref(),
            }
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}

struct ArithCmpPattern;

impl RewritePattern for ArithCmpPattern {
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

        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };

        let operand_ty = Some(ctx.value_ty(lhs));
        let category = type_category(ctx, operand_ty);
        let is_float = matches!(category, "f32" | "f64");
        let is_unsigned = !is_float && is_unsigned_int(ctx, operand_ty);

        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;

        let (cond, use_fcmp) = if name == Symbol::new("cmp_eq") {
            ("eq", is_float)
        } else if name == Symbol::new("cmp_ne") {
            ("ne", is_float)
        } else if name == Symbol::new("cmp_lt") {
            if is_float {
                ("lt", true)
            } else if is_unsigned {
                ("ult", false)
            } else {
                ("slt", false)
            }
        } else if name == Symbol::new("cmp_le") {
            if is_float {
                ("le", true)
            } else if is_unsigned {
                ("ule", false)
            } else {
                ("sle", false)
            }
        } else if name == Symbol::new("cmp_gt") {
            if is_float {
                ("gt", true)
            } else if is_unsigned {
                ("ugt", false)
            } else {
                ("sgt", false)
            }
        } else if name == Symbol::new("cmp_ge") {
            if is_float {
                ("ge", true)
            } else if is_unsigned {
                ("uge", false)
            } else {
                ("sge", false)
            }
        } else {
            return false;
        };

        let cond_sym = Symbol::new(cond);
        let new_op = if use_fcmp {
            arena_clif::fcmp(ctx, loc, lhs, rhs, result_ty, cond_sym).op_ref()
        } else {
            arena_clif::icmp(ctx, loc, lhs, rhs, result_ty, cond_sym).op_ref()
        };

        rewriter.replace_op(new_op);
        true
    }
}

struct ArithNegPattern;

impl RewritePattern for ArithNegPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(neg_op) = arith::Neg::from_op(ctx, op) else {
            return false;
        };

        let result_ty = rewriter.result_type(ctx, op, 0);
        let Some(ty) = result_ty else {
            return false;
        };
        let category = type_category(ctx, result_ty);
        let loc = ctx.op(op).location;
        let operand = neg_op.operand(ctx);

        let new_op = match category {
            "f32" | "f64" => arena_clif::fneg(ctx, loc, operand, ty).op_ref(),
            _ => arena_clif::ineg(ctx, loc, operand, ty).op_ref(),
        };

        rewriter.replace_op(new_op);
        true
    }
}

struct ArithBitwisePattern;

impl RewritePattern for ArithBitwisePattern {
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

        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("and") {
            arena_clif::band(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("or") {
            arena_clif::bor(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("xor") {
            arena_clif::bxor(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("shl") {
            arena_clif::ishl(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("shr") {
            arena_clif::sshr(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("shru") {
            arena_clif::ushr(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}

struct ArithConversionPattern;

impl RewritePattern for ArithConversionPattern {
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

        let operands = ctx.op_operands(op).to_vec();
        let Some(&operand) = operands.first() else {
            return false;
        };

        let src_ty = ctx.value_ty(operand);
        let src_cat = type_category(ctx, Some(src_ty));

        let Some(dst_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        // Use raw (pre-conversion) result type for signedness checks.
        let raw_dst_ty = ctx.op_result_types(op).first().copied();
        let dst_cat = type_category(ctx, Some(dst_ty));

        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("cast") {
            match (src_cat, dst_cat) {
                ("int", "int") => {
                    if is_wider_int(ctx, dst_ty, Some(src_ty)) {
                        arena_clif::sextend(ctx, loc, operand, dst_ty).op_ref()
                    } else {
                        arena_clif::ireduce(ctx, loc, operand, dst_ty).op_ref()
                    }
                }
                _ => return false,
            }
        } else if name == Symbol::new("trunc") {
            match (src_cat, dst_cat) {
                ("f32" | "f64", "int") => {
                    arena_clif::fcvt_to_sint(ctx, loc, operand, dst_ty).op_ref()
                }
                ("int", "int") => arena_clif::ireduce(ctx, loc, operand, dst_ty).op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("extend") {
            match (src_cat, dst_cat) {
                ("int", "int") if is_unsigned_int(ctx, Some(src_ty)) => {
                    arena_clif::uextend(ctx, loc, operand, dst_ty).op_ref()
                }
                ("int", "int") => arena_clif::sextend(ctx, loc, operand, dst_ty).op_ref(),
                ("f32", "f64") => arena_clif::fpromote(ctx, loc, operand, dst_ty).op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("convert") {
            match (src_cat, dst_cat) {
                ("int", "f32" | "f64") if is_unsigned_int(ctx, Some(src_ty)) => {
                    arena_clif::fcvt_from_uint(ctx, loc, operand, dst_ty).op_ref()
                }
                ("int", "f32" | "f64") => {
                    arena_clif::fcvt_from_sint(ctx, loc, operand, dst_ty).op_ref()
                }
                ("f32" | "f64", "int") if is_unsigned_int(ctx, raw_dst_ty) => {
                    arena_clif::fcvt_to_uint(ctx, loc, operand, dst_ty).op_ref()
                }
                ("f32" | "f64", "int") => {
                    arena_clif::fcvt_to_sint(ctx, loc, operand, dst_ty).op_ref()
                }
                ("f32", "f64") => arena_clif::fpromote(ctx, loc, operand, dst_ty).op_ref(),
                ("f64", "f32") => arena_clif::fdemote(ctx, loc, operand, dst_ty).op_ref(),
                _ => return false,
            }
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}
