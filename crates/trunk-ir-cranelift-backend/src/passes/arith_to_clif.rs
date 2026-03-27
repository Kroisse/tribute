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
                let Attribute::FloatBits(v) = value else {
                    return false;
                };
                arena_clif::f32const(ctx, loc, result_ty, f32::from_bits(v as u32)).op_ref()
            }
            "f64" => {
                let Attribute::FloatBits(v) = value else {
                    return false;
                };
                arena_clif::f64const(ctx, loc, result_ty, f64::from_bits(v)).op_ref()
            }
            _ => match value {
                Attribute::Int(v) => {
                    let Some(v) = i64::try_from(v).ok() else {
                        return false;
                    };
                    arena_clif::iconst(ctx, loc, result_ty, v).op_ref()
                }
                Attribute::Bool(b) => {
                    arena_clif::iconst(ctx, loc, result_ty, if b { 1 } else { 0 }).op_ref()
                }
                _ => return false,
            },
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

        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let name = data.name;

        let new_op = if name == Symbol::new("addi") {
            arena_clif::iadd(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("addf") {
            arena_clif::fadd(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("subi") {
            arena_clif::isub(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("subf") {
            arena_clif::fsub(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("muli") {
            arena_clif::imul(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("mulf") {
            arena_clif::fmul(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("divsi") {
            arena_clif::sdiv(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("divui") {
            arena_clif::udiv(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("divf") {
            arena_clif::fdiv(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("remsi") {
            arena_clif::srem(ctx, loc, lhs, rhs, result_ty).op_ref()
        } else if name == Symbol::new("remui") {
            arena_clif::urem(ctx, loc, lhs, rhs, result_ty).op_ref()
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
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;

        if let Ok(cmpi) = arith::Cmpi::from_op(ctx, op) {
            let lhs = cmpi.lhs(ctx);
            let rhs = cmpi.rhs(ctx);
            let cond = cmpi.predicate(ctx);
            let new_op = arena_clif::icmp(ctx, loc, lhs, rhs, result_ty, cond).op_ref();
            rewriter.replace_op(new_op);
            true
        } else if let Ok(cmpf) = arith::Cmpf::from_op(ctx, op) {
            let lhs = cmpf.lhs(ctx);
            let rhs = cmpf.rhs(ctx);
            let predicate = cmpf.predicate(ctx);
            // Map arith ordered predicates to clif conditions
            let cond_str = predicate.to_string();
            let cond = match cond_str.as_str() {
                "oeq" => Symbol::new("eq"),
                "one" => Symbol::new("ne"),
                "olt" => Symbol::new("lt"),
                "ole" => Symbol::new("le"),
                "ogt" => Symbol::new("gt"),
                "oge" => Symbol::new("ge"),
                _ => predicate,
            };
            let new_op = arena_clif::fcmp(ctx, loc, lhs, rhs, result_ty, cond).op_ref();
            rewriter.replace_op(new_op);
            true
        } else {
            false
        }
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
        let Some(ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;

        if let Ok(negi) = arith::Negi::from_op(ctx, op) {
            let operand = negi.operand(ctx);
            rewriter.replace_op(arena_clif::ineg(ctx, loc, operand, ty).op_ref());
            true
        } else if let Ok(negf) = arith::Negf::from_op(ctx, op) {
            let operand = negf.operand(ctx);
            rewriter.replace_op(arena_clif::fneg(ctx, loc, operand, ty).op_ref());
            true
        } else {
            false
        }
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

#[cfg(test)]
mod tests {
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::TypeConverter;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = TypeConverter::new();
        super::lower(&mut ctx, module, type_converter);
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_arith_const_bool() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i8 {
    %0 = arith.const {value = true} : core.bool
    %1 = arith.const {value = false} : core.bool
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }
}
