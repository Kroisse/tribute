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
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core::{self, FloatLike, IntegerLike};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::{Attribute, TypeDataBuilder};

/// Lower arith dialect to clif dialect.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<(), ConversionError> {
    let applicator = PatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(ArithConstPattern)
        .add_pattern(ArithBinOpPattern)
        .add_pattern(ArithCmpPattern)
        .add_pattern(ArithNegPattern)
        .add_pattern(ArithBitwisePattern)
        .add_pattern(ArithConversionPattern)
        .with_target(arith_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "arith-to-clif")?;
    Ok(())
}

fn arith_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("arith")
}

/// Classify a type into the clif lowering category.
///
/// `core` integers and `core.ptr` (a pointer-sized integer in clif) lower as
/// integers. `core` integer types carry no signedness, so signed conversions
/// are used unless an operation says otherwise. Other types have no category,
/// and patterns leave their operations unconverted.
fn type_category(ctx: &IrContext, ty: TypeRef) -> Option<&'static str> {
    if IntegerLike::matches(ctx, ty) || core::Ptr::matches(ctx, ty) {
        return Some("int");
    }
    match FloatLike::width(ctx, ty) {
        Some(32) => Some("f32"),
        Some(_) => Some("f64"),
        None if core::Nil::matches(ctx, ty) => Some("nil"),
        None => None,
    }
}

/// Bit width clif uses for an integer-category type.
///
/// `core.i1` is materialized as `i8`. `core.ptr` is 64 bits wide on the
/// supported native targets (x86_64 and aarch64).
fn clif_int_width(ctx: &IrContext, ty: TypeRef) -> u32 {
    match IntegerLike::width(ctx, ty) {
        Some(1) => 8,
        Some(width) => width,
        None => 64,
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

        let Some(category) = type_category(ctx, result_ty) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = const_op.value(ctx);

        if category == "nil" {
            let new_op = clif::Iconst::builder()
                .value(0)
                .results(result_ty)
                .build(ctx, loc);
            rewriter.replace_op(new_op.op_ref());
            return true;
        }

        let new_op_ref = match category {
            "f32" => {
                let Attribute::FloatBits(v) = value else {
                    return false;
                };
                clif::F32const::builder()
                    .value(f32::from_bits(v as u32))
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            "f64" => {
                let Attribute::FloatBits(v) = value else {
                    return false;
                };
                clif::F64const::builder()
                    .value(f64::from_bits(v))
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            _ => match value {
                Attribute::Int(v) => {
                    let Some(v) = i64::try_from(v).ok() else {
                        return false;
                    };
                    clif::Iconst::builder()
                        .value(v)
                        .results(result_ty)
                        .build(ctx, loc)
                        .op_ref()
                }
                Attribute::Bool(b) => clif::Iconst::builder()
                    .value(if b { 1 } else { 0 })
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
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
            clif::Iadd::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("addf") {
            clif::Fadd::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("subi") {
            clif::Isub::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("subf") {
            clif::Fsub::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("muli") {
            clif::Imul::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("mulf") {
            clif::Fmul::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("divsi") {
            clif::Sdiv::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("divui") {
            clif::Udiv::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("divf") {
            clif::Fdiv::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("remsi") {
            clif::Srem::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("remui") {
            clif::Urem::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else {
            return false;
        };

        rewriter.replace_op(new_op);
        true
    }
}

/// Emit the comparison result, extending from i8 to the expected width if needed.
///
/// Cranelift's `icmp`/`fcmp` always return i8. If the converted result type
/// is wider (e.g. i32), insert a `clif.uextend` after the comparison.
fn finalize_cmp(
    ctx: &mut IrContext,
    loc: trunk_ir::types::Location,
    rewriter: &mut trunk_ir::rewrite::PatternRewriter<'_>,
    cmp_op: OpRef,
    cmp_result: trunk_ir::refs::ValueRef,
    result_ty: TypeRef,
    i8_ty: TypeRef,
) {
    if result_ty == i8_ty {
        rewriter.replace_op(cmp_op);
    } else {
        rewriter.insert_op(cmp_op);
        let ext_op = clif::Uextend::operands(cmp_result)
            .results(result_ty)
            .build(ctx, loc)
            .op_ref();
        rewriter.replace_op(ext_op);
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
        // Cranelift's icmp/fcmp always return i8, so emit the comparison with
        // core.i8 result, then uextend to the converted result type (core.i32)
        // so downstream consumers get the expected width.
        let i8_ty = ctx.intern_type(TypeDataBuilder::new("core", "i8").build());

        if let Ok(cmpi) = arith::Cmpi::from_op(ctx, op) {
            let lhs = cmpi.lhs(ctx);
            let rhs = cmpi.rhs(ctx);
            let cond = cmpi.predicate(ctx);
            let cmp_op = clif::Icmp::operands(lhs, rhs)
                .cond(cond)
                .results(i8_ty)
                .build(ctx, loc);
            finalize_cmp(
                ctx,
                loc,
                rewriter,
                cmp_op.op_ref(),
                cmp_op.result(ctx),
                result_ty,
                i8_ty,
            );
            true
        } else if let Ok(cmpf) = arith::Cmpf::from_op(ctx, op) {
            let lhs = cmpf.lhs(ctx);
            let rhs = cmpf.rhs(ctx);
            let predicate = cmpf.predicate(ctx);
            // Map arith float predicates to clif conditions
            let cond_str = predicate.to_string();
            let cond = match cond_str.as_str() {
                "oeq" => Symbol::new("eq"),
                "une" => Symbol::new("ne"),
                "olt" => Symbol::new("lt"),
                "ole" => Symbol::new("le"),
                "ogt" => Symbol::new("gt"),
                "oge" => Symbol::new("ge"),
                _ => predicate,
            };
            let cmp_op = clif::Fcmp::operands(lhs, rhs)
                .cond(cond)
                .results(i8_ty)
                .build(ctx, loc);
            finalize_cmp(
                ctx,
                loc,
                rewriter,
                cmp_op.op_ref(),
                cmp_op.result(ctx),
                result_ty,
                i8_ty,
            );
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
            rewriter.replace_op(
                clif::Ineg::operands(operand)
                    .results(ty)
                    .build(ctx, loc)
                    .op_ref(),
            );
            true
        } else if let Ok(negf) = arith::Negf::from_op(ctx, op) {
            let operand = negf.operand(ctx);
            rewriter.replace_op(
                clif::Fneg::operands(operand)
                    .results(ty)
                    .build(ctx, loc)
                    .op_ref(),
            );
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
            clif::Band::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("or") {
            clif::Bor::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("xor") {
            clif::Bxor::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("shl") {
            clif::Ishl::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("shr") {
            clif::Sshr::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == Symbol::new("shru") {
            clif::Ushr::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
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
        let Some(src_cat) = type_category(ctx, src_ty) else {
            return false;
        };

        let Some(dst_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let Some(dst_cat) = type_category(ctx, dst_ty) else {
            return false;
        };

        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("cast") {
            match (src_cat, dst_cat) {
                ("int", "int") => {
                    if clif_int_width(ctx, dst_ty) > clif_int_width(ctx, src_ty) {
                        clif::Sextend::operands(operand)
                            .results(dst_ty)
                            .build(ctx, loc)
                            .op_ref()
                    } else {
                        clif::Ireduce::operands(operand)
                            .results(dst_ty)
                            .build(ctx, loc)
                            .op_ref()
                    }
                }
                _ => return false,
            }
        } else if name == Symbol::new("trunc") {
            match (src_cat, dst_cat) {
                ("f32" | "f64", "int") => clif::FcvtToSint::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("int", "int") => clif::Ireduce::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("extend") {
            match (src_cat, dst_cat) {
                ("int", "int") => clif::Sextend::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "f64") => clif::Fpromote::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("convert") {
            match (src_cat, dst_cat) {
                ("int", "f32" | "f64") => clif::FcvtFromSint::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32" | "f64", "int") => clif::FcvtToSint::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "f64") => clif::Fpromote::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "f32") => clif::Fdemote::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
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
        super::lower(&mut ctx, module, type_converter).unwrap();
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_arith_const_bool() {
        let result = run_pass(
            r#"core.module @test {
  func.func @test_fn() -> core.i1 {
    %0 = arith.const {value = true} : core.i1
    %1 = arith.const {value = false} : core.i1
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn pointers_lower_as_integers() {
        let result = run_pass(
            r#"core.module @test {
  func.func @f() {
    %null = arith.const {value = 0} : core.ptr
    func.return
  }
}"#,
        );
        assert!(
            result.contains("clif.iconst {value = 0} : core.ptr"),
            "{result}"
        );
    }

    #[test]
    fn pointer_casts_use_the_native_pointer_width() {
        let result = run_pass(
            r#"core.module @test {
  func.func @f(%index: core.i32, %address: core.ptr) {
    %widened = arith.cast %index : core.ptr
    %narrowed = arith.cast %address : core.i32
    func.return
  }
}"#,
        );
        assert!(result.contains("clif.sextend %0 : core.ptr"), "{result}");
        assert!(result.contains("clif.ireduce %1 : core.i32"), "{result}");
    }

    #[test]
    fn uncategorized_types_fail_the_conversion_boundary() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() {
    %opaque = arith.const {value = 0} : test.opaque
    func.return
  }
}"#,
        );
        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(format!("{error:?}").contains("IllegalOp"), "{error:?}");
    }

    #[test]
    fn conversions_follow_core_scalar_categories() {
        let result = run_pass(
            r#"core.module @test {
  func.func @convert(%b: core.i1, %w: core.i64, %n: core.i32, %x: core.f32, %y: core.f64) {
    %widened = arith.cast %b : core.i32
    %narrowed = arith.cast %w : core.i32
    %extended = arith.extend %n : core.i64
    %to_float = arith.convert %n : core.f64
    %to_int = arith.convert %y : core.i32
    %promoted = arith.extend %x : core.f64
    func.return
  }
}"#,
        );
        for (op, count) in [
            ("clif.sextend", 2),
            ("clif.ireduce", 1),
            ("clif.fcvt_from_sint", 1),
            ("clif.fcvt_to_sint", 1),
            ("clif.fpromote", 1),
        ] {
            assert_eq!(result.matches(op).count(), count, "{op}:\n{result}");
        }
        assert!(!result.contains("arith."), "{result}");
        assert!(!result.contains("uextend"), "{result}");
    }
}
