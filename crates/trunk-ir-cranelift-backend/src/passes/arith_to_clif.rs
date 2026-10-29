//! Lower arith dialect operations to clif dialect.
//!
//! This pass converts arithmetic operations to their Cranelift equivalents:
//! - `arith.const` -> `clif.iconst` / `clif.f32const` / `clif.f64const`
//! - `core.nil_value` -> `clif.iconst 0`
//! - `arith.{add,sub,mul,div,rem}` -> `clif.{iadd,isub,imul,sdiv,srem}` / `clif.{fadd,fsub,fmul,fdiv}`
//! - `arith.cmp_*` -> `clif.icmp` / `clif.fcmp` with `cond` attribute
//! - `arith.neg` -> `clif.ineg` / `clif.fneg` (native support, no expansion needed)
//! - `arith.{and,or,xor,shl,shr,shru}` -> `clif.{band,bor,bxor,ishl,sshr,ushr}`
//! - `arith.{extsi,extui,trunci}` -> `clif.{sextend,uextend,ireduce}`
//! - `arith.{sitofp,uitofp,fptosi,fptoui}` -> `clif.fcvt_{from,to}_{sint,uint}`
//! - `arith.{extf,truncf}` -> `clif.{fpromote,fdemote}`

use trunk_ir::StringArg;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::core::{self, FloatLike, IntegerLike};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, OpRewritePattern, PatternApplicator,
    PatternRewriter, RewritePattern, TypeConverter,
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
        .add_pattern(NilValuePattern)
        .add_pattern(ArithBinOpPattern)
        .add_pattern(ArithCmpPattern)
        .add_pattern(ArithNegPattern)
        .add_pattern(ArithBitwisePattern)
        .add_pattern(ArithConversionPattern)
        .add_pattern(ArithIndexCastPattern)
        .with_target(arith_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "arith-to-clif")?;
    Ok(())
}

fn arith_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect(clif::DIALECT_NAME)
        .illegal_dialect(arith::DIALECT_NAME)
        .illegal_op("core.nil_value")
}

/// Classify a type into the clif lowering category.
///
/// `core` integer types carry no signedness, so signed conversions are used
/// unless an operation says otherwise. `arith` does not handle pointers, so
/// `core.ptr` has no category. Types without a category leave their
/// operations unconverted.
fn type_category(ctx: &IrContext, ty: TypeRef) -> Option<&'static str> {
    if IntegerLike::matches(ctx, ty) {
        return Some("int");
    }
    match FloatLike::width(ctx, ty) {
        Some(32) => Some("f32"),
        Some(_) => Some("f64"),
        None => None,
    }
}

/// `core.nil_value` materializes as a zero placeholder. Nil results are
/// projected away at calls and returns.
struct NilValuePattern;

impl OpRewritePattern for NilValuePattern {
    type Op = core::NilValue;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        nil_value: core::NilValue,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = nil_value.op_ref();
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let new_op = clif::Iconst::operands()
            .value(0)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

struct ArithConstPattern;

impl OpRewritePattern for ArithConstPattern {
    type Op = arith::Const;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        const_op: arith::Const,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = const_op.op_ref();
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

        let new_op_ref = match category {
            "f32" => {
                let Attribute::FloatBits(v) = value else {
                    return false;
                };
                clif::F32const::operands()
                    .value(f32::from_bits(v as u32))
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            "f64" => {
                let Attribute::FloatBits(v) = value else {
                    return false;
                };
                clif::F64const::operands()
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
                    clif::Iconst::operands()
                        .value(v)
                        .results(result_ty)
                        .build(ctx, loc)
                        .op_ref()
                }
                Attribute::Bool(b) => clif::Iconst::operands()
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
        if data.name.dialect(ctx) != arith::DIALECT_NAME {
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

        let new_op = if name == arith::Addi::NAME {
            clif::Iadd::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Addf::NAME {
            clif::Fadd::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Subi::NAME {
            clif::Isub::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Subf::NAME {
            clif::Fsub::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Muli::NAME {
            clif::Imul::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Mulf::NAME {
            clif::Fmul::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Divsi::NAME {
            clif::Sdiv::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Divui::NAME {
            clif::Udiv::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Divf::NAME {
            clif::Fdiv::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Remsi::NAME {
            clif::Srem::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Remui::NAME {
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
pub fn finalize_cmp(
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
        let i8_ty = TypeDataBuilder::new("core.i8").intern(ctx);

        if let Ok(cmpi) = arith::Cmpi::from_op(ctx, op) {
            let lhs = cmpi.lhs(ctx);
            let rhs = cmpi.rhs(ctx);
            let cond = cmpi.predicate_ref(ctx);
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
            // Map arith float predicates to clif conditions
            let cond: StringArg = match cmpf.predicate(ctx) {
                "oeq" => "eq".into(),
                "une" => "ne".into(),
                "olt" => "lt".into(),
                "ole" => "le".into(),
                "ogt" => "gt".into(),
                "oge" => "ge".into(),
                _ => cmpf.predicate_ref(ctx).into(),
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
        if data.name.dialect(ctx) != arith::DIALECT_NAME {
            return false;
        }

        let name = data.name;
        let is_bitwise = [
            arith::And::NAME,
            arith::Or::NAME,
            arith::Xor::NAME,
            arith::Shl::NAME,
            arith::Shr::NAME,
            arith::Shru::NAME,
        ]
        .contains(&name);
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

        let new_op = if name == arith::And::NAME {
            clif::Band::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Or::NAME {
            clif::Bor::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Xor::NAME {
            clif::Bxor::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Shl::NAME {
            clif::Ishl::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Shr::NAME {
            clif::Sshr::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Shru::NAME {
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
        let Some(&operand) = ctx.op_operands(op).first() else {
            return false;
        };
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;
        // Cranelift holds `core.i1` as an `i8` 0 or 1, so only zero extension
        // of a boolean past 8 bits is a plain instruction. Other boolean
        // conversions stay unconverted and fail the conversion boundary.
        let width = |ctx: &IrContext, ty| IntegerLike::width(ctx, ty);
        let bool_source = width(ctx, ctx.value_ty(operand)) == Some(1);
        if (bool_source
            && !(arith::Extui::matches(ctx, op)
                && width(ctx, result_ty).is_some_and(|width| width > 8)))
            || width(ctx, result_ty) == Some(1)
        {
            return false;
        }
        // Operand and result types are constrained by each operation's
        // declaration, so the op alone selects the Cranelift instruction.
        let new_op = if arith::Extsi::matches(ctx, op) {
            clif::Sextend::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Extui::matches(ctx, op) {
            clif::Uextend::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Trunci::matches(ctx, op) {
            clif::Ireduce::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Sitofp::matches(ctx, op) {
            clif::FcvtFromSint::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Uitofp::matches(ctx, op) {
            clif::FcvtFromUint::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Fptosi::matches(ctx, op) {
            clif::FcvtToSint::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Fptoui::matches(ctx, op) {
            clif::FcvtToUint::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Extf::matches(ctx, op) {
            clif::Fpromote::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if arith::Truncf::matches(ctx, op) {
            clif::Fdemote::operands(operand)
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

/// `arith.index_castsi` / `arith.index_castui` between the fixed-width
/// integers that type conversion gave both sides: an extension by the
/// operation's signedness, a reduction, or nothing when the widths agree.
struct ArithIndexCastPattern;

impl RewritePattern for ArithIndexCastPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let signed = arith::IndexCastsi::matches(ctx, op);
        if !signed && !arith::IndexCastui::matches(ctx, op) {
            return false;
        }
        let Some(&operand) = ctx.op_operands(op).first() else {
            return false;
        };
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        // An index without a target width stays unconverted and fails the
        // conversion boundary.
        let (Some(from), Some(to)) = (
            IntegerLike::width(ctx, ctx.value_ty(operand)),
            IntegerLike::width(ctx, result_ty),
        ) else {
            return false;
        };
        // As for `arith.extsi`, a boolean held as an `i8` 0 or 1 has no
        // sign-extending instruction.
        if (from == 1 && signed) || to == 1 {
            return false;
        }
        if from == to {
            rewriter.erase_op(vec![operand]);
            return true;
        }
        let loc = ctx.op(op).location;
        let new_op = if from > to {
            clif::Ireduce::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if signed {
            clif::Sextend::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else {
            clif::Uextend::operands(operand)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
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

    /// Lower `ir` with `core.index` converted to `core.i64`.
    fn run_pass_with_64_bit_index(ir: &str) -> String {
        use trunk_ir::dialect::core;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let i64_ty = core::I64::type_ref(&mut ctx);
        let mut type_converter = TypeConverter::new();
        type_converter
            .add_conversion(move |ctx, ty| core::Index::matches(ctx, ty).then_some(i64_ty));
        super::lower(&mut ctx, module, type_converter).unwrap();
        print_module(&ctx, module.op())
    }

    #[test]
    fn index_casts_lower_to_extensions_and_reductions_at_the_converted_width() {
        let output = run_pass_with_64_bit_index(
            r#"core.module @test {
  func.func @f(%n: core.i32, %w: core.i64) -> core.i64 {
    %a = arith.index_castsi %n : core.index
    %b = arith.index_castui %n : core.index
    %c = arith.index_castsi %w : core.index
    %sum = arith.addi %a, %b : core.index
    %d = arith.addi %sum, %c : core.index
    %narrow = arith.index_castui %d : core.i32
    %wide = arith.index_castsi %d : core.i64
    func.return %wide
  }
}"#,
        );
        assert!(output.contains("clif.sextend"), "{output}");
        assert!(output.contains("clif.uextend"), "{output}");
        assert!(output.contains("clif.ireduce"), "{output}");
        assert!(!output.contains("arith."), "{output}");
        assert!(!output.contains(": core.index"), "{output}");
        insta::assert_snapshot!(output);
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
    fn boolean_conversions_other_than_zero_extension_are_rejected() {
        for body in [
            "%s = arith.extsi %b : core.i32",
            "%t = arith.trunci %n : core.i1",
            "%f = arith.uitofp %b : core.f64",
            "%u = arith.extui %b : core.i8",
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    "core.module @test {{\n  func.func @f(%b: core.i1, %n: core.i8) {{\n    {body}\n    func.return\n  }}\n}}"
                ),
            );
            let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
            assert!(
                format!("{error:?}").contains("IllegalOp"),
                "{body}: {error:?}"
            );
        }
    }

    #[test]
    fn arithmetic_constants_are_never_pointers() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() {
    %null = arith.const {value = 0} : core.ptr
    func.return
  }
}"#,
        );
        let error = super::lower(&mut ctx, module, TypeConverter::new()).unwrap_err();
        assert!(format!("{error:?}").contains("IllegalOp"), "{error:?}");
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
    fn each_conversion_selects_its_signedness() {
        let result = run_pass(
            r#"core.module @test {
  func.func @convert(%b: core.i1, %w: core.i64, %n: core.i32, %x: core.f32, %y: core.f64) {
    %s = arith.extsi %n : core.i64
    %u = arith.extui %b : core.i32
    %t = arith.trunci %w : core.i32
    %sf = arith.sitofp %n : core.f64
    %uf = arith.uitofp %n : core.f32
    %si = arith.fptosi %y : core.i32
    %ui = arith.fptoui %x : core.i64
    %e = arith.extf %x : core.f64
    %r = arith.truncf %y : core.f32
    func.return
  }
}"#,
        );
        for (op, count) in [
            ("clif.sextend", 1),
            ("clif.uextend", 1),
            ("clif.ireduce", 1),
            ("clif.fcvt_from_sint", 1),
            ("clif.fcvt_from_uint", 1),
            ("clif.fcvt_to_sint", 1),
            ("clif.fcvt_to_uint", 1),
            ("clif.fpromote", 1),
            ("clif.fdemote", 1),
        ] {
            assert_eq!(result.matches(op).count(), count, "{op}:\n{result}");
        }
        assert!(!result.contains("arith."), "{result}");
    }
}
