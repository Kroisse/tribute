//! Lower arith dialect operations to clif dialect.
//!
//! This pass converts arithmetic operations to their Cranelift equivalents:
//! - `arith.constant` -> `clif.iconst` / `clif.f32const` / `clif.f64const`
//! - `core.nil_value` -> `clif.iconst 0`
//! - `arith.{add,sub,mul,div,rem}` -> `clif.{iadd,isub,imul,sdiv,srem}` / `clif.{fadd,fsub,fmul,fdiv}`
//! - `arith.cmp_*` -> `clif.icmp` / `clif.fcmp` with `cond` attribute
//! - `arith.neg` -> `clif.ineg` / `clif.fneg` (native support, no expansion needed)
//! - `arith.{andi,ori,xori,shli,shrsi,shrui}` -> `clif.{band,bor,bxor,ishl,sshr,ushr}`
//! - `arith.{extsi,extui,trunci}` -> `clif.{sextend,uextend,ireduce}`
//! - `arith.{sitofp,uitofp,fptosi,fptoui}` -> `clif.fcvt_{from,to}_{sint,uint}`
//! - `arith.{extf,truncf}` -> `clif.{fpromote,fdemote}`
//!
//! An integer narrower than the Cranelift integer that holds it, such as
//! `core.i1` in an `I8`, stays zero-extended: results that can set the bits
//! above its declared width are masked back to it (see
//! `design/cranelift-backend.md`, "Cranelift 정수보다 좁은 정수").

use trunk_ir::OpName;
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
    // Type conversion may widen the operands of a comparison before it is
    // lowered, so signed comparisons of booleans become unsigned first.
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(BooleanSignedCmpPattern)
        .apply_partial(ctx, module);
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

/// A signed `arith.cmpi` of `core.i1` operands, rewritten to the unsigned
/// predicate that reads their zero-extended 0 or 1 the same way.
struct BooleanSignedCmpPattern;

impl OpRewritePattern for BooleanSignedCmpPattern {
    type Op = arith::Cmpi;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        cmpi: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let (lhs, rhs) = (cmpi.lhs(ctx), cmpi.rhs(ctx));
        if !core::I1::matches(ctx, ctx.value_ty(lhs)) {
            return None;
        }
        let predicate = cmpi.predicate(ctx);
        let unsigned = predicate.on_unsigned_booleans();
        if unsigned == predicate {
            return None;
        }
        let loc = ctx.op(cmpi.op_ref()).location;
        let new_op = arith::Cmpi::operands(lhs, rhs)
            .predicate(unsigned)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
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
        nil_value: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = nil_value.op_ref();
        let result_ty = rewriter.result_type(ctx, op, 0)?;
        let loc = ctx.op(op).location;
        let new_op = clif::Iconst::operands()
            .value(0)
            .results(result_ty)
            .build(ctx, loc);
        rewriter.replace_op(new_op.op_ref());
        Some(())
    }
}

/// The mask that keeps an integer of type `declared` zero-extended in the
/// Cranelift integer that holds its `converted` type, when the declared
/// width is the narrower one.
///
/// Cranelift's narrowest integer is `I8`, which also holds `core.i1`, and
/// type conversion may widen an integer further. Bits above the declared
/// width are zero in every value, so an operation whose result can set them
/// clears them.
fn zero_extension_mask(ctx: &IrContext, declared: TypeRef, converted: TypeRef) -> Option<i64> {
    let width = IntegerLike::width(ctx, declared)?;
    let held = IntegerLike::width(ctx, converted)?.max(8);
    (width < held).then(|| (1i64 << width) - 1)
}

/// Replace the operation being rewritten with `new_op`, followed by a
/// `clif.band` with `mask` when the result must be cleared above its
/// declared width.
fn replace_zero_extended(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    new_op: OpRef,
    mask: Option<i64>,
) {
    let Some(mask) = mask else {
        rewriter.replace_op(new_op);
        return;
    };
    rewriter.insert_op(new_op);
    let value = ctx.op_results(new_op)[0];
    let ty = ctx.value_ty(value);
    let loc = ctx.op(new_op).location;
    let mask = clif::Iconst::operands()
        .value(mask)
        .results(ty)
        .build(ctx, loc);
    rewriter.insert_op(mask.op_ref());
    let band = clif::Band::operands(value, mask.result(ctx))
        .results(ty)
        .build(ctx, loc);
    rewriter.replace_op(band.op_ref());
}

/// The `iconst` immediate of `value` in an integer type of `width` bits, if
/// `value` is in the signed or unsigned range of that width.
fn integer_immediate(value: i128, width: u32) -> Option<i64> {
    if width >= 64 {
        return i64::try_from(value)
            .ok()
            .or_else(|| u64::try_from(value).ok().map(|value| value as i64));
    }
    let min = -(1i128 << (width - 1));
    let max = (1i128 << width) - 1;
    (min..=max).contains(&value).then_some(value as i64)
}

struct ArithConstPattern;

impl OpRewritePattern for ArithConstPattern {
    type Op = arith::Constant;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        const_op: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = const_op.op_ref();
        let result_types = ctx.op_result_types(op);
        let &raw_result_ty = result_types.first()?;

        let result_ty = rewriter
            .type_converter()
            .convert_type(ctx, raw_result_ty)
            .unwrap_or(raw_result_ty);

        let category = type_category(ctx, result_ty)?;
        let loc = ctx.op(op).location;
        let value = const_op.value(ctx);

        let new_op_ref = match category {
            "f32" => {
                let Attribute::FloatBits(v) = value else {
                    return rewriter.no_match(format!("expected FloatBits for f32, got {value:?}"));
                };
                clif::F32const::operands()
                    .value(f32::from_bits(v as u32))
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            "f64" => {
                let Attribute::FloatBits(v) = value else {
                    return rewriter.no_match(format!("expected FloatBits for f64, got {value:?}"));
                };
                clif::F64const::operands()
                    .value(f64::from_bits(v))
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            _ => match value {
                Attribute::Int(v) => {
                    let width = IntegerLike::width(ctx, raw_result_ty).unwrap_or(64);
                    let v = rewriter.require(
                        integer_immediate(v, width),
                        "integer constant does not fit its type",
                    )?;
                    let v = match zero_extension_mask(ctx, raw_result_ty, result_ty) {
                        Some(mask) => v & mask,
                        None => v,
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
                _ => return rewriter.no_match(format!("expected Int or Bool, got {value:?}")),
            },
        };

        rewriter.replace_op(new_op_ref);
        Some(())
    }
}

struct ArithBinOpPattern;

impl RewritePattern for ArithBinOpPattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[
            arith::Addi::NAME,
            arith::Subi::NAME,
            arith::Muli::NAME,
            arith::Divsi::NAME,
            arith::Divui::NAME,
            arith::Remsi::NAME,
            arith::Remui::NAME,
            arith::Addf::NAME,
            arith::Subf::NAME,
            arith::Mulf::NAME,
            arith::Divf::NAME,
        ])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let data = ctx.op(op);
        if data.name.dialect(ctx) != arith::DIALECT_NAME {
            return None;
        }

        let result_ty = rewriter.result_type(ctx, op, 0)?;
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return None;
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
            return None;
        };

        // Wrapping addition, subtraction, and multiplication carry into the
        // bits above a narrower declared width.
        let mask = [arith::Addi::NAME, arith::Subi::NAME, arith::Muli::NAME]
            .contains(&name)
            .then(|| zero_extension_mask(ctx, ctx.op_result_types(op)[0], result_ty))
            .flatten();
        replace_zero_extended(ctx, rewriter, new_op, mask);
        Some(())
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
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[arith::Cmpi::NAME, arith::Cmpf::NAME])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let result_ty = rewriter.result_type(ctx, op, 0)?;
        let loc = ctx.op(op).location;
        // Cranelift's icmp/fcmp always return i8, so emit the comparison with
        // core.i8 result, then uextend to the converted result type (core.i32)
        // so downstream consumers get the expected width.
        let i8_ty = TypeDataBuilder::new("core.i8").intern(ctx);

        if let Ok(cmpi) = arith::Cmpi::from_op(ctx, op) {
            let lhs = cmpi.lhs(ctx);
            let rhs = cmpi.rhs(ctx);
            let cmp_op = clif::Icmp::operands(lhs, rhs)
                .cond(cmpi.predicate(ctx).into())
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
            Some(())
        } else if let Ok(cmpf) = arith::Cmpf::from_op(ctx, op) {
            let lhs = cmpf.lhs(ctx);
            let rhs = cmpf.rhs(ctx);
            let cmp_op = clif::Fcmp::operands(lhs, rhs)
                .cond(cmpf.predicate(ctx).into())
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
            Some(())
        } else {
            None
        }
    }
}

struct ArithNegPattern;

impl RewritePattern for ArithNegPattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[arith::Negi::NAME, arith::Negf::NAME])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let ty = rewriter.result_type(ctx, op, 0)?;
        let loc = ctx.op(op).location;

        if let Ok(negi) = arith::Negi::from_op(ctx, op) {
            let operand = negi.operand(ctx);
            let ineg = clif::Ineg::operands(operand)
                .results(ty)
                .build(ctx, loc)
                .op_ref();
            let mask = zero_extension_mask(ctx, ctx.op_result_types(op)[0], ty);
            replace_zero_extended(ctx, rewriter, ineg, mask);
            Some(())
        } else if let Ok(negf) = arith::Negf::from_op(ctx, op) {
            let operand = negf.operand(ctx);
            rewriter.replace_op(
                clif::Fneg::operands(operand)
                    .results(ty)
                    .build(ctx, loc)
                    .op_ref(),
            );
            Some(())
        } else {
            None
        }
    }
}

struct ArithBitwisePattern;

impl RewritePattern for ArithBitwisePattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[
            arith::Andi::NAME,
            arith::Ori::NAME,
            arith::Xori::NAME,
            arith::Shli::NAME,
            arith::Shrsi::NAME,
            arith::Shrui::NAME,
        ])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let data = ctx.op(op);
        if data.name.dialect(ctx) != arith::DIALECT_NAME {
            return None;
        }

        let name = data.name;
        let is_bitwise = [
            arith::Andi::NAME,
            arith::Ori::NAME,
            arith::Xori::NAME,
            arith::Shli::NAME,
            arith::Shrsi::NAME,
            arith::Shrui::NAME,
        ]
        .contains(&name);
        if !is_bitwise {
            return None;
        }

        let result_ty = rewriter.result_type(ctx, op, 0)?;
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return None;
        };
        let loc = ctx.op(op).location;

        let new_op = if name == arith::Andi::NAME {
            clif::Band::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Ori::NAME {
            clif::Bor::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Xori::NAME {
            clif::Bxor::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Shli::NAME {
            clif::Ishl::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Shrsi::NAME {
            clif::Sshr::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else if name == arith::Shrui::NAME {
            clif::Ushr::operands(lhs, rhs)
                .results(result_ty)
                .build(ctx, loc)
                .op_ref()
        } else {
            return None;
        };

        // A left shift moves bits past a narrower declared width.
        let mask = (name == arith::Shli::NAME)
            .then(|| zero_extension_mask(ctx, ctx.op_result_types(op)[0], result_ty))
            .flatten();
        replace_zero_extended(ctx, rewriter, new_op, mask);
        Some(())
    }
}

struct ArithConversionPattern;

impl RewritePattern for ArithConversionPattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[
            arith::Extsi::NAME,
            arith::Extui::NAME,
            arith::Trunci::NAME,
            arith::Sitofp::NAME,
            arith::Uitofp::NAME,
            arith::Fptosi::NAME,
            arith::Fptoui::NAME,
            arith::Extf::NAME,
            arith::Truncf::NAME,
        ])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let &operand = ctx.op_operands(op).first()?;
        let result_ty = rewriter.result_type(ctx, op, 0)?;
        let loc = ctx.op(op).location;
        // Cranelift holds `core.i1` as an `i8` 0 or 1, so only zero extension
        // of a boolean past 8 bits is a plain instruction. Other boolean
        // conversions stay unconverted and fail the conversion boundary.
        let width = |ctx: &IrContext, ty| IntegerLike::width(ctx, ty);
        let bool_source = width(ctx, ctx.value_ty(operand)) == Some(1);
        rewriter.ensure(
            !((bool_source
                && !(arith::Extui::matches(ctx, op)
                    && width(ctx, result_ty).is_some_and(|width| width > 8)))
                || width(ctx, result_ty) == Some(1)),
            "boolean conversion other than zero extension past 8 bits",
        )?;
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
            return None;
        };
        rewriter.replace_op(new_op);
        Some(())
    }
}

/// `arith.index_cast` / `arith.index_castui` between the fixed-width
/// integers that type conversion gave both sides: an extension by the
/// operation's signedness, a reduction, or nothing when the widths agree.
struct ArithIndexCastPattern;

impl RewritePattern for ArithIndexCastPattern {
    fn roots(&self) -> Option<&[OpName]> {
        Some(&[arith::IndexCast::NAME, arith::IndexCastui::NAME])
    }

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let signed = arith::IndexCast::matches(ctx, op);
        if !signed && !arith::IndexCastui::matches(ctx, op) {
            return None;
        }
        let &operand = ctx.op_operands(op).first()?;
        let result_ty = rewriter.result_type(ctx, op, 0)?;
        // An index without a target width stays unconverted and fails the
        // conversion boundary.
        let (Some(from), Some(to)) = (
            IntegerLike::width(ctx, ctx.value_ty(operand)),
            IntegerLike::width(ctx, result_ty),
        ) else {
            return None;
        };
        // As for `arith.extsi`, a boolean held as an `i8` 0 or 1 has no
        // sign-extending instruction.
        rewriter.ensure(
            !((from == 1 && signed) || to == 1),
            "index cast to or sign-extending from a boolean",
        )?;
        if from == to {
            rewriter.erase_op(vec![operand]);
            return Some(());
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
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::integer_immediate;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::rewrite::TypeConverter;

    fn run_pass(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let type_converter = TypeConverter::new();
        super::lower(&mut ctx, module, type_converter).unwrap();
        module.op().display(&ctx).to_string()
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
        module.op().display(&ctx).to_string()
    }

    /// Evaluate the value `@f` returns after lowering, with each `clif`
    /// integer operation computed at the width of the Cranelift integer
    /// that holds its type, and return its bits zero-extended.
    fn evaluate_lowered(ctx: &IrContext, module: trunk_ir::rewrite::Module) -> u64 {
        use std::ops::ControlFlow;
        use trunk_ir::dialect::core::IntegerLike;
        use trunk_ir::dialect::{clif, core, func};
        use trunk_ir::ops::DialectOp;
        use trunk_ir::walk::{WalkAction, walk_op};

        let held = |value| IntegerLike::width(ctx, ctx.value_ty(value)).unwrap().max(8);
        let mut values = rustc_hash::FxHashMap::<trunk_ir::refs::ValueRef, u64>::default();
        let mut returned = None;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let operand = |i: usize| values[&ctx.op_operands(op)[i]];
            let signed = |i: usize| {
                let shift = 64 - held(ctx.op_operands(op)[i]);
                ((operand(i) << shift) as i64) >> shift
            };
            let result = if let Ok(icmp) = clif::Icmp::from_op(ctx, op) {
                use clif::IntCond as C;
                let (a, b, sa, sb) = (operand(0), operand(1), signed(0), signed(1));
                Some(u64::from(match icmp.cond(ctx) {
                    C::Eq => a == b,
                    C::Ne => a != b,
                    C::Slt => sa < sb,
                    C::Sle => sa <= sb,
                    C::Sgt => sa > sb,
                    C::Sge => sa >= sb,
                    C::Ult => a < b,
                    C::Ule => a <= b,
                    C::Ugt => a > b,
                    C::Uge => a >= b,
                }))
            } else if clif::Uextend::matches(ctx, op) {
                Some(operand(0))
            } else if let Ok(iconst) = clif::Iconst::from_op(ctx, op) {
                Some(iconst.value(ctx) as u64)
            } else if clif::Iadd::matches(ctx, op) {
                Some(operand(0).wrapping_add(operand(1)))
            } else if clif::Isub::matches(ctx, op) {
                Some(operand(0).wrapping_sub(operand(1)))
            } else if clif::Imul::matches(ctx, op) {
                Some(operand(0).wrapping_mul(operand(1)))
            } else if clif::Ineg::matches(ctx, op) {
                Some(operand(0).wrapping_neg())
            } else if clif::Band::matches(ctx, op) {
                Some(operand(0) & operand(1))
            } else if core::UnrealizedConversionCast::matches(ctx, op) {
                Some(operand(0))
            } else if func::Return::matches(ctx, op) {
                returned = Some(operand(0));
                None
            } else {
                assert!(
                    ctx.op_results(op).is_empty(),
                    "unexpected operation {}",
                    ctx.op(op).name.display(ctx)
                );
                None
            };
            if let Some(bits) = result {
                let value = ctx.op_results(op)[0];
                let held = held(value);
                let mask = if held == 64 {
                    u64::MAX
                } else {
                    (1 << held) - 1
                };
                values.insert(value, bits & mask);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        returned.expect("@f returns a value")
    }

    /// The constant `@f` returns once arith folds ran, zero-extended from
    /// `core.i1`.
    fn fold_to_i1(ir: &str) -> u64 {
        use trunk_ir::analysis::AnalysisCache;
        use trunk_ir::dialect::{arith, func};
        use trunk_ir::ops::DialectOp;
        use trunk_ir::pass::Pass;
        use trunk_ir::transforms::canonicalize::Canonicalize;

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let f = module
            .ops(&ctx)
            .iter()
            .find_map(|&op| func::Func::from_op(&ctx, op).ok())
            .unwrap();
        Canonicalize
            .run(&mut ctx, f, &mut AnalysisCache::default())
            .unwrap();
        let body = f.body(&ctx);
        let block = ctx.region(body).blocks[0];
        let &ret = ctx.block(block).ops.last().unwrap();
        let value = ctx.op_operands(ret)[0];
        let trunk_ir::refs::ValueDef::OpResult(def, _) = ctx.value_def(value) else {
            panic!("@f returns a block argument");
        };
        let constant = arith::Constant::from_op(&ctx, def).unwrap();
        match constant.value(&ctx) {
            trunk_ir::types::Attribute::Int(v) => (v & 1) as u64,
            trunk_ir::types::Attribute::Bool(b) => u64::from(b),
            other => panic!("unexpected folded constant {other:?}"),
        }
    }

    /// Lower `ir`, with `core.i1` held in an `I8` or converted to `core.i32`
    /// as the native pipeline does, and evaluate what `@f` returns.
    fn lower_and_evaluate(ir: &str, i1_to_i32: bool) -> u64 {
        use trunk_ir::dialect::core;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let mut type_converter = TypeConverter::new();
        if i1_to_i32 {
            let i32_ty = core::I32::type_ref(&mut ctx);
            type_converter
                .add_conversion(move |ctx, ty| core::I1::matches(ctx, ty).then_some(i32_ty));
        }
        super::lower(&mut ctx, module, type_converter).unwrap();
        evaluate_lowered(&ctx, module)
    }

    #[test]
    fn boolean_arithmetic_agrees_with_its_folded_result() {
        // Integer constants, which the folds read; `true` reads as -1.
        let literal = |bit: bool| if bit { "-1" } else { "0" };
        for a in [false, true] {
            for b in [false, true] {
                let module = |body: &str| {
                    format!(
                        "core.module @test {{\n  func.func @f() -> core.i1 {{\n    \
                         %z = arith.constant 0 : core.i1\n    \
                         %a = arith.constant {} : core.i1\n    \
                         %b = arith.constant {} : core.i1\n    \
                         {body}\n    func.return %r\n  }}\n}}",
                        literal(a),
                        literal(b),
                    )
                };
                // `negi` has no fold; it agrees with the fold of `0 - a`.
                for (body, folded_body) in [
                    ("%r = arith.addi %a, %b : core.i1", None),
                    ("%r = arith.subi %a, %b : core.i1", None),
                    ("%r = arith.muli %a, %b : core.i1", None),
                    (
                        "%r = arith.negi %a : core.i1",
                        Some("%r = arith.subi %z, %a : core.i1"),
                    ),
                ] {
                    let folded = fold_to_i1(&module(folded_body.unwrap_or(body)));
                    for i1_to_i32 in [false, true] {
                        assert_eq!(
                            lower_and_evaluate(&module(body), i1_to_i32),
                            folded,
                            "{body} with a = {a}, b = {b}, i1 as i32: {i1_to_i32}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn boolean_comparisons_read_their_declared_width() {
        let literal = |bit: bool| if bit { "-1" } else { "0" };
        for a in [false, true] {
            for b in [false, true] {
                // Signed, a set bit of `core.i1` is -1.
                let (ua, ub) = (i8::from(a), i8::from(b));
                let (sa, sb) = (-ua, -ub);
                for (predicate, expected) in [
                    ("eq", a == b),
                    ("ne", a != b),
                    ("slt", sa < sb),
                    ("sle", sa <= sb),
                    ("sgt", sa > sb),
                    ("sge", sa >= sb),
                    ("ult", ua < ub),
                    ("ule", ua <= ub),
                    ("ugt", ua > ub),
                    ("uge", ua >= ub),
                ] {
                    // `%a` is a constant and `%b` the result of `0 - b`.
                    let ir = format!(
                        "core.module @test {{\n  func.func @f() -> core.i1 {{\n    \
                         %z = arith.constant 0 : core.i1\n    \
                         %a = arith.constant {} : core.i1\n    \
                         %nb = arith.constant {} : core.i1\n    \
                         %b = arith.subi %z, %nb : core.i1\n    \
                         %r = arith.cmpi {predicate}, %a, %b : core.i1\n    \
                         func.return %r\n  }}\n}}",
                        literal(a),
                        literal(b),
                    );
                    for i1_to_i32 in [false, true] {
                        assert_eq!(
                            lower_and_evaluate(&ir, i1_to_i32),
                            u64::from(expected),
                            "cmpi {predicate} with a = {a}, b = {b}, i1 as i32: {i1_to_i32}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn boolean_constants_and_left_shifts_stay_zero_extended() {
        let output = run_pass(
            r#"core.module @test {
  func.func @f(%a: core.i1, %b: core.i1) -> core.i1 {
    %t = arith.constant -1 : core.i1
    %s = arith.shli %a, %b : core.i1
    %r = arith.andi %s, %t : core.i1
    func.return %r
  }
}"#,
        );
        assert!(!output.contains("clif.iconst -1"), "{output}");
        assert_eq!(output.matches("clif.iconst 1 ").count(), 2, "{output}");
        assert_eq!(output.matches("clif.band").count(), 2, "{output}");
    }

    #[test]
    fn native_width_arithmetic_is_not_masked() {
        let output = run_pass(
            r#"core.module @test {
  func.func @f(%a: core.i8, %b: core.i8) -> core.i8 {
    %s = arith.addi %a, %b : core.i8
    %n = arith.negi %s : core.i8
    func.return %n
  }
}"#,
        );
        assert!(!output.contains("clif.band"), "{output}");
    }

    #[test]
    fn integer_immediates_accept_the_signed_and_unsigned_range_of_their_width() {
        assert_eq!(integer_immediate(-128, 8), Some(-128));
        assert_eq!(integer_immediate(255, 8), Some(255));
        assert_eq!(integer_immediate(256, 8), None);
        assert_eq!(integer_immediate(-129, 8), None);
        assert_eq!(integer_immediate(u64::MAX.into(), 64), Some(-1));
        assert_eq!(integer_immediate(i64::MIN.into(), 64), Some(i64::MIN));
        assert_eq!(integer_immediate(i128::from(u64::MAX) + 1, 64), None);
    }

    #[test]
    fn index_casts_lower_to_extensions_and_reductions_at_the_converted_width() {
        let output = run_pass_with_64_bit_index(
            r#"core.module @test {
  func.func @f(%n: core.i32, %w: core.i64) -> core.i64 {
    %a = arith.index_cast %n : core.i32 to core.index
    %b = arith.index_castui %n : core.i32 to core.index
    %c = arith.index_cast %w : core.i64 to core.index
    %sum = arith.addi %a, %b : core.index
    %d = arith.addi %sum, %c : core.index
    %narrow = arith.index_castui %d : core.index to core.i32
    %wide = arith.index_cast %d : core.index to core.i64
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
    %0 = arith.constant true : core.i1
    %1 = arith.constant false : core.i1
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn boolean_conversions_other_than_zero_extension_are_rejected() {
        for body in [
            "%s = arith.extsi %b : core.i1 to core.i32",
            "%t = arith.trunci %n : core.i8 to core.i1",
            "%f = arith.uitofp %b : core.i1 to core.f64",
            "%u = arith.extui %b : core.i1 to core.i8",
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
    %null = arith.constant 0 : core.ptr
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
    %opaque = arith.constant 0 : test.opaque
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
    %s = arith.extsi %n : core.i32 to core.i64
    %u = arith.extui %b : core.i1 to core.i32
    %t = arith.trunci %w : core.i64 to core.i32
    %sf = arith.sitofp %n : core.i32 to core.f64
    %uf = arith.uitofp %n : core.i32 to core.f32
    %si = arith.fptosi %y : core.f64 to core.i32
    %ui = arith.fptoui %x : core.f32 to core.i64
    %e = arith.extf %x : core.f32 to core.f64
    %r = arith.truncf %y : core.f64 to core.f32
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
