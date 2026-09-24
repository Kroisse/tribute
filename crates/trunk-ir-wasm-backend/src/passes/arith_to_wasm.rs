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

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::core::{self, FloatLike, IntegerLike};
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Attribute;

/// Lower arith dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
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

impl RewritePattern for ArithConstPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_const_op) = arith::Const::from_op(ctx, op) else {
            return false;
        };

        let result_types = ctx.op_result_types(op);
        let Some(&result_ty) = result_types.first() else {
            return false;
        };

        // Handle nil type constants specially
        let Some(type_name) = type_suffix(ctx, result_ty) else {
            return false;
        };
        if type_name == "nil" {
            let loc = ctx.op(op).location;
            let nop = wasm_dialect::Nop::operands()
                .results(result_ty)
                .build(ctx, loc);
            rewriter.replace_op(nop.op_ref());
            return true;
        }

        let loc = ctx.op(op).location;
        let value = _const_op.value(ctx);

        let new_op_ref = match type_name {
            "i32" => match value {
                Attribute::Int(v) => wasm_dialect::I32Const::operands()
                    .value(v as i32)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                Attribute::Bool(b) => wasm_dialect::I32Const::operands()
                    .value(if b { 1 } else { 0 })
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => {
                    warn!("arith.const: expected Int or Bool for i32, got {:?}", value);
                    return false;
                }
            },
            "i64" => {
                let Attribute::Int(v) = value else {
                    warn!("arith.const: expected Int for i64, got {:?}", value);
                    return false;
                };
                wasm_dialect::I64Const::operands()
                    .value(v as i64)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            "f32" => {
                let Attribute::FloatBits(v) = value else {
                    warn!("arith.const: expected FloatBits for f32, got {:?}", value);
                    return false;
                };
                wasm_dialect::F32Const::operands()
                    .value(f32::from_bits(v as u32))
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            "f64" => {
                let Attribute::FloatBits(v) = value else {
                    warn!("arith.const: expected FloatBits for f64, got {:?}", value);
                    return false;
                };
                wasm_dialect::F64Const::operands()
                    .value(f64::from_bits(v))
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref()
            }
            _ => {
                warn!("arith.const: unsupported type suffix '{}'", type_name);
                return false;
            }
        };

        rewriter.replace_op(new_op_ref);
        true
    }
}

/// Pattern for `arith.{addi,addf,subi,...}` -> `wasm.{type}_{op}`
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

        let result_types = ctx.op_result_types(op);
        let Some(&result_ty) = result_types.first() else {
            return false;
        };
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let Some(suffix) = type_suffix(ctx, result_ty) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let name = data.name;

        let new_op = if name == Symbol::new("addi") {
            match suffix {
                // `wasm.i32_add` is declared on `core.i32` and infers that
                // result. Narrower integers have no defined wrap-around
                // representation here, so they stay unlowered.
                "i32" if core::I32::matches(ctx, result_ty) => {
                    wasm_dialect::I32Add::operands(lhs, rhs)
                        .build(ctx, loc)
                        .op_ref()
                }
                "i64" => wasm_dialect::I64Add::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("addf") {
            match suffix {
                "f32" => wasm_dialect::F32Add::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "f64" => wasm_dialect::F64Add::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("subi") {
            match suffix {
                "i32" => wasm_dialect::I32Sub::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i64" => wasm_dialect::I64Sub::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("subf") {
            match suffix {
                "f32" => wasm_dialect::F32Sub::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "f64" => wasm_dialect::F64Sub::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("muli") {
            match suffix {
                "i32" => wasm_dialect::I32Mul::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i64" => wasm_dialect::I64Mul::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("mulf") {
            match suffix {
                "f32" => wasm_dialect::F32Mul::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "f64" => wasm_dialect::F64Mul::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("divsi") {
            match suffix {
                "i32" => wasm_dialect::I32DivS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i64" => wasm_dialect::I64DivS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("divui") {
            match suffix {
                "i32" => wasm_dialect::I32DivU::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i64" => wasm_dialect::I64DivU::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("divf") {
            match suffix {
                "f32" => wasm_dialect::F32Div::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "f64" => wasm_dialect::F64Div::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("remsi") {
            match suffix {
                "i32" => wasm_dialect::I32RemS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i64" => wasm_dialect::I64RemS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("remui") {
            match suffix {
                "i32" => wasm_dialect::I32RemU::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i64" => wasm_dialect::I64RemU::operands(lhs, rhs)
                    .results(result_ty)
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

/// Pattern for `arith.cmpi` / `arith.cmpf` -> `wasm.{type}_{cmp}`
struct ArithCmpPattern;

impl RewritePattern for ArithCmpPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let result_types = ctx.op_result_types(op);
        let Some(&result_ty) = result_types.first() else {
            return false;
        };
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let loc = ctx.op(op).location;

        if let Ok(cmpi) = arith::Cmpi::from_op(ctx, op) {
            let predicate = cmpi.predicate(ctx);
            let Some(suffix) = type_suffix(ctx, ctx.value_ty(lhs)) else {
                return false;
            };

            let pred_str = predicate.to_string();
            let new_op = match (suffix, pred_str.as_str()) {
                ("i32", "eq") => wasm_dialect::I32Eq::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "eq") => wasm_dialect::I64Eq::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "ne") => wasm_dialect::I32Ne::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "ne") => wasm_dialect::I64Ne::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "slt") => wasm_dialect::I32LtS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "slt") => wasm_dialect::I64LtS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "sle") => wasm_dialect::I32LeS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "sle") => wasm_dialect::I64LeS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "sgt") => wasm_dialect::I32GtS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "sgt") => wasm_dialect::I64GtS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "sge") => wasm_dialect::I32GeS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "sge") => wasm_dialect::I64GeS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "ult") => wasm_dialect::I32LtU::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            };
            rewriter.replace_op(new_op);
            true
        } else if let Ok(cmpf) = arith::Cmpf::from_op(ctx, op) {
            let predicate = cmpf.predicate(ctx);
            let Some(suffix) = type_suffix(ctx, ctx.value_ty(lhs)) else {
                return false;
            };

            let pred_str = predicate.to_string();
            let new_op = match (suffix, pred_str.as_str()) {
                ("f32", "oeq") => wasm_dialect::F32Eq::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "oeq") => wasm_dialect::F64Eq::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "une") => wasm_dialect::F32Ne::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "une") => wasm_dialect::F64Ne::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "olt") => wasm_dialect::F32Lt::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "olt") => wasm_dialect::F64Lt::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "ole") => wasm_dialect::F32Le::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "ole") => wasm_dialect::F64Le::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "ogt") => wasm_dialect::F32Gt::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "ogt") => wasm_dialect::F64Gt::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "oge") => wasm_dialect::F32Ge::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "oge") => wasm_dialect::F64Ge::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            };
            rewriter.replace_op(new_op);
            true
        } else {
            false
        }
    }
}

/// Pattern for `arith.negi` / `arith.negf` -> wasm ops
struct ArithNegPattern;

impl RewritePattern for ArithNegPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let loc = ctx.op(op).location;

        if let Ok(negf) = arith::Negf::from_op(ctx, op) {
            let Some(&result_ty) = ctx.op_result_types(op).first() else {
                return false;
            };
            let Some(suffix) = type_suffix(ctx, result_ty) else {
                return false;
            };
            let operand = negf.operand(ctx);

            match suffix {
                "f32" => {
                    rewriter.replace_op(
                        wasm_dialect::F32Neg::operands(operand)
                            .results(result_ty)
                            .build(ctx, loc)
                            .op_ref(),
                    );
                    true
                }
                "f64" => {
                    rewriter.replace_op(
                        wasm_dialect::F64Neg::operands(operand)
                            .results(result_ty)
                            .build(ctx, loc)
                            .op_ref(),
                    );
                    true
                }
                _ => false,
            }
        } else if let Ok(negi) = arith::Negi::from_op(ctx, op) {
            let Some(&result_ty) = ctx.op_result_types(op).first() else {
                return false;
            };
            let Some(suffix) = type_suffix(ctx, result_ty) else {
                return false;
            };
            let operand = negi.operand(ctx);

            match suffix {
                "i32" => {
                    let i32_ty = intern_i32_type(ctx);
                    let zero = wasm_dialect::I32Const::operands()
                        .value(0)
                        .results(i32_ty)
                        .build(ctx, loc);
                    let sub = wasm_dialect::I32Sub::operands(zero.result(ctx), operand)
                        .results(i32_ty)
                        .build(ctx, loc);
                    rewriter.insert_op(zero.op_ref());
                    rewriter.replace_op(sub.op_ref());
                    true
                }
                "i64" => {
                    let i64_ty = intern_i64_type(ctx);
                    let zero = wasm_dialect::I64Const::operands()
                        .value(0)
                        .results(i64_ty)
                        .build(ctx, loc);
                    let sub = wasm_dialect::I64Sub::operands(zero.result(ctx), operand)
                        .results(i64_ty)
                        .build(ctx, loc);
                    rewriter.insert_op(zero.op_ref());
                    rewriter.replace_op(sub.op_ref());
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }
}

/// Pattern for `arith.{and,or,xor,shl,shr,shru}` -> `wasm.i{32,64}_{op}`
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

        let result_types = ctx.op_result_types(op);
        let Some(&result_ty) = result_types.first() else {
            return false;
        };
        let operands = ctx.op_operands(op).to_vec();
        let (Some(&lhs), Some(&rhs)) = (operands.first(), operands.get(1)) else {
            return false;
        };
        let Some(suffix) = type_suffix(ctx, result_ty) else {
            return false;
        };
        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("and") {
            match suffix {
                "i64" => wasm_dialect::I64And::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i32" => wasm_dialect::I32And::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("or") {
            match suffix {
                "i64" => wasm_dialect::I64Or::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i32" => wasm_dialect::I32Or::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("xor") {
            match suffix {
                "i64" => wasm_dialect::I64Xor::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i32" => wasm_dialect::I32Xor::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("shl") {
            match suffix {
                "i64" => wasm_dialect::I64Shl::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i32" => wasm_dialect::I32Shl::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("shr") {
            match suffix {
                "i64" => wasm_dialect::I64ShrS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i32" => wasm_dialect::I32ShrS::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("shru") {
            match suffix {
                "i64" => wasm_dialect::I64ShrU::operands(lhs, rhs)
                    .results(result_ty)
                    .build(ctx, loc)
                    .op_ref(),
                "i32" => wasm_dialect::I32ShrU::operands(lhs, rhs)
                    .results(result_ty)
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

/// Pattern for `arith.{cast,trunc,extend,convert}` -> wasm conversion ops
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

        // Get source type from operand
        let operands = ctx.op_operands(op).to_vec();
        let Some(&operand) = operands.first() else {
            return false;
        };
        let src_ty = ctx.value_ty(operand);
        let Some(src_suffix) = type_suffix(ctx, src_ty) else {
            return false;
        };

        // Get destination type from result
        let result_types = ctx.op_result_types(op);
        let Some(&dst_ty) = result_types.first() else {
            return false;
        };
        let Some(dst_suffix) = type_suffix(ctx, dst_ty) else {
            return false;
        };

        let loc = ctx.op(op).location;

        let new_op = if name == Symbol::new("cast") {
            match (src_suffix, dst_suffix) {
                ("i64", "i32") => wasm_dialect::I32WrapI64::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "i64") => wasm_dialect::I64ExtendI32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("trunc") {
            match (src_suffix, dst_suffix) {
                ("f32", "i32") => wasm_dialect::I32TruncF32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "i32") => wasm_dialect::I32TruncF64S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "i64") => wasm_dialect::I64TruncF32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "i64") => wasm_dialect::I64TruncF64S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "i32") => wasm_dialect::I32WrapI64::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("extend") {
            match (src_suffix, dst_suffix) {
                ("i32", "i64") => wasm_dialect::I64ExtendI32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "f64") => wasm_dialect::F64PromoteF32::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                _ => return false,
            }
        } else if name == Symbol::new("convert") {
            match (src_suffix, dst_suffix) {
                ("i32", "f32") => wasm_dialect::F32ConvertI32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i32", "f64") => wasm_dialect::F64ConvertI32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "f32") => wasm_dialect::F32ConvertI64S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("i64", "f64") => wasm_dialect::F64ConvertI64S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "i32") => wasm_dialect::I32TruncF32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "i32") => wasm_dialect::I32TruncF64S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "i64") => wasm_dialect::I64TruncF32S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "i64") => wasm_dialect::I64TruncF64S::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f32", "f64") => wasm_dialect::F64PromoteF32::operands(operand)
                    .results(dst_ty)
                    .build(ctx, loc)
                    .op_ref(),
                ("f64", "f32") => wasm_dialect::F32DemoteF64::operands(operand)
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

// ============================================================================
// Helpers
// ============================================================================

/// Get the wasm value type suffix for an arith operand or result type.
///
/// Integers up to 32 bits use `i32`, 64-bit integers use `i64`, and `core`
/// floats use their own suffix. Other types have no suffix, and patterns
/// leave their operations unconverted so the conversion boundary rejects them.
pub(crate) fn type_suffix(ctx: &IrContext, ty: TypeRef) -> Option<&'static str> {
    if let Some(width) = FloatLike::width(ctx, ty) {
        return Some(if width == 32 { "f32" } else { "f64" });
    }
    match IntegerLike::width(ctx, ty) {
        Some(64) => Some("i64"),
        Some(width) if width <= 32 => Some("i32"),
        _ if core::Nil::matches(ctx, ty) => Some("nil"),
        _ => None,
    }
}

/// Intern a core.i32 type.
pub(crate) fn intern_i32_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::types::TypeDataBuilder;
    ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
}

/// Intern a core.i64 type.
pub(crate) fn intern_i64_type(ctx: &mut IrContext) -> TypeRef {
    use trunk_ir::types::TypeDataBuilder;
    ctx.intern_type(TypeDataBuilder::new("core", "i64").build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn uncategorized_types_are_left_unconverted() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
core.module @test {
  func.func @f(%a: test.opaque, %b: test.opaque, %x: core.f64, %y: core.f64) {
    %c = arith.const {value = 0} : test.opaque
    %sum = arith.addi %a, %b : test.opaque
    %bits = arith.and %x, %y : core.f64
    func.return
  }
}
"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = print_module(&ctx, module.op());
        for op in ["arith.const", "arith.addi", "arith.and"] {
            assert!(output.contains(op), "{op} should remain:\n{output}");
        }
        assert!(!output.contains("wasm.i32"), "{output}");
    }

    #[test]
    fn narrow_integer_addition_is_left_unconverted() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
core.module @test {
  func.func @f(%a: core.i8, %b: core.i8, %c: core.i32, %d: core.i32) {
    %narrow = arith.addi %a, %b : core.i8
    %wide = arith.addi %c, %d : core.i32
    func.return
  }
}
"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains("arith.addi %0, %1 : core.i8"),
            "i8 addition should remain:\n{output}"
        );
        assert_eq!(output.matches("wasm.i32_add").count(), 1, "{output}");
    }

    #[test]
    fn lowers_i32_unsigned_less_than() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"
core.module @test {
  func.func @compare(%0: core.i32, %1: core.i32) -> core.i1 {
    %2 = arith.cmpi %0, %1 {predicate = @ult} : core.i1
    func.return %2
  }
}
"#,
        );

        lower(&mut ctx, module, TypeConverter::new());

        let output = print_module(&ctx, module.op());
        assert!(
            output.contains("wasm.i32_lt_u"),
            "i32 ult should lower to wasm.i32_lt_u:\n{output}"
        );
        assert!(
            !output.contains("arith.cmpi"),
            "i32 ult should not remain as arith.cmpi:\n{output}"
        );
    }
}
