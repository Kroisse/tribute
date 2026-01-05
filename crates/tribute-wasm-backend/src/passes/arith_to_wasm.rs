//! Lower arith dialect operations to wasm dialect.
//!
//! This pass converts arithmetic operations to their wasm equivalents:
//! - `arith.const` -> `wasm.{i32,i64,f32,f64}_const`
//! - `arith.{add,sub,mul,div,rem}` -> `wasm.{type}_{op}`
//! - `arith.cmp_*` -> `wasm.{type}_{cmp}`
//! - `arith.neg` -> `wasm.{f32,f64}_neg` or 0 - x for integers
//! - `arith.{and,or,xor,shl,shr,shru}` -> `wasm.i{32,64}_{op}`
//! - `arith.{cast,trunc,extend,convert}` -> appropriate wasm conversion ops

use tracing::warn;

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{arith, core, wasm};
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol, Type};

/// Lower arith dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(ArithConstPattern)
        .add_pattern(ArithBinOpPattern)
        .add_pattern(ArithCmpPattern)
        .add_pattern(ArithNegPattern)
        .add_pattern(ArithBitwisePattern)
        .add_pattern(ArithConversionPattern)
        .apply(db, module)
        .module
}

/// Pattern for `arith.const` -> `wasm.{type}_const`
struct ArithConstPattern;

impl RewritePattern for ArithConstPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(const_op) = arith::Const::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let result_ty = op.results(db).first().copied();

        // Handle nil type constants specially
        let type_name = type_suffix(db, result_ty);
        if type_name == "nil" {
            // Nil constants have no runtime representation, but we need to preserve
            // SSA form for operations that reference them. Use wasm.nop which has
            // no runtime effect but maintains the value reference.
            let result_ty = result_ty.unwrap();
            let nop = wasm::nop(db, op.location(db), result_ty);
            return RewriteResult::Replace(nop.as_operation());
        }

        let location = op.location(db);
        let result_ty = result_ty.unwrap();
        let value = const_op.value(db).clone();

        let new_op = match type_name {
            "i32" => {
                if let Attribute::IntBits(v) = value {
                    wasm::i32_const(db, location, result_ty, v as i32).as_operation()
                } else {
                    wasm::i32_const(db, location, result_ty, 0).as_operation()
                }
            }
            "i64" => {
                if let Attribute::IntBits(v) = value {
                    wasm::i64_const(db, location, result_ty, v as i64).as_operation()
                } else {
                    wasm::i64_const(db, location, result_ty, 0).as_operation()
                }
            }
            "f32" => {
                if let Attribute::FloatBits(v) = value {
                    wasm::f32_const(db, location, result_ty, f32::from_bits(v as u32))
                        .as_operation()
                } else {
                    wasm::f32_const(db, location, result_ty, 0.0).as_operation()
                }
            }
            "f64" => {
                if let Attribute::FloatBits(v) = value {
                    wasm::f64_const(db, location, result_ty, f64::from_bits(v)).as_operation()
                } else {
                    wasm::f64_const(db, location, result_ty, 0.0).as_operation()
                }
            }
            _ => {
                if let Attribute::IntBits(v) = value {
                    wasm::i32_const(db, location, result_ty, v as i32).as_operation()
                } else {
                    wasm::i32_const(db, location, result_ty, 0).as_operation()
                }
            }
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.{add,sub,mul,div,rem}` -> `wasm.{type}_{op}`
struct ArithBinOpPattern;

impl RewritePattern for ArithBinOpPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_binop = name == arith::ADD()
            || name == arith::SUB()
            || name == arith::MUL()
            || name == arith::DIV()
            || name == arith::REM();

        if !is_binop {
            return RewriteResult::Unchanged;
        }

        let result_ty =
            op.results(db).first().copied().unwrap_or_else(|| {
                panic!("arith binop missing result type at {:?}", op.location(db))
            });
        let suffix = type_suffix(db, Some(result_ty));
        let location = op.location(db);
        let operands = op.operands(db);
        let lhs = operands[0];
        let rhs = operands[1];

        let new_op = if name == arith::ADD() {
            match suffix {
                "i32" => wasm::i32_add(db, location, lhs, rhs, result_ty).as_operation(),
                "i64" => wasm::i64_add(db, location, lhs, rhs, result_ty).as_operation(),
                "f32" => wasm::f32_add(db, location, lhs, rhs, result_ty).as_operation(),
                "f64" => wasm::f64_add(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_add(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::SUB() {
            match suffix {
                "i32" => wasm::i32_sub(db, location, lhs, rhs, result_ty).as_operation(),
                "i64" => wasm::i64_sub(db, location, lhs, rhs, result_ty).as_operation(),
                "f32" => wasm::f32_sub(db, location, lhs, rhs, result_ty).as_operation(),
                "f64" => wasm::f64_sub(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_sub(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::MUL() {
            match suffix {
                "i32" => wasm::i32_mul(db, location, lhs, rhs, result_ty).as_operation(),
                "i64" => wasm::i64_mul(db, location, lhs, rhs, result_ty).as_operation(),
                "f32" => wasm::f32_mul(db, location, lhs, rhs, result_ty).as_operation(),
                "f64" => wasm::f64_mul(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_mul(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::DIV() {
            match suffix {
                "i32" => wasm::i32_div_s(db, location, lhs, rhs, result_ty).as_operation(),
                "i64" => wasm::i64_div_s(db, location, lhs, rhs, result_ty).as_operation(),
                "f32" => wasm::f32_div(db, location, lhs, rhs, result_ty).as_operation(),
                "f64" => wasm::f64_div(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_div_s(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::REM() {
            match suffix {
                "i32" => wasm::i32_rem_s(db, location, lhs, rhs, result_ty).as_operation(),
                "i64" => wasm::i64_rem_s(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_rem_s(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else {
            return RewriteResult::Unchanged;
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.cmp_*` -> `wasm.{type}_{cmp}`
struct ArithCmpPattern;

impl RewritePattern for ArithCmpPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_cmp = name == arith::CMP_EQ()
            || name == arith::CMP_NE()
            || name == arith::CMP_LT()
            || name == arith::CMP_LE()
            || name == arith::CMP_GT()
            || name == arith::CMP_GE();

        if !is_cmp {
            return RewriteResult::Unchanged;
        }

        // Get operand type to determine wasm type
        let operand_ty = op.operands(db).first().and_then(|v| value_type(db, *v));
        let suffix = type_suffix(db, operand_ty);
        let is_integer = matches!(suffix, "i32" | "i64");

        let result_ty =
            op.results(db).first().copied().unwrap_or_else(|| {
                panic!("arith cmp missing result type at {:?}", op.location(db))
            });
        let location = op.location(db);
        let operands = op.operands(db);
        let lhs = operands[0];
        let rhs = operands[1];

        let new_op = if name == arith::CMP_EQ() {
            match suffix {
                "i32" => wasm::i32_eq(db, location, lhs, rhs, result_ty).as_operation(),
                "i64" => wasm::i64_eq(db, location, lhs, rhs, result_ty).as_operation(),
                "f32" => wasm::f32_eq(db, location, lhs, rhs, result_ty).as_operation(),
                "f64" => wasm::f64_eq(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_eq(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::CMP_NE() {
            match suffix {
                "i32" => wasm::i32_ne(db, location, lhs, rhs, result_ty).as_operation(),
                "i64" => wasm::i64_ne(db, location, lhs, rhs, result_ty).as_operation(),
                "f32" => wasm::f32_ne(db, location, lhs, rhs, result_ty).as_operation(),
                "f64" => wasm::f64_ne(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_ne(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::CMP_LT() {
            match (suffix, is_integer) {
                ("i32", true) => wasm::i32_lt_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("i64", true) => wasm::i64_lt_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("f32", false) => wasm::f32_lt(db, location, lhs, rhs, result_ty).as_operation(),
                ("f64", false) => wasm::f64_lt(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_lt_s(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::CMP_LE() {
            match (suffix, is_integer) {
                ("i32", true) => wasm::i32_le_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("i64", true) => wasm::i64_le_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("f32", false) => wasm::f32_le(db, location, lhs, rhs, result_ty).as_operation(),
                ("f64", false) => wasm::f64_le(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_le_s(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::CMP_GT() {
            match (suffix, is_integer) {
                ("i32", true) => wasm::i32_gt_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("i64", true) => wasm::i64_gt_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("f32", false) => wasm::f32_gt(db, location, lhs, rhs, result_ty).as_operation(),
                ("f64", false) => wasm::f64_gt(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_gt_s(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::CMP_GE() {
            match (suffix, is_integer) {
                ("i32", true) => wasm::i32_ge_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("i64", true) => wasm::i64_ge_s(db, location, lhs, rhs, result_ty).as_operation(),
                ("f32", false) => wasm::f32_ge(db, location, lhs, rhs, result_ty).as_operation(),
                ("f64", false) => wasm::f64_ge(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_ge_s(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else {
            return RewriteResult::Unchanged;
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.neg` -> `wasm.{f32,f64}_neg` or 0 - x for integers
struct ArithNegPattern;

impl RewritePattern for ArithNegPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(neg_op) = arith::Neg::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let result_ty = op.results(db).first().copied();
        let suffix = type_suffix(db, result_ty);
        let location = op.location(db);
        let operand = neg_op.operand(db);

        match suffix {
            "f32" => {
                let f32_ty = result_ty.unwrap_or_else(|| core::F32::new(db).as_type());
                let new_op = wasm::f32_neg(db, location, operand, f32_ty);
                RewriteResult::Replace(new_op.operation())
            }
            "f64" => {
                let f64_ty = result_ty.unwrap_or_else(|| core::F64::new(db).as_type());
                let new_op = wasm::f64_neg(db, location, operand, f64_ty);
                RewriteResult::Replace(new_op.operation())
            }
            "i64" => {
                // For i64: 0 - x
                let i64_ty = core::I64::new(db).as_type();
                let zero = wasm::i64_const(db, location, i64_ty, 0);
                let zero_val = zero.result(db);
                let sub = wasm::i64_sub(db, location, zero_val, operand, i64_ty);
                RewriteResult::Expand(vec![zero.operation(), sub.operation()])
            }
            _ => {
                // Default to i32: 0 - x
                let i32_ty = core::I32::new(db).as_type();
                let zero = wasm::i32_const(db, location, i32_ty, 0);
                let zero_val = zero.result(db);
                let sub = wasm::i32_sub(db, location, zero_val, operand, i32_ty);
                RewriteResult::Expand(vec![zero.operation(), sub.operation()])
            }
        }
    }
}

/// Pattern for `arith.{and,or,xor,shl,shr,shru}` -> `wasm.i{32,64}_{op}`
struct ArithBitwisePattern;

impl RewritePattern for ArithBitwisePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_bitwise = name == arith::AND()
            || name == arith::OR()
            || name == arith::XOR()
            || name == arith::SHL()
            || name == arith::SHR()
            || name == arith::SHRU();

        if !is_bitwise {
            return RewriteResult::Unchanged;
        }

        let result_ty = op.results(db).first().copied().unwrap_or_else(|| {
            panic!(
                "arith bitwise op missing result type at {:?}",
                op.location(db)
            )
        });
        let suffix = type_suffix(db, Some(result_ty));
        let location = op.location(db);
        let operands = op.operands(db);
        let lhs = operands[0];
        let rhs = operands[1];

        let new_op = if name == arith::AND() {
            match suffix {
                "i64" => wasm::i64_and(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_and(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::OR() {
            match suffix {
                "i64" => wasm::i64_or(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_or(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::XOR() {
            match suffix {
                "i64" => wasm::i64_xor(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_xor(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::SHL() {
            match suffix {
                "i64" => wasm::i64_shl(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_shl(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::SHR() {
            // Signed shift right
            match suffix {
                "i64" => wasm::i64_shr_s(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_shr_s(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else if name == arith::SHRU() {
            // Unsigned shift right
            match suffix {
                "i64" => wasm::i64_shr_u(db, location, lhs, rhs, result_ty).as_operation(),
                _ => wasm::i32_shr_u(db, location, lhs, rhs, result_ty).as_operation(),
            }
        } else {
            return RewriteResult::Unchanged;
        };

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `arith.{cast,trunc,extend,convert}` -> wasm conversion ops
struct ArithConversionPattern;

impl RewritePattern for ArithConversionPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != arith::DIALECT_NAME() {
            return RewriteResult::Unchanged;
        }

        let name = op.name(db);
        let is_conv = name == arith::CAST()
            || name == arith::TRUNC()
            || name == arith::EXTEND()
            || name == arith::CONVERT();

        if !is_conv {
            return RewriteResult::Unchanged;
        }

        // Get source type from operand
        let src_ty = op.operands(db).first().and_then(|v| value_type(db, *v));
        let src_suffix = type_suffix(db, src_ty);

        // Get destination type from result
        let dst_ty = op.results(db).first().copied().unwrap_or_else(|| {
            panic!(
                "arith conversion missing result type at {:?}",
                op.location(db)
            )
        });
        let dst_suffix = type_suffix(db, Some(dst_ty));

        let location = op.location(db);
        let operand = op.operands(db)[0];

        let new_op = if name == arith::CAST() {
            // cast: integer sign extension/truncation (i32 <-> i64)
            match (src_suffix, dst_suffix) {
                ("i64", "i32") => wasm::i32_wrap_i64(db, location, operand, dst_ty).as_operation(),
                ("i32", "i64") => {
                    wasm::i64_extend_i32_s(db, location, operand, dst_ty).as_operation()
                }
                _ => return RewriteResult::Unchanged, // unsupported cast
            }
        } else if name == arith::TRUNC() {
            // trunc: float -> int truncation (signed by default)
            match (src_suffix, dst_suffix) {
                ("f32", "i32") => {
                    wasm::i32_trunc_f32_s(db, location, operand, dst_ty).as_operation()
                }
                ("f64", "i32") => {
                    wasm::i32_trunc_f64_s(db, location, operand, dst_ty).as_operation()
                }
                ("f32", "i64") => {
                    wasm::i64_trunc_f32_s(db, location, operand, dst_ty).as_operation()
                }
                ("f64", "i64") => {
                    wasm::i64_trunc_f64_s(db, location, operand, dst_ty).as_operation()
                }
                ("i64", "i32") => wasm::i32_wrap_i64(db, location, operand, dst_ty).as_operation(), // integer truncation
                _ => return RewriteResult::Unchanged,
            }
        } else if name == arith::EXTEND() {
            // extend: smaller -> larger type
            match (src_suffix, dst_suffix) {
                ("i32", "i64") => {
                    wasm::i64_extend_i32_s(db, location, operand, dst_ty).as_operation()
                }
                ("f32", "f64") => {
                    wasm::f64_promote_f32(db, location, operand, dst_ty).as_operation()
                }
                _ => return RewriteResult::Unchanged,
            }
        } else if name == arith::CONVERT() {
            // convert: int <-> float conversion
            match (src_suffix, dst_suffix) {
                // int to float (signed by default)
                ("i32", "f32") => {
                    wasm::f32_convert_i32_s(db, location, operand, dst_ty).as_operation()
                }
                ("i32", "f64") => {
                    wasm::f64_convert_i32_s(db, location, operand, dst_ty).as_operation()
                }
                ("i64", "f32") => {
                    wasm::f32_convert_i64_s(db, location, operand, dst_ty).as_operation()
                }
                ("i64", "f64") => {
                    wasm::f64_convert_i64_s(db, location, operand, dst_ty).as_operation()
                }
                // float to int (signed by default)
                ("f32", "i32") => {
                    wasm::i32_trunc_f32_s(db, location, operand, dst_ty).as_operation()
                }
                ("f64", "i32") => {
                    wasm::i32_trunc_f64_s(db, location, operand, dst_ty).as_operation()
                }
                ("f32", "i64") => {
                    wasm::i64_trunc_f32_s(db, location, operand, dst_ty).as_operation()
                }
                ("f64", "i64") => {
                    wasm::i64_trunc_f64_s(db, location, operand, dst_ty).as_operation()
                }
                // float to float
                ("f32", "f64") => {
                    wasm::f64_promote_f32(db, location, operand, dst_ty).as_operation()
                }
                ("f64", "f32") => {
                    wasm::f32_demote_f64(db, location, operand, dst_ty).as_operation()
                }
                _ => return RewriteResult::Unchanged,
            }
        } else {
            return RewriteResult::Unchanged;
        };

        RewriteResult::Replace(new_op)
    }
}

/// Get the wasm type suffix from a type.
fn type_suffix<'db>(db: &'db dyn salsa::Database, ty: Option<Type<'db>>) -> &'static str {
    match ty {
        Some(t) => {
            let name = t.name(db);
            if name == Symbol::new("i32") {
                "i32"
            } else if name == Symbol::new("i64") {
                "i64"
            } else if name == Symbol::new("f32") {
                "f32"
            } else if name == Symbol::new("f64") {
                "f64"
            } else if name == Symbol::new("i1") {
                // Boolean type - represented as i32 in wasm
                "i32"
            } else if name == Symbol::new("int") || name == Symbol::new("nat") {
                // Arbitrary precision Int/Nat - lowered to i64 for Phase 1
                "i64"
            } else if name == Symbol::new("nil") {
                // Nil/void type - no runtime value
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

/// Get the type of a value.
fn value_type<'db>(db: &'db dyn salsa::Database, value: trunk_ir::Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        trunk_ir::ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        // BlockArg type lookup requires block context; callers handle None
        trunk_ir::ValueDef::BlockArg(_block_id) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockId, DialectOp, Location, PathId, Region, Span, idvec};

    /// Format module operations for snapshot testing
    fn format_module_ops(db: &dyn salsa::Database, module: &Module<'_>) -> String {
        let body = module.body(db);
        let ops = &body.blocks(db)[0].operations(db);
        ops.iter()
            .map(|op| format_op(db, op))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format a single operation for snapshot testing
    fn format_op(db: &dyn salsa::Database, op: &trunk_ir::Operation<'_>) -> String {
        let name = op.full_name(db);
        let operands = op.operands(db);
        let results = op.results(db);
        let attrs = op.attributes(db);

        let mut parts = vec![name];

        // Add key attributes
        for (key, attr) in attrs.iter() {
            if *key == "value" {
                parts.push(format!("value={}", format_attr(attr)));
            }
        }

        // Add operand count
        if !operands.is_empty() {
            parts.push(format!("operands={}", operands.len()));
        }

        // Add result types
        if !results.is_empty() {
            let result_types: Vec<_> = results.iter().map(|t| t.name(db).to_string()).collect();
            parts.push(format!("-> {}", result_types.join(", ")));
        }

        parts.join(" ")
    }

    fn format_attr(attr: &trunk_ir::Attribute<'_>) -> String {
        match attr {
            trunk_ir::Attribute::IntBits(v) => format!("{}", *v as i64),
            trunk_ir::Attribute::FloatBits(v) => format!("{}", f64::from_bits(*v)),
            _ => "...".to_string(),
        }
    }

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_arith_add_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create arith.const operations for operands
        let const1 = arith::Const::i32(db, location, 1);
        let const2 = arith::Const::i32(db, location, 2);

        // Create arith.add
        let add = trunk_ir::dialect::arith::add(
            db,
            location,
            const1.result(db),
            const2.result(db),
            i32_ty,
        );

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                const1.as_operation(),
                const2.as_operation(),
                add.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn format_lowered_module<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> String {
        let lowered = lower(db, module);
        format_module_ops(db, &lowered)
    }

    #[salsa_test]
    fn test_arith_const_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_arith_add_module(db);
        let formatted = format_lowered_module(db, module);

        // Snapshot test for visual verification
        assert_snapshot!(formatted, @r###"
        wasm.i32_const value=1 -> i32
        wasm.i32_const value=2 -> i32
        wasm.i32_add operands=2 -> i32
        "###);
    }

    #[salsa::tracked]
    fn make_convert_i32_to_f64_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let f64_ty = core::F64::new(db).as_type();

        // Create arith.const for i32 operand
        let int_const = arith::Const::i32(db, location, 42);

        // Create arith.convert to convert i32 -> f64
        let convert = arith::convert(db, location, int_const.result(db), f64_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![int_const.as_operation(), convert.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_convert_i32_to_f64(db: &salsa::DatabaseImpl) {
        let module = make_convert_i32_to_f64_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r###"
        wasm.i32_const value=42 -> i32
        wasm.f64_convert_i32_s operands=1 -> f64
        "###);
    }

    #[salsa::tracked]
    fn make_extend_i32_to_i64_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // Create arith.const for i32 operand
        let int_const = arith::Const::i32(db, location, 100);

        // Create arith.extend to extend i32 -> i64
        let extend = arith::extend(db, location, int_const.result(db), i64_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![int_const.as_operation(), extend.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_extend_i32_to_i64(db: &salsa::DatabaseImpl) {
        let module = make_extend_i32_to_i64_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r###"
        wasm.i32_const value=100 -> i32
        wasm.i64_extend_i32_s operands=1 -> i64
        "###);
    }

    #[salsa::tracked]
    fn make_trunc_f64_to_i32_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create arith.const for f64 operand
        let float_const = arith::Const::f64(db, location, 3.5);

        // Create arith.trunc to truncate f64 -> i32
        let trunc = arith::trunc(db, location, float_const.result(db), i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![float_const.as_operation(), trunc.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_arith_trunc_f64_to_i32(db: &salsa::DatabaseImpl) {
        let module = make_trunc_f64_to_i32_module(db);
        let formatted = format_lowered_module(db, module);

        assert_snapshot!(formatted, @r###"
        wasm.f64_const value=3.5 -> f64
        wasm.i32_trunc_f64_s operands=1 -> i32
        "###);
    }
}
