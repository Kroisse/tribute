//! Lower arith dialect operations to wasm dialect.
//!
//! This pass converts arithmetic operations to their wasm equivalents:
//! - `arith.const` -> `wasm.{i32,i64,f32,f64}_const`
//! - `arith.{add,sub,mul,div,rem}` -> `wasm.{type}_{op}`
//! - `arith.cmp_*` -> `wasm.{type}_{cmp}`
//! - `arith.neg` -> `wasm.{f32,f64}_neg` or 0 - x for integers
//! - `arith.{and,or,xor,shl,shr,shru}` -> `wasm.i{32,64}_{op}`

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{arith, core};
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectType, Operation, Symbol, Type};

/// Lower arith dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(ArithConstPattern)
        .add_pattern(ArithBinOpPattern)
        .add_pattern(ArithCmpPattern)
        .add_pattern(ArithNegPattern)
        .add_pattern(ArithBitwisePattern)
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
        if op.dialect(db) != arith::DIALECT_NAME() || op.name(db) != arith::CONST() {
            return RewriteResult::Unchanged;
        }

        let result_ty = op.results(db).first().copied();
        let wasm_op_name = match type_suffix(db, result_ty) {
            "i32" => "wasm.i32_const",
            "i64" => "wasm.i64_const",
            "f32" => "wasm.f32_const",
            "f64" => "wasm.f64_const",
            _ => "wasm.i32_const",
        };

        let mut builder = Operation::of_name(db, op.location(db), wasm_op_name)
            .results(op.results(db).clone());

        // Copy value attribute
        if let Some(value) = op.attributes(db).get(&Symbol::new("value")) {
            builder = builder.attr("value", value.clone());
        }

        RewriteResult::Replace(builder.build())
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

        let result_ty = op.results(db).first().copied();
        let suffix = type_suffix(db, result_ty);

        let wasm_op_name = if name == arith::ADD() {
            match suffix {
                "i32" => "wasm.i32_add",
                "i64" => "wasm.i64_add",
                "f32" => "wasm.f32_add",
                "f64" => "wasm.f64_add",
                _ => "wasm.i32_add",
            }
        } else if name == arith::SUB() {
            match suffix {
                "i32" => "wasm.i32_sub",
                "i64" => "wasm.i64_sub",
                "f32" => "wasm.f32_sub",
                "f64" => "wasm.f64_sub",
                _ => "wasm.i32_sub",
            }
        } else if name == arith::MUL() {
            match suffix {
                "i32" => "wasm.i32_mul",
                "i64" => "wasm.i64_mul",
                "f32" => "wasm.f32_mul",
                "f64" => "wasm.f64_mul",
                _ => "wasm.i32_mul",
            }
        } else if name == arith::DIV() {
            match suffix {
                "i32" => "wasm.i32_div_s",
                "i64" => "wasm.i64_div_s",
                "f32" => "wasm.f32_div",
                "f64" => "wasm.f64_div",
                _ => "wasm.i32_div_s",
            }
        } else if name == arith::REM() {
            match suffix {
                "i32" => "wasm.i32_rem_s",
                "i64" => "wasm.i64_rem_s",
                _ => "wasm.i32_rem_s",
            }
        } else {
            return RewriteResult::Unchanged;
        };

        // Note: op is already remapped by PatternApplicator
        let new_op = Operation::of_name(db, op.location(db), wasm_op_name)
            .operands(op.operands(db).clone())
            .results(op.results(db).clone())
            .build();

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
        let operand_ty = op
            .operands(db)
            .first()
            .and_then(|v| value_type(db, *v));
        let suffix = type_suffix(db, operand_ty);
        let is_integer = matches!(suffix, "i32" | "i64");

        let wasm_op_name = if name == arith::CMP_EQ() {
            match suffix {
                "i32" => "wasm.i32_eq",
                "i64" => "wasm.i64_eq",
                "f32" => "wasm.f32_eq",
                "f64" => "wasm.f64_eq",
                _ => "wasm.i32_eq",
            }
        } else if name == arith::CMP_NE() {
            match suffix {
                "i32" => "wasm.i32_ne",
                "i64" => "wasm.i64_ne",
                "f32" => "wasm.f32_ne",
                "f64" => "wasm.f64_ne",
                _ => "wasm.i32_ne",
            }
        } else if name == arith::CMP_LT() {
            match (suffix, is_integer) {
                ("i32", true) => "wasm.i32_lt_s",
                ("i64", true) => "wasm.i64_lt_s",
                ("f32", false) => "wasm.f32_lt",
                ("f64", false) => "wasm.f64_lt",
                _ => "wasm.i32_lt_s",
            }
        } else if name == arith::CMP_LE() {
            match (suffix, is_integer) {
                ("i32", true) => "wasm.i32_le_s",
                ("i64", true) => "wasm.i64_le_s",
                ("f32", false) => "wasm.f32_le",
                ("f64", false) => "wasm.f64_le",
                _ => "wasm.i32_le_s",
            }
        } else if name == arith::CMP_GT() {
            match (suffix, is_integer) {
                ("i32", true) => "wasm.i32_gt_s",
                ("i64", true) => "wasm.i64_gt_s",
                ("f32", false) => "wasm.f32_gt",
                ("f64", false) => "wasm.f64_gt",
                _ => "wasm.i32_gt_s",
            }
        } else if name == arith::CMP_GE() {
            match (suffix, is_integer) {
                ("i32", true) => "wasm.i32_ge_s",
                ("i64", true) => "wasm.i64_ge_s",
                ("f32", false) => "wasm.f32_ge",
                ("f64", false) => "wasm.f64_ge",
                _ => "wasm.i32_ge_s",
            }
        } else {
            return RewriteResult::Unchanged;
        };

        // Note: op is already remapped by PatternApplicator
        let new_op = Operation::of_name(db, op.location(db), wasm_op_name)
            .operands(op.operands(db).clone())
            .results(op.results(db).clone())
            .build();

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
        if op.dialect(db) != arith::DIALECT_NAME() || op.name(db) != arith::NEG() {
            return RewriteResult::Unchanged;
        }

        let result_ty = op.results(db).first().copied();
        let suffix = type_suffix(db, result_ty);
        let location = op.location(db);
        // Note: op is already remapped by PatternApplicator
        let operand = op.operands(db).first().copied();

        match suffix {
            "f32" => {
                let new_op = Operation::of_name(db, location, "wasm.f32_neg")
                    .operands(op.operands(db).clone())
                    .results(op.results(db).clone())
                    .build();
                RewriteResult::Replace(new_op)
            }
            "f64" => {
                let new_op = Operation::of_name(db, location, "wasm.f64_neg")
                    .operands(op.operands(db).clone())
                    .results(op.results(db).clone())
                    .build();
                RewriteResult::Replace(new_op)
            }
            "i64" => {
                // For i64: 0 - x
                let i64_ty = core::I64::new(db).as_type();
                let zero = Operation::of_name(db, location, "wasm.i64_const")
                    .attr("value", Attribute::IntBits(0))
                    .results(trunk_ir::idvec![i64_ty])
                    .build();
                let zero_val = zero.result(db, 0);

                let sub_operands = match operand {
                    Some(val) => trunk_ir::idvec![zero_val, val],
                    None => trunk_ir::idvec![zero_val],
                };

                let sub = Operation::of_name(db, location, "wasm.i64_sub")
                    .operands(sub_operands)
                    .results(op.results(db).clone())
                    .build();

                RewriteResult::Expand(vec![zero, sub])
            }
            _ => {
                // Default to i32: 0 - x
                let i32_ty = core::I32::new(db).as_type();
                let zero = Operation::of_name(db, location, "wasm.i32_const")
                    .attr("value", Attribute::IntBits(0))
                    .results(trunk_ir::idvec![i32_ty])
                    .build();
                let zero_val = zero.result(db, 0);

                let sub_operands = match operand {
                    Some(val) => trunk_ir::idvec![zero_val, val],
                    None => trunk_ir::idvec![zero_val],
                };

                let sub = Operation::of_name(db, location, "wasm.i32_sub")
                    .operands(sub_operands)
                    .results(op.results(db).clone())
                    .build();

                RewriteResult::Expand(vec![zero, sub])
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

        let result_ty = op.results(db).first().copied();
        let suffix = type_suffix(db, result_ty);

        let wasm_op_name = if name == arith::AND() {
            match suffix {
                "i64" => "wasm.i64_and",
                _ => "wasm.i32_and",
            }
        } else if name == arith::OR() {
            match suffix {
                "i64" => "wasm.i64_or",
                _ => "wasm.i32_or",
            }
        } else if name == arith::XOR() {
            match suffix {
                "i64" => "wasm.i64_xor",
                _ => "wasm.i32_xor",
            }
        } else if name == arith::SHL() {
            match suffix {
                "i64" => "wasm.i64_shl",
                _ => "wasm.i32_shl",
            }
        } else if name == arith::SHR() {
            // Signed shift right
            match suffix {
                "i64" => "wasm.i64_shr_s",
                _ => "wasm.i32_shr_s",
            }
        } else if name == arith::SHRU() {
            // Unsigned shift right
            match suffix {
                "i64" => "wasm.i64_shr_u",
                _ => "wasm.i32_shr_u",
            }
        } else {
            return RewriteResult::Unchanged;
        };

        // Note: op is already remapped by PatternApplicator
        let new_op = Operation::of_name(db, op.location(db), wasm_op_name)
            .operands(op.operands(db).clone())
            .results(op.results(db).clone())
            .build();

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
            } else {
                "i32" // Default to i32
            }
        }
        None => "i32",
    }
}

/// Get the type of a value.
fn value_type<'db>(db: &'db dyn salsa::Database, value: trunk_ir::Value<'db>) -> Option<Type<'db>> {
    match value.def(db) {
        trunk_ir::ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        trunk_ir::ValueDef::BlockArg(block) => block.args(db).get(value.index(db)).copied(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, DialectOp, Location, PathId, Region, Span, idvec};

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
            location,
            idvec![],
            idvec![const1.as_operation(), const2.as_operation(), add.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_and_check(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter().map(|op| op.full_name(db)).collect()
    }

    #[salsa_test]
    fn test_arith_const_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_arith_add_module(db);
        let op_names = lower_and_check(db, module);

        // Should have wasm.i32_const, wasm.i32_const, wasm.i32_add
        assert!(op_names.iter().all(|n| n.starts_with("wasm.")));
        assert!(op_names.iter().any(|n| n == "wasm.i32_const"));
        assert!(op_names.iter().any(|n| n == "wasm.i32_add"));
    }
}
