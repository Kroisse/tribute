//! Value integrity validation for TrunkIR modules.
//!
//! TrunkIR uses Salsa-interned immutable Values. When operations are replaced
//! during IR transformations, old Values become stale — they reference operations
//! that no longer exist in the current module. Without RAUW (Replace All Uses With),
//! stale values can silently propagate through the pipeline.
//!
//! This module provides utilities to detect stale values early, before they cause
//! confusing errors in backend emission.

use std::collections::HashSet;
use std::fmt;

use crate::Symbol;
use crate::dialect::func;
use crate::ir::{Region, Value, ValueDef};

/// Describes a stale value found during validation.
pub struct StaleValueError {
    /// Name of the function containing the stale reference.
    pub function_name: String,
    /// Full name of the consuming operation (e.g., "func.call").
    pub consumer_op: String,
    /// Index of the stale operand within the consuming operation.
    pub operand_index: usize,
    /// Human-readable description of the stale value.
    pub stale_value_description: String,
}

impl fmt::Display for StaleValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stale value in @{}: operand #{} of {} references {}",
            self.function_name, self.operand_index, self.consumer_op, self.stale_value_description,
        )
    }
}

impl fmt::Debug for StaleValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Result of value integrity validation.
pub struct ValidationResult {
    pub errors: Vec<StaleValueError>,
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.errors.is_empty() {
            write!(f, "validation passed")
        } else {
            writeln!(f, "{} stale value(s) found:", self.errors.len())?;
            for err in &self.errors {
                writeln!(f, "  - {}", err)?;
            }
            Ok(())
        }
    }
}

/// Recursively collect all values defined in a region (block args + op results),
/// including those in nested sub-regions.
fn collect_defined_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    defined: &mut HashSet<Value<'db>>,
) {
    for block in region.blocks(db).iter() {
        // Block arguments are defined values
        for (i, _) in block.args(db).iter().enumerate() {
            defined.insert(Value::new(db, ValueDef::BlockArg(block.id(db)), i));
        }

        // Operation results are defined values
        for op in block.operations(db).iter() {
            for (i, _) in op.results(db).iter().enumerate() {
                defined.insert(op.result(db, i));
            }

            // Recurse into nested regions (scf.if bodies, cont.push_prompt, etc.)
            for nested_region in op.regions(db).iter() {
                collect_defined_in_region(db, nested_region, defined);
            }
        }
    }
}

/// Describe a value for diagnostic purposes.
fn describe_value<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> String {
    match value.def(db) {
        ValueDef::OpResult(op) => {
            let full_name = format!("{}.{}", op.dialect(db), op.name(db));
            // Try to get a callee/sym_name attribute for more context
            let sym = op
                .attributes(db)
                .get(&Symbol::new("sym_name"))
                .or_else(|| op.attributes(db).get(&Symbol::new("callee")));
            match sym {
                Some(crate::Attribute::Symbol(s)) => {
                    format!("result #{} of {} (@{})", value.index(db), full_name, s)
                }
                _ => format!("result #{} of {}", value.index(db), full_name),
            }
        }
        ValueDef::BlockArg(block_id) => {
            format!("block arg #{} of block {}", value.index(db), block_id.0)
        }
    }
}

/// Check that all operands in a region reference values defined within `defined_set`.
fn check_operands_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    defined_set: &HashSet<Value<'db>>,
    function_name: &str,
    errors: &mut Vec<StaleValueError>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            for (i, operand) in op.operands(db).iter().enumerate() {
                if !defined_set.contains(operand) {
                    errors.push(StaleValueError {
                        function_name: function_name.to_string(),
                        consumer_op: op.full_name(db),
                        operand_index: i,
                        stale_value_description: describe_value(db, *operand),
                    });
                }
            }

            // Also check nested regions
            for nested_region in op.regions(db).iter() {
                check_operands_in_region(db, nested_region, defined_set, function_name, errors);
            }
        }
    }
}

/// Validate value integrity for all `func.func` operations in a module.
///
/// For each function, builds the set of values defined in its region tree
/// (block args + op results, recursively), then checks that every operand
/// in the function body references a value in that set.
///
/// Values defined outside a function (e.g., in a different function) are
/// considered stale — TrunkIR functions are self-contained.
pub fn validate_value_integrity<'db>(
    db: &'db dyn salsa::Database,
    module: crate::dialect::core::Module<'db>,
) -> ValidationResult {
    let mut errors = Vec::new();
    let body = module.body(db);
    validate_functions_in_region(db, &body, &mut errors);
    ValidationResult { errors }
}

/// Recursively walk a region to find and validate all `func.func` operations,
/// including those inside nested `core.module` ops.
fn validate_functions_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    errors: &mut Vec<StaleValueError>,
) {
    use crate::DialectOp;

    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let func_name = func_op.sym_name(db).to_string();
                let func_body = func_op.body(db);

                let mut defined = HashSet::new();
                collect_defined_in_region(db, &func_body, &mut defined);
                check_operands_in_region(db, &func_body, &defined, &func_name, errors);
            }

            // Recurse into nested regions (e.g., core.module bodies)
            for nested_region in op.regions(db).iter() {
                validate_functions_in_region(db, nested_region, errors);
            }
        }
    }
}

/// Debug-only validation that panics on stale values.
///
/// Only runs under `cfg!(debug_assertions)`. Useful for inserting validation
/// checkpoints after IR transformation passes.
pub fn debug_assert_value_integrity<'db>(
    db: &'db dyn salsa::Database,
    module: crate::dialect::core::Module<'db>,
    pass_name: &str,
) {
    if !cfg!(debug_assertions) {
        return;
    }

    let result = validate_value_integrity(db, module);
    if !result.is_ok() {
        panic!(
            "Value integrity check failed after `{}`:\n{}",
            pass_name, result,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core, func, scf};
    use crate::ir::BlockBuilder;
    use crate::{DialectType, Location, PathId, Span, idvec};
    use salsa_test_macros::salsa_test;

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Build a valid module with one function: fn add() { 40 + 2 }
    #[salsa::tracked]
    fn build_valid_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let loc = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let add_fn = func::Func::build(db, loc, "add", idvec![], i32_ty, |entry| {
            let c0 = entry.op(arith::Const::i32(db, loc, 40));
            let c1 = entry.op(arith::Const::i32(db, loc, 2));
            let sum = entry.op(arith::add(db, loc, c0.result(db), c1.result(db), i32_ty));
            entry.op(func::Return::value(db, loc, sum.result(db)));
        });

        core::Module::build(db, loc, "test".into(), |top| {
            top.op(add_fn);
        })
    }

    #[salsa_test]
    fn valid_module_passes(db: &salsa::DatabaseImpl) {
        let module = build_valid_module(db);
        let result = validate_value_integrity(db, module);
        assert!(result.is_ok(), "Valid module should pass: {}", result);
    }

    /// Build a module where function B uses an op result from function A.
    #[salsa::tracked]
    fn build_stale_op_result_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let loc = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Function A: creates a constant
        let const_in_a = arith::Const::i32(db, loc, 99);
        let stale_value = const_in_a.result(db); // This value belongs to func A

        let func_a = func::Func::build(db, loc, "func_a", idvec![], i32_ty, |entry| {
            entry.op(const_in_a);
            entry.op(func::Return::value(db, loc, const_in_a.result(db)));
        });

        // Function B: uses the stale value from func A
        let func_b = func::Func::build(db, loc, "func_b", idvec![], i32_ty, |entry| {
            // This operand references a value defined in func_a — stale!
            entry.op(func::Return::value(db, loc, stale_value));
        });

        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func_a);
            top.op(func_b);
        })
    }

    #[salsa_test]
    fn stale_op_result_detected(db: &salsa::DatabaseImpl) {
        let module = build_stale_op_result_module(db);
        let result = validate_value_integrity(db, module);
        assert!(!result.is_ok(), "Should detect stale op result");
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].function_name, "func_b");
        assert!(
            result.errors[0]
                .stale_value_description
                .contains("arith.const")
        );
    }

    /// Build a module where function B uses a block arg from function A's entry block.
    #[salsa::tracked]
    fn build_stale_block_arg_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let loc = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Function A: has a parameter — build manually to capture block arg
        let func_a_entry = BlockBuilder::new(db, loc).arg(i32_ty);
        // Capture the block arg value before building
        let stale_block_arg = func_a_entry.block_arg(db, 0);
        let mut func_a_entry = func_a_entry;
        func_a_entry.op(func::Return::value(db, loc, stale_block_arg));
        let func_a_block = func_a_entry.build();

        let func_a_body = Region::new(db, loc, idvec![func_a_block]);
        let func_a_ty = core::Func::new(db, idvec![i32_ty], i32_ty);
        let func_a = func::func(
            db,
            loc,
            Symbol::new("func_a"),
            func_a_ty.as_type(),
            func_a_body,
        );

        // Function B: uses the block arg from func_a — stale!
        let func_b = func::Func::build(db, loc, "func_b", idvec![], i32_ty, |entry| {
            entry.op(func::Return::value(db, loc, stale_block_arg));
        });

        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func_a);
            top.op(func_b);
        })
    }

    #[salsa_test]
    fn stale_block_arg_detected(db: &salsa::DatabaseImpl) {
        let module = build_stale_block_arg_module(db);
        let result = validate_value_integrity(db, module);
        assert!(!result.is_ok(), "Should detect stale block arg");
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].function_name, "func_b");
        assert!(
            result.errors[0]
                .stale_value_description
                .contains("block arg")
        );
    }

    /// Build a module where an inner region references a value from its
    /// enclosing function's entry block — this is valid (lexical scoping).
    #[salsa::tracked]
    fn build_nested_region_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let loc = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        let func_op = func::Func::build(db, loc, "nested_fn", idvec![i32_ty], i32_ty, |entry| {
            let param = entry.block_arg(db, 0);
            // Use arith.cmp_eq to produce a bool (i1) value
            let zero = entry.op(arith::Const::i32(db, loc, 0));
            let cond_op = entry.op(arith::cmp_eq(db, loc, param, zero.result(db), i1_ty));

            // scf.if with branches that reference `param` from outer scope
            let then_region = {
                let mut then_block = BlockBuilder::new(db, loc);
                then_block.op(scf::r#yield(db, loc, vec![param]));
                Region::new(db, loc, idvec![then_block.build()])
            };
            let else_region = {
                let mut else_block = BlockBuilder::new(db, loc);
                let c1 = else_block.op(arith::Const::i32(db, loc, 1));
                let sum = else_block.op(arith::add(db, loc, param, c1.result(db), i32_ty));
                else_block.op(scf::r#yield(db, loc, vec![sum.result(db)]));
                Region::new(db, loc, idvec![else_block.build()])
            };
            let if_op = entry.op(scf::r#if(
                db,
                loc,
                cond_op.result(db),
                i32_ty,
                then_region,
                else_region,
            ));
            entry.op(func::Return::value(db, loc, if_op.result(db)));
        });

        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func_op);
        })
    }

    #[salsa_test]
    fn nested_region_cross_ref_valid(db: &salsa::DatabaseImpl) {
        let module = build_nested_region_module(db);
        let result = validate_value_integrity(db, module);
        assert!(
            result.is_ok(),
            "Inner region referencing outer block arg should be valid: {}",
            result,
        );
    }

    /// Build a module where function B uses a value from function A — invalid.
    #[salsa::tracked]
    fn build_cross_function_ref_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let loc = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Function A: creates a constant
        let const_a = arith::Const::i32(db, loc, 42);
        let value_from_a = const_a.result(db);

        let func_a = func::Func::build(db, loc, "func_a", idvec![], i32_ty, |entry| {
            entry.op(const_a);
            entry.op(func::Return::value(db, loc, value_from_a));
        });

        // Function B: references value_from_a — cross-function, invalid!
        let func_b = func::Func::build(db, loc, "func_b", idvec![], i32_ty, |entry| {
            let local = entry.op(arith::Const::i32(db, loc, 1));
            let _sum = entry.op(arith::add(
                db,
                loc,
                value_from_a, // stale: from func_a
                local.result(db),
                i32_ty,
            ));
            entry.op(func::Return::value(db, loc, value_from_a)); // also stale
        });

        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func_a);
            top.op(func_b);
        })
    }

    #[salsa_test]
    fn cross_function_ref_invalid(db: &salsa::DatabaseImpl) {
        let module = build_cross_function_ref_module(db);
        let result = validate_value_integrity(db, module);
        assert!(
            !result.is_ok(),
            "Cross-function value ref should be invalid"
        );
        // func_b uses value_from_a in two places: arith.add operand #0 and func.return operand #0
        assert_eq!(result.errors.len(), 2);
        for err in &result.errors {
            assert_eq!(err.function_name, "func_b");
        }
    }
}
