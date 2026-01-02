//! Lower closure dialect operations to wasm dialect.
//!
//! This pass converts closure operations to WasmGC struct operations:
//!
//! ## Closure Representation
//!
//! Closures are represented as WasmGC structs with two fields:
//! - Field 0: function index (i32) - index into function table
//! - Field 1: environment (anyref) - captured variables as struct
//!
//! ## Transformations
//!
//! - `closure.new(env) @func_ref` -> `wasm.struct_new` with func index and env
//! - `closure.func(closure)` -> `wasm.struct_get` field 0
//! - `closure.env(closure)` -> `wasm.struct_get` field 1
//!
//! ## Function Table
//!
//! Functions referenced by closures are collected and placed in a function table.
//! The `func.constant` operation produces the function index.

use tribute_ir::dialect::closure;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, DialectType, IdVec, Operation};

/// Lower closure dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(ClosureNewPattern)
        .add_pattern(ClosureFuncPattern)
        .add_pattern(ClosureEnvPattern)
        .apply(db, module)
        .module
}

/// Pattern for `closure.new(env) @func_ref` -> `wasm.struct_new`
///
/// Creates a closure struct with:
/// - Field 0: function reference (obtained via wasm.ref_func SSA value)
/// - Field 1: environment struct
struct ClosureNewPattern;

impl RewritePattern for ClosureNewPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(closure_new) = closure::New::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let func_ref = closure_new.func_ref(db);
        let env = closure_new.env(db);

        // Get the closure type from the result
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .expect("closure.new must have exactly one result type");

        // Create wasm.ref_func to get the function reference as an SSA value
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let ref_func = wasm::ref_func(db, location, funcref_ty, func_ref);
        let func_ref_val = ref_func.as_operation().result(db, 0);

        // Create wasm.struct_new with both the function reference and environment
        // as operands.
        //
        // Layout: (func_ref: funcref, env: anyref)
        let struct_new = Operation::of_name(db, location, "wasm.struct_new")
            .operands(IdVec::from(vec![func_ref_val, env]))
            .attr("type", Attribute::Type(result_ty))
            .attr("is_closure", Attribute::Bool(true))
            .results(IdVec::from(vec![result_ty]))
            .build();

        RewriteResult::Expand(vec![ref_func.as_operation(), struct_new])
    }
}

/// Pattern for `closure.func(closure)` -> `wasm.struct_get` field 0
///
/// Extracts the function reference from a closure struct.
struct ClosureFuncPattern;

impl RewritePattern for ClosureFuncPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_closure_func) = closure::Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Get the closure type from operand for struct.get
        let closure_ty = op.operands(db).first().and_then(|v| get_value_type(db, *v));

        // Create wasm.struct_get for field 0 (function reference)
        let mut struct_get = Operation::of_name(db, location, "wasm.struct_get")
            .operands(op.operands(db).clone())
            .attr("field_idx", Attribute::IntBits(0))
            .results(op.results(db).clone());

        if let Some(ty) = closure_ty {
            struct_get = struct_get.attr("type", Attribute::Type(ty));
        }

        RewriteResult::Replace(struct_get.build())
    }
}

/// Pattern for `closure.env(closure)` -> `wasm.struct_get` field 1
///
/// Extracts the environment struct from a closure.
struct ClosureEnvPattern;

impl RewritePattern for ClosureEnvPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_closure_env) = closure::Env::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);

        // Get the closure type from operand for struct.get
        let closure_ty = op.operands(db).first().and_then(|v| get_value_type(db, *v));

        // Create wasm.struct_get for field 1 (environment)
        let mut struct_get = Operation::of_name(db, location, "wasm.struct_get")
            .operands(op.operands(db).clone())
            .attr("field_idx", Attribute::IntBits(1))
            .results(op.results(db).clone());

        if let Some(ty) = closure_ty {
            struct_get = struct_get.attr("type", Attribute::Type(ty));
        }

        RewriteResult::Replace(struct_get.build())
    }
}

/// Get the type of a value from its defining operation's result type.
fn get_value_type<'db>(
    db: &'db dyn salsa::Database,
    value: trunk_ir::Value<'db>,
) -> Option<trunk_ir::Type<'db>> {
    match value.def(db) {
        trunk_ir::ValueDef::OpResult(op) => op.results(db).get(value.index(db)).copied(),
        trunk_ir::ValueDef::BlockArg(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;
    use trunk_ir::{
        Block, BlockId, DialectType, Location, PathId, Region, Span, Symbol, Value, ValueDef, idvec,
    };

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_closure_new_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, i32_ty).as_type();

        // Create a dummy env value
        let env_op = Operation::of_name(db, location, "test.env")
            .results(idvec![i32_ty])
            .build();
        let env_val = Value::new(db, ValueDef::OpResult(env_op), 0);

        // Create closure.new
        let closure_new = closure::new(db, location, env_val, closure_ty, Symbol::new("test_func"));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![env_op, closure_new.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn lower_and_check_names(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter().map(|op| op.full_name(db)).collect()
    }

    #[salsa_test]
    fn test_closure_new_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_closure_new_module(db);
        let op_names = lower_and_check_names(db, module);

        assert!(op_names.iter().any(|n| n == "wasm.ref_func"));
        assert!(op_names.iter().any(|n| n == "wasm.struct_new"));
        assert!(!op_names.iter().any(|n| n == "closure.new"));
    }

    #[salsa::tracked]
    fn make_closure_func_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, i32_ty).as_type();

        // Create a dummy env value
        let env_op = Operation::of_name(db, location, "test.env")
            .results(idvec![i32_ty])
            .build();
        let env_val = Value::new(db, ValueDef::OpResult(env_op), 0);

        // Create closure.new
        let closure_new = closure::new(db, location, env_val, closure_ty, Symbol::new("test_func"));
        let closure_val = Value::new(db, ValueDef::OpResult(closure_new.as_operation()), 0);

        // Create closure.func
        let closure_func = closure::func(db, location, closure_val, i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                env_op,
                closure_new.as_operation(),
                closure_func.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn make_closure_env_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, i32_ty).as_type();

        // Create a dummy env value
        let env_op = Operation::of_name(db, location, "test.env")
            .results(idvec![i32_ty])
            .build();
        let env_val = Value::new(db, ValueDef::OpResult(env_op), 0);

        // Create closure.new
        let closure_new = closure::new(db, location, env_val, closure_ty, Symbol::new("test_func"));
        let closure_val = Value::new(db, ValueDef::OpResult(closure_new.as_operation()), 0);

        // Create closure.env
        let closure_env = closure::env(db, location, closure_val, i32_ty);

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                env_op,
                closure_new.as_operation(),
                closure_env.as_operation()
            ],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    /// Helper to check operation attributes
    #[salsa::tracked]
    fn lower_and_get_field_idx(db: &dyn salsa::Database, module: Module<'_>) -> Option<i64> {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        // Find the wasm.struct_get operation
        for op in ops.iter() {
            if op.full_name(db) == "wasm.struct_get"
                && let Some(Attribute::IntBits(idx)) =
                    op.attributes(db).get(&Symbol::new("field_idx"))
            {
                return Some(*idx as i64);
            }
        }
        None
    }

    #[salsa_test]
    fn test_closure_func_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_closure_func_module(db);
        let op_names = lower_and_check_names(db, module);

        // closure.func should become wasm.struct_get
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(!op_names.iter().any(|n| n == "closure.func"));

        // Verify field_idx is 0 (function reference)
        let field_idx = lower_and_get_field_idx(db, module);
        assert_eq!(field_idx, Some(0));
    }

    #[salsa_test]
    fn test_closure_env_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_closure_env_module(db);
        let op_names = lower_and_check_names(db, module);

        // closure.env should become wasm.struct_get
        assert!(op_names.iter().any(|n| n == "wasm.struct_get"));
        assert!(!op_names.iter().any(|n| n == "closure.env"));

        // Verify field_idx is 1 (environment)
        let field_idx = lower_and_get_field_idx(db, module);
        assert_eq!(field_idx, Some(1));
    }
}
