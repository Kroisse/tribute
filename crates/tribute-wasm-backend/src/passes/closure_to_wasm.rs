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
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, Operation};

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
/// - Field 0: function reference (preserved as func.constant for later resolution)
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
            .expect("closure.new should have a result type");

        // Create wasm.struct_new with the function reference as an attribute
        // and the environment as an operand.
        //
        // The func_ref is stored as an attribute because it's resolved to a
        // table index at emit time, not as an SSA value.
        //
        // Layout: (func_ref: funcref, env: anyref)
        let struct_new = Operation::of_name(db, location, "wasm.struct_new")
            .operands(trunk_ir::idvec![env])
            .attr("func_ref", Attribute::QualifiedName(func_ref))
            .attr("type", Attribute::Type(result_ty))
            .attr("is_closure", Attribute::Bool(true))
            .results(trunk_ir::idvec![result_ty])
            .build();

        RewriteResult::Replace(struct_new)
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
        Block, BlockId, DialectType, Location, PathId, QualifiedName, Region, Span, Symbol, Value,
        ValueDef, idvec,
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
        let closure_new = closure::new(
            db,
            location,
            env_val,
            closure_ty,
            QualifiedName::simple(Symbol::new("test_func")),
        );

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

        assert!(op_names.iter().any(|n| n == "wasm.struct_new"));
        assert!(!op_names.iter().any(|n| n == "closure.new"));
    }
}
