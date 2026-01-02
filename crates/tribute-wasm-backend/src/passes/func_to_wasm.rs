//! Lower func dialect operations to wasm dialect.
//!
//! This pass converts function-level operations to wasm operations:
//! - `func.func` -> `wasm.func`
//! - `func.call` -> `wasm.call`
//! - `func.call_indirect` -> `wasm.call_indirect`
//! - `func.return` -> `wasm.return`
//! - `func.tail_call` -> `wasm.return_call`
//! - `func.unreachable` -> `wasm.unreachable`
//! - `func.constant` -> `wasm.ref_func`

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{func, wasm};
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation};

/// Lower func dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncCallIndirectPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncUnreachablePattern)
        .add_pattern(FuncConstantPattern)
        .apply(db, module)
        .module
}

/// Pattern for `func.func` -> `wasm.func`
struct FuncFuncPattern;

impl RewritePattern for FuncFuncPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_func_op) = func::Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // wasm.func has the same structure: sym_name, type attributes, body region
        // PatternApplicator will recursively process the body region
        let new_op = op.modify(db).dialect_str("wasm").name_str("func").build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.call` -> `wasm.call`
struct FuncCallPattern;

impl RewritePattern for FuncCallPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(call_op) = func::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Build wasm.call with same callee and operands
        // Note: we use modify() but need to update the callee attribute format
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("call")
            .attr("callee", Attribute::Symbol(call_op.callee(db)))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.call_indirect` -> `wasm.call_indirect`
///
/// Transforms indirect function calls for closures.
/// The callee (funcref) is the first operand, followed by arguments.
struct FuncCallIndirectPattern;

impl RewritePattern for FuncCallIndirectPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_call_indirect) = func::CallIndirect::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Build wasm.call_indirect with same operands
        // The emit phase will resolve the type_idx and table attributes
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("call_indirect")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.return` -> `wasm.return`
struct FuncReturnPattern;

impl RewritePattern for FuncReturnPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_return_op) = func::Return::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op.modify(db).dialect_str("wasm").name_str("return").build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.tail_call` -> `wasm.return_call`
struct FuncTailCallPattern;

impl RewritePattern for FuncTailCallPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(tail_call_op) = func::TailCall::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Build wasm.return_call with same callee and operands
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("return_call")
            .attr("callee", Attribute::Symbol(tail_call_op.callee(db)))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.unreachable` -> `wasm.unreachable`
struct FuncUnreachablePattern;

impl RewritePattern for FuncUnreachablePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_unreachable_op) = func::Unreachable::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("unreachable")
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `func.constant` -> `wasm.ref_func`
///
/// Transforms function constant references to WASM function references.
/// Used for closures where lifted functions need to be stored as first-class values.
struct FuncConstantPattern;

impl RewritePattern for FuncConstantPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(const_op) = func::Constant::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let func_ref = const_op.func_ref(db);

        // Transform to wasm.ref_func with the same function reference
        // The result type becomes wasm.funcref
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("ref_func")
            .attr("func_name", Attribute::Symbol(func_ref))
            .result(funcref_ty)
            .build();

        RewriteResult::Replace(new_op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;
    use trunk_ir::{Block, BlockId, DialectType, Location, PathId, Region, Span, Symbol, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_func_call_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a simple func.call
        let func_call = Operation::of_name(db, location, "func.call")
            .attr("callee", Attribute::Symbol(Symbol::new("foo")))
            .results(idvec![i32_ty])
            .build();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_call]);
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa::tracked]
    fn make_func_func_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();
        let func_ty = core::Func::new(db, idvec![], nil_ty).as_type();

        // Create func.return inside func.func body
        let func_return = Operation::of_name(db, location, "func.return").build();

        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_return],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // Create func.func
        let func_func = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("test_fn")))
            .attr("type", Attribute::Type(func_ty))
            .region(body_region)
            .build();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![func_func]);
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

    #[salsa::tracked]
    fn lower_and_check_nested(db: &dyn salsa::Database, module: Module<'_>) -> (String, String) {
        let lowered = lower(db, module);
        let body = lowered.body(db);
        let func_op = &body.blocks(db)[0].operations(db)[0];
        let func_name = func_op.full_name(db);

        // Get the operation inside the function body
        let func_body = func_op.regions(db)[0].blocks(db)[0].operations(db);
        let inner_name = func_body[0].full_name(db);

        (func_name, inner_name)
    }

    #[salsa_test]
    fn test_func_call_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_func_call_module(db);
        let op_names = lower_and_check_names(db, module);

        assert!(op_names.iter().any(|n| n == "wasm.call"));
        assert!(!op_names.iter().any(|n| n == "func.call"));
    }

    #[salsa_test]
    fn test_func_func_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_func_func_module(db);
        let (func_name, inner_name) = lower_and_check_nested(db, module);

        // func.func should become wasm.func
        assert_eq!(func_name, "wasm.func");
        // func.return inside should become wasm.return
        assert_eq!(inner_name, "wasm.return");
    }

    #[salsa::tracked]
    fn make_call_indirect_module(db: &dyn salsa::Database) -> Module<'_> {
        use trunk_ir::{Value, ValueDef};

        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a dummy callee value (function reference)
        let callee_op = Operation::of_name(db, location, "test.callee")
            .results(idvec![i32_ty])
            .build();
        let callee_val = Value::new(db, ValueDef::OpResult(callee_op), 0);

        // Create a dummy argument value
        let arg_op = Operation::of_name(db, location, "test.arg")
            .results(idvec![i32_ty])
            .build();
        let arg_val = Value::new(db, ValueDef::OpResult(arg_op), 0);

        // Create func.call_indirect
        let call_indirect = Operation::of_name(db, location, "func.call_indirect")
            .operands(idvec![callee_val, arg_val])
            .results(idvec![i32_ty])
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![callee_op, arg_op, call_indirect],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_call_indirect_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_call_indirect_module(db);
        let op_names = lower_and_check_names(db, module);

        // func.call_indirect should become wasm.call_indirect
        assert!(op_names.iter().any(|n| n == "wasm.call_indirect"));
        assert!(!op_names.iter().any(|n| n == "func.call_indirect"));
    }

    #[salsa::tracked]
    fn make_func_constant_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let func_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type()).as_type();

        // Create func.constant
        let func_constant = Operation::of_name(db, location, "func.constant")
            .attr("func_ref", Attribute::Symbol(Symbol::new("test_func")))
            .results(idvec![func_ty])
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![func_constant],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_func_constant_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_func_constant_module(db);
        let op_names = lower_and_check_names(db, module);

        // func.constant should become wasm.ref_func
        assert!(op_names.iter().any(|n| n == "wasm.ref_func"));
        assert!(!op_names.iter().any(|n| n == "func.constant"));
    }
}
