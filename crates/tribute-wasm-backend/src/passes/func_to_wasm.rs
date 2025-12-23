//! Lower func dialect operations to wasm dialect.
//!
//! This pass converts function-level operations to wasm operations:
//! - `func.func` -> `wasm.func`
//! - `func.call` -> `wasm.call`
//! - `func.return` -> `wasm.return`
//! - `func.tail_call` -> `wasm.return_call`
//! - `func.unreachable` -> `wasm.unreachable`
//!
//! Note: `func.call_indirect` and `func.constant` are preserved for now
//! (closure support to be added later).

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::func;
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, Operation};

/// Lower func dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(FuncFuncPattern)
        .add_pattern(FuncCallPattern)
        .add_pattern(FuncReturnPattern)
        .add_pattern(FuncTailCallPattern)
        .add_pattern(FuncUnreachablePattern)
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
        if op.dialect(db) != func::DIALECT_NAME() || op.name(db) != func::FUNC() {
            return RewriteResult::Unchanged;
        }

        // wasm.func has the same structure: sym_name, type attributes, body region
        // PatternApplicator will recursively process the body region
        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("func")
            .build();

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
        if op.dialect(db) != func::DIALECT_NAME() || op.name(db) != func::CALL() {
            return RewriteResult::Unchanged;
        }

        let op_call = func::Call::from_operation(db, *op).expect("already matched func.call");
        let location = op.location(db);

        // Build wasm.call with same callee and operands
        let new_op = Operation::of_name(db, location, "wasm.call")
            .operands(op.operands(db).clone())
            .results(op.results(db).clone())
            .attr("callee", Attribute::QualifiedName(op_call.callee(db)))
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
        if op.dialect(db) != func::DIALECT_NAME() || op.name(db) != func::RETURN() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("return")
            .build();

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
        if op.dialect(db) != func::DIALECT_NAME() || op.name(db) != func::TAIL_CALL() {
            return RewriteResult::Unchanged;
        }

        let op_tail = func::TailCall::from_operation(db, *op).expect("already matched func.tail_call");
        let location = op.location(db);

        // Build wasm.return_call with same callee and operands
        let new_op = Operation::of_name(db, location, "wasm.return_call")
            .operands(op.operands(db).clone())
            .attr("callee", Attribute::QualifiedName(op_tail.callee(db)))
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
        if op.dialect(db) != func::DIALECT_NAME() || op.name(db) != func::UNREACHABLE() {
            return RewriteResult::Unchanged;
        }

        let new_op = op
            .modify(db)
            .dialect_str("wasm")
            .name_str("unreachable")
            .build();

        RewriteResult::Replace(new_op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;
    use trunk_ir::{Block, DialectType, Location, PathId, QualifiedName, Region, Span, Symbol, idvec};

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
            .attr(
                "callee",
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("foo"))),
            )
            .results(idvec![i32_ty])
            .build();

        let block = Block::new(db, location, idvec![], idvec![func_call]);
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

        let body_block = Block::new(db, location, idvec![], idvec![func_return]);
        let body_region = Region::new(db, location, idvec![body_block]);

        // Create func.func
        let func_func = Operation::of_name(db, location, "func.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("test_fn")))
            .attr("type", Attribute::Type(func_ty))
            .region(body_region)
            .build();

        let block = Block::new(db, location, idvec![], idvec![func_func]);
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
}
