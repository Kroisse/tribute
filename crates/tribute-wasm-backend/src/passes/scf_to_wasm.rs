//! Lower scf dialect operations to wasm dialect.
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> removed (values stay on stack)
//! - `scf.continue` -> `wasm.br(target=1)` (branch to loop)
//! - `scf.break` -> `wasm.br(target=0)` (branch to block)

use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::{scf, wasm};
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, Block, DialectOp, DialectType, IdVec, Operation, Region, idvec};

/// Lower scf dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new()
        .add_pattern(ScfIfPattern)
        .add_pattern(ScfLoopPattern)
        .add_pattern(ScfYieldPattern)
        .add_pattern(ScfContinuePattern)
        .add_pattern(ScfBreakPattern)
        .apply(db, module)
        .module
}

/// Pattern for `scf.if` -> `wasm.if`
struct ScfIfPattern;

impl RewritePattern for ScfIfPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_if_op) = scf::If::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // wasm.if has the same structure: cond operand, result, then/else regions
        // PatternApplicator will recursively process the regions
        let new_op = Operation::of_name(db, op.location(db), "wasm.if")
            .operands(op.operands(db).clone())
            .results(op.results(db).clone())
            .regions(op.regions(db).clone())
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `scf.loop` -> `wasm.block(wasm.loop(...))`
///
/// The loop is wrapped in a block to provide a break target:
/// - `wasm.br(target=0)` branches to the block (break)
/// - `wasm.br(target=1)` branches to the loop (continue)
struct ScfLoopPattern;

impl RewritePattern for ScfLoopPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(loop_op) = scf::Loop::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let body = loop_op.body(db);

        // Create wasm.loop with the body region
        // PatternApplicator will recursively process the body
        let wasm_loop = Operation::of_name(db, location, "wasm.loop")
            .regions(idvec![body])
            .build();

        // Wrap in wasm.block for break target
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(db).as_type());

        let block_body_block = Block::new(db, location, IdVec::new(), idvec![wasm_loop]);
        let block_body = Region::new(db, location, idvec![block_body_block]);

        let wasm_block = Operation::of_name(db, location, "wasm.block")
            .results(idvec![result_ty])
            .regions(idvec![block_body])
            .build();

        RewriteResult::Replace(wasm_block)
    }
}

/// Pattern for `scf.yield` -> removed
///
/// In wasm, block results are implicit - the last value on the stack is the result.
/// We erase the yield operation; the yielded values are already produced by
/// preceding operations.
struct ScfYieldPattern;

impl RewritePattern for ScfYieldPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_yield_op) = scf::Yield::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Erase the yield - values are already on the stack
        RewriteResult::Erase {
            replacement_values: vec![],
        }
    }
}

/// Pattern for `scf.continue` -> `wasm.br(target=1)`
///
/// Branches to the enclosing wasm.loop (depth 1, since loop is inside block).
struct ScfContinuePattern;

impl RewritePattern for ScfContinuePattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_continue_op) = scf::Continue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Branch to loop (depth 1: block=0, loop=1)
        let br_op = Operation::of_name(db, op.location(db), "wasm.br")
            .attr("target", Attribute::IntBits(1))
            .build();

        RewriteResult::Replace(br_op)
    }
}

/// Pattern for `scf.break` -> `wasm.br(target=0)`
///
/// Branches to the enclosing wasm.block (depth 0) with a result value.
struct ScfBreakPattern;

impl RewritePattern for ScfBreakPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> RewriteResult<'db> {
        let Ok(_break_op) = scf::Break::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Branch to block (depth 0) with result value
        let br_op = Operation::of_name(db, op.location(db), "wasm.br")
            .attr("target", Attribute::IntBits(0))
            .operands(op.operands(db).clone())
            .build();

        RewriteResult::Replace(br_op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Location, PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_scf_if_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a simple scf.if with empty then/else regions
        let then_block = Block::new(db, location, idvec![], idvec![]);
        let then_region = Region::new(db, location, idvec![then_block]);

        let else_block = Block::new(db, location, idvec![], idvec![]);
        let else_region = Region::new(db, location, idvec![else_block]);

        // Create a dummy condition value
        let cond_const = wasm::i32_const(db, location, i32_ty, Attribute::IntBits(1));

        let scf_if = Operation::of_name(db, location, "scf.if")
            .operands(idvec![cond_const.result(db)])
            .results(idvec![i32_ty])
            .regions(idvec![then_region, else_region])
            .build();

        let block = Block::new(db, location, idvec![], idvec![cond_const.operation(), scf_if]);
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
    fn test_scf_if_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_scf_if_module(db);
        let op_names = lower_and_check(db, module);

        // scf.if should become wasm.if
        assert!(op_names.iter().any(|n| n == "wasm.if"));
        assert!(!op_names.iter().any(|n| n == "scf.if"));
    }
}
