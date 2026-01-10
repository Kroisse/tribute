//! Lower scf dialect operations to wasm dialect.
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> `wasm.yield` (tracks region result value)
//! - `scf.continue` -> `wasm.br(target=1)` (branch to loop)
//! - `scf.break` -> `wasm.br(target=0)` (branch to block)

use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::scf;
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, idvec,
};

use crate::type_converter::wasm_type_converter;

/// Lower scf dialect to wasm dialect.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    PatternApplicator::new(wasm_type_converter())
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
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(scf_if_op) = scf::If::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // wasm.if has the same structure: cond operand, result, then/else regions
        // PatternApplicator will recursively process the regions
        let cond = scf_if_op.cond(db);
        let then_region = scf_if_op.then(db);
        let else_region = scf_if_op.r#else(db);
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(db).as_type());

        let new_op = wasm::r#if(
            db,
            op.location(db),
            cond,
            result_ty,
            then_region,
            else_region,
        );

        RewriteResult::Replace(new_op.as_operation())
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
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(loop_op) = scf::Loop::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let body = loop_op.body(db);

        // Get result type first
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(db).as_type());

        // Create wasm.loop with the body region
        // PatternApplicator will recursively process the body
        let wasm_loop =
            wasm::r#loop(db, location, result_ty, trunk_ir::Symbol::new(""), body).as_operation();

        let block_body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            idvec![wasm_loop],
        );
        let block_body = Region::new(db, location, idvec![block_body_block]);

        let wasm_block = wasm::block(
            db,
            location,
            result_ty,
            trunk_ir::Symbol::new(""),
            block_body,
        )
        .as_operation();

        RewriteResult::Replace(wasm_block)
    }
}

/// Pattern for `scf.yield` -> `wasm.yield`
///
/// In wasm, block results are implicit - the last value on the stack is the result.
/// We convert scf.yield to wasm.yield to track which value should be the region's
/// result. This is especially important for handler dispatch where the result value
/// may be defined outside the region (e.g., the scrutinee in `{ result } -> result`).
///
/// At emit time, wasm.yield is handled specially: its operand is emitted as a
/// local.get, and the wasm.yield itself produces no Wasm instruction.
struct ScfYieldPattern;

impl RewritePattern for ScfYieldPattern {
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(yield_op) = scf::Yield::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Convert to wasm.yield which tracks the result value
        // This is needed because the yielded value may be defined outside the region
        let Some(value) = yield_op.values(db).first().copied() else {
            // No value to yield - just erase
            return RewriteResult::Erase {
                replacement_values: vec![],
            };
        };

        let new_op = wasm::r#yield(db, op.location(db), value);
        RewriteResult::Replace(new_op.as_operation())
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
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_continue_op) = scf::Continue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Branch to loop (depth 1: block=0, loop=1)
        // Note: wasm::br typed helper expects Symbol (label name), but we use integer depths.
        // Use Operation::of_name for depth-based branching.
        let br_op = Operation::of(db, op.location(db), wasm::DIALECT_NAME(), wasm::BR())
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
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_break_op) = scf::Break::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Branch to block (depth 0) with result value
        // Note: The wasm.br typed helper doesn't support operands, but we need to
        // pass the break value. Use Operation::of_name for this case.
        let br_op = Operation::of(db, op.location(db), wasm::DIALECT_NAME(), wasm::BR())
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
    use trunk_ir::dialect::wasm;
    use trunk_ir::{BlockId, Location, PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_scf_if_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a simple scf.if with empty then/else regions
        let then_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![]);
        let then_region = Region::new(db, location, idvec![then_block]);

        let else_block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![]);
        let else_region = Region::new(db, location, idvec![else_block]);

        // Create a dummy condition value
        let cond_const = wasm::i32_const(db, location, i32_ty, 1);

        let scf_if = Operation::of(db, location, scf::DIALECT_NAME(), scf::IF())
            .operands(idvec![cond_const.result(db)])
            .results(idvec![i32_ty])
            .regions(idvec![then_region, else_region])
            .build();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![cond_const.operation(), scf_if],
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
    fn test_scf_if_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_scf_if_module(db);
        let op_names = lower_and_check(db, module);

        // scf.if should become wasm.if
        assert!(op_names.iter().any(|n| n == "wasm.if"));
        assert!(!op_names.iter().any(|n| n == "scf.if"));
    }
}
