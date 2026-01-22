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
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, idvec};

/// Lower scf dialect to wasm dialect.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    type_converter: TypeConverter,
) -> Module<'db> {
    // No specific conversion target - scf lowering is a dialect transformation
    let target = ConversionTarget::new();
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(ScfIfPattern)
        .add_pattern(ScfLoopPattern)
        .add_pattern(ScfYieldPattern)
        .add_pattern(ScfContinuePattern)
        .add_pattern(ScfBreakPattern);
    applicator.apply_partial(db, module, target).module
}

/// Pattern for `scf.if` -> `wasm.if`
struct ScfIfPattern;

impl<'db> RewritePattern<'db> for ScfIfPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(scf_if_op) = scf::If::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // wasm.if has the same structure: cond operand, result, then/else regions
        // PatternApplicator will recursively process the regions
        // Use adaptor to get remapped cond operand (important when cond is a result of another converted op)
        let cond = adaptor.operand(0).unwrap_or_else(|| scf_if_op.cond(db));
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

impl<'db> RewritePattern<'db> for ScfLoopPattern {
    fn match_and_rewrite(
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
        let wasm_loop = wasm::r#loop(db, location, result_ty, body).as_operation();

        let block_body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            idvec![wasm_loop],
        );
        let block_body = Region::new(db, location, idvec![block_body_block]);

        let wasm_block = wasm::block(db, location, result_ty, block_body).as_operation();

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

impl<'db> RewritePattern<'db> for ScfYieldPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if !scf::Yield::matches(db, *op) {
            return RewriteResult::Unchanged;
        };

        // Convert to wasm.yield which tracks the result value
        // This is needed because the yielded value may be defined outside the region
        // Use adaptor to get remapped operand
        let Some(value) = adaptor.operand(0) else {
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

impl<'db> RewritePattern<'db> for ScfContinuePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_continue_op) = scf::Continue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Branch to loop (depth 1: block=0, loop=1)
        let br_op = wasm::br(db, op.location(db), 1).as_operation();

        RewriteResult::Replace(br_op)
    }
}

/// Pattern for `scf.break` -> `wasm.yield(value) + wasm.br(target=0)`
///
/// Branches to the enclosing wasm.block (depth 0) with a result value.
///
/// According to WASM spec, `br` instruction takes no operands - values are
/// passed via the stack. We use `wasm.yield` to mark the break value as the
/// region's result, then branch without operands.
struct ScfBreakPattern;

impl<'db> RewritePattern<'db> for ScfBreakPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(break_op) = scf::Break::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let value = break_op.value(db);

        // Emit the break value via wasm.yield (marks it as region result)
        let yield_op = wasm::r#yield(db, location, value).as_operation();

        // Branch to block (depth 0) without operands (WASM spec compliant)
        let br_op = wasm::br(db, location, 0).as_operation();

        RewriteResult::Expand(vec![yield_op, br_op])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::wasm;
    use trunk_ir::{BlockId, Location, PathId, Span};

    fn test_converter() -> TypeConverter {
        TypeConverter::new()
    }

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

        let scf_if = scf::r#if(
            db,
            location,
            cond_const.result(db),
            i32_ty,
            then_region,
            else_region,
        )
        .as_operation();

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
        let lowered = lower(db, module, test_converter());
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
