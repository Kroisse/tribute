//! Lower scf dialect operations to wasm dialect.
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> `wasm.yield` (tracks region result value)
//! - `scf.continue` -> `wasm.br(target=1)` (branch to loop)
//! - `scf.break` -> `wasm.br(target=2)` (branch to outer block, past if and loop)

use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::scf;
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{
    ConversionTarget, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
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
    let target = ConversionTarget::new()
        .legal_dialect("wasm")
        .illegal_dialect("scf");
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(scf_if_op) = scf::If::from_operation(db, *op) else {
            return false;
        };

        // wasm.if has the same structure: cond operand, result, then/else regions
        // PatternApplicator will recursively process the regions
        // Use rewriter to get remapped cond operand (important when cond is a result of another converted op)
        let cond = rewriter.operand(0).unwrap_or_else(|| scf_if_op.cond(db));
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

        rewriter.replace_op(new_op.as_operation());
        true
    }
}

/// Pattern for `scf.loop` -> `wasm.block(wasm.loop(...))`
///
/// The loop is wrapped in a block to provide a break target.
/// From inside a `wasm.if` within the loop body:
/// - `wasm.br(target=1)` branches to the loop (continue)
/// - `wasm.br(target=2)` branches to the block (break)
struct ScfLoopPattern;

impl<'db> RewritePattern<'db> for ScfLoopPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(loop_op) = scf::Loop::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let body = loop_op.body(db);

        // Get result type first
        let result_ty = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(db).as_type());

        // Remap init operands through the rewriter
        let init = loop_op.init(db);
        let remapped_init: Vec<_> = init
            .iter()
            .enumerate()
            .map(|(i, v)| rewriter.operand(i).unwrap_or(*v))
            .collect();

        // Create wasm.loop with init operands and the body region
        // PatternApplicator will recursively process the body
        let wasm_loop = wasm::r#loop(db, location, remapped_init, result_ty, body).as_operation();

        let block_body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            idvec![wasm_loop],
        );
        let block_body = Region::new(db, location, idvec![block_body_block]);

        let wasm_block = wasm::block(db, location, result_ty, block_body).as_operation();

        rewriter.replace_op(wasm_block);
        true
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        if !scf::Yield::matches(db, *op) {
            return false;
        };

        // Convert to wasm.yield which tracks the result value
        // This is needed because the yielded value may be defined outside the region
        // Use rewriter to get remapped operand
        let Some(value) = rewriter.operand(0) else {
            // No value to yield - just erase
            rewriter.erase_op(vec![]);
            return true;
        };

        let new_op = wasm::r#yield(db, op.location(db), value);
        rewriter.replace_op(new_op.as_operation());
        true
    }
}

/// Pattern for `scf.continue` -> `wasm.br(target=1)`
///
/// Branches to the enclosing wasm.loop. Depth 1 is correct because
/// `scf.continue` is always inside a `scf.if` (depth 0 = wasm.if,
/// depth 1 = wasm.loop) within a `scf.loop`.
struct ScfContinuePattern;

impl<'db> RewritePattern<'db> for ScfContinuePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(continue_op) = scf::Continue::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let values = continue_op.values(db);
        assert!(
            values.len() <= 1,
            "scf.continue with multiple loop-carried values not yet supported"
        );

        if values.is_empty() {
            // No loop-carried values -- simple branch
            let br_op = wasm::br(db, location, 1).as_operation();
            rewriter.replace_op(br_op);
            return true;
        }

        // Emit wasm.yield(value) + wasm.br(1) for each loop-carried value.
        // The emit layer will translate yield+br targeting a loop into
        // local.set for the loop arg followed by br.
        let value = rewriter.operand(0).unwrap_or(values[0]);
        let yield_op = wasm::r#yield(db, location, value).as_operation();
        let br_op = wasm::br(db, location, 1).as_operation();

        rewriter.insert_op(yield_op);
        rewriter.replace_op(br_op);
        true
    }
}

/// Pattern for `scf.break` -> `wasm.yield(value) + wasm.br(target=2)`
///
/// Branches to the enclosing wasm.block with a result value.
/// `scf.break` is always inside a `scf.if` within a `scf.loop`, so after
/// lowering the nesting is: wasm.block > wasm.loop > wasm.if. From inside
/// the wasm.if, depth 2 targets the outer wasm.block (break out of loop).
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
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(break_op) = scf::Break::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let value = rewriter.operand(0).unwrap_or_else(|| break_op.value(db));

        // Emit the break value via wasm.yield (marks it as region result)
        let yield_op = wasm::r#yield(db, location, value).as_operation();

        // Branch to outer block (depth 2: if=0, loop=1, block=2)
        let br_op = wasm::br(db, location, 2).as_operation();

        rewriter.insert_op(yield_op);
        rewriter.replace_op(br_op);
        true
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

    /// Recursively collect all operation names from a region (including nested regions).
    fn collect_all_op_names<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
    ) -> Vec<String> {
        let mut names = Vec::new();
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                names.push(op.full_name(db));
                for nested_region in op.regions(db).iter() {
                    names.extend(collect_all_op_names(db, nested_region));
                }
            }
        }
        names
    }

    /// Recursively collect `wasm.br` target depths from a region.
    fn collect_br_targets<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> Vec<u32> {
        let mut targets = Vec::new();
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                if let Ok(br) = wasm::Br::from_operation(db, *op) {
                    targets.push(br.target(db));
                }
                for nested_region in op.regions(db).iter() {
                    targets.extend(collect_br_targets(db, nested_region));
                }
            }
        }
        targets
    }

    /// Build a module with scf.break inside scf.if inside scf.loop.
    ///
    /// Structure: scf.loop { scf.if(cond) { then: scf.break(val) } { else: scf.continue } }
    #[salsa::tracked]
    fn make_scf_break_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Value to break with -- will be provided as loop init arg
        // The loop body receives it as block arg
        let loop_body_block_id = BlockId::fresh();

        // Create a condition value inside the loop body
        let cond_const = wasm::i32_const(db, location, i32_ty, 1);

        // The loop block arg represents the loop-carried value
        let loop_arg = trunk_ir::BlockArg::of_type(db, i32_ty);
        let loop_arg_val =
            trunk_ir::Value::new(db, trunk_ir::ValueDef::BlockArg(loop_body_block_id), 0);

        // then region: scf.break(loop_arg)
        let break_op = scf::r#break(db, location, loop_arg_val);
        let then_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![break_op.as_operation()],
        );
        let then_region = Region::new(db, location, idvec![then_block]);

        // else region: scf.continue(loop_arg)
        let continue_op = scf::r#continue(db, location, vec![loop_arg_val]);
        let else_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![continue_op.as_operation()],
        );
        let else_region = Region::new(db, location, idvec![else_block]);

        // scf.if inside loop body
        let scf_if = scf::r#if(
            db,
            location,
            cond_const.result(db),
            i32_ty,
            then_region,
            else_region,
        );

        // Loop body block: cond_const + scf.if
        let loop_body_block = Block::new(
            db,
            loop_body_block_id,
            location,
            IdVec::from(vec![loop_arg]),
            idvec![cond_const.operation(), scf_if.as_operation()],
        );
        let loop_body_region = Region::new(db, location, idvec![loop_body_block]);

        // Loop init value
        let init_val = wasm::i32_const(db, location, i32_ty, 42);

        // scf.loop with init and body
        let scf_loop = scf::r#loop(
            db,
            location,
            vec![init_val.result(db)],
            i32_ty,
            loop_body_region,
        );

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![init_val.operation(), scf_loop.as_operation()],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test_break".into(), region)
    }

    #[salsa::tracked]
    fn lower_and_collect_all(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let lowered = lower(db, module, test_converter());
        let body = lowered.body(db);
        collect_all_op_names(db, &body)
    }

    #[salsa::tracked]
    fn lower_and_collect_br_targets(db: &dyn salsa::Database, module: Module<'_>) -> Vec<u32> {
        let lowered = lower(db, module, test_converter());
        let body = lowered.body(db);
        collect_br_targets(db, &body)
    }

    #[salsa_test]
    fn test_scf_break_to_wasm_yield_and_br(db: &salsa::DatabaseImpl) {
        let module = make_scf_break_module(db);
        let all_ops = lower_and_collect_all(db, module);

        // scf.break should be gone
        assert!(
            !all_ops.iter().any(|n| n == "scf.break"),
            "scf.break should be lowered away, but found in: {:?}",
            all_ops
        );

        // wasm.yield and wasm.br should be present (from scf.break lowering)
        assert!(
            all_ops.iter().any(|n| n == "wasm.yield"),
            "Expected wasm.yield from scf.break lowering, got: {:?}",
            all_ops
        );
        assert!(
            all_ops.iter().any(|n| n == "wasm.br"),
            "Expected wasm.br from scf.break lowering, got: {:?}",
            all_ops
        );

        // scf.if and scf.loop should also be gone
        assert!(
            !all_ops.iter().any(|n| n == "scf.if"),
            "scf.if should be lowered, got: {:?}",
            all_ops
        );
        assert!(
            !all_ops.iter().any(|n| n == "scf.loop"),
            "scf.loop should be lowered, got: {:?}",
            all_ops
        );

        // Verify br target depths: continue=1 (loop), break=2 (outer block)
        let br_targets = lower_and_collect_br_targets(db, module);
        assert!(
            br_targets.contains(&1),
            "Expected br target=1 (continue -> loop), got targets: {:?}",
            br_targets
        );
        assert!(
            br_targets.contains(&2),
            "Expected br target=2 (break -> outer block), got targets: {:?}",
            br_targets
        );
    }
}
