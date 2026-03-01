//! Lower scf dialect operations to wasm dialect (arena IR).
//!
//! This pass converts structured control flow operations to wasm control:
//! - `scf.if` -> `wasm.if`
//! - `scf.loop` -> `wasm.block(wasm.loop(...))`
//! - `scf.yield` -> `wasm.yield` (tracks region result value)
//! - `scf.continue` -> `wasm.br(target=1)` (branch to loop)
//! - `scf.break` -> `wasm.br(target=2)` (branch to outer block, past if and loop)

use trunk_ir::arena::context::{BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::scf as arena_scf;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::OpRef;
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::smallvec::smallvec;

/// Lower scf dialect to wasm dialect using arena IR.
///
/// The `type_converter` parameter allows language-specific backends to provide
/// their own type conversion rules.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(ScfIfPattern)
        .add_pattern(ScfLoopPattern)
        .add_pattern(ScfYieldPattern)
        .add_pattern(ScfContinuePattern)
        .add_pattern(ScfBreakPattern);
    applicator.apply_partial(ctx, module);
}

/// Pattern for `scf.if` -> `wasm.if`
struct ScfIfPattern;

impl ArenaRewritePattern for ScfIfPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(scf_if_op) = arena_scf::If::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;

        // Get the condition operand
        let cond = scf_if_op.cond(ctx);

        // Get then/else regions and detach them from the original op
        let then_region = scf_if_op.then_region(ctx);
        let else_region = scf_if_op.else_region(ctx);
        ctx.detach_region(then_region);
        ctx.detach_region(else_region);

        // Get result type (default to nil if none)
        let result_types = ctx.op_result_types(op);
        let result_ty = result_types
            .first()
            .copied()
            .unwrap_or_else(|| intern_nil_type(ctx));

        let new_op = arena_wasm::r#if(ctx, loc, cond, result_ty, then_region, else_region);
        rewriter.replace_op(new_op.op_ref());
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

impl ArenaRewritePattern for ScfLoopPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(loop_op) = arena_scf::Loop::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;

        // Get result type
        let result_types = ctx.op_result_types(op);
        let result_ty = result_types
            .first()
            .copied()
            .unwrap_or_else(|| intern_nil_type(ctx));

        // Get init operands
        let init: Vec<_> = loop_op.init(ctx).to_vec();

        // Detach the body region from the original loop op
        let body = loop_op.body(ctx);
        ctx.detach_region(body);

        // Create wasm.loop with init operands and the body region
        let wasm_loop = arena_wasm::r#loop(ctx, loc, init, result_ty, body);

        // Create a block containing just the wasm.loop, to serve as the break target
        let block_body_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block_body_block, wasm_loop.op_ref());

        let block_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block_body_block],
            parent_op: None,
        });

        let wasm_block = arena_wasm::block(ctx, loc, result_ty, block_body);
        rewriter.replace_op(wasm_block.op_ref());
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

impl ArenaRewritePattern for ScfYieldPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_scf::Yield::matches(ctx, op) {
            return false;
        }

        // Get yield values (variadic operands)
        let operands = ctx.op_operands(op).to_vec();

        if operands.is_empty() {
            // No value to yield - just erase
            rewriter.erase_op(vec![]);
            return true;
        }

        if operands.len() > 1 {
            // Multi-value yields are not yet supported; leave unlowered.
            return false;
        }

        let value = operands[0];
        let loc = ctx.op(op).location;
        let new_op = arena_wasm::r#yield(ctx, loc, value);
        rewriter.replace_op(new_op.op_ref());
        true
    }
}

/// Pattern for `scf.continue` -> `wasm.br(target=1)`
///
/// Branches to the enclosing wasm.loop. Depth 1 is correct because
/// `scf.continue` is always inside a `scf.if` (depth 0 = wasm.if,
/// depth 1 = wasm.loop) within a `scf.loop`.
struct ScfContinuePattern;

impl ArenaRewritePattern for ScfContinuePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_scf::Continue::matches(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;

        // Get loop-carried values (variadic operands)
        let values = ctx.op_operands(op).to_vec();
        if values.len() > 1 {
            // Multiple loop-carried values not yet supported; leave unlowered.
            return false;
        }

        if values.is_empty() {
            // No loop-carried values -- simple branch
            let br_op = arena_wasm::br(ctx, loc, 1);
            rewriter.replace_op(br_op.op_ref());
            return true;
        }

        // Emit wasm.yield(value) + wasm.br(1) for each loop-carried value.
        // The emit layer will translate yield+br targeting a loop into
        // local.set for the loop arg followed by br.
        let value = values[0];
        let yield_op = arena_wasm::r#yield(ctx, loc, value);
        let br_op = arena_wasm::br(ctx, loc, 1);

        rewriter.insert_op(yield_op.op_ref());
        rewriter.replace_op(br_op.op_ref());
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

impl ArenaRewritePattern for ScfBreakPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(break_op) = arena_scf::Break::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let value = break_op.value(ctx);

        // Emit the break value via wasm.yield (marks it as region result)
        let yield_op = arena_wasm::r#yield(ctx, loc, value);

        // Branch to outer block (depth 2: if=0, loop=1, block=2)
        let br_op = arena_wasm::br(ctx, loc, 2);

        rewriter.insert_op(yield_op.op_ref());
        rewriter.replace_op(br_op.op_ref());
        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Intern a core.nil type.
fn intern_nil_type(ctx: &mut IrContext) -> trunk_ir::arena::refs::TypeRef {
    use trunk_ir::arena::types::TypeDataBuilder;
    use trunk_ir::ir::Symbol;
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
}
