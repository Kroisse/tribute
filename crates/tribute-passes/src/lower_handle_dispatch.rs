//! Lower `ability.handle_dispatch` to inline done-handler application.
//!
//! In the tail-call CPS design, effect operations are handled via tail calls
//! to handler_dispatch closures (see `lower_ability_perform`). By the time
//! `ability.handle_dispatch` is reached, the body result is already the final
//! value. This pass simply applies the done handler to the body result.

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func, scf};
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Location;

use tribute_ir::dialect::ability as arena_ability;

use crate::cont_to_yield_bubbling::types::YieldBubblingTypes;
use crate::cont_util::get_done_region;

/// Lower all `ability.handle_dispatch` ops in the module.
pub fn lower_handle_dispatch(ctx: &mut IrContext, module: Module) {
    let types = YieldBubblingTypes::new(ctx);

    let func_ops: Vec<OpRef> = module.ops(ctx);
    for func_op_ref in func_ops {
        let Ok(func_op) = func::Func::from_op(ctx, func_op_ref) else {
            continue;
        };

        let body = func_op.body(ctx);
        let blocks: Vec<BlockRef> = ctx.region(body).blocks.to_vec();
        lower_dispatches_in_blocks(ctx, &blocks, &types);
    }
}

fn lower_dispatches_in_blocks(
    ctx: &mut IrContext,
    blocks: &[BlockRef],
    types: &YieldBubblingTypes,
) {
    for &block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            if arena_ability::HandleDispatch::matches(ctx, op) {
                lower_single_dispatch(ctx, block, op, types);
            }

            // Recurse into nested regions (but not the dispatch's own body —
            // it gets consumed during lowering).
            if !arena_ability::HandleDispatch::matches(ctx, op) {
                let regions: Vec<_> = ctx.op(op).regions.to_vec();
                for region in regions {
                    let inner_blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
                    lower_dispatches_in_blocks(ctx, &inner_blocks, types);
                }
            }
        }
    }
}

fn lower_single_dispatch(
    ctx: &mut IrContext,
    block: BlockRef,
    op: OpRef,
    _types: &YieldBubblingTypes,
) {
    let Ok(dispatch_op) = arena_ability::HandleDispatch::from_op(ctx, op) else {
        return;
    };

    let location = ctx.op(op).location;
    // operand[0] = body result (anyref), operand[1] = handler_fn (unused here)
    let body_result = ctx.op_operands(op)[0];
    let user_result_ty = dispatch_op.result_type(ctx);
    let handler_body = dispatch_op.body(ctx);

    // In the tail-call CPS design, the body result is the final value
    // (effects are handled via tail calls, not YieldResult dispatch).
    // Just apply the done handler to the body result.
    let done_region = get_done_region(ctx, handler_body);

    let final_result = if let Some(done_body) = done_region {
        inline_done_body(ctx, block, location, done_body, body_result, op)
    } else {
        body_result
    };

    // Cast to user result type if needed
    let result_val = if ctx.value_ty(final_result) != user_result_ty {
        let cast = core::unrealized_conversion_cast(ctx, location, final_result, user_result_ty);
        ctx.insert_op_before(block, op, cast.op_ref());
        cast.result(ctx)
    } else {
        final_result
    };

    // Replace dispatch op result with the done handler result.
    let old_result = ctx.op_result(op, 0);
    ctx.replace_all_uses(old_result, result_val);
    ctx.remove_op_from_block(block, op);
}

/// Inline the `cont.done` region's body before `insert_before`.
///
/// The done region has a single block argument (the body result value).
/// We map that argument to `done_value` and clone the ops into `dest_block`.
/// `scf.yield` terminators are skipped — their operand becomes the result.
fn inline_done_body(
    ctx: &mut IrContext,
    dest_block: BlockRef,
    _location: Location,
    done_body: trunk_ir::refs::RegionRef,
    done_value: ValueRef,
    insert_before: OpRef,
) -> ValueRef {
    let done_blocks = &ctx.region(done_body).blocks;
    let Some(&done_block) = done_blocks.first() else {
        return done_value;
    };

    let mut mapping = IrMapping::new();
    let done_block_args = ctx.block_args(done_block).to_vec();
    if !done_block_args.is_empty() {
        mapping.map_value(done_block_args[0], done_value);
    }

    let mut final_result = done_value;
    let done_ops: Vec<OpRef> = ctx.block(done_block).ops.clone().to_vec();
    for &done_op in &done_ops {
        if scf::Yield::matches(ctx, done_op) {
            let yielded = ctx.op_operands(done_op).to_vec();
            if let Some(&result) = yielded.first() {
                final_result = mapping.lookup_value_or_default(result);
            }
            continue;
        }
        let cloned = ctx.clone_op(done_op, &mut mapping);
        ctx.insert_op_before(dest_block, insert_before, cloned);
        let cloned_results = ctx.op_results(cloned);
        if !cloned_results.is_empty() {
            final_result = cloned_results[0];
        }
    }

    final_result
}
