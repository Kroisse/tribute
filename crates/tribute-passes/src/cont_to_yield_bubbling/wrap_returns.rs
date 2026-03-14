//! Wrap function returns in effectful functions with YieldResult::Done.

use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};

use trunk_ir::rewrite::Module;

use super::types::{YieldBubblingTypes, is_yield_result_type};

// ============================================================================
// Pattern: Wrap returns in effectful functions with YieldResult::Done
// ============================================================================

pub(crate) struct WrapReturnsPattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
    pub(crate) types: YieldBubblingTypes,
}

impl RewritePattern for WrapReturnsPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        _rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_name = func.sym_name(ctx);
        if !self.effectful_funcs.contains(&func_name) {
            return false;
        }

        let body = func.body(ctx);
        wrap_returns_in_region(ctx, body, &self.types);

        // Return false - mutated in place, no replacement needed
        false
    }
}

/// Wrap returns only in the specified target functions.
///
/// Used for post-processing newly generated resume/chain functions.
pub(crate) fn wrap_returns_for_funcs(
    ctx: &mut IrContext,
    module: Module,
    func_names: &[Symbol],
    types: &YieldBubblingTypes,
) {
    let module_body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };

    let blocks: Vec<BlockRef> = ctx.region(module_body).blocks.to_vec();
    for block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
        for op in ops {
            if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                let func_name = func.sym_name(ctx);
                if func_names.contains(&func_name) {
                    let body = func.body(ctx);
                    wrap_returns_in_region(ctx, body, types);
                }
            }
        }
    }
}

/// Recursively wrap returns in a region with YieldResult::Done.
fn wrap_returns_in_region(ctx: &mut IrContext, region: RegionRef, types: &YieldBubblingTypes) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        wrap_returns_in_block(ctx, block, types);
    }
}

/// Wrap returns in a block.
fn wrap_returns_in_block(ctx: &mut IrContext, block: BlockRef, types: &YieldBubblingTypes) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        // Recursively process nested regions
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for r in regions {
            wrap_returns_in_region(ctx, r, types);
        }

        // Check if this is func.return
        if arena_func::Return::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op).to_vec();

            if let Some(&value) = operands.first() {
                let is_yr = is_yield_result_type(ctx, ctx.value_ty(value));
                if !is_yr {
                    let location = ctx.op(op).location;

                    // Cast to anyref
                    let anyref_val =
                        arena_core::unrealized_conversion_cast(ctx, location, value, types.anyref);
                    ctx.insert_op_before(block, op, anyref_val.op_ref());

                    // Create YieldResult::Done
                    let done_op = arena_adt::variant_new(
                        ctx,
                        location,
                        [anyref_val.result(ctx)],
                        types.yield_result,
                        types.yield_result,
                        Symbol::new("Done"),
                    );
                    ctx.insert_op_before(block, op, done_op.op_ref());

                    // Update return operand
                    ctx.set_op_operand(op, 0, done_op.result(ctx));
                }
            }
        }
    }
}
