use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::dialect::trampoline as arena_trampoline;
use trunk_ir::arena::ops::DialectOp;
use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef, ValueRef};
use trunk_ir::arena::rewrite::{PatternRewriter as ArenaPatternRewriter, RewritePattern};

use super::patterns::is_step_type;
use super::shift_lower::step_type;

// ============================================================================
// Pattern: Wrap returns in effectful functions with step_done
// ============================================================================

pub(crate) struct WrapReturnsInEffectfulFuncsPattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
}

impl RewritePattern for WrapReturnsInEffectfulFuncsPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        _rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(func) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_name = func.sym_name(ctx);
        if !self.effectful_funcs.contains(&func_name) {
            return false;
        }

        tracing::debug!(
            "WrapReturnsInEffectfulFuncsPattern: processing {}",
            func_name
        );

        let body = func.body(ctx);
        let modified = wrap_returns_in_region(ctx, body);

        tracing::debug!(
            "WrapReturnsInEffectfulFuncsPattern: {} modified={}",
            func_name,
            modified
        );

        // Return false - we mutated in place so no replacement needed
        false
    }
}

/// Recursively wrap returns in a region with step_done.
fn wrap_returns_in_region(ctx: &mut IrContext, region: RegionRef) -> bool {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    let mut any_modified = false;

    for block in blocks {
        any_modified |= wrap_returns_in_block(ctx, block);
    }

    any_modified
}

/// Wrap returns in a block with step_done.
fn wrap_returns_in_block(ctx: &mut IrContext, block: BlockRef) -> bool {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let mut modified = false;

    for op in ops {
        // First, recursively process nested regions
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for r in regions {
            modified |= wrap_returns_in_region(ctx, r);
        }

        // Check if this is a func.return
        if arena_func::Return::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op).to_vec();

            if let Some(&value) = operands.first() {
                let is_step = is_step_value(ctx, value);
                tracing::debug!(
                    "wrap_returns_in_block: found func.return, value is_step={}",
                    is_step
                );
                if !is_step {
                    let location = ctx.op(op).location;
                    let step_ty = step_type(ctx);

                    // Create step_done(value)
                    let step_done = arena_trampoline::step_done(ctx, location, value, step_ty);
                    let step_value = step_done.result(ctx);
                    ctx.insert_op_before(block, op, step_done.op_ref());

                    // Update return operand
                    ctx.set_op_operand(op, 0, step_value);
                    modified = true;
                    tracing::debug!("wrap_returns_in_block: wrapped return with step_done");
                }
            }
        }
    }

    modified
}

/// Check if a value is already a Step type.
pub(crate) fn is_step_value(ctx: &IrContext, value: ValueRef) -> bool {
    let ty = ctx.value_ty(value);
    is_step_type(ctx, ty)
}
