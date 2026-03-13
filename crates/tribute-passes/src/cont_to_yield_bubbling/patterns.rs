//! Additional rewrite patterns for yield bubbling.
//!
//! - `LowerResumePattern`: Lower `cont.resume` to call_indirect via Continuation struct
//! - `UpdateEffectfulCallResultTypePattern`: Update effectful call result types to YieldResult
//! - `UpdateScfIfResultTypePattern`: Update scf.if result types that contain YieldResult
//! - `UpdateScfYieldToYieldResultPattern`: Ensure scf.yield operands match YieldResult type
//! - `LowerPushPromptPattern`: Lower `cont.push_prompt` to body call + handler dispatch loop

use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};

use super::types::{YieldBubblingTypes, is_yield_result_type};

// ============================================================================
// Pattern: Lower cont.resume → call_indirect via Continuation struct
// ============================================================================

pub(crate) struct LowerResumePattern {
    pub(crate) types: YieldBubblingTypes,
}

impl RewritePattern for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(_resume) = arena_cont::Resume::from_op(ctx, op) else {
            return false;
        };

        let operands = ctx.op_operands(op).to_vec();
        if operands.len() < 2 {
            return false;
        }

        let location = ctx.op(op).location;
        let t = &self.types;

        let k_operand = operands[0]; // continuation (anyref)
        let value_operand = operands[1]; // resume value

        let mut ops = Vec::new();

        // Cast continuation to Continuation struct
        let k_cast =
            arena_core::unrealized_conversion_cast(ctx, location, k_operand, t.continuation);
        ops.push(k_cast.op_ref());

        // Extract resume_fn (field 0) — typed as core.ptr (not RC-managed)
        let get_fn =
            arena_adt::struct_get(ctx, location, k_cast.result(ctx), t.ptr, t.continuation, 0);
        ops.push(get_fn.op_ref());

        // Extract state (field 1)
        let get_state = arena_adt::struct_get(
            ctx,
            location,
            k_cast.result(ctx),
            t.anyref,
            t.continuation,
            1,
        );
        ops.push(get_state.op_ref());

        // Cast resume_value to anyref
        let rv_anyref =
            arena_core::unrealized_conversion_cast(ctx, location, value_operand, t.anyref);
        ops.push(rv_anyref.op_ref());

        // Build ResumeWrapper { state, resume_value }
        let wrapper = arena_adt::struct_new(
            ctx,
            location,
            vec![get_state.result(ctx), rv_anyref.result(ctx)],
            t.anyref,
            t.resume_wrapper,
        );
        ops.push(wrapper.op_ref());

        // Cast wrapper to anyref for call_indirect
        let wrapper_anyref =
            arena_core::unrealized_conversion_cast(ctx, location, wrapper.result(ctx), t.anyref);
        ops.push(wrapper_anyref.op_ref());

        // Get evidence (use a null evidence for now — evidence is passed through separately)
        let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);
        let null_ev = arena_adt::ref_null(ctx, location, evidence_ty, evidence_ty);
        ops.push(null_ev.op_ref());

        // call_indirect(resume_fn, evidence, wrapper) -> YieldResult
        let call = arena_func::call_indirect(
            ctx,
            location,
            get_fn.result(ctx),
            vec![null_ev.result(ctx), wrapper_anyref.result(ctx)],
            t.yield_result,
        );
        ops.push(call.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Pattern: Update effectful call result types to YieldResult
// ============================================================================

pub(crate) struct UpdateEffectfulCallResultTypePattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
    pub(crate) types: YieldBubblingTypes,
}

impl RewritePattern for UpdateEffectfulCallResultTypePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        _rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call) = arena_func::Call::from_op(ctx, op) else {
            return false;
        };

        let callee = call.callee(ctx);
        if !self.effectful_funcs.contains(&callee) {
            return false;
        }

        let result_types = ctx.op_result_types(op);
        if result_types.is_empty() {
            return false;
        }

        if is_yield_result_type(ctx, result_types[0]) {
            return false;
        }

        ctx.set_op_result_type(op, 0, self.types.yield_result);
        true
    }
}

// ============================================================================
// Pattern: Update scf.if result types
// ============================================================================

pub(crate) struct UpdateScfIfResultTypePattern {
    pub(crate) types: YieldBubblingTypes,
}

impl RewritePattern for UpdateScfIfResultTypePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        _rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_scf::If::from_op(ctx, op).is_err() {
            return false;
        }

        let result_types = ctx.op_result_types(op);
        if result_types.is_empty() {
            return false;
        }

        // Check if any branch yields a YieldResult
        let regions: Vec<trunk_ir::refs::RegionRef> = ctx.op(op).regions.to_vec();
        let has_yr_branch = regions.iter().any(|&r| region_yields_yr(ctx, r));

        if !has_yr_branch || is_yield_result_type(ctx, result_types[0]) {
            return false;
        }

        ctx.set_op_result_type(op, 0, self.types.yield_result);
        true
    }
}

use trunk_ir::dialect::scf as arena_scf;

fn region_yields_yr(ctx: &IrContext, region: trunk_ir::refs::RegionRef) -> bool {
    let blocks = &ctx.region(region).blocks;
    for &block in blocks {
        let ops = &ctx.block(block).ops;
        for &op in ops {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
                return true;
            }
        }
    }
    false
}

// ============================================================================
// Pattern: Update scf.yield to YieldResult type
// ============================================================================

pub(crate) struct UpdateScfYieldToYieldResultPattern {
    pub(crate) _types: YieldBubblingTypes,
}

impl RewritePattern for UpdateScfYieldToYieldResultPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        _rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_scf::Yield::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.is_empty() {
            return false;
        }

        let val = operands[0];
        let val_ty = ctx.value_ty(val);

        // If value is already YieldResult, nothing to do
        if is_yield_result_type(ctx, val_ty) {
            return false;
        }

        // Check if this yield is in a context expecting YieldResult
        // For now, skip — this pattern is applied after other patterns
        false
    }
}

// ============================================================================
// Pattern: Lower cont.push_prompt
// ============================================================================

pub(crate) struct LowerPushPromptPattern {
    pub(crate) types: YieldBubblingTypes,
}

impl RewritePattern for LowerPushPromptPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(push_prompt) = arena_cont::PushPrompt::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let t = &self.types;

        // Get the body region and inline its operations
        let body = push_prompt.body(ctx);
        let body_result = crate::cont_util::get_region_result_value(ctx, body);

        let mut all_ops = Vec::new();

        // Inline body operations (skip trailing scf.yield — we handle the result ourselves)
        let blocks = &ctx.region(body).blocks;
        if let Some(&body_block) = blocks.first() {
            let ops = ctx.block(body_block).ops.to_vec();
            for body_op in ops {
                // Skip scf.yield — we'll use body_result directly
                if arena_scf::Yield::from_op(ctx, body_op).is_ok() {
                    continue;
                }
                // Detach from the original block before re-inserting
                ctx.detach_op(body_op);
                all_ops.push(body_op);
            }
        }

        // Determine if body result is already a YieldResult
        let yr_value = if let Some(result) = body_result {
            let result_ty = ctx.value_ty(result);
            if is_yield_result_type(ctx, result_ty) {
                // Already a YieldResult — pass through
                result
            } else {
                // Wrap in YieldResult::Done
                let anyref_val =
                    arena_core::unrealized_conversion_cast(ctx, location, result, t.anyref);
                all_ops.push(anyref_val.op_ref());

                let done_op = arena_adt::variant_new(
                    ctx,
                    location,
                    [anyref_val.result(ctx)],
                    t.yield_result,
                    t.yield_result,
                    Symbol::new("Done"),
                );
                all_ops.push(done_op.op_ref());
                done_op.result(ctx)
            }
        } else {
            // No body result — create a YieldResult::Done with null
            let null_op = arena_adt::ref_null(ctx, location, t.anyref, t.anyref);
            all_ops.push(null_op.op_ref());

            let done_op = arena_adt::variant_new(
                ctx,
                location,
                [null_op.result(ctx)],
                t.yield_result,
                t.yield_result,
                Symbol::new("Done"),
            );
            all_ops.push(done_op.op_ref());
            done_op.result(ctx)
        };

        // Now inline the handlers region operations
        let handlers = push_prompt.handlers(ctx);
        let handler_blocks = &ctx.region(handlers).blocks;
        if let Some(&handler_block) = handler_blocks.first() {
            let handler_ops = ctx.block(handler_block).ops.to_vec();
            for handler_op in handler_ops {
                // Wire the YieldResult into handler_dispatch's operand
                if arena_cont::HandlerDispatch::from_op(ctx, handler_op).is_ok() {
                    let operands = ctx.op_operands(handler_op).to_vec();
                    if !operands.is_empty() {
                        ctx.set_op_operand(handler_op, 0, yr_value);
                    }
                }

                // Skip scf.yield in handlers region
                if arena_scf::Yield::from_op(ctx, handler_op).is_ok() {
                    continue;
                }
                // Detach from the original block before re-inserting
                ctx.detach_op(handler_op);
                all_ops.push(handler_op);
            }
        }

        if all_ops.is_empty() {
            return false;
        }

        let last = all_ops.pop().unwrap();
        for o in all_ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}
