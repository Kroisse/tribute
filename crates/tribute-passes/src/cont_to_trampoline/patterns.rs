use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::dialect::trampoline as arena_trampoline;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::types::Attribute;

use super::get_region_result_value;
use super::shift_lower::{anyref_type, i32_type, step_type};

// ============================================================================
// Pattern: Lower cont.resume
// ============================================================================

pub(crate) struct LowerResumePattern;

impl RewritePattern for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_cont::Resume::from_op(ctx, op).is_err() {
            return false;
        }

        let location = ctx.op(op).location;
        let i32_ty = i32_type(ctx);
        let anyref_ty = anyref_type(ctx);

        let operands = ctx.op_operands(op).to_vec();
        let continuation = *operands.first().expect("resume requires continuation");
        let value = operands.get(1).copied();

        let mut ops = Vec::new();

        // === 1. Reset yield state ===
        let reset = arena_trampoline::reset_yield_state(ctx, location);
        ops.push(reset.op_ref());

        // === 2. Get resume_fn from continuation (field 0 = resume_fn) ===
        let get_resume_fn =
            arena_trampoline::continuation_get(ctx, location, continuation, i32_ty, 0);
        let resume_fn_val = get_resume_fn.result(ctx);
        ops.push(get_resume_fn.op_ref());

        // === 3. Get state from continuation (field 1 = state) ===
        let get_state =
            arena_trampoline::continuation_get(ctx, location, continuation, anyref_ty, 1);
        let state_val = get_state.result(ctx);
        ops.push(get_state.op_ref());

        // === 4. Build resume wrapper ===
        let wrapper_ty = super::shift_lower::resume_wrapper_type(ctx);
        let resume_value = value.unwrap_or(state_val);

        let wrapper_op = arena_trampoline::build_resume_wrapper(
            ctx,
            location,
            state_val,
            resume_value,
            wrapper_ty,
        );
        let wrapper_val = wrapper_op.result(ctx);
        ops.push(wrapper_op.op_ref());

        // === 5. Call resume function ===
        let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);

        // Create null evidence
        let null_evidence = arena_adt::ref_null(ctx, location, evidence_ty, evidence_ty);
        let evidence_val = null_evidence.result(ctx);
        ops.push(null_evidence.op_ref());

        let step_ty = step_type(ctx);
        let call_op = arena_func::call_indirect(
            ctx,
            location,
            resume_fn_val,
            [evidence_val, wrapper_val],
            step_ty,
        );
        ops.push(call_op.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

use trunk_ir::dialect::adt as arena_adt;

// ============================================================================
// Pattern: Update func.call result type for effectful functions
// ============================================================================

pub(crate) struct UpdateEffectfulCallResultTypePattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
}

impl RewritePattern for UpdateEffectfulCallResultTypePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(call) = arena_func::Call::from_op(ctx, op) else {
            return false;
        };

        let callee = call.callee(ctx);

        if !self.effectful_funcs.contains(&callee) {
            return false;
        }

        let step_ty = step_type(ctx);
        let result_types = ctx.op_result_types(op);

        // Skip if already returns Step or no results
        if result_types.is_empty() {
            return false;
        }
        if is_step_type(ctx, result_types[0]) {
            return false;
        }

        let location = ctx.op(op).location;

        tracing::debug!(
            "UpdateEffectfulCallResultTypePattern: updating call to {} to Step",
            callee,
        );

        // Create new call with Step result type
        let op_data = ctx.op(op);
        let mut builder =
            trunk_ir::context::OperationDataBuilder::new(location, op_data.dialect, op_data.name)
                .operands(ctx.op_operands(op).to_vec())
                .result(step_ty);
        for (k, v) in &op_data.attributes {
            builder = builder.attr(*k, v.clone());
        }
        for &r in &op_data.regions {
            builder = builder.region(r);
        }
        let new_data = builder.build(ctx);
        let new_op = ctx.create_op(new_data);

        rewriter.replace_op(new_op);
        true
    }
}

// ============================================================================
// Pattern: Update scf.if result type when branches contain effectful calls
// ============================================================================

pub(crate) struct UpdateScfIfResultTypePattern;

impl RewritePattern for UpdateScfIfResultTypePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_scf::If::from_op(ctx, op).is_err() {
            return false;
        }

        let result_types = ctx.op_result_types(op).to_vec();
        if result_types.is_empty() {
            return false;
        }

        let step_ty = step_type(ctx);
        if is_step_type(ctx, result_types[0]) {
            return false;
        }

        // Check if any branch contains operations that return Step type
        let branches_have_step_ops = ctx.op(op).regions.iter().any(|&region| {
            ctx.region(region).blocks.iter().any(|&block| {
                ctx.block(block).ops.iter().any(|&branch_op| {
                    let rtypes = ctx.op_result_types(branch_op);
                    !rtypes.is_empty() && is_step_type(ctx, rtypes[0])
                })
            })
        });

        if !branches_have_step_ops {
            return false;
        }

        let location = ctx.op(op).location;
        tracing::debug!(
            "UpdateScfIfResultTypePattern: updating scf.if result to Step at {:?}",
            location
        );

        let op_data = ctx.op(op);
        let mut builder =
            trunk_ir::context::OperationDataBuilder::new(location, op_data.dialect, op_data.name)
                .operands(ctx.op_operands(op).to_vec())
                .result(step_ty);
        for (k, v) in &op_data.attributes {
            builder = builder.attr(*k, v.clone());
        }
        for &r in &op_data.regions {
            builder = builder.region(r);
        }
        let new_data = builder.build(ctx);
        let new_op = ctx.create_op(new_data);

        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern that updates scf.yield to yield Step value when it's inside
/// a block that contains effectful operations returning Step.
pub(crate) struct UpdateScfYieldToStepPattern;

impl RewritePattern for UpdateScfYieldToStepPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_scf::Yield::from_op(ctx, op).is_err() {
            return false;
        }

        let operands = ctx.op_operands(op).to_vec();
        if operands.is_empty() {
            return false;
        }

        // Skip if already yielding Step
        let yielded_value = operands[0];
        let yielded_ty = ctx.value_ty(yielded_value);
        if is_step_type(ctx, yielded_ty) {
            return false;
        }

        // Check if we can find the Step value through cast chain
        if let Some(step_value) = find_step_source(ctx, yielded_value) {
            tracing::debug!(
                "UpdateScfYieldToStepPattern: updating scf.yield to yield Step at {:?}",
                ctx.op(op).location
            );
            let location = ctx.op(op).location;
            let new_yield = arena_scf::r#yield(ctx, location, [step_value]);
            rewriter.replace_op(new_yield.op_ref());
            return true;
        }

        false
    }
}

/// Find the Step source value by tracing through the cast chain.
fn find_step_source(ctx: &IrContext, value: ValueRef) -> Option<ValueRef> {
    let ty = ctx.value_ty(value);
    if is_step_type(ctx, ty) {
        return Some(value);
    }

    // Check if the value definition is a cast from Step
    let def_op = value_def_op(ctx, value)?;
    if let Ok(cast) = arena_core::UnrealizedConversionCast::from_op(ctx, def_op) {
        let input = cast.value(ctx);
        let input_ty = ctx.value_ty(input);
        if is_step_type(ctx, input_ty) {
            return Some(input);
        }
    }

    // Check if the defining op produces Step
    let result_types = ctx.op_result_types(def_op);
    if !result_types.is_empty() && is_step_type(ctx, result_types[0]) {
        return Some(ctx.op_results(def_op)[0]);
    }

    None
}

// ============================================================================
// Pattern: Lower cont.push_prompt
// ============================================================================

pub(crate) struct LowerPushPromptPattern;

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
        let i32_ty = i32_type(ctx);
        let step_ty = step_type(ctx);
        let tag_attr = push_prompt.tag(ctx);
        let tag = match tag_attr {
            Attribute::Int(v) => u32::try_from(v).expect("push_prompt tag must fit in u32"),
            _ => panic!("push_prompt tag must be Int"),
        };

        // Get the body region
        let body = push_prompt.body(ctx);
        let body_result = get_region_result_value(ctx, body);

        let mut all_ops = Vec::new();

        // Add all body operations
        let blocks = &ctx.region(body).blocks;
        if let Some(&body_block) = blocks.first() {
            let ops = ctx.block(body_block).ops.to_vec();
            for body_op in ops {
                all_ops.push(body_op);
            }
        }

        // check_yield
        let check_yield = arena_trampoline::check_yield(ctx, location, i32_ty);
        let is_yielding = check_yield.result(ctx);
        all_ops.push(check_yield.op_ref());

        // Build yield handling branches
        let then_region = build_yield_then_branch(ctx, location, tag, step_ty);
        let else_region = build_yield_else_branch(ctx, location, body_result, step_ty);

        // scf.if for yield check
        let if_op = arena_scf::r#if(
            ctx,
            location,
            is_yielding,
            step_ty,
            then_region,
            else_region,
        );
        all_ops.push(if_op.op_ref());

        let last = all_ops.pop().unwrap();
        for o in all_ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

fn build_yield_then_branch(
    ctx: &mut IrContext,
    location: trunk_ir::types::Location,
    tag: u32,
    step_ty: TypeRef,
) -> trunk_ir::refs::RegionRef {
    let i32_ty = i32_type(ctx);
    let cont_ty = super::shift_lower::continuation_type(ctx);

    let tag_const = arith::r#const(ctx, location, i32_ty, Attribute::Int(tag as i128));
    let tag_val = tag_const.result(ctx);

    let get_cont = arena_trampoline::get_yield_continuation(ctx, location, cont_ty);
    let cont_val = get_cont.result(ctx);

    let step_shift = arena_trampoline::step_shift(ctx, location, tag_val, cont_val, step_ty, 0);
    let step_shift_val = step_shift.result(ctx);

    let yield_op = arena_scf::r#yield(ctx, location, [step_shift_val]);

    let block = ctx.create_block(trunk_ir::context::BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });
    ctx.push_op(block, tag_const.op_ref());
    ctx.push_op(block, get_cont.op_ref());
    ctx.push_op(block, step_shift.op_ref());
    ctx.push_op(block, yield_op.op_ref());

    ctx.create_region(trunk_ir::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

fn build_yield_else_branch(
    ctx: &mut IrContext,
    location: trunk_ir::types::Location,
    body_result: Option<ValueRef>,
    step_ty: TypeRef,
) -> trunk_ir::refs::RegionRef {
    let block = ctx.create_block(trunk_ir::context::BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    let step_value = if let Some(result) = body_result {
        // Check if result is already a Step
        let is_step = {
            if let Some(def_op) = value_def_op(ctx, result) {
                let rtypes = ctx.op_result_types(def_op);
                !rtypes.is_empty() && is_step_type(ctx, rtypes[0])
            } else {
                false
            }
        };

        if is_step {
            result
        } else {
            let step_done = arena_trampoline::step_done(ctx, location, result, step_ty);
            ctx.push_op(block, step_done.op_ref());
            step_done.result(ctx)
        }
    } else {
        // No body result - create a step_done with zero value
        let i32_ty = i32_type(ctx);
        let zero = arith::r#const(ctx, location, i32_ty, Attribute::Int(0));
        ctx.push_op(block, zero.op_ref());
        let step_done = arena_trampoline::step_done(ctx, location, zero.result(ctx), step_ty);
        ctx.push_op(block, step_done.op_ref());
        step_done.result(ctx)
    };

    let yield_op = arena_scf::r#yield(ctx, location, [step_value]);
    ctx.push_op(block, yield_op.op_ref());

    ctx.create_region(trunk_ir::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Check if a TypeRef is the trampoline.step type.
pub(crate) fn is_step_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("trampoline") && data.name == Symbol::new("step")
}

/// Get the defining OpRef of a value, if it's an operation result.
fn value_def_op(ctx: &IrContext, value: ValueRef) -> Option<OpRef> {
    match ctx.value_def(value) {
        trunk_ir::refs::ValueDef::OpResult(op, _) => Some(op),
        trunk_ir::refs::ValueDef::BlockArg(_, _) => None,
    }
}
