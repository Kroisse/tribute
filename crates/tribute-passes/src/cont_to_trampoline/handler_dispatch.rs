use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::arena::context::{BlockData, IrContext, OperationDataBuilder, RegionData};
use trunk_ir::arena::dialect::arith as arena_arith;
use trunk_ir::arena::dialect::cont as arena_cont;
use trunk_ir::arena::dialect::core as arena_core;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::dialect::scf as arena_scf;
use trunk_ir::arena::dialect::trampoline as arena_trampoline;
use trunk_ir::arena::ops::DialectOp;
use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{PatternRewriter as ArenaPatternRewriter, RewritePattern};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, Location};
use trunk_ir::location::Span;

use super::patterns::is_step_type;
use super::shift_lower::{anyref_type, continuation_type, i32_type, step_type};
use crate::cont_util::{SuspendArm, collect_suspend_arms_arena};

// ============================================================================
// Pattern: Lower cont.handler_dispatch
// ============================================================================

pub(crate) struct LowerHandlerDispatchPattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
    /// Spans of handler_dispatch operations that are inside effectful functions.
    pub(crate) handlers_in_effectful_funcs: Rc<HashSet<Span>>,
}

impl RewritePattern for LowerHandlerDispatchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch) = arena_cont::HandlerDispatch::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let i32_ty = i32_type(ctx);
        let i1_ty = ctx.types.intern(
            trunk_ir::arena::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1"))
                .build(),
        );
        let step_ty = step_type(ctx);
        let anyref_ty = anyref_type(ctx);

        // Get the step operand (result of push_prompt)
        let step_operand = ctx.op_operands(op)[0];

        // Get the handler's tag
        let our_tag = dispatch.tag(ctx);

        // Get the user's result type
        let user_result_ty = dispatch.result_type(ctx);

        // Get the body region with child ops
        let body_region = dispatch.body(ctx);

        // Collect suspend arms
        let suspend_arms = collect_suspend_arms_arena(ctx, body_region);

        // Check if this handler is inside an effectful function
        let is_in_effectful_func = self.handlers_in_effectful_funcs.contains(&location.span);
        let loop_result_ty = if is_in_effectful_func {
            tracing::debug!(
                "LowerHandlerDispatchPattern: handler in effectful func, returning Step"
            );
            step_ty
        } else {
            user_result_ty
        };

        // Build the trampoline loop body
        let loop_body = self.build_trampoline_loop_body(
            ctx,
            location,
            our_tag,
            &suspend_arms,
            user_result_ty,
            step_ty,
            i32_ty,
            i1_ty,
            anyref_ty,
            is_in_effectful_func,
        );

        // Create scf.loop with step_operand as initial value
        let loop_op = arena_scf::r#loop(ctx, location, [step_operand], loop_result_ty, loop_body);

        rewriter.replace_op(loop_op.op_ref());
        true
    }
}

impl LowerHandlerDispatchPattern {
    #[allow(clippy::too_many_arguments)]
    fn build_trampoline_loop_body(
        &self,
        ctx: &mut IrContext,
        location: Location,
        our_tag: u32,
        suspend_arms: &[SuspendArm],
        user_result_ty: TypeRef,
        step_ty: TypeRef,
        i32_ty: TypeRef,
        i1_ty: TypeRef,
        anyref_ty: TypeRef,
        is_in_effectful_func: bool,
    ) -> RegionRef {
        // Create block with current_step as argument
        let block = ctx.create_block(BlockData {
            location,
            args: vec![trunk_ir::arena::context::BlockArgData {
                ty: step_ty,
                attrs: Default::default(),
            }],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let current_step = ctx.block_args(block)[0];

        // Extract tag field from Step (field 0 = tag)
        let get_tag = arena_trampoline::step_get(ctx, location, current_step, i32_ty, 0);
        ctx.push_op(block, get_tag.op_ref());
        let step_tag = get_tag.result(ctx);

        // Compare with DONE (0)
        let done_const = arena_arith::r#const(ctx, location, i32_ty, ArenaAttribute::IntBits(0));
        ctx.push_op(block, done_const.op_ref());
        let is_done = arena_arith::cmp_eq(ctx, location, step_tag, done_const.result(ctx), i1_ty);
        ctx.push_op(block, is_done.op_ref());

        // Build Done branch
        let done_branch = self.build_done_branch(
            ctx,
            location,
            current_step,
            user_result_ty,
            anyref_ty,
            step_ty,
            is_in_effectful_func,
        );

        // Build Shift branch
        let shift_branch = self.build_shift_branch(
            ctx,
            location,
            our_tag,
            current_step,
            suspend_arms,
            step_ty,
            i32_ty,
            i1_ty,
        );

        // scf.if
        let nil_ty = arena_core::nil(ctx).as_type_ref();
        let if_op = arena_scf::r#if(
            ctx,
            location,
            is_done.result(ctx),
            nil_ty,
            done_branch,
            shift_branch,
        );
        ctx.push_op(block, if_op.op_ref());

        ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_done_branch(
        &self,
        ctx: &mut IrContext,
        location: Location,
        current_step: ValueRef,
        user_result_ty: TypeRef,
        anyref_ty: TypeRef,
        step_ty: TypeRef,
        is_in_effectful_func: bool,
    ) -> RegionRef {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });

        // Extract value from Step (field 1 = value)
        let get_value = arena_trampoline::step_get(ctx, location, current_step, anyref_ty, 1);
        ctx.push_op(block, get_value.op_ref());
        let step_value = get_value.result(ctx);

        // Cast anyref to user result type if needed
        let result_value = if anyref_ty != user_result_ty {
            let cast =
                arena_core::unrealized_conversion_cast(ctx, location, step_value, user_result_ty);
            ctx.push_op(block, cast.op_ref());
            cast.result(ctx)
        } else {
            step_value
        };

        if is_in_effectful_func {
            let step_done = arena_trampoline::step_done(ctx, location, result_value, step_ty);
            ctx.push_op(block, step_done.op_ref());
            let break_op = arena_scf::r#break(ctx, location, step_done.result(ctx));
            ctx.push_op(block, break_op.op_ref());
        } else {
            let break_op = arena_scf::r#break(ctx, location, result_value);
            ctx.push_op(block, break_op.op_ref());
        }

        ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_shift_branch(
        &self,
        ctx: &mut IrContext,
        location: Location,
        our_tag: u32,
        current_step: ValueRef,
        suspend_arms: &[SuspendArm],
        step_ty: TypeRef,
        i32_ty: TypeRef,
        i1_ty: TypeRef,
    ) -> RegionRef {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });

        // Extract prompt tag from Step (field 2 = prompt)
        let get_prompt = arena_trampoline::step_get(ctx, location, current_step, i32_ty, 2);
        ctx.push_op(block, get_prompt.op_ref());
        let step_prompt = get_prompt.result(ctx);

        // Compare with our handler's tag
        let our_tag_const = arena_arith::r#const(
            ctx,
            location,
            i32_ty,
            ArenaAttribute::IntBits(our_tag as u64),
        );
        ctx.push_op(block, our_tag_const.op_ref());
        let tag_matches =
            arena_arith::cmp_eq(ctx, location, step_prompt, our_tag_const.result(ctx), i1_ty);
        ctx.push_op(block, tag_matches.op_ref());

        // Build dispatch region (when tag matches)
        let dispatch_region = self.build_dispatch_region(ctx, location, suspend_arms, step_ty);

        // Build propagate region (when tag doesn't match)
        let propagate_region = self.build_propagate_region(ctx, location, current_step);

        // scf.if
        let if_op = arena_scf::r#if(
            ctx,
            location,
            tag_matches.result(ctx),
            step_ty,
            dispatch_region,
            propagate_region,
        );
        ctx.push_op(block, if_op.op_ref());

        // Continue loop with new step
        let continue_op = arena_scf::r#continue(ctx, location, [if_op.result(ctx)]);
        ctx.push_op(block, continue_op.op_ref());

        ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        })
    }

    fn build_dispatch_region(
        &self,
        ctx: &mut IrContext,
        location: Location,
        suspend_arms: &[SuspendArm],
        step_ty: TypeRef,
    ) -> RegionRef {
        build_suspend_dispatch_region(ctx, location, step_ty, suspend_arms, &self.effectful_funcs)
    }

    fn build_propagate_region(
        &self,
        ctx: &mut IrContext,
        location: Location,
        current_step: ValueRef,
    ) -> RegionRef {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });

        let yield_op = arena_scf::r#yield(ctx, location, [current_step]);
        ctx.push_op(block, yield_op.op_ref());

        ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        })
    }
}

/// Build a single-block region for suspend dispatch using nested scf.if.
fn build_suspend_dispatch_region(
    ctx: &mut IrContext,
    location: Location,
    result_ty: TypeRef,
    suspend_arms: &[SuspendArm],
    effectful_funcs: &HashSet<Symbol>,
) -> RegionRef {
    let i32_ty = i32_type(ctx);

    if suspend_arms.is_empty() {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let unreachable = arena_func::unreachable(ctx, location);
        ctx.push_op(block, unreachable.op_ref());
        return ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        });
    }

    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Get current op_idx
    let get_op_idx = arena_trampoline::get_yield_op_idx(ctx, location, i32_ty);
    ctx.push_op(block, get_op_idx.op_ref());
    let current_op_idx = get_op_idx.result(ctx);

    // Build nested if-else dispatch
    let final_result = build_nested_dispatch(
        ctx,
        block,
        location,
        result_ty,
        current_op_idx,
        suspend_arms,
        0,
        effectful_funcs,
    );

    // Yield the result
    let yield_op = arena_scf::r#yield(ctx, location, [final_result]);
    ctx.push_op(block, yield_op.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_nested_dispatch(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    result_ty: TypeRef,
    current_op_idx: ValueRef,
    suspend_arms: &[SuspendArm],
    arm_index: usize,
    effectful_funcs: &HashSet<Symbol>,
) -> ValueRef {
    let i1_ty = ctx.types.intern(
        trunk_ir::arena::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1"))
            .build(),
    );

    if arm_index >= suspend_arms.len() {
        panic!("build_nested_dispatch: arm_index out of bounds");
    }

    let arm = &suspend_arms[arm_index];
    let is_last_arm = arm_index + 1 >= suspend_arms.len();

    // Build then region
    let then_region = build_arm_region(ctx, location, arm.body, effectful_funcs);

    if is_last_arm {
        // Last arm: always-true condition
        let true_const = arena_arith::r#const(ctx, location, i1_ty, ArenaAttribute::IntBits(1));
        ctx.push_op(block, true_const.op_ref());
        let else_region = build_arm_region(ctx, location, arm.body, effectful_funcs);

        let if_op = arena_scf::r#if(
            ctx,
            location,
            true_const.result(ctx),
            result_ty,
            then_region,
            else_region,
        );
        ctx.push_op(block, if_op.op_ref());
        return if_op.result(ctx);
    }

    // Compare current op_idx with expected
    let i32_ty = i32_type(ctx);
    let expected_const = arena_arith::r#const(
        ctx,
        location,
        i32_ty,
        ArenaAttribute::IntBits(arm.expected_op_idx as u64),
    );
    ctx.push_op(block, expected_const.op_ref());
    let cmp_op = arena_arith::cmp_eq(
        ctx,
        location,
        current_op_idx,
        expected_const.result(ctx),
        i1_ty,
    );
    ctx.push_op(block, cmp_op.op_ref());

    // Build else region: recurse to next arm
    let else_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });
    let else_result = build_nested_dispatch(
        ctx,
        else_block,
        location,
        result_ty,
        current_op_idx,
        suspend_arms,
        arm_index + 1,
        effectful_funcs,
    );
    let else_yield = arena_scf::r#yield(ctx, location, [else_result]);
    ctx.push_op(else_block, else_yield.op_ref());
    let else_region = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });

    let if_op = arena_scf::r#if(
        ctx,
        location,
        cmp_op.result(ctx),
        result_ty,
        then_region,
        else_region,
    );
    ctx.push_op(block, if_op.op_ref());

    if_op.result(ctx)
}

/// Build a single-block region from a handler arm's body region.
pub(crate) fn build_arm_region(
    ctx: &mut IrContext,
    location: Location,
    arm_body: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
) -> RegionRef {
    let blocks = &ctx.region(arm_body).blocks;
    let Some(&arm_block) = blocks.first() else {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let unreachable = arena_func::unreachable(ctx, location);
        ctx.push_op(block, unreachable.op_ref());
        return ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        });
    };

    let step_ty = step_type(ctx);
    let original_ops: Vec<OpRef> = ctx.block(arm_block).ops.to_vec();

    // Collect cast ops to skip
    let mut value_remap: std::collections::HashMap<ValueRef, ValueRef> =
        std::collections::HashMap::new();

    for &op in &original_ops {
        if let Ok(cast) = arena_core::UnrealizedConversionCast::from_op(ctx, op) {
            let cast_input = cast.value(ctx);
            let cast_output = ctx.op_results(op)[0];
            value_remap.insert(cast_output, cast_input);
        }
    }

    // Replace suspend body block args with trampoline ops
    let mut prefix_ops: Vec<OpRef> = Vec::new();
    {
        let ba = ctx.block_args(arm_block).to_vec();
        if !ba.is_empty() {
            let cont_ty = continuation_type(ctx);
            let get_cont = arena_trampoline::get_yield_continuation(ctx, location, cont_ty);
            prefix_ops.push(get_cont.op_ref());
            value_remap.insert(ba[0], get_cont.result(ctx));
        }
        if ba.len() >= 2 {
            let anyref = anyref_type(ctx);
            let get_shift = arena_trampoline::get_yield_shift_value(ctx, location, anyref);
            prefix_ops.push(get_shift.op_ref());
            value_remap.insert(ba[1], get_shift.result(ctx));
        }
    }

    let new_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Add prefix ops
    for op in &prefix_ops {
        ctx.push_op(new_block, *op);
    }

    // Build operations, skipping casts and scf.yield
    let mut last_step_value: Option<ValueRef> = None;

    for &op in &original_ops {
        // Skip all unrealized_conversion_cast
        if arena_core::UnrealizedConversionCast::from_op(ctx, op).is_ok() {
            if let Some(&input) = ctx.op_operands(op).first()
                && is_step_type(ctx, ctx.value_ty(input))
            {
                last_step_value = Some(input);
            }
            continue;
        }

        // Skip existing scf.yield
        if arena_scf::Yield::from_op(ctx, op).is_ok() {
            continue;
        }

        // Track operations that produce Step
        let result_types = ctx.op_result_types(op);
        if !result_types.is_empty() && is_step_type(ctx, result_types[0]) {
            last_step_value = Some(ctx.op_results(op)[0]);
        }

        // Detect effectful function calls
        let is_effectful_call = if let Ok(call) = arena_func::Call::from_op(ctx, op) {
            let callee = call.callee(ctx);
            effectful_funcs.contains(&callee) && !ctx.op_results(op).is_empty()
        } else {
            false
        };

        let is_resume = arena_cont::Resume::from_op(ctx, op).is_ok();
        let produces_step = is_effectful_call || is_resume;

        // Remap operands (follow chains transitively)
        let operands = ctx.op_operands(op).to_vec();
        let remapped_operands: Vec<ValueRef> = operands
            .iter()
            .map(|&v| {
                let mut current = v;
                while let Some(&next) = value_remap.get(&current) {
                    if next == current {
                        break;
                    }
                    current = next;
                }
                current
            })
            .collect();

        let result_types = if produces_step {
            vec![step_ty]
        } else {
            ctx.op_result_types(op).to_vec()
        };

        let needs_rebuild = remapped_operands != operands || produces_step;
        if needs_rebuild {
            let op_data = ctx.op(op);
            let mut builder =
                OperationDataBuilder::new(op_data.location, op_data.dialect, op_data.name)
                    .operands(remapped_operands)
                    .results(result_types);
            for (k, v) in &op_data.attributes {
                builder = builder.attr(*k, v.clone());
            }
            for &r in &op_data.regions {
                builder = builder.region(r);
            }
            let new_data = builder.build(ctx);
            let new_op = ctx.create_op(new_data);
            // Map old result values → new result values
            let old_results = ctx.op_results(op).to_vec();
            let new_results = ctx.op_results(new_op).to_vec();
            for (old_v, new_v) in old_results.iter().zip(new_results.iter()) {
                value_remap.insert(*old_v, *new_v);
            }
            ctx.push_op(new_block, new_op);
            if produces_step {
                last_step_value = Some(ctx.op_results(new_op)[0]);
                break;
            }
        } else {
            ctx.push_op(new_block, op);
        }
    }

    // Add yield for the result
    if let Some(step_val) = last_step_value {
        let yield_op = arena_scf::r#yield(ctx, location, [step_val]);
        ctx.push_op(new_block, yield_op.op_ref());
    } else {
        // Check if last op has results
        let ops = ctx.block(new_block).ops.to_vec();
        if let Some(&last_op) = ops.last() {
            let results = ctx.op_results(last_op);
            if !results.is_empty() {
                let result_value = results[0];
                let step_done = arena_trampoline::step_done(ctx, location, result_value, step_ty);
                ctx.push_op(new_block, step_done.op_ref());
                let yield_op = arena_scf::r#yield(ctx, location, [step_done.result(ctx)]);
                ctx.push_op(new_block, yield_op.op_ref());
            } else {
                let unreachable = arena_func::unreachable(ctx, location);
                ctx.push_op(new_block, unreachable.op_ref());
            }
        } else {
            let unreachable = arena_func::unreachable(ctx, location);
            ctx.push_op(new_block, unreachable.op_ref());
        }
    }

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![new_block],
        parent_op: None,
    })
}
