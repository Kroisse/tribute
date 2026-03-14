//! Lower `cont.handler_dispatch` to scf.loop with YieldResult dispatch.
//!
//! Transforms handler_dispatch into:
//! ```text
//! %final = scf.loop(%initial_yr) -> YieldResult {
//!   ^bb0(%yr: YieldResult):
//!     %is_done = adt.variant_is(@YieldResult, "Done", %yr)
//!     scf.if %is_done {
//!       // Done arm
//!       scf.break %done_result
//!     } else {
//!       // Shift arm: check prompt tag
//!       scf.if %matches {
//!         // Our handler → dispatch on op_idx
//!         scf.continue %arm_result
//!       } else {
//!         // Propagate to outer handler
//!         scf.break %yr
//!       }
//!     }
//! }
//! ```

use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, OperationDataBuilder, RegionData};
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::arith;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::location::Span;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::types::{Attribute, Location};

use super::types::{YieldBubblingTypes, is_yield_result_type};
use crate::cont_util::{SuspendArm, collect_suspend_arms};

// ============================================================================
// Pattern: Lower cont.handler_dispatch
// ============================================================================

pub(crate) struct LowerHandlerDispatchPattern {
    pub(crate) types: YieldBubblingTypes,
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
    pub(crate) handlers_in_effectful_funcs: Rc<HashSet<Span>>,
}

impl RewritePattern for LowerHandlerDispatchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch) = arena_cont::HandlerDispatch::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;

        // Get operands: [0] = YieldResult, [1] = runtime prompt tag (optional,
        // added by LowerPushPromptPattern when the handler_dispatch is paired
        // with a push_prompt).
        let operands = ctx.op_operands(op).to_vec();
        let yr_operand = operands[0];
        let runtime_tag_operand = operands.get(1).copied();

        let is_in_effectful_func = self.handlers_in_effectful_funcs.contains(&location.span);

        let loop_op = build_handler_dispatch_loop(
            ctx,
            &dispatch,
            op,
            yr_operand,
            runtime_tag_operand,
            is_in_effectful_func,
            &self.types,
            &self.effectful_funcs,
        );

        rewriter.replace_op(loop_op);
        true
    }
}

/// Build the scf.loop that implements handler dispatch.
///
/// This is called both by `LowerHandlerDispatchPattern` (for standalone handler_dispatch)
/// and by `LowerPushPromptPattern` (for handler_dispatch inside push_prompt, with runtime tag).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_handler_dispatch_loop(
    ctx: &mut IrContext,
    dispatch: &arena_cont::HandlerDispatch,
    op: OpRef,
    yr_operand: ValueRef,
    runtime_tag_operand: Option<ValueRef>,
    is_in_effectful_func: bool,
    types: &YieldBubblingTypes,
    effectful_funcs: &Rc<HashSet<Symbol>>,
) -> OpRef {
    let location = ctx.op(op).location;
    let compile_time_tag = dispatch.tag(ctx);
    let user_result_ty = dispatch.result_type(ctx);
    let body_region = dispatch.body(ctx);
    let suspend_arms = collect_suspend_arms(ctx, body_region);
    tracing::debug!(
        "build_handler_dispatch_loop: compile_tag={}, runtime_tag={}, arms={}",
        compile_time_tag,
        runtime_tag_operand.is_some(),
        suspend_arms.len(),
    );

    let loop_result_ty = if is_in_effectful_func {
        types.yield_result
    } else {
        user_result_ty
    };

    let loop_body = build_loop_body(
        ctx,
        location,
        compile_time_tag,
        &suspend_arms,
        user_result_ty,
        is_in_effectful_func,
        runtime_tag_operand,
        types,
        effectful_funcs,
    );

    let loop_op = arena_scf::r#loop(ctx, location, [yr_operand], loop_result_ty, loop_body);
    loop_op.op_ref()
}

#[allow(clippy::too_many_arguments)]
fn build_loop_body(
    ctx: &mut IrContext,
    location: Location,
    our_tag: u32,
    suspend_arms: &[SuspendArm],
    user_result_ty: TypeRef,
    is_in_effectful_func: bool,
    runtime_tag_operand: Option<ValueRef>,
    types: &YieldBubblingTypes,
    effectful_funcs: &HashSet<Symbol>,
) -> RegionRef {
    let t = types;

    // Create block with current YieldResult as argument
    let block = ctx.create_block(BlockData {
        location,
        args: vec![trunk_ir::context::BlockArgData {
            ty: t.yield_result,
            attrs: Default::default(),
        }],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });
    let current_yr = ctx.block_args(block)[0];

    // Check if Done variant
    let is_done = arena_adt::variant_is(
        ctx,
        location,
        current_yr,
        t.i1,
        t.yield_result,
        Symbol::new("Done"),
    );
    ctx.push_op(block, is_done.op_ref());

    // Build Done branch
    let done_branch = build_done_branch(
        ctx,
        location,
        current_yr,
        user_result_ty,
        is_in_effectful_func,
        types,
    );

    // Build Shift branch
    let shift_branch = build_shift_branch(
        ctx,
        location,
        our_tag,
        current_yr,
        suspend_arms,
        runtime_tag_operand,
        types,
        effectful_funcs,
    );

    // Result type for the if: void (both branches break/continue the loop)
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

fn build_done_branch(
    ctx: &mut IrContext,
    location: Location,
    current_yr: ValueRef,
    user_result_ty: TypeRef,
    is_in_effectful_func: bool,
    types: &YieldBubblingTypes,
) -> RegionRef {
    let t = types;
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Extract value: adt.variant_get(@YieldResult, "Done", 0, %yr)
    let get_val = arena_adt::variant_get(
        ctx,
        location,
        current_yr,
        t.anyref,
        t.yield_result,
        Symbol::new("Done"),
        0,
    );
    ctx.push_op(block, get_val.op_ref());
    let done_value = get_val.result(ctx);

    if is_in_effectful_func {
        // Re-wrap as YieldResult::Done and break
        let rewrap = arena_adt::variant_new(
            ctx,
            location,
            [done_value],
            t.yield_result,
            t.yield_result,
            Symbol::new("Done"),
        );
        ctx.push_op(block, rewrap.op_ref());
        let break_op = arena_scf::r#break(ctx, location, rewrap.result(ctx));
        ctx.push_op(block, break_op.op_ref());
    } else {
        // Cast anyref to user result type
        let result_value = if t.anyref != user_result_ty {
            let cast =
                arena_core::unrealized_conversion_cast(ctx, location, done_value, user_result_ty);
            ctx.push_op(block, cast.op_ref());
            cast.result(ctx)
        } else {
            done_value
        };
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
    ctx: &mut IrContext,
    location: Location,
    our_tag: u32,
    current_yr: ValueRef,
    suspend_arms: &[SuspendArm],
    runtime_tag_operand: Option<ValueRef>,
    types: &YieldBubblingTypes,
    effectful_funcs: &HashSet<Symbol>,
) -> RegionRef {
    let t = types;
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Extract ShiftInfo from YieldResult::Shift
    let get_info = arena_adt::variant_get(
        ctx,
        location,
        current_yr,
        t.shift_info,
        t.yield_result,
        Symbol::new("Shift"),
        0,
    );
    ctx.push_op(block, get_info.op_ref());
    let shift_info = get_info.result(ctx);

    // Extract prompt from ShiftInfo (field 1)
    let get_prompt = arena_adt::struct_get(ctx, location, shift_info, t.i32, t.shift_info, 1);
    ctx.push_op(block, get_prompt.op_ref());
    let prompt_val = get_prompt.result(ctx);

    // Compare with our handler's tag (prefer runtime tag over compile-time constant)
    let our_tag_val = if let Some(rt_tag) = runtime_tag_operand {
        rt_tag
    } else {
        let our_tag_const = arith::r#const(ctx, location, t.i32, Attribute::Int(our_tag as i128));
        ctx.push_op(block, our_tag_const.op_ref());
        our_tag_const.result(ctx)
    };
    let tag_matches = arith::cmp_eq(ctx, location, prompt_val, our_tag_val, t.i1);
    ctx.push_op(block, tag_matches.op_ref());

    // Build dispatch region (when tag matches)
    let dispatch_region =
        build_dispatch_region(ctx, location, shift_info, suspend_arms, t, effectful_funcs);

    // Build propagate region (when tag doesn't match → break with current yr)
    let propagate_region = {
        let pb = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let break_op = arena_scf::r#break(ctx, location, current_yr);
        ctx.push_op(pb, break_op.op_ref());
        ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![pb],
            parent_op: None,
        })
    };

    // scf.if for tag match
    let nil_ty = arena_core::nil(ctx).as_type_ref();
    let if_op = arena_scf::r#if(
        ctx,
        location,
        tag_matches.result(ctx),
        nil_ty,
        dispatch_region,
        propagate_region,
    );
    ctx.push_op(block, if_op.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// Build dispatch region for suspend arms using nested scf.if.
fn build_dispatch_region(
    ctx: &mut IrContext,
    location: Location,
    shift_info: ValueRef,
    suspend_arms: &[SuspendArm],
    types: &YieldBubblingTypes,
    effectful_funcs: &HashSet<Symbol>,
) -> RegionRef {
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

    // Get op_idx from ShiftInfo (field 2)
    let get_op_idx =
        arena_adt::struct_get(ctx, location, shift_info, types.i32, types.shift_info, 2);
    ctx.push_op(block, get_op_idx.op_ref());
    let current_op_idx = get_op_idx.result(ctx);

    // Build nested if-else dispatch
    let final_result = build_nested_dispatch(
        ctx,
        block,
        location,
        shift_info,
        current_op_idx,
        suspend_arms,
        0,
        types,
        effectful_funcs,
    );

    // Continue loop with result
    let continue_op = arena_scf::r#continue(ctx, location, [final_result]);
    ctx.push_op(block, continue_op.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_nested_dispatch(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    shift_info: ValueRef,
    current_op_idx: ValueRef,
    suspend_arms: &[SuspendArm],
    arm_index: usize,
    types: &YieldBubblingTypes,
    effectful_funcs: &HashSet<Symbol>,
) -> ValueRef {
    if arm_index >= suspend_arms.len() {
        panic!("build_nested_dispatch: arm_index out of bounds");
    }

    let arm = &suspend_arms[arm_index];
    let is_last_arm = arm_index + 1 >= suspend_arms.len();

    // Build then region (handler arm)
    let then_region = build_arm_region(ctx, location, arm.body, shift_info, types, effectful_funcs);

    if is_last_arm {
        // Last arm: unconditional
        let true_const = arith::r#const(ctx, location, types.i1, Attribute::Int(1));
        ctx.push_op(block, true_const.op_ref());
        let else_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let unreachable = arena_func::unreachable(ctx, location);
        ctx.push_op(else_block, unreachable.op_ref());
        let else_region = ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![else_block],
            parent_op: None,
        });

        let if_op = arena_scf::r#if(
            ctx,
            location,
            true_const.result(ctx),
            types.yield_result,
            then_region,
            else_region,
        );
        ctx.push_op(block, if_op.op_ref());
        return if_op.result(ctx);
    }

    // Compare current op_idx with expected
    let expected_const = arith::r#const(
        ctx,
        location,
        types.i32,
        Attribute::Int(arm.expected_op_idx as i128),
    );
    ctx.push_op(block, expected_const.op_ref());
    let cmp_op = arith::cmp_eq(
        ctx,
        location,
        current_op_idx,
        expected_const.result(ctx),
        types.i1,
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
        shift_info,
        current_op_idx,
        suspend_arms,
        arm_index + 1,
        types,
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
        types.yield_result,
        then_region,
        else_region,
    );
    ctx.push_op(block, if_op.op_ref());

    if_op.result(ctx)
}

/// Build a handler arm region.
///
/// The arm body's block args become:
/// - arg[0] → continuation (from ShiftInfo field 3)
/// - arg[1] → shift_value (from ShiftInfo field 0)
///
/// Operations are cloned with value remapping. cont.resume is lowered inline.
fn build_arm_region(
    ctx: &mut IrContext,
    location: Location,
    arm_body: RegionRef,
    shift_info: ValueRef,
    types: &YieldBubblingTypes,
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

    let original_ops: Vec<OpRef> = ctx.block(arm_block).ops.to_vec();
    let mut value_remap: std::collections::HashMap<ValueRef, ValueRef> =
        std::collections::HashMap::new();

    // Map block args to ShiftInfo fields
    let ba = ctx.block_args(arm_block).to_vec();
    let block_arg_set: std::collections::HashSet<ValueRef> = ba.iter().copied().collect();

    // Only skip casts that directly convert block args (continuation, shift_value).
    // Casts added by LowerResumePattern (e.g., i32→anyref, wrapper→anyref) must be preserved.
    for &op in &original_ops {
        if let Ok(cast) = arena_core::UnrealizedConversionCast::from_op(ctx, op) {
            let cast_input = cast.value(ctx);
            if block_arg_set.contains(&cast_input) {
                let cast_output = ctx.op_results(op)[0];
                value_remap.insert(cast_output, cast_input);
            }
        }
    }
    let new_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // arg[0] → continuation (ShiftInfo field 3)
    if !ba.is_empty() {
        let get_cont =
            arena_adt::struct_get(ctx, location, shift_info, types.anyref, types.shift_info, 3);
        ctx.push_op(new_block, get_cont.op_ref());
        value_remap.insert(ba[0], get_cont.result(ctx));
    }
    // arg[1] → shift_value (ShiftInfo field 0)
    if ba.len() >= 2 {
        let get_sv =
            arena_adt::struct_get(ctx, location, shift_info, types.anyref, types.shift_info, 0);
        ctx.push_op(new_block, get_sv.op_ref());
        value_remap.insert(ba[1], get_sv.result(ctx));
    }

    // Clone operations with remapping
    let mut last_yr_value: Option<ValueRef> = None;

    // Track which casts were recorded as block-arg casts (to skip in clone loop)
    let skipped_cast_ops: std::collections::HashSet<OpRef> = original_ops
        .iter()
        .filter(|&&op| {
            if let Ok(cast) = arena_core::UnrealizedConversionCast::from_op(ctx, op) {
                block_arg_set.contains(&cast.value(ctx))
            } else {
                false
            }
        })
        .copied()
        .collect();

    for &op in &original_ops {
        // Only skip block-arg casts (not casts added by LowerResumePattern)
        if skipped_cast_ops.contains(&op) {
            continue;
        }

        // Skip existing scf.yield
        if arena_scf::Yield::from_op(ctx, op).is_ok() {
            continue;
        }

        // Track operations that produce YieldResult
        let result_types = ctx.op_result_types(op);
        if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
            last_yr_value = Some(ctx.op_results(op)[0]);
        }

        // Detect effectful function calls
        let is_effectful_call = if let Ok(call) = arena_func::Call::from_op(ctx, op) {
            let callee = call.callee(ctx);
            effectful_funcs.contains(&callee) && !ctx.op_results(op).is_empty()
        } else {
            false
        };

        let is_resume = arena_cont::Resume::from_op(ctx, op).is_ok();
        let produces_yr = is_effectful_call || is_resume;

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

        let result_types = if produces_yr {
            vec![types.yield_result]
        } else {
            ctx.op_result_types(op).to_vec()
        };

        let needs_rebuild = remapped_operands != operands || produces_yr;
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
            let old_results = ctx.op_results(op).to_vec();
            let new_results = ctx.op_results(new_op).to_vec();
            for (old_v, new_v) in old_results.iter().zip(new_results.iter()) {
                value_remap.insert(*old_v, *new_v);
            }
            ctx.push_op(new_block, new_op);
            if produces_yr {
                last_yr_value = Some(ctx.op_results(new_op)[0]);
                break;
            }
        } else {
            ctx.detach_op(op);
            ctx.push_op(new_block, op);
        }
    }

    // Add yield for the result (remap through value_remap to avoid stale refs)
    if let Some(mut yr_val) = last_yr_value {
        while let Some(&next) = value_remap.get(&yr_val) {
            if next == yr_val {
                break;
            }
            yr_val = next;
        }
        let yield_op = arena_scf::r#yield(ctx, location, [yr_val]);
        ctx.push_op(new_block, yield_op.op_ref());
    } else {
        // Check if last op has results and wrap in YieldResult::Done
        let ops = ctx.block(new_block).ops.to_vec();
        if let Some(&last_op) = ops.last() {
            let results = ctx.op_results(last_op);
            if !results.is_empty() {
                let result_value = results[0];
                let anyref_val = arena_core::unrealized_conversion_cast(
                    ctx,
                    location,
                    result_value,
                    types.anyref,
                );
                ctx.push_op(new_block, anyref_val.op_ref());
                let done_op = arena_adt::variant_new(
                    ctx,
                    location,
                    [anyref_val.result(ctx)],
                    types.yield_result,
                    types.yield_result,
                    Symbol::new("Done"),
                );
                ctx.push_op(new_block, done_op.op_ref());
                let yield_op = arena_scf::r#yield(ctx, location, [done_op.result(ctx)]);
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
