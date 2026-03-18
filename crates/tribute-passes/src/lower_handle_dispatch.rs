//! Lower `ability.handle_dispatch` to scf.loop with YieldResult dispatch.
//!
//! Transforms `ability.handle_dispatch` into a dispatch loop:
//!
//! ```text
//! %final = scf.loop(%yield_result) -> result_type {
//!   ^bb0(%yr: YieldResult):
//!     if Done → extract value, apply done handler, break
//!     if Shift → check prompt tag
//!       if match → dispatch on op_idx, handler arm returns YieldResult, continue
//!       if no match → break (propagate to outer handler)
//! }
//! ```
//!
//! The body region of `ability.handle_dispatch` uses `cont.done` and
//! `cont.suspend`/`cont.yield` ops to define handler arms, reusing the
//! existing infrastructure from `cont_util`.

use trunk_ir::Symbol;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{adt, arith, core, func, scf};
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, Location};

use tribute_ir::dialect::ability as arena_ability;

use crate::cont_to_yield_bubbling::types::{YieldBubblingTypes, is_yield_result_type};
use crate::cont_util::{SuspendArm, collect_suspend_arms, get_done_region};

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

// ============================================================================
// Loop construction
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn build_dispatch_loop(
    ctx: &mut IrContext,
    location: Location,
    yr_operand: ValueRef,
    tag: u32,
    user_result_ty: TypeRef,
    done_region: Option<RegionRef>,
    suspend_arms: &[SuspendArm],
    types: &YieldBubblingTypes,
) -> OpRef {
    let loop_body = build_loop_body(
        ctx,
        location,
        tag,
        user_result_ty,
        done_region,
        suspend_arms,
        types,
    );
    let loop_op = scf::r#loop(ctx, location, [yr_operand], user_result_ty, loop_body);
    loop_op.op_ref()
}

#[allow(clippy::too_many_arguments)]
fn build_loop_body(
    ctx: &mut IrContext,
    location: Location,
    tag: u32,
    user_result_ty: TypeRef,
    done_region: Option<RegionRef>,
    suspend_arms: &[SuspendArm],
    types: &YieldBubblingTypes,
) -> RegionRef {
    let t = types;

    let block = ctx.create_block(BlockData {
        location,
        args: vec![BlockArgData {
            ty: t.yield_result,
            attrs: Default::default(),
        }],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });
    let current_yr = ctx.block_args(block)[0];

    // Check Done variant.
    let is_done = adt::variant_is(
        ctx,
        location,
        current_yr,
        t.i1,
        t.yield_result,
        Symbol::new("Done"),
    );
    ctx.push_op(block, is_done.op_ref());

    let done_branch = build_done_branch(ctx, location, current_yr, user_result_ty, done_region, t);
    let shift_branch = build_shift_branch(ctx, location, current_yr, tag, suspend_arms, t);

    let nil_ty = core::nil(ctx).as_type_ref();
    let if_op = scf::r#if(
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

// ============================================================================
// Done branch
// ============================================================================

fn build_done_branch(
    ctx: &mut IrContext,
    location: Location,
    current_yr: ValueRef,
    user_result_ty: TypeRef,
    done_region: Option<RegionRef>,
    t: &YieldBubblingTypes,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Extract Done value.
    let get_val = adt::variant_get(
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

    // Apply done handler if present, then break.
    // Use a sentinel break op as insert_before target so inline_done_body
    // inserts ops before the break rather than after it.
    let sentinel_break = scf::r#break(ctx, location, done_value);
    ctx.push_op(block, sentinel_break.op_ref());

    let result = if let Some(done_body) = done_region {
        inline_done_body(
            ctx,
            block,
            location,
            done_body,
            done_value,
            sentinel_break.op_ref(),
        )
    } else {
        done_value
    };

    // Remove sentinel and add final break with correct value.
    ctx.remove_op_from_block(block, sentinel_break.op_ref());
    let result_value = if ctx.value_ty(result) != user_result_ty {
        let cast = core::unrealized_conversion_cast(ctx, location, result, user_result_ty);
        ctx.push_op(block, cast.op_ref());
        cast.result(ctx)
    } else {
        result
    };
    let break_op = scf::r#break(ctx, location, result_value);
    ctx.push_op(block, break_op.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

fn inline_done_body(
    ctx: &mut IrContext,
    dest_block: BlockRef,
    _location: Location,
    done_body: RegionRef,
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

// ============================================================================
// Shift branch
// ============================================================================

fn build_shift_branch(
    ctx: &mut IrContext,
    location: Location,
    current_yr: ValueRef,
    tag: u32,
    suspend_arms: &[SuspendArm],
    t: &YieldBubblingTypes,
) -> RegionRef {
    let block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Extract ShiftInfo.
    let get_info = adt::variant_get(
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

    // Extract prompt from ShiftInfo (field 1).
    let get_prompt = adt::struct_get(ctx, location, shift_info, t.i32, t.shift_info, 1);
    ctx.push_op(block, get_prompt.op_ref());
    let prompt_val = get_prompt.result(ctx);

    // Compare with our tag.
    let our_tag_const = arith::r#const(ctx, location, t.i32, Attribute::Int(tag as i128));
    ctx.push_op(block, our_tag_const.op_ref());
    let tag_matches = arith::cmp_eq(ctx, location, prompt_val, our_tag_const.result(ctx), t.i1);
    ctx.push_op(block, tag_matches.op_ref());

    // Dispatch region (tag matches).
    let dispatch_region = build_dispatch_region(ctx, location, shift_info, suspend_arms, t);

    // Propagate region (tag doesn't match → break with current yr).
    let propagate_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });
    let break_op = scf::r#break(ctx, location, current_yr);
    ctx.push_op(propagate_block, break_op.op_ref());
    let propagate_region = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![propagate_block],
        parent_op: None,
    });

    let nil_ty = core::nil(ctx).as_type_ref();
    let if_op = scf::r#if(
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

fn build_dispatch_region(
    ctx: &mut IrContext,
    location: Location,
    shift_info: ValueRef,
    suspend_arms: &[SuspendArm],
    t: &YieldBubblingTypes,
) -> RegionRef {
    if suspend_arms.is_empty() {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let unreachable = func::unreachable(ctx, location);
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

    // Get op_idx from ShiftInfo (field 2).
    let get_op_idx = adt::struct_get(ctx, location, shift_info, t.i32, t.shift_info, 2);
    ctx.push_op(block, get_op_idx.op_ref());
    let current_op_idx = get_op_idx.result(ctx);

    let final_result = build_nested_dispatch(
        ctx,
        block,
        location,
        shift_info,
        current_op_idx,
        0,
        suspend_arms,
        t,
    );

    let continue_op = scf::r#continue(ctx, location, [final_result]);
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
    arm_index: usize,
    suspend_arms: &[SuspendArm],
    t: &YieldBubblingTypes,
) -> ValueRef {
    let arm = &suspend_arms[arm_index];
    let is_last = arm_index + 1 >= suspend_arms.len();

    let then_region = build_arm_region(ctx, location, arm.body, shift_info, t);

    if is_last {
        // Last arm: unconditional.
        let true_const = arith::r#const(ctx, location, t.i1, Attribute::Int(1));
        ctx.push_op(block, true_const.op_ref());

        let else_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let unreachable = func::unreachable(ctx, location);
        ctx.push_op(else_block, unreachable.op_ref());
        let else_region = ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![else_block],
            parent_op: None,
        });

        let if_op = scf::r#if(
            ctx,
            location,
            true_const.result(ctx),
            t.yield_result,
            then_region,
            else_region,
        );
        ctx.push_op(block, if_op.op_ref());
        return if_op.result(ctx);
    }

    // Compare op_idx.
    let expected_const = arith::r#const(
        ctx,
        location,
        t.i32,
        Attribute::Int(arm.expected_op_idx as i128),
    );
    ctx.push_op(block, expected_const.op_ref());
    let cmp = arith::cmp_eq(
        ctx,
        location,
        current_op_idx,
        expected_const.result(ctx),
        t.i1,
    );
    ctx.push_op(block, cmp.op_ref());

    // Else: recurse to next arm.
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
        arm_index + 1,
        suspend_arms,
        t,
    );
    let else_yield = scf::r#yield(ctx, location, [else_result]);
    ctx.push_op(else_block, else_yield.op_ref());
    let else_region = ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![else_block],
        parent_op: None,
    });

    let if_op = scf::r#if(
        ctx,
        location,
        cmp.result(ctx),
        t.yield_result,
        then_region,
        else_region,
    );
    ctx.push_op(block, if_op.op_ref());
    if_op.result(ctx)
}

// ============================================================================
// Handler arm
// ============================================================================

/// Build a handler arm region.
///
/// In CPS, the continuation (ShiftInfo field 3) is a closure. Handler arms
/// receive it as their first block arg and call it via func.call_indirect.
fn build_arm_region(
    ctx: &mut IrContext,
    location: Location,
    arm_body: RegionRef,
    shift_info: ValueRef,
    t: &YieldBubblingTypes,
) -> RegionRef {
    let blocks = &ctx.region(arm_body).blocks;
    let Some(&arm_block) = blocks.first() else {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        let unreachable = func::unreachable(ctx, location);
        ctx.push_op(block, unreachable.op_ref());
        return ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        });
    };

    let new_block = ctx.create_block(BlockData {
        location,
        args: vec![],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    let mut mapping = IrMapping::new();

    // Map arm block args to ShiftInfo fields:
    // arg[0] → continuation (field 3)
    // arg[1] → shift_value (field 0)
    let ba = ctx.block_args(arm_block).to_vec();
    if !ba.is_empty() {
        let get_cont = adt::struct_get(ctx, location, shift_info, t.anyref, t.shift_info, 3);
        ctx.push_op(new_block, get_cont.op_ref());
        mapping.map_value(ba[0], get_cont.result(ctx));
    }
    if ba.len() >= 2 {
        let get_sv = adt::struct_get(ctx, location, shift_info, t.anyref, t.shift_info, 0);
        ctx.push_op(new_block, get_sv.op_ref());
        mapping.map_value(ba[1], get_sv.result(ctx));
    }

    // Clone arm body ops, collecting the yielded result.
    let original_ops: Vec<OpRef> = ctx.block(arm_block).ops.to_vec();
    let mut last_result: Option<ValueRef> = None;

    for &arm_op in &original_ops {
        // Skip scf.yield — we produce scf.yield ourselves.
        if scf::Yield::matches(ctx, arm_op) {
            let yielded = ctx.op_operands(arm_op).to_vec();
            if let Some(&v) = yielded.first() {
                last_result = Some(mapping.lookup_value_or_default(v));
            }
            continue;
        }

        let cloned = ctx.clone_op_into_block(new_block, arm_op, &mut mapping);
        let results = ctx.op_results(cloned);
        if !results.is_empty() {
            // Track YieldResult-typed results for the loop continue.
            if is_yield_result_type(ctx, ctx.value_ty(results[0])) {
                last_result = Some(results[0]);
            }
        }
    }

    // Yield the result for the dispatch loop.
    let result_val = last_result.unwrap_or_else(|| {
        let null_op = adt::ref_null(ctx, location, t.yield_result, t.yield_result);
        ctx.push_op(new_block, null_op.op_ref());
        null_op.result(ctx)
    });
    let yield_op = scf::r#yield(ctx, location, [result_val]);
    ctx.push_op(new_block, yield_op.op_ref());

    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![new_block],
        parent_op: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::RegionData;
    use trunk_ir::dialect::{cont, core as arena_core};
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::TypeDataBuilder;
    use trunk_ir::{IrContext, OperationDataBuilder, Span};

    use tribute_ir::dialect::tribute_rt;

    fn test_ctx() -> (IrContext, Location) {
        let ctx = IrContext::new();
        let loc = Location::new(PathRef::from_u32(0), Span::default());
        (ctx, loc)
    }

    fn make_module(ctx: &mut IrContext, loc: Location) -> (Module, BlockRef) {
        let module_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![module_block],
            parent_op: None,
        });
        let module_op = OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
            .region(module_region)
            .build(ctx);
        let module_ref = ctx.create_op(module_op);
        (Module::new(ctx, module_ref).unwrap(), module_block)
    }

    fn make_ability_ref_type(ctx: &mut IrContext, name: &str) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::from_dynamic(name)))
                .build(),
        )
    }

    #[test]
    fn test_lower_handle_dispatch_basic() {
        let (mut ctx, loc) = test_ctx();
        let (module, module_block) = make_module(&mut ctx, loc);

        let anyref_ty = tribute_rt::anyref(&mut ctx).as_type_ref();
        let types = YieldBubblingTypes::new(&mut ctx);

        // Build handler body region with:
        //   cont.done { ^bb0(%v): scf.yield %v }
        //   cont.suspend { @State, @get } { ^bb0(%k, %sv): scf.yield %sv }

        // Done handler
        let done_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: anyref_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let done_val = ctx.block_arg(done_block, 0);
        let done_yield = scf::r#yield(&mut ctx, loc, [done_val]);
        ctx.push_op(done_block, done_yield.op_ref());
        let done_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![done_block],
            parent_op: None,
        });
        let done_op = cont::done(&mut ctx, loc, done_body);

        // Suspend handler: returns shift_value directly
        let state_ref = make_ability_ref_type(&mut ctx, "State");
        let suspend_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: anyref_ty,
                    attrs: Default::default(),
                },
                BlockArgData {
                    ty: anyref_ty,
                    attrs: Default::default(),
                },
            ],
            ops: Default::default(),
            parent_region: None,
        });
        let _k_val = ctx.block_arg(suspend_block, 0);
        let sv_val = ctx.block_arg(suspend_block, 1);
        // Wrap return as YieldResult::Done for simplicity
        let done_wrap = adt::variant_new(
            &mut ctx,
            loc,
            [sv_val],
            types.yield_result,
            types.yield_result,
            Symbol::new("Done"),
        );
        ctx.push_op(suspend_block, done_wrap.op_ref());
        let done_wrap_result = done_wrap.result(&ctx);
        let suspend_yield = scf::r#yield(&mut ctx, loc, [done_wrap_result]);
        ctx.push_op(suspend_block, suspend_yield.op_ref());
        let suspend_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![suspend_block],
            parent_op: None,
        });
        let suspend_op = cont::suspend(&mut ctx, loc, state_ref, Symbol::new("get"), suspend_body);

        // Handler body region
        let handler_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        ctx.push_op(handler_block, done_op.op_ref());
        ctx.push_op(handler_block, suspend_op.op_ref());
        let handler_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![handler_block],
            parent_op: None,
        });

        // Build function with ability.handle_dispatch
        let func_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: types.yield_result,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let yr_input = ctx.block_arg(func_entry, 0);

        // Null handler_fn placeholder
        let null_handler = adt::ref_null(&mut ctx, loc, anyref_ty, anyref_ty);
        ctx.push_op(func_entry, null_handler.op_ref());
        let handler_fn_val = null_handler.result(&ctx);

        let dispatch_op = arena_ability::handle_dispatch(
            &mut ctx,
            loc,
            yr_input,
            handler_fn_val,
            anyref_ty,
            42,
            anyref_ty,
            handler_body,
        );
        ctx.push_op(func_entry, dispatch_op.op_ref());

        let dispatch_result = dispatch_op.result(&ctx);
        let ret_op = func::r#return(&mut ctx, loc, [dispatch_result]);
        ctx.push_op(func_entry, ret_op.op_ref());

        let func_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![func_entry],
            parent_op: None,
        });
        let func_ty =
            arena_core::func(&mut ctx, anyref_ty, [types.yield_result], None).as_type_ref();
        let test_func = func::func(&mut ctx, loc, Symbol::new("handler"), func_ty, func_body);
        ctx.push_op(module_block, test_func.op_ref());

        // Run the pass.
        lower_handle_dispatch(&mut ctx, module);

        // Verify: dispatch should be inlined (done body inlined directly, no scf.loop).
        let handler_fn = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let body = handler_fn.body(&ctx);
        let entry = ctx.region(body).blocks[0];
        let ops: Vec<OpRef> = ctx.block(entry).ops.to_vec();

        assert!(
            !ops.iter()
                .any(|&o| arena_ability::HandleDispatch::matches(&ctx, o)),
            "ability.handle_dispatch should be replaced"
        );

        // In the new CPS design, done handler is inlined directly (no loop).
        // The function should end with func.return.
        let last_op = *ops.last().unwrap();
        assert!(
            func::Return::matches(&ctx, last_op),
            "last op should be func.return after done handler inlining"
        );
    }
}
