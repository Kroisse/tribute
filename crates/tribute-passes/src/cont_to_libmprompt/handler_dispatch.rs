//! Handler dispatch lowering for libmprompt backend.
//!
//! Transforms `cont.handler_dispatch` into an `scf.loop` that checks
//! the TLS yield state and dispatches to the appropriate handler arm.
//!
//! Unlike the trampoline backend, libmprompt eliminates the need for:
//! - Step type wrapping/unwrapping
//! - Tag matching (mp_yield reaches the correct prompt directly)
//! - Effectful function tracking
//!
//! The generated loop:
//! ```text
//! scf.loop(%current = %prompt_result) : user_result_ty {
//!   %is_yield = func.call @__tribute_yield_active()
//!   scf.if(%is_yield) {
//!     // Shift path: dispatch by op_idx
//!     %op_idx = func.call @__tribute_get_yield_op_idx()
//!     %k = func.call @__tribute_get_yield_continuation()
//!     %v = func.call @__tribute_get_yield_shift_value()
//!     func.call @__tribute_reset_yield_state()
//!     <nested if-else dispatch by op_idx>
//!     scf.continue(%arm_result)
//!   } else {
//!     // Done path: %current is the normal return value
//!     <done arm body>
//!     scf.break(%done_result)
//!   }
//! }
//! ```

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::arena::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::{
    arith as arena_arith, cont as arena_cont, core as arena_core, func as arena_func,
    scf as arena_scf,
};
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{ArenaRewritePattern, PatternRewriter as ArenaPatternRewriter};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::smallvec::smallvec;

use crate::cont_util::{ArenaSuspendArm, collect_suspend_arms_arena, get_done_region_arena};

/// Pattern: Lower `cont.handler_dispatch` -> `scf.loop` with yield dispatch.
pub(crate) struct LowerHandlerDispatchPattern;

impl ArenaRewritePattern for LowerHandlerDispatchPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch) = arena_cont::HandlerDispatch::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let user_result_ty = dispatch.result_type(ctx);

        let prompt_result = ctx.op_operands(op)[0];
        let body_region = dispatch.body(ctx);

        let suspend_arms = collect_suspend_arms_arena(ctx, body_region);

        let loop_body = build_loop_body(ctx, loc, body_region, &suspend_arms, user_result_ty);

        let loop_op = arena_scf::r#loop(ctx, loc, [prompt_result], user_result_ty, loop_body);
        rewriter.replace_op(loop_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "LowerHandlerDispatchPattern"
    }
}

fn build_loop_body(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::types::Location,
    body_region: RegionRef,
    suspend_arms: &[ArenaSuspendArm],
    user_result_ty: TypeRef,
) -> RegionRef {
    let ptr_ty = arena_core::ptr(ctx).as_type_ref();
    let i1_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
    let nil_ty = arena_core::nil(ctx).as_type_ref();

    // Create loop body block with %current as ptr argument
    let block = ctx.create_block(BlockData {
        location: loc,
        args: vec![BlockArgData {
            ty: ptr_ty,
            attrs: Default::default(),
        }],
        ops: smallvec![],
        parent_region: None,
    });
    let current = ctx.block_args(block)[0];

    // %is_yield = func.call @__tribute_yield_active()
    let is_yield_call =
        arena_func::call(ctx, loc, [], i1_ty, Symbol::new("__tribute_yield_active"));
    ctx.push_op(block, is_yield_call.op_ref());
    let is_yield = is_yield_call.result(ctx);

    // Build shift branch (then -- yield is active)
    let shift_branch = build_shift_branch(ctx, loc, suspend_arms, ptr_ty);

    // Build done branch (else -- normal return)
    let done_branch = build_done_branch(ctx, loc, body_region, current, user_result_ty, ptr_ty);

    // scf.if(%is_yield) { shift } else { done }
    let if_op = arena_scf::r#if(ctx, loc, is_yield, nil_ty, shift_branch, done_branch);
    ctx.push_op(block, if_op.op_ref());

    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    })
}

fn build_done_branch(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::types::Location,
    body_region: RegionRef,
    current: ValueRef,
    user_result_ty: TypeRef,
    ptr_ty: TypeRef,
) -> RegionRef {
    let Some(done_region) = get_done_region_arena(ctx, body_region) else {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let result = cast_if_needed(ctx, block, loc, current, ptr_ty, user_result_ty);
        let brk = arena_scf::r#break(ctx, loc, result);
        ctx.push_op(block, brk.op_ref());
        return ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
    };

    let done_blocks = &ctx.region(done_region).blocks;
    let Some(&done_block) = done_blocks.first() else {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let result = cast_if_needed(ctx, block, loc, current, ptr_ty, user_result_ty);
        let brk = arena_scf::r#break(ctx, loc, result);
        ctx.push_op(block, brk.op_ref());
        return ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
    };

    let new_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // Cast %current to done arg type
    let done_args = ctx.block_args(done_block);
    let done_arg_ty = if !done_args.is_empty() {
        ctx.value_ty(done_args[0])
    } else {
        ptr_ty
    };

    let done_value = cast_if_needed(ctx, new_block, loc, current, ptr_ty, done_arg_ty);

    // Build value remap: done block args -> done_value
    let mut value_remap: HashMap<ValueRef, ValueRef> = HashMap::new();
    let done_block_args = ctx.block_args(done_block).to_vec();
    if !done_block_args.is_empty() {
        value_remap.insert(done_block_args[0], done_value);
    }

    // Copy done block operations, replacing scf.yield with scf.break
    let done_ops: Vec<OpRef> = ctx.block(done_block).ops.clone().to_vec();
    for &done_op in &done_ops {
        if arena_scf::Yield::matches(ctx, done_op) {
            let yielded_operands: Vec<ValueRef> = ctx.op_operands(done_op).to_vec();
            if let Some(&result) = yielded_operands.first() {
                let remapped = value_remap.get(&result).copied().unwrap_or(result);
                let brk = arena_scf::r#break(ctx, loc, remapped);
                ctx.push_op(new_block, brk.op_ref());
            }
            continue;
        }
        // Clone op into the new block with remapping
        clone_op_into_block_with_remap(ctx, new_block, done_op, &value_remap);

        // Map old results -> new results
        let new_ops = ctx.block(new_block).ops.clone();
        if let Some(&new_op) = new_ops.last() {
            let old_results: Vec<ValueRef> = ctx.op_results(done_op).to_vec();
            let new_results: Vec<ValueRef> = ctx.op_results(new_op).to_vec();
            for (old_r, new_r) in old_results.into_iter().zip(new_results) {
                value_remap.insert(old_r, new_r);
            }
        }
    }

    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![new_block],
        parent_op: None,
    })
}

fn build_shift_branch(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::types::Location,
    suspend_arms: &[ArenaSuspendArm],
    ptr_ty: TypeRef,
) -> RegionRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let nil_ty = arena_core::nil(ctx).as_type_ref();

    let block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    // Get yield state from TLS
    let op_idx_call = arena_func::call(
        ctx,
        loc,
        [],
        i32_ty,
        Symbol::new("__tribute_get_yield_op_idx"),
    );
    ctx.push_op(block, op_idx_call.op_ref());
    let op_idx = op_idx_call.result(ctx);

    let k_call = arena_func::call(
        ctx,
        loc,
        [],
        ptr_ty,
        Symbol::new("__tribute_get_yield_continuation"),
    );
    ctx.push_op(block, k_call.op_ref());
    let k = k_call.result(ctx);

    let v_call = arena_func::call(
        ctx,
        loc,
        [],
        ptr_ty,
        Symbol::new("__tribute_get_yield_shift_value"),
    );
    ctx.push_op(block, v_call.op_ref());
    let v = v_call.result(ctx);

    // Reset yield state
    let reset_call = arena_func::call(
        ctx,
        loc,
        [],
        nil_ty,
        Symbol::new("__tribute_reset_yield_state"),
    );
    ctx.push_op(block, reset_call.op_ref());

    // Build nested if-else dispatch
    let arm_result = build_nested_dispatch(ctx, block, loc, ptr_ty, op_idx, k, v, suspend_arms, 0);

    // Continue loop with arm result
    let cont_op = arena_scf::r#continue(ctx, loc, [arm_result]);
    ctx.push_op(block, cont_op.op_ref());

    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_nested_dispatch(
    ctx: &mut IrContext,
    block: trunk_ir::arena::refs::BlockRef,
    loc: trunk_ir::arena::types::Location,
    result_ty: TypeRef,
    current_op_idx: ValueRef,
    k: ValueRef,
    v: ValueRef,
    suspend_arms: &[ArenaSuspendArm],
    arm_index: usize,
) -> ValueRef {
    let i1_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());

    if arm_index >= suspend_arms.len() {
        let dummy = arena_arith::r#const(ctx, loc, result_ty, ArenaAttribute::IntBits(0));
        ctx.push_op(block, dummy.op_ref());
        let unreachable = arena_func::unreachable(ctx, loc);
        ctx.push_op(block, unreachable.op_ref());
        return dummy.result(ctx);
    }

    let arm = &suspend_arms[arm_index];
    let is_last = arm_index + 1 >= suspend_arms.len();

    let then_region = build_arm_region(ctx, loc, arm.body, k, v, result_ty);

    if is_last {
        let true_const = arena_arith::r#const(ctx, loc, i1_ty, ArenaAttribute::IntBits(1));
        ctx.push_op(block, true_const.op_ref());
        let else_region = build_arm_region(ctx, loc, arm.body, k, v, result_ty);
        let if_op = arena_scf::r#if(
            ctx,
            loc,
            true_const.result(ctx),
            result_ty,
            then_region,
            else_region,
        );
        ctx.push_op(block, if_op.op_ref());
        return if_op.result(ctx);
    }

    // Compare op_idx
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let expected = arena_arith::r#const(
        ctx,
        loc,
        i32_ty,
        ArenaAttribute::IntBits(arm.expected_op_idx as u64),
    );
    ctx.push_op(block, expected.op_ref());
    let cmp = arena_arith::cmp_eq(ctx, loc, current_op_idx, expected.result(ctx), i1_ty);
    ctx.push_op(block, cmp.op_ref());

    // Else region: recurse
    let else_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });
    let else_result = build_nested_dispatch(
        ctx,
        else_block,
        loc,
        result_ty,
        current_op_idx,
        k,
        v,
        suspend_arms,
        arm_index + 1,
    );
    let else_yield = arena_scf::r#yield(ctx, loc, [else_result]);
    ctx.push_op(else_block, else_yield.op_ref());
    let else_region = ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![else_block],
        parent_op: None,
    });

    let if_op = arena_scf::r#if(
        ctx,
        loc,
        cmp.result(ctx),
        result_ty,
        then_region,
        else_region,
    );
    ctx.push_op(block, if_op.op_ref());
    if_op.result(ctx)
}

/// Build a single-block region from a suspend arm body.
fn build_arm_region(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::types::Location,
    arm_body: RegionRef,
    k: ValueRef,
    v: ValueRef,
    result_ty: TypeRef,
) -> RegionRef {
    let arm_blocks = &ctx.region(arm_body).blocks;
    let Some(&arm_block) = arm_blocks.first() else {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let unreachable = arena_func::unreachable(ctx, loc);
        ctx.push_op(block, unreachable.op_ref());
        return ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
    };

    // Build value remap: block args -> FFI getter values
    let mut value_remap: HashMap<ValueRef, ValueRef> = HashMap::new();
    let block_args = ctx.block_args(arm_block).to_vec();
    if !block_args.is_empty() {
        value_remap.insert(block_args[0], k);
    }
    if block_args.len() >= 2 {
        value_remap.insert(block_args[1], v);
    }

    // Clone arm block ops into new block, replacing scf.yield
    let new_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    });

    let arm_ops: Vec<OpRef> = ctx.block(arm_block).ops.clone().to_vec();
    let mut has_yield = false;

    for &op in &arm_ops {
        if arena_scf::Yield::matches(ctx, op) {
            let yielded_operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
            if let Some(&result) = yielded_operands.first() {
                let remapped = value_remap.get(&result).copied().unwrap_or(result);
                let y = arena_scf::r#yield(ctx, loc, [remapped]);
                ctx.push_op(new_block, y.op_ref());
                has_yield = true;
            }
            continue;
        }
        clone_op_into_block_with_remap(ctx, new_block, op, &value_remap);

        // Map old results -> new results
        let new_ops = ctx.block(new_block).ops.clone();
        if let Some(&new_op) = new_ops.last() {
            let old_results: Vec<ValueRef> = ctx.op_results(op).to_vec();
            let new_results: Vec<ValueRef> = ctx.op_results(new_op).to_vec();
            for (old_r, new_r) in old_results.into_iter().zip(new_results) {
                value_remap.insert(old_r, new_r);
            }
        }
    }

    if !has_yield {
        let last_ops: Vec<OpRef> = ctx.block(new_block).ops.clone().to_vec();
        if let Some(&last_op) = last_ops.last() {
            let results = ctx.op_results(last_op);
            if !results.is_empty() {
                let result_val = results[0];
                let y = arena_scf::r#yield(ctx, loc, [result_val]);
                ctx.push_op(new_block, y.op_ref());
            } else {
                let dummy = arena_arith::r#const(ctx, loc, result_ty, ArenaAttribute::IntBits(0));
                ctx.push_op(new_block, dummy.op_ref());
                let y = arena_scf::r#yield(ctx, loc, [dummy.result(ctx)]);
                ctx.push_op(new_block, y.op_ref());
            }
        }
    }

    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![new_block],
        parent_op: None,
    })
}

/// Cast a value if types differ, inserting ops into the given block.
fn cast_if_needed(
    ctx: &mut IrContext,
    block: trunk_ir::arena::refs::BlockRef,
    loc: trunk_ir::arena::types::Location,
    value: ValueRef,
    from_ty: TypeRef,
    to_ty: TypeRef,
) -> ValueRef {
    if from_ty == to_ty {
        value
    } else {
        let cast = arena_core::unrealized_conversion_cast(ctx, loc, value, to_ty);
        ctx.push_op(block, cast.op_ref());
        cast.result(ctx)
    }
}

/// Clone an operation into a new block, applying a value remap to operands.
///
/// Results of the cloned operation are added to the remap so that subsequent
/// cloned ops pick them up automatically.
///
/// Nested regions are deep-cloned so that captured values inside them
/// are remapped correctly.
pub(crate) fn clone_op_into_block_with_remap(
    ctx: &mut IrContext,
    dest_block: trunk_ir::arena::refs::BlockRef,
    src_op: OpRef,
    value_remap: &HashMap<ValueRef, ValueRef>,
) {
    use trunk_ir::arena::context::OperationDataBuilder;

    let data = ctx.op(src_op);
    let loc = data.location;
    let dialect = data.dialect;
    let name = data.name;
    let attrs = data.attributes.clone();
    let regions: Vec<_> = data.regions.to_vec();
    let successors: Vec<_> = data.successors.to_vec();
    let operands: Vec<_> = ctx.op_operands(src_op).to_vec();
    let result_types: Vec<_> = ctx.op_result_types(src_op).to_vec();

    let mut builder = OperationDataBuilder::new(loc, dialect, name);
    for &v in &operands {
        let remapped = value_remap.get(&v).copied().unwrap_or(v);
        builder = builder.operand(remapped);
    }
    for &ty in &result_types {
        builder = builder.result(ty);
    }
    for (k, v) in attrs {
        builder = builder.attr(k, v);
    }
    for r in regions {
        let cloned_region = deep_clone_region(ctx, r, value_remap);
        builder = builder.region(cloned_region);
    }
    for s in successors {
        builder = builder.successor(s);
    }

    let data = builder.build(ctx);
    let new_op = ctx.create_op(data);
    ctx.push_op(dest_block, new_op);
}

/// Deep-clone a region, recursively remapping values in all nested ops.
fn deep_clone_region(
    ctx: &mut IrContext,
    region: RegionRef,
    parent_remap: &HashMap<ValueRef, ValueRef>,
) -> RegionRef {
    let region_data = ctx.region(region);
    let loc = region_data.location;
    let src_blocks: Vec<_> = region_data.blocks.to_vec();

    let mut remap = parent_remap.clone();
    let mut new_blocks = Vec::with_capacity(src_blocks.len());

    for &src_block in &src_blocks {
        let src_args = ctx.block_args(src_block).to_vec();
        let arg_data: Vec<BlockArgData> = src_args
            .iter()
            .map(|&v| BlockArgData {
                ty: ctx.value_ty(v),
                attrs: Default::default(),
            })
            .collect();

        let new_block = ctx.create_block(BlockData {
            location: ctx.block(src_block).location,
            args: arg_data,
            ops: smallvec![],
            parent_region: None,
        });

        // Map old block args -> new block args
        let new_args = ctx.block_args(new_block).to_vec();
        for (old_arg, new_arg) in src_args.into_iter().zip(new_args) {
            remap.insert(old_arg, new_arg);
        }

        new_blocks.push((src_block, new_block));
    }

    // Clone ops in each block
    for &(src_block, new_block) in &new_blocks {
        let src_ops: Vec<OpRef> = ctx.block(src_block).ops.clone().to_vec();
        for &op in &src_ops {
            clone_op_into_block_with_remap(ctx, new_block, op, &remap);

            // Map old results -> new results
            let new_ops_list = ctx.block(new_block).ops.clone();
            if let Some(&new_op) = new_ops_list.last() {
                let old_results: Vec<ValueRef> = ctx.op_results(op).to_vec();
                let new_results: Vec<ValueRef> = ctx.op_results(new_op).to_vec();
                for (old_r, new_r) in old_results.into_iter().zip(new_results) {
                    remap.insert(old_r, new_r);
                }
            }
        }
    }

    let block_refs: Vec<_> = new_blocks.into_iter().map(|(_, b)| b).collect();
    ctx.create_region(RegionData {
        location: loc,
        blocks: block_refs.into(),
        parent_op: None,
    })
}
