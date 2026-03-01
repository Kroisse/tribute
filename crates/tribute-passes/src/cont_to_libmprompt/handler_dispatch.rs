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

use trunk_ir::arena::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::{
    arith as arena_arith, cont as arena_cont, core as arena_core, func as arena_func,
    scf as arena_scf,
};
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{ArenaRewritePattern, PatternRewriter as ArenaPatternRewriter};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{arith, cont, scf};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::smallvec::smallvec;
use trunk_ir::{
    Block, BlockArg, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type,
    Value,
};

use crate::cont_util::{
    ArenaSuspendArm, SuspendArm, collect_suspend_arms, collect_suspend_arms_arena, get_done_region,
    get_done_region_arena, rebuild_op_with_remap, remap_value,
};

// ============================================================================
// Pattern: Lower cont.handler_dispatch
// ============================================================================

pub(crate) struct LowerHandlerDispatchPattern;

impl<'db> RewritePattern<'db> for LowerHandlerDispatchPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(dispatch) = cont::HandlerDispatch::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let user_result_ty = dispatch.result_type(db);

        // Get the prompt result operand (from push_prompt's __tribute_prompt call)
        let prompt_result = rewriter.operand(0).unwrap();

        // Get the body region with child ops (cont.done + cont.suspend)
        let body_region = dispatch.body(db);

        // Collect suspend arms with their expected op_idx
        let suspend_arms = collect_suspend_arms(db, &body_region);

        // Build the loop body
        let loop_body = build_loop_body(db, location, &body_region, &suspend_arms, user_result_ty);

        // Create scf.loop with prompt_result as initial value
        let loop_op = scf::r#loop(db, location, vec![prompt_result], user_result_ty, loop_body);

        let ops = [loop_op.as_operation()];
        let last = *ops.last().unwrap();
        for op in &ops[..ops.len() - 1] {
            rewriter.insert_op(*op);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Loop body construction
// ============================================================================

/// Build the loop body region.
///
/// The loop receives `%current` as a block argument (ptr type — the prompt result).
/// It checks `__tribute_yield_active()` to decide between done and shift paths.
fn build_loop_body<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    body_region: &Region<'db>,
    suspend_arms: &[SuspendArm<'db>],
    user_result_ty: Type<'db>,
) -> Region<'db> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i1_ty = core::I::<1>::new(db).as_type();
    let nil_ty = core::Nil::new(db).as_type();

    // Create block with %current as argument
    let block_id = trunk_ir::BlockId::fresh();
    let current_arg = BlockArg::of_type(db, ptr_ty);
    let current = Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);

    let mut builder = BlockBuilder::new(db, location);

    // %is_yield = func.call @__tribute_yield_active()
    let is_yield_call = builder.op(func::call(
        db,
        location,
        vec![],
        i1_ty,
        Symbol::new("__tribute_yield_active"),
    ));
    let is_yield = is_yield_call.result(db);

    // Build shift branch (then — yield is active)
    let shift_branch = build_shift_branch(db, location, suspend_arms, ptr_ty);

    // Build done branch (else — normal return)
    let done_branch = build_done_branch(db, location, body_region, current, user_result_ty);

    // scf.if(%is_yield) { shift } else { done }
    // Both branches use scf.break/scf.continue, so result type is nil
    builder.op(scf::r#if(
        db,
        location,
        is_yield,
        nil_ty,
        shift_branch,
        done_branch,
    ));

    let body_block = Block::new(
        db,
        block_id,
        location,
        IdVec::from(vec![current_arg]),
        builder.build().operations(db).clone(),
    );

    Region::new(db, location, IdVec::from(vec![body_block]))
}

/// Build the done branch (yield_active == false).
///
/// The done arm receives the done value from %current (the loop argument).
/// It executes the done arm body and breaks with the result.
fn build_done_branch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    body_region: &Region<'db>,
    current: Value<'db>,
    user_result_ty: Type<'db>,
) -> Region<'db> {
    let ptr_ty = core::Ptr::new(db).as_type();

    let Some(done_region) = get_done_region(db, body_region) else {
        // No done arm — just break with current value cast to user type
        let mut builder = BlockBuilder::new(db, location);
        let result = cast_if_needed(db, &mut builder, location, current, ptr_ty, user_result_ty);
        builder.op(scf::r#break(db, location, result));
        return Region::new(db, location, IdVec::from(vec![builder.build()]));
    };

    let done_blocks = done_region.blocks(db);
    let Some(done_block) = done_blocks.first() else {
        let mut builder = BlockBuilder::new(db, location);
        let result = cast_if_needed(db, &mut builder, location, current, ptr_ty, user_result_ty);
        builder.op(scf::r#break(db, location, result));
        return Region::new(db, location, IdVec::from(vec![builder.build()]));
    };

    // Build using Vec<Operation> + Block::new (to handle raw remapped ops)
    let mut ops: Vec<Operation<'db>> = Vec::new();

    // Cast %current to done arg type
    let done_arg_ty = done_block
        .args(db)
        .first()
        .map(|a| a.ty(db))
        .unwrap_or(ptr_ty);
    let done_value = if ptr_ty != done_arg_ty {
        let cast = core::unrealized_conversion_cast(db, location, current, done_arg_ty);
        ops.push(cast.as_operation());
        cast.as_operation().result(db, 0)
    } else {
        current
    };

    // Build value remap: done block arg → done_value
    let mut value_remap: HashMap<Value<'db>, Value<'db>> = HashMap::new();
    if !done_block.args(db).is_empty() {
        let orig_arg = Value::new(db, trunk_ir::ValueDef::BlockArg(done_block.id(db)), 0);
        value_remap.insert(orig_arg, done_value);
    }

    // Copy done block operations with remapping
    for op in done_block.operations(db).iter() {
        // Skip scf.yield — we'll add scf.break instead
        if scf::Yield::from_operation(db, *op).is_ok() {
            let yielded = op
                .operands(db)
                .first()
                .copied()
                .map(|v| remap_value(v, &value_remap));
            if let Some(result) = yielded {
                ops.push(scf::r#break(db, location, result).as_operation());
            }
            continue;
        }

        let remapped_op = rebuild_op_with_remap(db, op, &value_remap);
        if remapped_op != *op {
            for (i, _) in op.results(db).iter().enumerate() {
                value_remap.insert(op.result(db, i), remapped_op.result(db, i));
            }
        }
        ops.push(remapped_op);
    }

    let block = Block::new(
        db,
        trunk_ir::BlockId::fresh(),
        location,
        IdVec::new(),
        IdVec::from(ops),
    );
    Region::new(db, location, IdVec::from(vec![block]))
}

/// Build the shift branch (yield_active == true).
///
/// Calls FFI getters, resets yield state, then dispatches by op_idx.
fn build_shift_branch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    suspend_arms: &[SuspendArm<'db>],
    ptr_ty: Type<'db>,
) -> Region<'db> {
    let i32_ty = core::I32::new(db).as_type();
    let nil_ty = core::Nil::new(db).as_type();

    let mut builder = BlockBuilder::new(db, location);

    // Get yield state from TLS
    let op_idx_call = builder.op(func::call(
        db,
        location,
        vec![],
        i32_ty,
        Symbol::new("__tribute_get_yield_op_idx"),
    ));
    let op_idx = op_idx_call.result(db);

    let k_call = builder.op(func::call(
        db,
        location,
        vec![],
        ptr_ty,
        Symbol::new("__tribute_get_yield_continuation"),
    ));
    let k = k_call.result(db);

    let v_call = builder.op(func::call(
        db,
        location,
        vec![],
        ptr_ty,
        Symbol::new("__tribute_get_yield_shift_value"),
    ));
    let v = v_call.result(db);

    // Reset yield state
    builder.op(func::call(
        db,
        location,
        vec![],
        nil_ty,
        Symbol::new("__tribute_reset_yield_state"),
    ));

    // Build nested if-else dispatch by op_idx
    let arm_result = build_nested_dispatch(
        db,
        &mut builder,
        location,
        ptr_ty,
        op_idx,
        k,
        v,
        suspend_arms,
        0,
    );

    // Continue loop with arm result
    builder.op(scf::r#continue(db, location, vec![arm_result]));

    Region::new(db, location, IdVec::from(vec![builder.build()]))
}

// ============================================================================
// Nested dispatch
// ============================================================================

/// Build nested if-else dispatch for suspend arms by op_idx.
///
/// Last arm becomes the default else case.
#[allow(clippy::too_many_arguments)]
fn build_nested_dispatch<'db>(
    db: &'db dyn salsa::Database,
    builder: &mut BlockBuilder<'db>,
    location: Location<'db>,
    result_ty: Type<'db>,
    current_op_idx: Value<'db>,
    k: Value<'db>,
    v: Value<'db>,
    suspend_arms: &[SuspendArm<'db>],
    arm_index: usize,
) -> Value<'db> {
    let i1_ty = core::I::<1>::new(db).as_type();

    if arm_index >= suspend_arms.len() {
        // Create a dummy value first — needed for SSA even in unreachable code
        let dummy = builder.op(arith::r#const(
            db,
            location,
            result_ty,
            trunk_ir::Attribute::IntBits(0),
        ));
        // func.unreachable is a block terminator, so it must come last
        builder.op(func::unreachable(db, location));
        return dummy.result(db);
    }

    let arm = &suspend_arms[arm_index];
    let is_last = arm_index + 1 >= suspend_arms.len();

    // Build then region: execute this arm
    let then_region = build_arm_region(db, location, &arm.body, k, v, result_ty);

    if is_last {
        // Last arm: use always-true condition
        let true_const = builder.op(arith::r#const(db, location, i1_ty, true.into()));
        let else_region = build_arm_region(db, location, &arm.body, k, v, result_ty);
        let if_op = builder.op(scf::r#if(
            db,
            location,
            true_const.result(db),
            result_ty,
            then_region,
            else_region,
        ));
        return if_op.result(db);
    }

    // Not last: compare op_idx
    let expected = builder.op(arith::Const::i32(db, location, arm.expected_op_idx as i32));
    let cmp = builder.op(arith::cmp_eq(
        db,
        location,
        current_op_idx,
        expected.result(db),
        i1_ty,
    ));

    // Else region: recurse
    let mut else_builder = BlockBuilder::new(db, location);
    let else_result = build_nested_dispatch(
        db,
        &mut else_builder,
        location,
        result_ty,
        current_op_idx,
        k,
        v,
        suspend_arms,
        arm_index + 1,
    );
    else_builder.op(scf::r#yield(db, location, vec![else_result]));
    let else_region = Region::new(db, location, IdVec::from(vec![else_builder.build()]));

    let if_op = builder.op(scf::r#if(
        db,
        location,
        cmp.result(db),
        result_ty,
        then_region,
        else_region,
    ));

    if_op.result(db)
}

/// Build a single-block region from a suspend arm body.
///
/// The arm body's entry block has block arguments (%k: continuation, %v: shift_value).
/// These are replaced by the FFI getter results.
fn build_arm_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    arm_body: &Region<'db>,
    k: Value<'db>,
    v: Value<'db>,
    result_ty: Type<'db>,
) -> Region<'db> {
    let Some(arm_block) = arm_body.blocks(db).first() else {
        let mut b = BlockBuilder::new(db, location);
        b.op(func::unreachable(db, location));
        return Region::new(db, location, IdVec::from(vec![b.build()]));
    };

    // Build value remap: block args → FFI getter values
    let mut value_remap: HashMap<Value<'db>, Value<'db>> = HashMap::new();
    let block_id = arm_block.id(db);
    let ba = arm_block.args(db);

    // First block arg = continuation → k
    if !ba.is_empty() {
        let orig_k = Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);
        value_remap.insert(orig_k, k);
    }
    // Second block arg = shift_value → v
    if ba.len() >= 2 {
        let orig_v = Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 1);
        value_remap.insert(orig_v, v);
    }

    // Use Vec<Operation> + Block::new pattern for raw remapped ops
    let mut ops: Vec<Operation<'db>> = Vec::new();
    let mut has_yield = false;

    let arm_ops = arm_block.operations(db);
    for op in arm_ops.iter() {
        // Skip scf.yield — we'll add our own
        if scf::Yield::from_operation(db, *op).is_ok() {
            let yielded = op
                .operands(db)
                .first()
                .copied()
                .map(|val| remap_value(val, &value_remap));
            if let Some(result) = yielded {
                ops.push(scf::r#yield(db, location, vec![result]).as_operation());
                has_yield = true;
            }
            continue;
        }

        let remapped_op = rebuild_op_with_remap(db, op, &value_remap);
        if remapped_op != *op {
            for (i, _) in op.results(db).iter().enumerate() {
                value_remap.insert(op.result(db, i), remapped_op.result(db, i));
            }
        }
        ops.push(remapped_op);
    }

    // If no scf.yield was encountered, add one with the last result
    if !has_yield && let Some(last_op) = ops.last().copied() {
        if !last_op.results(db).is_empty() {
            let result_val = last_op.result(db, 0);
            ops.push(scf::r#yield(db, location, vec![result_val]).as_operation());
        } else {
            let dummy = arith::r#const(db, location, result_ty, trunk_ir::Attribute::IntBits(0));
            ops.push(dummy.as_operation());
            ops.push(scf::r#yield(db, location, vec![dummy.result(db)]).as_operation());
        }
    }

    let block = Block::new(
        db,
        trunk_ir::BlockId::fresh(),
        location,
        IdVec::new(),
        IdVec::from(ops),
    );
    Region::new(db, location, IdVec::from(vec![block]))
}

/// Cast a value to a target type if different from current type.
fn cast_if_needed<'db>(
    db: &'db dyn salsa::Database,
    builder: &mut BlockBuilder<'db>,
    location: Location<'db>,
    value: Value<'db>,
    from_ty: Type<'db>,
    to_ty: Type<'db>,
) -> Value<'db> {
    if from_ty == to_ty {
        value
    } else {
        let cast = builder.op(core::unrealized_conversion_cast(db, location, value, to_ty));
        cast.result(db)
    }
}

// ============================================================================
// Arena version
// ============================================================================

/// Arena pattern: Lower `cont.handler_dispatch` → `scf.loop` with yield dispatch.
pub(crate) struct ArenaLowerHandlerDispatchPattern;

impl ArenaRewritePattern for ArenaLowerHandlerDispatchPattern {
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

        let loop_body = build_loop_body_arena(ctx, loc, body_region, &suspend_arms, user_result_ty);

        let loop_op = arena_scf::r#loop(ctx, loc, [prompt_result], user_result_ty, loop_body);
        rewriter.replace_op(loop_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "ArenaLowerHandlerDispatchPattern"
    }
}

fn build_loop_body_arena(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::types::Location,
    body_region: RegionRef,
    suspend_arms: &[ArenaSuspendArm],
    user_result_ty: TypeRef,
) -> RegionRef {
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i1_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
    let nil_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());

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

    // Build shift branch (then — yield is active)
    let shift_branch = build_shift_branch_arena(ctx, loc, suspend_arms, ptr_ty);

    // Build done branch (else — normal return)
    let done_branch =
        build_done_branch_arena(ctx, loc, body_region, current, user_result_ty, ptr_ty);

    // scf.if(%is_yield) { shift } else { done }
    let if_op = arena_scf::r#if(ctx, loc, is_yield, nil_ty, shift_branch, done_branch);
    ctx.push_op(block, if_op.op_ref());

    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    })
}

fn build_done_branch_arena(
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
        let result = arena_cast_if_needed(ctx, block, loc, current, ptr_ty, user_result_ty);
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
        let result = arena_cast_if_needed(ctx, block, loc, current, ptr_ty, user_result_ty);
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

    let done_value = arena_cast_if_needed(ctx, new_block, loc, current, ptr_ty, done_arg_ty);

    // Build value remap: done block args → done_value
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

        // Map old results → new results
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

fn build_shift_branch_arena(
    ctx: &mut IrContext,
    loc: trunk_ir::arena::types::Location,
    suspend_arms: &[ArenaSuspendArm],
    ptr_ty: TypeRef,
) -> RegionRef {
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let nil_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build());

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
    let arm_result =
        build_nested_dispatch_arena(ctx, block, loc, ptr_ty, op_idx, k, v, suspend_arms, 0);

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
fn build_nested_dispatch_arena(
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

    let then_region = build_arm_region_arena(ctx, loc, arm.body, k, v, result_ty);

    if is_last {
        let true_const = arena_arith::r#const(ctx, loc, i1_ty, ArenaAttribute::IntBits(1));
        ctx.push_op(block, true_const.op_ref());
        let else_region = build_arm_region_arena(ctx, loc, arm.body, k, v, result_ty);
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
    let else_result = build_nested_dispatch_arena(
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

/// Build a single-block region from a suspend arm body (arena version).
fn build_arm_region_arena(
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

    // Build value remap: block args → FFI getter values
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

        // Map old results → new results
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

/// Cast a value if types differ, inserting ops into the given block (arena version).
fn arena_cast_if_needed(
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
        builder = builder.region(r);
    }
    for s in successors {
        builder = builder.successor(s);
    }

    let data = builder.build(ctx);
    let new_op = ctx.create_op(data);
    ctx.push_op(dest_block, new_op);
}
