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

use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{arith, cont, scf};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::rewrite::{OpAdaptor, RewritePattern, RewriteResult};
use trunk_ir::{
    Block, BlockArg, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type,
    Value,
};

use crate::cont_util::{SuspendArm, collect_suspend_arms, get_done_region};

// ============================================================================
// Pattern: Lower cont.handler_dispatch
// ============================================================================

pub(crate) struct LowerHandlerDispatchPattern;

impl<'db> RewritePattern<'db> for LowerHandlerDispatchPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(dispatch) = cont::HandlerDispatch::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let user_result_ty = dispatch.result_type(db);

        // Get the prompt result operand (from push_prompt's __tribute_prompt call)
        let prompt_result = op.operands(db).first().copied().unwrap();

        // Get the body region with child ops (cont.done + cont.suspend)
        let body_region = dispatch.body(db);

        // Collect suspend arms with their expected op_idx
        let suspend_arms = collect_suspend_arms(db, &body_region);

        // Build the loop body
        let loop_body = build_loop_body(db, location, &body_region, &suspend_arms, user_result_ty);

        // Create scf.loop with prompt_result as initial value
        let loop_op = scf::r#loop(db, location, vec![prompt_result], user_result_ty, loop_body);

        RewriteResult::expand(vec![loop_op.as_operation()])
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
        // Unreachable — emit func.unreachable for safety
        builder.op(func::unreachable(db, location));
        // Return a dummy value — unreachable code, but need a value for SSA
        let dummy = builder.op(arith::r#const(
            db,
            location,
            result_ty,
            trunk_ir::Attribute::IntBits(0),
        ));
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

// ============================================================================
// Value remapping helpers (local, since cont_util ones aren't public functions)
// ============================================================================

fn remap_value<'db>(v: Value<'db>, value_remap: &HashMap<Value<'db>, Value<'db>>) -> Value<'db> {
    let mut current = v;
    let mut steps = 0u32;
    while let Some(&remapped) = value_remap.get(&current) {
        current = remapped;
        steps += 1;
        assert!(
            steps < 1000,
            "cycle detected in value_remap after {steps} steps"
        );
    }
    current
}

fn rebuild_op_with_remap<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Operation<'db> {
    let operands = op.operands(db);
    let remapped_operands: IdVec<Value<'db>> = operands
        .iter()
        .map(|v| remap_value(*v, value_remap))
        .collect::<Vec<_>>()
        .into();

    let regions = op.regions(db);
    let remapped_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|r| rebuild_region_with_remap(db, r, value_remap))
        .collect::<Vec<_>>()
        .into();

    if remapped_operands == *operands && remapped_regions == *regions {
        return *op;
    }

    Operation::new(
        db,
        op.location(db),
        op.dialect(db),
        op.name(db),
        remapped_operands,
        op.results(db).clone(),
        op.attributes(db).clone(),
        remapped_regions,
        op.successors(db).clone(),
    )
}

fn rebuild_region_with_remap<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Region<'db> {
    let new_blocks: Vec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let new_ops: Vec<Operation<'db>> = block
                .operations(db)
                .iter()
                .map(|op| rebuild_op_with_remap(db, op, value_remap))
                .collect();
            Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                IdVec::from(new_ops),
            )
        })
        .collect();
    Region::new(db, region.location(db), IdVec::from(new_blocks))
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
