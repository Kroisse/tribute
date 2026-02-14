use std::collections::HashSet;
use std::rc::Rc;

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{arith, cont, scf, trampoline};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::rewrite::{OpAdaptor, RewritePattern, RewriteResult};
use trunk_ir::{
    Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Span, Symbol, Type, Value,
};

use super::collect_suspend_arms;
use crate::cont_util::{SuspendArm, rebuild_region_with_remap, remap_value};

// ============================================================================
// Pattern: Lower cont.handler_dispatch
// ============================================================================

pub(crate) struct LowerHandlerDispatchPattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
    /// Spans of handler_dispatch operations that are inside effectful functions.
    /// These handlers should return Step type for propagation.
    pub(crate) handlers_in_effectful_funcs: Rc<HashSet<Span>>,
}

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
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I::<1>::new(db).as_type();
        let step_ty = trampoline::Step::new(db).as_type();
        let anyref_ty = tribute_rt::Any::new(db).as_type();

        // Get the step operand (result of push_prompt)
        let step_operand = op.operands(db).first().copied().unwrap();

        // Get the handler's tag
        let our_tag = dispatch.tag(db);

        // Get the user's result type from the result_type attribute
        // This was set by tribute_to_cont when creating the handler_dispatch
        let user_result_ty = dispatch.result_type(db);

        // Get the body region with child ops (cont.done + cont.suspend)
        let body_region = dispatch.body(db);

        // Collect suspend arms with their expected op_idx
        let suspend_arms = collect_suspend_arms(db, &body_region);

        // Check if this handler is inside an effectful function
        // If so, the loop should return Step type for propagation up the call stack
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
            db,
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
        let loop_op = scf::r#loop(db, location, vec![step_operand], loop_result_ty, loop_body);

        RewriteResult::expand(vec![loop_op.as_operation()])
    }
}

impl LowerHandlerDispatchPattern {
    /// Build the trampoline loop body region.
    ///
    /// The loop body receives `current_step` (Step type) as a block argument and:
    /// 1. Checks if step is Done or Shift
    /// 2. If Done: extracts value, runs done block, breaks with result
    /// 3. If Shift: checks tag, dispatches to appropriate arm, continues with new step
    #[allow(clippy::too_many_arguments)]
    fn build_trampoline_loop_body<'db>(
        &self,
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        our_tag: u32,
        suspend_arms: &[SuspendArm<'db>],
        user_result_ty: Type<'db>,
        step_ty: Type<'db>,
        i32_ty: Type<'db>,
        i1_ty: Type<'db>,
        anyref_ty: Type<'db>,
        is_in_effectful_func: bool,
    ) -> Region<'db> {
        use trunk_ir::BlockArg;

        // Create block with current_step as argument
        let block_id = trunk_ir::BlockId::fresh();
        let current_step_arg = BlockArg::of_type(db, step_ty);
        let current_step = Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);

        let mut builder = BlockBuilder::new(db, location);

        // Extract tag field from Step (field 0: tag, 0=Done, 1=Shift)
        let get_tag = builder.op(trampoline::step_get(
            db,
            location,
            current_step,
            i32_ty,
            Symbol::new("tag"),
        ));
        let step_tag = get_tag.result(db);

        // Compare with DONE (0)
        let done_const = builder.op(arith::Const::i32(db, location, 0));
        let is_done = builder.op(arith::cmp_eq(
            db,
            location,
            step_tag,
            done_const.result(db),
            i1_ty,
        ));

        // Build Done branch: extract value and break with result
        // If in effectful func, wrap result in Step.Done for propagation
        let done_branch = self.build_done_branch(
            db,
            location,
            current_step,
            user_result_ty,
            anyref_ty,
            step_ty,
            is_in_effectful_func,
        );

        // Build Shift branch: check tag and dispatch
        let shift_branch = self.build_shift_branch(
            db,
            location,
            our_tag,
            current_step,
            suspend_arms,
            step_ty,
            i32_ty,
            i1_ty,
        );

        // scf.if: if is_done { done_branch } else { shift_branch }
        // Note: Both branches use scf.break/scf.continue to control loop
        // Result type is nil because both branches terminate via break/continue
        let nil_ty = core::Nil::new(db).as_type();
        builder.op(scf::r#if(
            db,
            location,
            is_done.result(db),
            nil_ty, // void result - both branches use break/continue terminators
            done_branch,
            shift_branch,
        ));

        let body_block = Block::new(
            db,
            block_id,
            location,
            IdVec::from(vec![current_step_arg]),
            builder.build().operations(db).clone(),
        );

        Region::new(db, location, IdVec::from(vec![body_block]))
    }

    /// Build the Done branch of the trampoline loop.
    /// Extracts value from Step and breaks with result.
    /// If `is_in_effectful_func` is true, wraps result in Step.Done for propagation.
    #[allow(clippy::too_many_arguments)]
    fn build_done_branch<'db>(
        &self,
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        current_step: Value<'db>,
        user_result_ty: Type<'db>,
        anyref_ty: Type<'db>,
        step_ty: Type<'db>,
        is_in_effectful_func: bool,
    ) -> Region<'db> {
        let mut builder = BlockBuilder::new(db, location);

        // Extract value from Step (field 1: value, anyref type)
        let get_value = builder.op(trampoline::step_get(
            db,
            location,
            current_step,
            anyref_ty,
            Symbol::new("value"),
        ));
        let step_value = get_value.result(db);

        // Cast anyref to user result type if needed
        let result_value = if anyref_ty != user_result_ty {
            let cast = builder.op(core::unrealized_conversion_cast(
                db,
                location,
                step_value,
                user_result_ty,
            ));
            cast.result(db)
        } else {
            step_value
        };

        if is_in_effectful_func {
            // In effectful function: wrap result in Step.Done for propagation
            let step_done = builder.op(trampoline::step_done(db, location, result_value, step_ty));
            builder.op(scf::r#break(db, location, step_done.result(db)));
        } else {
            // Closed handler: break directly with the extracted value
            builder.op(scf::r#break(db, location, result_value));
        }

        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
    }

    /// Build the Shift branch of the trampoline loop.
    /// Checks prompt tag and dispatches to appropriate handler arm.
    #[allow(clippy::too_many_arguments)]
    fn build_shift_branch<'db>(
        &self,
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        our_tag: u32,
        current_step: Value<'db>,
        suspend_arms: &[SuspendArm<'db>],
        step_ty: Type<'db>,
        i32_ty: Type<'db>,
        i1_ty: Type<'db>,
    ) -> Region<'db> {
        let mut builder = BlockBuilder::new(db, location);

        // Extract prompt tag from Step (field 2: prompt)
        let get_prompt = builder.op(trampoline::step_get(
            db,
            location,
            current_step,
            i32_ty,
            Symbol::new("prompt"),
        ));
        let step_prompt = get_prompt.result(db);

        // Compare with our handler's tag
        let our_tag_const = builder.op(arith::Const::i32(db, location, our_tag as i32));
        let tag_matches = builder.op(arith::cmp_eq(
            db,
            location,
            step_prompt,
            our_tag_const.result(db),
            i1_ty,
        ));

        // Build dispatch region (when tag matches)
        let dispatch_region = self.build_dispatch_region(db, location, suspend_arms, step_ty);

        // Build propagate region (when tag doesn't match - for open handlers)
        // For closed handlers, this should never be reached
        let propagate_region = self.build_propagate_region(db, location, current_step);

        // scf.if: if tag_matches { dispatch } else { propagate }
        let if_op = builder.op(scf::r#if(
            db,
            location,
            tag_matches.result(db),
            step_ty, // Both branches return Step for continue
            dispatch_region,
            propagate_region,
        ));

        // Continue loop with new step
        builder.op(scf::r#continue(db, location, vec![if_op.result(db)]));

        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
    }

    /// Build dispatch region that handles the effect based on op_idx.
    fn build_dispatch_region<'db>(
        &self,
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        suspend_arms: &[SuspendArm<'db>],
        step_ty: Type<'db>,
    ) -> Region<'db> {
        // This is similar to the existing build_suspend_dispatch_region
        // but the result is Step (from resume calls) for continuing the loop
        build_suspend_dispatch_region(db, location, step_ty, suspend_arms, &self.effectful_funcs)
    }

    /// Build propagate region for unhandled effects (open handlers).
    /// For closed handlers this should never be reached.
    fn build_propagate_region<'db>(
        &self,
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        current_step: Value<'db>,
    ) -> Region<'db> {
        let mut builder = BlockBuilder::new(db, location);

        // For closed handlers, this path should never be reached at runtime
        // (the tag always matches). We keep the yield (instead of trapping)
        // for forward compatibility with open handlers, which will need to
        // propagate unhandled effects up the call stack.
        builder.op(scf::r#yield(db, location, vec![current_step]));

        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
    }
}

// SuspendArm and collect_suspend_arms are defined in crate::cont_util

/// Build a single-block region for suspend dispatch using nested scf.if.
fn build_suspend_dispatch_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    result_ty: Type<'db>,
    suspend_arms: &[SuspendArm<'db>],
    effectful_funcs: &HashSet<Symbol>,
) -> Region<'db> {
    let i32_ty = core::I32::new(db).as_type();

    if suspend_arms.is_empty() {
        // No suspend arms - this path is unreachable in practice
        let mut builder = BlockBuilder::new(db, location);
        builder.op(func::unreachable(db, location));
        return Region::new(db, location, IdVec::from(vec![builder.build()]));
    }

    // Build a single block that does:
    // 1. Get current op_idx
    // 2. Build nested if-else chain to dispatch
    let mut builder = BlockBuilder::new(db, location);

    // Get current op_idx from global state
    let get_op_idx = builder.op(trampoline::get_yield_op_idx(db, location, i32_ty));
    let current_op_idx = get_op_idx.result(db);

    // Build nested if-else dispatch
    let final_result = build_nested_dispatch(
        db,
        &mut builder,
        location,
        result_ty,
        current_op_idx,
        suspend_arms,
        0,
        effectful_funcs,
    );

    // Yield the result
    builder.op(scf::r#yield(db, location, vec![final_result]));

    Region::new(db, location, IdVec::from(vec![builder.build()]))
}

/// Build nested if-else dispatch for suspend arms.
/// Returns the final result value.
///
/// Strategy: The last arm becomes the default else case (no condition check needed).
/// - For 1 arm: if (true) { arm0 } else { arm0 } (always executes arm0)
/// - For 2 arms: if (op_idx == 0) { arm0 } else { arm1 }
/// - For 3 arms: if (op_idx == 0) { arm0 } else { if (op_idx == 1) { arm1 } else { arm2 } }
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_nested_dispatch<'db>(
    db: &'db dyn salsa::Database,
    builder: &mut BlockBuilder<'db>,
    location: Location<'db>,
    result_ty: Type<'db>,
    current_op_idx: Value<'db>,
    suspend_arms: &[SuspendArm<'db>],
    arm_index: usize,
    effectful_funcs: &HashSet<Symbol>,
) -> Value<'db> {
    let i1_ty = core::I::<1>::new(db).as_type();

    // Safety check
    if arm_index >= suspend_arms.len() {
        panic!("build_nested_dispatch: arm_index out of bounds");
    }

    let arm = &suspend_arms[arm_index];
    let is_last_arm = arm_index + 1 >= suspend_arms.len();

    // Build then region: execute this arm's body code
    let then_region = build_arm_region(db, location, &arm.body, effectful_funcs);

    if is_last_arm {
        // Last arm (or only arm): use always-true condition, duplicate arm for else
        let true_const = builder.op(arith::r#const(db, location, i1_ty, true.into()));
        let else_region = build_arm_region(db, location, &arm.body, effectful_funcs);

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

    // Not the last arm: build if-else with condition check

    // Compare current op_idx with expected
    let expected_const = builder.op(arith::Const::i32(db, location, arm.expected_op_idx as i32));
    let cmp_op = builder.op(arith::cmp_eq(
        db,
        location,
        current_op_idx,
        expected_const.result(db),
        i1_ty,
    ));
    let is_match = cmp_op.result(db);

    // Build else region: recurse to next arm (which may be last and become default)
    let mut else_builder = BlockBuilder::new(db, location);
    let else_result = build_nested_dispatch(
        db,
        &mut else_builder,
        location,
        result_ty,
        current_op_idx,
        suspend_arms,
        arm_index + 1,
        effectful_funcs,
    );
    else_builder.op(scf::r#yield(db, location, vec![else_result]));
    let else_region = Region::new(db, location, IdVec::from(vec![else_builder.build()]));

    // Create scf.if for this dispatch level
    let if_op = builder.op(scf::r#if(
        db,
        location,
        is_match,
        result_ty,
        then_region,
        else_region,
    ));

    if_op.result(db)
}

/// Build a single-block region from a handler arm's body region.
///
/// The arm body contains user's handler code ending with cont.resume (lowered to
/// trampoline.resume → func.call_indirect which returns Step).
/// We need to ensure the region yields this Step value.
///
/// IMPORTANT: The arm body may contain unrealized_conversion_cast operations that
/// convert the Step result to the user's expected type (e.g., i32). We need to find
/// the actual Step result and yield that, not the converted result.
pub(crate) fn build_arm_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    arm_body: &Region<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> Region<'db> {
    let Some(arm_block) = arm_body.blocks(db).first() else {
        // Empty body: emit unreachable
        let mut builder = BlockBuilder::new(db, location);
        builder.op(func::unreachable(db, location));
        return Region::new(db, location, IdVec::from(vec![builder.build()]));
    };

    // Remove block args — arm will be placed inside scf.if which has no block args.
    // Block arg values (continuation, shift_value) are replaced by trampoline ops below.
    let new_args = IdVec::new();

    let original_ops = arm_block.operations(db);

    // Process operations to:
    // 1. Find all Step->* unrealized_conversion_cast operations
    // 2. Skip them all (handler arms should propagate Step, not convert it)
    // 3. Yield the last Step value
    //
    // NOTE: Handler arms may contain multiple effectful function calls (e.g., run_state),
    // each of which returns Step. The UpdateFuncCallResultTypePattern adds casts to
    // convert Step -> user_result_ty for each such call. But in handler arms, we need
    // to propagate the Step value to the trampoline loop, not convert it.
    //
    // Example arm structure:
    //   Op 0-4: setup operations
    //   Op 5: func.call (run_state) -> Step
    //   Op 6: unrealized_conversion_cast (Step -> i32)  <- skip this
    //   Op 7: scf.yield (i32)  <- skip this, add new yield for Step
    let new_ops = {
        let step_ty = trampoline::Step::new(db).as_type();

        // Build a map from value (result) to (operation, result_index) for tracing types
        let mut result_to_op: std::collections::HashMap<Value<'db>, (&Operation<'db>, usize)> =
            std::collections::HashMap::new();
        for op in original_ops.iter() {
            for (i, _ty) in op.results(db).iter().enumerate() {
                result_to_op.insert(op.result(db, i), (op, i));
            }
        }

        // Helper to check if a value has Step type
        let value_has_step_type = |value: Value<'db>| -> bool {
            if let Some((defining_op, result_idx)) = result_to_op.get(&value) {
                let result_types = defining_op.results(db);
                if let Some(ty) = result_types.get(*result_idx) {
                    return *ty == step_ty;
                }
            }
            false
        };

        // Skip ALL unrealized_conversion_cast operations in handler arms.
        // These casts are inserted during type conversion but handler arms need to
        // work with the original (often anyref-boxed) values since they will be
        // processed by the trampoline loop which handles Step types directly.
        //
        // We build a value remapping so that references to cast outputs use the
        // cast inputs instead.
        //
        // We also remap effectful function call results to their new Step-typed results
        // so downstream operations reference the correct values.
        let mut value_remap: std::collections::HashMap<Value<'db>, Value<'db>> =
            std::collections::HashMap::new();

        for op in original_ops.iter() {
            if let Ok(cast) = core::UnrealizedConversionCast::from_operation(db, *op) {
                let cast_input = cast.value(db);
                let cast_output = op.result(db, 0);
                value_remap.insert(cast_output, cast_input);
            }
        }

        // Replace suspend body block args with trampoline ops.
        // cont.suspend body has block args (%k: continuation, %v: shift_value)
        // which become trampoline.get_yield_* ops in the lowered output.
        let mut prefix_ops: Vec<Operation<'db>> = Vec::new();
        {
            let block_id = arm_block.id(db);
            let ba = arm_block.args(db);
            if !ba.is_empty() {
                let cont_ty = trampoline::Continuation::new(db).as_type();
                let get_cont = trampoline::get_yield_continuation(db, location, cont_ty);
                prefix_ops.push(get_cont.as_operation());
                let orig = Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);
                value_remap.insert(orig, get_cont.result(db));
            }
            if ba.len() >= 2 {
                let anyref = tribute_rt::Any::new(db).as_type();
                let get_shift = trampoline::get_yield_shift_value(db, location, anyref);
                prefix_ops.push(get_shift.as_operation());
                let orig = Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 1);
                value_remap.insert(orig, get_shift.result(db));
            }
        }

        // Build operations, skipping ALL casts and scf.yield
        let mut ops: Vec<Operation<'db>> = Vec::new();
        let mut last_step_value: Option<Value<'db>> = None;

        for op in original_ops.iter() {
            // Skip all unrealized_conversion_cast
            if core::UnrealizedConversionCast::from_operation(db, *op).is_ok() {
                // If cast input has Step type, track it
                if let Some(input) = op.operands(db).first()
                    && value_has_step_type(*input)
                {
                    last_step_value = Some(*input);
                }
                continue;
            }

            // Skip existing scf.yield - we'll add our own with the Step value
            if scf::Yield::from_operation(db, *op).is_ok() {
                continue;
            }

            // Track operations that produce Step
            for (i, ty) in op.results(db).iter().enumerate() {
                if *ty == step_ty {
                    last_step_value = Some(op.result(db, i));
                }
            }

            // Detect effectful function calls - their results are Step at runtime
            // even though their IR type might be the original return type (e.g., i32)
            let is_effectful_call = if let Ok(call) = func::Call::from_operation(db, *op) {
                let callee = call.callee(db);
                let is_effectful = effectful_funcs.contains(&callee);
                let has_results = !op.results(db).is_empty();
                if is_effectful && has_results {
                    tracing::debug!(
                        "build_arm_region: found effectful call to {}, changing result type to Step",
                        callee
                    );
                }
                is_effectful && has_results
            } else {
                false
            };

            // Detect cont.resume - it gets lowered to func.call_indirect and returns Step
            let is_resume = cont::Resume::from_operation(db, *op).is_ok();
            if is_resume {
                tracing::debug!("build_arm_region: found cont.resume, will produce Step");
            }

            // Either effectful call or resume produces Step
            let produces_step = is_effectful_call || is_resume;

            // Remap operands if needed
            let operands = op.operands(db);
            let remapped_operands: IdVec<Value<'db>> = operands
                .iter()
                .map(|v| remap_value(*v, &value_remap))
                .collect::<Vec<_>>()
                .into();

            // Determine result types - effectful calls and resume need Step type
            let result_types = if produces_step {
                assert_eq!(
                    op.results(db).len(),
                    1,
                    "build_arm_region: produces_step op must have exactly 1 result, got {}",
                    op.results(db).len(),
                );
                IdVec::from(vec![step_ty])
            } else {
                op.results(db).clone()
            };

            // Rebuild nested regions with remap so block-arg references inside
            // e.g. scf.if branches are updated too.
            let remapped_regions: IdVec<Region<'db>> = op
                .regions(db)
                .iter()
                .map(|r| rebuild_region_with_remap(db, r, &value_remap))
                .collect::<Vec<_>>()
                .into();

            // If operands, regions, or result types changed, create new operation
            let needs_rebuild = remapped_operands != *operands
                || remapped_regions != *op.regions(db)
                || produces_step;
            if needs_rebuild {
                let new_op = Operation::new(
                    db,
                    op.location(db),
                    op.dialect(db),
                    op.name(db),
                    remapped_operands,
                    result_types,
                    op.attributes(db).clone(),
                    remapped_regions,
                    op.successors(db).clone(),
                );
                // Map old result values → new result values so subsequent ops
                // that reference the old results pick up the recreated ones.
                for (i, _ty) in op.results(db).iter().enumerate() {
                    value_remap.insert(op.result(db, i), new_op.result(db, i));
                }
                ops.push(new_op);
                if produces_step {
                    // Effectful call or resume returns Step at runtime - subsequent ops are handled by continuation
                    last_step_value = Some(new_op.result(db, 0));
                    break;
                }
            } else {
                ops.push(*op);
            }
        }

        // Add yield for the result (either Step directly, or wrapped in Step.Done)
        if let Some(step_val) = last_step_value {
            // Already have a Step value (from effectful call)
            ops.push(scf::r#yield(db, location, vec![step_val]).as_operation());
        } else if let Some(last_op) = ops.last().copied()
            && !last_op.results(db).is_empty()
        {
            // No Step found - the arm calls a non-effectful function
            // Wrap the last operation's result in Step.Done for the trampoline loop
            let result_value = last_op.result(db, 0);
            // Wrap in Step.Done so the trampoline loop can process it
            let step_done_op = trampoline::step_done(db, location, result_value, step_ty);
            ops.push(step_done_op.as_operation());
            ops.push(scf::r#yield(db, location, vec![step_done_op.result(db)]).as_operation());
        } else {
            // Defensive fallback: no Step value and no usable result from last op.
            // This path should be unreachable in well-formed IR, but emit a
            // terminator to keep the region valid (consistent with
            // build_suspend_dispatch_region's empty-arms fallback).
            ops.push(func::unreachable(db, location).as_operation());
        }

        let mut all_ops = prefix_ops;
        all_ops.extend(ops);
        IdVec::from(all_ops)
    };

    // Create new block with filtered args and possibly modified operations
    let new_block = Block::new(
        db,
        arm_block.id(db),
        arm_block.location(db),
        new_args,
        new_ops,
    );

    Region::new(db, location, IdVec::from(vec![new_block]))
}
