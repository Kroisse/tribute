use std::collections::HashSet;

use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func::{self, Func};
use trunk_ir::dialect::{cont, scf, trampoline};
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value,
};

use super::calls_effectful_function;

// ============================================================================
// Truncate After Shift
// ============================================================================

/// Truncate effectful function bodies after the first effect point.
///
/// An effect point is either:
/// 1. A `step_shift` operation (from transformed cont.shift)
/// 2. A call to an effectful function (which may return Step)
///
/// After these points, continuation operations are stored in ResumeFuncSpec
/// for resume function generation, but they still remain in the original
/// function body. This causes type mismatches because the effect point
/// returns `Step` but continuation ops expect the original type.
///
/// This function removes all operations after an effect point and adds a proper
/// `func.return` for the Step result.
pub(crate) fn truncate_after_shift<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> Module<'db> {
    tracing::debug!(
        "truncate_after_shift: processing {} effectful functions: {:?}",
        effectful_funcs.len(),
        effectful_funcs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    );
    let body = module.body(db);
    let (new_body, modified) =
        find_and_truncate_effectful_funcs_in_region(db, &body, effectful_funcs);

    if !modified {
        return module;
    }

    Module::create(db, module.location(db), module.name(db), new_body)
}

/// Recursively find and truncate effectful functions in a region.
/// This includes functions nested inside other operations (e.g., inside enum definitions).
/// Returns (modified_region, was_modified).
fn find_and_truncate_effectful_funcs_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> (Region<'db>, bool) {
    let mut new_blocks = Vec::new();
    let mut region_modified = false;

    for block in region.blocks(db).iter() {
        let mut new_ops = Vec::new();
        let mut block_modified = false;

        for op in block.operations(db).iter() {
            if let Ok(func) = Func::from_operation(db, *op) {
                let func_name = func.sym_name(db);
                if effectful_funcs.contains(&func_name) {
                    // Process effectful function
                    let (new_func, func_modified) =
                        truncate_func_after_shift(db, *op, effectful_funcs);
                    new_ops.push(new_func);
                    block_modified |= func_modified;
                    continue;
                }
            }

            // Recursively process nested regions to find more effectful functions
            let regions = op.regions(db);
            if regions.is_empty() {
                new_ops.push(*op);
            } else {
                let mut new_regions = Vec::new();
                let mut op_modified = false;

                for nested_region in regions.iter() {
                    let (new_nested, nested_modified) = find_and_truncate_effectful_funcs_in_region(
                        db,
                        nested_region,
                        effectful_funcs,
                    );
                    new_regions.push(new_nested);
                    op_modified |= nested_modified;
                }

                if op_modified {
                    let new_op = op.modify(db).regions(IdVec::from(new_regions)).build();
                    new_ops.push(new_op);
                    block_modified = true;
                } else {
                    new_ops.push(*op);
                }
            }
        }

        if block_modified {
            new_blocks.push(Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                IdVec::from(new_ops),
            ));
            region_modified = true;
        } else {
            new_blocks.push(*block);
        }
    }

    if region_modified {
        (
            Region::new(db, region.location(db), IdVec::from(new_blocks)),
            true,
        )
    } else {
        (*region, false)
    }
}

/// Truncate a single function's body after the first effect point.
/// Returns (modified_operation, was_modified).
fn truncate_func_after_shift<'db>(
    db: &'db dyn salsa::Database,
    func_op: Operation<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> (Operation<'db>, bool) {
    let func = match Func::from_operation(db, func_op) {
        Ok(f) => f,
        Err(_) => return (func_op, false),
    };

    let func_name = func.sym_name(db);
    let body = func.body(db);
    let (new_body, body_modified) = truncate_region_after_shift(db, body, effectful_funcs);

    // Effectful functions need Step return type even if body wasn't truncated
    // (e.g., push_prompt may be in a nested region like handler_dispatch)
    let is_effectful = effectful_funcs.contains(&func_name);
    let modified = body_modified || is_effectful;

    tracing::debug!(
        "truncate_func_after_shift: {} body_modified={} is_effectful={} modified={}",
        func_name,
        body_modified,
        is_effectful,
        modified
    );

    if !modified {
        return (func_op, false);
    }

    // Change return type to Step for effectful functions
    // The function type is stored in the "type" attribute, not in results
    let step_ty = trampoline::Step::new(db).as_type();

    let mut builder = func_op.modify(db).regions(IdVec::from(vec![new_body]));

    if is_effectful {
        // Get the existing function type and modify its return type
        let old_func_ty = func.r#type(db);
        let func_ty = core::Func::from_type(db, old_func_ty).unwrap_or_else(|| {
            panic!(
                "truncate_func_after_shift: effectful function {} has non-function type {:?}",
                func_name, old_func_ty
            )
        });
        let original_result = func_ty.result(db);
        let params = func_ty.params(db);
        let effect = func_ty.effect(db);
        let new_func_ty = core::Func::with_effect(db, params, step_ty, effect);
        builder = builder
            .attr(Symbol::new("type"), Attribute::Type(new_func_ty.as_type()))
            .attr(
                Symbol::new("original_result_type"),
                Attribute::Type(original_result),
            );
        tracing::debug!(
            "truncate_func_after_shift: {} changed return type to Step (original: {:?})",
            func_name,
            original_result
        );
    }

    let new_op = builder.build();
    (new_op, true)
}

/// Truncate a region after the first effect point.
/// Returns (modified_region, was_modified).
fn truncate_region_after_shift<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> (Region<'db>, bool) {
    let mut new_blocks = Vec::new();
    let mut any_modified = false;

    for block in region.blocks(db).iter() {
        let (new_block, modified) = truncate_block_after_shift(db, *block, effectful_funcs);
        new_blocks.push(new_block);
        any_modified |= modified;
    }

    if !any_modified {
        return (region, false);
    }

    (
        Region::new(db, region.location(db), IdVec::from(new_blocks)),
        true,
    )
}

/// Truncate an scf.if branch region, keeping only operations up to the first
/// effectful call and ending with scf.yield(Step value).
pub(crate) fn truncate_scf_if_branch<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    effectful_funcs: &HashSet<Symbol>,
    step_ty: Type<'db>,
) -> Region<'db> {
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| truncate_scf_if_block(db, block, effectful_funcs, step_ty))
        .collect();
    Region::new(db, region.location(db), new_blocks)
}

/// Truncate an scf.if block, keeping operations up to the first effectful call
/// and replacing the terminator with scf.yield(Step value).
pub(crate) fn truncate_scf_if_block<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    effectful_funcs: &HashSet<Symbol>,
    step_ty: Type<'db>,
) -> Block<'db> {
    let ops = block.operations(db);
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    let mut step_value: Option<Value<'db>> = None;
    let mut original_yield_operand: Option<Value<'db>> = None;

    for op in ops.iter() {
        // Capture scf.yield operand before skipping — needed to wrap pure
        // branches in step_done when the branch has no effectful call.
        if scf::Yield::from_operation(db, *op).is_ok() {
            original_yield_operand = op.operands(db).first().copied();
            continue;
        }

        // Check if this is a call to an effectful function
        if let Ok(call) = func::Call::from_operation(db, *op)
            && effectful_funcs.contains(&call.callee(db))
        {
            // Create new call with Step result type
            let new_call = Operation::new(
                db,
                op.location(db),
                op.dialect(db),
                op.name(db),
                op.operands(db).clone(),
                IdVec::from(vec![step_ty]),
                op.attributes(db).clone(),
                op.regions(db).clone(),
                op.successors(db).clone(),
            );
            step_value = Some(new_call.result(db, 0));
            new_ops.push(new_call);
            break; // Stop processing after effectful call
        }

        // Check if this operation already produces Step
        if op
            .results(db)
            .first()
            .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
        {
            step_value = Some(op.result(db, 0));
            new_ops.push(*op);
            break; // Stop processing after Step-producing op
        }

        // Keep other operations
        new_ops.push(*op);
    }

    // Add scf.yield with Step value
    if let Some(val) = step_value {
        let yield_op = scf::r#yield(db, block.location(db), vec![val]);
        new_ops.push(yield_op.as_operation());
    } else if let Some(original_val) = original_yield_operand {
        // Non-effectful branch: wrap the original yield value in step_done
        // so both branches of the scf.if produce a Step-typed result.
        let step_done_op = trampoline::step_done(db, block.location(db), original_val, step_ty);
        new_ops.push(step_done_op.as_operation());
        let yield_op = scf::r#yield(db, block.location(db), vec![step_done_op.result(db)]);
        new_ops.push(yield_op.as_operation());
    }

    Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops.into(),
    )
}

/// Rebuild an operation with its result type changed to `Step`.
/// If the operation has no results, returns it unchanged.
fn rebuild_with_step_result<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> Operation<'db> {
    if op.results(db).is_empty() {
        return *op;
    }
    let step_ty = trampoline::Step::new(db).as_type();
    op.modify(db).results(IdVec::from(vec![step_ty])).build()
}

/// Truncate a block after the first effect point (step_shift or effectful call).
/// Returns (modified_block, was_modified).
fn truncate_block_after_shift<'db>(
    db: &'db dyn salsa::Database,
    block: Block<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> (Block<'db>, bool) {
    let ops = block.operations(db);
    let mut new_ops = Vec::new();
    let mut found_effect_point = false;
    let mut effect_result: Option<Value<'db>> = None;
    let mut effect_location: Option<trunk_ir::Location<'db>> = None;

    tracing::debug!(
        "truncate_block_after_shift: checking block with {} ops: [{}]",
        ops.len(),
        ops.iter()
            .map(|op| format!("{}.{}", op.dialect(db), op.name(db)))
            .collect::<Vec<_>>()
            .join(", ")
    );

    for op in ops.iter() {
        tracing::trace!(
            "truncate_block_after_shift: op = {}.{}",
            op.dialect(db),
            op.name(db)
        );
        if found_effect_point {
            // Special case: scf.loop after an effect point is the handler trampoline
            // (generated by LowerHandlerDispatchPattern). It consumes the Step result
            // from push_prompt and drives the trampoline loop. Must be preserved.
            if scf::Loop::from_operation(db, *op).is_ok() {
                // The scf.loop's init operand may reference the old scf.if result.
                // Replace it with the new effect_result from the recreated scf.if.
                let new_loop = if let Some(new_step) = effect_result {
                    let old_operands = op.operands(db);
                    debug_assert!(
                        old_operands.len() == 1,
                        "handler trampoline scf.loop should have exactly one init operand (the Step value), got {}",
                        old_operands.len()
                    );
                    if !old_operands.is_empty() {
                        let mut new_operands: Vec<Value<'db>> =
                            old_operands.iter().copied().collect();
                        new_operands[0] = new_step;
                        op.modify(db).operands(IdVec::from(new_operands)).build()
                    } else {
                        *op
                    }
                } else {
                    *op
                };
                new_ops.push(new_loop);
                // Update effect result to the loop's result (may be Step or user_result_ty)
                if !new_loop.results(db).is_empty() {
                    effect_result = Some(new_loop.result(db, 0));
                }
                effect_location = Some(op.location(db));
                tracing::debug!(
                    "truncate_block_after_shift: preserved handler trampoline scf.loop after effect point"
                );
                continue;
            }
            // Skip all other operations after effect point (they're now in resume functions)
            continue;
        }

        // Check if this is a step_shift operation
        if trampoline::StepShift::from_operation(db, *op).is_ok() {
            new_ops.push(*op);
            found_effect_point = true;
            effect_result = Some(op.result(db, 0));
            effect_location = Some(op.location(db));
            continue;
        }

        // Check if this is a push_prompt operation (establishes an effect handler)
        // The push_prompt result is the Step value from the handler
        if cont::PushPrompt::from_operation(db, *op).is_ok() {
            // Create new operation with Step result type
            let new_op = rebuild_with_step_result(db, op);
            new_ops.push(new_op);
            found_effect_point = true;
            if !new_op.results(db).is_empty() {
                effect_result = Some(new_op.result(db, 0));
            }
            effect_location = Some(op.location(db));
            tracing::debug!(
                "truncate_block_after_shift: found push_prompt, changed result type to Step"
            );
            continue;
        }

        // Check if this is a call to an effectful function
        if let Ok(call) = func::Call::from_operation(db, *op)
            && effectful_funcs.contains(&call.callee(db))
        {
            // Create new operation with Step result type
            let new_op = rebuild_with_step_result(db, op);
            new_ops.push(new_op);
            found_effect_point = true;
            if !new_op.results(db).is_empty() {
                effect_result = Some(new_op.result(db, 0));
            }
            effect_location = Some(op.location(db));
            tracing::debug!(
                "truncate_block_after_shift: found effectful call to {}, changed result type to Step",
                call.callee(db)
            );
            continue;
        }

        // Check if this is a call_indirect (indirect function call through closure).
        // In effectful functions, call_indirect may invoke effectful closures, so we
        // must treat the result as Step to avoid double-wrapping in step_done.
        if func::CallIndirect::from_operation(db, *op).is_ok() {
            let new_op = rebuild_with_step_result(db, op);
            new_ops.push(new_op);
            found_effect_point = true;
            if !new_op.results(db).is_empty() {
                effect_result = Some(new_op.result(db, 0));
            }
            effect_location = Some(op.location(db));
            tracing::debug!(
                "truncate_block_after_shift: found call_indirect, changed result type to Step"
            );
            continue;
        }

        // Check if this is a scf.if that contains effectful code or returns Step
        if scf::If::from_operation(db, *op).is_ok() {
            // Check if result type is Step (from push_prompt/check_yield)
            let returns_step = op
                .results(db)
                .first()
                .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some());

            // Check if any branch contains effectful code (calls to effectful funcs)
            let has_effectful_code = op
                .regions(db)
                .iter()
                .any(|r| calls_effectful_function(db, r, effectful_funcs));

            if has_effectful_code || returns_step {
                // Recursively process scf.if regions to truncate branches
                let step_ty = trampoline::Step::new(db).as_type();
                let new_regions: IdVec<Region<'db>> = op
                    .regions(db)
                    .iter()
                    .map(|region| truncate_scf_if_branch(db, region, effectful_funcs, step_ty))
                    .collect();

                // Create new scf.if with Step result type and truncated regions
                let new_op = Operation::new(
                    db,
                    op.location(db),
                    op.dialect(db),
                    op.name(db),
                    op.operands(db).clone(),
                    IdVec::from(vec![step_ty]),
                    op.attributes(db).clone(),
                    new_regions,
                    op.successors(db).clone(),
                );
                new_ops.push(new_op);
                found_effect_point = true;
                effect_result = Some(new_op.result(db, 0));
                effect_location = Some(op.location(db));
                let result_ty = op.results(db).first();
                tracing::debug!(
                    "truncate_block_after_shift: found scf.if with effectful code or Step result (returns_step={}, has_effectful_code={}, result_ty={:?}, location={:?})",
                    returns_step,
                    has_effectful_code,
                    result_ty,
                    op.location(db)
                );
                continue;
            }
        }

        // Check if this is a scf.loop with Step result type (trampoline loop)
        // These loops are generated by LowerHandlerDispatchPattern in effectful functions
        if scf::Loop::from_operation(db, *op).is_ok()
            && op
                .results(db)
                .first()
                .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
        {
            new_ops.push(*op);
            found_effect_point = true;
            effect_result = Some(op.result(db, 0));
            effect_location = Some(op.location(db));
            tracing::debug!("truncate_block_after_shift: found scf.loop with Step result");
            continue;
        }

        new_ops.push(*op);
    }

    if !found_effect_point {
        return (block, false);
    }

    // Add func.return for the effect result
    if let (Some(result), Some(location)) = (effect_result, effect_location) {
        let return_op = func::r#return(db, location, Some(result));
        new_ops.push(return_op.as_operation());
    }

    tracing::debug!(
        "truncate_block_after_shift: truncated {} ops to {} ops: [{}]",
        ops.len(),
        new_ops.len(),
        new_ops
            .iter()
            .map(|op| format!("{}.{}", op.dialect(db), op.name(db)))
            .collect::<Vec<_>>()
            .join(", ")
    );

    (
        Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            IdVec::from(new_ops),
        ),
        true,
    )
}
