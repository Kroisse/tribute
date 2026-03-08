use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::dialect::trampoline as arena_trampoline;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, TypeDataBuilder};

use super::analysis::calls_effectful_function;
use super::patterns::is_step_type;
use super::shift_lower::step_type;

// ============================================================================
// Truncate After Shift
// ============================================================================

/// Truncate effectful function bodies after the first effect point.
pub(crate) fn truncate_after_shift(
    ctx: &mut IrContext,
    module: Module,
    effectful_funcs: &HashSet<Symbol>,
) {
    let module_body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };

    tracing::debug!(
        "truncate_after_shift: processing {} effectful functions: {:?}",
        effectful_funcs.len(),
        effectful_funcs
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    );

    find_and_truncate_effectful_funcs_in_region(ctx, module_body, effectful_funcs);
}

/// Recursively find and truncate effectful functions in a region.
fn find_and_truncate_effectful_funcs_in_region(
    ctx: &mut IrContext,
    region: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();

    for block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

        for op in ops {
            if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                let func_name = func.sym_name(ctx);
                if effectful_funcs.contains(&func_name) {
                    truncate_func_after_shift(ctx, op, effectful_funcs);
                    continue;
                }
            }

            // Recursively process nested regions
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for nested_region in regions {
                find_and_truncate_effectful_funcs_in_region(ctx, nested_region, effectful_funcs);
            }
        }
    }
}

/// Truncate a single function's body after the first effect point.
fn truncate_func_after_shift(
    ctx: &mut IrContext,
    func_op: OpRef,
    effectful_funcs: &HashSet<Symbol>,
) {
    let Ok(func) = arena_func::Func::from_op(ctx, func_op) else {
        return;
    };

    let func_name = func.sym_name(ctx);
    let body = func.body(ctx);

    truncate_region_after_shift(ctx, body, effectful_funcs);

    // Effectful functions need Step return type
    let is_effectful = effectful_funcs.contains(&func_name);
    if is_effectful {
        let step_ty = step_type(ctx);
        let func_ty = func.r#type(ctx);
        let func_ty_data = ctx.types.get(func_ty);

        // Build new func type with Step return
        if func_ty_data.dialect == Symbol::new("core") && func_ty_data.name == Symbol::new("func") {
            let original_result = func_ty_data
                .attrs
                .get(&Symbol::new("result"))
                .and_then(|a| match a {
                    Attribute::Type(t) => Some(*t),
                    _ => None,
                });

            let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"));
            for &p in &func_ty_data.params {
                builder = builder.param(p);
            }
            builder = builder.attr("result", Attribute::Type(step_ty));
            if let Some(Attribute::Type(effect)) = func_ty_data.attrs.get(&Symbol::new("effect")) {
                builder = builder.attr("effect", Attribute::Type(*effect));
            }
            let new_func_ty = ctx.types.intern(builder.build());

            // Update the function's type attribute
            let mut new_attrs = ctx.op(func_op).attributes.clone();
            new_attrs.insert(Symbol::new("type"), Attribute::Type(new_func_ty));
            if let Some(original) = original_result {
                new_attrs.insert(
                    Symbol::new("original_result_type"),
                    Attribute::Type(original),
                );
            }
            ctx.op_mut(func_op).attributes = new_attrs;

            tracing::debug!(
                "truncate_func_after_shift: {} changed return type to Step",
                func_name,
            );
        }
    }
}

/// Truncate a region after the first effect point.
fn truncate_region_after_shift(
    ctx: &mut IrContext,
    region: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();

    for block in blocks {
        truncate_block_after_shift(ctx, block, effectful_funcs);
    }
}

/// Truncate a block after the first effect point.
fn truncate_block_after_shift(
    ctx: &mut IrContext,
    block: BlockRef,
    effectful_funcs: &HashSet<Symbol>,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let mut found_effect_point = false;
    let mut effect_result: Option<ValueRef> = None;
    let mut effect_location = None;
    let mut ops_to_remove: Vec<OpRef> = Vec::new();

    for &op in &ops {
        if found_effect_point {
            // Special case: scf.loop after an effect point (handler trampoline)
            if arena_scf::Loop::from_op(ctx, op).is_ok() {
                // Update the loop's init operand to point to the new effect_result
                if let Some(new_step) = effect_result {
                    let operands = ctx.op_operands(op).to_vec();
                    if !operands.is_empty() {
                        ctx.set_op_operand(op, 0, new_step);
                    }
                }
                // Update effect result to the loop's result
                let results = ctx.op_results(op);
                if !results.is_empty() {
                    effect_result = Some(results[0]);
                }
                effect_location = Some(ctx.op(op).location);
                continue;
            }
            // Skip all other operations after effect point
            ops_to_remove.push(op);
            continue;
        }

        // Check if this is a step_shift operation
        if arena_trampoline::StepShift::from_op(ctx, op).is_ok() {
            found_effect_point = true;
            let results = ctx.op_results(op);
            if !results.is_empty() {
                effect_result = Some(results[0]);
            }
            effect_location = Some(ctx.op(op).location);
            continue;
        }

        // Check if this is a push_prompt operation
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            // Change result type to Step
            let step_ty = step_type(ctx);
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && !is_step_type(ctx, result_types[0]) {
                ctx.set_op_result_type(op, 0, step_ty);
            }
            found_effect_point = true;
            let results = ctx.op_results(op);
            if !results.is_empty() {
                effect_result = Some(results[0]);
            }
            effect_location = Some(ctx.op(op).location);
            continue;
        }

        // Check if this is a call to an effectful function
        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful_funcs.contains(&call.callee(ctx))
        {
            let step_ty = step_type(ctx);
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && !is_step_type(ctx, result_types[0]) {
                ctx.set_op_result_type(op, 0, step_ty);
            }
            found_effect_point = true;
            let results = ctx.op_results(op);
            if !results.is_empty() {
                effect_result = Some(results[0]);
            }
            effect_location = Some(ctx.op(op).location);
            continue;
        }

        // Check if this is call_indirect
        if arena_func::CallIndirect::from_op(ctx, op).is_ok() {
            let step_ty = step_type(ctx);
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() {
                ctx.set_op_result_type(op, 0, step_ty);
            }
            found_effect_point = true;
            let results = ctx.op_results(op);
            if !results.is_empty() {
                effect_result = Some(results[0]);
            }
            effect_location = Some(ctx.op(op).location);
            continue;
        }

        // Check if this is scf.if with effectful code or Step result
        if arena_scf::If::from_op(ctx, op).is_ok() {
            let result_types = ctx.op_result_types(op);
            let returns_step = !result_types.is_empty() && is_step_type(ctx, result_types[0]);

            let has_effectful_code = ctx
                .op(op)
                .regions
                .iter()
                .any(|&r| calls_effectful_function(ctx, r, effectful_funcs));

            if has_effectful_code || returns_step {
                let step_ty = step_type(ctx);
                // Truncate branches
                let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
                for region in regions {
                    truncate_scf_if_branch(ctx, region, effectful_funcs, step_ty);
                }
                // Update result type to Step
                let result_types = ctx.op_result_types(op);
                if !result_types.is_empty() && !is_step_type(ctx, result_types[0]) {
                    ctx.set_op_result_type(op, 0, step_ty);
                }
                found_effect_point = true;
                let results = ctx.op_results(op);
                if !results.is_empty() {
                    effect_result = Some(results[0]);
                }
                effect_location = Some(ctx.op(op).location);
                continue;
            }
        }

        // Check if this is scf.loop with Step result type (trampoline loop)
        if arena_scf::Loop::from_op(ctx, op).is_ok() {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && is_step_type(ctx, result_types[0]) {
                found_effect_point = true;
                effect_result = Some(ctx.op_results(op)[0]);
                effect_location = Some(ctx.op(op).location);
                continue;
            }
        }
    }

    if !found_effect_point {
        return;
    }

    // Remove ops after effect point
    for op in ops_to_remove {
        ctx.remove_op_from_block(block, op);
    }

    // Add func.return for the effect result
    if let (Some(result), Some(location)) = (effect_result, effect_location) {
        let return_op = arena_func::r#return(ctx, location, [result]);
        ctx.push_op(block, return_op.op_ref());
    }
}

/// Truncate scf.if branch region.
pub(crate) fn truncate_scf_if_branch(
    ctx: &mut IrContext,
    region: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
    step_ty: TypeRef,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        truncate_scf_if_block(ctx, block, effectful_funcs, step_ty);
    }
}

/// Truncate scf.if block.
fn truncate_scf_if_block(
    ctx: &mut IrContext,
    block: BlockRef,
    effectful_funcs: &HashSet<Symbol>,
    step_ty: TypeRef,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let mut step_value: Option<ValueRef> = None;
    let mut original_yield_operand: Option<ValueRef> = None;
    let mut ops_to_remove: Vec<OpRef> = Vec::new();
    let mut found_effect = false;

    for &op in &ops {
        if found_effect {
            ops_to_remove.push(op);
            continue;
        }

        // Capture scf.yield operand before skipping
        if arena_scf::Yield::from_op(ctx, op).is_ok() {
            original_yield_operand = ctx.op_operands(op).first().copied();
            ops_to_remove.push(op);
            continue;
        }

        // Check if this is a call to an effectful function
        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful_funcs.contains(&call.callee(ctx))
        {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() {
                ctx.set_op_result_type(op, 0, step_ty);
                step_value = Some(ctx.op_results(op)[0]);
            }
            found_effect = true;
            continue;
        }

        // Check if this operation already produces Step
        let result_types = ctx.op_result_types(op);
        if !result_types.is_empty() && is_step_type(ctx, result_types[0]) {
            step_value = Some(ctx.op_results(op)[0]);
            found_effect = true;
            continue;
        }
    }

    // Remove ops
    for op in ops_to_remove {
        ctx.remove_op_from_block(block, op);
    }

    let location = ctx.block(block).location;

    // Add scf.yield with Step value
    if let Some(val) = step_value {
        let yield_op = arena_scf::r#yield(ctx, location, [val]);
        ctx.push_op(block, yield_op.op_ref());
    } else if let Some(original_val) = original_yield_operand {
        // Non-effectful branch: wrap in step_done
        let step_done = arena_trampoline::step_done(ctx, location, original_val, step_ty);
        ctx.push_op(block, step_done.op_ref());
        let yield_op = arena_scf::r#yield(ctx, location, [step_done.result(ctx)]);
        ctx.push_op(block, yield_op.op_ref());
    } else {
        tracing::warn!("truncate_scf_if_block: block has neither step_value nor yield operand");
        let unreachable = arena_func::unreachable(ctx, location);
        ctx.push_op(block, unreachable.op_ref());
    }
}
