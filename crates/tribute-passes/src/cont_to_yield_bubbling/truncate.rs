//! Truncate effectful function bodies after the first effect point.
//!
//! After shift lowering, effectful functions need their bodies truncated
//! so they return YieldResult at the first effect point.

use std::collections::HashSet;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::dialect::scf as arena_scf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, TypeDataBuilder};

use super::analysis::{calls_effectful_function, has_effectful_type};
use super::types::{YieldBubblingTypes, is_yield_result_type};

// ============================================================================
// Truncate After Shift
// ============================================================================

/// Truncate effectful function bodies after the first effect point.
pub(crate) fn truncate_after_shift(
    ctx: &mut IrContext,
    module: Module,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
) {
    let module_body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };

    tracing::debug!(
        "truncate_after_shift: processing {} effectful functions",
        effectful_funcs.len(),
    );

    find_and_truncate_effectful_funcs_in_region(ctx, module_body, effectful_funcs, types);
}

/// Recursively find and truncate effectful functions in a region.
fn find_and_truncate_effectful_funcs_in_region(
    ctx: &mut IrContext,
    region: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();

    for block in blocks {
        let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

        for op in ops {
            if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                let func_name = func.sym_name(ctx);
                if effectful_funcs.contains(&func_name) {
                    truncate_func_after_shift(ctx, op, effectful_funcs, types);
                    continue;
                }
            }

            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for nested_region in regions {
                find_and_truncate_effectful_funcs_in_region(
                    ctx,
                    nested_region,
                    effectful_funcs,
                    types,
                );
            }
        }
    }
}

/// Truncate a single function's body after the first effect point.
fn truncate_func_after_shift(
    ctx: &mut IrContext,
    func_op: OpRef,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
) {
    let Ok(func) = arena_func::Func::from_op(ctx, func_op) else {
        return;
    };

    let func_name = func.sym_name(ctx);
    let body = func.body(ctx);

    truncate_region_after_shift(ctx, body, effectful_funcs, types);

    // Effectful functions need YieldResult return type
    if effectful_funcs.contains(&func_name) {
        let func_ty = func.r#type(ctx);
        let func_ty_data = ctx.types.get(func_ty);

        if func_ty_data.dialect == Symbol::new("core") && func_ty_data.name == Symbol::new("func") {
            // Detect Layout A (params[0]=return) vs Layout B (attrs.result=return)
            let has_result_attr = func_ty_data.attrs.contains_key(&Symbol::new("result"));
            let (original_result, arg_params) = if has_result_attr {
                // Layout B: params are actual parameters, return type in attrs
                let ret = func_ty_data
                    .attrs
                    .get(&Symbol::new("result"))
                    .and_then(|a| match a {
                        Attribute::Type(t) => Some(*t),
                        _ => None,
                    });
                (ret, &func_ty_data.params[..])
            } else if !func_ty_data.params.is_empty() {
                // Layout A: params[0] = return type, params[1..] = actual params
                (Some(func_ty_data.params[0]), &func_ty_data.params[1..])
            } else {
                (None, &func_ty_data.params[..])
            };

            // Build new type in Layout A: params[0] = return type, params[1..] = arg params
            // (Cranelift backend's translate_signature expects Layout A)
            let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"));
            builder = builder.param(types.yield_result); // return type
            for &p in arg_params {
                builder = builder.param(p);
            }
            if let Some(Attribute::Type(effect)) = func_ty_data.attrs.get(&Symbol::new("effect")) {
                builder = builder.attr("effect", Attribute::Type(*effect));
            }
            let new_func_ty = ctx.types.intern(builder.build());

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
                "truncate_func_after_shift: {} changed return type to YieldResult",
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
    types: &YieldBubblingTypes,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        truncate_block_after_shift(ctx, block, effectful_funcs, types);
    }
}

/// Truncate a block after the first effect point.
fn truncate_block_after_shift(
    ctx: &mut IrContext,
    block: BlockRef,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let mut found_effect_point = false;
    let mut effect_result: Option<ValueRef> = None;
    let mut effect_location = None;
    let mut ops_to_remove: Vec<OpRef> = Vec::new();

    for &op in &ops {
        if found_effect_point {
            // scf.loop after an effect point (handler trampoline)
            if arena_scf::Loop::from_op(ctx, op).is_ok() {
                if let Some(new_step) = effect_result {
                    let operands = ctx.op_operands(op).to_vec();
                    if !operands.is_empty() {
                        ctx.set_op_operand(op, 0, new_step);
                    }
                }
                let results = ctx.op_results(op);
                if !results.is_empty() {
                    effect_result = Some(results[0]);
                }
                effect_location = Some(ctx.op(op).location);
                continue;
            }
            ops_to_remove.push(op);
            continue;
        }

        // Check if this operation produces a YieldResult
        let result_types = ctx.op_result_types(op);
        if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
            found_effect_point = true;
            effect_result = Some(ctx.op_results(op)[0]);
            effect_location = Some(ctx.op(op).location);
            continue;
        }

        // Check cont.push_prompt
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && !is_yield_result_type(ctx, result_types[0]) {
                ctx.set_op_result_type(op, 0, types.yield_result);
            }
            found_effect_point = true;
            let results = ctx.op_results(op);
            if !results.is_empty() {
                effect_result = Some(results[0]);
            }
            effect_location = Some(ctx.op(op).location);
            continue;
        }

        // Check call to effectful function
        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful_funcs.contains(&call.callee(ctx))
        {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && !is_yield_result_type(ctx, result_types[0]) {
                ctx.set_op_result_type(op, 0, types.yield_result);
            }
            found_effect_point = true;
            let results = ctx.op_results(op);
            if !results.is_empty() {
                effect_result = Some(results[0]);
            }
            effect_location = Some(ctx.op(op).location);
            continue;
        }

        // Check call_indirect to effectful function
        if arena_func::CallIndirect::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op).to_vec();
            let is_effectful = if !operands.is_empty() {
                let callee_ty = ctx.value_ty(operands[0]);
                has_effectful_type(ctx, callee_ty)
            } else {
                false
            };
            if is_effectful {
                let result_types = ctx.op_result_types(op);
                if !result_types.is_empty() {
                    ctx.set_op_result_type(op, 0, types.yield_result);
                }
                found_effect_point = true;
                let results = ctx.op_results(op);
                if !results.is_empty() {
                    effect_result = Some(results[0]);
                }
                effect_location = Some(ctx.op(op).location);
            }
            continue;
        }

        // Check scf.if with effectful code or YieldResult result
        if arena_scf::If::from_op(ctx, op).is_ok() {
            let result_types = ctx.op_result_types(op);
            let returns_yr = !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]);

            let has_effectful_code = ctx
                .op(op)
                .regions
                .iter()
                .any(|&r| calls_effectful_function(ctx, r, effectful_funcs));

            if has_effectful_code || returns_yr {
                let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
                for region in regions {
                    truncate_scf_if_branch(ctx, region, effectful_funcs, types);
                }
                let result_types = ctx.op_result_types(op);
                if !result_types.is_empty() && !is_yield_result_type(ctx, result_types[0]) {
                    ctx.set_op_result_type(op, 0, types.yield_result);
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

        // Check scf.loop with YieldResult result type
        if arena_scf::Loop::from_op(ctx, op).is_ok() {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
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

    for op in ops_to_remove {
        ctx.remove_op_from_block(block, op);
    }

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
    types: &YieldBubblingTypes,
) {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        truncate_scf_if_block(ctx, block, effectful_funcs, types);
    }
}

/// Truncate scf.if block.
fn truncate_scf_if_block(
    ctx: &mut IrContext,
    block: BlockRef,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    let mut yr_value: Option<ValueRef> = None;
    let mut original_yield_operand: Option<ValueRef> = None;
    let mut ops_to_remove: Vec<OpRef> = Vec::new();
    let mut found_effect = false;

    for &op in &ops {
        if found_effect {
            ops_to_remove.push(op);
            continue;
        }

        if arena_scf::Yield::from_op(ctx, op).is_ok() {
            original_yield_operand = ctx.op_operands(op).first().copied();
            ops_to_remove.push(op);
            continue;
        }

        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful_funcs.contains(&call.callee(ctx))
        {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() {
                ctx.set_op_result_type(op, 0, types.yield_result);
                yr_value = Some(ctx.op_results(op)[0]);
            }
            found_effect = true;
            continue;
        }

        let result_types = ctx.op_result_types(op);
        if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
            yr_value = Some(ctx.op_results(op)[0]);
            found_effect = true;
            continue;
        }
    }

    for op in ops_to_remove {
        ctx.remove_op_from_block(block, op);
    }

    let location = ctx.block(block).location;

    if let Some(val) = yr_value {
        let yield_op = arena_scf::r#yield(ctx, location, [val]);
        ctx.push_op(block, yield_op.op_ref());
    } else if let Some(original_val) = original_yield_operand {
        // Non-effectful branch: wrap in YieldResult::Done
        let anyref_val =
            arena_core::unrealized_conversion_cast(ctx, location, original_val, types.anyref);
        ctx.push_op(block, anyref_val.op_ref());

        let done_op = arena_adt::variant_new(
            ctx,
            location,
            [anyref_val.result(ctx)],
            types.yield_result,
            types.yield_result,
            Symbol::new("Done"),
        );
        ctx.push_op(block, done_op.op_ref());

        let yield_op = arena_scf::r#yield(ctx, location, [done_op.result(ctx)]);
        ctx.push_op(block, yield_op.op_ref());
    } else {
        tracing::warn!("truncate_scf_if_block: block has neither yr_value nor yield operand");
        let unreachable = arena_func::unreachable(ctx, location);
        ctx.push_op(block, unreachable.op_ref());
    }
}
