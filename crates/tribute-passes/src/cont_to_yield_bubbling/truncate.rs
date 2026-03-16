//! Truncate effectful function bodies after the first effect point.
//!
//! After shift lowering, effectful functions need their bodies truncated
//! so they return YieldResult at the first effect point.

use std::collections::HashSet;
use std::ops::ControlFlow;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::dialect::scf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueDef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, TypeDataBuilder};
use trunk_ir::walk::{self, WalkAction};

use super::types::{YieldBubblingTypes, is_yield_result_type};

// ============================================================================
// Helpers
// ============================================================================

/// Collect all `func.func` ops in a region matching a predicate.
///
/// Uses `walk_region` to traverse the module. All func bodies are skipped
/// (not descended into), so only top-level funcs are collected.
fn collect_func_ops(
    ctx: &IrContext,
    region: RegionRef,
    predicate: impl Fn(Symbol) -> bool,
) -> Vec<OpRef> {
    let mut result = Vec::new();
    let _: ControlFlow<(), ()> = walk::walk_region(ctx, region, &mut |op| {
        if let Ok(func) = func::Func::from_op(ctx, op) {
            if predicate(func.sym_name(ctx)) {
                result.push(op);
            }
            ControlFlow::Continue(WalkAction::Skip)
        } else {
            ControlFlow::Continue(WalkAction::Advance)
        }
    });
    result
}

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

    let func_ops = collect_func_ops(ctx, module_body, |name| effectful_funcs.contains(&name));
    for func_op in func_ops {
        truncate_func_after_shift(ctx, func_op, effectful_funcs, types);
    }
}

/// Truncate a single function's body after the first effect point.
fn truncate_func_after_shift(
    ctx: &mut IrContext,
    func_op: OpRef,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
) {
    let Ok(func) = func::Func::from_op(ctx, func_op) else {
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
            // Preserve all type attrs except "result" (normalized away into Layout A)
            let result_key = Symbol::new("result");
            for (attr_name, attr_val) in &func_ty_data.attrs {
                if *attr_name != result_key {
                    builder = builder.attr(*attr_name, attr_val.clone());
                }
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
///
/// Handles three cases:
/// 1. **Shift-lowered code**: `adt.variant_new "Shift"` → always truncate (dead code)
/// 2. **Handler bodies**: scf.loop returning YieldResult → truncate after loop
/// 3. **Effectful calls**: Only truncate if ALL remaining ops are effectful calls
///    or returns (dead code from resume chain). If non-effectful ops follow
///    (like `__tribute_print_nat`), leave for `lower_effectful_calls`.
fn truncate_block_after_shift(
    ctx: &mut IrContext,
    block: BlockRef,
    effectful_funcs: &HashSet<Symbol>,
    _types: &YieldBubblingTypes,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    tracing::debug!("truncate_block_after_shift: block has {} ops", ops.len(),);
    for (i, &op) in ops.iter().enumerate() {
        let op_data = ctx.op(op);
        let rt = ctx.op_result_types(op);
        let is_yr = !rt.is_empty() && is_yield_result_type(ctx, rt[0]);
        tracing::debug!(
            "  op[{}]: {}.{} result_is_yr={} results={}",
            i,
            op_data.dialect,
            op_data.name,
            is_yr,
            rt.len(),
        );
    }

    // Recurse into nested regions first (scf.if branches, scf.loop bodies, etc.)
    for &op in &ops {
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for nested_region in regions {
            truncate_region_after_shift(ctx, nested_region, effectful_funcs, _types);
        }
    }

    let mut truncation_result: Option<ValueRef> = None;
    let mut truncation_location = None;
    let mut truncation_index = None;

    for (i, &op) in ops.iter().enumerate() {
        // Case 1: adt.variant_new with "Shift" tag → shift-lowered dead code follows
        if adt::VariantNew::from_op(ctx, op).is_ok() {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
                let attrs = &ctx.op(op).attributes;
                if let Some(Attribute::Symbol(tag)) = attrs.get(&Symbol::new("tag"))
                    && *tag == Symbol::new("Shift")
                {
                    // Check if there's an scf.loop immediately after (handler dispatch)
                    if i + 1 < ops.len() && scf::Loop::from_op(ctx, ops[i + 1]).is_ok() {
                        let loop_op = ops[i + 1];
                        let after_loop = &ops[i + 2..];
                        // Only truncate after the loop if remaining ops are dead code.
                        // This prevents removing meaningful code (like print_nat).
                        if remaining_are_dead_code(ctx, after_loop, effectful_funcs) {
                            let loop_results = ctx.op_results(loop_op);
                            if !loop_results.is_empty() {
                                truncation_result = Some(loop_results[0]);
                                truncation_location = Some(ctx.op(loop_op).location);
                                truncation_index = Some(i + 2);
                            }
                        }
                    } else {
                        let results = ctx.op_results(op);
                        if !results.is_empty() {
                            truncation_result = Some(results[0]);
                            truncation_location = Some(ctx.op(op).location);
                            truncation_index = Some(i + 1);
                        }
                    }
                    break;
                }
            }
        }

        // Case 2: scf.loop returning YieldResult (handler dispatch)
        // Only truncate if ALL remaining ops after the loop are dead code.
        // This prevents removing meaningful code (like print_nat) after
        // a handler dispatch loop in functions like main.
        if scf::Loop::from_op(ctx, op).is_ok() {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
                let remaining = &ops[i + 1..];
                if remaining_are_dead_code(ctx, remaining, effectful_funcs) {
                    let results = ctx.op_results(op);
                    if !results.is_empty() {
                        truncation_result = Some(results[0]);
                        truncation_location = Some(ctx.op(op).location);
                        truncation_index = Some(i + 1);
                    }
                    break;
                }
            }
        }

        // Case 3: named effectful call producing YieldResult
        // Only truncate if it's a direct func.call to a known effectful function
        // AND all remaining ops are dead code. Skip call_indirect (may be push_prompt
        // body call) and other YieldResult-producing ops to avoid removing handler dispatch.
        if let Ok(call) = func::Call::from_op(ctx, op)
            && effectful_funcs.contains(&call.callee(ctx))
        {
            let result_types = ctx.op_result_types(op);
            if !result_types.is_empty() && is_yield_result_type(ctx, result_types[0]) {
                let remaining = &ops[i + 1..];
                if remaining_are_dead_code(ctx, remaining, effectful_funcs) {
                    let results = ctx.op_results(op);
                    if !results.is_empty() {
                        truncation_result = Some(results[0]);
                        truncation_location = Some(ctx.op(op).location);
                        truncation_index = Some(i + 1);
                    }
                    break;
                }
            }
        }
    }

    let Some(trunc_idx) = truncation_index else {
        return;
    };

    // Remove all ops after the truncation point
    for &op in &ops[trunc_idx..] {
        ctx.remove_op_from_block(block, op);
    }

    // Add func.return with the truncation result
    if let (Some(result), Some(location)) = (truncation_result, truncation_location) {
        let block_ops = ctx.block(block).ops.to_vec();
        let already_has_return = block_ops
            .last()
            .is_some_and(|&last| func::Return::from_op(ctx, last).is_ok());
        if !already_has_return {
            let return_op = func::r#return(ctx, location, [result]);
            ctx.push_op(block, return_op.op_ref());
        }
    }
}

// ============================================================================
// Fix push_prompt body call_indirect types (double-wrapping)
// ============================================================================

/// Fix double-wrapped push_prompt body calls in effectful functions.
///
/// After truncation, effectful body thunks return YieldResult. But the
/// call_indirect in the push_prompt body (inlined by LowerPushPromptPattern)
/// still has the original result type, and its result is wrapped in Done.
/// This creates Done(YieldResult_as_anyref) — a double-wrap that prevents
/// the handler dispatch from seeing Shift variants.
///
/// Pattern detected:
///   %r = func.call_indirect ... : <NOT YieldResult>
///   %c = core.unrealized_conversion_cast %r : anyref
///   %d = adt.variant_new(%c, Done) : YieldResult
///
/// Fixed to:
///   %r = func.call_indirect ... : YieldResult
///   (users of %d now use %r)
pub(crate) fn fix_body_call_types(
    ctx: &mut IrContext,
    module: Module,
    effectful_funcs: &HashSet<Symbol>,
    _types: &YieldBubblingTypes,
) {
    let module_body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };

    let func_ops = collect_func_ops(ctx, module_body, |name| effectful_funcs.contains(&name));
    for func_op in func_ops {
        let Ok(func) = func::Func::from_op(ctx, func_op) else {
            continue;
        };
        let body = func.body(ctx);
        let blocks: Vec<_> = ctx.region(body).blocks.to_vec();
        for block in blocks {
            fix_body_call_types_in_block(ctx, block);
        }
    }
}

fn fix_body_call_types_in_block(ctx: &mut IrContext, block: BlockRef) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for (i, &op) in ops.iter().enumerate() {
        // Look for adt.variant_new with "Done" tag producing YieldResult
        if adt::VariantNew::from_op(ctx, op).is_err() {
            continue;
        }
        let result_types = ctx.op_result_types(op);
        if result_types.is_empty() || !is_yield_result_type(ctx, result_types[0]) {
            continue;
        }
        let attrs = &ctx.op(op).attributes;
        let Some(Attribute::Symbol(tag)) = attrs.get(&Symbol::new("tag")) else {
            continue;
        };
        if *tag != Symbol::new("Done") {
            continue;
        }

        // Check that Done's input comes from unrealized_conversion_cast
        let done_operands = ctx.op_operands(op).to_vec();
        if done_operands.is_empty() {
            continue;
        }
        let cast_val = done_operands[0];
        let ValueDef::OpResult(cast_op, _) = ctx.value_def(cast_val) else {
            continue;
        };
        if core::UnrealizedConversionCast::from_op(ctx, cast_op).is_err() {
            continue;
        }

        // Check that the cast's input comes from func.call_indirect
        let cast_operands = ctx.op_operands(cast_op).to_vec();
        if cast_operands.is_empty() {
            continue;
        }
        let call_val = cast_operands[0];
        let ValueDef::OpResult(call_op, _) = ctx.value_def(call_val) else {
            continue;
        };
        if func::CallIndirect::from_op(ctx, call_op).is_err() {
            continue;
        }

        // Check that call_indirect result is NOT already YieldResult
        let call_result_types = ctx.op_result_types(call_op);
        if !call_result_types.is_empty() && is_yield_result_type(ctx, call_result_types[0]) {
            continue;
        }

        // Check that the call_indirect callee is effectful (has effects in type)
        let call_operands = ctx.op_operands(call_op).to_vec();
        if call_operands.is_empty() {
            continue;
        }
        // Trace back through struct_get to find the closure/function type
        let callee_is_effectful = {
            let callee_val = call_operands[0];
            let callee_ty = ctx.value_ty(callee_val);
            if super::analysis::has_effectful_type(ctx, callee_ty) {
                true
            } else if let ValueDef::OpResult(struct_get_op, _) = ctx.value_def(callee_val) {
                if adt::StructGet::from_op(ctx, struct_get_op).is_ok() {
                    let sg_operands = ctx.op_operands(struct_get_op).to_vec();
                    if !sg_operands.is_empty() {
                        let closure_ty = ctx.value_ty(sg_operands[0]);
                        super::analysis::has_effectful_type(ctx, closure_ty)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };

        if !callee_is_effectful {
            continue;
        }

        tracing::debug!(
            "fix_body_call_types: found call_indirect → cast → Done pattern at op[{}], fixing",
            i,
        );

        // Get the YieldResult type from the Done op's result
        let yr_ty = result_types[0];

        // Fix 1: Change call_indirect result type to YieldResult
        ctx.set_op_result_type(call_op, 0, yr_ty);

        // Fix 2: Replace all uses of Done result with call_indirect result
        let done_result = ctx.op_results(op)[0];
        let call_result = ctx.op_results(call_op)[0];
        ctx.replace_all_uses(done_result, call_result);

        // Fix 3: Remove the cast and Done ops (they're now dead)
        ctx.remove_op_from_block(block, op);
        ctx.remove_op_from_block(block, cast_op);

        // Don't process more patterns in this block (we've modified it)
        break;
    }

    // Also recurse into nested regions (e.g., scf.loop body, scf.if branches)
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
    for op in ops {
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            let blocks: Vec<_> = ctx.region(region).blocks.to_vec();
            for b in blocks {
                fix_body_call_types_in_block(ctx, b);
            }
        }
    }
}

// ============================================================================
// Unwrap YieldResult in non-effectful functions
// ============================================================================

/// In non-effectful functions that call effectful functions, the call returns
/// YieldResult but the surrounding code expects the original type.
/// Since the function is not effectful, all effects are fully handled by
/// the callee, so the result is always Done. We add code to extract the
/// Done value and cast it to the original type.
pub(crate) fn unwrap_yr_in_non_effectful_funcs(
    ctx: &mut IrContext,
    module: Module,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
) {
    let module_body = match module.body(ctx) {
        Some(r) => r,
        None => return,
    };

    let func_ops = collect_func_ops(ctx, module_body, |name| !effectful_funcs.contains(&name));
    for func_op in func_ops {
        unwrap_yr_calls_in_func(ctx, func_op, effectful_funcs, types, module_body);
    }
}

fn unwrap_yr_calls_in_func(
    ctx: &mut IrContext,
    func_op: OpRef,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
    module_body: RegionRef,
) {
    let Ok(func) = func::Func::from_op(ctx, func_op) else {
        return;
    };
    let body = func.body(ctx);
    let blocks: Vec<_> = ctx.region(body).blocks.to_vec();
    for block in blocks {
        unwrap_yr_calls_in_block(ctx, block, effectful_funcs, types, module_body);
    }
}

fn unwrap_yr_calls_in_block(
    ctx: &mut IrContext,
    block: BlockRef,
    effectful_funcs: &HashSet<Symbol>,
    types: &YieldBubblingTypes,
    module_body: RegionRef,
) {
    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for &op in &ops {
        // Find func.call to effectful functions with YieldResult result
        let Ok(call) = func::Call::from_op(ctx, op) else {
            // Recurse into nested regions
            let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
            for region in regions {
                let blocks: Vec<_> = ctx.region(region).blocks.to_vec();
                for b in blocks {
                    unwrap_yr_calls_in_block(ctx, b, effectful_funcs, types, module_body);
                }
            }
            continue;
        };

        let callee = call.callee(ctx);
        if !effectful_funcs.contains(&callee) {
            continue;
        }

        let result_types = ctx.op_result_types(op);
        if result_types.is_empty() || !is_yield_result_type(ctx, result_types[0]) {
            continue;
        }

        let call_result = ctx.op_results(op)[0];
        let location = ctx.op(op).location;

        // Skip if the call result is consumed by handler_dispatch infrastructure
        // (scf.loop, adt.variant_new Done). These are already handled by the
        // handler dispatch loop and should not be unwrapped.
        // Note: the VariantNew check is intentionally broad — in non-effectful
        // functions (the only context where this runs), the only VariantNew ops
        // consuming a YieldResult are handler dispatch wrappers.
        let uses = ctx.uses(call_result).to_vec();
        let consumed_by_handler = uses.iter().any(|u| {
            scf::Loop::from_op(ctx, u.user).is_ok() || adt::VariantNew::from_op(ctx, u.user).is_ok()
        });
        if consumed_by_handler {
            continue;
        }

        // Look up the callee's original result type
        let original_result_ty = find_original_result_type(ctx, module_body, callee);
        let Some(original_ty) = original_result_ty else {
            tracing::warn!(
                "unwrap_yr_calls: could not find original result type for {}",
                callee,
            );
            continue;
        };

        tracing::debug!(
            "unwrap_yr_calls: unwrapping YieldResult for call to {} in non-effectful function",
            callee,
        );

        // Extract Done value: adt.variant_get(@YieldResult, "Done", 0, %r) : anyref
        let get_done = adt::variant_get(
            ctx,
            location,
            call_result,
            types.anyref,
            types.yield_result,
            Symbol::new("Done"),
            0,
        );

        // Cast anyref to original type
        let cast =
            core::unrealized_conversion_cast(ctx, location, get_done.result(ctx), original_ty);

        // Replace all uses of call_result (except by get_done) with cast result
        // Must do this before inserting ops to avoid stale use lists
        ctx.replace_all_uses(call_result, cast.result(ctx));
        // Restore the get_done's operand (replace_all_uses changed it too)
        ctx.set_op_operand(get_done.op_ref(), 0, call_result);

        // Find the position of the call op and insert after it
        let block_ops = ctx.block(block).ops.to_vec();
        if let Some(pos) = block_ops.iter().position(|&o| o == op) {
            // Insert in reverse order so they end up in correct order
            if pos + 1 < block_ops.len() {
                let next_op = block_ops[pos + 1];
                ctx.insert_op_before(block, next_op, cast.op_ref());
                ctx.insert_op_before(block, cast.op_ref(), get_done.op_ref());
            } else {
                ctx.push_op(block, get_done.op_ref());
                ctx.push_op(block, cast.op_ref());
            }
        }
    }
}

/// Find the original result type of a function (before yield bubbling changed it).
/// Walks the module to find the function definition and reads its
/// `original_result_type` attribute (set by truncation).
pub(crate) fn find_original_result_type(
    ctx: &IrContext,
    module_body: RegionRef,
    func_name: Symbol,
) -> Option<TypeRef> {
    find_func_original_result_in_region(ctx, module_body, func_name)
}

fn find_func_original_result_in_region(
    ctx: &IrContext,
    region: RegionRef,
    func_name: Symbol,
) -> Option<TypeRef> {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            if let Ok(func) = func::Func::from_op(ctx, op)
                && func.sym_name(ctx) == func_name
            {
                let attrs = &ctx.op(op).attributes;
                // Check for original_result_type attribute (set by truncation)
                if let Some(Attribute::Type(ty)) = attrs.get(&Symbol::new("original_result_type")) {
                    return Some(*ty);
                }
                // Fallback: extract from function type
                let func_ty = func.r#type(ctx);
                let td = ctx.types.get(func_ty);
                if td.dialect == Symbol::new("core") && td.name == Symbol::new("func") {
                    if let Some(Attribute::Type(ret)) = td.attrs.get(&Symbol::new("result")) {
                        return Some(*ret);
                    }
                    if !td.params.is_empty() {
                        return Some(td.params[0]);
                    }
                }
            }
            // Recurse into nested regions
            for &nested in &ctx.op(op).regions {
                if let Some(ty) = find_func_original_result_in_region(ctx, nested, func_name) {
                    return Some(ty);
                }
            }
        }
    }
    None
}

/// Check if remaining ops are "dead code" that can be safely truncated.
///
/// Returns false if the remaining ops contain a non-effectful function call
/// (like `__tribute_print_nat`) — those must be kept for `lower_effectful_calls`.
/// Pure operations (arith, casts, constants) and effectful calls are safe to drop.
/// Recursively descends into nested regions (scf.if, scf.loop, etc.).
fn remaining_are_dead_code(
    ctx: &IrContext,
    remaining: &[OpRef],
    effectful_funcs: &HashSet<Symbol>,
) -> bool {
    for &op in remaining {
        if !op_is_dead_code(ctx, op, effectful_funcs) {
            return false;
        }
    }
    true
}

/// Check if a single op (and its nested regions) is dead code.
fn op_is_dead_code(ctx: &IrContext, op: OpRef, effectful_funcs: &HashSet<Symbol>) -> bool {
    // Direct call to a non-effectful function → has side effects, can't truncate
    if let Ok(call) = func::Call::from_op(ctx, op) {
        if !effectful_funcs.contains(&call.callee(ctx)) {
            return false;
        }
        return true;
    }
    // call_indirect to non-effectful function → has side effects
    if func::CallIndirect::from_op(ctx, op).is_ok() {
        let operands = ctx.op_operands(op).to_vec();
        if !operands.is_empty() {
            let callee_ty = ctx.value_ty(operands[0]);
            if !super::analysis::has_effectful_type(ctx, callee_ty) {
                return false;
            }
        }
        return true;
    }
    // Recurse into nested regions
    for &region in &ctx.op(op).regions {
        for &block in &ctx.region(region).blocks {
            for &nested_op in &ctx.block(block).ops {
                if !op_is_dead_code(ctx, nested_op, effectful_funcs) {
                    return false;
                }
            }
        }
    }
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::IrContext;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn collect_func_ops_finds_matching_funcs() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @foo() -> core.i32 {
    %c = arith.const {value = 1} : core.i32
    func.return %c
  }
  func.func @bar() -> core.i32 {
    %c = arith.const {value = 2} : core.i32
    func.return %c
  }
}"#,
        );
        let module_body = module.body(&ctx).unwrap();

        // Find only "foo"
        let found = collect_func_ops(&ctx, module_body, |name| name == Symbol::new("foo"));
        assert_eq!(found.len(), 1);
        let found_func = func::Func::from_op(&ctx, found[0]).unwrap();
        assert_eq!(found_func.sym_name(&ctx), Symbol::new("foo"));

        // Find all
        let found_all = collect_func_ops(&ctx, module_body, |_| true);
        assert_eq!(found_all.len(), 2);

        // Find none
        let found_none = collect_func_ops(&ctx, module_body, |_| false);
        assert!(found_none.is_empty());
    }

    #[test]
    fn collect_func_ops_skips_nested_funcs() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @outer() -> core.i32 {
    func.func @inner() -> core.i32 {
      %c = arith.const {value = 1} : core.i32
      func.return %c
    }
    %c = arith.const {value = 2} : core.i32
    func.return %c
  }
}"#,
        );
        let module_body = module.body(&ctx).unwrap();

        // Should find "outer" but not "inner" (skip func bodies)
        let found = collect_func_ops(&ctx, module_body, |_| true);
        assert_eq!(found.len(), 1);
        let found_func = func::Func::from_op(&ctx, found[0]).unwrap();
        assert_eq!(found_func.sym_name(&ctx), Symbol::new("outer"));
    }

    #[test]
    fn find_original_result_type_reads_attribute() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @my_func() -> core.i32 {
    %c = arith.const {value = 42} : core.i32
    func.return %c
  }
}"#,
        );
        let module_body = module.body(&ctx).unwrap();

        // Manually set original_result_type attribute (as truncation would)
        let func_ops = collect_func_ops(&ctx, module_body, |n| n == Symbol::new("my_func"));
        let func_op = func_ops[0];
        let i32_ty = func::Func::from_op(&ctx, func_op).unwrap().r#type(&ctx);
        let i32_ty_data = ctx.types.get(i32_ty);
        // Extract return type from Layout A (params[0])
        let original_ret = i32_ty_data.params[0];
        ctx.op_mut(func_op).attributes.insert(
            Symbol::new("original_result_type"),
            Attribute::Type(original_ret),
        );

        let result = find_original_result_type(&ctx, module_body, Symbol::new("my_func"));
        assert_eq!(result, Some(original_ret));
    }
}
