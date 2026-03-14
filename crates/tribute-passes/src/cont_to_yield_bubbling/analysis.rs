//! Effectful function and shift point analysis for yield bubbling.
//!
//! This module re-exports and adapts the shared analysis infrastructure
//! from `cont_to_trampoline::analysis`. The analysis logic is identical —
//! only the lowering output differs.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::live_vars::FunctionAnalysis;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::location::Span;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{RegionRef, TypeRef};
use trunk_ir::types::Attribute;
use trunk_ir::walk;

use super::ShiftAnalysis;
use super::ShiftPointInfo;

// ============================================================================
// Shift Point Analysis
// ============================================================================

/// Analyze all effectful functions for shift points.
/// Returns a map from shift operation span to shift point info.
pub(crate) fn analyze_shift_points(
    ctx: &IrContext,
    module_body: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
) -> ShiftAnalysis {
    let mut analysis = HashMap::new();

    analyze_shift_points_in_region(ctx, module_body, effectful_funcs, &mut analysis);

    tracing::debug!(
        "analyze_shift_points: found {} shift points",
        analysis.len()
    );
    Rc::new(analysis)
}

/// Helper to recursively analyze shift points in a region.
fn analyze_shift_points_in_region(
    ctx: &IrContext,
    region: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
    analysis: &mut HashMap<Span, ShiftPointInfo>,
) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            // Recursively check nested regions
            for &nested_region in &ctx.op(op).regions {
                analyze_shift_points_in_region(ctx, nested_region, effectful_funcs, analysis);
            }

            let Ok(func) = arena_func::Func::from_op(ctx, op) else {
                continue;
            };
            let func_name = func.sym_name(ctx);
            if !effectful_funcs.contains(&func_name) {
                continue;
            }
            let body = func.body(ctx);
            let Some(func_analysis) = FunctionAnalysis::analyze(ctx, body) else {
                continue;
            };
            let total_shifts = func_analysis.shift_points.len();
            for shift_point in func_analysis.shift_points {
                let span = ctx.op(shift_point.shift_op).location.span;
                let results = ctx.op_results(shift_point.shift_op);
                let (shift_result_value, shift_result_type) = if !results.is_empty() {
                    (Some(results[0]), Some(ctx.value_ty(results[0])))
                } else {
                    (None, None)
                };
                analysis.insert(
                    span,
                    ShiftPointInfo {
                        index: shift_point.index,
                        total_shifts,
                        live_values: shift_point.live_values,
                        shift_result_value,
                        shift_result_type,
                        continuation_ops: shift_point.continuation_ops,
                    },
                );
            }
        }
    }
}

/// Collect spans of handler_dispatch operations inside effectful functions.
pub(crate) fn collect_handlers_in_effectful_funcs(
    ctx: &IrContext,
    module_body: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
) -> HashSet<Span> {
    let mut handler_spans = HashSet::new();

    fn collect_handlers_in_region(
        ctx: &IrContext,
        region: RegionRef,
        handler_spans: &mut HashSet<Span>,
    ) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                if arena_cont::HandlerDispatch::from_op(ctx, op).is_ok() {
                    handler_spans.insert(ctx.op(op).location.span);
                }
                for &region in &ctx.op(op).regions {
                    collect_handlers_in_region(ctx, region, handler_spans);
                }
            }
        }
    }

    fn find_effectful_funcs_and_collect(
        ctx: &IrContext,
        region: RegionRef,
        effectful_funcs: &HashSet<Symbol>,
        handler_spans: &mut HashSet<Span>,
    ) {
        for &block in &ctx.region(region).blocks {
            for &op in &ctx.block(block).ops {
                if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                    let func_name = func.sym_name(ctx);
                    if effectful_funcs.contains(&func_name) {
                        collect_handlers_in_region(ctx, func.body(ctx), handler_spans);
                    }
                }
                for &nested_region in &ctx.op(op).regions {
                    find_effectful_funcs_and_collect(
                        ctx,
                        nested_region,
                        effectful_funcs,
                        handler_spans,
                    );
                }
            }
        }
    }

    find_effectful_funcs_and_collect(ctx, module_body, effectful_funcs, &mut handler_spans);

    tracing::debug!(
        "collect_handlers_in_effectful_funcs: found {} handlers in effectful functions",
        handler_spans.len()
    );
    handler_spans
}

// ============================================================================
// Effectful Function Analysis
// ============================================================================

/// Identify all effectful functions in the module.
///
/// A function is effectful if its type signature has a non-empty effect row,
/// it contains a `cont.push_prompt` (handler function), it contains
/// `cont.resume` (outside of handler dispatch), or if it calls another
/// effectful function (transitive closure).
pub(crate) fn identify_effectful_functions(
    ctx: &IrContext,
    module_body: RegionRef,
) -> Rc<HashSet<Symbol>> {
    let mut effectful = HashSet::new();
    let mut all_funcs: Vec<(Symbol, RegionRef, TypeRef)> = Vec::new();

    collect_direct_effectful_funcs(ctx, module_body, &mut effectful, &mut all_funcs);

    // Propagate effectfulness through the call graph until fixpoint.
    // Only propagate to functions that have a tail variable in their effect row,
    // meaning they are effect-polymorphic and can forward unhandled shifts.
    // Functions without tail_var (like main) should NOT become effectful
    // just because they call an effectful function — they handle all effects locally.
    let mut changed = true;
    while changed {
        changed = false;
        for (func_name, body, func_ty) in &all_funcs {
            if effectful.contains(func_name) {
                continue;
            }
            if !has_tail_var(ctx, *func_ty) {
                continue;
            }
            if calls_effectful_function(ctx, *body, &effectful) {
                effectful.insert(*func_name);
                changed = true;
            }
        }
    }

    tracing::debug!(
        "identify_effectful_functions: found {} effectful functions: {:?}",
        effectful.len(),
        effectful.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
    Rc::new(effectful)
}

/// Collect directly effectful functions.
///
/// A function is directly effectful if:
/// - Its type signature has a non-empty effect row (concrete effects), OR
/// - It contains a `cont.push_prompt` AND its type has a tail variable
///   (effect-polymorphic handler that may propagate unhandled shifts)
fn collect_direct_effectful_funcs(
    ctx: &IrContext,
    region: RegionRef,
    effectful: &mut HashSet<Symbol>,
    all_funcs: &mut Vec<(Symbol, RegionRef, TypeRef)>,
) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                let func_name = func.sym_name(ctx);
                let body = func.body(ctx);
                let func_ty = func.r#type(ctx);

                all_funcs.push((func_name, body, func_ty));

                if has_effectful_type(ctx, func_ty) {
                    effectful.insert(func_name);
                }

                // Handler functions (containing push_prompt) are effectful only
                // if they also have a tail variable in their effect row.
                // This means they are effect-polymorphic and may propagate
                // unhandled shifts to their callers.
                // Functions without tail_var (like main) that contain push_prompt
                // handle all effects locally and should NOT be marked effectful.
                if region_contains_push_prompt(ctx, body) && has_tail_var(ctx, func_ty) {
                    effectful.insert(func_name);
                }
            }

            for &nested_region in &ctx.op(op).regions {
                collect_direct_effectful_funcs(ctx, nested_region, effectful, all_funcs);
            }
        }
    }
}

/// Check if a region (non-recursively into nested funcs) contains a `cont.push_prompt`.
fn region_contains_push_prompt(ctx: &IrContext, region: RegionRef) -> bool {
    use std::ops::ControlFlow;

    walk::walk_region::<()>(ctx, region, &mut |op| {
        if arena_func::Func::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(walk::WalkAction::Advance)
    })
    .is_break()
}

/// Check if a region (non-recursively into nested funcs or handlers) contains
/// a `cont.resume`. Skips nested func definitions, push_prompt, and
/// handler_dispatch to avoid marking the enclosing function as effectful
/// when the resume is inside a handler arm.
#[allow(dead_code)]
fn region_contains_resume(ctx: &IrContext, region: RegionRef) -> bool {
    use std::ops::ControlFlow;

    walk::walk_region::<()>(ctx, region, &mut |op| {
        if arena_func::Func::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if arena_cont::HandlerDispatch::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if arena_cont::Resume::from_op(ctx, op).is_ok() {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(walk::WalkAction::Advance)
    })
    .is_break()
}

/// Check if a function type has a tail variable in its effect row.
/// This indicates an effect-polymorphic function that may propagate
/// unhandled effects to its caller.
fn has_tail_var(ctx: &IrContext, func_ty: TypeRef) -> bool {
    let data = ctx.types.get(func_ty);
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
        return false;
    }
    let Some(Attribute::Type(effect)) = data.attrs.get(&Symbol::new("effect")) else {
        return false;
    };
    let effect_data = ctx.types.get(*effect);
    if effect_data.dialect != Symbol::new("core") || effect_data.name != Symbol::new("effect_row") {
        return false;
    }
    effect_data.attrs.contains_key(&Symbol::new("tail_var_id"))
}

/// Check if a function type has a non-empty effect row.
pub(crate) fn has_effectful_type(ctx: &IrContext, func_ty: TypeRef) -> bool {
    let data = ctx.types.get(func_ty);
    if data.dialect != Symbol::new("core") || data.name != Symbol::new("func") {
        return false;
    }
    let Some(Attribute::Type(effect)) = data.attrs.get(&Symbol::new("effect")) else {
        return false;
    };
    let effect_data = ctx.types.get(*effect);
    if effect_data.dialect != Symbol::new("core") || effect_data.name != Symbol::new("effect_row") {
        return false;
    }
    !effect_data.params.is_empty()
}

/// Check if a region calls any effectful function.
pub(crate) fn calls_effectful_function(
    ctx: &IrContext,
    region: RegionRef,
    effectful: &HashSet<Symbol>,
) -> bool {
    use std::ops::ControlFlow;

    walk::walk_region::<()>(ctx, region, &mut |op| {
        if arena_func::Func::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if let Ok(dispatch) = arena_cont::HandlerDispatch::from_op(ctx, op) {
            let body_region = dispatch.body(ctx);
            let blocks = &ctx.region(body_region).blocks;
            if let Some(&first_block) = blocks.first() {
                for &child_op in &ctx.block(first_block).ops {
                    if let Ok(done_op) = arena_cont::Done::from_op(ctx, child_op) {
                        let done_body = done_op.body(ctx);
                        for &block in &ctx.region(done_body).blocks {
                            if block_calls_effectful_inner(ctx, block, effectful) {
                                return ControlFlow::Break(());
                            }
                        }
                    }
                    if let Ok(suspend_op) = arena_cont::Suspend::from_op(ctx, child_op) {
                        let suspend_body = suspend_op.body(ctx);
                        for &block in &ctx.region(suspend_body).blocks {
                            if block_calls_effectful_inner(ctx, block, effectful) {
                                return ControlFlow::Break(());
                            }
                        }
                    }
                    if let Ok(yield_op) = arena_cont::Yield::from_op(ctx, child_op) {
                        let yield_body = yield_op.body(ctx);
                        for &block in &ctx.region(yield_body).blocks {
                            if block_calls_effectful_inner(ctx, block, effectful) {
                                return ControlFlow::Break(());
                            }
                        }
                    }
                }
            }
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful.contains(&call.callee(ctx))
        {
            return ControlFlow::Break(());
        }
        if arena_func::CallIndirect::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op).to_vec();
            if !operands.is_empty() {
                let callee_ty = ctx.value_ty(operands[0]);
                if has_effectful_type(ctx, callee_ty) {
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(walk::WalkAction::Advance)
    })
    .is_break()
}

/// Helper to check if a block calls effectful functions.
fn block_calls_effectful_inner(
    ctx: &IrContext,
    block: trunk_ir::refs::BlockRef,
    effectful: &HashSet<Symbol>,
) -> bool {
    for &op in &ctx.block(block).ops {
        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful.contains(&call.callee(ctx))
        {
            return true;
        }
        if arena_func::CallIndirect::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op).to_vec();
            if !operands.is_empty() {
                let callee_ty = ctx.value_ty(operands[0]);
                if has_effectful_type(ctx, callee_ty) {
                    return true;
                }
            }
        }
        if arena_func::Func::from_op(ctx, op).is_ok() {
            continue;
        }
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            continue;
        }
        for &region in &ctx.op(op).regions {
            for &nested_block in &ctx.region(region).blocks {
                if block_calls_effectful_inner(ctx, nested_block, effectful) {
                    return true;
                }
            }
        }
    }
    false
}
