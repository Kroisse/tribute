use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::live_vars::ArenaFunctionAnalysis;
use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::cont as arena_cont;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::ops::DialectOp;
use trunk_ir::arena::refs::{RegionRef, TypeRef};
use trunk_ir::arena::types::Attribute;
use trunk_ir::arena::walk;
use trunk_ir::location::Span;

use super::{ShiftAnalysis, ShiftPointInfo};

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

    // Recursively walk through all functions (including those in nested regions)
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
            if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                let func_name = func.sym_name(ctx);
                if effectful_funcs.contains(&func_name) {
                    // Analyze this effectful function
                    let body = func.body(ctx);
                    if let Some(func_analysis) = ArenaFunctionAnalysis::analyze(ctx, body) {
                        let total_shifts = func_analysis.shift_points.len();
                        for shift_point in func_analysis.shift_points {
                            let span = ctx.op(shift_point.shift_op).location.span;
                            // Get the shift result value and type if the operation has results
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

            // Recursively check nested regions
            for &nested_region in &ctx.op(op).regions {
                analyze_shift_points_in_region(ctx, nested_region, effectful_funcs, analysis);
            }
        }
    }
}

/// Collect spans of handler_dispatch operations inside effectful functions.
/// These handlers should return Step type (to be propagated by the effectful function).
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
                // Check if this is a handler_dispatch
                if arena_cont::HandlerDispatch::from_op(ctx, op).is_ok() {
                    handler_spans.insert(ctx.op(op).location.span);
                }
                // Recursively check nested regions
                for &region in &ctx.op(op).regions {
                    collect_handlers_in_region(ctx, region, handler_spans);
                }
            }
        }
    }

    // Helper to recursively find and process effectful functions
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
                        // Collect handler_dispatch spans in this effectful function
                        collect_handlers_in_region(ctx, func.body(ctx), handler_spans);
                    }
                }
                // Recursively check nested regions for more functions
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

    // Walk through all functions (including those in nested regions)
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
/// A function is effectful if it contains `cont.shift` or `cont.push_prompt` operations,
/// or if it calls another effectful function (transitive closure).
pub(crate) fn identify_effectful_functions(
    ctx: &IrContext,
    module_body: RegionRef,
) -> Rc<HashSet<Symbol>> {
    let mut effectful = HashSet::new();
    let mut all_funcs: Vec<(Symbol, RegionRef)> = Vec::new();

    // First pass: identify direct effectful functions and collect all functions
    collect_direct_effectful_funcs(ctx, module_body, &mut effectful, &mut all_funcs);

    // Second pass: propagate effectfulness through the call graph
    // Keep iterating until no new effectful functions are found
    let mut changed = true;
    while changed {
        changed = false;
        for (func_name, body) in &all_funcs {
            if effectful.contains(func_name) {
                continue;
            }
            if calls_effectful_function(ctx, *body, &effectful) {
                effectful.insert(*func_name);
                changed = true;
            }
        }
    }

    tracing::debug!(
        "identify_effectful_functions: collected {} functions: {:?}",
        all_funcs.len(),
        all_funcs
            .iter()
            .map(|(s, _)| s.to_string())
            .collect::<Vec<_>>()
    );
    tracing::debug!(
        "identify_effectful_functions: found {} effectful functions: {:?}",
        effectful.len(),
        effectful.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
    Rc::new(effectful)
}

/// Collect directly effectful functions and all functions for later propagation.
///
/// A function is considered effectful if its type signature has a non-empty effect row,
/// which means either:
/// - It has concrete abilities (e.g., `->{State(Int)}`)
/// - It has a polymorphic effect row (e.g., `->{e}` with a tail variable)
fn collect_direct_effectful_funcs(
    ctx: &IrContext,
    region: RegionRef,
    effectful: &mut HashSet<Symbol>,
    all_funcs: &mut Vec<(Symbol, RegionRef)>,
) {
    for &block in &ctx.region(region).blocks {
        tracing::trace!(
            "collect_direct_effectful_funcs: block has {} operations",
            ctx.block(block).ops.len()
        );
        for &op in &ctx.block(block).ops {
            tracing::trace!(
                "collect_direct_effectful_funcs: op {}.{}",
                ctx.op(op).dialect,
                ctx.op(op).name
            );
            if let Ok(func) = arena_func::Func::from_op(ctx, op) {
                let func_name = func.sym_name(ctx);
                let body = func.body(ctx);
                tracing::trace!(
                    "collect_direct_effectful_funcs: found func.func '{}'",
                    func_name
                );

                all_funcs.push((func_name, body));

                // Check the function's type signature for effectfulness
                let func_ty = func.r#type(ctx);
                let is_effectful = has_effectful_type(ctx, func_ty);
                tracing::trace!(
                    "collect_direct_effectful_funcs: '{}' is_effectful={}",
                    func_name,
                    is_effectful
                );
                if is_effectful {
                    effectful.insert(func_name);
                }
            }

            // Recursively check nested regions
            for &nested_region in &ctx.op(op).regions {
                collect_direct_effectful_funcs(ctx, nested_region, effectful, all_funcs);
            }
        }
    }
}

/// Check if a function type has a non-empty effect row.
///
/// Returns true if the effect row:
/// - Has any concrete abilities, OR
/// - Has a tail variable (is polymorphic)
fn has_effectful_type(ctx: &IrContext, func_ty: TypeRef) -> bool {
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
    // Non-empty if it has ability params or a tail variable
    !effect_data.params.is_empty() || effect_data.attrs.contains_key(&Symbol::new("tail_var"))
}

/// Check if a region calls any effectful function.
/// NOTE: Calls inside push_prompt body are skipped (handled by the handler),
/// but calls in handler_dispatch ARMS are checked (they may return Step).
pub(crate) fn calls_effectful_function(
    ctx: &IrContext,
    region: RegionRef,
    effectful: &HashSet<Symbol>,
) -> bool {
    use std::ops::ControlFlow;

    walk::walk_region::<()>(ctx, region, &mut |op| {
        // Skip nested function definitions - they're analyzed separately
        if arena_func::Func::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        // Skip push_prompt body - effects there are handled by the enclosing handler
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        // For handler_dispatch: check if handler ARMS call effectful functions
        if let Ok(dispatch) = arena_cont::HandlerDispatch::from_op(ctx, op) {
            let body_region = dispatch.body(ctx);
            // Check cont.done and cont.suspend child ops' body regions
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
                }
            }
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful.contains(&call.callee(ctx))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(walk::WalkAction::Advance)
        }
    })
    .is_break()
}

/// Helper to check if a block calls effectful functions (for handler arm checking).
fn block_calls_effectful_inner(
    ctx: &IrContext,
    block: trunk_ir::arena::refs::BlockRef,
    effectful: &HashSet<Symbol>,
) -> bool {
    for &op in &ctx.block(block).ops {
        // Check direct calls to effectful functions
        if let Ok(call) = arena_func::Call::from_op(ctx, op)
            && effectful.contains(&call.callee(ctx))
        {
            return true;
        }
        // Skip nested function definitions
        if arena_func::Func::from_op(ctx, op).is_ok() {
            continue;
        }
        // Skip push_prompt body
        if arena_cont::PushPrompt::from_op(ctx, op).is_ok() {
            continue;
        }
        // Recursively check nested regions
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
