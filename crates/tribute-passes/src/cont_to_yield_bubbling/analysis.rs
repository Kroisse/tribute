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
use trunk_ir::dialect::cont;
use trunk_ir::dialect::func;
use trunk_ir::location::Span;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
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
    use std::ops::ControlFlow;

    let _ = walk::walk_region::<()>(ctx, region, &mut |op| {
        let Ok(func) = func::Func::from_op(ctx, op) else {
            return ControlFlow::Continue(walk::WalkAction::Advance);
        };
        let func_name = func.sym_name(ctx);
        if !effectful_funcs.contains(&func_name) {
            return ControlFlow::Continue(walk::WalkAction::Advance);
        }
        let Some(func_analysis) = FunctionAnalysis::analyze(ctx, func.body(ctx)) else {
            return ControlFlow::Continue(walk::WalkAction::Advance);
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
        ControlFlow::Continue(walk::WalkAction::Advance)
    });
}

/// Collect spans of handler_dispatch operations inside effectful functions.
pub(crate) fn collect_handlers_in_effectful_funcs(
    ctx: &IrContext,
    module_body: RegionRef,
    effectful_funcs: &HashSet<Symbol>,
) -> HashSet<Span> {
    use std::ops::ControlFlow;

    let mut handler_spans = HashSet::new();

    // Find effectful func.func ops, then walk their bodies for handler_dispatch ops.
    let _ = walk::walk_region::<()>(ctx, module_body, &mut |op| {
        let Ok(func) = func::Func::from_op(ctx, op) else {
            return ControlFlow::Continue(walk::WalkAction::Advance);
        };
        if !effectful_funcs.contains(&func.sym_name(ctx)) {
            return ControlFlow::Continue(walk::WalkAction::Advance);
        }
        // Walk the effectful function's body for handler_dispatch ops
        let _ = walk::walk_region::<()>(ctx, func.body(ctx), &mut |inner_op| {
            if cont::HandlerDispatch::from_op(ctx, inner_op).is_ok() {
                handler_spans.insert(ctx.op(inner_op).location.span);
            }
            ControlFlow::Continue(walk::WalkAction::Advance)
        });
        ControlFlow::Continue(walk::WalkAction::Advance)
    });

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
    use std::ops::ControlFlow;

    let _ = walk::walk_region::<()>(ctx, region, &mut |op| {
        let Ok(func) = func::Func::from_op(ctx, op) else {
            return ControlFlow::Continue(walk::WalkAction::Advance);
        };
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
        ControlFlow::Continue(walk::WalkAction::Advance)
    });
}

/// Check if a region (non-recursively into nested funcs) contains a `cont.push_prompt`.
fn region_contains_push_prompt(ctx: &IrContext, region: RegionRef) -> bool {
    use std::ops::ControlFlow;

    walk::walk_region::<()>(ctx, region, &mut |op| {
        if func::Func::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if cont::PushPrompt::from_op(ctx, op).is_ok() {
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
        if func::Func::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if cont::PushPrompt::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if cont::HandlerDispatch::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if cont::Resume::from_op(ctx, op).is_ok() {
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
///
/// Skips nested `func.func` and `cont.push_prompt` (they are separate scopes).
/// For `cont.handler_dispatch`, walks into the body regions of child
/// `cont.done`/`cont.suspend`/`cont.yield` ops but skips the dispatch itself.
pub(crate) fn calls_effectful_function(
    ctx: &IrContext,
    region: RegionRef,
    effectful: &HashSet<Symbol>,
) -> bool {
    use std::ops::ControlFlow;

    /// Check if a single op is an effectful call/call_indirect.
    fn is_effectful_call(ctx: &IrContext, op: OpRef, effectful: &HashSet<Symbol>) -> bool {
        if let Ok(call) = func::Call::from_op(ctx, op) {
            return effectful.contains(&call.callee(ctx));
        }
        if func::CallIndirect::from_op(ctx, op).is_ok() {
            let operands = ctx.op_operands(op).to_vec();
            if let Some(&callee) = operands.first() {
                return has_effectful_type(ctx, ctx.value_ty(callee));
            }
        }
        false
    }

    walk::walk_region::<()>(ctx, region, &mut |op| {
        if func::Func::from_op(ctx, op).is_ok() || cont::PushPrompt::from_op(ctx, op).is_ok() {
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if let Ok(dispatch) = cont::HandlerDispatch::from_op(ctx, op) {
            // Walk into child ops' body regions (done/suspend/yield)
            let body_region = dispatch.body(ctx);
            for &block in &ctx.region(body_region).blocks {
                for &child_op in &ctx.block(block).ops {
                    // Each cont.done/cont.suspend/cont.yield has a body region
                    for &child_region in &ctx.op(child_op).regions {
                        let found = walk::walk_region::<()>(ctx, child_region, &mut |inner_op| {
                            if is_effectful_call(ctx, inner_op, effectful) {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(walk::WalkAction::Advance)
                            }
                        })
                        .is_break();
                        if found {
                            return ControlFlow::Break(());
                        }
                    }
                }
            }
            return ControlFlow::Continue(walk::WalkAction::Skip);
        }
        if is_effectful_call(ctx, op, effectful) {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(walk::WalkAction::Advance)
    })
    .is_break()
}
