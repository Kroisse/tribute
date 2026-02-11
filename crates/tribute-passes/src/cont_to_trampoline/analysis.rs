use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::live_vars::FunctionAnalysis;
use trunk_ir::dialect::cont;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func::{self, Func};
use trunk_ir::{Block, DialectOp, DialectType, Region, Span, Symbol, Type};

use super::{ShiftAnalysis, ShiftPointInfo};

// ============================================================================
// Shift Point Analysis
// ============================================================================

/// Analyze all effectful functions for shift points.
/// Returns a map from shift operation span to shift point info.
pub(crate) fn analyze_shift_points<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> ShiftAnalysis<'db> {
    let mut analysis = HashMap::new();

    // Recursively walk through all functions (including those in nested regions)
    analyze_shift_points_in_region(db, &module.body(db), effectful_funcs, &mut analysis);

    tracing::debug!(
        "analyze_shift_points: found {} shift points",
        analysis.len()
    );
    Rc::new(analysis)
}

/// Helper to recursively analyze shift points in a region.
fn analyze_shift_points_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    effectful_funcs: &HashSet<Symbol>,
    analysis: &mut HashMap<Span, ShiftPointInfo<'db>>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func) = Func::from_operation(db, *op) {
                let func_name = func.sym_name(db);
                if effectful_funcs.contains(&func_name) {
                    // Analyze this effectful function
                    let body = func.body(db);
                    if let Some(func_analysis) = FunctionAnalysis::analyze(db, &body) {
                        let total_shifts = func_analysis.shift_points.len();
                        for shift_point in func_analysis.shift_points {
                            let span = shift_point.shift_op.location(db).span;
                            // Get the shift result value and type if the operation has results
                            let (shift_result_value, shift_result_type) = if let Some(result_type) =
                                shift_point.shift_op.results(db).first()
                            {
                                (Some(shift_point.shift_op.result(db, 0)), Some(*result_type))
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
            for nested_region in op.regions(db).iter() {
                analyze_shift_points_in_region(db, nested_region, effectful_funcs, analysis);
            }
        }
    }
}

/// Collect spans of handler_dispatch operations inside effectful functions.
/// These handlers should return Step type (to be propagated by the effectful function).
pub(crate) fn collect_handlers_in_effectful_funcs<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> HashSet<Span> {
    let mut handler_spans = HashSet::new();

    fn collect_handlers_in_region<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        handler_spans: &mut HashSet<Span>,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                // Check if this is a handler_dispatch
                if cont::HandlerDispatch::from_operation(db, *op).is_ok() {
                    handler_spans.insert(op.location(db).span);
                }
                // Recursively check nested regions
                for region in op.regions(db).iter() {
                    collect_handlers_in_region(db, region, handler_spans);
                }
            }
        }
    }

    // Helper to recursively find and process effectful functions
    fn find_effectful_funcs_and_collect<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        effectful_funcs: &HashSet<Symbol>,
        handler_spans: &mut HashSet<Span>,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                if let Ok(func) = Func::from_operation(db, *op) {
                    let func_name = func.sym_name(db);
                    if effectful_funcs.contains(&func_name) {
                        // Collect handler_dispatch spans in this effectful function
                        collect_handlers_in_region(db, &func.body(db), handler_spans);
                    }
                }
                // Recursively check nested regions for more functions
                for nested_region in op.regions(db).iter() {
                    find_effectful_funcs_and_collect(
                        db,
                        nested_region,
                        effectful_funcs,
                        handler_spans,
                    );
                }
            }
        }
    }

    // Walk through all functions (including those in nested regions)
    find_effectful_funcs_and_collect(db, &module.body(db), effectful_funcs, &mut handler_spans);

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
pub(crate) fn identify_effectful_functions<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<'db>,
) -> Rc<HashSet<Symbol>> {
    let mut effectful = HashSet::new();
    let mut all_funcs: Vec<(Symbol, Region<'db>)> = Vec::new();

    // First pass: identify direct effectful functions and collect all functions
    collect_direct_effectful_funcs(db, &module.body(db), &mut effectful, &mut all_funcs);

    // Second pass: propagate effectfulness through the call graph
    // Keep iterating until no new effectful functions are found
    let mut changed = true;
    while changed {
        changed = false;
        for (func_name, body) in &all_funcs {
            if effectful.contains(func_name) {
                continue;
            }
            if calls_effectful_function(db, body, &effectful) {
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
fn collect_direct_effectful_funcs<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    effectful: &mut HashSet<Symbol>,
    all_funcs: &mut Vec<(Symbol, Region<'db>)>,
) {
    for block in region.blocks(db).iter() {
        tracing::trace!(
            "collect_direct_effectful_funcs: block has {} operations",
            block.operations(db).len()
        );
        for op in block.operations(db).iter() {
            tracing::trace!(
                "collect_direct_effectful_funcs: op {}.{}",
                op.dialect(db),
                op.name(db)
            );
            if let Ok(func) = Func::from_operation(db, *op) {
                let func_name = func.sym_name(db);
                let body = func.body(db);
                tracing::trace!(
                    "collect_direct_effectful_funcs: found func.func '{}'",
                    func_name
                );

                all_funcs.push((func_name, body));

                // Check the function's type signature for effectfulness
                let func_ty = func.ty(db);
                let is_effectful = has_effectful_type(db, func_ty);
                tracing::trace!(
                    "collect_direct_effectful_funcs: '{}' type={:?}, is_effectful={}",
                    func_name,
                    func_ty,
                    is_effectful
                );
                if is_effectful {
                    effectful.insert(func_name);
                }
            }

            // Recursively check nested regions
            for nested_region in op.regions(db).iter() {
                collect_direct_effectful_funcs(db, nested_region, effectful, all_funcs);
            }
        }
    }
}

/// Check if a function type has a non-empty effect row.
///
/// Returns true if the effect row:
/// - Has any concrete abilities, OR
/// - Has a tail variable (is polymorphic)
fn has_effectful_type<'db>(db: &'db dyn salsa::Database, func_ty: Type<'db>) -> bool {
    let Some(func) = core::Func::from_type(db, func_ty) else {
        return false;
    };
    let Some(effect) = func.effect(db) else {
        return false;
    };
    let Some(row) = core::EffectRowType::from_type(db, effect) else {
        return false;
    };
    let abilities = row.abilities(db);
    let tail_var = row.tail_var(db);
    !abilities.is_empty() || tail_var.is_some()
}

/// Check if a region calls any effectful function.
/// NOTE: Calls inside push_prompt body are skipped (handled by the handler),
/// but calls in handler_dispatch ARMS are checked (they may return Step).
pub(crate) fn calls_effectful_function<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    effectful: &HashSet<Symbol>,
) -> bool {
    use std::ops::ControlFlow;
    use trunk_ir::{OperationWalk, WalkAction};

    region
        .walk_all::<()>(db, |op| {
            // Skip nested function definitions - they're analyzed separately
            if Func::from_operation(db, op).is_ok() {
                return ControlFlow::Continue(WalkAction::Skip);
            }
            // Skip push_prompt body - effects there are handled by the enclosing handler
            if cont::PushPrompt::from_operation(db, op).is_ok() {
                return ControlFlow::Continue(WalkAction::Skip);
            }
            // For handler_dispatch: check if handler ARMS call effectful functions
            // Handler arms can call effectful functions that return Step
            if let Ok(dispatch) = cont::HandlerDispatch::from_operation(db, op) {
                let body_region = dispatch.body(db);
                // Skip block 0 (done case), check suspend arms (blocks 1+)
                for block in body_region.blocks(db).iter().skip(1) {
                    if block_calls_effectful_inner(db, block, effectful) {
                        return ControlFlow::Break(());
                    }
                }
                return ControlFlow::Continue(WalkAction::Skip);
            }
            if let Ok(call) = func::Call::from_operation(db, op)
                && effectful.contains(&call.callee(db))
            {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(WalkAction::Advance)
            }
        })
        .is_break()
}

/// Helper to check if a block calls effectful functions (for handler arm checking).
/// Only returns true if the block calls a function that is already known to be effectful.
/// This is used for propagating effectfulness through the call graph.
fn block_calls_effectful_inner<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    effectful: &HashSet<Symbol>,
) -> bool {
    for op in block.operations(db).iter() {
        // Check direct calls to effectful functions
        if let Ok(call) = func::Call::from_operation(db, *op)
            && effectful.contains(&call.callee(db))
        {
            return true;
        }
        // Recursively check nested regions (but skip nested functions and push_prompt bodies)
        if Func::from_operation(db, *op).is_ok() {
            continue; // Skip nested function definitions
        }
        if cont::PushPrompt::from_operation(db, *op).is_ok() {
            continue; // Skip push_prompt body — effects there are handled by the enclosing handler
        }
        for region in op.regions(db).iter() {
            for nested_block in region.blocks(db).iter() {
                if block_calls_effectful_inner(db, nested_block, effectful) {
                    return true;
                }
            }
        }
    }
    false
}

// NOTE: handler_dispatch is NOT considered effectful here because:
// - For closed handlers, the trampoline loop returns user_result_ty (not Step)
// - The function's return type should remain user_result_ty
// - Open handlers (which propagate effects) are not yet supported
//
// If we need to support open handlers in the future, we'll need a different
// approach (e.g., an attribute on the handler to indicate open vs closed).
