//! Lower cont dialect operations to trampoline dialect.
//!
//! This pass transforms continuation operations to trampoline operations:
//! - `cont.shift` → build_state + build_continuation + set_yield_state + step_shift
//! - `cont.resume` → reset_yield_state + continuation_get + build_resume_wrapper + call
//! - `cont.push_prompt` → trampoline yield check loop + dispatch
//! - `cont.handler_dispatch` → yield check + multi-arm dispatch
//! - `cont.get_continuation` → `trampoline.get_yield_continuation`
//! - `cont.get_shift_value` → `trampoline.get_yield_shift_value`
//! - `cont.get_done_value` → `trampoline.step_get(field="value")`
//! - `cont.drop` → pass through (handled later)
//!
//! This pass is backend-agnostic and should run after `tribute_to_cont`/`handler_lower`
//! and before `trampoline_to_adt`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::live_vars::FunctionAnalysis;
use tribute_ir::dialect::{adt, trampoline, tribute_rt};
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func::{self, Func};
use trunk_ir::dialect::{arith, cont, scf};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Span, Symbol,
    Type, Value,
};

// ============================================================================
// Public API
// ============================================================================

/// Create a TypeConverter with standard tribute_rt -> core type conversions.
fn standard_type_converter() -> TypeConverter {
    TypeConverter::new()
        .add_conversion(|db, ty| {
            tribute_rt::Int::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Nat::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Bool::from_type(db, ty).map(|_| core::I::<1>::new(db).as_type())
        })
        .add_conversion(|db, ty| {
            tribute_rt::Float::from_type(db, ty).map(|_| core::F64::new(db).as_type())
        })
}

/// Metadata for generating resume functions with continuation code.
struct ResumeFuncSpec<'db> {
    /// Name of the resume function
    name: String,
    /// State struct type (used to extract captured values)
    state_type: Type<'db>,
    /// Fields in the state struct (field_name, field_type)
    state_fields: Vec<(Symbol, Type<'db>)>,
    /// Original values that were captured (for value remapping)
    original_live_values: Vec<Value<'db>>,
    /// The original shift result value (maps to resume_value)
    shift_result_value: Option<Value<'db>>,
    /// The type of the shift result (used to cast resume_value)
    shift_result_type: Option<Type<'db>>,
    /// Operations that form the continuation (ops after shift)
    continuation_ops: Vec<Operation<'db>>,
    /// Name of next resume function (if not last)
    next_resume_name: Option<String>,
    /// Location for generating code
    location: Location<'db>,
    /// Shift analysis for handling nested shifts in continuation
    shift_analysis: ShiftAnalysis<'db>,
}

/// Shared storage for resume function specs during pattern matching.
type ResumeSpecs<'db> = Rc<RefCell<Vec<ResumeFuncSpec<'db>>>>;

/// Shared counter for generating unique resume function names.
type ResumeCounter = Rc<RefCell<u32>>;

/// Analysis results for shift points, keyed by shift operation's span.
/// Using Span as key because Operation identity may not be stable across phases.
type ShiftAnalysis<'db> = Rc<HashMap<Span, ShiftPointInfo<'db>>>;

/// Information about a shift point for code generation.
#[derive(Clone)]
struct ShiftPointInfo<'db> {
    /// Index of this shift point in the function (0, 1, 2, ...)
    index: usize,
    /// Total number of shift points in the function
    total_shifts: usize,
    /// Live values at this shift point (defined before, used after) with their types
    live_values: Vec<(Value<'db>, Type<'db>)>,
    /// The result value of the shift operation (maps to resume_value)
    shift_result_value: Option<Value<'db>>,
    /// The type of the shift result (for casting resume_value)
    shift_result_type: Option<Type<'db>>,
    /// Operations after this shift until next shift or function end
    continuation_ops: Vec<Operation<'db>>,
}

/// Update type signatures for effectful functions.
///
/// This pass updates:
/// - Function return types: effectful functions return `trampoline.step` instead of their original type
/// - Function call result types: calls to effectful functions return `trampoline.step`
/// - `scf.if` result types: when branches call effectful functions
/// - `cont.push_prompt` result types: always returns `trampoline.step`
///
/// This MUST run before `lower_cont_to_trampoline` to ensure correct type information
/// is available when lowering patterns run.
pub fn update_effectful_types<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    // Step 1: Identify effectful functions (before transformation)
    let effectful_funcs = identify_effectful_functions(db, &module);

    // Step 2: Update type signatures
    // NOTE: We only update push_prompt and handler_dispatch result types to Step.
    // Effectful functions (like counter) do NOT have their return types changed to Step.
    // When a shift occurs inside an effectful function, push_prompt catches it and returns Step.
    // The effectful function itself never completes normally when a shift happens.
    let empty_target = ConversionTarget::new();
    let applicator = PatternApplicator::new(TypeConverter::new())
        // NOTE: These type update patterns are intentionally disabled for closed handlers.
        //
        // Why disabled:
        // 1. UpdateFuncTypePattern/UpdateFuncCallResultTypePattern:
        //    - Effectful functions should keep their original return types
        //    - push_prompt handles the Step conversion, not the effectful function itself
        //    - When shift occurs, control jumps to push_prompt, not back to the caller
        //
        // 2. UpdateScfIfTypePattern:
        //    - Closed handlers have their own trampoline loop that processes Step internally
        //    - The handler returns user_result_ty, not Step
        //    - Changing scf.if result types to Step would cause type mismatches
        //
        // NOTE: All type update patterns are now disabled because:
        // - tribute_to_cont now sets push_prompt result type to Step directly
        // - handler_dispatch operand is already Step-typed
        // - Closed handlers process Step internally and return user_result_ty
        //
        // .add_pattern(UpdateFuncTypePattern { effectful_funcs: effectful_funcs.clone() })
        // .add_pattern(UpdateFuncCallResultTypePattern { effectful_funcs: effectful_funcs.clone() })
        // .add_pattern(UpdateScfIfTypePattern { effectful_funcs: effectful_funcs.clone() })
        // .add_pattern(UpdatePushPromptResultTypePattern)
        ;
    // NOTE: UpdateHandlerDispatchResultTypePattern is disabled for closed handlers.
    // Closed handlers have a trampoline loop that returns user_result_ty directly,
    // so handler_dispatch should also return user_result_ty (not Step).
    // Open handlers (not yet supported) would need this pattern enabled.
    // .add_pattern(UpdateHandlerDispatchResultTypePattern)
    let _ = effectful_funcs; // suppress unused warning
    let result = applicator.apply_partial(db, module, empty_target);

    result.module
}

/// Lower cont dialect operations to trampoline dialect.
///
/// This pass transforms:
/// - `cont.shift` → state capture + continuation build + `trampoline.step_shift`
/// - `cont.resume` → continuation extraction + resume wrapper call
/// - `cont.push_prompt` → yield check + dispatch
/// - `cont.handler_dispatch` → yield check + multi-arm dispatch
/// - `cont.get_continuation` → `trampoline.get_yield_continuation`
/// - `cont.get_shift_value` → `trampoline.get_yield_shift_value`
/// - `cont.get_done_value` → `trampoline.step_get(field="value")`
///
/// IMPORTANT: `update_effectful_types` MUST be called before this pass to ensure
/// correct type information is available.
///
/// Returns an error if any `cont.*` operations (except `cont.drop`) remain after conversion.
pub fn lower_cont_to_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Result<Module<'db>, ConversionError> {
    // Shared state for resume function generation (no global state!)
    let resume_specs: ResumeSpecs<'db> = Rc::new(RefCell::new(Vec::new()));
    let resume_counter: ResumeCounter = Rc::new(RefCell::new(0));

    // Step 1: Identify effectful functions (types are already updated by update_effectful_types)
    let effectful_funcs = identify_effectful_functions(db, &module);

    // Step 2: Analyze shift points in effectful functions
    let shift_analysis = analyze_shift_points(db, &module, &effectful_funcs);

    // Step 2.5: Collect handler_dispatch ops that are inside effectful functions
    // These handlers should return Step type (for propagation up the call stack)
    let handlers_in_effectful_funcs =
        collect_handlers_in_effectful_funcs(db, &module, &effectful_funcs);

    // Step 3: Lower cont.* operations to trampoline.*
    let empty_target = ConversionTarget::new();
    let applicator = PatternApplicator::new(standard_type_converter())
        .add_pattern(LowerShiftPattern {
            resume_specs: Rc::clone(&resume_specs),
            resume_counter: Rc::clone(&resume_counter),
            shift_analysis: Rc::clone(&shift_analysis),
        })
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerGetContinuationPattern)
        .add_pattern(LowerGetShiftValuePattern)
        .add_pattern(LowerGetDoneValuePattern)
        .add_pattern(UpdateEffectfulCallResultTypePattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
        })
        .add_pattern(UpdateScfIfResultTypePattern)
        .add_pattern(UpdateScfYieldToStepPattern)
        .add_pattern(LowerPushPromptPattern)
        .add_pattern(LowerHandlerDispatchPattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
            handlers_in_effectful_funcs: Rc::new(handlers_in_effectful_funcs),
        });

    let result = applicator.apply_partial(db, module, empty_target.clone());

    // Step 3.5: Truncate effectful function bodies after step_shift
    // After LowerShiftPattern, continuation ops are stored in ResumeFuncSpec but still
    // remain in the original function body. Remove them so only step_shift and its
    // return statement remain.
    let module = truncate_after_shift(db, result.module, &effectful_funcs);

    // Step 4: Wrap returns in effectful functions with step_done
    let applicator = PatternApplicator::new(standard_type_converter()).add_pattern(
        WrapReturnsInEffectfulFuncsPattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
        },
    );
    let result = applicator.apply_partial(db, module, empty_target);
    let module = result.module;

    // Step 5: Wrap main with global trampoline (_start)
    // This handles the case where main returns Step (effectful main)
    let module = wrap_main_with_global_trampoline(db, module, &effectful_funcs);

    // Verify all cont.* ops (except cont.drop) are converted
    let conversion_target = ConversionTarget::new()
        .illegal_dialect("cont")
        .legal_op("cont", "drop");

    // Generate resume functions from collected specs
    let specs = resume_specs.borrow();
    if specs.is_empty() {
        conversion_target.verify(db, &module)?;
        return Ok(module);
    }

    let _location = module.location(db);
    let resume_funcs: Vec<Operation<'db>> = specs
        .iter()
        .map(|spec| create_resume_function_with_continuation(db, spec))
        .collect();

    // Add resume functions to module body
    let body = module.body(db);
    let mut blocks: Vec<Block<'db>> = body.blocks(db).iter().copied().collect();
    if let Some(block) = blocks.first_mut() {
        let mut ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();
        ops.extend(resume_funcs);
        *block = Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            IdVec::from(ops),
        );
    }

    let new_body = Region::new(db, body.location(db), IdVec::from(blocks));
    let module = Module::create(db, module.location(db), module.name(db), new_body);

    conversion_target.verify(db, &module)?;
    Ok(module)
}

// ============================================================================
// Shift Point Analysis
// ============================================================================

/// Analyze all effectful functions for shift points.
/// Returns a map from shift operation span to shift point info.
fn analyze_shift_points<'db>(
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
fn collect_handlers_in_effectful_funcs<'db>(
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
fn truncate_after_shift<'db>(
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
        if let Some(func_ty) = core::Func::from_type(db, old_func_ty) {
            let params = func_ty.params(db);
            let effect = func_ty.effect(db);
            let new_func_ty = core::Func::with_effect(db, params, step_ty, effect);
            builder = builder.attr(Symbol::new("type"), Attribute::Type(new_func_ty.as_type()));
            tracing::debug!(
                "truncate_func_after_shift: {} changed return type to Step",
                func_name
            );
        }
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
fn truncate_scf_if_branch<'db>(
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
fn truncate_scf_if_block<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    effectful_funcs: &HashSet<Symbol>,
    step_ty: Type<'db>,
) -> Block<'db> {
    let ops = block.operations(db);
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    let mut step_value: Option<Value<'db>> = None;

    for op in ops.iter() {
        // Skip scf.yield - we'll add our own at the end
        if scf::Yield::from_operation(db, *op).is_ok() {
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
    }

    Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops.into(),
    )
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
    let mut effect_location: Option<Location<'db>> = None;

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
            // Skip all operations after effect point (they're now in resume functions)
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
            let step_ty = trampoline::Step::new(db).as_type();
            let new_result_types = if !op.results(db).is_empty() {
                IdVec::from(vec![step_ty])
            } else {
                op.results(db).clone()
            };
            let new_op = Operation::new(
                db,
                op.location(db),
                op.dialect(db),
                op.name(db),
                op.operands(db).clone(),
                new_result_types,
                op.attributes(db).clone(),
                op.regions(db).clone(),
                op.successors(db).clone(),
            );
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
            let step_ty = trampoline::Step::new(db).as_type();
            let new_result_types = if !op.results(db).is_empty() {
                IdVec::from(vec![step_ty])
            } else {
                op.results(db).clone()
            };
            let new_op = Operation::new(
                db,
                op.location(db),
                op.dialect(db),
                op.name(db),
                op.operands(db).clone(),
                new_result_types,
                op.attributes(db).clone(),
                op.regions(db).clone(),
                op.successors(db).clone(),
            );
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

// ============================================================================
// Effectful Function Analysis
// ============================================================================

/// Identify all effectful functions in the module.
/// A function is effectful if it contains `cont.shift` or `cont.push_prompt` operations,
/// or if it calls another effectful function (transitive closure).
fn identify_effectful_functions<'db>(
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
                if has_effectful_type(db, func.ty(db)) {
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
fn calls_effectful_function<'db>(
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
        // Recursively check nested regions (but skip nested functions)
        if Func::from_operation(db, *op).is_ok() {
            continue; // Skip nested function definitions
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
// ============================================================================
// Resume Function Helpers
// ============================================================================

/// Generate a unique resume function name using the shared counter.
fn fresh_resume_name(counter: &ResumeCounter) -> String {
    let mut counter = counter.borrow_mut();
    let id = *counter;
    *counter += 1;
    format!("__resume_{}", id)
}

/// Generate a unique state type name based on ability, operation info, and shift index.
/// The shift_index ensures different shift points for the same ability/op have distinct state types.
fn state_type_name(
    ability_name: Option<Symbol>,
    op_name: Option<Symbol>,
    tag: u32,
    shift_index: usize,
) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    if let Some(ability) = ability_name {
        ability.to_string().hash(&mut hasher);
    }
    if let Some(name) = op_name {
        name.to_string().hash(&mut hasher);
    }
    tag.hash(&mut hasher);
    shift_index.hash(&mut hasher);

    let hash = hasher.finish();
    format!("__State_{:x}", hash & 0xFFFFFF)
}

// ============================================================================
// Pattern: Lower cont.shift
// ============================================================================

struct LowerShiftPattern<'db> {
    resume_specs: ResumeSpecs<'db>,
    resume_counter: ResumeCounter,
    shift_analysis: ShiftAnalysis<'db>,
}

impl<'db> RewritePattern<'db> for LowerShiftPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let shift_op = match cont::Shift::from_operation(db, *op) {
            Ok(s) => s,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let tag = shift_op.tag(db);

        // Compute op_idx from ability_ref and op_name
        let ability_ref_type = shift_op.ability_ref(db);
        let ability_name =
            core::AbilityRefType::from_type(db, ability_ref_type).and_then(|ar| ar.name(db));
        let op_name = Some(shift_op.op_name(db));
        let op_idx = compute_op_idx(ability_name, op_name);

        // Look up shift point analysis - fail fast if missing
        let shift_info = self.shift_analysis.get(&location.span).unwrap_or_else(|| {
            panic!(
                "missing shift analysis for cont.shift at {:?} (ability: {:?}, op: {:?})",
                location.span, ability_name, op_name
            )
        });

        let mut ops = Vec::new();

        // === 1. Build State Struct with live values ===
        let state_name = Symbol::from_dynamic(&state_type_name(
            ability_name,
            op_name,
            tag,
            shift_info.index,
        ));

        // Get live values and their types from analysis
        let (state_values, state_fields): (Vec<Value<'db>>, Vec<(Symbol, Type<'db>)>) = shift_info
            .live_values
            .iter()
            .enumerate()
            .map(|(i, (v, ty))| {
                let field_name = Symbol::from_dynamic(&format!("field_{}", i));
                (*v, (field_name, *ty))
            })
            .unzip();

        let state_adt_ty = adt::struct_type(db, state_name, state_fields.clone());
        let state_op = trampoline::build_state(
            db,
            location,
            state_values.clone(),
            state_adt_ty,
            state_adt_ty,
        );
        let state_val = state_op.as_operation().result(db, 0);
        ops.push(state_op.as_operation());

        // === 2. Get resume function reference (i32 table index) ===
        let i32_ty = core::I32::new(db).as_type();
        let resume_name = fresh_resume_name(&self.resume_counter);

        // Determine next resume name if not the last shift point
        let next_resume_name = if shift_info.index + 1 >= shift_info.total_shifts {
            None
        } else {
            // Pre-compute next resume function name
            let counter = self.resume_counter.borrow();
            let next_id = *counter; // This will be the next ID
            Some(format!("__resume_{}", next_id))
        };

        // Record resume function spec with continuation info
        self.resume_specs.borrow_mut().push(ResumeFuncSpec {
            name: resume_name.clone(),
            state_type: state_adt_ty,
            state_fields,
            original_live_values: state_values.clone(),
            shift_result_value: shift_info.shift_result_value,
            shift_result_type: shift_info.shift_result_type,
            continuation_ops: shift_info.continuation_ops.clone(),
            next_resume_name,
            location,
            shift_analysis: self.shift_analysis.clone(),
        });

        let resume_name_sym = Symbol::from_dynamic(&resume_name);
        let const_op = func::constant(db, location, i32_ty, resume_name_sym);
        let resume_fn_val = const_op.as_operation().result(db, 0);
        ops.push(const_op.as_operation());

        // === 3. Get shift value (the value passed to the effect operation) ===
        // Note: shift value may be absent if the ability operation has no arguments.
        // In that case, we use state_val as a placeholder (will be ignored by resume).
        let shift_value_val = adaptor.operands().first().copied().unwrap_or(state_val);

        // === 4. Build Continuation ===
        let cont_ty = trampoline::Continuation::new(db).as_type();
        let cont_op = trampoline::build_continuation(
            db,
            location,
            resume_fn_val,
            state_val,
            shift_value_val,
            cont_ty,
            tag,
            op_idx,
        );
        let cont_val = cont_op.as_operation().result(db, 0);
        ops.push(cont_op.as_operation());

        // === 5. Set Yield State ===
        let set_yield_op = trampoline::set_yield_state(db, location, cont_val, tag, op_idx);
        ops.push(set_yield_op.as_operation());

        // === 6. Return Step::Shift ===
        let step_ty = trampoline::Step::new(db).as_type();
        let step_op = trampoline::step_shift(db, location, cont_val, step_ty, tag, op_idx);
        ops.push(step_op.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Create a resume function with continuation code.
///
/// The resume function:
/// 1. Extracts state and resume_value from the wrapper
/// 2. Restores captured local values from state
/// 3. Executes the continuation operations (with value remapping)
/// 4. Returns the final result (or yields again for chained shifts)
///
/// NOTE: Resume functions take evidence as first parameter for calling convention
/// consistency with lifted lambdas. All functions in the function table use the
/// same signature: (evidence, env/wrapper, args...). The evidence parameter may
/// be unused in resume functions but is required for call_indirect type compatibility.
fn create_resume_function_with_continuation<'db>(
    db: &'db dyn salsa::Database,
    spec: &ResumeFuncSpec<'db>,
) -> Operation<'db> {
    let evidence_ty = tribute_ir::dialect::ability::EvidencePtr::new(db).as_type();
    let wrapper_ty = trampoline::ResumeWrapper::new(db).as_type();
    let step_ty = trampoline::Step::new(db).as_type();
    let anyref_ty = tribute_rt::Any::new(db).as_type();
    let location = spec.location;
    let name = Symbol::from_dynamic(&spec.name);

    // Resume functions take (evidence, wrapper as anyref) for calling convention consistency
    // with lifted lambdas. Using anyref allows uniform call_indirect type.
    // The function casts anyref to the specific wrapper_ty at the start.
    let func_op = Func::build(
        db,
        location,
        name,
        IdVec::from(vec![evidence_ty, anyref_ty]),
        step_ty,
        |builder| {
            // Evidence is at index 0 (unused but required for calling convention)
            // Wrapper is at index 1 as anyref, needs cast to specific type
            let wrapper_anyref = builder.block_arg(db, 1);
            let wrapper_cast = builder.op(core::unrealized_conversion_cast(
                db,
                location,
                wrapper_anyref,
                wrapper_ty,
            ));
            let wrapper_arg = wrapper_cast.result(db);

            // Build value map for remapping
            let mut value_map: HashMap<Value<'db>, Value<'db>> = HashMap::new();

            // Extract resume_value from wrapper
            let get_resume_value = builder.op(trampoline::resume_wrapper_get(
                db,
                location,
                wrapper_arg,
                anyref_ty,
                Symbol::new("resume_value"),
            ));
            let mut resume_value = get_resume_value.result(db);

            // Cast resume_value to the original shift result type if needed
            if let Some(result_type) = spec.shift_result_type {
                // Insert unrealized_conversion_cast to convert anyref -> original type
                let cast_op = builder.op(core::unrealized_conversion_cast(
                    db,
                    location,
                    resume_value,
                    result_type,
                ));
                resume_value = cast_op.result(db);
            }

            // Map shift result to resume_value (now properly typed)
            if let Some(shift_result) = spec.shift_result_value {
                value_map.insert(shift_result, resume_value);
            }

            // If we have state fields, extract the state and get captured values
            if !spec.state_fields.is_empty() {
                let get_state = builder.op(trampoline::resume_wrapper_get(
                    db,
                    location,
                    wrapper_arg,
                    spec.state_type,
                    Symbol::new("state"),
                ));
                let state_val = get_state.result(db);

                // Extract each captured local from state and map to original value
                // Note: State fields are stored as anyref after lowering (in trampoline_to_wasm),
                // so we use anyref as the field type here and add a cast to the original type.
                let anyref_ty = tribute_rt::any_type(db);
                for (i, ((_field_name, field_type), original_value)) in spec
                    .state_fields
                    .iter()
                    .zip(spec.original_live_values.iter())
                    .enumerate()
                {
                    // Get field as anyref (consistent with how LowerBuildStatePattern stores it)
                    let get_field = builder.op(adt::struct_get(
                        db,
                        location,
                        state_val,
                        anyref_ty,
                        spec.state_type,
                        Attribute::IntBits(i as u64),
                    ));
                    let anyref_value = get_field.result(db);

                    // Cast anyref to the expected field type
                    let cast_op = builder.op(core::unrealized_conversion_cast(
                        db,
                        location,
                        anyref_value,
                        *field_type,
                    ));
                    let extracted_value = cast_op.result(db);
                    value_map.insert(*original_value, extracted_value);
                }
            }

            // Execute continuation operations with value remapping
            let mut last_result: Option<Value<'db>> = None;
            let mut encountered_shift = false;

            for op in &spec.continuation_ops {
                // Skip func.return - we'll handle the final return ourselves
                if func::Return::from_operation(db, *op).is_ok() {
                    // Get the return value and use it as last_result
                    if let Some(&return_val) = op.operands(db).first() {
                        last_result = Some(*value_map.get(&return_val).unwrap_or(&return_val));
                    }
                    continue;
                }

                // Handle cont.shift - transform to step_shift with next resume function
                if let Ok(shift_op) = cont::Shift::from_operation(db, *op) {
                    // Get next resume function name
                    let next_resume_name = spec.next_resume_name.as_ref().expect(
                        "encountered shift in continuation but no next_resume_name specified",
                    );

                    // Look up shift analysis for this shift point
                    let shift_span = op.location(db).span;
                    let next_shift_info = spec.shift_analysis.get(&shift_span);

                    // Get shift properties
                    let tag = shift_op.tag(db);
                    let ability_ref_type = shift_op.ability_ref(db);
                    let ability_name = core::AbilityRefType::from_type(db, ability_ref_type)
                        .and_then(|ar| ar.name(db));
                    let op_name = Some(shift_op.op_name(db));
                    let op_idx = compute_op_idx(ability_name, op_name);

                    // Build state struct with current live values (remapped)
                    let shift_index = next_shift_info.map(|info| info.index).unwrap_or(0);
                    let state_name = Symbol::from_dynamic(&state_type_name(
                        ability_name,
                        op_name,
                        tag,
                        shift_index,
                    ));
                    let (state_values, state_fields): (Vec<Value<'db>>, Vec<(Symbol, Type<'db>)>) =
                        if let Some(info) = next_shift_info {
                            info.live_values
                                .iter()
                                .enumerate()
                                .map(|(i, (v, ty))| {
                                    let remapped_v = *value_map.get(v).unwrap_or(v);
                                    let field_name = Symbol::from_dynamic(&format!("field_{}", i));
                                    (remapped_v, (field_name, *ty))
                                })
                                .unzip()
                        } else {
                            (vec![], vec![])
                        };

                    let state_adt_ty = adt::struct_type(db, state_name, state_fields);
                    let state_op = builder.op(trampoline::build_state(
                        db,
                        location,
                        state_values,
                        state_adt_ty,
                        state_adt_ty,
                    ));
                    let state_val = state_op.result(db);

                    // Get resume function reference (i32 table index)
                    let i32_ty = core::I32::new(db).as_type();
                    let resume_name_sym = Symbol::from_dynamic(next_resume_name);
                    let const_op =
                        builder.op(func::constant(db, location, i32_ty, resume_name_sym));
                    let resume_fn_val = const_op.result(db);

                    // Get shift value (remap if needed)
                    let shift_value_val = op
                        .operands(db)
                        .first()
                        .map(|&v| *value_map.get(&v).unwrap_or(&v))
                        .unwrap_or(state_val);

                    // Build continuation
                    let cont_ty = trampoline::Continuation::new(db).as_type();
                    let cont_op = builder.op(trampoline::build_continuation(
                        db,
                        location,
                        resume_fn_val,
                        state_val,
                        shift_value_val,
                        cont_ty,
                        tag,
                        op_idx,
                    ));
                    let cont_val = cont_op.result(db);

                    // Set yield state and return step_shift
                    builder.op(trampoline::set_yield_state(
                        db, location, cont_val, tag, op_idx,
                    ));
                    let step_shift = builder.op(trampoline::step_shift(
                        db, location, cont_val, step_ty, tag, op_idx,
                    ));
                    builder.op(func::r#return(db, location, Some(step_shift.result(db))));

                    encountered_shift = true;
                    break; // Stop processing further ops
                }

                // Remap operands
                let remapped_operands: IdVec<Value<'db>> = op
                    .operands(db)
                    .iter()
                    .map(|&v| *value_map.get(&v).unwrap_or(&v))
                    .collect();

                // Build new operation with remapped operands
                let new_op = if remapped_operands != *op.operands(db) {
                    let new_op = op.modify(db).operands(remapped_operands).build();
                    builder.op(new_op);
                    new_op
                } else {
                    builder.op(*op);
                    *op
                };

                // Map old results to new results
                for i in 0..op.results(db).len() {
                    let old_result = op.result(db, i);
                    let new_result = new_op.result(db, i);
                    if old_result != new_result {
                        value_map.insert(old_result, new_result);
                    }
                }

                // Track last result for final return
                if !new_op.results(db).is_empty() {
                    last_result = Some(new_op.result(db, 0));
                }
            }

            // Return Step.Done with the final result (only if no shift was encountered)
            if !encountered_shift {
                let final_value = last_result.unwrap_or(resume_value);
                let step_done =
                    builder.op(trampoline::step_done(db, location, final_value, step_ty));
                builder.op(func::r#return(db, location, Some(step_done.result(db))));
            }
        },
    );

    func_op.as_operation()
}

// ============================================================================
// Pattern: Lower cont.resume
// ============================================================================

struct LowerResumePattern;

impl<'db> RewritePattern<'db> for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::Resume::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let anyref_ty = tribute_rt::Any::new(db).as_type();

        let operands = adaptor.operands();
        let continuation = operands
            .first()
            .copied()
            .expect("resume requires continuation");
        let value = operands.get(1).copied();

        let mut ops = Vec::new();

        // === 1. Reset yield state ===
        ops.push(trampoline::reset_yield_state(db, location).as_operation());

        // === 2. Get resume_fn from continuation (i32 table index) ===
        let get_resume_fn = trampoline::continuation_get(
            db,
            location,
            continuation,
            i32_ty,
            Symbol::new("resume_fn"),
        );
        let resume_fn_val = get_resume_fn.as_operation().result(db, 0);
        ops.push(get_resume_fn.as_operation());

        // === 3. Get state from continuation ===
        let get_state = trampoline::continuation_get(
            db,
            location,
            continuation,
            anyref_ty,
            Symbol::new("state"),
        );
        let state_val = get_state.as_operation().result(db, 0);
        ops.push(get_state.as_operation());

        // === 4. Build resume wrapper ===
        let wrapper_ty = trampoline::ResumeWrapper::new(db).as_type();
        let resume_value = value.unwrap_or(state_val);

        let wrapper_op =
            trampoline::build_resume_wrapper(db, location, state_val, resume_value, wrapper_ty);
        let wrapper_val = wrapper_op.as_operation().result(db, 0);
        ops.push(wrapper_op.as_operation());

        // === 5. Call resume function ===
        // Resume functions take (evidence, wrapper) for calling convention consistency
        // with lifted lambdas. We pass null evidence since we're inside a handler arm.
        let evidence_ty = tribute_ir::dialect::ability::EvidencePtr::new(db).as_type();
        let null_evidence = adt::ref_null(db, location, evidence_ty, anyref_ty);
        ops.push(null_evidence.as_operation());
        let evidence_val = null_evidence.as_operation().result(db, 0);

        let step_ty = trampoline::Step::new(db).as_type();
        let call_op = func::call_indirect(
            db,
            location,
            resume_fn_val,
            IdVec::from(vec![evidence_val, wrapper_val]),
            step_ty,
        );
        ops.push(call_op.as_operation());

        RewriteResult::expand(ops)
    }
}

// ============================================================================
// Pattern: Lower cont.get_continuation
// ============================================================================

struct LowerGetContinuationPattern;

impl<'db> RewritePattern<'db> for LowerGetContinuationPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::GetContinuation::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let result_type = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| trampoline::Continuation::new(db).as_type());

        let trampoline_op = trampoline::get_yield_continuation(db, location, result_type);
        RewriteResult::Replace(trampoline_op.as_operation())
    }
}

// ============================================================================
// Pattern: Lower cont.get_shift_value
// ============================================================================

struct LowerGetShiftValuePattern;

impl<'db> RewritePattern<'db> for LowerGetShiftValuePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::GetShiftValue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let result_type = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(db).as_type());

        // get_yield_shift_value returns anyref
        let anyref_ty = tribute_rt::Any::new(db).as_type();
        let get_shift_op = trampoline::get_yield_shift_value(db, location, anyref_ty);
        let anyref_value = get_shift_op.as_operation().result(db, 0);

        // Insert explicit cast from anyref to the expected result type
        let cast_op = core::unrealized_conversion_cast(db, location, anyref_value, result_type);

        RewriteResult::expand(vec![get_shift_op.as_operation(), cast_op.as_operation()])
    }
}

// ============================================================================
// Pattern: Lower cont.get_done_value
// ============================================================================

struct LowerGetDoneValuePattern;

impl<'db> RewritePattern<'db> for LowerGetDoneValuePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::GetDoneValue::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let result_type = op
            .results(db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(db).as_type());

        let step_value = adaptor
            .operands()
            .first()
            .copied()
            .expect("get_done_value requires step operand");

        // Debug: log the step_value type and its source
        tracing::debug!(
            "LowerGetDoneValuePattern: step_value def = {:?}, index = {}",
            step_value.def(db),
            step_value.index(db)
        );
        if let Some(step_value_ty) = adaptor.operand_type(0) {
            tracing::debug!(
                "LowerGetDoneValuePattern: step_value type = {}.{}, result_type = {}.{}",
                step_value_ty.dialect(db),
                step_value_ty.name(db),
                result_type.dialect(db),
                result_type.name(db)
            );
        }
        // Also check the raw type from the operation's result
        if let trunk_ir::ValueDef::OpResult(source_op) = step_value.def(db)
            && let Some(raw_ty) = source_op.results(db).get(step_value.index(db))
        {
            tracing::debug!(
                "LowerGetDoneValuePattern: source op = {}.{}, raw result type = {}.{}",
                source_op.dialect(db),
                source_op.name(db),
                raw_ty.dialect(db),
                raw_ty.name(db)
            );
        }

        // step_get extracts the value field which is anyref
        let anyref_ty = tribute_rt::Any::new(db).as_type();
        let step_get_op =
            trampoline::step_get(db, location, step_value, anyref_ty, Symbol::new("value"));
        let anyref_value = step_get_op.as_operation().result(db, 0);

        // Insert explicit cast from anyref to the expected result type
        let cast_op = core::unrealized_conversion_cast(db, location, anyref_value, result_type);

        RewriteResult::expand(vec![step_get_op.as_operation(), cast_op.as_operation()])
    }
}

// ============================================================================
// Pattern: Update func.call result type for effectful functions
// ============================================================================

/// Pattern that updates func.call to effectful functions to return Step type.
/// This handles calls inside scf.if/scf.loop regions that weren't processed
/// by truncate_after_shift (which only processes function entry blocks).
struct UpdateEffectfulCallResultTypePattern {
    effectful_funcs: Rc<HashSet<Symbol>>,
}

impl<'db> RewritePattern<'db> for UpdateEffectfulCallResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle func.call operations
        let Ok(call) = func::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let callee = call.callee(db);

        // Skip if not an effectful function
        if !self.effectful_funcs.contains(&callee) {
            return RewriteResult::Unchanged;
        }

        // Skip if already returns Step
        let step_ty = trampoline::Step::new(db).as_type();
        if op
            .results(db)
            .first()
            .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
        {
            return RewriteResult::Unchanged;
        }

        // Skip if no results
        if op.results(db).is_empty() {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let original_result_ty = op.results(db).first().copied().unwrap();

        tracing::debug!(
            "UpdateEffectfulCallResultTypePattern: updating call to {} from {} to Step",
            callee,
            original_result_ty.name(db)
        );

        // Create new call with Step result type
        let new_call = Operation::new(
            db,
            location,
            op.dialect(db),
            op.name(db),
            adaptor.operands().clone(),
            IdVec::from(vec![step_ty]),
            op.attributes(db).clone(),
            op.regions(db).clone(),
            op.successors(db).clone(),
        );

        // Return the new call directly without adding a cast.
        // The Step value should propagate up through the effectful context.
        // Any downstream operations that need the original type will need to handle
        // Step unpacking appropriately.
        RewriteResult::replace(new_call)
    }
}

// ============================================================================
// Pattern: Update scf.if result type when branches contain effectful calls
// ============================================================================

/// Pattern that updates scf.if result type to Step when its branches
/// contain calls that return Step (effectful calls that have been transformed).
///
/// NOTE: This pattern only changes the result type. The internal scf.yield
/// operations are updated by UpdateScfYieldToStepPattern which runs after
/// the nested regions have been processed by PatternApplicator.
struct UpdateScfIfResultTypePattern;

impl<'db> RewritePattern<'db> for UpdateScfIfResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle scf.if operations
        if scf::If::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        // Skip if no results
        if op.results(db).is_empty() {
            return RewriteResult::Unchanged;
        }

        // Skip if already returns Step
        let step_ty = trampoline::Step::new(db).as_type();
        if op
            .results(db)
            .first()
            .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
        {
            return RewriteResult::Unchanged;
        }

        // Check if any branch contains operations that return Step type
        // This includes func.call that have been transformed to return Step
        let branches_have_step_ops = op.regions(db).iter().any(|region| {
            region.blocks(db).iter().any(|block| {
                block.operations(db).iter().any(|branch_op| {
                    // Check if any operation produces Step type result
                    branch_op
                        .results(db)
                        .first()
                        .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
                })
            })
        });

        if !branches_have_step_ops {
            return RewriteResult::Unchanged;
        }

        let loc = op.location(db);
        tracing::debug!(
            "UpdateScfIfResultTypePattern: updating scf.if result to Step at {:?}",
            loc
        );

        // Only change the result type - let PatternApplicator handle nested regions
        // and UpdateScfYieldToStepPattern handle the yield updates
        let new_if = Operation::new(
            db,
            op.location(db),
            op.dialect(db),
            op.name(db),
            op.operands(db).clone(),
            IdVec::from(vec![step_ty]),
            op.attributes(db).clone(),
            op.regions(db).clone(), // Keep original regions - will be processed recursively
            op.successors(db).clone(),
        );

        RewriteResult::replace(new_if)
    }
}

/// Pattern that updates scf.yield to yield Step value when it's inside
/// a block that contains effectful operations returning Step.
///
/// This pattern is applied after UpdateEffectfulCallResultTypePattern has
/// changed func.call results to Step, so we can find the actual Step values.
struct UpdateScfYieldToStepPattern;

impl<'db> RewritePattern<'db> for UpdateScfYieldToStepPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Only handle scf.yield operations
        let Ok(_yield_op) = scf::Yield::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Skip if already yielding Step
        if let Some(operand) = adaptor.operands().first()
            && let Some(ty) = adaptor.get_raw_value_type(db, *operand)
            && trampoline::Step::from_type(db, ty).is_some()
        {
            return RewriteResult::Unchanged;
        }

        // Check the context to find if there's a Step value we should yield instead
        // We look at the remapped operand types to see what the current value types are
        let current_operands = adaptor.operands();
        if current_operands.is_empty() {
            return RewriteResult::Unchanged;
        }

        // Get the actual type of the yielded value
        let yielded_value = current_operands[0];
        let Some(yielded_ty) = adaptor.get_raw_value_type(db, yielded_value) else {
            return RewriteResult::Unchanged;
        };

        // If the yielded value is already Step, no change needed
        if trampoline::Step::from_type(db, yielded_ty).is_some() {
            return RewriteResult::Unchanged;
        }

        // Check if we can find the Step value through cast chain
        // The yielded value might be a cast result where the input is Step
        if let Some(step_value) = find_step_source(db, yielded_value, adaptor) {
            tracing::debug!(
                "UpdateScfYieldToStepPattern: updating scf.yield to yield Step at {:?}",
                op.location(db)
            );
            let new_yield = scf::r#yield(db, op.location(db), vec![step_value]);
            return RewriteResult::replace(new_yield.as_operation());
        }

        RewriteResult::Unchanged
    }
}

/// Find the Step source value by tracing through the value map and cast chain.
fn find_step_source<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    adaptor: &OpAdaptor<'db, '_>,
) -> Option<Value<'db>> {
    // Check if the value itself is Step (after remapping)
    if let Some(ty) = adaptor.get_raw_value_type(db, value)
        && trampoline::Step::from_type(db, ty).is_some()
    {
        return Some(value);
    }

    // Check if the value definition is a cast from Step
    if let trunk_ir::ValueDef::OpResult(defining_op) = value.def(db) {
        // If this is a cast, check the input
        if let Ok(cast) = core::UnrealizedConversionCast::from_operation(db, defining_op) {
            let input = cast.value(db);
            if let Some(input_ty) = adaptor.get_raw_value_type(db, input)
                && trampoline::Step::from_type(db, input_ty).is_some()
            {
                return Some(input);
            }
        }

        // Check if the defining op produces Step
        if let Some(ty) = defining_op.results(db).first()
            && trampoline::Step::from_type(db, *ty).is_some()
        {
            return Some(defining_op.result(db, 0));
        }
    }

    None
}

// ============================================================================
// Pattern: Lower cont.push_prompt
// ============================================================================

struct LowerPushPromptPattern;

impl<'db> RewritePattern<'db> for LowerPushPromptPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let push_prompt = match cont::PushPrompt::from_operation(db, *op) {
            Ok(p) => p,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let step_ty = trampoline::Step::new(db).as_type();
        let tag = push_prompt.tag(db);

        // Get the body region (already recursively transformed by applicator)
        let body = push_prompt.body(db);

        // Get body result value
        let body_result = get_region_result_value(db, &body);

        let mut all_ops = Vec::new();

        // Add all body operations
        if let Some(body_block) = body.blocks(db).first() {
            for body_op in body_block.operations(db).iter() {
                all_ops.push(*body_op);
            }
        }

        // check_yield
        let check_yield = trampoline::check_yield(db, location, i32_ty);
        let is_yielding = check_yield.as_operation().result(db, 0);
        all_ops.push(check_yield.as_operation());

        // Build yield handling branches
        let then_region = build_yield_then_branch(db, location, tag, step_ty);
        let else_region = build_yield_else_branch(db, location, body_result, step_ty);

        // scf.if for yield check
        let if_op = scf::r#if(db, location, is_yielding, step_ty, then_region, else_region);
        all_ops.push(if_op.as_operation());

        RewriteResult::expand(all_ops)
    }
}

fn build_yield_then_branch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    our_tag: u32,
    step_ty: Type<'db>,
) -> Region<'db> {
    let mut builder = BlockBuilder::new(db, location);

    let cont_ty = trampoline::Continuation::new(db).as_type();
    let get_cont = builder.op(trampoline::get_yield_continuation(db, location, cont_ty));
    let cont_val = get_cont.result(db);

    let step_shift = builder.op(trampoline::step_shift(
        db, location, cont_val, step_ty, our_tag, 0,
    ));
    builder.op(scf::r#yield(db, location, vec![step_shift.result(db)]));

    let block = builder.build();
    Region::new(db, location, IdVec::from(vec![block]))
}

fn build_yield_else_branch<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    body_result: Option<Value<'db>>,
    step_ty: Type<'db>,
) -> Region<'db> {
    use trunk_ir::ValueDef;

    let mut builder = BlockBuilder::new(db, location);

    let step_value = if let Some(result) = body_result {
        // Check if result is already a Step (from effectful call or scf.if returning Step)
        let is_step = match result.def(db) {
            ValueDef::OpResult(def_op) => {
                // Check if this operation produces Step
                def_op
                    .results(db)
                    .first()
                    .map(|ty| trampoline::Step::from_type(db, *ty).is_some())
                    .unwrap_or(false)
            }
            ValueDef::BlockArg(_) => false,
        };

        if is_step {
            // Already a Step, return it directly
            result
        } else {
            // Wrap the value in step_done
            let step_done = builder.op(trampoline::step_done(db, location, result, step_ty));
            step_done.result(db)
        }
    } else {
        // No body result - create a step_done with zero value (unit placeholder)
        let zero = builder.op(arith::Const::i32(db, location, 0));
        let step_done = builder.op(trampoline::step_done(
            db,
            location,
            zero.result(db),
            step_ty,
        ));
        step_done.result(db)
    };

    builder.op(scf::r#yield(db, location, vec![step_value]));

    let block = builder.build();
    Region::new(db, location, IdVec::from(vec![block]))
}

// ============================================================================
// Pattern: Lower cont.handler_dispatch
// ============================================================================

struct LowerHandlerDispatchPattern {
    effectful_funcs: Rc<HashSet<Symbol>>,
    /// Spans of handler_dispatch operations that are inside effectful functions.
    /// These handlers should return Step type for propagation.
    handlers_in_effectful_funcs: Rc<HashSet<Span>>,
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

        // Get the body region with multiple blocks
        let body_region = op
            .regions(db)
            .first()
            .cloned()
            .unwrap_or_else(|| Region::new(db, location, IdVec::new()));
        let blocks = body_region.blocks(db);

        // Collect suspend arms with their expected op_idx
        let suspend_arms = collect_suspend_arms(db, blocks);

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

        // For closed handlers, we could trap here
        // For now, just yield the current step (will cause type error if reached)
        builder.op(scf::r#yield(db, location, vec![current_step]));

        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
    }
}

/// Information about a suspend arm for dispatch.
struct SuspendArm<'db> {
    /// Expected op_idx for this arm
    expected_op_idx: u32,
    /// The block containing the handler arm code
    block: Block<'db>,
}

/// Collect suspend arms from handler blocks with their expected op_idx.
fn collect_suspend_arms<'db>(
    db: &'db dyn salsa::Database,
    blocks: &IdVec<Block<'db>>,
) -> Vec<SuspendArm<'db>> {
    let mut arms = Vec::new();

    // Skip block 0 (done case), process blocks 1+ (suspend cases)
    for block in blocks.iter().skip(1) {
        // Extract ability_ref and op_name from marker block arg
        let block_args = block.args(db);
        if let Some(marker_arg) = block_args.first() {
            let attrs = marker_arg.attrs(db);
            let ability_ref = attrs.get(&Symbol::new("ability_ref")).and_then(|a| {
                if let Attribute::Type(ty) = a {
                    core::AbilityRefType::from_type(db, *ty).and_then(|ar| ar.name(db))
                } else {
                    None
                }
            });
            let op_name = attrs.get(&Symbol::new("op_name")).and_then(|a| {
                if let Attribute::Symbol(s) = a {
                    Some(*s)
                } else {
                    None
                }
            });

            let expected_op_idx = compute_op_idx(ability_ref, op_name);
            arms.push(SuspendArm {
                expected_op_idx,
                block: *block,
            });
        }
    }

    arms
}

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
fn build_nested_dispatch<'db>(
    db: &'db dyn salsa::Database,
    builder: &mut BlockBuilder<'db>,
    location: Location<'db>,
    result_ty: Type<'db>,
    current_op_idx: Value<'db>,
    suspend_arms: &[SuspendArm<'db>],
    arm_index: usize,
    effectful_funcs: &HashSet<Symbol>,
) -> Value<'db> {
    let i32_ty = core::I32::new(db).as_type();

    // Safety check
    if arm_index >= suspend_arms.len() {
        panic!("build_nested_dispatch: arm_index out of bounds");
    }

    let arm = &suspend_arms[arm_index];
    let is_last_arm = arm_index + 1 >= suspend_arms.len();

    // Build then region: execute this arm's block code
    let then_region = build_arm_region(db, location, &arm.block, effectful_funcs);

    if is_last_arm {
        // Last arm (or only arm): use always-true condition, duplicate arm for else
        let true_const = builder.op(arith::Const::i32(db, location, 1));
        let else_region = build_arm_region(db, location, &arm.block, effectful_funcs);

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
        i32_ty,
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

/// Build a single-block region from a handler arm block.
///
/// The arm block contains user's handler code ending with cont.resume (lowered to
/// trampoline.resume → func.call_indirect which returns Step).
/// We need to ensure the region yields this Step value.
///
/// IMPORTANT: The arm block may contain unrealized_conversion_cast operations that
/// convert the Step result to the user's expected type (e.g., i32). We need to find
/// the actual Step result and yield that, not the converted result.
fn build_arm_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    arm_block: &Block<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> Region<'db> {
    // Skip the first block arg (marker arg) if present
    let original_args = arm_block.args(db);
    let new_args = if !original_args.is_empty() {
        IdVec::from(original_args.iter().skip(1).cloned().collect::<Vec<_>>())
    } else {
        IdVec::new()
    };

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

        // Helper to remap a value through the remap chain
        // Note: This closure is called in the loop below, where value_remap may be mutated
        // So we use a function that takes the map as a parameter instead
        fn remap_value_inner<'db>(
            v: Value<'db>,
            value_remap: &std::collections::HashMap<Value<'db>, Value<'db>>,
        ) -> Value<'db> {
            let mut current = v;
            while let Some(&remapped) = value_remap.get(&current) {
                current = remapped;
            }
            current
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

            // Remap operands if needed
            let operands = op.operands(db);
            let remapped_operands: IdVec<Value<'db>> = operands
                .iter()
                .map(|v| remap_value_inner(*v, &value_remap))
                .collect::<Vec<_>>()
                .into();

            // Determine result types - effectful calls need Step type
            let result_types = if is_effectful_call {
                // Replace the first result type with Step
                let mut types = op.results(db).clone();
                if !types.is_empty() {
                    types = IdVec::from(vec![step_ty]);
                }
                types
            } else {
                op.results(db).clone()
            };

            // If operands or result types changed, create new operation
            if remapped_operands != *operands || is_effectful_call {
                let new_op = Operation::new(
                    db,
                    op.location(db),
                    op.dialect(db),
                    op.name(db),
                    remapped_operands,
                    result_types,
                    op.attributes(db).clone(),
                    op.regions(db).clone(),
                    op.successors(db).clone(),
                );
                ops.push(new_op);
                if is_effectful_call {
                    // Effectful call returns Step at runtime - subsequent ops are handled by continuation
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
        }

        IdVec::from(ops)
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

// ============================================================================
// Global Trampoline Wrapper (_start)
// ============================================================================

/// Wrap main with a global trampoline if main is effectful (returns Step).
///
/// Creates `_start` function that:
/// 1. Calls `main()` which returns Step
/// 2. Loops to process the Step:
///    - Done: extract value and return
///    - Shift: panic (unhandled effect at top level)
fn wrap_main_with_global_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> Module<'db> {
    let main_symbol = Symbol::new("main");

    // Check if main is effectful
    if !effectful_funcs.contains(&main_symbol) {
        return module;
    }

    // Find main function and get its return type
    let mut main_func_op: Option<Operation<'db>> = None;
    let mut main_original_return_ty: Option<Type<'db>> = None;

    for block in module.body(db).blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func) = Func::from_operation(db, *op)
                && func.sym_name(db) == main_symbol
            {
                main_func_op = Some(*op);
                // Get the function type to extract original return type
                let func_ty = func.r#type(db);
                if let Some(fn_ty) = core::Func::from_type(db, func_ty) {
                    // After update_effectful_types, return type is Step
                    // We need the original return type for _start's return
                    // For now, assume i32 for main (common case)
                    // TODO: Store original return type before Step conversion
                    main_original_return_ty = Some(core::I32::new(db).as_type());
                    let _ = fn_ty; // suppress unused warning
                }
                break;
            }
        }
        if main_func_op.is_some() {
            break;
        }
    }

    let Some(_main_op) = main_func_op else {
        // No main function found
        return module;
    };

    let original_return_ty =
        main_original_return_ty.unwrap_or_else(|| core::I32::new(db).as_type());
    let location = module.location(db);

    // Create _start function
    let start_func = build_start_function(db, location, original_return_ty);

    // Add _start to module
    let body = module.body(db);
    let mut blocks: Vec<Block<'db>> = body.blocks(db).iter().copied().collect();
    if let Some(block) = blocks.first_mut() {
        let mut ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();
        ops.push(start_func);
        *block = Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            IdVec::from(ops),
        );
    }

    let new_body = Region::new(db, body.location(db), IdVec::from(blocks));
    Module::create(db, module.location(db), module.name(db), new_body)
}

/// Build the _start function that wraps main with a trampoline loop.
fn build_start_function<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    return_ty: Type<'db>,
) -> Operation<'db> {
    let step_ty = trampoline::Step::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();
    let i1_ty = core::I1::new(db).as_type();
    let anyref_ty = tribute_rt::Any::new(db).as_type();

    let mut builder = BlockBuilder::new(db, location);

    // Call main() -> Step
    let main_call = builder.op(func::call(
        db,
        location,
        IdVec::new(), // no args
        step_ty,
        Symbol::new("main"),
    ));
    let initial_step = main_call.result(db);

    // Build trampoline loop body
    let loop_body = build_start_trampoline_loop_body(
        db, location, step_ty, return_ty, i32_ty, i1_ty, anyref_ty,
    );

    // Create scf.loop
    let loop_op = builder.op(scf::r#loop(
        db,
        location,
        vec![initial_step],
        return_ty,
        loop_body,
    ));

    // Return the loop result
    builder.op(func::r#return(db, location, Some(loop_op.result(db))));

    // Build function body region
    let body_block = builder.build();
    let body_region = Region::new(db, location, IdVec::from(vec![body_block]));

    // Create function type: () -> return_ty
    let func_ty = core::Func::new(db, IdVec::new(), return_ty).as_type();

    func::func(db, location, Symbol::new("_start"), func_ty, body_region).as_operation()
}

/// Build the trampoline loop body for _start.
fn build_start_trampoline_loop_body<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    step_ty: Type<'db>,
    return_ty: Type<'db>,
    i32_ty: Type<'db>,
    i1_ty: Type<'db>,
    anyref_ty: Type<'db>,
) -> Region<'db> {
    use trunk_ir::BlockArg;

    // Block argument: current_step (Step type)
    let block_id = trunk_ir::BlockId::fresh();
    let current_step_arg = BlockArg::of_type(db, step_ty);
    let current_step = Value::new(db, trunk_ir::ValueDef::BlockArg(block_id), 0);

    let mut builder = BlockBuilder::new(db, location);

    // Extract tag from Step (0 = Done, 1 = Shift)
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

    // Done branch: extract value and break
    let done_region = build_start_done_region(db, location, current_step, return_ty, anyref_ty);

    // Shift branch: unhandled effect - unreachable/panic
    let shift_region = build_start_shift_region(db, location, return_ty);

    // scf.if: if is_done { done } else { shift }
    let if_op = builder.op(scf::r#if(
        db,
        location,
        is_done.result(db),
        return_ty,
        done_region,
        shift_region,
    ));

    // Yield the if result (for loop body)
    builder.op(scf::r#yield(db, location, vec![if_op.result(db)]));

    let body_block = Block::new(
        db,
        block_id,
        location,
        IdVec::from(vec![current_step_arg]),
        builder.build().operations(db).clone(),
    );

    Region::new(db, location, IdVec::from(vec![body_block]))
}

/// Build Done region for _start: extract value and break.
fn build_start_done_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    current_step: Value<'db>,
    return_ty: Type<'db>,
    anyref_ty: Type<'db>,
) -> Region<'db> {
    let mut builder = BlockBuilder::new(db, location);

    // Extract value from Step
    let get_value = builder.op(trampoline::step_get(
        db,
        location,
        current_step,
        anyref_ty,
        Symbol::new("value"),
    ));
    let step_value = get_value.result(db);

    // Cast anyref to return type
    let result_value = if anyref_ty != return_ty {
        let cast = builder.op(core::unrealized_conversion_cast(
            db, location, step_value, return_ty,
        ));
        cast.result(db)
    } else {
        step_value
    };

    // Break from loop with the value
    builder.op(scf::r#break(db, location, result_value));

    let block = builder.build();
    Region::new(db, location, IdVec::from(vec![block]))
}

/// Build Shift region for _start: unhandled effect (unreachable).
fn build_start_shift_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    return_ty: Type<'db>,
) -> Region<'db> {
    let mut builder = BlockBuilder::new(db, location);

    // Unhandled effect at top level - this should never happen in well-formed programs
    // Use unreachable to indicate this is an error path
    builder.op(func::unreachable(db, location));

    // Need a dummy break to satisfy type checker (unreachable, but needed for IR validity)
    // Create a dummy value of return_ty - use i32(0) as a placeholder since unreachable
    let const_op = builder.op(arith::Const::i32(db, location, 0));
    let dummy = if return_ty == core::I32::new(db).as_type() {
        const_op.result(db)
    } else {
        // Cast to the expected return type
        let cast = builder.op(core::unrealized_conversion_cast(
            db,
            location,
            const_op.result(db),
            return_ty,
        ));
        cast.result(db)
    };

    builder.op(scf::r#break(db, location, dummy));

    let block = builder.build();
    Region::new(db, location, IdVec::from(vec![block]))
}

// ============================================================================
// Pattern: Wrap returns in effectful functions with step_done
// ============================================================================

struct WrapReturnsInEffectfulFuncsPattern {
    effectful_funcs: Rc<HashSet<Symbol>>,
}

impl<'db> RewritePattern<'db> for WrapReturnsInEffectfulFuncsPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match func.func operations
        let Ok(func) = Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Only process effectful functions
        let func_name = func.sym_name(db);
        if !self.effectful_funcs.contains(&func_name) {
            return RewriteResult::Unchanged;
        }

        tracing::debug!(
            "WrapReturnsInEffectfulFuncsPattern: processing {}",
            func_name
        );

        // Transform the function body - wrap non-Step returns with step_done
        let body = func.body(db);
        let (new_body, modified) = wrap_returns_in_region(db, body);

        tracing::debug!(
            "WrapReturnsInEffectfulFuncsPattern: {} modified={}",
            func_name,
            modified
        );

        if !modified {
            return RewriteResult::Unchanged;
        }

        // Rebuild the function with the transformed body
        let new_op = op.modify(db).regions(IdVec::from(vec![new_body])).build();
        RewriteResult::Replace(new_op)
    }
}

/// Recursively wrap returns in a region with step_done.
/// Returns (new_region, was_modified).
fn wrap_returns_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
) -> (Region<'db>, bool) {
    let mut new_blocks = Vec::new();
    let mut any_modified = false;

    for block in region.blocks(db).iter() {
        let (new_block, modified) = wrap_returns_in_block(db, *block);
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

/// Wrap returns in a block with step_done.
/// Returns (new_block, was_modified).
fn wrap_returns_in_block<'db>(
    db: &'db dyn salsa::Database,
    block: Block<'db>,
) -> (Block<'db>, bool) {
    let mut new_ops = Vec::new();
    let mut modified = false;

    for op in block.operations(db).iter() {
        // First, recursively process nested regions
        let mut op_modified = false;
        let op_with_transformed_regions = if !op.regions(db).is_empty() {
            let mut new_regions = Vec::new();
            for r in op.regions(db).iter() {
                let (new_r, r_modified) = wrap_returns_in_region(db, *r);
                new_regions.push(new_r);
                op_modified |= r_modified;
            }
            if op_modified {
                op.modify(db).regions(IdVec::from(new_regions)).build()
            } else {
                *op
            }
        } else {
            *op
        };

        modified |= op_modified;

        // Check if this is a func.return
        if func::Return::from_operation(db, op_with_transformed_regions).is_ok() {
            let operands = op_with_transformed_regions.operands(db);

            if let Some(&value) = operands.first() {
                // Check if already returning Step
                let is_step = is_step_value(db, value);
                tracing::debug!(
                    "wrap_returns_in_block: found func.return, value is_step={}",
                    is_step
                );
                if !is_step {
                    let location = op_with_transformed_regions.location(db);
                    let step_ty = trampoline::Step::new(db).as_type();

                    // Create step_done(value)
                    let step_done = trampoline::step_done(db, location, value, step_ty);
                    let step_value = step_done.as_operation().result(db, 0);
                    new_ops.push(step_done.as_operation());

                    // Create new return with step value
                    let new_return = func::r#return(db, location, Some(step_value));
                    new_ops.push(new_return.as_operation());
                    modified = true;
                    tracing::debug!("wrap_returns_in_block: wrapped return with step_done");
                    continue;
                }
            }
        }

        new_ops.push(op_with_transformed_regions);
    }

    if !modified {
        return (block, false);
    }

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

/// Check if a value is already a Step type (from step_shift, step_done, or check_yield result).
fn is_step_value<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> bool {
    use trunk_ir::ValueDef;

    match value.def(db) {
        ValueDef::OpResult(def_op) => {
            // Check if the defining operation produces Step
            trampoline::StepShift::from_operation(db, def_op).is_ok()
                || trampoline::StepDone::from_operation(db, def_op).is_ok()
                // Check if scf.if returns Step type (not all scf.if return Step)
                || (scf::If::from_operation(db, def_op).is_ok()
                    && def_op
                        .results(db)
                        .first()
                        .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some()))
        }
        ValueDef::BlockArg(_) => {
            // Block args could be Step, but we can't easily tell from here
            // For safety, don't wrap block args
            false
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn get_region_result_value<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Value<'db>> {
    let blocks = region.blocks(db);
    let last_block = blocks.last()?;
    let ops = last_block.operations(db);
    let last_op = ops.last()?;

    // If the last op is scf.yield, return its first operand (the yielded value)
    if let Ok(yield_op) = scf::Yield::from_operation(db, *last_op) {
        return yield_op.values(db).first().copied();
    }

    // Otherwise, return the first result of the last op
    last_op.results(db).first().map(|_| last_op.result(db, 0))
}

fn compute_op_idx(ability_ref: Option<Symbol>, op_name: Option<Symbol>) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    if let Some(ability) = ability_ref {
        ability.to_string().hash(&mut hasher);
    }
    if let Some(name) = op_name {
        name.to_string().hash(&mut hasher);
    }

    (hasher.finish() % 0x7FFFFFFF) as u32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::ir::BlockBuilder;
    use trunk_ir::rewrite::RewriteContext;
    use trunk_ir::{BlockArg, BlockId, PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    // ========================================================================
    // Test: Multi-suspend block handling
    // ========================================================================

    /// Test helper: builds handler_dispatch with multiple suspend blocks and applies pattern.
    /// Helper function to count nested scf.if operations in a region.
    /// This counts the total number of scf.if operations, including nested ones.
    fn count_scf_if_in_region<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> usize {
        let mut count = 0;
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                if scf::If::from_operation(db, *op).is_ok() {
                    count += 1;
                    // Count nested scf.if in the then/else regions
                    for nested_region in op.regions(db).iter() {
                        count += count_scf_if_in_region(db, nested_region);
                    }
                }
            }
        }
        count
    }

    /// Returns the number of nested scf.if operations in the suspend region.
    /// With the new dispatch structure, each suspend arm becomes a branch in nested scf.if.
    /// For 2 suspend arms: if (op_idx == 0) { arm0 } else { if (true) { arm1 } else { arm1 } }
    /// So we expect 2 scf.if operations (outer dispatch + inner for last arm).
    #[salsa::tracked]
    fn handler_dispatch_scf_if_count(db: &dyn salsa::Database) -> usize {
        let location = test_location(db);
        let step_ty = trampoline::Step::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();

        // Create 3 blocks: done block + 2 suspend blocks
        let done_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());

        // Create marker block args for suspend blocks (required by collect_suspend_arms)
        let marker_arg1 = {
            let mut attrs = std::collections::BTreeMap::new();
            attrs.insert(
                Symbol::new("op_name"),
                Attribute::Symbol(Symbol::new("get")),
            );
            BlockArg::new(db, i32_ty, attrs)
        };

        let marker_arg2 = {
            let mut attrs = std::collections::BTreeMap::new();
            attrs.insert(
                Symbol::new("op_name"),
                Attribute::Symbol(Symbol::new("set")),
            );
            BlockArg::new(db, i32_ty, attrs)
        };

        let mut builder1 = BlockBuilder::new(db, location);
        let zero1 = builder1.op(arith::Const::i32(db, location, 1));
        let step1 = builder1.op(trampoline::step_done(
            db,
            location,
            zero1.result(db),
            step_ty,
        ));
        builder1.op(scf::r#yield(db, location, vec![step1.result(db)]));
        let suspend_block1 = {
            let block = builder1.build();
            // Rebuild block with marker arg
            Block::new(
                db,
                block.id(db),
                location,
                IdVec::from(vec![marker_arg1]),
                block.operations(db).clone(),
            )
        };

        let mut builder2 = BlockBuilder::new(db, location);
        let zero2 = builder2.op(arith::Const::i32(db, location, 2));
        let step2 = builder2.op(trampoline::step_done(
            db,
            location,
            zero2.result(db),
            step_ty,
        ));
        builder2.op(scf::r#yield(db, location, vec![step2.result(db)]));
        let suspend_block2 = {
            let block = builder2.build();
            // Rebuild block with marker arg
            Block::new(
                db,
                block.id(db),
                location,
                IdVec::from(vec![marker_arg2]),
                block.operations(db).clone(),
            )
        };

        let body_region = Region::new(
            db,
            location,
            IdVec::from(vec![done_block, suspend_block1, suspend_block2]),
        );

        // Create a dummy result value for handler_dispatch
        let dummy_const = arith::Const::i32(db, location, 0);
        let result_val = dummy_const.as_operation().result(db, 0);

        // Create handler_dispatch with 3 blocks
        let test_tag: u32 = 12345; // Dummy tag for testing
        let i32_ty = core::I32::new(db).as_type();
        let dispatch_op = cont::handler_dispatch(
            db,
            location,
            result_val,
            step_ty,
            test_tag,
            i32_ty,
            body_region,
        )
        .as_operation();

        // Apply pattern
        let effectful_funcs = Rc::new(HashSet::new());
        let handlers_in_effectful_funcs = Rc::new(HashSet::new());
        let pattern = LowerHandlerDispatchPattern {
            effectful_funcs,
            handlers_in_effectful_funcs,
        };
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let adaptor = OpAdaptor::new(
            dispatch_op,
            dispatch_op.operands(db).clone(),
            vec![],
            &ctx,
            &type_converter,
        );
        let result = pattern.match_and_rewrite(db, &dispatch_op, &adaptor);

        // Count scf.if operations in the loop body
        // With trampoline loop, the result is a single scf.loop operation
        match result {
            RewriteResult::Expand(ops) if ops.len() == 1 => {
                let loop_op = &ops[0];
                let regions = loop_op.regions(db);
                if !regions.is_empty() {
                    // Count scf.if in the loop body region
                    count_scf_if_in_region(db, &regions[0])
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    #[salsa_test]
    fn test_handler_dispatch_collects_all_suspend_blocks(db: &salsa::DatabaseImpl) {
        // With trampoline loop structure and 2 suspend arms, we expect:
        // - 1 scf.if for is_done check (done vs shift branch)
        // - 1 scf.if for tag_matches check (in shift branch)
        // - 2 scf.if for arm dispatch (outer dispatch + last arm always-true)
        // Total: 4 scf.if operations
        let count = handler_dispatch_scf_if_count(db);
        assert_eq!(
            count, 4,
            "Loop body should have 4 scf.if: is_done + tag_matches + 2 arm dispatch"
        );
    }

    // ========================================================================
    // Test: is_step_value with scf.if type verification
    // ========================================================================

    /// Test helper: creates scf.if returning Step type and checks is_step_value.
    #[salsa::tracked]
    fn is_step_value_for_step_if(db: &dyn salsa::Database) -> bool {
        let location = test_location(db);
        let step_ty = trampoline::Step::new(db).as_type();

        // Create scf.if returning Step type
        let cond_op = arith::Const::i32(db, location, 1);
        let cond_val = cond_op.as_operation().result(db, 0);

        let mut then_builder = BlockBuilder::new(db, location);
        let zero = then_builder.op(arith::Const::i32(db, location, 0));
        let step = then_builder.op(trampoline::step_done(
            db,
            location,
            zero.result(db),
            step_ty,
        ));
        then_builder.op(scf::r#yield(db, location, vec![step.result(db)]));
        let then_region = Region::new(db, location, IdVec::from(vec![then_builder.build()]));

        let mut else_builder = BlockBuilder::new(db, location);
        let one = else_builder.op(arith::Const::i32(db, location, 1));
        let step2 = else_builder.op(trampoline::step_done(db, location, one.result(db), step_ty));
        else_builder.op(scf::r#yield(db, location, vec![step2.result(db)]));
        let else_region = Region::new(db, location, IdVec::from(vec![else_builder.build()]));

        let if_op = scf::r#if(db, location, cond_val, step_ty, then_region, else_region);
        let if_result = if_op.as_operation().result(db, 0);

        is_step_value(db, if_result)
    }

    /// Test helper: creates scf.if returning i32 (non-Step) and checks is_step_value.
    #[salsa::tracked]
    fn is_step_value_for_i32_if(db: &dyn salsa::Database) -> bool {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create scf.if returning i32 (not Step)
        let cond_op = arith::Const::i32(db, location, 1);
        let cond_val = cond_op.as_operation().result(db, 0);

        let mut then_builder = BlockBuilder::new(db, location);
        let val1 = then_builder.op(arith::Const::i32(db, location, 42));
        then_builder.op(scf::r#yield(db, location, vec![val1.result(db)]));
        let then_region = Region::new(db, location, IdVec::from(vec![then_builder.build()]));

        let mut else_builder = BlockBuilder::new(db, location);
        let val2 = else_builder.op(arith::Const::i32(db, location, 0));
        else_builder.op(scf::r#yield(db, location, vec![val2.result(db)]));
        let else_region = Region::new(db, location, IdVec::from(vec![else_builder.build()]));

        let if_op = scf::r#if(db, location, cond_val, i32_ty, then_region, else_region);
        let if_result = if_op.as_operation().result(db, 0);

        is_step_value(db, if_result)
    }

    #[salsa_test]
    fn test_is_step_value_scf_if_with_step_type(db: &salsa::DatabaseImpl) {
        assert!(
            is_step_value_for_step_if(db),
            "scf.if returning Step should be detected as Step value"
        );
    }

    #[salsa_test]
    fn test_is_step_value_scf_if_with_non_step_type(db: &salsa::DatabaseImpl) {
        assert!(
            !is_step_value_for_i32_if(db),
            "scf.if returning i32 should NOT be detected as Step value"
        );
    }

    // ========================================================================
    // Test: Resume function generation (no global state)
    // ========================================================================

    #[test]
    fn test_fresh_resume_name_generates_unique_names() {
        let counter = Rc::new(RefCell::new(0u32));

        let name1 = fresh_resume_name(&counter);
        let name2 = fresh_resume_name(&counter);
        let name3 = fresh_resume_name(&counter);

        assert_eq!(name1, "__resume_0");
        assert_eq!(name2, "__resume_1");
        assert_eq!(name3, "__resume_2");
        assert_eq!(*counter.borrow(), 3);
    }

    #[test]
    fn test_resume_specs_isolation() {
        // Test that different ResumeCounter instances are independent
        let counter1 = Rc::new(RefCell::new(0u32));
        let counter2 = Rc::new(RefCell::new(0u32));

        // Increment counter1 twice
        *counter1.borrow_mut() += 1;
        *counter1.borrow_mut() += 1;

        // counter2 should still be 0
        assert_eq!(*counter1.borrow(), 2);
        assert_eq!(*counter2.borrow(), 0, "counter2 should be independent");
    }

    // ========================================================================
    // Test: Utility functions
    // ========================================================================

    #[test]
    fn test_compute_op_idx_deterministic() {
        // Same inputs should produce same output
        let idx1 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        let idx2 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        assert_eq!(idx1, idx2, "Same inputs should produce same op_idx");

        // Different op names should produce different indices
        let idx_get = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        let idx_set = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("set")));
        assert_ne!(
            idx_get, idx_set,
            "Different ops should have different indices"
        );

        // Different abilities should produce different indices
        let idx_state = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        let idx_console = compute_op_idx(Some(Symbol::new("Console")), Some(Symbol::new("get")));
        assert_ne!(
            idx_state, idx_console,
            "Different abilities should have different indices"
        );
    }

    #[test]
    fn test_state_type_name_deterministic() {
        // Same inputs should produce same output
        let name1 = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("get")), 0, 0);
        let name2 = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("get")), 0, 0);
        assert_eq!(name1, name2, "Same inputs should produce same name");

        // Name should start with __State_ prefix
        assert!(
            name1.starts_with("__State_"),
            "State type name should have __State_ prefix"
        );

        // Different tags should produce different names
        let name_tag0 = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("get")), 0, 0);
        let name_tag1 = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("get")), 1, 0);
        assert_ne!(
            name_tag0, name_tag1,
            "Different tags should produce different names"
        );

        // Different ops should produce different names
        let name_get = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("get")), 0, 0);
        let name_set = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("set")), 0, 0);
        assert_ne!(
            name_get, name_set,
            "Different ops should produce different names"
        );

        // Different shift indices should produce different names
        let name_idx0 = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("get")), 0, 0);
        let name_idx1 = state_type_name(Some(Symbol::new("State")), Some(Symbol::new("get")), 0, 1);
        assert_ne!(
            name_idx0, name_idx1,
            "Different shift indices should produce different names"
        );
    }
}
