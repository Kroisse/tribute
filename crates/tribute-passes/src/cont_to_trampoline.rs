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
use tribute_ir::dialect::{adt, trampoline};
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func::{self, Func};
use trunk_ir::dialect::{arith, cont, scf, wasm};
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
    /// Operations after this shift until next shift or function end
    continuation_ops: Vec<Operation<'db>>,
}

/// Lower cont dialect operations to trampoline dialect.
///
/// Returns an error if any `cont.*` operations (except `cont.drop`) remain after conversion.
pub fn lower_cont_to_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Result<Module<'db>, ConversionError> {
    // Shared state for resume function generation (no global state!)
    let resume_specs: ResumeSpecs<'db> = Rc::new(RefCell::new(Vec::new()));
    let resume_counter: ResumeCounter = Rc::new(RefCell::new(0));

    // Step 1: Identify effectful functions (before transformation)
    let effectful_funcs = identify_effectful_functions(db, &module);

    // Step 2: Update function signatures, call result types, and scf.if types
    // Use empty target for intermediate steps (no verification needed)
    let empty_target = ConversionTarget::new();
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(UpdateFuncTypePattern {
            effectful_funcs: effectful_funcs.clone(),
        })
        .add_pattern(UpdateFuncCallResultTypePattern {
            effectful_funcs: effectful_funcs.clone(),
        })
        .add_pattern(UpdateScfIfTypePattern {
            effectful_funcs: effectful_funcs.clone(),
        });
    let result = applicator.apply_partial(db, module, empty_target.clone());

    // Step 2.5: Analyze shift points in effectful functions (before transforming them)
    let shift_analysis = analyze_shift_points(db, &result.module, &effectful_funcs);

    // Step 3: Transform cont.* operations to trampoline.* operations
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(LowerShiftPattern {
            resume_specs: Rc::clone(&resume_specs),
            resume_counter: Rc::clone(&resume_counter),
            shift_analysis: Rc::clone(&shift_analysis),
        })
        .add_pattern(LowerResumePattern)
        .add_pattern(LowerGetContinuationPattern)
        .add_pattern(LowerGetShiftValuePattern)
        .add_pattern(LowerGetDoneValuePattern)
        .add_pattern(LowerPushPromptPattern)
        .add_pattern(LowerHandlerDispatchPattern);

    let result = applicator.apply_partial(db, result.module, empty_target.clone());

    // Step 4: Wrap returns in effectful functions with step_done
    let applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(
        WrapReturnsInEffectfulFuncsPattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
        },
    );
    let result = applicator.apply_partial(db, result.module, empty_target);
    let module = result.module;

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

    // Walk through all functions
    for block in module.body(db).blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func) = Func::from_operation(db, *op) {
                let func_name = func.sym_name(db);
                if !effectful_funcs.contains(&func_name) {
                    continue;
                }

                // Analyze this effectful function
                let body = func.body(db);
                if let Some(func_analysis) = FunctionAnalysis::analyze(db, &body) {
                    let total_shifts = func_analysis.shift_points.len();
                    for shift_point in func_analysis.shift_points {
                        let span = shift_point.shift_op.location(db).span;
                        // Get the shift result value if the operation has results
                        let shift_result_value = if !shift_point.shift_op.results(db).is_empty() {
                            Some(shift_point.shift_op.result(db, 0))
                        } else {
                            None
                        };
                        analysis.insert(
                            span,
                            ShiftPointInfo {
                                index: shift_point.index,
                                total_shifts,
                                live_values: shift_point.live_values,
                                shift_result_value,
                                continuation_ops: shift_point.continuation_ops,
                            },
                        );
                    }
                }
            }
        }
    }

    tracing::debug!(
        "analyze_shift_points: found {} shift points",
        analysis.len()
    );
    Rc::new(analysis)
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
        "identify_effectful_functions: found {} effectful functions: {:?}",
        effectful.len(),
        effectful.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
    Rc::new(effectful)
}

/// Collect directly effectful functions and all functions for later propagation.
fn collect_direct_effectful_funcs<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    effectful: &mut HashSet<Symbol>,
    all_funcs: &mut Vec<(Symbol, Region<'db>)>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func) = Func::from_operation(db, *op) {
                let func_name = func.sym_name(db);
                let body = func.body(db);

                all_funcs.push((func_name, body));

                if contains_effectful_ops(db, &body) {
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

/// Check if a region calls any effectful function.
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

/// Recursively check if a region contains effectful operations:
/// - cont.shift (direct yield)
/// - cont.push_prompt (prompt installation)
/// - cont.handler_dispatch (handler arm that processes Step)
/// - cont.resume (resumes a continuation, returns Step)
fn contains_effectful_ops<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> bool {
    use std::ops::ControlFlow;
    use trunk_ir::{OperationWalk, WalkAction};

    region
        .walk_all::<()>(db, |op| {
            // Skip nested function definitions - they're analyzed separately
            if Func::from_operation(db, op).is_ok() {
                return ControlFlow::Continue(WalkAction::Skip);
            }
            // cont.shift is effectful
            if cont::Shift::from_operation(db, op).is_ok() {
                return ControlFlow::Break(());
            }
            // These are also effectful operations
            if cont::PushPrompt::from_operation(db, op).is_ok()
                || cont::HandlerDispatch::from_operation(db, op).is_ok()
                || cont::Resume::from_operation(db, op).is_ok()
            {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(WalkAction::Advance)
        })
        .is_break()
}

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

        // === 2. Get resume function reference ===
        let funcref_ty = wasm::Funcref::new(db).as_type();
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
            continuation_ops: shift_info.continuation_ops.clone(),
            next_resume_name,
            location,
            shift_analysis: self.shift_analysis.clone(),
        });

        let resume_name_sym = Symbol::from_dynamic(&resume_name);
        let const_op = func::constant(db, location, funcref_ty, resume_name_sym);
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
fn create_resume_function_with_continuation<'db>(
    db: &'db dyn salsa::Database,
    spec: &ResumeFuncSpec<'db>,
) -> Operation<'db> {
    let wrapper_ty = trampoline::ResumeWrapper::new(db).as_type();
    let step_ty = trampoline::Step::new(db).as_type();
    let anyref_ty = wasm::Anyref::new(db).as_type();
    let location = spec.location;
    let name = Symbol::from_dynamic(&spec.name);

    let func_op = Func::build(
        db,
        location,
        name,
        IdVec::from(vec![wrapper_ty]),
        step_ty,
        |builder| {
            let wrapper_arg = builder.block_arg(db, 0);

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
            let resume_value = get_resume_value.result(db);

            // Map shift result to resume_value
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
                for (i, ((_field_name, field_type), original_value)) in spec
                    .state_fields
                    .iter()
                    .zip(spec.original_live_values.iter())
                    .enumerate()
                {
                    let get_field = builder.op(adt::struct_get(
                        db,
                        location,
                        state_val,
                        *field_type,
                        spec.state_type,
                        Attribute::IntBits(i as u64),
                    ));
                    let extracted_value = get_field.result(db);
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

                    // Get resume function reference
                    let funcref_ty = wasm::Funcref::new(db).as_type();
                    let resume_name_sym = Symbol::from_dynamic(next_resume_name);
                    let const_op =
                        builder.op(func::constant(db, location, funcref_ty, resume_name_sym));
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
        let funcref_ty = wasm::Funcref::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();

        let operands = adaptor.operands();
        let continuation = operands
            .first()
            .copied()
            .expect("resume requires continuation");
        let value = operands.get(1).copied();

        let mut ops = Vec::new();

        // === 1. Reset yield state ===
        ops.push(trampoline::reset_yield_state(db, location).as_operation());

        // === 2. Get resume_fn from continuation ===
        let get_resume_fn = trampoline::continuation_get(
            db,
            location,
            continuation,
            funcref_ty,
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
        let step_ty = trampoline::Step::new(db).as_type();
        let call_op = func::call_indirect(
            db,
            location,
            resume_fn_val,
            IdVec::from(vec![wrapper_val]),
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

        let trampoline_op = trampoline::get_yield_shift_value(db, location, result_type);
        RewriteResult::Replace(trampoline_op.as_operation())
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

        let trampoline_op =
            trampoline::step_get(db, location, step_value, result_type, Symbol::new("value"));
        RewriteResult::Replace(trampoline_op.as_operation())
    }
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
    let mut builder = BlockBuilder::new(db, location);

    let step_value = if let Some(result) = body_result {
        let step_done = builder.op(trampoline::step_done(db, location, result, step_ty));
        step_done.result(db)
    } else {
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

struct LowerHandlerDispatchPattern;

impl<'db> RewritePattern<'db> for LowerHandlerDispatchPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = cont::HandlerDispatch::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let step_ty = trampoline::Step::new(db).as_type();

        // Get the body region with multiple blocks
        let body_region = op
            .regions(db)
            .first()
            .cloned()
            .unwrap_or_else(|| Region::new(db, location, IdVec::new()));
        let blocks = body_region.blocks(db);

        // Block 0 = done case
        let done_region = if let Some(done_block) = blocks.first() {
            Region::new(db, location, IdVec::from(vec![*done_block]))
        } else {
            Region::new(db, location, IdVec::new())
        };

        // Block 1+ = suspend cases (include all suspend blocks, not just the first one)
        let suspend_region = if blocks.len() > 1 {
            let suspend_blocks: Vec<_> = blocks.iter().skip(1).copied().collect();
            Region::new(db, location, IdVec::from(suspend_blocks))
        } else {
            let mut builder = BlockBuilder::new(db, location);
            let zero = builder.op(arith::Const::i32(db, location, 0));
            let step_done = builder.op(trampoline::step_done(
                db,
                location,
                zero.result(db),
                step_ty,
            ));
            builder.op(scf::r#yield(db, location, vec![step_done.result(db)]));
            Region::new(db, location, IdVec::from(vec![builder.build()]))
        };

        // check_yield
        let check_yield = trampoline::check_yield(db, location, i32_ty);
        let is_yielding = check_yield.as_operation().result(db, 0);

        // scf.if
        let if_op = scf::r#if(
            db,
            location,
            is_yielding,
            step_ty,
            suspend_region,
            done_region,
        );

        RewriteResult::expand(vec![check_yield.as_operation(), if_op.as_operation()])
    }
}

// ============================================================================
// Pattern: Update func.func type for effectful functions
// ============================================================================

struct UpdateFuncTypePattern {
    effectful_funcs: Rc<HashSet<Symbol>>,
}

impl<'db> RewritePattern<'db> for UpdateFuncTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(func) = Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let func_name = func.sym_name(db);
        if !self.effectful_funcs.contains(&func_name) {
            return RewriteResult::Unchanged;
        }

        // Get the current function type
        let func_ty = func.r#type(db);
        let Some(fn_ty) = core::Func::from_type(db, func_ty) else {
            return RewriteResult::Unchanged;
        };

        // Create new function type with Step as return type
        let step_ty = trampoline::Step::new(db).as_type();
        let original_result = fn_ty.result(db);

        // Skip if already returning Step
        if trampoline::Step::from_type(db, original_result).is_some() {
            return RewriteResult::Unchanged;
        }

        let params = fn_ty.params(db);
        let effect = fn_ty.effect(db);
        let new_fn_ty = core::Func::with_effect(db, params, step_ty, effect);

        // Rebuild the function operation with updated type and original_result_type attribute
        let new_op = op
            .modify(db)
            .attr("type", Attribute::Type(new_fn_ty.as_type()))
            .attr("original_result_type", Attribute::Type(original_result))
            .build();

        RewriteResult::Replace(new_op)
    }
}

// ============================================================================
// Pattern: Update scf.if result types in effectful functions
// ============================================================================

struct UpdateScfIfTypePattern {
    effectful_funcs: Rc<HashSet<Symbol>>,
}

impl<'db> RewritePattern<'db> for UpdateScfIfTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = scf::If::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Check if result type needs update
        let result_types = op.results(db);
        if result_types.is_empty() {
            return RewriteResult::Unchanged;
        }

        let step_ty = trampoline::Step::new(db).as_type();

        // Skip if already Step
        if trampoline::Step::from_type(db, result_types[0]).is_some() {
            return RewriteResult::Unchanged;
        }

        // Check if any branch yields a call to an effectful function or returns Step
        // by looking at the scf.yield operands in the then/else regions
        let regions = op.regions(db);
        let mut needs_step = false;

        for region in regions.iter() {
            if region_yields_effectful_result(db, region, &self.effectful_funcs) {
                needs_step = true;
                break;
            }
        }

        if !needs_step {
            return RewriteResult::Unchanged;
        }

        // Update result type to Step
        let new_op = op.modify(db).results(IdVec::from(vec![step_ty])).build();

        RewriteResult::Replace(new_op)
    }
}

/// Check if a region's scf.yield yields a value from an effectful function call.
fn region_yields_effectful_result<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> bool {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check if this is scf.yield
            if let Ok(yield_op) = scf::Yield::from_operation(db, *op) {
                // Check if any yielded value comes from an effectful call
                for value in yield_op.values(db).iter() {
                    if value_from_effectful_call(db, *value, effectful_funcs) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if a value comes from a call to an effectful function.
fn value_from_effectful_call<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    effectful_funcs: &HashSet<Symbol>,
) -> bool {
    use trunk_ir::ValueDef;

    match value.def(db) {
        ValueDef::OpResult(def_op) => {
            // Check if this is a func.call to an effectful function
            if let Ok(call) = func::Call::from_operation(db, def_op)
                && effectful_funcs.contains(&call.callee(db))
            {
                return true;
            }
            // Also check if it's a Step-producing operation
            if trampoline::StepDone::from_operation(db, def_op).is_ok()
                || trampoline::StepShift::from_operation(db, def_op).is_ok()
            {
                return true;
            }
            // Check if nested if already returns Step
            if scf::If::from_operation(db, def_op).is_ok()
                && let Some(result_ty) = def_op.results(db).first()
            {
                return trampoline::Step::from_type(db, *result_ty).is_some();
            }
            false
        }
        ValueDef::BlockArg(_) => false,
    }
}

// ============================================================================
// Pattern: Update func.call result types for calls to effectful functions
// ============================================================================

struct UpdateFuncCallResultTypePattern {
    effectful_funcs: Rc<HashSet<Symbol>>,
}

impl<'db> RewritePattern<'db> for UpdateFuncCallResultTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(call) = func::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Check if calling an effectful function
        let callee = call.callee(db);
        if !self.effectful_funcs.contains(&callee) {
            return RewriteResult::Unchanged;
        }

        // Check if result type already Step
        let result_types = op.results(db);
        if result_types.is_empty() {
            return RewriteResult::Unchanged;
        }

        let step_ty = trampoline::Step::new(db).as_type();
        if trampoline::Step::from_type(db, result_types[0]).is_some() {
            return RewriteResult::Unchanged;
        }

        // Update result type to Step
        let new_op = op.modify(db).results(IdVec::from(vec![step_ty])).build();

        RewriteResult::Replace(new_op)
    }
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

        // Transform the function body - wrap non-Step returns with step_done
        let body = func.body(db);
        let (new_body, modified) = wrap_returns_in_region(db, body);

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
                if !is_step_value(db, value) {
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
    use trunk_ir::{BlockId, PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    // ========================================================================
    // Test: Multi-suspend block handling
    // ========================================================================

    /// Test helper: builds handler_dispatch with multiple suspend blocks and applies pattern.
    /// Returns the number of blocks in the suspend region.
    #[salsa::tracked]
    fn handler_dispatch_suspend_block_count(db: &dyn salsa::Database) -> usize {
        let location = test_location(db);
        let step_ty = trampoline::Step::new(db).as_type();

        // Create 3 blocks: done block + 2 suspend blocks
        let done_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());

        let mut builder1 = BlockBuilder::new(db, location);
        let zero1 = builder1.op(arith::Const::i32(db, location, 1));
        let step1 = builder1.op(trampoline::step_done(
            db,
            location,
            zero1.result(db),
            step_ty,
        ));
        builder1.op(scf::r#yield(db, location, vec![step1.result(db)]));
        let suspend_block1 = builder1.build();

        let mut builder2 = BlockBuilder::new(db, location);
        let zero2 = builder2.op(arith::Const::i32(db, location, 2));
        let step2 = builder2.op(trampoline::step_done(
            db,
            location,
            zero2.result(db),
            step_ty,
        ));
        builder2.op(scf::r#yield(db, location, vec![step2.result(db)]));
        let suspend_block2 = builder2.build();

        let body_region = Region::new(
            db,
            location,
            IdVec::from(vec![done_block, suspend_block1, suspend_block2]),
        );

        // Create a dummy result value for handler_dispatch
        let dummy_const = arith::Const::i32(db, location, 0);
        let result_val = dummy_const.as_operation().result(db, 0);

        // Create handler_dispatch with 3 blocks
        let dispatch_op =
            cont::handler_dispatch(db, location, result_val, step_ty, body_region).as_operation();

        // Apply pattern
        let pattern = LowerHandlerDispatchPattern;
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

        // Extract suspend region block count
        match result {
            RewriteResult::Expand(ops) if ops.len() == 2 => {
                let if_op = &ops[1];
                let regions = if_op.regions(db);
                if !regions.is_empty() {
                    regions[0].blocks(db).len()
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    #[salsa_test]
    fn test_handler_dispatch_collects_all_suspend_blocks(db: &salsa::DatabaseImpl) {
        let count = handler_dispatch_suspend_block_count(db);
        assert_eq!(count, 2, "Suspend region should have all 2 suspend blocks");
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

    // ========================================================================
    // Test: contains_effectful_ops skips inner functions
    // ========================================================================

    /// Test helper: creates a region with an inner function containing cont.shift.
    /// The outer region should NOT be considered effectful since inner functions are skipped.
    #[salsa::tracked]
    fn contains_effectful_ops_with_inner_func(db: &dyn salsa::Database) -> bool {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        // Create inner function with cont.shift
        let inner_func = Func::build(db, location, "inner_fn", IdVec::new(), i32_ty, |entry| {
            let handler_region = Region::new(db, location, IdVec::new());
            let shift_op = entry.op(cont::shift(
                db,
                location,
                vec![],
                i32_ty,
                0,
                ability_ref_ty,
                Symbol::new("get"),
                handler_region,
            ));
            entry.op(func::r#return(db, location, Some(shift_op.result(db))));
        });

        // Outer region contains only the inner function definition (no direct effectful ops)
        let mut outer_builder = BlockBuilder::new(db, location);
        outer_builder.op(inner_func.as_operation());
        outer_builder.op(arith::Const::i32(db, location, 42));
        let outer_block = outer_builder.build();
        let outer_region = Region::new(db, location, IdVec::from(vec![outer_block]));

        contains_effectful_ops(db, &outer_region)
    }

    /// Test helper: creates a region with cont.shift directly (no inner function).
    #[salsa::tracked]
    fn contains_effectful_ops_direct_shift(db: &dyn salsa::Database) -> bool {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        // Region with direct cont.shift (not inside inner function)
        let mut builder = BlockBuilder::new(db, location);
        let handler_region = Region::new(db, location, IdVec::new());
        builder.op(cont::shift(
            db,
            location,
            vec![],
            i32_ty,
            0,
            ability_ref_ty,
            Symbol::new("get"),
            handler_region,
        ));
        let block = builder.build();
        let region = Region::new(db, location, IdVec::from(vec![block]));

        contains_effectful_ops(db, &region)
    }

    #[salsa_test]
    fn test_contains_effectful_ops_skips_inner_functions(db: &salsa::DatabaseImpl) {
        // Inner function's effectful ops should NOT make the outer region effectful
        assert!(
            !contains_effectful_ops_with_inner_func(db),
            "Outer region should not be effectful when shift is only in inner function"
        );
    }

    #[salsa_test]
    fn test_contains_effectful_ops_detects_direct_shift(db: &salsa::DatabaseImpl) {
        // Direct shift in the region should be detected
        assert!(
            contains_effectful_ops_direct_shift(db),
            "Region with direct cont.shift should be effectful"
        );
    }
}
