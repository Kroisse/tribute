use std::collections::HashMap;

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self, Func};
use trunk_ir::dialect::{adt, cont, trampoline};
use trunk_ir::rewrite::{OpAdaptor, RewritePattern, RewriteResult};
use trunk_ir::{DialectOp, DialectType, IdVec, Operation, Span, Symbol, Type, Value};

use super::{ResumeCounter, ResumeFuncSpec, ResumeSpecs, ShiftAnalysis, compute_op_idx};

// ============================================================================
// Resume Function Helpers
// ============================================================================

/// Generate a unique resume function name using the shared counter.
pub(crate) fn fresh_resume_name(counter: &ResumeCounter) -> String {
    let mut counter = counter.borrow_mut();
    let id = *counter;
    *counter += 1;
    format!("__resume_{}", id)
}

/// Information used for generating unique state type names.
pub(crate) struct StateTypeKey {
    pub(crate) ability_name: Option<Symbol>,
    pub(crate) op_name: Option<Symbol>,
    pub(crate) module_name: Symbol,
    pub(crate) span: Span,
    pub(crate) shift_index: usize,
}

/// Generate a unique state type name based on ability, operation info, module, span and shift index.
/// Uses module name and span for uniqueness since the tag is runtime-determined.
pub(crate) fn state_type_name(key: StateTypeKey) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    if let Some(ability) = key.ability_name {
        ability.to_string().hash(&mut hasher);
    }
    if let Some(name) = key.op_name {
        name.to_string().hash(&mut hasher);
    }
    // Include module name for cross-module uniqueness
    key.module_name.to_string().hash(&mut hasher);
    // Use span (start, end) for uniqueness
    key.span.start.hash(&mut hasher);
    key.span.end.hash(&mut hasher);
    key.shift_index.hash(&mut hasher);

    let hash = hasher.finish();
    format!("__State_{:012x}", hash & 0xFFFF_FFFF_FFFF)
}

// ============================================================================
// Pattern: Lower cont.shift
// ============================================================================

pub(crate) struct LowerShiftPattern<'db> {
    pub(crate) resume_specs: ResumeSpecs<'db>,
    pub(crate) resume_counter: ResumeCounter,
    pub(crate) shift_analysis: ShiftAnalysis<'db>,
    pub(crate) module_name: Symbol,
}

impl<'db> RewritePattern<'db> for LowerShiftPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match cont.shift with tag as first operand
        let Ok(shift_op) = cont::Shift::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let operands: Vec<_> = adaptor.operands().iter().copied().collect();
        let tag_operand = operands[0]; // First operand is tag
        let value_operands: Vec<Value<'db>> = operands.into_iter().skip(1).collect(); // Rest are values
        let ability_ref_type = shift_op.ability_ref(db);
        let op_name_sym = shift_op.op_name(db);

        let location = op.location(db);

        // Use op_offset from the shift operation if available (set by resolve_evidence)
        // Otherwise fall back to hash-based op_idx for backwards compatibility
        let ability_name =
            core::AbilityRefType::from_type(db, ability_ref_type).and_then(|ar| ar.name(db));
        let op_name = Some(op_name_sym);
        let op_idx = match shift_op.op_offset(db) {
            Some(offset) => offset,
            None => compute_op_idx(ability_name, op_name),
        };

        // Look up shift point analysis - fail fast if missing
        let shift_point_info = self.shift_analysis.get(&location.span).unwrap_or_else(|| {
            panic!(
                "missing shift analysis for cont.shift at {:?} (ability: {:?}, op: {:?})",
                location.span, ability_name, op_name
            )
        });

        let mut ops = Vec::new();

        // === 1. Build State Struct with live values ===
        let state_type_key = StateTypeKey {
            ability_name,
            op_name,
            module_name: self.module_name,
            span: location.span,
            shift_index: shift_point_info.index,
        };
        let state_name = Symbol::from_dynamic(&state_type_name(state_type_key));

        // Get live values and their types from analysis.
        // State fields use anyref as field type to match the WASM lowering
        // (LowerBuildStatePattern converts all fields to anyref). This ensures
        // the state struct's nominal type is consistent between creation and extraction.
        let anyref_ty = tribute_rt::any_type(db);
        let state_values: Vec<Value<'db>> = shift_point_info
            .live_values
            .iter()
            .map(|(v, _ty)| *v)
            .collect();
        let original_field_types: Vec<Type<'db>> = shift_point_info
            .live_values
            .iter()
            .map(|(_v, ty)| *ty)
            .collect();
        let state_fields: Vec<(Symbol, Type<'db>)> = (0..shift_point_info.live_values.len())
            .map(|i| {
                let field_name = Symbol::from_dynamic(&format!("field_{}", i));
                (field_name, anyref_ty)
            })
            .collect();

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
        let next_resume_name = if shift_point_info.index + 1 >= shift_point_info.total_shifts {
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
            original_field_types,
            original_live_values: state_values.clone(),
            shift_result_value: shift_point_info.shift_result_value,
            shift_result_type: shift_point_info.shift_result_type,
            continuation_ops: shift_point_info.continuation_ops.clone(),
            next_resume_name,
            location,
            shift_analysis: self.shift_analysis.clone(),
            module_name: self.module_name,
        });

        let resume_name_sym = Symbol::from_dynamic(&resume_name);
        let const_op = func::constant(db, location, i32_ty, resume_name_sym);
        let resume_fn_val = const_op.as_operation().result(db, 0);
        ops.push(const_op.as_operation());

        // === 3. Get shift value (the value passed to the effect operation) ===
        // Note: shift value may be absent if the ability operation has no arguments.
        // In that case, we use state_val as a placeholder (will be ignored by resume).
        let shift_value_val = value_operands.first().copied().unwrap_or(state_val);

        // === 4. Build Continuation with tag operand ===
        let cont_ty = trampoline::Continuation::new(db).as_type();
        let step_ty = trampoline::Step::new(db).as_type();

        let cont_op = trampoline::build_continuation(
            db,
            location,
            tag_operand,
            resume_fn_val,
            state_val,
            shift_value_val,
            cont_ty,
            op_idx,
        );
        let cont_val = cont_op.as_operation().result(db, 0);
        ops.push(cont_op.as_operation());

        // === 5. Set Yield State ===
        let set_yield_op = trampoline::set_yield_state(db, location, tag_operand, cont_val, op_idx);
        ops.push(set_yield_op.as_operation());

        // === 6. Return Step::Shift ===
        let step_op = trampoline::step_shift(db, location, tag_operand, cont_val, step_ty, op_idx);
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
pub(crate) fn create_resume_function_with_continuation<'db>(
    db: &'db dyn salsa::Database,
    spec: &ResumeFuncSpec<'db>,
) -> Operation<'db> {
    let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type(db);
    let wrapper_ty = trampoline::ResumeWrapper::new(db).as_type();
    let step_ty = trampoline::Step::new(db).as_type();
    let anyref_ty = tribute_rt::Any::new(db).as_type();
    let location = spec.location;
    let name = Symbol::from_dynamic(&spec.name);

    // Resume functions take (evidence, wrapper) where wrapper is
    // trampoline.resume_wrapper. ConvertFuncTypePattern will later convert
    // this to _ResumeWrapper ADT, matching the call_indirect signature.
    let func_op = Func::build(
        db,
        location,
        name,
        IdVec::from(vec![evidence_ty, wrapper_ty]),
        step_ty,
        |builder| {
            // Evidence is at index 0 (unused but required for calling convention)
            // Wrapper is at index 1, already the correct type
            let wrapper_arg = builder.block_arg(db, 1);

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

                // Extract each captured local from state and map to original value.
                // State fields are all anyref (matching LowerBuildStatePattern in WASM lowering).
                // We cast each extracted anyref value to the original field type.
                let anyref_ty = tribute_rt::any_type(db);
                for (i, (original_field_type, original_value)) in spec
                    .original_field_types
                    .iter()
                    .zip(spec.original_live_values.iter())
                    .enumerate()
                {
                    let get_field = builder.op(adt::struct_get(
                        db,
                        location,
                        state_val,
                        anyref_ty,
                        spec.state_type,
                        i as u64,
                    ));
                    let anyref_value = get_field.result(db);

                    // Cast anyref to the original field type
                    let cast_op = builder.op(core::unrealized_conversion_cast(
                        db,
                        location,
                        anyref_value,
                        *original_field_type,
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
                // cont.shift takes tag as first operand (dynamic)
                if let Ok(shift_op) = cont::Shift::from_operation(db, *op) {
                    let operands = op.operands(db);
                    let tag_operand = operands
                        .first()
                        .copied()
                        .expect("shift missing tag operand");
                    let ability_ref_type = shift_op.ability_ref(db);
                    let op_name_sym = shift_op.op_name(db);
                    let shift_value_operand = operands.get(1).copied(); // Skip tag, get first value operand
                    // Get next resume function name
                    let next_resume_name = spec.next_resume_name.as_ref().expect(
                        "encountered shift in continuation but no next_resume_name specified",
                    );

                    // Look up shift analysis for this shift point - fail fast if missing
                    let shift_span = op.location(db).span;
                    let next_shift_info =
                        spec.shift_analysis.get(&shift_span).unwrap_or_else(|| {
                            panic!(
                                "missing shift analysis for nested shift at {:?}",
                                shift_span
                            )
                        });

                    // Get shift properties
                    // Use op_offset from the shift operation if available (set by resolve_evidence)
                    // Otherwise fall back to hash-based op_idx for backwards compatibility
                    let ability_name = core::AbilityRefType::from_type(db, ability_ref_type)
                        .and_then(|ar| ar.name(db));
                    let op_name = Some(op_name_sym);
                    let op_idx = match shift_op.op_offset(db) {
                        Some(offset) => offset,
                        None => compute_op_idx(ability_name, op_name),
                    };

                    // Build state struct with current live values (remapped)
                    let shift_index = next_shift_info.index;
                    let state_type_key = StateTypeKey {
                        ability_name,
                        op_name,
                        module_name: spec.module_name,
                        span: shift_span,
                        shift_index,
                    };
                    let state_name = Symbol::from_dynamic(&state_type_name(state_type_key));
                    let anyref_ty = tribute_rt::any_type(db);
                    let state_values: Vec<Value<'db>> = next_shift_info
                        .live_values
                        .iter()
                        .map(|(v, _ty)| *value_map.get(v).unwrap_or(v))
                        .collect();
                    let state_fields: Vec<(Symbol, Type<'db>)> =
                        (0..next_shift_info.live_values.len())
                            .map(|i| {
                                let field_name = Symbol::from_dynamic(&format!("field_{}", i));
                                (field_name, anyref_ty)
                            })
                            .collect();

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
                    let shift_value_val = shift_value_operand
                        .map(|v| *value_map.get(&v).unwrap_or(&v))
                        .unwrap_or(state_val);

                    // Remap tag operand
                    let tag_val = *value_map.get(&tag_operand).unwrap_or(&tag_operand);

                    // Build continuation with tag operand
                    let cont_ty = trampoline::Continuation::new(db).as_type();
                    let cont_op = builder.op(trampoline::build_continuation(
                        db,
                        location,
                        tag_val,
                        resume_fn_val,
                        state_val,
                        shift_value_val,
                        cont_ty,
                        op_idx,
                    ));
                    let cont_val = cont_op.result(db);

                    // Set yield state with tag operand
                    builder.op(trampoline::set_yield_state(
                        db, location, tag_val, cont_val, op_idx,
                    ));

                    // Create step_shift with tag operand
                    let step_shift = builder.op(trampoline::step_shift(
                        db, location, tag_val, cont_val, step_ty, op_idx,
                    ));
                    let step_shift_val = step_shift.result(db);
                    builder.op(func::r#return(db, location, Some(step_shift_val)));

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
