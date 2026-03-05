use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::cont as arena_cont;
use trunk_ir::arena::dialect::core as arena_core;
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::dialect::trampoline as arena_trampoline;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::PatternRewriter as ArenaPatternRewriter;
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::location::Span;

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
pub(crate) fn state_type_name(key: StateTypeKey) -> String {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();

    if let Some(ability) = key.ability_name {
        ability.to_string().hash(&mut hasher);
    }
    if let Some(name) = key.op_name {
        name.to_string().hash(&mut hasher);
    }
    key.module_name.to_string().hash(&mut hasher);
    key.span.start.hash(&mut hasher);
    key.span.end.hash(&mut hasher);
    key.shift_index.hash(&mut hasher);

    let hash = hasher.finish();
    format!("__State_{:012x}", hash & 0xFFFF_FFFF_FFFF)
}

/// Extract ability name and compute op_idx from a cont.shift operation.
fn resolve_shift_op_idx(
    ctx: &IrContext,
    shift_op: arena_cont::Shift,
) -> (Option<Symbol>, Option<Symbol>, u32) {
    let ability_ref_type = shift_op.ability_ref(ctx);
    let ability_data = ctx.types.get(ability_ref_type);
    let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
        Some(ArenaAttribute::Symbol(s)) => Some(*s),
        _ => None,
    };
    let op_name_sym = shift_op.op_name(ctx);
    let op_name = Some(op_name_sym);
    let op_idx = match shift_op.op_offset(ctx) {
        Some(offset) => offset,
        None => compute_op_idx(ability_name, op_name),
    };
    (ability_name, op_name, op_idx)
}

/// Create state field descriptors with anyref types for WASM lowering compatibility.
fn create_state_fields(count: usize, anyref_ty: TypeRef) -> Vec<(Symbol, TypeRef)> {
    (0..count)
        .map(|i| {
            let field_name = Symbol::from_dynamic(&format!("field_{}", i));
            (field_name, anyref_ty)
        })
        .collect()
}

/// Helper to intern the anyref type.
pub(crate) fn anyref_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("tribute_rt"), Symbol::new("Any")).build())
}

/// Helper to intern the step type.
pub(crate) fn step_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("trampoline"), Symbol::new("step")).build())
}

/// Helper to intern the continuation type.
pub(crate) fn continuation_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("trampoline"), Symbol::new("continuation")).build(),
    )
}

/// Helper to intern the resume_wrapper type.
pub(crate) fn resume_wrapper_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("trampoline"), Symbol::new("resume_wrapper")).build(),
    )
}

/// Helper to intern i32 type.
pub(crate) fn i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Helper to create an ADT struct type.
pub(crate) fn adt_struct_type(
    ctx: &mut IrContext,
    name: Symbol,
    fields: &[(Symbol, TypeRef)],
) -> TypeRef {
    let fields_attr = ArenaAttribute::List(
        fields
            .iter()
            .map(|(fname, fty)| {
                ArenaAttribute::List(vec![
                    ArenaAttribute::Symbol(*fname),
                    ArenaAttribute::Type(*fty),
                ])
            })
            .collect(),
    );

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", ArenaAttribute::Symbol(name))
            .attr("fields", fields_attr)
            .build(),
    )
}

// ============================================================================
// Pattern: Lower cont.shift
// ============================================================================

pub(crate) struct LowerShiftPattern {
    pub(crate) resume_specs: ResumeSpecs,
    pub(crate) resume_counter: ResumeCounter,
    pub(crate) shift_analysis: ShiftAnalysis,
    pub(crate) module_name: Symbol,
}

impl trunk_ir::arena::rewrite::ArenaRewritePattern for LowerShiftPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        // Match cont.shift with tag as first operand
        let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) else {
            return false;
        };

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let tag_operand = *operands.first().expect("cont.shift requires a tag operand");
        let value_operands: Vec<ValueRef> = operands.into_iter().skip(1).collect();

        let location = ctx.op(op).location;
        let (ability_name, op_name, op_idx) = resolve_shift_op_idx(ctx, shift_op);

        // Look up shift point analysis - fail fast if missing
        let shift_point_info = self.shift_analysis.get(&location.span).unwrap_or_else(|| {
            panic!(
                "missing shift analysis for cont.shift at {:?} (ability: {:?}, op: {:?})",
                location.span, ability_name, op_name
            )
        });

        let anyref_ty = anyref_type(ctx);
        let i32_ty = i32_type(ctx);
        let step_ty = step_type(ctx);
        let cont_ty = continuation_type(ctx);

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

        let state_values: Vec<ValueRef> = shift_point_info
            .live_values
            .iter()
            .map(|(v, _ty)| *v)
            .collect();
        let original_field_types: Vec<TypeRef> = shift_point_info
            .live_values
            .iter()
            .map(|(_v, ty)| *ty)
            .collect();
        let state_fields = create_state_fields(shift_point_info.live_values.len(), anyref_ty);

        let state_adt_ty = adt_struct_type(ctx, state_name, &state_fields);
        let state_op = arena_trampoline::build_state(
            ctx,
            location,
            state_values.clone(),
            state_adt_ty,
            state_adt_ty,
        );
        let state_val = state_op.result(ctx);
        ops.push(state_op.op_ref());

        // === 2. Get resume function reference (i32 table index) ===
        let resume_name = fresh_resume_name(&self.resume_counter);

        let next_resume_name = if shift_point_info.index + 1 >= shift_point_info.total_shifts {
            None
        } else {
            let counter = self.resume_counter.borrow();
            let next_id = *counter;
            Some(format!("__resume_{}", next_id))
        };

        // Record resume function spec with continuation info
        self.resume_specs.borrow_mut().push(ResumeFuncSpec {
            name: resume_name.clone(),
            state_type: state_adt_ty,
            state_fields,
            original_field_types,
            original_live_values: state_values,
            shift_result_value: shift_point_info.shift_result_value,
            shift_result_type: shift_point_info.shift_result_type,
            continuation_ops: shift_point_info.continuation_ops.clone(),
            next_resume_name,
            location,
            shift_analysis: self.shift_analysis.clone(),
            module_name: self.module_name,
        });

        let resume_name_sym = Symbol::from_dynamic(&resume_name);
        let const_op = arena_func::constant(ctx, location, i32_ty, resume_name_sym);
        let resume_fn_val = const_op.result(ctx);
        ops.push(const_op.op_ref());

        // === 3. Get shift value ===
        let shift_value_val = value_operands.first().copied().unwrap_or(state_val);

        // === 4. Build Continuation with tag operand ===
        let cont_op = arena_trampoline::build_continuation(
            ctx,
            location,
            tag_operand,
            resume_fn_val,
            state_val,
            shift_value_val,
            cont_ty,
            op_idx,
        );
        let cont_val = cont_op.result(ctx);
        ops.push(cont_op.op_ref());

        // === 5. Set Yield State ===
        let set_yield_op =
            arena_trampoline::set_yield_state(ctx, location, tag_operand, cont_val, op_idx);
        ops.push(set_yield_op.op_ref());

        // === 6. Return Step::Shift ===
        let step_op =
            arena_trampoline::step_shift(ctx, location, tag_operand, cont_val, step_ty, op_idx);
        ops.push(step_op.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

/// Create a resume function with continuation code.
pub(crate) fn create_resume_function_with_continuation(
    ctx: &mut IrContext,
    spec: &ResumeFuncSpec,
) -> OpRef {
    let evidence_ty = tribute_ir::arena::dialect::ability::evidence_adt_type_ref(ctx);
    let wrapper_ty = resume_wrapper_type(ctx);
    let step_ty = step_type(ctx);
    let anyref_ty = anyref_type(ctx);
    let location = spec.location;
    let name = Symbol::from_dynamic(&spec.name);

    // Create function type: (evidence, wrapper) -> Step
    let func_ty_ref = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .param(evidence_ty)
            .param(wrapper_ty)
            .attr("result", ArenaAttribute::Type(step_ty))
            .build(),
    );

    // Build function body
    let body_block = ctx.create_block(trunk_ir::arena::context::BlockData {
        location,
        args: vec![
            trunk_ir::arena::context::BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            },
            trunk_ir::arena::context::BlockArgData {
                ty: wrapper_ty,
                attrs: Default::default(),
            },
        ],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    // Evidence is at index 0 (unused but required for calling convention)
    // Wrapper is at index 1
    let wrapper_arg = ctx.block_args(body_block)[1];

    // Build value map for remapping
    let mut value_map: HashMap<ValueRef, ValueRef> = HashMap::new();

    // Extract resume_value from wrapper (field 1 = resume_value)
    let get_resume_value =
        arena_trampoline::resume_wrapper_get(ctx, location, wrapper_arg, anyref_ty, 1);
    ctx.push_op(body_block, get_resume_value.op_ref());
    let mut resume_value = get_resume_value.result(ctx);

    // Cast resume_value to the original shift result type if needed
    if let Some(result_type) = spec.shift_result_type {
        let cast_op =
            arena_core::unrealized_conversion_cast(ctx, location, resume_value, result_type);
        ctx.push_op(body_block, cast_op.op_ref());
        resume_value = cast_op.result(ctx);
    }

    // Map shift result to resume_value (now properly typed)
    if let Some(shift_result) = spec.shift_result_value {
        value_map.insert(shift_result, resume_value);
    }

    // If we have state fields, extract the state and get captured values
    if !spec.state_fields.is_empty() {
        let get_state =
            arena_trampoline::resume_wrapper_get(ctx, location, wrapper_arg, spec.state_type, 0);
        ctx.push_op(body_block, get_state.op_ref());
        let state_val = get_state.result(ctx);

        // Extract each captured local from state and map to original value.
        for (i, (original_field_type, original_value)) in spec
            .original_field_types
            .iter()
            .zip(spec.original_live_values.iter())
            .enumerate()
        {
            let get_field = arena_adt::struct_get(
                ctx,
                location,
                state_val,
                anyref_ty,
                spec.state_type,
                i as u32,
            );
            ctx.push_op(body_block, get_field.op_ref());
            let anyref_value = get_field.result(ctx);

            // Cast anyref to the original field type
            let cast_op = arena_core::unrealized_conversion_cast(
                ctx,
                location,
                anyref_value,
                *original_field_type,
            );
            ctx.push_op(body_block, cast_op.op_ref());
            let extracted_value = cast_op.result(ctx);
            value_map.insert(*original_value, extracted_value);
        }
    }

    // Execute continuation operations with value remapping
    let mut last_result: Option<ValueRef> = None;
    let mut encountered_shift = false;

    for &op in &spec.continuation_ops {
        // Skip func.return - we'll handle the final return ourselves
        if arena_func::Return::from_op(ctx, op).is_ok() {
            if let Some(&return_val) = ctx.op_operands(op).first() {
                last_result = Some(*value_map.get(&return_val).unwrap_or(&return_val));
            }
            continue;
        }

        // Handle cont.shift - transform to step_shift with next resume function
        if let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) {
            let operands = ctx.op_operands(op).to_vec();
            let tag_operand = operands
                .first()
                .copied()
                .expect("shift missing tag operand");
            let shift_value_operand = operands.get(1).copied();

            let next_resume_name = spec
                .next_resume_name
                .as_ref()
                .expect("encountered shift in continuation but no next_resume_name specified");

            let shift_span = ctx.op(op).location.span;
            let next_shift_info = spec.shift_analysis.get(&shift_span).unwrap_or_else(|| {
                panic!(
                    "missing shift analysis for nested shift at {:?}",
                    shift_span
                )
            });

            let (ability_name, op_name, op_idx) = resolve_shift_op_idx(ctx, shift_op);

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
            let state_values: Vec<ValueRef> = next_shift_info
                .live_values
                .iter()
                .map(|(v, _ty)| *value_map.get(v).unwrap_or(v))
                .collect();
            let state_fields = create_state_fields(next_shift_info.live_values.len(), anyref_ty);

            let state_adt_ty = adt_struct_type(ctx, state_name, &state_fields);
            let state_op = arena_trampoline::build_state(
                ctx,
                location,
                state_values,
                state_adt_ty,
                state_adt_ty,
            );
            ctx.push_op(body_block, state_op.op_ref());
            let state_val = state_op.result(ctx);

            // Get resume function reference
            let i32_ty = i32_type(ctx);
            let resume_name_sym = Symbol::from_dynamic(next_resume_name);
            let const_op = arena_func::constant(ctx, location, i32_ty, resume_name_sym);
            ctx.push_op(body_block, const_op.op_ref());
            let resume_fn_val = const_op.result(ctx);

            // Get shift value (remap if needed)
            let shift_value_val = shift_value_operand
                .map(|v| *value_map.get(&v).unwrap_or(&v))
                .unwrap_or(state_val);

            // Remap tag operand
            let tag_val = *value_map.get(&tag_operand).unwrap_or(&tag_operand);

            // Build continuation with tag operand
            let cont_ty = continuation_type(ctx);
            let cont_op = arena_trampoline::build_continuation(
                ctx,
                location,
                tag_val,
                resume_fn_val,
                state_val,
                shift_value_val,
                cont_ty,
                op_idx,
            );
            ctx.push_op(body_block, cont_op.op_ref());
            let cont_val = cont_op.result(ctx);

            // Set yield state with tag operand
            let set_yield =
                arena_trampoline::set_yield_state(ctx, location, tag_val, cont_val, op_idx);
            ctx.push_op(body_block, set_yield.op_ref());

            // Create step_shift with tag operand
            let step_shift =
                arena_trampoline::step_shift(ctx, location, tag_val, cont_val, step_ty, op_idx);
            ctx.push_op(body_block, step_shift.op_ref());
            let step_shift_val = step_shift.result(ctx);

            let ret = arena_func::r#return(ctx, location, [step_shift_val]);
            ctx.push_op(body_block, ret.op_ref());

            encountered_shift = true;
            break;
        }

        // Remap operands
        let old_operands = ctx.op_operands(op).to_vec();
        let remapped_operands: Vec<ValueRef> = old_operands
            .iter()
            .map(|&v| *value_map.get(&v).unwrap_or(&v))
            .collect();

        // Clone the operation with remapped operands
        let needs_rebuild = remapped_operands != old_operands;
        if needs_rebuild {
            let op_data = ctx.op(op);
            let mut builder = trunk_ir::arena::context::OperationDataBuilder::new(
                op_data.location,
                op_data.dialect,
                op_data.name,
            )
            .operands(remapped_operands)
            .results(ctx.op_result_types(op).to_vec());
            for (k, v) in &op_data.attributes {
                builder = builder.attr(*k, v.clone());
            }
            for &r in &op_data.regions {
                builder = builder.region(r);
            }
            let new_data = builder.build(ctx);
            let new_op = ctx.create_op(new_data);
            ctx.push_op(body_block, new_op);

            // Map old results to new results
            let old_results = ctx.op_results(op).to_vec();
            let new_results = ctx.op_results(new_op).to_vec();
            for (old_r, new_r) in old_results.iter().zip(new_results.iter()) {
                if old_r != new_r {
                    value_map.insert(*old_r, *new_r);
                }
            }

            let new_results = ctx.op_results(new_op);
            if !new_results.is_empty() {
                last_result = Some(new_results[0]);
            }
        } else {
            ctx.push_op(body_block, op);
            let results = ctx.op_results(op);
            if !results.is_empty() {
                last_result = Some(results[0]);
            }
        }
    }

    // Return Step.Done with the final result (only if no shift was encountered)
    if !encountered_shift {
        let final_value = last_result.unwrap_or(resume_value);
        let step_done = arena_trampoline::step_done(ctx, location, final_value, step_ty);
        ctx.push_op(body_block, step_done.op_ref());
        let step_done_val = step_done.result(ctx);
        let ret = arena_func::r#return(ctx, location, [step_done_val]);
        ctx.push_op(body_block, ret.op_ref());
    }

    let body_region = ctx.create_region(trunk_ir::arena::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![body_block],
        parent_op: None,
    });

    // Create the func.func operation
    let func_data = trunk_ir::arena::context::OperationDataBuilder::new(
        location,
        Symbol::new("func"),
        Symbol::new("func"),
    )
    .attr("sym_name", ArenaAttribute::Symbol(name))
    .attr("type", ArenaAttribute::Type(func_ty_ref))
    .region(body_region)
    .build(ctx);

    ctx.create_op(func_data)
}
