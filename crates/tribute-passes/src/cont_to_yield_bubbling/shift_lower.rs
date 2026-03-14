//! Lower `cont.shift` to YieldResult::Shift construction.
//!
//! Each `cont.shift` is replaced by:
//! 1. Capture live variables into a state struct
//! 2. Create a Continuation with resume_fn reference and state
//! 3. Create ShiftInfo with value, prompt tag, op_idx, and continuation
//! 4. Return `YieldResult::Shift(shift_info)`

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::location::Span;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::types::{Attribute, TypeDataBuilder};

use super::types::{YieldBubblingTypes, adt_struct_type};
use super::{ResumeCounter, ResumeFuncSpec, ResumeSpecs, ShiftAnalysis};
use crate::cont_util::compute_op_idx;

// ============================================================================
// Resume Function Helpers
// ============================================================================

/// Generate a unique resume function name using the shared counter.
pub(crate) fn fresh_resume_name(counter: &ResumeCounter) -> String {
    let mut counter = counter.borrow_mut();
    let id = *counter;
    *counter += 1;
    format!("__yb_resume_{}", id)
}

/// Information used for generating unique state type names.
pub(crate) struct StateTypeKey {
    pub(crate) ability_name: Option<Symbol>,
    pub(crate) op_name: Option<Symbol>,
    pub(crate) module_name: Symbol,
    pub(crate) span: Span,
    pub(crate) shift_index: usize,
}

/// Generate a unique state type name.
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
    format!("__YBState_{:012x}", hash & 0xFFFF_FFFF_FFFF)
}

/// Extract ability name and compute op_idx from a cont.shift operation.
fn resolve_shift_op_idx(
    ctx: &IrContext,
    shift_op: arena_cont::Shift,
) -> (Option<Symbol>, Option<Symbol>, u32) {
    let ability_ref_type = shift_op.ability_ref(ctx);
    let ability_data = ctx.types.get(ability_ref_type);
    let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
        Some(Attribute::Symbol(s)) => Some(*s),
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

/// Create state field descriptors with anyref types.
fn create_state_fields(count: usize, anyref_ty: TypeRef) -> Vec<(Symbol, TypeRef)> {
    (0..count)
        .map(|i| {
            let field_name = Symbol::from_dynamic(&format!("field_{}", i));
            (field_name, anyref_ty)
        })
        .collect()
}

// ============================================================================
// Pattern: Lower cont.shift → YieldResult::Shift
// ============================================================================

pub(crate) struct LowerShiftPattern {
    pub(crate) types: YieldBubblingTypes,
    pub(crate) resume_specs: ResumeSpecs,
    pub(crate) resume_counter: ResumeCounter,
    pub(crate) shift_analysis: ShiftAnalysis,
    pub(crate) module_name: Symbol,
}

impl RewritePattern for LowerShiftPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) else {
            return false;
        };

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let tag_operand = *operands.first().expect("cont.shift requires a tag operand");
        let value_operands: Vec<ValueRef> = operands.into_iter().skip(1).collect();

        let location = ctx.op(op).location;
        let (ability_name, op_name, op_idx) = resolve_shift_op_idx(ctx, shift_op);

        // Look up shift point analysis
        let shift_point_info = self.shift_analysis.get(&location.span).unwrap_or_else(|| {
            panic!(
                "missing shift analysis for cont.shift at {:?} (ability: {:?}, op: {:?})",
                location.span, ability_name, op_name
            )
        });

        let t = &self.types;
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
        let state_fields = create_state_fields(shift_point_info.live_values.len(), t.anyref);

        let state_adt_ty = adt_struct_type(ctx, state_name, &state_fields);

        // Cast live values to anyref, then create state struct
        let mut anyref_state_values: Vec<ValueRef> = Vec::new();
        for &(val, _ty) in &shift_point_info.live_values {
            let cast = arena_core::unrealized_conversion_cast(ctx, location, val, t.anyref);
            ops.push(cast.op_ref());
            anyref_state_values.push(cast.result(ctx));
        }

        let state_op =
            arena_adt::struct_new(ctx, location, anyref_state_values, t.anyref, state_adt_ty);
        let state_val = state_op.result(ctx);
        ops.push(state_op.op_ref());

        // === 2. Get resume function reference (i32 table index) ===
        let resume_name = fresh_resume_name(&self.resume_counter);

        let next_resume_name = if shift_point_info.index + 1 >= shift_point_info.total_shifts {
            None
        } else {
            let counter = self.resume_counter.borrow();
            let next_id = *counter;
            Some(format!("__yb_resume_{}", next_id))
        };

        // Record resume function spec
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
        let const_op = arena_func::constant(ctx, location, t.ptr, resume_name_sym);
        let resume_fn_val = const_op.result(ctx);
        ops.push(const_op.op_ref());

        // === 3. Build Continuation struct ===
        let cont_op = arena_adt::struct_new(
            ctx,
            location,
            vec![resume_fn_val, state_val],
            t.anyref,
            t.continuation,
        );
        let cont_val = cont_op.result(ctx);
        ops.push(cont_op.op_ref());

        // Cast continuation to anyref for ShiftInfo
        let cont_anyref = arena_core::unrealized_conversion_cast(ctx, location, cont_val, t.anyref);
        ops.push(cont_anyref.op_ref());

        // === 4. Build ShiftInfo struct ===
        // shift_value: first value operand cast to anyref, or null
        let shift_value_val = if let Some(&sv) = value_operands.first() {
            let cast = arena_core::unrealized_conversion_cast(ctx, location, sv, t.anyref);
            ops.push(cast.op_ref());
            cast.result(ctx)
        } else {
            let null_op = arena_adt::ref_null(ctx, location, t.anyref, t.anyref);
            ops.push(null_op.op_ref());
            null_op.result(ctx)
        };

        // prompt tag: cast to i32
        let prompt_val = arena_core::unrealized_conversion_cast(ctx, location, tag_operand, t.i32);
        ops.push(prompt_val.op_ref());

        // op_idx constant
        let op_idx_const =
            trunk_ir::dialect::arith::r#const(ctx, location, t.i32, Attribute::Int(op_idx as i128));
        ops.push(op_idx_const.op_ref());

        let shift_info_op = arena_adt::struct_new(
            ctx,
            location,
            vec![
                shift_value_val,
                prompt_val.result(ctx),
                op_idx_const.result(ctx),
                cont_anyref.result(ctx),
            ],
            t.shift_info,
            t.shift_info,
        );
        let shift_info_val = shift_info_op.result(ctx);
        ops.push(shift_info_op.op_ref());

        // === 5. Build YieldResult::Shift ===
        let yr_op = arena_adt::variant_new(
            ctx,
            location,
            [shift_info_val],
            t.yield_result,
            t.yield_result,
            Symbol::new("Shift"),
        );
        ops.push(yr_op.op_ref());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Resume Function Generation
// ============================================================================

/// Create a resume function from a spec.
///
/// Resume function signature: `(Evidence, anyref) -> YieldResult`
/// where the anyref argument is a ResumeWrapper.
pub(crate) fn create_resume_function(
    ctx: &mut IrContext,
    spec: &ResumeFuncSpec,
    types: &YieldBubblingTypes,
) -> OpRef {
    let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);
    let location = spec.location;
    let name = Symbol::from_dynamic(&spec.name);

    // Function type: (evidence, anyref) -> YieldResult
    // Use Layout A: params[0] = return type, params[1..] = parameter types
    // (Cranelift backend's translate_signature expects Layout A)
    let func_ty_ref = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
            .param(types.yield_result)
            .param(evidence_ty)
            .param(types.anyref)
            .build(),
    );

    // Build function body
    let body_block = ctx.create_block(trunk_ir::context::BlockData {
        location,
        args: vec![
            trunk_ir::context::BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            },
            trunk_ir::context::BlockArgData {
                ty: types.anyref,
                attrs: Default::default(),
            },
        ],
        ops: trunk_ir::smallvec::smallvec![],
        parent_region: None,
    });

    let wrapper_arg = ctx.block_args(body_block)[1];

    let mut mapping = IrMapping::new();

    // Cast wrapper to ResumeWrapper
    let wrapper_cast =
        arena_core::unrealized_conversion_cast(ctx, location, wrapper_arg, types.resume_wrapper);
    ctx.push_op(body_block, wrapper_cast.op_ref());
    let wrapper_val = wrapper_cast.result(ctx);

    // Extract resume_value from wrapper (field 1)
    let get_rv = arena_adt::struct_get(
        ctx,
        location,
        wrapper_val,
        types.anyref,
        types.resume_wrapper,
        1,
    );
    ctx.push_op(body_block, get_rv.op_ref());
    let mut resume_value = get_rv.result(ctx);

    // Cast resume_value to the original shift result type if needed
    if let Some(result_type) = spec.shift_result_type {
        let cast = arena_core::unrealized_conversion_cast(ctx, location, resume_value, result_type);
        ctx.push_op(body_block, cast.op_ref());
        resume_value = cast.result(ctx);
    }

    if let Some(shift_result) = spec.shift_result_value {
        mapping.map_value(shift_result, resume_value);
    }

    // Extract state from wrapper (field 0) and restore captured values
    let ev_arg = ctx.block_args(body_block)[0]; // evidence parameter
    let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(ctx);

    if !spec.state_fields.is_empty() {
        let get_state = arena_adt::struct_get(
            ctx,
            location,
            wrapper_val,
            types.anyref,
            types.resume_wrapper,
            0,
        );
        ctx.push_op(body_block, get_state.op_ref());
        let state_anyref = get_state.result(ctx);

        // Cast anyref to state struct type
        let state_cast =
            arena_core::unrealized_conversion_cast(ctx, location, state_anyref, spec.state_type);
        ctx.push_op(body_block, state_cast.op_ref());
        let state_val = state_cast.result(ctx);

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
                types.anyref,
                spec.state_type,
                i as u32,
            );
            ctx.push_op(body_block, get_field.op_ref());
            let anyref_value = get_field.result(ctx);

            let cast = arena_core::unrealized_conversion_cast(
                ctx,
                location,
                anyref_value,
                *original_field_type,
            );
            ctx.push_op(body_block, cast.op_ref());
            mapping.map_value(*original_value, cast.result(ctx));
        }
    }

    // Override evidence mapping: use the resume function's evidence parameter
    // instead of the restored-from-state value. After a recursive handler call,
    // the evidence may have new prompt tags that the restored value lacks.
    for (original_value, original_type) in spec
        .original_live_values
        .iter()
        .zip(spec.original_field_types.iter())
    {
        if *original_type == evidence_ty {
            mapping.map_value(*original_value, ev_arg);
        }
    }

    // After RAUW by replace_op, continuation ops may reference the
    // replacement value (e.g., adt.variant_new result) instead of the
    // original shift result or a live value. Detect and map RAUW'd values.
    // This must run AFTER state restoration so that live value mappings
    // are available for distinguishing RAUW'd live values from RAUW'd
    // shift results.
    if let Some(shift_result) = spec.shift_result_value {
        let live_value_set: std::collections::HashSet<ValueRef> =
            spec.original_live_values.iter().copied().collect();
        let cont_op_set: std::collections::HashSet<OpRef> =
            spec.continuation_ops.iter().copied().collect();
        for &cont_op in &spec.continuation_ops {
            for &operand in ctx.op_operands(cont_op) {
                if operand == shift_result || live_value_set.contains(&operand) {
                    continue;
                }
                if mapping.contains_value(operand) {
                    continue;
                }
                if let trunk_ir::refs::ValueDef::OpResult(def_op, _) = ctx.value_def(operand)
                    && !cont_op_set.contains(&def_op)
                {
                    // This operand was RAUW'd from either the current shift's result
                    // or a previous shift's result (which is now a live value).
                    // Use the defining op's span to look up the original shift result
                    // in shift_analysis, then reuse whatever mapping was established.
                    let def_span = ctx.op(def_op).location.span;
                    let mut mapped_via_analysis = false;
                    if let Some(info) = spec.shift_analysis.get(&def_span) {
                        if let Some(original_result) = info.shift_result_value {
                            let mapped = mapping.lookup_value_or_default(original_result);
                            if mapped != original_result {
                                mapping.map_value(operand, mapped);
                                mapped_via_analysis = true;
                            }
                        }
                    }
                    if !mapped_via_analysis {
                        mapping.map_value(operand, resume_value);
                    }
                }
            }
        }
    }

    // Execute continuation operations with value remapping using IrMapping.
    // clone_op_into_block automatically remaps operands via the mapping and
    // registers new result↔old result correspondences.
    let mut last_result: Option<ValueRef> = None;
    let mut encountered_shift = false;

    for &op in &spec.continuation_ops {
        // Skip func.return - handle final return ourselves
        if arena_func::Return::from_op(ctx, op).is_ok() {
            if let Some(&return_val) = ctx.op_operands(op).first() {
                last_result = Some(mapping.lookup_value_or_default(return_val));
            }
            continue;
        }

        // Handle cont.shift - transform to YieldResult::Shift with next resume function
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
            let state_fields = create_state_fields(next_shift_info.live_values.len(), types.anyref);
            let state_adt_ty = adt_struct_type(ctx, state_name, &state_fields);

            // Cast live values to anyref and create state
            let mut anyref_vals: Vec<ValueRef> = Vec::new();
            for (v, _ty) in &next_shift_info.live_values {
                let remapped = mapping.lookup_value_or_default(*v);
                let cast =
                    arena_core::unrealized_conversion_cast(ctx, location, remapped, types.anyref);
                ctx.push_op(body_block, cast.op_ref());
                anyref_vals.push(cast.result(ctx));
            }

            let state_op =
                arena_adt::struct_new(ctx, location, anyref_vals, types.anyref, state_adt_ty);
            ctx.push_op(body_block, state_op.op_ref());
            let state_val = state_op.result(ctx);

            // Resume function reference
            let resume_name_sym = Symbol::from_dynamic(next_resume_name);
            let const_op = arena_func::constant(ctx, location, types.ptr, resume_name_sym);
            ctx.push_op(body_block, const_op.op_ref());

            // Continuation struct
            let cont_op = arena_adt::struct_new(
                ctx,
                location,
                vec![const_op.result(ctx), state_val],
                types.anyref,
                types.continuation,
            );
            ctx.push_op(body_block, cont_op.op_ref());

            // Cast continuation to anyref
            let cont_anyref = arena_core::unrealized_conversion_cast(
                ctx,
                location,
                cont_op.result(ctx),
                types.anyref,
            );
            ctx.push_op(body_block, cont_anyref.op_ref());

            // Shift value
            let sv = if let Some(sv_operand) = shift_value_operand {
                let remapped = mapping.lookup_value_or_default(sv_operand);
                let cast =
                    arena_core::unrealized_conversion_cast(ctx, location, remapped, types.anyref);
                ctx.push_op(body_block, cast.op_ref());
                cast.result(ctx)
            } else {
                let null_op = arena_adt::ref_null(ctx, location, types.anyref, types.anyref);
                ctx.push_op(body_block, null_op.op_ref());
                null_op.result(ctx)
            };

            // Prompt tag
            let tag_val = mapping.lookup_value_or_default(tag_operand);
            let prompt_val =
                arena_core::unrealized_conversion_cast(ctx, location, tag_val, types.i32);
            ctx.push_op(body_block, prompt_val.op_ref());

            // Op idx
            let op_idx_const = trunk_ir::dialect::arith::r#const(
                ctx,
                location,
                types.i32,
                Attribute::Int(op_idx as i128),
            );
            ctx.push_op(body_block, op_idx_const.op_ref());

            // ShiftInfo
            let shift_info = arena_adt::struct_new(
                ctx,
                location,
                vec![
                    sv,
                    prompt_val.result(ctx),
                    op_idx_const.result(ctx),
                    cont_anyref.result(ctx),
                ],
                types.shift_info,
                types.shift_info,
            );
            ctx.push_op(body_block, shift_info.op_ref());

            // YieldResult::Shift
            let yr = arena_adt::variant_new(
                ctx,
                location,
                [shift_info.result(ctx)],
                types.yield_result,
                types.yield_result,
                Symbol::new("Shift"),
            );
            ctx.push_op(body_block, yr.op_ref());

            let ret = arena_func::r#return(ctx, location, [yr.result(ctx)]);
            ctx.push_op(body_block, ret.op_ref());

            encountered_shift = true;
            break;
        }

        // Clone operation into the resume function body using IrMapping
        let new_op = ctx.clone_op_into_block(body_block, op, &mut mapping);

        let new_results = ctx.op_results(new_op);
        if !new_results.is_empty() {
            last_result = Some(new_results[0]);
        }
    }

    // Return YieldResult::Done with the final result
    if !encountered_shift {
        let final_value = last_result.unwrap_or(resume_value);
        // Cast to anyref
        let final_anyref =
            arena_core::unrealized_conversion_cast(ctx, location, final_value, types.anyref);
        ctx.push_op(body_block, final_anyref.op_ref());

        let done_op = arena_adt::variant_new(
            ctx,
            location,
            [final_anyref.result(ctx)],
            types.yield_result,
            types.yield_result,
            Symbol::new("Done"),
        );
        ctx.push_op(body_block, done_op.op_ref());

        let ret = arena_func::r#return(ctx, location, [done_op.result(ctx)]);
        ctx.push_op(body_block, ret.op_ref());
    }

    let body_region = ctx.create_region(trunk_ir::context::RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![body_block],
        parent_op: None,
    });

    let func_data = trunk_ir::context::OperationDataBuilder::new(
        location,
        Symbol::new("func"),
        Symbol::new("func"),
    )
    .attr("sym_name", Attribute::Symbol(name))
    .attr("type", Attribute::Type(func_ty_ref))
    .region(body_region)
    .build(ctx);

    ctx.create_op(func_data)
}
