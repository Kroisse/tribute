//! Lower cont dialect operations to yield bubbling (ADT-based).
//!
//! This pass replaces `cont_to_trampoline`
//! with a unified transformation that uses ADT enum/struct types:
//!
//! - `cont.shift` → capture state + build Continuation + ShiftInfo → YieldResult::Shift
//! - `cont.resume` → extract resume_fn from Continuation + call_indirect
//! - `cont.handler_dispatch` → scf.loop with YieldResult dispatch
//! - `cont.push_prompt` → body call + handler dispatch loop
//!
//! Output uses only `func`, `adt`, `scf`, `arith`, `core` dialects —
//! no intermediate `trampoline.*` dialect. Both Cranelift and WASM backends
//! can process the lowered IR through their existing `adt_to_*` passes.

pub(crate) mod analysis;
pub(crate) mod call_lower;
pub(crate) mod handler_dispatch;
pub(crate) mod patterns;
pub(crate) mod shift_lower;
pub(crate) mod truncate;
pub(crate) mod types;
pub(crate) mod wrap_returns;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::location::Span;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::rewrite::{ConversionTarget, PatternApplicator, TypeConverter};

use types::YieldBubblingTypes;

// ============================================================================
// Shared Types
// ============================================================================

/// Metadata for generating resume functions with continuation code.
pub(crate) struct ResumeFuncSpec {
    pub(crate) name: String,
    pub(crate) state_type: TypeRef,
    pub(crate) state_fields: Vec<(Symbol, TypeRef)>,
    pub(crate) original_field_types: Vec<TypeRef>,
    pub(crate) original_live_values: Vec<ValueRef>,
    pub(crate) shift_result_value: Option<ValueRef>,
    pub(crate) shift_result_type: Option<TypeRef>,
    pub(crate) continuation_ops: Vec<OpRef>,
    pub(crate) next_resume_name: Option<String>,
    pub(crate) location: trunk_ir::types::Location,
    pub(crate) shift_analysis: ShiftAnalysis,
    pub(crate) module_name: Symbol,
}

pub(crate) type ResumeSpecs = Rc<RefCell<Vec<ResumeFuncSpec>>>;
pub(crate) type ResumeCounter = Rc<RefCell<u32>>;
pub(crate) type ShiftAnalysis = Rc<HashMap<Span, ShiftPointInfo>>;

#[derive(Clone)]
pub(crate) struct ShiftPointInfo {
    pub(crate) index: usize,
    pub(crate) total_shifts: usize,
    pub(crate) live_values: Vec<(ValueRef, TypeRef)>,
    pub(crate) shift_result_value: Option<ValueRef>,
    pub(crate) shift_result_type: Option<TypeRef>,
    pub(crate) continuation_ops: Vec<OpRef>,
}

// ============================================================================
// Type Converter
// ============================================================================

/// Create a TypeConverter with standard tribute_rt → core type conversions.
fn standard_type_converter(ctx: &mut IrContext) -> TypeConverter {
    use trunk_ir::types::TypeDataBuilder;

    let int_dialect = Symbol::new("tribute_rt");
    let core_dialect = Symbol::new("core");

    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(core_dialect, Symbol::new("i32")).build());
    let i1_ty = ctx
        .types
        .intern(TypeDataBuilder::new(core_dialect, Symbol::new("i1")).build());
    let f64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(core_dialect, Symbol::new("f64")).build());

    let mut converter = TypeConverter::new();
    converter.add_conversion(move |ctx, ty| {
        let data = ctx.types.get(ty);
        if data.dialect == int_dialect && data.name == Symbol::new("Int") {
            Some(i32_ty)
        } else {
            None
        }
    });
    converter.add_conversion(move |ctx, ty| {
        let data = ctx.types.get(ty);
        if data.dialect == int_dialect && data.name == Symbol::new("Nat") {
            Some(i32_ty)
        } else {
            None
        }
    });
    converter.add_conversion(move |ctx, ty| {
        let data = ctx.types.get(ty);
        if data.dialect == int_dialect && data.name == Symbol::new("Bool") {
            Some(i1_ty)
        } else {
            None
        }
    });
    converter.add_conversion(move |ctx, ty| {
        let data = ctx.types.get(ty);
        if data.dialect == int_dialect && data.name == Symbol::new("Float") {
            Some(f64_ty)
        } else {
            None
        }
    });
    converter
}

// ============================================================================
// Main Pass
// ============================================================================

/// Lower cont dialect operations to yield bubbling (ADT-based).
///
/// This is the unified replacement for `lower_cont_to_trampoline`.
pub fn lower_cont_to_yield_bubbling(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), Vec<trunk_ir::rewrite::conversion_target::IllegalOp>> {
    let module_body = module.body(ctx).expect("module should have a body");
    let module_name = module.name(ctx).unwrap_or_else(|| Symbol::new(""));

    // Create yield bubbling types (YieldResult, ShiftInfo, Continuation, ResumeWrapper)
    let types = YieldBubblingTypes::new(ctx);

    // Shared state for resume function generation
    let resume_specs: ResumeSpecs = Rc::new(RefCell::new(Vec::new()));
    let resume_counter: ResumeCounter = Rc::new(RefCell::new(0));

    // Step 1: Identify effectful functions
    let effectful_funcs = analysis::identify_effectful_functions(ctx, module_body);

    // Step 2: Analyze shift points in effectful functions
    let shift_analysis = analysis::analyze_shift_points(ctx, module_body, &effectful_funcs);

    // Step 2.5: Collect handler_dispatch ops that are inside effectful functions
    let handlers_in_effectful_funcs =
        analysis::collect_handlers_in_effectful_funcs(ctx, module_body, &effectful_funcs);

    // Step 3: Lower cont.* operations
    let type_converter = standard_type_converter(ctx);

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(shift_lower::LowerShiftPattern {
            types,
            resume_specs: Rc::clone(&resume_specs),
            resume_counter: Rc::clone(&resume_counter),
            shift_analysis: Rc::clone(&shift_analysis),
            module_name,
        })
        .add_pattern(patterns::LowerResumePattern { types })
        .add_pattern(patterns::UpdateEffectfulCallResultTypePattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
            types,
        })
        .add_pattern(patterns::UpdateScfIfResultTypePattern { types })
        .add_pattern(patterns::LowerPushPromptPattern { types })
        .add_pattern(handler_dispatch::LowerHandlerDispatchPattern {
            types,
            effectful_funcs: Rc::clone(&effectful_funcs),
            handlers_in_effectful_funcs: Rc::new(handlers_in_effectful_funcs),
        });

    applicator.apply_partial(ctx, module);

    // Step 3.5: Truncate effectful function bodies after first effect point.
    // Must run BEFORE lower_effectful_calls so that dead code from shift lowering
    // (sequential effectful calls in shift-proxy functions) is removed first.
    truncate::truncate_after_shift(ctx, module, &effectful_funcs, &types);

    // Step 3.55: Fix push_prompt body call_indirect types.
    // After truncation, effectful body thunks return YieldResult, but the
    // call_indirect in the (already-inlined) push_prompt body still has the
    // original type, wrapped in Done → double-wrapping. Fix by changing
    // call_indirect result type to YieldResult and removing the Done wrap.
    truncate::fix_body_call_types(ctx, module, &effectful_funcs, &types);

    // Step 3.6: Expand effectful calls into Done/Shift branches with chaining (#336)
    // Runs after truncate so only meaningful remaining ops are expanded.
    let chain_specs: call_lower::ChainSpecs = Rc::new(RefCell::new(Vec::new()));
    let chain_counter: ResumeCounter = Rc::new(RefCell::new(1000)); // offset to avoid name collisions
    call_lower::lower_effectful_calls(
        ctx,
        module,
        &effectful_funcs,
        &types,
        &chain_specs,
        &chain_counter,
        module_name,
    );

    // Step 4: Wrap returns in effectful functions with YieldResult::Done
    let type_converter2 = standard_type_converter(ctx);
    let applicator2 =
        PatternApplicator::new(type_converter2).add_pattern(wrap_returns::WrapReturnsPattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
            types,
        });
    applicator2.apply_partial(ctx, module);

    // Step 5: Unwrap YieldResult in non-effectful functions that call effectful ones.
    // Since these functions handle all effects locally, the result is always Done.
    // We extract the Done value and cast to the original type.
    truncate::unwrap_yr_in_non_effectful_funcs(ctx, module, &effectful_funcs, &types);

    // Verify all cont.* ops (except cont.drop, cont.done, cont.suspend, cont.yield) are converted
    let mut conversion_target = ConversionTarget::new();
    conversion_target.add_illegal_dialect("cont");
    conversion_target.add_legal_op("cont", "drop");
    conversion_target.add_legal_op("cont", "done");
    conversion_target.add_legal_op("cont", "suspend");
    conversion_target.add_legal_op("cont", "yield");

    // Generate resume/chain functions and post-process them iteratively.
    // Newly generated functions may contain effectful calls that need
    // lower_effectful_calls + wrap_returns, which may produce more chain specs.
    let module_block = module
        .first_block(ctx)
        .expect("expected module first_block for inserting resume funcs");

    // Check if there are any specs to process
    if resume_specs.borrow().is_empty() && chain_specs.borrow().is_empty() {
        let illegal = conversion_target.verify(ctx, module_body);
        if illegal.is_empty() {
            return Ok(());
        }
        return Err(illegal);
    }

    // Validate next_resume_name references before draining
    {
        let specs = resume_specs.borrow();
        for spec in specs.iter() {
            if let Some(ref next_name) = spec.next_resume_name {
                debug_assert!(
                    specs.iter().any(|s| s.name == *next_name),
                    "resume spec '{}' references next_resume_name '{}' which does not exist",
                    spec.name,
                    next_name,
                );
            }
        }
    }

    // Iterative generation + post-processing loop.
    // Each iteration drains current specs, generates functions, then
    // runs lower_effectful_calls + wrap_returns on them. Those passes
    // may produce new chain specs, so we loop until no new specs appear.
    let mut effectful_set = (*effectful_funcs).clone();

    loop {
        let cur_resume: Vec<_> = resume_specs.borrow_mut().drain(..).collect();
        let cur_chain: Vec<_> = chain_specs.borrow_mut().drain(..).collect();

        if cur_resume.is_empty() && cur_chain.is_empty() {
            break;
        }

        // 1) Generate functions
        let resume_ops: Vec<OpRef> = cur_resume
            .iter()
            .map(|s| shift_lower::create_resume_function(ctx, s, &types))
            .collect();
        let chain_ops: Vec<OpRef> = cur_chain
            .iter()
            .map(|s| call_lower::create_chain_function(ctx, s, &types))
            .collect();

        // 2) Collect names, add to effectful set
        let new_names: Vec<Symbol> = cur_resume
            .iter()
            .map(|s| Symbol::from_dynamic(&s.name))
            .chain(cur_chain.iter().map(|s| Symbol::from_dynamic(&s.name)))
            .collect();
        for name in &new_names {
            effectful_set.insert(*name);
        }

        // 3) Add to module
        for op in resume_ops {
            ctx.push_op(module_block, op);
        }
        for op in chain_ops {
            ctx.push_op(module_block, op);
        }

        // 4) Post-process new functions:
        //    a) Update effectful call result types to YieldResult.
        //       Reuse UpdateEffectfulCallResultTypePattern via PatternApplicator.
        //       The pattern is idempotent — already-processed calls are skipped.
        let ef_for_update = Rc::new(effectful_set.clone());
        let update_applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(
            patterns::UpdateEffectfulCallResultTypePattern {
                effectful_funcs: ef_for_update,
                types,
            },
        );
        update_applicator.apply_partial(ctx, module);
        //    b) Expand effectful calls into Done/Shift branches
        let ef_rc = Rc::new(effectful_set.clone());
        let lc = call_lower::CallLowerCtx {
            effectful_funcs: &ef_rc,
            types: &types,
            chain_specs: &chain_specs,
            chain_counter: &chain_counter,
            module_name,
            module_body,
        };
        call_lower::lower_effectful_calls_for_funcs(ctx, module, &lc, &new_names);
        wrap_returns::wrap_returns_for_funcs(ctx, module, &new_names, &types);
    }

    let illegal = conversion_target.verify(ctx, module_body);
    if illegal.is_empty() {
        Ok(())
    } else {
        Err(illegal)
    }
}
