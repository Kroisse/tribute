//! Lower cont dialect operations to yield bubbling (ADT-based).
//!
//! This pass replaces both `cont_to_trampoline` and `cont_to_libmprompt`
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
/// This is the unified replacement for both `lower_cont_to_trampoline`
/// and `lower_cont_to_libmprompt`.
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

    // Create second set of types for patterns that need owned copies
    let types_for_shift = YieldBubblingTypes::new(ctx);
    let types_for_resume = YieldBubblingTypes::new(ctx);
    let types_for_call = YieldBubblingTypes::new(ctx);
    let types_for_if = YieldBubblingTypes::new(ctx);
    let types_for_yield = YieldBubblingTypes::new(ctx);
    let types_for_dispatch = YieldBubblingTypes::new(ctx);
    let types_for_push = YieldBubblingTypes::new(ctx);

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(shift_lower::LowerShiftPattern {
            types: types_for_shift,
            resume_specs: Rc::clone(&resume_specs),
            resume_counter: Rc::clone(&resume_counter),
            shift_analysis: Rc::clone(&shift_analysis),
            module_name,
        })
        .add_pattern(patterns::LowerResumePattern {
            types: types_for_resume,
        })
        .add_pattern(patterns::UpdateEffectfulCallResultTypePattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
            types: types_for_call,
        })
        .add_pattern(patterns::UpdateScfIfResultTypePattern {
            types: types_for_if,
        })
        .add_pattern(patterns::UpdateScfYieldToYieldResultPattern {
            _types: types_for_yield,
        })
        .add_pattern(patterns::LowerPushPromptPattern {
            types: types_for_push,
        })
        .add_pattern(handler_dispatch::LowerHandlerDispatchPattern {
            types: types_for_dispatch,
            effectful_funcs: Rc::clone(&effectful_funcs),
            handlers_in_effectful_funcs: Rc::new(handlers_in_effectful_funcs),
        });

    applicator.apply_partial(ctx, module);

    // Step 3.5: Expand effectful calls into Done/Shift branches with chaining (#336)
    let chain_specs: call_lower::ChainSpecs = Rc::new(RefCell::new(Vec::new()));
    let chain_counter: ResumeCounter = Rc::new(RefCell::new(1000)); // offset to avoid name collisions
    let types_for_chain = YieldBubblingTypes::new(ctx);
    call_lower::lower_effectful_calls(
        ctx,
        module,
        &effectful_funcs,
        &types_for_chain,
        &chain_specs,
        &chain_counter,
        module_name,
    );

    // Step 3.6: Truncate effectful function bodies after first effect point
    truncate::truncate_after_shift(ctx, module, &effectful_funcs, &types);

    // Step 4: Wrap returns in effectful functions with YieldResult::Done
    let type_converter2 = standard_type_converter(ctx);
    let types_for_wrap = YieldBubblingTypes::new(ctx);
    let applicator2 =
        PatternApplicator::new(type_converter2).add_pattern(wrap_returns::WrapReturnsPattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
            types: types_for_wrap,
        });
    applicator2.apply_partial(ctx, module);

    // Verify all cont.* ops (except cont.drop) are converted
    let mut conversion_target = ConversionTarget::new();
    conversion_target.add_illegal_dialect("cont");
    conversion_target.add_legal_op("cont", "drop");
    conversion_target.add_legal_op("cont", "done");
    conversion_target.add_legal_op("cont", "suspend");
    conversion_target.add_legal_op("cont", "yield");

    // Generate resume functions from collected specs
    let specs = resume_specs.borrow();
    let chain_specs_ref = chain_specs.borrow();
    if specs.is_empty() && chain_specs_ref.is_empty() {
        let illegal = conversion_target.verify(ctx, module_body);
        if illegal.is_empty() {
            return Ok(());
        }
        return Err(illegal);
    }

    // Validate next_resume_name references
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

    let resume_funcs: Vec<OpRef> = specs
        .iter()
        .map(|spec| shift_lower::create_resume_function(ctx, spec, &types))
        .collect();

    let chain_funcs: Vec<OpRef> = chain_specs_ref
        .iter()
        .map(|spec| call_lower::create_chain_function(ctx, spec, &types))
        .collect();

    // Add resume and chain functions to module body
    let module_block = module
        .first_block(ctx)
        .expect("expected module first_block for inserting resume funcs");
    for func_op in resume_funcs {
        ctx.push_op(module_block, func_op);
    }
    for func_op in chain_funcs {
        ctx.push_op(module_block, func_op);
    }

    let illegal = conversion_target.verify(ctx, module_body);
    if illegal.is_empty() {
        Ok(())
    } else {
        Err(illegal)
    }
}
