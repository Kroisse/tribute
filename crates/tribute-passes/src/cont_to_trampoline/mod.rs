//! Lower cont dialect operations to trampoline dialect.
//!
//! This pass transforms continuation operations to trampoline operations:
//! - `cont.shift` → build_state + build_continuation + set_yield_state + step_shift
//! - `cont.resume` → reset_yield_state + continuation_get + build_resume_wrapper + call
//! - `cont.push_prompt` → trampoline yield check loop + dispatch
//! - `cont.handler_dispatch` → yield check + multi-arm dispatch
//! - `cont.done` / `cont.suspend` → consumed by handler_dispatch lowering
//! - `cont.drop` → pass through (handled later)
//!
//! This pass is backend-agnostic and should run after `tribute_to_cont`/`handler_lower`
//! and before `trampoline_to_adt`.

pub(crate) mod analysis;
pub(crate) mod handler_dispatch;
pub(crate) mod patterns;
pub(crate) mod shift_lower;
pub(crate) mod truncate;
pub(crate) mod wrap_returns;

#[cfg(test)]
mod tests;

pub(crate) use analysis::*;
pub(crate) use handler_dispatch::*;
pub(crate) use patterns::*;
pub(crate) use shift_lower::*;
pub(crate) use truncate::*;
pub(crate) use wrap_returns::*;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::ArenaModule;
use trunk_ir::arena::rewrite::{
    ArenaConversionTarget, ArenaTypeConverter, PatternApplicator as ArenaPatternApplicator,
};
use trunk_ir::location::Span;

// Re-export shared utilities from cont_util so existing internal callers still work.
pub(crate) use crate::cont_util::{compute_op_idx, get_region_result_value_arena};

// ============================================================================
// Public API
// ============================================================================

/// Create an ArenaTypeConverter with standard tribute_rt -> core type conversions.
fn standard_type_converter(ctx: &mut IrContext) -> ArenaTypeConverter {
    use trunk_ir::arena::types::TypeDataBuilder;

    let int_dialect = Symbol::new("tribute_rt");
    let core_dialect = Symbol::new("core");

    // Pre-intern target types
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(core_dialect, Symbol::new("i32")).build());
    let i1_ty = ctx
        .types
        .intern(TypeDataBuilder::new(core_dialect, Symbol::new("i1")).build());
    let f64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(core_dialect, Symbol::new("f64")).build());

    let mut converter = ArenaTypeConverter::new();
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
// Shared Types
// ============================================================================

/// Metadata for generating resume functions with continuation code.
pub(crate) struct ResumeFuncSpec {
    /// Name of the resume function
    pub(crate) name: String,
    /// State struct type (used to extract captured values)
    pub(crate) state_type: TypeRef,
    /// Fields in the state struct (field_name, field_type) — field types are anyref
    pub(crate) state_fields: Vec<(Symbol, TypeRef)>,
    /// Original (pre-anyref) field types for casting extracted values back
    pub(crate) original_field_types: Vec<TypeRef>,
    /// Original values that were captured (for value remapping)
    pub(crate) original_live_values: Vec<ValueRef>,
    /// The original shift result value (maps to resume_value)
    pub(crate) shift_result_value: Option<ValueRef>,
    /// The type of the shift result (used to cast resume_value)
    pub(crate) shift_result_type: Option<TypeRef>,
    /// Operations that form the continuation (ops after shift)
    pub(crate) continuation_ops: Vec<OpRef>,
    /// Name of next resume function (if not last)
    pub(crate) next_resume_name: Option<String>,
    /// Location for generating code
    pub(crate) location: trunk_ir::arena::types::Location,
    /// Shift analysis for handling nested shifts in continuation
    pub(crate) shift_analysis: ShiftAnalysis,
    /// Module name for generating unique state type names (for dynamic shifts)
    pub(crate) module_name: Symbol,
}

/// Shared storage for resume function specs during pattern matching.
pub(crate) type ResumeSpecs = Rc<RefCell<Vec<ResumeFuncSpec>>>;

/// Shared counter for generating unique resume function names.
pub(crate) type ResumeCounter = Rc<RefCell<u32>>;

/// Analysis results for shift points, keyed by shift operation's span.
/// Using Span as key because Operation identity may not be stable across phases.
pub(crate) type ShiftAnalysis = Rc<HashMap<Span, ShiftPointInfo>>;

/// Information about a shift point for code generation.
#[derive(Clone)]
pub(crate) struct ShiftPointInfo {
    /// Index of this shift point in the function (0, 1, 2, ...)
    pub(crate) index: usize,
    /// Total number of shift points in the function
    pub(crate) total_shifts: usize,
    /// Live values at this shift point (defined before, used after) with their types
    pub(crate) live_values: Vec<(ValueRef, TypeRef)>,
    /// The result value of the shift operation (maps to resume_value)
    pub(crate) shift_result_value: Option<ValueRef>,
    /// The type of the shift result (for casting resume_value)
    pub(crate) shift_result_type: Option<TypeRef>,
    /// Operations after this shift until next shift or function end
    pub(crate) continuation_ops: Vec<OpRef>,
}

// ============================================================================
// Main Pass
// ============================================================================

/// Lower cont dialect operations to trampoline dialect.
///
/// This pass transforms:
/// - `cont.shift` → state capture + continuation build + `trampoline.step_shift`
/// - `cont.resume` → continuation extraction + resume wrapper call
/// - `cont.push_prompt` → yield check + dispatch
/// - `cont.handler_dispatch` → yield check + multi-arm dispatch
/// - `cont.done` / `cont.suspend` → consumed by handler_dispatch lowering
pub fn lower_cont_to_trampoline(
    ctx: &mut IrContext,
    module: ArenaModule,
) -> Result<(), Vec<trunk_ir::arena::rewrite::conversion_target::IllegalOp>> {
    let module_body = module.body(ctx).expect("module should have a body");
    let module_name = module.name(ctx).unwrap_or_else(|| Symbol::new(""));

    // Shared state for resume function generation (no global state!)
    let resume_specs: ResumeSpecs = Rc::new(RefCell::new(Vec::new()));
    let resume_counter: ResumeCounter = Rc::new(RefCell::new(0));

    // Step 1: Identify effectful functions
    let effectful_funcs = identify_effectful_functions(ctx, module_body);

    // Step 2: Analyze shift points in effectful functions
    let shift_analysis = analyze_shift_points(ctx, module_body, &effectful_funcs);

    // Step 2.5: Collect handler_dispatch ops that are inside effectful functions
    let handlers_in_effectful_funcs =
        collect_handlers_in_effectful_funcs(ctx, module_body, &effectful_funcs);

    // Step 3: Lower cont.* operations to trampoline.*
    let type_converter = standard_type_converter(ctx);
    let applicator = ArenaPatternApplicator::new(type_converter)
        .add_pattern(LowerShiftPattern {
            resume_specs: Rc::clone(&resume_specs),
            resume_counter: Rc::clone(&resume_counter),
            shift_analysis: Rc::clone(&shift_analysis),
            module_name,
        })
        .add_pattern(LowerResumePattern)
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

    applicator.apply_partial(ctx, module);

    // Step 3.5: Truncate effectful function bodies after step_shift
    truncate_after_shift(ctx, module, &effectful_funcs);

    // Step 4: Wrap returns in effectful functions with step_done
    let type_converter2 = standard_type_converter(ctx);
    let applicator2 = ArenaPatternApplicator::new(type_converter2).add_pattern(
        WrapReturnsInEffectfulFuncsPattern {
            effectful_funcs: Rc::clone(&effectful_funcs),
        },
    );
    applicator2.apply_partial(ctx, module);

    // Verify all cont.* ops (except cont.drop) are converted.
    let mut conversion_target = ArenaConversionTarget::new();
    conversion_target.add_illegal_dialect("cont");
    conversion_target.add_legal_op("cont", "drop");
    conversion_target.add_legal_op("cont", "done");
    conversion_target.add_legal_op("cont", "suspend");

    // Generate resume functions from collected specs
    let specs = resume_specs.borrow();
    if specs.is_empty() {
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
                "resume spec '{}' references next_resume_name '{}' which does not exist \
                 in the collected specs ({:?}). This may indicate a resume counter \
                 synchronization bug.",
                spec.name,
                next_name,
                specs.iter().map(|s| &s.name).collect::<Vec<_>>(),
            );
        }
    }

    let resume_funcs: Vec<OpRef> = specs
        .iter()
        .map(|spec| create_resume_function_with_continuation(ctx, spec))
        .collect();

    // Add resume functions to module body
    let module_block = module
        .first_block(ctx)
        .expect("expected module first_block for inserting resume funcs");
    for func_op in resume_funcs {
        ctx.push_op(module_block, func_op);
    }

    let illegal = conversion_target.verify(ctx, module_body);
    if illegal.is_empty() {
        Ok(())
    } else {
        Err(illegal)
    }
}
