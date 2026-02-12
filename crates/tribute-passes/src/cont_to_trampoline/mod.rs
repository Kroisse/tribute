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

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::rewrite::{ConversionError, ConversionTarget, PatternApplicator, TypeConverter};
use trunk_ir::{Block, DialectType, IdVec, Location, Operation, Region, Span, Symbol, Type, Value};

// Re-export shared utilities from cont_util so existing internal callers still work.
pub(crate) use crate::cont_util::{collect_suspend_arms, compute_op_idx, get_region_result_value};

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

// ============================================================================
// Shared Types
// ============================================================================

/// Metadata for generating resume functions with continuation code.
pub(crate) struct ResumeFuncSpec<'db> {
    /// Name of the resume function
    pub(crate) name: String,
    /// State struct type (used to extract captured values)
    pub(crate) state_type: Type<'db>,
    /// Fields in the state struct (field_name, field_type) — field types are anyref
    /// to match the WASM lowering (LowerBuildStatePattern converts all fields to anyref).
    pub(crate) state_fields: Vec<(Symbol, Type<'db>)>,
    /// Original (pre-anyref) field types for casting extracted values back
    pub(crate) original_field_types: Vec<Type<'db>>,
    /// Original values that were captured (for value remapping)
    pub(crate) original_live_values: Vec<Value<'db>>,
    /// The original shift result value (maps to resume_value)
    pub(crate) shift_result_value: Option<Value<'db>>,
    /// The type of the shift result (used to cast resume_value)
    pub(crate) shift_result_type: Option<Type<'db>>,
    /// Operations that form the continuation (ops after shift)
    pub(crate) continuation_ops: Vec<Operation<'db>>,
    /// Name of next resume function (if not last)
    pub(crate) next_resume_name: Option<String>,
    /// Location for generating code
    pub(crate) location: Location<'db>,
    /// Shift analysis for handling nested shifts in continuation
    pub(crate) shift_analysis: ShiftAnalysis<'db>,
    /// Module name for generating unique state type names (for dynamic shifts)
    pub(crate) module_name: Symbol,
}

/// Shared storage for resume function specs during pattern matching.
pub(crate) type ResumeSpecs<'db> = Rc<RefCell<Vec<ResumeFuncSpec<'db>>>>;

/// Shared counter for generating unique resume function names.
pub(crate) type ResumeCounter = Rc<RefCell<u32>>;

/// Analysis results for shift points, keyed by shift operation's span.
/// Using Span as key because Operation identity may not be stable across phases.
pub(crate) type ShiftAnalysis<'db> = Rc<HashMap<Span, ShiftPointInfo<'db>>>;

/// Information about a shift point for code generation.
#[derive(Clone)]
pub(crate) struct ShiftPointInfo<'db> {
    /// Index of this shift point in the function (0, 1, 2, ...)
    pub(crate) index: usize,
    /// Total number of shift points in the function
    pub(crate) total_shifts: usize,
    /// Live values at this shift point (defined before, used after) with their types
    pub(crate) live_values: Vec<(Value<'db>, Type<'db>)>,
    /// The result value of the shift operation (maps to resume_value)
    pub(crate) shift_result_value: Option<Value<'db>>,
    /// The type of the shift result (for casting resume_value)
    pub(crate) shift_result_type: Option<Type<'db>>,
    /// Operations after this shift until next shift or function end
    pub(crate) continuation_ops: Vec<Operation<'db>>,
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
/// - `cont.get_continuation` → `trampoline.get_yield_continuation`
/// - `cont.get_shift_value` → `trampoline.get_yield_shift_value`
/// - `cont.get_done_value` → `trampoline.step_get(field="value")`
///
/// Returns an error if any `cont.*` operations (except `cont.drop`) remain after conversion.
pub fn lower_cont_to_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Result<Module<'db>, ConversionError> {
    // Shared state for resume function generation (no global state!)
    let resume_specs: ResumeSpecs<'db> = Rc::new(RefCell::new(Vec::new()));
    let resume_counter: ResumeCounter = Rc::new(RefCell::new(0));

    // Step 1: Identify effectful functions
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
            module_name: module.name(db),
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

    // Validate next_resume_name references: every referenced name must exist
    // in the collected specs. A mismatch indicates a resume counter
    // synchronization bug (see shift_lower.rs next_resume_name invariant).
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

// Helpers: see crate::cont_util for shared utilities (compute_op_idx, get_region_result_value, etc.)
