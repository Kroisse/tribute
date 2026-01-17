//! Lower trampoline dialect operations to WASM and ADT operations.
//!
//! This pass converts all trampoline operations using RewritePattern infrastructure:
//!
//! ## Struct operations → ADT operations
//! - `trampoline.build_continuation` → `adt.struct_new` (Continuation type)
//! - `trampoline.step_done` → `adt.struct_new` (Step type with tag=0)
//! - `trampoline.step_shift` → `adt.struct_new` (Step type with tag=1)
//! - `trampoline.step_get` → `adt.struct_get`
//! - `trampoline.continuation_get` → `adt.struct_get`
//! - `trampoline.build_state` → `adt.struct_new` (custom state type)
//! - `trampoline.build_resume_wrapper` → `adt.struct_new` (ResumeWrapper type)
//! - `trampoline.resume_wrapper_get` → `adt.struct_get`
//! - `trampoline.state_get` → `adt.struct_get`
//!
//! ## Global state operations → WASM operations
//! - `trampoline.set_yield_state` → wasm.global_set (multiple)
//! - `trampoline.reset_yield_state` → wasm.global_set
//! - `trampoline.get_yield_continuation` → wasm.global_get + wasm.ref_cast
//! - `trampoline.get_yield_shift_value` → wasm.global_get + adt.struct_get
//! - `trampoline.check_yield` → wasm.global_get
//!
//! This pass uses TypeConverter to consistently convert trampoline types to ADT types.

use tribute_ir::dialect::adt;
use tribute_ir::dialect::trampoline::{self, STEP_TAG_DONE, STEP_TAG_SHIFT};
use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func::{self, Func};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{
    ConversionTarget, MaterializeResult, OpAdaptor, PatternApplicator, RewritePattern,
    RewriteResult, TypeConverter,
};
use trunk_ir::{
    Attribute, Block, BlockArg, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol,
    Type, Value,
};

use crate::constants::yield_globals;

/// Lower all trampoline operations to WASM/ADT using RewritePattern infrastructure.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let type_converter = create_type_converter();

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(ConvertFuncTypePattern)
        .add_pattern(ConvertCallTypePattern)
        .add_pattern(ConvertCallIndirectTypePattern)
        .add_pattern(ConvertWasmIfTypePattern)
        .add_pattern(LowerBuildContinuationPattern)
        .add_pattern(LowerStepDonePattern)
        .add_pattern(LowerStepShiftPattern)
        .add_pattern(LowerStepGetPattern)
        .add_pattern(LowerContinuationGetPattern)
        .add_pattern(LowerBuildStatePattern)
        .add_pattern(LowerBuildResumeWrapperPattern)
        .add_pattern(LowerResumeWrapperGetPattern)
        .add_pattern(LowerStateGetPattern)
        .add_pattern(LowerSetYieldStatePattern)
        .add_pattern(LowerResetYieldStatePattern)
        .add_pattern(LowerGetYieldContinuationPattern)
        .add_pattern(LowerGetYieldShiftValuePattern)
        .add_pattern(LowerCheckYieldPattern)
        .add_pattern(LowerGetYieldOpIdxPattern);

    // No specific conversion target - trampoline lowering is a dialect transformation
    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
}

// ============================================================================
// Type Definitions (Single Source of Truth)
// ============================================================================

/// Get the canonical Step ADT type.
/// Layout: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
///
/// IMPORTANT: Must use wasm::Anyref to match step_marker_type in gc_types.rs.
/// Using tribute_rt::Any would create a different type identity, causing
/// type_idx_by_type lookup failures.
fn step_adt_type(db: &dyn salsa::Database) -> Type<'_> {
    let i32_ty = core::I32::new(db).as_type();
    let anyref_ty = wasm::Anyref::new(db).as_type();

    adt::struct_type(
        db,
        "_Step",
        vec![
            (Symbol::new("tag"), i32_ty),
            (Symbol::new("value"), anyref_ty),
            (Symbol::new("prompt"), i32_ty),
            (Symbol::new("op_idx"), i32_ty),
        ],
    )
}

/// Get the canonical Continuation ADT type.
/// Layout: (resume_fn: funcref, state: anyref, tag: i32, shift_value: anyref)
///
/// Uses wasm::Anyref for consistency with step_adt_type.
fn continuation_adt_type(db: &dyn salsa::Database) -> Type<'_> {
    let i32_ty = core::I32::new(db).as_type();
    let funcref_ty = wasm::Funcref::new(db).as_type();
    let anyref_ty = wasm::Anyref::new(db).as_type();

    adt::struct_type(
        db,
        "_Continuation",
        vec![
            (Symbol::new("resume_fn"), funcref_ty),
            (Symbol::new("state"), anyref_ty),
            (Symbol::new("tag"), i32_ty),
            (Symbol::new("shift_value"), anyref_ty),
        ],
    )
}

/// Get the canonical ResumeWrapper ADT type.
/// Layout: (state: anyref, resume_value: anyref)
///
/// Uses wasm::Anyref for consistency with step_adt_type.
fn resume_wrapper_adt_type(db: &dyn salsa::Database) -> Type<'_> {
    let anyref_ty = wasm::Anyref::new(db).as_type();

    adt::struct_type(
        db,
        "_ResumeWrapper",
        vec![
            (Symbol::new("state"), anyref_ty),
            (Symbol::new("resume_value"), anyref_ty),
        ],
    )
}

// ============================================================================
// Type Converter
// ============================================================================

/// Create a TypeConverter that converts trampoline types to ADT types.
fn create_type_converter() -> TypeConverter {
    TypeConverter::new()
        // Convert trampoline.step → _Step ADT
        .add_conversion(|db, ty| trampoline::Step::from_type(db, ty).map(|_| step_adt_type(db)))
        // Convert trampoline.continuation → _Continuation ADT
        .add_conversion(|db, ty| {
            trampoline::Continuation::from_type(db, ty).map(|_| continuation_adt_type(db))
        })
        // Convert trampoline.resume_wrapper → _ResumeWrapper ADT
        .add_conversion(|db, ty| {
            trampoline::ResumeWrapper::from_type(db, ty).map(|_| resume_wrapper_adt_type(db))
        })
        // Materialize: insert ref_cast when converting anyref ↔ concrete types
        .add_materialization(|db, location, value, from_ty, to_ty| {
            if from_ty == to_ty {
                return MaterializeResult::Skip;
            }
            // Insert ref_cast for reference type conversions
            let cast_op = adt::ref_cast(db, location, value, to_ty, to_ty);
            MaterializeResult::single(cast_op.as_operation())
        })
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_i32_const<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    value: i32,
) -> Operation<'db> {
    let i32_ty = core::I32::new(db).as_type();
    wasm::i32_const(db, location, i32_ty, value).as_operation()
}

/// Step fields: tag=0, value=1, prompt=2, op_idx=3
fn step_field_index(field_name: Symbol) -> u32 {
    field_name.with_str(|s| match s {
        "tag" => 0,
        "value" => 1,
        "prompt" => 2,
        "op_idx" => 3,
        _ => 1, // default to value
    })
}

/// Continuation fields: resume_fn=0, state=1, tag=2, shift_value=3
fn continuation_field_index(field_name: Symbol) -> u32 {
    field_name.with_str(|s| match s {
        "resume_fn" => 0,
        "state" => 1,
        "tag" => 2,
        "shift_value" => 3,
        _ => 0,
    })
}

/// ResumeWrapper fields: state=0, resume_value=1
fn resume_wrapper_field_index(field_name: Symbol) -> u32 {
    field_name.with_str(|s| match s {
        "state" => 0,
        "resume_value" => 1,
        _ => 0,
    })
}

/// Cast a value to `wasm::Anyref` type if needed.
fn cast_to_any<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    value: Value<'db>,
    ops: &mut Vec<Operation<'db>>,
) -> Value<'db> {
    // In WASM GC, anyref is a supertype of all reference types.
    // Upcasting to anyref is implicit at runtime, but we need to express
    // this in the IR for proper type tracking.
    // We use wasm.ref_cast with anyref as target, which will be lowered
    // to a no-op in emit (since upcasting to anyref is always valid).
    let anyref_ty = wasm::Anyref::new(db).as_type();
    let cast_op = wasm::ref_cast(db, location, value, anyref_ty, anyref_ty, None);
    let cast_val = cast_op.as_operation().result(db, 0);
    ops.push(cast_op.as_operation());
    cast_val
}

/// Create a null reference of `wasm::Anyref` type.
fn null_any<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    ops: &mut Vec<Operation<'db>>,
) -> Value<'db> {
    let anyref_ty = wasm::Anyref::new(db).as_type();
    let null_op = adt::ref_null(db, location, anyref_ty, anyref_ty);
    let null_val = null_op.as_operation().result(db, 0);
    ops.push(null_op.as_operation());
    null_val
}

/// Create a null reference of `wasm::Funcref` type.
fn null_funcref<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    ops: &mut Vec<Operation<'db>>,
) -> Value<'db> {
    let funcref_ty = wasm::Funcref::new(db).as_type();
    let null_op = adt::ref_null(db, location, funcref_ty, funcref_ty);
    let null_val = null_op.as_operation().result(db, 0);
    ops.push(null_op.as_operation());
    null_val
}

// ============================================================================
// Patterns: Struct Operations → ADT
// ============================================================================

/// Lower `trampoline.build_continuation` → `adt.struct_new`
struct LowerBuildContinuationPattern;

impl<'db> RewritePattern<'db> for LowerBuildContinuationPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let build_cont = match trampoline::BuildContinuation::from_operation(db, *op) {
            Ok(bc) => bc,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let cont_type = continuation_adt_type(db);

        let tag = build_cont.tag(db);

        let operands = adaptor.operands();
        let resume_fn = operands.first().copied();
        let state = operands.get(1).copied();
        let shift_value = operands.get(2).copied();

        let mut ops = Vec::new();

        // Create tag constant
        let tag_const = create_i32_const(db, location, tag as i32);
        let tag_value = tag_const.result(db, 0);
        ops.push(tag_const);

        // resume_fn field (funcref, no casting needed)
        let resume_fn_field = if let Some(v) = resume_fn {
            v
        } else {
            null_funcref(db, location, &mut ops)
        };

        // state field - cast to any
        let state_field = if let Some(v) = state {
            cast_to_any(db, location, v, &mut ops)
        } else {
            null_any(db, location, &mut ops)
        };

        // shift_value field - cast to any
        let shift_value_field = if let Some(v) = shift_value {
            cast_to_any(db, location, v, &mut ops)
        } else {
            null_any(db, location, &mut ops)
        };

        let fields = vec![resume_fn_field, state_field, tag_value, shift_value_field];

        let struct_new = adt::struct_new(db, location, fields, cont_type, cont_type);
        ops.push(struct_new.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.step_done` → `adt.struct_new` with tag=0
struct LowerStepDonePattern;

impl<'db> RewritePattern<'db> for LowerStepDonePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME() || op.name(db) != trampoline::STEP_DONE() {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let step_type = step_adt_type(db);

        let value = adaptor.operands().first().copied();

        let mut ops = Vec::new();

        // Create constants for tag=0, prompt=0, op_idx=0
        let tag_const = create_i32_const(db, location, STEP_TAG_DONE);
        let prompt_const = create_i32_const(db, location, 0);
        let op_idx_const = create_i32_const(db, location, 0);

        let tag_value = tag_const.result(db, 0);
        let prompt_value = prompt_const.result(db, 0);
        let op_idx_value = op_idx_const.result(db, 0);

        ops.push(tag_const);
        ops.push(prompt_const);
        ops.push(op_idx_const);

        // Value field - cast to any
        let value_field = if let Some(v) = value {
            cast_to_any(db, location, v, &mut ops)
        } else {
            null_any(db, location, &mut ops)
        };

        let fields = vec![tag_value, value_field, prompt_value, op_idx_value];
        let struct_new = adt::struct_new(db, location, fields, step_type, step_type);
        ops.push(struct_new.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.step_shift` → `adt.struct_new` with tag=1
struct LowerStepShiftPattern;

impl<'db> RewritePattern<'db> for LowerStepShiftPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let step_shift = match trampoline::StepShift::from_operation(db, *op) {
            Ok(ss) => ss,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let step_type = step_adt_type(db);

        let prompt = step_shift.prompt(db);
        let op_idx = step_shift.op_idx(db);

        let continuation = adaptor.operands().first().copied();

        let mut ops = Vec::new();

        // Create constants
        let tag_const = create_i32_const(db, location, STEP_TAG_SHIFT);
        let prompt_const = create_i32_const(db, location, prompt as i32);
        let op_idx_const = create_i32_const(db, location, op_idx as i32);

        let tag_value = tag_const.result(db, 0);
        let prompt_value = prompt_const.result(db, 0);
        let op_idx_value = op_idx_const.result(db, 0);

        ops.push(tag_const);
        ops.push(prompt_const);
        ops.push(op_idx_const);

        // Value field - cast continuation to any
        let value_field = if let Some(v) = continuation {
            cast_to_any(db, location, v, &mut ops)
        } else {
            null_any(db, location, &mut ops)
        };

        let fields = vec![tag_value, value_field, prompt_value, op_idx_value];
        let struct_new = adt::struct_new(db, location, fields, step_type, step_type);
        ops.push(struct_new.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.step_get` → `adt.struct_get`
struct LowerStepGetPattern;

impl<'db> RewritePattern<'db> for LowerStepGetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let step_get = match trampoline::StepGet::from_operation(db, *op) {
            Ok(sg) => sg,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let step_type = step_adt_type(db);
        let any_ty = wasm::Anyref::new(db).as_type();

        let field_name = step_get.field(db);
        let field_idx = step_field_index(field_name);
        let expected_result_type = op.results(db).first().copied().unwrap_or(any_ty);

        let step_value = adaptor
            .operands()
            .first()
            .copied()
            .expect("step_get requires operand");

        // field 1 (value) is any type and needs casting or unboxing
        let is_any_field = field_idx == 1;

        if is_any_field {
            let mut ops = Vec::new();

            let struct_get = adt::struct_get(
                db,
                location,
                step_value,
                any_ty,
                step_type,
                Attribute::IntBits(field_idx.into()),
            );
            let any_value = struct_get.as_operation().result(db, 0);
            ops.push(struct_get.as_operation());

            // Check if expected_result_type is a primitive type that needs unboxing
            // After normalize_primitive_types pass, types are already core.i32/f64
            let is_int_primitive = core::I32::from_type(db, expected_result_type).is_some();
            let is_float_primitive = core::F64::from_type(db, expected_result_type).is_some();

            if is_int_primitive {
                // For int primitives: anyref -> i31ref -> i32 (via unbox_int)
                let i31ref_ty = wasm::I31ref::new(db).as_type();
                let i32_ty = core::I32::new(db).as_type();

                // Cast anyref to i31ref
                let ref_cast_op =
                    wasm::ref_cast(db, location, any_value, i31ref_ty, i31ref_ty, None);
                let i31ref_value = ref_cast_op.as_operation().result(db, 0);
                ops.push(ref_cast_op.as_operation());

                // Unbox i31ref to i32
                let unbox_op = tribute_rt::unbox_int(db, location, i31ref_value, i32_ty);
                ops.push(unbox_op.as_operation());
            } else if is_float_primitive {
                // For float primitives: anyref -> boxed_f64 struct -> f64
                // Currently just ref_cast (may need struct_get for BoxedF64 struct)
                let cast_op = adt::ref_cast(
                    db,
                    location,
                    any_value,
                    expected_result_type,
                    expected_result_type,
                );
                ops.push(cast_op.as_operation());
            } else {
                // For reference types: just ref_cast
                let cast_op = adt::ref_cast(
                    db,
                    location,
                    any_value,
                    expected_result_type,
                    expected_result_type,
                );
                ops.push(cast_op.as_operation());
            }

            RewriteResult::expand(ops)
        } else {
            let struct_get = adt::struct_get(
                db,
                location,
                step_value,
                expected_result_type,
                step_type,
                Attribute::IntBits(field_idx.into()),
            );

            RewriteResult::Replace(struct_get.as_operation())
        }
    }
}

/// Lower `trampoline.continuation_get` → `adt.struct_get`
struct LowerContinuationGetPattern;

impl<'db> RewritePattern<'db> for LowerContinuationGetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let cont_get = match trampoline::ContinuationGet::from_operation(db, *op) {
            Ok(cg) => cg,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let cont_type = continuation_adt_type(db);
        let any_ty = wasm::Anyref::new(db).as_type();

        let field_name = cont_get.field(db);
        let field_idx = continuation_field_index(field_name);
        let expected_result_type = op.results(db).first().copied().unwrap_or(any_ty);

        let cont_value = adaptor
            .operands()
            .first()
            .copied()
            .expect("continuation_get requires operand");

        // Fields 1 (state) and 3 (shift_value) are any type
        let is_any_field = field_idx == 1 || field_idx == 3;

        if is_any_field {
            let mut ops = Vec::new();

            let struct_get = adt::struct_get(
                db,
                location,
                cont_value,
                any_ty,
                cont_type,
                Attribute::IntBits(field_idx.into()),
            );
            let any_value = struct_get.as_operation().result(db, 0);
            ops.push(struct_get.as_operation());

            let cast_op = adt::ref_cast(
                db,
                location,
                any_value,
                expected_result_type,
                expected_result_type,
            );
            ops.push(cast_op.as_operation());

            RewriteResult::expand(ops)
        } else {
            let struct_get = adt::struct_get(
                db,
                location,
                cont_value,
                expected_result_type,
                cont_type,
                Attribute::IntBits(field_idx.into()),
            );

            RewriteResult::Replace(struct_get.as_operation())
        }
    }
}

/// Lower `trampoline.build_state` → `adt.struct_new`
struct LowerBuildStatePattern;

impl<'db> RewritePattern<'db> for LowerBuildStatePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let build_state = match trampoline::BuildState::from_operation(db, *op) {
            Ok(bs) => bs,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let operands = adaptor.operands();

        // Get state type from attribute or create dynamic one
        let state_type = build_state.state_type(db);

        let mut ops = Vec::new();
        let mut fields = Vec::new();

        // Cast all operands to any type
        for v in operands.iter() {
            let casted = cast_to_any(db, location, *v, &mut ops);
            fields.push(casted);
        }

        let struct_new = adt::struct_new(db, location, fields, state_type, state_type);
        ops.push(struct_new.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.build_resume_wrapper` → `adt.struct_new`
struct LowerBuildResumeWrapperPattern;

impl<'db> RewritePattern<'db> for LowerBuildResumeWrapperPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME()
            || op.name(db) != trampoline::BUILD_RESUME_WRAPPER()
        {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let wrapper_type = resume_wrapper_adt_type(db);

        let operands = adaptor.operands();
        let state = operands.first().copied();
        let resume_value = operands.get(1).copied();

        let mut ops = Vec::new();

        // Cast state to any
        let state_field = if let Some(v) = state {
            cast_to_any(db, location, v, &mut ops)
        } else {
            null_any(db, location, &mut ops)
        };

        // Cast resume_value to any
        let resume_value_field = if let Some(v) = resume_value {
            cast_to_any(db, location, v, &mut ops)
        } else {
            null_any(db, location, &mut ops)
        };

        let fields = vec![state_field, resume_value_field];
        let struct_new = adt::struct_new(db, location, fields, wrapper_type, wrapper_type);
        ops.push(struct_new.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.resume_wrapper_get` → `adt.struct_get`
struct LowerResumeWrapperGetPattern;

impl<'db> RewritePattern<'db> for LowerResumeWrapperGetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let wrapper_get = match trampoline::ResumeWrapperGet::from_operation(db, *op) {
            Ok(wg) => wg,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let wrapper_type = resume_wrapper_adt_type(db);
        let any_ty = wasm::Anyref::new(db).as_type();

        let field_name = wrapper_get.field(db);
        let field_idx = resume_wrapper_field_index(field_name);
        let expected_result_type = op.results(db).first().copied().unwrap_or(any_ty);

        let wrapper_value = adaptor
            .operands()
            .first()
            .copied()
            .expect("resume_wrapper_get requires operand");

        // Both fields are any type
        let mut ops = Vec::new();

        let struct_get = adt::struct_get(
            db,
            location,
            wrapper_value,
            any_ty,
            wrapper_type,
            Attribute::IntBits(field_idx.into()),
        );
        let any_value = struct_get.as_operation().result(db, 0);
        ops.push(struct_get.as_operation());

        let cast_op = adt::ref_cast(
            db,
            location,
            any_value,
            expected_result_type,
            expected_result_type,
        );
        ops.push(cast_op.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.state_get` → `adt.struct_get`
struct LowerStateGetPattern;

impl<'db> RewritePattern<'db> for LowerStateGetPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME() || op.name(db) != trampoline::STATE_GET() {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let any_ty = wasm::Anyref::new(db).as_type();

        // Extract field index from "field" attribute (Symbol type, e.g., "field0", "field1")
        let field_idx =
            op.attributes(db)
                .get(&Symbol::new("field"))
                .and_then(|a| match a {
                    Attribute::Symbol(sym) => sym
                        .with_str(|s| s.strip_prefix("field").and_then(|n| n.parse::<u64>().ok())),
                    _ => None,
                })
                .unwrap_or(0u64);

        let state_type = op
            .attributes(db)
            .get(&Symbol::new("state_type"))
            .and_then(|a| match a {
                Attribute::Type(ty) => Some(*ty),
                _ => None,
            })
            .unwrap_or(any_ty);

        let expected_result_type = op.results(db).first().copied().unwrap_or(any_ty);

        let state_value = adaptor
            .operands()
            .first()
            .copied()
            .expect("state_get requires operand");

        let mut ops = Vec::new();

        let struct_get = adt::struct_get(
            db,
            location,
            state_value,
            any_ty,
            state_type,
            Attribute::IntBits(field_idx),
        );
        let any_value = struct_get.as_operation().result(db, 0);
        ops.push(struct_get.as_operation());

        let cast_op = adt::ref_cast(
            db,
            location,
            any_value,
            expected_result_type,
            expected_result_type,
        );
        ops.push(cast_op.as_operation());

        RewriteResult::expand(ops)
    }
}

// ============================================================================
// Patterns: Global State Operations → WASM
// ============================================================================

/// Lower `trampoline.set_yield_state` → wasm.global_set (multiple)
struct LowerSetYieldStatePattern;

impl<'db> RewritePattern<'db> for LowerSetYieldStatePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let set_yield = match trampoline::SetYieldState::from_operation(db, *op) {
            Ok(sy) => sy,
            Err(_) => return RewriteResult::Unchanged,
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        let tag = set_yield.tag(db);
        let op_idx = set_yield.op_idx(db);

        let cont_val = adaptor
            .operands()
            .first()
            .copied()
            .expect("set_yield_state requires continuation operand");

        let mut ops = Vec::new();

        // Set $yield_state = 1 (yielding)
        let const_1 = wasm::i32_const(db, location, i32_ty, 1);
        let const_1_val = const_1.as_operation().result(db, 0);
        ops.push(const_1.as_operation());
        ops.push(
            wasm::global_set(db, location, const_1_val, yield_globals::STATE_IDX).as_operation(),
        );

        // Set $yield_tag = tag
        let tag_const = wasm::i32_const(db, location, i32_ty, tag as i32);
        let tag_val = tag_const.as_operation().result(db, 0);
        ops.push(tag_const.as_operation());
        ops.push(wasm::global_set(db, location, tag_val, yield_globals::TAG_IDX).as_operation());

        // Set $yield_cont = continuation
        ops.push(wasm::global_set(db, location, cont_val, yield_globals::CONT_IDX).as_operation());

        // Set $yield_op_idx = op_idx
        let op_idx_const = wasm::i32_const(db, location, i32_ty, op_idx as i32);
        let op_idx_val = op_idx_const.as_operation().result(db, 0);
        ops.push(op_idx_const.as_operation());
        ops.push(wasm::global_set(db, location, op_idx_val, yield_globals::OP_IDX).as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.reset_yield_state` → wasm.global_set ($yield_state = 0)
struct LowerResetYieldStatePattern;

impl<'db> RewritePattern<'db> for LowerResetYieldStatePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME()
            || op.name(db) != trampoline::RESET_YIELD_STATE()
        {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        let mut ops = Vec::new();

        // Set $yield_state = 0 (not yielding)
        let const_0 = wasm::i32_const(db, location, i32_ty, 0);
        let const_0_val = const_0.as_operation().result(db, 0);
        ops.push(const_0.as_operation());
        ops.push(
            wasm::global_set(db, location, const_0_val, yield_globals::STATE_IDX).as_operation(),
        );

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.get_yield_continuation` → wasm.global_get + wasm.ref_cast
struct LowerGetYieldContinuationPattern;

impl<'db> RewritePattern<'db> for LowerGetYieldContinuationPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME()
            || op.name(db) != trampoline::GET_YIELD_CONTINUATION()
        {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let cont_type = continuation_adt_type(db);

        let mut ops = Vec::new();

        // Load continuation from $yield_cont global
        let get_cont = wasm::global_get(db, location, anyref_ty, yield_globals::CONT_IDX);
        let cont_anyref = get_cont.as_operation().result(db, 0);
        ops.push(get_cont.as_operation());

        // Cast anyref to continuation type
        let cont_cast_op = wasm::ref_cast(db, location, cont_anyref, cont_type, cont_type, None);
        ops.push(cont_cast_op.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.get_yield_shift_value` → wasm.global_get + adt.struct_get
struct LowerGetYieldShiftValuePattern;

impl<'db> RewritePattern<'db> for LowerGetYieldShiftValuePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME()
            || op.name(db) != trampoline::GET_YIELD_SHIFT_VALUE()
        {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let cont_type = continuation_adt_type(db);

        let mut ops = Vec::new();

        // Load continuation from $yield_cont global
        let get_cont = wasm::global_get(db, location, anyref_ty, yield_globals::CONT_IDX);
        let cont_anyref = get_cont.as_operation().result(db, 0);
        ops.push(get_cont.as_operation());

        // Cast anyref to continuation type
        let cont_cast = wasm::ref_cast(db, location, cont_anyref, cont_type, cont_type, None);
        let cont_ref = cont_cast.as_operation().result(db, 0);
        ops.push(cont_cast.as_operation());

        // Extract shift_value from continuation (field 3)
        let get_shift_value = adt::struct_get(
            db,
            location,
            cont_ref,
            anyref_ty,
            cont_type,
            Attribute::IntBits(3),
        );
        ops.push(get_shift_value.as_operation());

        RewriteResult::expand(ops)
    }
}

/// Lower `trampoline.check_yield` → wasm.global_get (yield_state)
struct LowerCheckYieldPattern;

impl<'db> RewritePattern<'db> for LowerCheckYieldPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME() || op.name(db) != trampoline::CHECK_YIELD()
        {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Get yield state from global
        let get_yield = wasm::global_get(db, location, i32_ty, yield_globals::STATE_IDX);

        RewriteResult::Replace(get_yield.as_operation())
    }
}

/// Lower `trampoline.get_yield_op_idx` → wasm.global_get (yield_op_idx)
struct LowerGetYieldOpIdxPattern;

impl<'db> RewritePattern<'db> for LowerGetYieldOpIdxPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        if op.dialect(db) != trampoline::DIALECT_NAME()
            || op.name(db) != trampoline::GET_YIELD_OP_IDX()
        {
            return RewriteResult::Unchanged;
        }

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Get op_idx from global
        let get_op_idx = wasm::global_get(db, location, i32_ty, yield_globals::OP_IDX);

        RewriteResult::Replace(get_op_idx.as_operation())
    }
}

// ============================================================================
// Pattern: Convert function type with trampoline.Step return type
// ============================================================================

/// Convert function signatures that return trampoline.Step to return _Step ADT.
///
/// This pattern matches func.func operations and updates their type attribute
/// when the return type is trampoline.Step.
struct ConvertFuncTypePattern;

impl<'db> RewritePattern<'db> for ConvertFuncTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(func) = Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let func_ty = func.r#type(db);
        let Some(fn_ty) = core::Func::from_type(db, func_ty) else {
            return RewriteResult::Unchanged;
        };

        // Convert parameter types
        let mut params_changed = false;
        let new_params: Vec<Type<'db>> = fn_ty
            .params(db)
            .iter()
            .map(|&ty| {
                if trampoline::Step::from_type(db, ty).is_some() {
                    params_changed = true;
                    step_adt_type(db)
                } else if trampoline::Continuation::from_type(db, ty).is_some() {
                    params_changed = true;
                    continuation_adt_type(db)
                } else if trampoline::ResumeWrapper::from_type(db, ty).is_some() {
                    params_changed = true;
                    resume_wrapper_adt_type(db)
                } else {
                    ty
                }
            })
            .collect();

        // Convert result type
        let result_ty = fn_ty.result(db);
        let (new_result, result_changed) = if trampoline::Step::from_type(db, result_ty).is_some() {
            (step_adt_type(db), true)
        } else if trampoline::Continuation::from_type(db, result_ty).is_some() {
            (continuation_adt_type(db), true)
        } else if trampoline::ResumeWrapper::from_type(db, result_ty).is_some() {
            (resume_wrapper_adt_type(db), true)
        } else {
            (result_ty, false)
        };

        if !params_changed && !result_changed {
            return RewriteResult::Unchanged;
        }

        tracing::debug!(
            "ConvertFuncTypePattern: converting func {} signature (params_changed={}, result_changed={}, orig_params={:?}, new_params={:?})",
            func.sym_name(db),
            params_changed,
            result_changed,
            fn_ty
                .params(db)
                .iter()
                .map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
                .collect::<Vec<_>>(),
            new_params
                .iter()
                .map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
                .collect::<Vec<_>>()
        );

        // Rebuild function with new type AND update block arguments
        // The block arguments must match the function signature parameter types
        // IMPORTANT: Build new_blocks first (using new_params.iter()), then new_fn_ty
        let body = func.body(db);
        let blocks = body.blocks(db);
        let new_blocks: IdVec<Block<'db>> = blocks
            .iter()
            .enumerate()
            .map(|(block_idx, block)| {
                if block_idx == 0 && params_changed {
                    // Entry block: update block args to match new param types
                    let new_block_args: IdVec<BlockArg<'db>> = new_params
                        .iter()
                        .enumerate()
                        .map(|(i, ty)| {
                            // Preserve any existing attributes from the original block arg
                            let orig_arg = block.args(db).get(i);
                            let attrs = orig_arg.map(|a| a.attrs(db).clone()).unwrap_or_default();
                            BlockArg::new(db, *ty, attrs)
                        })
                        .collect();
                    Block::new(
                        db,
                        block.id(db),
                        block.location(db),
                        new_block_args,
                        block.operations(db).clone(),
                    )
                } else {
                    *block
                }
            })
            .collect();
        let new_body = Region::new(db, body.location(db), new_blocks);

        // Build new function type (consumes new_params)
        let new_fn_ty = core::Func::with_effect(
            db,
            new_params.into_iter().collect(),
            new_result,
            fn_ty.effect(db),
        );

        tracing::debug!(
            "ConvertFuncTypePattern: new func type has result={}.{}, params={:?}",
            new_fn_ty.result(db).dialect(db),
            new_fn_ty.result(db).name(db),
            new_fn_ty
                .params(db)
                .iter()
                .map(|t| format!("{}.{}", t.dialect(db), t.name(db)))
                .collect::<Vec<_>>()
        );

        let new_op = op
            .modify(db)
            .attr(Symbol::new("type"), Attribute::Type(new_fn_ty.as_type()))
            .regions(IdVec::from(vec![new_body]))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Convert func.call result types from trampoline types to ADT types.
struct ConvertCallTypePattern;

impl<'db> RewritePattern<'db> for ConvertCallTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = func::Call::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let results = op.results(db);
        if results.is_empty() {
            return RewriteResult::Unchanged;
        }

        let result_ty = results[0];
        let new_result_ty = convert_trampoline_type(db, result_ty);

        if new_result_ty == result_ty {
            return RewriteResult::Unchanged;
        }

        tracing::debug!(
            "ConvertCallTypePattern: converting result type from {}.{} to {}.{}",
            result_ty.dialect(db),
            result_ty.name(db),
            new_result_ty.dialect(db),
            new_result_ty.name(db)
        );

        let new_op = op
            .modify(db)
            .results(IdVec::from(vec![new_result_ty]))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Convert func.call_indirect result types from trampoline types to ADT types.
struct ConvertCallIndirectTypePattern;

impl<'db> RewritePattern<'db> for ConvertCallIndirectTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = func::CallIndirect::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let results = op.results(db);
        if results.is_empty() {
            return RewriteResult::Unchanged;
        }

        let result_ty = results[0];
        let new_result_ty = convert_trampoline_type(db, result_ty);

        if new_result_ty == result_ty {
            return RewriteResult::Unchanged;
        }

        tracing::debug!(
            "ConvertCallIndirectTypePattern: converting result type from {}.{} to {}.{}",
            result_ty.dialect(db),
            result_ty.name(db),
            new_result_ty.dialect(db),
            new_result_ty.name(db)
        );

        let new_op = op
            .modify(db)
            .results(IdVec::from(vec![new_result_ty]))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Convert wasm.if result types from trampoline types to ADT types.
/// Note: scf.if is converted to wasm.if by scf_to_wasm before this pass runs.
struct ConvertWasmIfTypePattern;

impl<'db> RewritePattern<'db> for ConvertWasmIfTypePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        let Ok(_) = wasm::If::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let results = op.results(db);
        if results.is_empty() {
            return RewriteResult::Unchanged;
        }

        let result_ty = results[0];
        let new_result_ty = convert_trampoline_type(db, result_ty);

        if new_result_ty == result_ty {
            return RewriteResult::Unchanged;
        }

        tracing::debug!(
            "ConvertWasmIfTypePattern: converting result type from {}.{} to {}.{}",
            result_ty.dialect(db),
            result_ty.name(db),
            new_result_ty.dialect(db),
            new_result_ty.name(db)
        );

        let new_op = op
            .modify(db)
            .results(IdVec::from(vec![new_result_ty]))
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Convert a trampoline type to its corresponding ADT type.
fn convert_trampoline_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Type<'db> {
    if trampoline::Step::from_type(db, ty).is_some() {
        step_adt_type(db)
    } else if trampoline::Continuation::from_type(db, ty).is_some() {
        continuation_adt_type(db)
    } else if trampoline::ResumeWrapper::from_type(db, ty).is_some() {
        resume_wrapper_adt_type(db)
    } else {
        ty
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::rewrite::RewriteContext;
    use trunk_ir::{PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    // ========================================================================
    // Test: yield_globals constants
    // ========================================================================

    #[test]
    fn test_yield_globals_indices_are_sequential() {
        // Verify indices are sequential starting from 0
        assert_eq!(yield_globals::STATE_IDX, 0);
        assert_eq!(yield_globals::TAG_IDX, 1);
        assert_eq!(yield_globals::CONT_IDX, 2);
        assert_eq!(yield_globals::OP_IDX, 3);
    }

    #[test]
    fn test_yield_globals_indices_are_unique() {
        let indices = [
            yield_globals::STATE_IDX,
            yield_globals::TAG_IDX,
            yield_globals::CONT_IDX,
            yield_globals::OP_IDX,
        ];

        // Check all indices are unique
        for (i, &idx1) in indices.iter().enumerate() {
            for (j, &idx2) in indices.iter().enumerate() {
                if i != j {
                    assert_ne!(idx1, idx2, "Indices {} and {} should be unique", i, j);
                }
            }
        }
    }

    // ========================================================================
    // Test: LowerBuildContinuationPattern
    // ========================================================================

    /// Test helper: creates build_continuation and applies the pattern.
    /// Returns the struct_new operation's field count.
    #[salsa::tracked]
    fn build_continuation_produces_struct_new(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        let location = test_location(db);
        let cont_ty = trampoline::Continuation::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        let funcref_ty = wasm::Funcref::new(db).as_type();

        // Create operands: resume_fn (funcref), state (anyref), shift_value (anyref)
        let resume_fn_op = adt::ref_null(db, location, funcref_ty, funcref_ty);
        let state_op = adt::ref_null(db, location, anyref_ty, anyref_ty);
        let shift_value_op = adt::ref_null(db, location, anyref_ty, anyref_ty);

        let resume_fn = resume_fn_op.as_operation().result(db, 0);
        let state = state_op.as_operation().result(db, 0);
        let shift_value = shift_value_op.as_operation().result(db, 0);

        // Create trampoline.build_continuation
        let build_cont_op = trampoline::build_continuation(
            db,
            location,
            resume_fn,
            state,
            shift_value,
            cont_ty,
            1,
            0,
        );

        // Apply the pattern
        let pattern = LowerBuildContinuationPattern;
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let op = build_cont_op.as_operation();
        let adaptor = OpAdaptor::new(op, op.operands(db).clone(), vec![], &ctx, &type_converter);
        let result = pattern.match_and_rewrite(db, &op, &adaptor);

        // Extract the final operation (should be adt.struct_new)
        match result {
            RewriteResult::Expand(ops) => {
                let last_op = ops.last().unwrap();
                (last_op.dialect(db), last_op.name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("unexpected")),
        }
    }

    #[salsa_test]
    fn test_build_continuation_lowers_to_struct_new(db: &salsa::DatabaseImpl) {
        let (dialect, name) = build_continuation_produces_struct_new(db);
        assert_eq!(dialect, adt::DIALECT_NAME());
        assert_eq!(name, adt::STRUCT_NEW());
    }

    // ========================================================================
    // Test: LowerStateGetPattern
    // ========================================================================

    /// Test helper: creates state_get with field attribute and applies pattern.
    /// Returns the field index used in the lowered struct_get.
    #[salsa::tracked]
    fn state_get_extracts_field_index(db: &dyn salsa::Database) -> u64 {
        let location = test_location(db);
        let anyref_ty = wasm::Anyref::new(db).as_type();

        // Create a dummy state value
        let state_op = adt::ref_null(db, location, anyref_ty, anyref_ty);
        let state_val = state_op.as_operation().result(db, 0);

        // Create trampoline.state_get with field="field2"
        let fields = vec![
            (Symbol::new("field0"), anyref_ty),
            (Symbol::new("field1"), anyref_ty),
            (Symbol::new("field2"), anyref_ty),
        ];
        let state_type = adt::struct_type(db, Symbol::new("TestState"), fields);
        let state_get_op =
            trampoline::state_get(db, location, state_val, anyref_ty, Symbol::new("field2"));

        // Add state_type attribute manually (needed by the pattern)
        let op = state_get_op.as_operation();
        let op = op
            .modify(db)
            .attr(Symbol::new("state_type"), Attribute::Type(state_type))
            .build();

        // Apply the pattern
        let pattern = LowerStateGetPattern;
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let adaptor = OpAdaptor::new(op, op.operands(db).clone(), vec![], &ctx, &type_converter);
        let result = pattern.match_and_rewrite(db, &op, &adaptor);

        // Extract field_idx from the resulting adt.struct_get
        match result {
            RewriteResult::Expand(ops) => {
                for op in ops.iter() {
                    if op.dialect(db) == adt::DIALECT_NAME()
                        && op.name(db) == adt::STRUCT_GET()
                        && let Some(Attribute::IntBits(idx)) =
                            op.attributes(db).get(&Symbol::new("field"))
                    {
                        return *idx;
                    }
                }
                999 // Not found
            }
            _ => 999,
        }
    }

    #[salsa_test]
    fn test_state_get_parses_field_attribute(db: &salsa::DatabaseImpl) {
        let field_idx = state_get_extracts_field_index(db);
        assert_eq!(field_idx, 2, "field2 should be parsed to index 2");
    }

    // ========================================================================
    // Test: LowerSetYieldStatePattern
    // ========================================================================

    /// Test helper: creates set_yield_state and applies pattern.
    /// Returns the global indices used in the lowered global_set operations.
    #[salsa::tracked]
    fn set_yield_state_uses_correct_globals(db: &dyn salsa::Database) -> Vec<u32> {
        let location = test_location(db);
        let anyref_ty = wasm::Anyref::new(db).as_type();

        // Create a dummy continuation value
        let cont_op = adt::ref_null(db, location, anyref_ty, anyref_ty);
        let cont_val = cont_op.as_operation().result(db, 0);

        // Create trampoline.set_yield_state
        let set_yield_op = trampoline::set_yield_state(db, location, cont_val, 42, 7);

        // Apply the pattern
        let pattern = LowerSetYieldStatePattern;
        let ctx = RewriteContext::new();
        let type_converter = TypeConverter::new();
        let op = set_yield_op.as_operation();
        let adaptor = OpAdaptor::new(op, op.operands(db).clone(), vec![], &ctx, &type_converter);
        let result = pattern.match_and_rewrite(db, &op, &adaptor);

        // Extract global indices from global_set operations
        let mut indices = Vec::new();
        if let RewriteResult::Expand(ops) = result {
            for op in ops.iter() {
                if op.dialect(db) == wasm::DIALECT_NAME()
                    && op.name(db) == wasm::GLOBAL_SET()
                    && let Some(Attribute::IntBits(idx)) =
                        op.attributes(db).get(&Symbol::new("index"))
                {
                    indices.push(*idx as u32);
                }
            }
        }
        indices
    }

    #[salsa_test]
    fn test_set_yield_state_uses_yield_globals(db: &salsa::DatabaseImpl) {
        let indices = set_yield_state_uses_correct_globals(db);

        // Should set: STATE_IDX (0), TAG_IDX (1), CONT_IDX (2), OP_IDX (3)
        assert!(
            indices.contains(&yield_globals::STATE_IDX),
            "Should set yield_state global"
        );
        assert!(
            indices.contains(&yield_globals::TAG_IDX),
            "Should set yield_tag global"
        );
        assert!(
            indices.contains(&yield_globals::CONT_IDX),
            "Should set yield_cont global"
        );
        assert!(
            indices.contains(&yield_globals::OP_IDX),
            "Should set yield_op_idx global"
        );
    }
}
