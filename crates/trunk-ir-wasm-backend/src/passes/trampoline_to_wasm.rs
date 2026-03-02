//! Lower trampoline dialect operations to WASM and ADT operations.
//!
//! This pass converts all trampoline operations using ArenaRewritePattern infrastructure:
//!
//! ## Struct operations -> ADT operations
//! - `trampoline.build_continuation` -> `adt.struct_new` (Continuation type)
//! - `trampoline.step_done` -> `adt.struct_new` (Step type with tag=0)
//! - `trampoline.step_shift` -> `adt.struct_new` (Step type with tag=1)
//! - `trampoline.step_get` -> `adt.struct_get`
//! - `trampoline.continuation_get` -> `adt.struct_get`
//! - `trampoline.build_state` -> `adt.struct_new` (custom state type)
//! - `trampoline.build_resume_wrapper` -> `adt.struct_new` (ResumeWrapper type)
//! - `trampoline.resume_wrapper_get` -> `adt.struct_get`
//! - `trampoline.state_get` -> `adt.struct_get`
//!
//! ## Global state operations -> WASM operations
//! - `trampoline.set_yield_state` -> wasm.global_set (multiple)
//! - `trampoline.reset_yield_state` -> wasm.global_set
//! - `trampoline.get_yield_continuation` -> wasm.global_get + wasm.ref_cast
//! - `trampoline.get_yield_shift_value` -> wasm.global_get + adt.struct_get
//! - `trampoline.check_yield` -> wasm.global_get
//!
//! This pass uses ArenaTypeConverter to consistently convert trampoline types to ADT types.

use trunk_ir::arena::context::{IrContext, OperationDataBuilder};
use trunk_ir::arena::dialect::{
    adt as arena_adt, func as arena_func, trampoline as arena_trampoline, wasm as arena_wasm,
};
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, Location, TypeDataBuilder};
use trunk_ir::ir::Symbol;

use trunk_ir::arena::rewrite::type_converter::MaterializeResult;

/// Tag value for Step::Done variant.
const STEP_TAG_DONE: i32 = 0;
/// Tag value for Step::Shift variant.
const STEP_TAG_SHIFT: i32 = 1;

/// Global variable indices for yield/trampoline state.
///
/// These globals are created in the WASM lowering pipeline and accessed by trampoline operations.
/// The order must be kept consistent between global creation and access.
pub mod yield_globals {
    /// `$yield_state`: i32 (0 = normal, 1 = yielding)
    pub const STATE_IDX: u32 = 0;
    /// `$yield_tag`: i32 (prompt tag being yielded to)
    pub const TAG_IDX: u32 = 1;
    /// `$yield_cont`: anyref (captured continuation)
    pub const CONT_IDX: u32 = 2;
    /// `$yield_op_idx`: i32 (operation index within ability)
    pub const OP_IDX: u32 = 3;
}

/// Lower all trampoline operations to WASM/ADT using ArenaRewritePattern infrastructure.
pub fn lower(ctx: &mut IrContext, module: ArenaModule) {
    let type_converter = create_type_converter(ctx);

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(ConvertFuncTypePattern)
        .add_pattern(ConvertTrampolineResultTypePattern)
        .add_pattern(LowerBuildContinuationPattern)
        .add_pattern(LowerStepDonePattern)
        .add_pattern(LowerStepShiftPattern)
        .add_pattern(LowerTrampolineStructGetPattern)
        .add_pattern(LowerBuildStatePattern)
        .add_pattern(LowerBuildResumeWrapperPattern)
        .add_pattern(LowerSetYieldStatePattern)
        .add_pattern(LowerResetYieldStatePattern)
        .add_pattern(LowerYieldContinuationAccessPattern)
        .add_pattern(LowerYieldGlobalGetPattern);

    applicator.apply_partial(ctx, module);
}

// ============================================================================
// Type Helpers
// ============================================================================

fn intern_type(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new(dialect), Symbol::new(name)).build())
}

fn is_type(ctx: &IrContext, ty: TypeRef, dialect: &'static str, name: &'static str) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new(dialect) && data.name == Symbol::new(name)
}

// ============================================================================
// Type Definitions (Single Source of Truth)
// ============================================================================

/// Get the canonical Step ADT type.
/// Layout: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
///
/// IMPORTANT: Must use wasm.anyref to match step_marker_type in gc_types.rs.
/// Using tribute_rt.any would create a different type identity, causing
/// type_idx_by_type lookup failures.
pub fn step_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = intern_type(ctx, "core", "i32");
    let anyref_ty = intern_type(ctx, "wasm", "anyref");

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(i32_ty)
            .param(anyref_ty)
            .param(i32_ty)
            .param(i32_ty)
            .attr("name", ArenaAttribute::Symbol(Symbol::new("_Step")))
            .build(),
    )
}

/// Get the canonical Continuation ADT type.
/// Layout: (resume_fn: i32, state: anyref, tag: i32, shift_value: anyref)
///
/// resume_fn is stored as i32 (function table index), same as closures.
/// Uses wasm.anyref for consistency with step_adt_type.
pub fn continuation_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = intern_type(ctx, "core", "i32");
    let anyref_ty = intern_type(ctx, "wasm", "anyref");

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(i32_ty)
            .param(anyref_ty)
            .param(i32_ty)
            .param(anyref_ty)
            .attr("name", ArenaAttribute::Symbol(Symbol::new("_Continuation")))
            .build(),
    )
}

/// Get the canonical ResumeWrapper ADT type.
/// Layout: (state: anyref, resume_value: anyref)
///
/// Uses wasm.anyref for consistency with step_adt_type.
pub fn resume_wrapper_adt_type(ctx: &mut IrContext) -> TypeRef {
    let anyref_ty = intern_type(ctx, "wasm", "anyref");

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(anyref_ty)
            .param(anyref_ty)
            .attr(
                "name",
                ArenaAttribute::Symbol(Symbol::new("_ResumeWrapper")),
            )
            .build(),
    )
}

/// Get the BoxedF64 struct type for f64 boxing.
/// Layout: (value: f64)
///
/// Used to box f64 values into anyref for storage in generic containers.
fn boxed_f64_type(ctx: &mut IrContext) -> TypeRef {
    let f64_ty = intern_type(ctx, "core", "f64");
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(f64_ty)
            .attr("name", ArenaAttribute::Symbol(Symbol::new("_BoxedF64")))
            .build(),
    )
}

// ============================================================================
// Type Converter
// ============================================================================

/// Create an ArenaTypeConverter that converts trampoline types to ADT types.
fn create_type_converter(ctx: &mut IrContext) -> ArenaTypeConverter {
    // Pre-compute types
    let step_ty = step_adt_type(ctx);
    let cont_ty = continuation_adt_type(ctx);
    let rw_ty = resume_wrapper_adt_type(ctx);
    let i32_ty = intern_type(ctx, "core", "i32");
    let f64_ty = intern_type(ctx, "core", "f64");
    let anyref_ty = intern_type(ctx, "wasm", "anyref");
    let i31ref_ty = intern_type(ctx, "wasm", "i31ref");
    let boxed_f64_ty = boxed_f64_type(ctx);

    let mut tc = ArenaTypeConverter::new();

    // Convert trampoline.step -> _Step ADT
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, "trampoline", "step") {
            Some(step_ty)
        } else {
            None
        }
    });

    // Convert trampoline.continuation -> _Continuation ADT
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, "trampoline", "continuation") {
            Some(cont_ty)
        } else {
            None
        }
    });

    // Convert trampoline.resume_wrapper -> _ResumeWrapper ADT
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, "trampoline", "resume_wrapper") {
            Some(rw_ty)
        } else {
            None
        }
    });

    // Convert cont.prompt_tag -> core.i32 (same representation at runtime)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, "cont", "prompt_tag") {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Combined materializer for all type conversions
    tc.set_materializer(move |ctx, location, value, from_ty, to_ty| {
        if from_ty == to_ty {
            return None;
        }

        let is_from_trampoline_step = is_type(ctx, from_ty, "trampoline", "step");
        let is_from_trampoline_cont = is_type(ctx, from_ty, "trampoline", "continuation");
        let is_from_trampoline_rw = is_type(ctx, from_ty, "trampoline", "resume_wrapper");

        // trampoline types -> ADT types (no-op, same representation)
        if is_from_trampoline_step && to_ty == step_ty {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        if is_from_trampoline_cont && to_ty == cont_ty {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        if is_from_trampoline_rw && to_ty == rw_ty {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // i32 -> anyref (box via i31ref, then upcast to anyref)
        let is_from_i32 = from_ty == i32_ty;
        let is_to_anyref = to_ty == anyref_ty;

        if is_from_i32 && is_to_anyref {
            // wasm.ref_i31: i32 -> i31ref
            let box_op = arena_wasm::ref_i31(ctx, location, value, i31ref_ty);
            let i31ref_value = ctx.op_result(box_op.op_ref(), 0);

            // Upcast i31ref -> anyref (no-op in runtime, but needed for IR type correctness)
            let upcast_op =
                arena_wasm::ref_cast(ctx, location, i31ref_value, anyref_ty, anyref_ty, None);
            let final_value = ctx.op_result(upcast_op.op_ref(), 0);

            return Some(MaterializeResult {
                value: final_value,
                ops: vec![box_op.op_ref(), upcast_op.op_ref()],
            });
        }

        // f64 -> anyref (box via BoxedF64 struct)
        let is_from_f64 = from_ty == f64_ty;

        if is_from_f64 && is_to_anyref {
            // Create BoxedF64 struct containing the f64 value
            let box_op =
                arena_adt::struct_new(ctx, location, vec![value], boxed_f64_ty, boxed_f64_ty);
            let boxed_value = ctx.op_result(box_op.op_ref(), 0);

            // Upcast BoxedF64 to anyref
            let upcast_op =
                arena_wasm::ref_cast(ctx, location, boxed_value, anyref_ty, anyref_ty, None);
            let final_value = ctx.op_result(upcast_op.op_ref(), 0);

            return Some(MaterializeResult {
                value: final_value,
                ops: vec![box_op.op_ref(), upcast_op.op_ref()],
            });
        }

        // anyref -> i32 (unbox via i31ref)
        let is_from_anyref = from_ty == anyref_ty;
        let is_to_i32 = to_ty == i32_ty;

        if is_from_anyref && is_to_i32 {
            // Cast anyref to i31ref
            let ref_cast_op =
                arena_wasm::ref_cast(ctx, location, value, i31ref_ty, i31ref_ty, None);
            let i31ref_value = ctx.op_result(ref_cast_op.op_ref(), 0);

            // Unbox i31ref to i32
            let unbox_op = arena_wasm::i31_get_s(ctx, location, i31ref_value, i32_ty);
            let final_value = ctx.op_result(unbox_op.op_ref(), 0);

            return Some(MaterializeResult {
                value: final_value,
                ops: vec![ref_cast_op.op_ref(), unbox_op.op_ref()],
            });
        }

        // anyref -> f64 (unbox via BoxedF64 struct)
        let is_to_f64 = to_ty == f64_ty;

        if is_from_anyref && is_to_f64 {
            // Cast anyref to BoxedF64
            let ref_cast_op = arena_adt::ref_cast(ctx, location, value, boxed_f64_ty, boxed_f64_ty);
            let boxed_value = ctx.op_result(ref_cast_op.op_ref(), 0);

            // Extract f64 from BoxedF64.value (field 0)
            let unbox_op =
                arena_adt::struct_get(ctx, location, boxed_value, f64_ty, boxed_f64_ty, 0);
            let final_value = ctx.op_result(unbox_op.op_ref(), 0);

            return Some(MaterializeResult {
                value: final_value,
                ops: vec![ref_cast_op.op_ref(), unbox_op.op_ref()],
            });
        }

        // cont.prompt_tag -> i32 (no-op, same representation)
        let is_from_prompt_tag = is_type(ctx, from_ty, "cont", "prompt_tag");

        if is_from_prompt_tag && is_to_i32 {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        None
    });

    tc
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_i32_const(ctx: &mut IrContext, location: Location, value: i32) -> OpRef {
    let i32_ty = intern_type(ctx, "core", "i32");
    arena_wasm::i32_const(ctx, location, i32_ty, value).op_ref()
}

/// Push an i32 constant and a global_set operation to set a global variable.
fn push_set_i32_global(
    ctx: &mut IrContext,
    location: Location,
    value: i32,
    global_idx: u32,
    ops: &mut Vec<OpRef>,
) {
    let const_op = create_i32_const(ctx, location, value);
    let const_val = ctx.op_result(const_op, 0);
    ops.push(const_op);
    ops.push(arena_wasm::global_set(ctx, location, const_val, global_idx).op_ref());
}

/// Cast a value to anyref type using TypeConverter materializations.
///
/// Uses the rewriter's type converter to generate boxing operations:
/// - For i32: boxes to i31ref then casts to anyref
/// - For f64: boxes using adt.struct_new (BoxedF64)
/// - For reference types: uses ref_cast upcast
fn materialize_to_any(
    ctx: &mut IrContext,
    location: Location,
    value: ValueRef,
    rewriter: &PatternRewriter<'_>,
    ops: &mut Vec<OpRef>,
) -> ValueRef {
    let anyref_ty = intern_type(ctx, "wasm", "anyref");

    // Get the value's type
    let value_ty = ctx.value_ty(value);

    // Already anyref - no conversion needed
    if value_ty == anyref_ty {
        return value;
    }

    // Try to materialize the conversion via type converter
    let tc = rewriter.type_converter();
    if let Some(result) = tc.materialize(ctx, location, value, value_ty, anyref_ty) {
        if result.ops.is_empty() {
            // NoOp - value is already correct
            return result.value;
        }
        let final_value = result.value;
        ops.extend(result.ops);
        return final_value;
    }

    // Fallback: upcast to anyref using ref_cast
    let cast_op = arena_wasm::ref_cast(ctx, location, value, anyref_ty, anyref_ty, None);
    let cast_val = ctx.op_result(cast_op.op_ref(), 0);
    ops.push(cast_op.op_ref());
    cast_val
}

/// Unbox a value from anyref to a target type using TypeConverter materializations.
///
/// Uses the rewriter's type converter to generate unboxing operations:
/// - For i32: ref_cast to i31ref then i31_get_s
/// - For f64: ref_cast to BoxedF64 then struct_get
/// - For reference types: ref_cast
fn materialize_from_any(
    ctx: &mut IrContext,
    location: Location,
    value: ValueRef,
    target_ty: TypeRef,
    rewriter: &PatternRewriter<'_>,
    ops: &mut Vec<OpRef>,
) -> ValueRef {
    let anyref_ty = intern_type(ctx, "wasm", "anyref");

    // Already the target type - no conversion needed
    if target_ty == anyref_ty {
        return value;
    }

    // Try to materialize the conversion via type converter
    let tc = rewriter.type_converter();
    if let Some(result) = tc.materialize(ctx, location, value, anyref_ty, target_ty) {
        if result.ops.is_empty() {
            return result.value;
        }
        let final_value = result.value;
        ops.extend(result.ops);
        return final_value;
    }

    // Fallback: ref_cast to target type (for reference types)
    let cast_op = arena_adt::ref_cast(ctx, location, value, target_ty, target_ty);
    let cast_val = ctx.op_result(cast_op.op_ref(), 0);
    ops.push(cast_op.op_ref());
    cast_val
}

/// Create a null reference of anyref type.
fn null_any(ctx: &mut IrContext, location: Location, ops: &mut Vec<OpRef>) -> ValueRef {
    let anyref_ty = intern_type(ctx, "wasm", "anyref");
    let null_op = arena_adt::ref_null(ctx, location, anyref_ty, anyref_ty);
    let null_val = ctx.op_result(null_op.op_ref(), 0);
    ops.push(null_op.op_ref());
    null_val
}

/// Helper to insert prefix ops and replace op from an ops vec.
///
/// All ops except the last are inserted before, the last replaces the current op.
fn expand_ops(rewriter: &mut PatternRewriter<'_>, ops: Vec<OpRef>) {
    let len = ops.len();
    for (i, op) in ops.into_iter().enumerate() {
        if i < len - 1 {
            rewriter.insert_op(op);
        } else {
            rewriter.replace_op(op);
        }
    }
}

/// Convert a trampoline type to its corresponding ADT type.
fn convert_trampoline_type(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    if is_type(ctx, ty, "trampoline", "step") {
        step_adt_type(ctx)
    } else if is_type(ctx, ty, "trampoline", "continuation") {
        continuation_adt_type(ctx)
    } else if is_type(ctx, ty, "trampoline", "resume_wrapper") {
        resume_wrapper_adt_type(ctx)
    } else {
        ty
    }
}

// ============================================================================
// Patterns: Struct Operations -> ADT
// ============================================================================

/// Lower `trampoline.build_continuation` -> `adt.struct_new`
///
/// Takes tag as first operand (dynamic tag from evidence lookup).
struct LowerBuildContinuationPattern;

impl ArenaRewritePattern for LowerBuildContinuationPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_trampoline::BuildContinuation::from_op(ctx, op).is_err() {
            return false;
        }

        let location = ctx.op(op).location;
        let cont_type = continuation_adt_type(ctx);

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        // Operands are (tag, resume_fn, state, shift_value)
        let tag_operand = operands.first().copied();
        let resume_fn = operands.get(1).copied();
        let state = operands.get(2).copied();
        let shift_value = operands.get(3).copied();

        let mut ops = Vec::new();

        // Tag comes from operand (cont.prompt_tag has same representation as i32)
        let tag_value = tag_operand.expect("build_continuation requires tag operand");

        // resume_fn field
        let resume_fn_field = if let Some(v) = resume_fn {
            v
        } else {
            let null_const = create_i32_const(ctx, location, -1);
            let val = ctx.op_result(null_const, 0);
            ops.push(null_const);
            val
        };

        // state field - cast to any
        let state_field = if let Some(v) = state {
            materialize_to_any(ctx, location, v, rewriter, &mut ops)
        } else {
            null_any(ctx, location, &mut ops)
        };

        // shift_value field - cast to any
        let shift_value_field = if let Some(v) = shift_value {
            materialize_to_any(ctx, location, v, rewriter, &mut ops)
        } else {
            null_any(ctx, location, &mut ops)
        };

        let fields = vec![resume_fn_field, state_field, tag_value, shift_value_field];

        let struct_new =
            arena_adt::struct_new(ctx, location, fields, cont_type, cont_type).op_ref();
        ops.push(struct_new);

        expand_ops(rewriter, ops);
        true
    }
}

/// Lower `trampoline.step_done` -> `adt.struct_new` with tag=0
struct LowerStepDonePattern;

impl ArenaRewritePattern for LowerStepDonePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_trampoline::StepDone::from_op(ctx, op).is_err() {
            return false;
        }

        let location = ctx.op(op).location;
        let step_type = step_adt_type(ctx);

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let value = operands.first().copied();

        let mut ops = Vec::new();

        // Create constants for tag=0, prompt=0, op_idx=0
        let tag_const = create_i32_const(ctx, location, STEP_TAG_DONE);
        let prompt_const = create_i32_const(ctx, location, 0);
        let op_idx_const = create_i32_const(ctx, location, 0);

        let tag_value = ctx.op_result(tag_const, 0);
        let prompt_value = ctx.op_result(prompt_const, 0);
        let op_idx_value = ctx.op_result(op_idx_const, 0);

        ops.push(tag_const);
        ops.push(prompt_const);
        ops.push(op_idx_const);

        // Value field - cast to any using TypeConverter materialization
        let value_field = if let Some(v) = value {
            materialize_to_any(ctx, location, v, rewriter, &mut ops)
        } else {
            null_any(ctx, location, &mut ops)
        };

        let fields = vec![tag_value, value_field, prompt_value, op_idx_value];
        let struct_new =
            arena_adt::struct_new(ctx, location, fields, step_type, step_type).op_ref();
        ops.push(struct_new);

        expand_ops(rewriter, ops);
        true
    }
}

/// Lower `trampoline.step_shift` -> `adt.struct_new` with tag=1
///
/// Takes prompt tag as first operand (dynamic tag from evidence lookup).
struct LowerStepShiftPattern;

impl ArenaRewritePattern for LowerStepShiftPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(step_shift) = arena_trampoline::StepShift::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let step_type = step_adt_type(ctx);

        let op_idx = step_shift.op_idx(ctx);

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        // Operands are (prompt, continuation)
        let prompt_operand = operands.first().copied();
        let continuation = operands.get(1).copied();

        let mut ops = Vec::new();

        // Create constants for tag
        let tag_const = create_i32_const(ctx, location, STEP_TAG_SHIFT);
        let tag_value = ctx.op_result(tag_const, 0);
        ops.push(tag_const);

        // Prompt comes from operand (cont.prompt_tag has same representation as i32)
        let prompt_value = prompt_operand.expect("step_shift requires prompt operand");

        // op_idx from attribute
        let op_idx_const = create_i32_const(ctx, location, op_idx as i32);
        let op_idx_value = ctx.op_result(op_idx_const, 0);
        ops.push(op_idx_const);

        // Value field - cast continuation to any
        let value_field = if let Some(v) = continuation {
            materialize_to_any(ctx, location, v, rewriter, &mut ops)
        } else {
            null_any(ctx, location, &mut ops)
        };

        let fields = vec![tag_value, value_field, prompt_value, op_idx_value];
        let struct_new =
            arena_adt::struct_new(ctx, location, fields, step_type, step_type).op_ref();
        ops.push(struct_new);

        expand_ops(rewriter, ops);
        true
    }
}

/// Lower trampoline struct-get operations -> `adt.struct_get` [+ materialize_from_any]
///
/// Applies to:
/// - `trampoline.step_get` (field 1 is anyref)
/// - `trampoline.continuation_get` (fields 1, 3 are anyref)
/// - `trampoline.resume_wrapper_get` (all fields are anyref)
/// - `trampoline.state_get` (all fields are anyref)
struct LowerTrampolineStructGetPattern;

impl ArenaRewritePattern for LowerTrampolineStructGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let anyref_ty = intern_type(ctx, "wasm", "anyref");

        // Try to match each struct-get operation type and extract parameters
        let (struct_type, field_idx, is_any_field) =
            if let Ok(step_get) = arena_trampoline::StepGet::from_op(ctx, op) {
                let idx = step_get.field(ctx);
                (step_adt_type(ctx), idx, idx == 1)
            } else if let Ok(cont_get) = arena_trampoline::ContinuationGet::from_op(ctx, op) {
                let idx = cont_get.field(ctx);
                (continuation_adt_type(ctx), idx, idx == 1 || idx == 3)
            } else if let Ok(wrapper_get) = arena_trampoline::ResumeWrapperGet::from_op(ctx, op) {
                let idx = wrapper_get.field(ctx);
                (resume_wrapper_adt_type(ctx), idx, true)
            } else if let Ok(state_get) = arena_trampoline::StateGet::from_op(ctx, op) {
                let field_idx = state_get.field(ctx);
                // state_type is added manually as an attribute
                let state_type = ctx
                    .op(op)
                    .attributes
                    .get(&Symbol::new("state_type"))
                    .and_then(|a| match a {
                        ArenaAttribute::Type(ty) => Some(*ty),
                        _ => None,
                    })
                    .unwrap_or(anyref_ty);
                (state_type, field_idx, true)
            } else {
                return false;
            };

        let location = ctx.op(op).location;
        let expected_result_type = ctx
            .op_result_types(op)
            .first()
            .copied()
            .unwrap_or(anyref_ty);
        let struct_value = ctx
            .op_operands(op)
            .first()
            .copied()
            .expect("struct_get requires operand");

        if is_any_field {
            let mut ops = Vec::new();

            let struct_get = arena_adt::struct_get(
                ctx,
                location,
                struct_value,
                anyref_ty,
                struct_type,
                field_idx,
            )
            .op_ref();
            let any_value = ctx.op_result(struct_get, 0);
            ops.push(struct_get);

            materialize_from_any(
                ctx,
                location,
                any_value,
                expected_result_type,
                rewriter,
                &mut ops,
            );

            expand_ops(rewriter, ops);
            true
        } else {
            let struct_get = arena_adt::struct_get(
                ctx,
                location,
                struct_value,
                expected_result_type,
                struct_type,
                field_idx,
            )
            .op_ref();

            rewriter.replace_op(struct_get);
            true
        }
    }
}

/// Lower `trampoline.build_state` -> `adt.struct_new`
struct LowerBuildStatePattern;

impl ArenaRewritePattern for LowerBuildStatePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(build_state) = arena_trampoline::BuildState::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();

        // Use the original state type directly. State fields are already set to
        // anyref (tribute_rt.any) in cont_to_trampoline, ensuring the same nominal
        // type is used for both creation (build_state) and extraction (struct_get).
        let state_type = build_state.state_type(ctx);

        let mut ops = Vec::new();
        let mut fields = Vec::new();

        // Cast all operands to any type using TypeConverter materialization
        for v in operands.iter() {
            let casted = materialize_to_any(ctx, location, *v, rewriter, &mut ops);
            fields.push(casted);
        }

        let struct_new =
            arena_adt::struct_new(ctx, location, fields, state_type, state_type).op_ref();
        ops.push(struct_new);

        expand_ops(rewriter, ops);
        true
    }
}

/// Lower `trampoline.build_resume_wrapper` -> `adt.struct_new`
struct LowerBuildResumeWrapperPattern;

impl ArenaRewritePattern for LowerBuildResumeWrapperPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_trampoline::BuildResumeWrapper::from_op(ctx, op).is_err() {
            return false;
        }

        let location = ctx.op(op).location;
        let wrapper_type = resume_wrapper_adt_type(ctx);

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let state = operands.first().copied();
        let resume_value = operands.get(1).copied();

        let mut ops = Vec::new();

        // Cast state to any using TypeConverter materialization
        let state_field = if let Some(v) = state {
            materialize_to_any(ctx, location, v, rewriter, &mut ops)
        } else {
            null_any(ctx, location, &mut ops)
        };

        // Cast resume_value to any using TypeConverter materialization
        let resume_value_field = if let Some(v) = resume_value {
            materialize_to_any(ctx, location, v, rewriter, &mut ops)
        } else {
            null_any(ctx, location, &mut ops)
        };

        let fields = vec![state_field, resume_value_field];
        let struct_new =
            arena_adt::struct_new(ctx, location, fields, wrapper_type, wrapper_type).op_ref();
        ops.push(struct_new);

        expand_ops(rewriter, ops);
        true
    }
}

// ============================================================================
// Patterns: Global State Operations -> WASM
// ============================================================================

/// Lower `trampoline.set_yield_state` -> wasm.global_set (multiple)
///
/// Takes tag as first operand (dynamic tag from evidence lookup).
struct LowerSetYieldStatePattern;

impl ArenaRewritePattern for LowerSetYieldStatePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(set_yield) = arena_trampoline::SetYieldState::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let op_idx = set_yield.op_idx(ctx);

        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        // Operands are (tag, continuation)
        let tag_operand = operands
            .first()
            .copied()
            .expect("set_yield_state requires tag operand");
        let cont_val = operands
            .get(1)
            .copied()
            .expect("set_yield_state requires continuation operand");

        let mut ops = Vec::new();

        // Set $yield_state = 1 (yielding)
        push_set_i32_global(ctx, location, 1, yield_globals::STATE_IDX, &mut ops);

        // Set $yield_tag = tag (cont.prompt_tag has same representation as i32)
        ops.push(
            arena_wasm::global_set(ctx, location, tag_operand, yield_globals::TAG_IDX).op_ref(),
        );

        // Set $yield_cont = continuation (as anyref)
        let cont_any = materialize_to_any(ctx, location, cont_val, rewriter, &mut ops);
        ops.push(arena_wasm::global_set(ctx, location, cont_any, yield_globals::CONT_IDX).op_ref());

        // Set $yield_op_idx = op_idx
        push_set_i32_global(
            ctx,
            location,
            op_idx as i32,
            yield_globals::OP_IDX,
            &mut ops,
        );

        expand_ops(rewriter, ops);
        true
    }
}

/// Lower `trampoline.reset_yield_state` -> wasm.global_set ($yield_state = 0)
struct LowerResetYieldStatePattern;

impl ArenaRewritePattern for LowerResetYieldStatePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_trampoline::ResetYieldState::from_op(ctx, op).is_err() {
            return false;
        }

        let location = ctx.op(op).location;
        let mut ops = Vec::new();

        // Set $yield_state = 0 (not yielding)
        push_set_i32_global(ctx, location, 0, yield_globals::STATE_IDX, &mut ops);

        expand_ops(rewriter, ops);
        true
    }
}

/// Lower yield continuation access operations -> wasm.global_get + wasm.ref_cast [+ adt.struct_get]
///
/// Applies to:
/// - `trampoline.get_yield_continuation` -> load and cast continuation
/// - `trampoline.get_yield_shift_value` -> load, cast, and extract shift_value field
struct LowerYieldContinuationAccessPattern;

impl ArenaRewritePattern for LowerYieldContinuationAccessPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let data = ctx.op(op);
        if data.dialect != Symbol::new("trampoline") {
            return false;
        }

        let extract_shift_value = if data.name == Symbol::new("get_yield_continuation") {
            false
        } else if data.name == Symbol::new("get_yield_shift_value") {
            true
        } else {
            return false;
        };

        let location = ctx.op(op).location;
        let anyref_ty = intern_type(ctx, "wasm", "anyref");
        let cont_type = continuation_adt_type(ctx);

        let mut ops = Vec::new();

        // Load continuation from $yield_cont global
        let get_cont =
            arena_wasm::global_get(ctx, location, anyref_ty, yield_globals::CONT_IDX).op_ref();
        let cont_anyref = ctx.op_result(get_cont, 0);
        ops.push(get_cont);

        // Cast anyref to continuation type
        let cont_cast =
            arena_wasm::ref_cast(ctx, location, cont_anyref, cont_type, cont_type, None).op_ref();
        let cont_ref = ctx.op_result(cont_cast, 0);
        ops.push(cont_cast);

        if extract_shift_value {
            // Extract shift_value from continuation (field 3)
            let get_shift_value =
                arena_adt::struct_get(ctx, location, cont_ref, anyref_ty, cont_type, 3).op_ref();
            ops.push(get_shift_value);
        }

        expand_ops(rewriter, ops);
        true
    }
}

/// Lower yield global getter operations -> wasm.global_get
///
/// Applies to:
/// - `trampoline.check_yield` -> yield_state global
/// - `trampoline.get_yield_op_idx` -> yield_op_idx global
struct LowerYieldGlobalGetPattern;

impl ArenaRewritePattern for LowerYieldGlobalGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let data = ctx.op(op);
        if data.dialect != Symbol::new("trampoline") {
            return false;
        }

        let global_idx = if data.name == Symbol::new("check_yield") {
            yield_globals::STATE_IDX
        } else if data.name == Symbol::new("get_yield_op_idx") {
            yield_globals::OP_IDX
        } else {
            return false;
        };

        let location = ctx.op(op).location;
        let i32_ty = intern_type(ctx, "core", "i32");
        let get_global = arena_wasm::global_get(ctx, location, i32_ty, global_idx).op_ref();

        rewriter.replace_op(get_global);
        true
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

impl ArenaRewritePattern for ConvertFuncTypePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(func_op) = arena_func::Func::from_op(ctx, op) else {
            return false;
        };

        let func_ty = func_op.r#type(ctx);
        let func_data = ctx.types.get(func_ty);
        if func_data.dialect != Symbol::new("core") || func_data.name != Symbol::new("func") {
            return false;
        }

        // params[0] = return type, params[1..] = parameter types
        if func_data.params.is_empty() {
            return false;
        }

        // Clone data we need before mutating ctx.types
        let params: Vec<TypeRef> = func_data.params.to_vec();
        let effect_attr = func_data.attrs.get(&Symbol::new("effect")).cloned();

        // Convert return type (params[0]) and parameter types (params[1..])
        let new_result = convert_trampoline_type(ctx, params[0]);
        let result_changed = new_result != params[0];

        let mut params_changed = false;
        let mut new_params = Vec::with_capacity(params.len());
        new_params.push(new_result); // return type at index 0

        for &param_ty in &params[1..] {
            let new_param = convert_trampoline_type(ctx, param_ty);
            if new_param != param_ty {
                params_changed = true;
            }
            new_params.push(new_param);
        }

        if !params_changed && !result_changed {
            return false;
        }

        // Build new function type preserving effect attribute
        let mut builder =
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).params(new_params);
        if let Some(eff) = effect_attr {
            builder = builder.attr("effect", eff);
        }
        let new_func_ty = ctx.types.intern(builder.build());

        // Update block arguments in the entry block to match new param types
        let body = func_op.body(ctx);
        let blocks = ctx.region(body).blocks.to_vec();
        if let Some(&entry_block) = blocks.first()
            && params_changed
        {
            let block_args = ctx.block(entry_block).args.clone();
            for (i, _arg) in block_args.iter().enumerate() {
                let new_func_ty_data = ctx.types.get(new_func_ty);
                // params[1..] maps to block args[0..]
                if i + 1 < new_func_ty_data.params.len() {
                    let new_param_ty = new_func_ty_data.params[i + 1];
                    ctx.set_block_arg_type(entry_block, i as u32, new_param_ty);
                }
            }
        }

        // Rebuild function with new type, reusing detached body region
        let func_name = func_op.sym_name(ctx);
        let loc = ctx.op(op).location;
        ctx.detach_region(body);
        let new_op = arena_func::func(ctx, loc, func_name, new_func_ty, body).op_ref();

        rewriter.replace_op(new_op);
        true
    }
}

/// Convert result types from trampoline types to ADT types.
///
/// Applies to: func.call, func.call_indirect, wasm.if
struct ConvertTrampolineResultTypePattern;

impl ArenaRewritePattern for ConvertTrampolineResultTypePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        // Check if this is a supported operation type
        let is_supported = arena_func::Call::from_op(ctx, op).is_ok()
            || arena_func::CallIndirect::from_op(ctx, op).is_ok()
            || arena_wasm::If::from_op(ctx, op).is_ok();

        if !is_supported {
            return false;
        }

        let result_types = ctx.op_result_types(op).to_vec();
        if result_types.is_empty() {
            return false;
        }

        let result_ty = result_types[0];
        let new_result_ty = convert_trampoline_type(ctx, result_ty);

        if new_result_ty == result_ty {
            return false;
        }

        tracing::debug!("ConvertTrampolineResultTypePattern: converting result type");

        // Rebuild the op with updated result types
        let data = ctx.op(op);
        let loc = data.location;
        let dialect = data.dialect;
        let name = data.name;
        let attrs = data.attributes.clone();
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let regions: Vec<_> = data.regions.to_vec();
        let successors: Vec<_> = data.successors.to_vec();

        let new_result_types: Vec<TypeRef> = result_types
            .iter()
            .enumerate()
            .map(|(i, &ty)| if i == 0 { new_result_ty } else { ty })
            .collect();

        // Detach regions from the old op before building the new one
        for &r in &regions {
            ctx.detach_region(r);
        }

        let mut builder = OperationDataBuilder::new(loc, dialect, name)
            .operands(operands)
            .results(new_result_types);
        for (key, val) in attrs {
            builder = builder.attr(key, val);
        }
        for r in regions {
            builder = builder.region(r);
        }
        for s in successors {
            builder = builder.successor(s);
        }
        let new_data = builder.build(ctx);
        let new_op = ctx.create_op(new_data);

        rewriter.replace_op(new_op);
        true
    }
}
