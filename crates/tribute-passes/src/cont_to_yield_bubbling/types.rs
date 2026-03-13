//! ADT type definitions for yield bubbling.
//!
//! Creates the YieldResult enum, ShiftInfo struct, Continuation struct,
//! and ResumeWrapper struct types used by the yield bubbling transformation.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{Attribute, TypeDataBuilder};

use tribute_ir::dialect::tribute_rt as arena_tribute_rt;

/// Cached type references for yield bubbling types.
///
/// Created once per pass invocation via `YieldBubblingTypes::new()`.
pub(crate) struct YieldBubblingTypes {
    /// `adt.enum @YieldResult { Done(anyref), Shift(ShiftInfo) }`
    pub(crate) yield_result: TypeRef,
    /// `adt.struct @ShiftInfo { value: anyref, prompt: i32, op_idx: i32, continuation: anyref }`
    pub(crate) shift_info: TypeRef,
    /// `adt.struct @Continuation { resume_fn: core.ptr, state: anyref }`
    pub(crate) continuation: TypeRef,
    /// `adt.struct @ResumeWrapper { state: anyref, resume_value: anyref }`
    pub(crate) resume_wrapper: TypeRef,
    /// `tribute_rt.anyref`
    pub(crate) anyref: TypeRef,
    /// `core.ptr` (raw pointer, not RC-managed)
    pub(crate) ptr: TypeRef,
    /// `core.i32`
    pub(crate) i32: TypeRef,
    /// `core.i1`
    pub(crate) i1: TypeRef,
}

impl YieldBubblingTypes {
    /// Create and intern all yield bubbling types.
    pub(crate) fn new(ctx: &mut IrContext) -> Self {
        let anyref_ty = arena_tribute_rt::anyref(ctx).as_type_ref();
        let ptr_ty = intern_ptr(ctx);
        let i32_ty = intern_i32(ctx);
        let i1_ty = intern_i1(ctx);

        let shift_info_ty = make_shift_info_type(ctx, anyref_ty, i32_ty);
        let continuation_ty = make_continuation_type(ctx, ptr_ty, anyref_ty);
        let resume_wrapper_ty = make_resume_wrapper_type(ctx, anyref_ty);
        let yield_result_ty = make_yield_result_type(ctx, anyref_ty, shift_info_ty);

        Self {
            yield_result: yield_result_ty,
            shift_info: shift_info_ty,
            continuation: continuation_ty,
            resume_wrapper: resume_wrapper_ty,
            anyref: anyref_ty,
            ptr: ptr_ty,
            i32: i32_ty,
            i1: i1_ty,
        }
    }
}

// ============================================================================
// Type Construction Helpers
// ============================================================================

/// Intern `core.ptr` (raw pointer, not RC-managed).
fn intern_ptr(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build())
}

/// Intern `core.i32`.
fn intern_i32(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Intern `core.i1`.
fn intern_i1(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build())
}

/// Create `adt.struct @ShiftInfo { value: anyref, prompt: i32, op_idx: i32, continuation: anyref }`.
fn make_shift_info_type(ctx: &mut IrContext, anyref: TypeRef, i32_ty: TypeRef) -> TypeRef {
    adt_struct_type(
        ctx,
        Symbol::new("@ShiftInfo"),
        &[
            (Symbol::new("value"), anyref),
            (Symbol::new("prompt"), i32_ty),
            (Symbol::new("op_idx"), i32_ty),
            (Symbol::new("continuation"), anyref),
        ],
    )
}

/// Create `adt.struct @Continuation { resume_fn: core.ptr, state: anyref }`.
///
/// `resume_fn` is typed as `core.ptr` (raw pointer) so the RC insertion pass
/// does NOT add retain/release operations for function pointers. On native,
/// `func.constant` produces a code address that must not be treated as a
/// heap-allocated RC object.
fn make_continuation_type(ctx: &mut IrContext, ptr_ty: TypeRef, anyref: TypeRef) -> TypeRef {
    adt_struct_type(
        ctx,
        Symbol::new("@Continuation"),
        &[
            (Symbol::new("resume_fn"), ptr_ty),
            (Symbol::new("state"), anyref),
        ],
    )
}

/// Create `adt.struct @ResumeWrapper { state: anyref, resume_value: anyref }`.
fn make_resume_wrapper_type(ctx: &mut IrContext, anyref: TypeRef) -> TypeRef {
    adt_struct_type(
        ctx,
        Symbol::new("@ResumeWrapper"),
        &[
            (Symbol::new("state"), anyref),
            (Symbol::new("resume_value"), anyref),
        ],
    )
}

/// Create `adt.enum @YieldResult { Done(anyref), Shift(ShiftInfo) }`.
fn make_yield_result_type(ctx: &mut IrContext, anyref: TypeRef, shift_info: TypeRef) -> TypeRef {
    adt_enum_type(
        ctx,
        Symbol::new("@YieldResult"),
        &[
            (Symbol::new("Done"), vec![anyref]),
            (Symbol::new("Shift"), vec![shift_info]),
        ],
    )
}

// ============================================================================
// ADT Type Builders (shared with other modules)
// ============================================================================

/// Create an `adt.struct` type with named fields.
pub(crate) fn adt_struct_type(
    ctx: &mut IrContext,
    name: Symbol,
    fields: &[(Symbol, TypeRef)],
) -> TypeRef {
    let fields_attr = Attribute::List(
        fields
            .iter()
            .map(|(fname, fty)| {
                Attribute::List(vec![Attribute::Symbol(*fname), Attribute::Type(*fty)])
            })
            .collect(),
    );

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", Attribute::Symbol(name))
            .attr("fields", fields_attr)
            .build(),
    )
}

/// Create an `adt.enum` type with named variants.
pub(crate) fn adt_enum_type(
    ctx: &mut IrContext,
    name: Symbol,
    variants: &[(Symbol, Vec<TypeRef>)],
) -> TypeRef {
    let variants_attr = Attribute::List(
        variants
            .iter()
            .map(|(vname, field_tys)| {
                Attribute::List(vec![
                    Attribute::Symbol(*vname),
                    Attribute::List(field_tys.iter().map(|ty| Attribute::Type(*ty)).collect()),
                ])
            })
            .collect(),
    );

    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("enum"))
            .attr("name", Attribute::Symbol(name))
            .attr("variants", variants_attr)
            .build(),
    )
}

/// Check if a type is the YieldResult enum type.
pub(crate) fn is_yield_result_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") || data.name != Symbol::new("enum") {
        return false;
    }
    matches!(
        data.attrs.get(&Symbol::new("name")),
        Some(Attribute::Symbol(s)) if *s == Symbol::new("@YieldResult")
    )
}
