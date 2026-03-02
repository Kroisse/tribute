//! WASM type converter for arena IR-level type transformations.
//!
//! This module provides an `ArenaTypeConverter` configuration for converting
//! high-level Tribute types to their WASM representations during IR lowering.
//!
//! ## Type Conversion Rules
//!
//! | Source Type         | Target Type     | Notes                              |
//! |---------------------|-----------------|-------------------------------------|
//! | `tribute_rt.int`    | `core.i32`      | Arbitrary precision -> i32 (Phase 1) |
//! | `tribute_rt.nat`    | `core.i32`      | Arbitrary precision -> i32 (Phase 1) |
//! | `tribute_rt.bool`   | `core.i32`      | Boolean as i32                      |
//! | `core.i1`           | `core.i32`      | WASM doesn't have i1                |
//! | `tribute_rt.float`  | `core.f64`      | Float as f64                        |
//! | `tribute_rt.intref` | `wasm.i31ref`   | Boxed integer reference             |
//! | `tribute_rt.any`    | `wasm.anyref`   | Any reference type                  |
//! | `adt.typeref<T>`    | `wasm.structref`| Generic struct reference            |
//!
//! ## Materializations
//!
//! When values need to be bridged between different struct types (e.g., from
//! a base enum type to a variant type), the converter can insert `wasm.ref_cast`
//! operations.

use tribute_ir::arena::dialect::ability::{is_evidence_type_ref, is_marker_type_ref};
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::type_converter::{ArenaTypeConverter, MaterializeResult};
use trunk_ir::arena::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir::ir::Symbol;

// =============================================================================
// Helper: intern a simple type (no params, no attrs)
// =============================================================================

fn intern_type(ctx: &mut IrContext, dialect: Symbol, name: Symbol) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(dialect, name).build())
}

// =============================================================================
// Helper: check if a TypeRef matches a given dialect.name
// =============================================================================

fn is_type(ctx: &IrContext, ty: TypeRef, dialect: Symbol, name: Symbol) -> bool {
    ctx.types.is_dialect(ty, dialect, name)
}

// =============================================================================
// ADT struct type constructor helper
// =============================================================================

fn make_adt_struct_type(
    ctx: &mut IrContext,
    name: Symbol,
    fields: Vec<(Symbol, TypeRef)>,
) -> TypeRef {
    let fields_attr = Attribute::List(
        fields
            .into_iter()
            .map(|(field_name, field_type)| {
                Attribute::List(vec![
                    Attribute::Symbol(field_name),
                    Attribute::Type(field_type),
                ])
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

// =============================================================================
// ADT type helpers (arena versions)
// =============================================================================

/// Get the canonical Closure ADT type (arena version).
///
/// Layout: (table_idx: i32, env: anyref)
///
/// IMPORTANT: Must use wasm.anyref (not tribute_rt.any) to ensure consistent
/// type identity for emit lookups. This matches the pattern used by step_adt_type.
pub fn closure_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = intern_type(ctx, Symbol::new("core"), Symbol::new("i32"));
    let anyref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("anyref"));

    make_adt_struct_type(
        ctx,
        Symbol::new("_closure"),
        vec![
            (Symbol::new("table_idx"), i32_ty),
            (Symbol::new("env"), anyref_ty),
        ],
    )
}

// Re-export marker_adt_type_ref from ability module for backward compatibility
pub use tribute_ir::arena::dialect::ability::marker_adt_type_ref as marker_adt_type;

/// Get the canonical Evidence ADT type for WASM representation (arena version).
///
/// At the WASM level, evidence is represented as `wasm.arrayref`.
/// This is distinct from `ability::evidence_adt_type_ref()` which returns
/// `core.array(Marker)` for high-level IR representation.
pub fn evidence_wasm_type(ctx: &mut IrContext) -> TypeRef {
    intern_type(ctx, Symbol::new("wasm"), Symbol::new("arrayref"))
}

/// Get the canonical Step ADT type (arena version).
///
/// Layout: (tag: i32, value: anyref, prompt: i32, op_idx: i32)
///
/// IMPORTANT: Must use wasm.anyref to match step_marker_type in gc_types.rs.
pub fn step_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = intern_type(ctx, Symbol::new("core"), Symbol::new("i32"));
    let anyref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("anyref"));

    make_adt_struct_type(
        ctx,
        Symbol::new("_Step"),
        vec![
            (Symbol::new("tag"), i32_ty),
            (Symbol::new("value"), anyref_ty),
            (Symbol::new("prompt"), i32_ty),
            (Symbol::new("op_idx"), i32_ty),
        ],
    )
}

/// Get the canonical Continuation ADT type (arena version).
///
/// Layout: (resume_fn: i32, state: anyref, tag: i32, shift_value: anyref)
///
/// resume_fn is stored as i32 (function table index), same as closures.
/// Uses wasm.anyref for consistency with step_adt_type.
pub fn continuation_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = intern_type(ctx, Symbol::new("core"), Symbol::new("i32"));
    let anyref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("anyref"));

    make_adt_struct_type(
        ctx,
        Symbol::new("_Continuation"),
        vec![
            (Symbol::new("resume_fn"), i32_ty),
            (Symbol::new("state"), anyref_ty),
            (Symbol::new("tag"), i32_ty),
            (Symbol::new("shift_value"), anyref_ty),
        ],
    )
}

/// Get the canonical ResumeWrapper ADT type (arena version).
///
/// Layout: (state: anyref, resume_value: anyref)
///
/// Uses wasm.anyref for consistency with step_adt_type.
pub fn resume_wrapper_adt_type(ctx: &mut IrContext) -> TypeRef {
    let anyref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("anyref"));

    make_adt_struct_type(
        ctx,
        Symbol::new("_ResumeWrapper"),
        vec![
            (Symbol::new("state"), anyref_ty),
            (Symbol::new("resume_value"), anyref_ty),
        ],
    )
}

// =============================================================================
// Type inspection helpers
// =============================================================================

/// Check if a type is an `adt.struct` type.
fn is_adt_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, Symbol::new("adt"), Symbol::new("struct"))
}

/// Check if a type is an `adt.typeref` type.
fn is_adt_typeref(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, Symbol::new("adt"), Symbol::new("typeref"))
}

/// Check if a type has the `is_variant` attribute set to true.
fn is_variant_instance_type(ctx: &IrContext, ty: TypeRef) -> bool {
    matches!(
        ctx.types.get(ty).attrs.get(&Symbol::new("is_variant")),
        Some(Attribute::Bool(true))
    )
}

/// Check if a type is a struct-like reference type.
///
/// This includes `wasm.structref`, `wasm.anyref`, ADT struct/typeref types,
/// variant instance types, and trampoline types that get lowered to ADT structs.
fn is_struct_like(ctx: &IrContext, ty: TypeRef) -> bool {
    // wasm.structref or wasm.anyref
    if is_type(ctx, ty, Symbol::new("wasm"), Symbol::new("structref"))
        || is_type(ctx, ty, Symbol::new("wasm"), Symbol::new("anyref"))
    {
        return true;
    }

    // adt.typeref (generic struct references)
    if is_adt_typeref(ctx, ty) {
        return true;
    }

    // adt.struct (concrete struct types like _ResumeWrapper, _Continuation, _Step)
    if is_adt_struct_type(ctx, ty) {
        return true;
    }

    // Check for variant instance types (have is_variant attribute)
    if is_variant_instance_type(ctx, ty) {
        return true;
    }

    // trampoline types that get lowered to ADT structs
    if is_type(ctx, ty, Symbol::new("trampoline"), Symbol::new("step"))
        || is_type(
            ctx,
            ty,
            Symbol::new("trampoline"),
            Symbol::new("continuation"),
        )
        || is_type(
            ctx,
            ty,
            Symbol::new("trampoline"),
            Symbol::new("resume_wrapper"),
        )
    {
        return true;
    }

    false
}

/// Check if a type is a `closure.closure` type.
fn is_closure_type(ctx: &IrContext, ty: TypeRef) -> bool {
    is_type(ctx, ty, Symbol::new("closure"), Symbol::new("closure"))
}

// =============================================================================
// Boxing / Unboxing helpers
// =============================================================================

/// Helper to generate i31 boxing operations (ref_i31 + explicit upcast to anyref).
///
/// This is used when converting i32/i64 values to anyref for polymorphic function calls.
/// WasmGC's i31ref is a subtype of anyref, but an explicit `ref_cast` is emitted for IR type correctness.
/// If the input is i64, an `i32_wrap_i64` truncation is emitted first.
fn box_via_i31(
    ctx: &mut IrContext,
    loc: Location,
    value: ValueRef,
    from_ty: TypeRef,
    i31ref_ty: TypeRef,
    anyref_ty: TypeRef,
    i32_ty: TypeRef,
) -> Option<MaterializeResult> {
    let is_i64 = is_type(ctx, from_ty, Symbol::new("core"), Symbol::new("i64"));
    let mut ops: Vec<OpRef> = Vec::new();

    let val = if is_i64 {
        let wrap = arena_wasm::i32_wrap_i64(ctx, loc, value, i32_ty);
        ops.push(wrap.op_ref());
        wrap.result(ctx)
    } else {
        value
    };

    let ref_op = arena_wasm::ref_i31(ctx, loc, val, i31ref_ty);
    ops.push(ref_op.op_ref());
    let i31_val = ref_op.result(ctx);

    // Upcast i31ref -> anyref (needed for IR type correctness)
    let upcast = arena_wasm::ref_cast(ctx, loc, i31_val, anyref_ty, Symbol::new("anyref"), None);
    ops.push(upcast.op_ref());

    Some(MaterializeResult {
        value: upcast.result(ctx),
        ops,
    })
}

/// Helper to generate i31 unboxing operations (ref_cast to i31ref + i31_get_s).
///
/// This is used when converting anyref-typed values back to i32, such as
/// extracting values from Step structs which store all values as anyref.
fn unbox_via_i31(
    ctx: &mut IrContext,
    loc: Location,
    value: ValueRef,
    i31ref_ty: TypeRef,
    i32_ty: TypeRef,
) -> Option<MaterializeResult> {
    // Cast anyref to i31ref
    let cast_op = arena_wasm::ref_cast(ctx, loc, value, i31ref_ty, Symbol::new("i31ref"), None);
    let cast_val = cast_op.result(ctx);

    // Extract i32 from i31ref
    let get_op = arena_wasm::i31_get_s(ctx, loc, cast_val, i32_ty);

    Some(MaterializeResult {
        value: get_op.result(ctx),
        ops: vec![cast_op.op_ref(), get_op.op_ref()],
    })
}

// =============================================================================
// Main entry point
// =============================================================================

/// Create an ArenaTypeConverter configured for WASM backend type conversions.
///
/// This converter handles the IR-level type transformations needed during
/// lowering passes. It complements the emit-phase `gc_types::type_to_field_type`
/// by performing conversions at the IR level.
pub fn wasm_type_converter(ctx: &mut IrContext) -> ArenaTypeConverter {
    // Pre-compute commonly used types (TypeRef is Copy)
    let i32_ty = intern_type(ctx, Symbol::new("core"), Symbol::new("i32"));
    let f64_ty = intern_type(ctx, Symbol::new("core"), Symbol::new("f64"));
    let anyref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("anyref"));
    let i31ref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("i31ref"));
    let structref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("structref"));
    let arrayref_ty = intern_type(ctx, Symbol::new("wasm"), Symbol::new("arrayref"));
    let closure_ty = closure_adt_type(ctx);
    let step_ty = step_adt_type(ctx);
    let cont_ty = continuation_adt_type(ctx);
    let rw_ty = resume_wrapper_adt_type(ctx);
    let evidence_ty = evidence_wasm_type(ctx);
    let marker_ty = marker_adt_type(ctx);

    let mut tc = ArenaTypeConverter::new();

    // =========================================================================
    // Type conversions
    // =========================================================================

    // Convert tribute_rt.int -> core.i32 (Phase 1: arbitrary precision as i32)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("tribute_rt"), Symbol::new("int")) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.nat -> core.i32 (Phase 1: arbitrary precision as i32)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("tribute_rt"), Symbol::new("nat")) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.bool -> core.i32 (boolean as i32)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("tribute_rt"), Symbol::new("bool")) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert core.i1 -> core.i32 (WASM doesn't have i1, use i32)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("core"), Symbol::new("i1")) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.float -> core.f64 (float as f64)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("tribute_rt"), Symbol::new("float")) {
            Some(f64_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.intref -> wasm.i31ref (boxed integer reference)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("tribute_rt"), Symbol::new("intref")) {
            Some(i31ref_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.any -> wasm.anyref (any reference type)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("tribute_rt"), Symbol::new("any")) {
            Some(anyref_ty)
        } else {
            None
        }
    });

    // Convert trampoline.step -> _Step ADT
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("trampoline"), Symbol::new("step")) {
            Some(step_ty)
        } else {
            None
        }
    });

    // Convert trampoline.continuation -> _Continuation ADT
    tc.add_conversion(move |ctx, ty| {
        if is_type(
            ctx,
            ty,
            Symbol::new("trampoline"),
            Symbol::new("continuation"),
        ) {
            Some(cont_ty)
        } else {
            None
        }
    });

    // Convert trampoline.resume_wrapper -> _ResumeWrapper ADT
    tc.add_conversion(move |ctx, ty| {
        if is_type(
            ctx,
            ty,
            Symbol::new("trampoline"),
            Symbol::new("resume_wrapper"),
        ) {
            Some(rw_ty)
        } else {
            None
        }
    });

    // Convert adt.typeref -> wasm.structref (generic struct reference)
    tc.add_conversion(move |ctx, ty| {
        if is_adt_typeref(ctx, ty) {
            Some(structref_ty)
        } else {
            None
        }
    });

    // Convert closure.closure -> adt.struct(name="_closure")
    // This ensures emit can identify closures by ADT name rather than tribute-ir type
    tc.add_conversion(move |ctx, ty| {
        if is_closure_type(ctx, ty) {
            Some(closure_ty)
        } else {
            None
        }
    });

    // Convert evidence ADT type (core.array(Marker)) -> wasm.arrayref
    tc.add_conversion(move |ctx, ty| {
        if is_evidence_type_ref(ctx, ty) {
            Some(evidence_ty)
        } else {
            None
        }
    });

    // Convert generic core.array -> wasm.arrayref
    // This handles array types that are not evidence types (e.g., user arrays)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("core"), Symbol::new("array")) {
            Some(arrayref_ty)
        } else {
            None
        }
    });

    // Convert cont.prompt_tag -> core.i32 (same representation at runtime)
    tc.add_conversion(move |ctx, ty| {
        if is_type(ctx, ty, Symbol::new("cont"), Symbol::new("prompt_tag")) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert marker ADT type -> pass through (already standard ADT struct)
    // This conversion is a no-op since marker_adt_type() returns a standard adt.struct
    tc.add_conversion(move |ctx, ty| {
        if is_marker_type_ref(ctx, ty) {
            Some(marker_ty)
        } else {
            None
        }
    });

    // =========================================================================
    // Single materializer combining all materialization rules
    // =========================================================================

    tc.set_materializer(move |ctx, location, value, from_ty, to_ty| {
        // Same type - no materialization needed
        if from_ty == to_ty {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // Struct-like bridging materializations
        // -----------------------------------------------------------------

        // adt.typeref -> wasm.structref is a safe upcast (no-op).
        // structref -> adt.typeref is a downcast and needs ref.cast (handled below).
        let from_is_typeref = is_adt_typeref(ctx, from_ty);
        let to_is_structref = is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("structref"));

        if from_is_typeref && to_is_structref {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // Both are struct-like types but need actual bridging - generate ref_cast
        let from_is_struct_like = is_struct_like(ctx, from_ty);
        let to_is_struct_like = is_struct_like(ctx, to_ty);

        if from_is_struct_like && to_is_struct_like {
            // Skip ref_cast for safe upcasts to abstract supertypes
            // (e.g., concrete struct -> anyref, concrete struct -> structref).
            let to_is_anyref = is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"));
            if to_is_anyref || to_is_structref {
                return Some(MaterializeResult { value, ops: vec![] });
            }

            let cast_op =
                arena_wasm::ref_cast(ctx, location, value, to_ty, Symbol::new("struct"), None);
            return Some(MaterializeResult {
                value: cast_op.result(ctx),
                ops: vec![cast_op.op_ref()],
            });
        }

        // anyref -> concrete struct type: use ref_cast
        // This handles cases like closure env or resume wrapper parameters
        // that are passed as anyref for uniform calling convention.
        // We EXCLUDE wasm.anyref as target since that's handled by primitive equivalence.
        let from_is_anyref = is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("anyref"))
            || is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("any"));
        let to_is_abstract_anyref = is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"));
        if from_is_anyref && to_is_struct_like && !to_is_abstract_anyref {
            // Trampoline types must already be converted to ADT types by add_conversion
            // rules before materialization runs (convert_type is applied to to_ty in
            // resolve_unrealized_casts before calling materialize).
            debug_assert!(
                !is_type(
                    ctx,
                    to_ty,
                    Symbol::new("trampoline"),
                    Symbol::new("resume_wrapper")
                ) && !is_type(ctx, to_ty, Symbol::new("trampoline"), Symbol::new("step"))
                    && !is_type(
                        ctx,
                        to_ty,
                        Symbol::new("trampoline"),
                        Symbol::new("continuation")
                    ),
                "ICE: trampoline type reached materialization without conversion"
            );
            let cast_op =
                arena_wasm::ref_cast(ctx, location, value, to_ty, Symbol::new("struct"), None);
            return Some(MaterializeResult {
                value: cast_op.result(ctx),
                ops: vec![cast_op.op_ref()],
            });
        }

        // -----------------------------------------------------------------
        // Primitive type equivalences (no-op materializations)
        // -----------------------------------------------------------------

        // tribute_rt.int -> core.i32 (same representation)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("int"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("i32"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.nat -> core.i32 (same representation)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("nat"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("i32"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.bool -> core.i32 (same representation)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("bool"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("i32"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // core.i1 -> core.i32 (same representation for wasm)
        if is_type(ctx, from_ty, Symbol::new("core"), Symbol::new("i1"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("i32"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.float -> core.f64 (same representation)
        if is_type(
            ctx,
            from_ty,
            Symbol::new("tribute_rt"),
            Symbol::new("float"),
        ) && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("f64"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.intref -> wasm.i31ref (same representation)
        if is_type(
            ctx,
            from_ty,
            Symbol::new("tribute_rt"),
            Symbol::new("intref"),
        ) && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("i31ref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.any -> wasm.anyref (same representation)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("any"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // wasm.i31ref -> wasm.anyref (i31ref is a subtype of anyref)
        if is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("i31ref"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // Boxing materializations (emit ops)
        // -----------------------------------------------------------------

        // core.i32 -> wasm.anyref (box via i31)
        if is_type(ctx, from_ty, Symbol::new("core"), Symbol::new("i32"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return box_via_i31(ctx, location, value, from_ty, i31ref_ty, anyref_ty, i32_ty);
        }
        // core.i64 -> wasm.anyref (box via i31, with i32 truncation)
        if is_type(ctx, from_ty, Symbol::new("core"), Symbol::new("i64"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return box_via_i31(ctx, location, value, from_ty, i31ref_ty, anyref_ty, i32_ty);
        }
        // core.nil -> wasm.anyref (nil is represented as null reference)
        if is_type(ctx, from_ty, Symbol::new("core"), Symbol::new("nil"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            let null_op =
                arena_wasm::ref_null(ctx, location, anyref_ty, Symbol::new("anyref"), None);
            return Some(MaterializeResult {
                value: null_op.result(ctx),
                ops: vec![null_op.op_ref()],
            });
        }

        // -----------------------------------------------------------------
        // Subtype no-ops (continued)
        // -----------------------------------------------------------------

        // wasm.structref -> wasm.anyref (structref is a subtype of anyref)
        if is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("structref"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // wasm.arrayref -> wasm.anyref (arrayref is a subtype of anyref in WasmGC)
        if is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("arrayref"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // adt.struct -> wasm.anyref (GC struct is a subtype of anyref)
        if is_adt_struct_type(ctx, from_ty)
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // closure.closure -> adt.struct(name="_closure") (same representation)
        if is_closure_type(ctx, from_ty) && to_ty == closure_ty {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // closure.closure -> wasm.structref (subtype relationship)
        if is_closure_type(ctx, from_ty)
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("structref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // Unboxing materializations
        // -----------------------------------------------------------------

        // wasm.anyref -> core.i32 (unbox via i31)
        if is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("anyref"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("i32"))
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }
        // tribute_rt.any -> core.i32 (unbox via i31, same as wasm.anyref)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("any"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("i32"))
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }
        // wasm.anyref -> tribute_rt.int (unbox via i31)
        if is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("anyref"))
            && is_type(ctx, to_ty, Symbol::new("tribute_rt"), Symbol::new("int"))
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }
        // tribute_rt.any -> tribute_rt.int (unbox via i31)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("any"))
            && is_type(ctx, to_ty, Symbol::new("tribute_rt"), Symbol::new("int"))
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }

        // -----------------------------------------------------------------
        // Pointer / nil equivalences
        // -----------------------------------------------------------------

        // wasm.anyref -> core.ptr (treat pointer as anyref subtype)
        if is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("anyref"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("ptr"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.any -> core.ptr (same representation in WasmGC)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("any"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("ptr"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // wasm.anyref -> core.nil (unit type, value ignored)
        if is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("anyref"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("nil"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // tribute_rt.any -> core.nil (unit type, value ignored)
        if is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("any"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("nil"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // Trampoline type equivalences
        // -----------------------------------------------------------------

        // trampoline.step -> _Step ADT (same representation after conversion)
        if is_type(ctx, from_ty, Symbol::new("trampoline"), Symbol::new("step")) && to_ty == step_ty
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // trampoline.continuation -> _Continuation ADT (same representation)
        if is_type(
            ctx,
            from_ty,
            Symbol::new("trampoline"),
            Symbol::new("continuation"),
        ) && to_ty == cont_ty
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // trampoline.resume_wrapper -> _ResumeWrapper ADT (same representation)
        if is_type(
            ctx,
            from_ty,
            Symbol::new("trampoline"),
            Symbol::new("resume_wrapper"),
        ) && to_ty == rw_ty
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // Evidence / marker equivalences
        // -----------------------------------------------------------------

        // evidence (core.array(Marker)) -> wasm.anyref (same representation)
        if is_evidence_type_ref(ctx, from_ty)
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("anyref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // marker (adt.struct) -> wasm.structref (marker is a struct)
        if is_marker_type_ref(ctx, from_ty)
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("structref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // cont.prompt_tag -> core.i32 (same representation at runtime)
        if is_type(ctx, from_ty, Symbol::new("cont"), Symbol::new("prompt_tag"))
            && is_type(ctx, to_ty, Symbol::new("core"), Symbol::new("i32"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }
        // core.array -> wasm.arrayref (same representation)
        if is_type(ctx, from_ty, Symbol::new("core"), Symbol::new("array"))
            && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("arrayref"))
        {
            return Some(MaterializeResult { value, ops: vec![] });
        }

        // -----------------------------------------------------------------
        // anyref -> arrayref materialization (requires ref_cast)
        // -----------------------------------------------------------------

        let from_is_any = is_type(ctx, from_ty, Symbol::new("wasm"), Symbol::new("anyref"))
            || is_type(ctx, from_ty, Symbol::new("tribute_rt"), Symbol::new("any"));
        if from_is_any && is_type(ctx, to_ty, Symbol::new("wasm"), Symbol::new("arrayref")) {
            let cast_op = arena_wasm::ref_cast(
                ctx,
                location,
                value,
                arrayref_ty,
                Symbol::new("arrayref"),
                None,
            );
            return Some(MaterializeResult {
                value: cast_op.result(ctx),
                ops: vec![cast_op.op_ref()],
            });
        }

        // Cannot materialize this conversion
        None
    });

    tc
}

// =============================================================================
// Salsa-based type converter (for resolve_unrealized_casts in pipeline.rs)
// =============================================================================

/// Salsa-based WASM type converter for `resolve_unrealized_casts`.
///
/// This is the Salsa-level counterpart of `wasm_type_converter()` (arena).
/// It provides type conversion and materialization rules for resolving
/// `core.unrealized_conversion_cast` operations at the Salsa IR level.
pub mod salsa_converter {
    use tribute_ir::dialect::{ability, closure, tribute_rt};
    use trunk_ir::dialect::{adt, cont, trampoline};
    use trunk_ir::dialect::{core, wasm};
    use trunk_ir::ir::Symbol;
    use trunk_ir::rewrite::{MaterializeResult, OpVec, TypeConverter};
    use trunk_ir::{DialectOp, DialectType, Type};

    /// Get the canonical Closure ADT type (Salsa version).
    pub fn closure_adt_type(db: &dyn salsa::Database) -> Type<'_> {
        let i32_ty = core::I32::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        adt::struct_type(
            db,
            Symbol::new("_closure"),
            vec![
                (Symbol::new("table_idx"), i32_ty),
                (Symbol::new("env"), anyref_ty),
            ],
        )
    }

    /// Get the evidence WASM type (Salsa version).
    pub fn evidence_wasm_type(db: &dyn salsa::Database) -> Type<'_> {
        wasm::Arrayref::new(db).as_type()
    }

    /// Step ADT type (Salsa version).
    /// Layout must match the canonical definition in gc_types.rs and the arena version.
    fn step_adt_type(db: &dyn salsa::Database) -> Type<'_> {
        let i32_ty = core::I32::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();
        adt::struct_type(
            db,
            Symbol::new("_Step"),
            vec![
                (Symbol::new("tag"), i32_ty),
                (Symbol::new("value"), anyref_ty),
                (Symbol::new("prompt"), i32_ty),
                (Symbol::new("op_idx"), i32_ty),
            ],
        )
    }

    /// Continuation ADT type (Salsa version).
    /// Delegates to the canonical definition in gc_types.
    fn continuation_adt_type(db: &dyn salsa::Database) -> Type<'_> {
        trunk_ir_wasm_backend::gc_types::continuation_adt_type(db)
    }

    /// ResumeWrapper ADT type (Salsa version).
    /// Delegates to the canonical definition in gc_types.
    fn resume_wrapper_adt_type(db: &dyn salsa::Database) -> Type<'_> {
        trunk_ir_wasm_backend::gc_types::resume_wrapper_adt_type(db)
    }

    /// Re-export Salsa marker_adt_type.
    pub fn marker_adt_type(db: &dyn salsa::Database) -> Type<'_> {
        ability::marker_adt_type(db)
    }

    fn box_via_i31<'db>(
        db: &'db dyn salsa::Database,
        location: trunk_ir::Location<'db>,
        value: trunk_ir::Value<'db>,
        from_ty: trunk_ir::Type<'db>,
    ) -> MaterializeResult<'db> {
        let i31ref_ty = wasm::I31ref::new(db).as_type();
        let anyref_ty = wasm::Anyref::new(db).as_type();

        if core::I64::from_type(db, from_ty).is_some() {
            let i32_ty = core::I32::new(db).as_type();
            let wrap_op = wasm::i32_wrap_i64(db, location, value, i32_ty);
            let wrapped = wrap_op.as_operation().result(db, 0);
            let ref_op = wasm::ref_i31(db, location, wrapped, i31ref_ty);
            let i31ref_val = ref_op.as_operation().result(db, 0);
            let upcast = wasm::ref_cast(db, location, i31ref_val, anyref_ty, anyref_ty, None);
            return MaterializeResult::ops([
                wrap_op.as_operation(),
                ref_op.as_operation(),
                upcast.as_operation(),
            ]);
        }

        let ref_op = wasm::ref_i31(db, location, value, i31ref_ty);
        let i31ref_val = ref_op.as_operation().result(db, 0);
        let upcast = wasm::ref_cast(db, location, i31ref_val, anyref_ty, anyref_ty, None);
        MaterializeResult::ops([ref_op.as_operation(), upcast.as_operation()])
    }

    fn unbox_via_i31<'db>(
        db: &'db dyn salsa::Database,
        location: trunk_ir::Location<'db>,
        value: trunk_ir::Value<'db>,
    ) -> MaterializeResult<'db> {
        let i31ref_ty = wasm::I31ref::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let cast_op = wasm::ref_cast(db, location, value, i31ref_ty, i31ref_ty, None);
        let get_op = wasm::i31_get_s(db, location, cast_op.as_operation().result(db, 0), i32_ty);
        let mut ops = OpVec::new();
        ops.push(cast_op.as_operation());
        ops.push(get_op.as_operation());
        MaterializeResult::Ops(ops)
    }

    fn is_struct_like(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
        if wasm::Structref::from_type(db, ty).is_some() || wasm::Anyref::from_type(db, ty).is_some()
        {
            return true;
        }
        if adt::is_typeref(db, ty) {
            return true;
        }
        if adt::is_struct_type(db, ty) {
            return true;
        }
        if adt::is_variant_instance_type(db, ty) {
            return true;
        }
        if trampoline::Step::from_type(db, ty).is_some()
            || trampoline::Continuation::from_type(db, ty).is_some()
            || trampoline::ResumeWrapper::from_type(db, ty).is_some()
        {
            return true;
        }
        false
    }

    /// Create a Salsa TypeConverter configured for WASM backend type conversions.
    ///
    /// Used by `resolve_unrealized_casts` in the pipeline to resolve casts
    /// that were inserted by earlier Salsa-based passes (boxing, etc.).
    pub fn wasm_type_converter() -> TypeConverter {
        TypeConverter::new()
            .add_conversion(|db, ty| {
                tribute_rt::Int::from_type(db, ty).map(|_| core::I32::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                tribute_rt::Nat::from_type(db, ty).map(|_| core::I32::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                tribute_rt::Bool::from_type(db, ty).map(|_| core::I32::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                core::I::<1>::from_type(db, ty).map(|_| core::I32::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                tribute_rt::Float::from_type(db, ty).map(|_| core::F64::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                tribute_rt::Intref::from_type(db, ty).map(|_| wasm::I31ref::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                tribute_rt::Any::from_type(db, ty).map(|_| wasm::Anyref::new(db).as_type())
            })
            .add_conversion(|db, ty| trampoline::Step::from_type(db, ty).map(|_| step_adt_type(db)))
            .add_conversion(|db, ty| {
                trampoline::Continuation::from_type(db, ty).map(|_| continuation_adt_type(db))
            })
            .add_conversion(|db, ty| {
                trampoline::ResumeWrapper::from_type(db, ty).map(|_| resume_wrapper_adt_type(db))
            })
            .add_conversion(|db, ty| {
                if adt::is_typeref(db, ty) {
                    Some(wasm::Structref::new(db).as_type())
                } else {
                    None
                }
            })
            .add_conversion(|db, ty| {
                if closure::Closure::from_type(db, ty).is_some() {
                    Some(closure_adt_type(db))
                } else {
                    None
                }
            })
            .add_conversion(|db, ty| {
                if ability::is_evidence_type(db, ty) {
                    Some(evidence_wasm_type(db))
                } else {
                    None
                }
            })
            .add_conversion(|db, ty| {
                core::Array::from_type(db, ty).map(|_| wasm::Arrayref::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                cont::PromptTag::from_type(db, ty).map(|_| core::I32::new(db).as_type())
            })
            .add_conversion(|db, ty| {
                if ability::is_marker_type(db, ty) {
                    Some(marker_adt_type(db))
                } else {
                    None
                }
            })
            // Materialization for struct type bridging
            .add_materialization(|db, location, value, from_ty, to_ty| {
                if from_ty == to_ty {
                    return MaterializeResult::NoOp;
                }
                let from_is_struct_like = is_struct_like(db, from_ty);
                let to_is_struct_like = is_struct_like(db, to_ty);
                // Safe upcasts to abstract ref types need no runtime cast
                let to_is_abstract = wasm::Structref::from_type(db, to_ty).is_some()
                    || wasm::Anyref::from_type(db, to_ty).is_some();
                if to_is_abstract && from_is_struct_like {
                    return MaterializeResult::NoOp;
                }
                // Downcasts and lateral casts between struct-like types need ref_cast
                if from_is_struct_like && to_is_struct_like {
                    let cast_op = wasm::ref_cast(db, location, value, to_ty, to_ty, None);
                    let mut ops = OpVec::new();
                    ops.push(cast_op.as_operation());
                    return MaterializeResult::Ops(ops);
                }
                let from_is_anyref = wasm::Anyref::from_type(db, from_ty).is_some()
                    || tribute_rt::Any::from_type(db, from_ty).is_some();
                let to_is_abstract_anyref = wasm::Anyref::from_type(db, to_ty).is_some();
                if from_is_anyref && to_is_struct_like && !to_is_abstract_anyref {
                    debug_assert!(
                        trampoline::ResumeWrapper::from_type(db, to_ty).is_none()
                            && trampoline::Step::from_type(db, to_ty).is_none()
                            && trampoline::Continuation::from_type(db, to_ty).is_none(),
                        "ICE: trampoline type {to_ty:?} reached materialization without conversion"
                    );
                    let cast_op = wasm::ref_cast(db, location, value, to_ty, to_ty, None);
                    let mut ops = OpVec::new();
                    ops.push(cast_op.as_operation());
                    return MaterializeResult::Ops(ops);
                }
                MaterializeResult::Skip
            })
            // Primitive type equivalence materializations
            .add_materialization(|db, location, value, from_ty, to_ty| {
                if tribute_rt::Int::from_type(db, from_ty).is_some()
                    && core::I32::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if tribute_rt::Nat::from_type(db, from_ty).is_some()
                    && core::I32::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if tribute_rt::Bool::from_type(db, from_ty).is_some()
                    && core::I32::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if core::I::<1>::from_type(db, from_ty).is_some()
                    && core::I32::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if tribute_rt::Float::from_type(db, from_ty).is_some()
                    && core::F64::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if tribute_rt::Intref::from_type(db, from_ty).is_some()
                    && wasm::I31ref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if tribute_rt::Any::from_type(db, from_ty).is_some()
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if wasm::I31ref::from_type(db, from_ty).is_some()
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if core::I32::from_type(db, from_ty).is_some()
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return box_via_i31(db, location, value, from_ty);
                }
                if core::I64::from_type(db, from_ty).is_some()
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return box_via_i31(db, location, value, from_ty);
                }
                if core::Nil::from_type(db, from_ty).is_some()
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    let anyref_ty = wasm::Anyref::new(db).as_type();
                    let null_op = wasm::ref_null(db, location, anyref_ty, anyref_ty, None);
                    return MaterializeResult::single(null_op.as_operation());
                }
                if wasm::Structref::from_type(db, from_ty).is_some()
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if wasm::Arrayref::from_type(db, from_ty).is_some()
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if adt::is_struct_type(db, from_ty) && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if closure::Closure::from_type(db, from_ty).is_some()
                    && to_ty == closure_adt_type(db)
                {
                    return MaterializeResult::NoOp;
                }
                if closure::Closure::from_type(db, from_ty).is_some()
                    && wasm::Structref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if wasm::Anyref::from_type(db, from_ty).is_some()
                    && core::I32::from_type(db, to_ty).is_some()
                {
                    return unbox_via_i31(db, location, value);
                }
                if tribute_rt::Any::from_type(db, from_ty).is_some()
                    && core::I32::from_type(db, to_ty).is_some()
                {
                    return unbox_via_i31(db, location, value);
                }
                if wasm::Anyref::from_type(db, from_ty).is_some()
                    && tribute_rt::Int::from_type(db, to_ty).is_some()
                {
                    return unbox_via_i31(db, location, value);
                }
                if tribute_rt::Any::from_type(db, from_ty).is_some()
                    && tribute_rt::Int::from_type(db, to_ty).is_some()
                {
                    return unbox_via_i31(db, location, value);
                }
                if wasm::Anyref::from_type(db, from_ty).is_some()
                    && core::Ptr::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if tribute_rt::Any::from_type(db, from_ty).is_some()
                    && core::Ptr::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if wasm::Anyref::from_type(db, from_ty).is_some()
                    && core::Nil::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if tribute_rt::Any::from_type(db, from_ty).is_some()
                    && core::Nil::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if trampoline::Step::from_type(db, from_ty).is_some() && to_ty == step_adt_type(db)
                {
                    return MaterializeResult::NoOp;
                }
                if trampoline::Continuation::from_type(db, from_ty).is_some()
                    && to_ty == continuation_adt_type(db)
                {
                    return MaterializeResult::NoOp;
                }
                if trampoline::ResumeWrapper::from_type(db, from_ty).is_some()
                    && to_ty == resume_wrapper_adt_type(db)
                {
                    return MaterializeResult::NoOp;
                }
                if ability::is_evidence_type(db, from_ty)
                    && wasm::Anyref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if ability::is_marker_type(db, from_ty)
                    && wasm::Structref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if cont::PromptTag::from_type(db, from_ty).is_some()
                    && core::I32::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                if core::Array::from_type(db, from_ty).is_some()
                    && wasm::Arrayref::from_type(db, to_ty).is_some()
                {
                    return MaterializeResult::NoOp;
                }
                MaterializeResult::Skip
            })
            // anyref → arrayref materialization
            .add_materialization(|db, location, value, from_ty, to_ty| {
                let from_is_any = wasm::Anyref::from_type(db, from_ty).is_some()
                    || tribute_rt::Any::from_type(db, from_ty).is_some();
                if from_is_any && wasm::Arrayref::from_type(db, to_ty).is_some() {
                    let arrayref_ty = wasm::Arrayref::new(db).as_type();
                    let cast_op =
                        wasm::ref_cast(db, location, value, arrayref_ty, arrayref_ty, None);
                    let mut ops = OpVec::new();
                    ops.push(cast_op.as_operation());
                    return MaterializeResult::Ops(ops);
                }
                MaterializeResult::Skip
            })
    }
}
