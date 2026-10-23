//! WASM type converter for arena IR-level type transformations.
//!
//! This module provides a `TypeConverter` configuration for converting
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
//! | `core.index`        | `core.i32`      | WasmGC array index width            |
//! | `tribute_rt.float`  | `core.f64`      | Float as f64                        |
//! | `tribute_rt.intref` | `wasm.i31ref`   | Boxed integer reference             |
//! | `tribute_rt.anyref` | `wasm.anyref`   | Any reference type                  |
//! | `adt.typeref<T>`    | `wasm.structref`| Generic struct reference            |
//!
//! ## Materializations
//!
//! When values need to be bridged between different struct types (e.g., from
//! a base enum type to a variant type), the converter can insert `wasm.ref_cast`
//! operations.

use tribute_ir::dialect::ability::is_evidence_type_ref;
use tribute_ir::dialect::adt;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::wasm_gc as wasm_gc_dialect;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::type_converter::{MaterializeResult, TypeConverter};
use trunk_ir::types::{AttributeMap, Location};

// =============================================================================
// Helper: intern a simple type (no params, no attrs)
// =============================================================================

// =============================================================================
// Helper: check if a TypeRef matches a given dialect.name
// =============================================================================

// =============================================================================
// ADT type helpers (arena versions)
// =============================================================================

/// Get the canonical Closure ADT type (arena version).
///
/// Layout: (table_idx: i32, env: anyref)
///
/// IMPORTANT: Must use wasm.anyref (not tribute_rt.anyref) to ensure consistent
/// type identity for emit lookups.
pub fn closure_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = core::I32::type_ref(ctx);
    crate::closure_lower::closure_base_layout(ctx, i32_ty)
}

/// Get the `Described` type: the supertype of every user struct and variant,
/// holding only the runtime type descriptor field.
pub fn described_adt_type(ctx: &mut IrContext) -> TypeRef {
    let i32_ty = core::I32::type_ref(ctx);
    let mut attrs = AttributeMap::new();
    let layout = ctx.string_attr(tribute_ir::runtime_layout::DESCRIBED);
    attrs
        .view_mut(ctx)
        .insert(tribute_ir::runtime_layout::LAYOUT_ATTR, layout);
    adt::Struct::from_fields(ctx, "_described", [("descriptor", i32_ty)], attrs).as_type_ref()
}

/// Get the Evidence type of the WASM representation.
///
/// Evidence is a concrete `wasm_gc.array` through Wasm lowering, never an
/// erased `wasm.arrayref`, so the evidence runtime and its callers agree on
/// one array type. Its elements are `wasm.structref`, as every marker value
/// is: a marker holds evidence, so no structural element type can name it.
pub fn evidence_wasm_type(ctx: &mut IrContext) -> TypeRef {
    let structref = wasm_dialect::Structref::params().intern(ctx).as_type_ref();
    trunk_ir::types::TypeDataBuilder::new(trunk_ir::dialect::wasm_gc::Array::NAME)
        .param(structref)
        .intern(ctx)
}

/// Replace the evidence type with its GC array everywhere in `module`, once
/// no `adt` operation reads it.
pub(crate) fn convert_evidence_arrays(ctx: &mut IrContext, module: trunk_ir::rewrite::Module) {
    let evidence = evidence_wasm_type(ctx);
    crate::closure_lower::substitute_module_types(ctx, module, |ctx, ty| {
        is_evidence_type_ref(ctx, ty).then_some(evidence)
    });
}

/// The GC array type that Wasm operations use for `ty`: an `adt.array`
/// becomes the `wasm_gc.array` of the same element, keeping its runtime
/// [`layout`](tribute_ir::runtime_layout::LAYOUT_ATTR) identifier, and any
/// other type is returned unchanged.
pub fn gc_array_type(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    use tribute_ir::runtime_layout::LAYOUT_ATTR;

    if is_evidence_type_ref(ctx, ty) {
        return evidence_wasm_type(ctx);
    }
    let Some(array) = tribute_ir::dialect::adt::Array::from_type_ref(ctx, ty) else {
        return ty;
    };
    let element = array.element(ctx);
    let layout = ctx.type_attrs(ty).get_string_ref(LAYOUT_ATTR);
    let mut builder =
        trunk_ir::types::TypeDataBuilder::new(trunk_ir::dialect::wasm_gc::Array::NAME)
            .param(element);
    if let Some(layout) = layout {
        builder = builder.attr(LAYOUT_ATTR, trunk_ir::types::Attribute::String(layout));
    }
    builder.intern(ctx)
}

// =============================================================================
// Type inspection helpers
// =============================================================================

/// Check if a type is an `adt.struct` type.
fn is_adt_struct_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::adt::Struct::matches(ctx, ty)
}

/// Check if a type is an `adt.typeref` type.
fn is_adt_typeref(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::adt::Typeref::matches(ctx, ty)
}

/// Check if a type is a struct-like reference type.
///
/// This includes `wasm.structref`, `wasm.anyref`, ADT struct/typeref types,
/// and the structural `wasm_gc.struct` of user structs and variants.
fn is_struct_like(ctx: &IrContext, ty: TypeRef) -> bool {
    // wasm.structref or wasm.anyref
    if trunk_ir::dialect::wasm::Structref::matches(ctx, ty)
        || trunk_ir::dialect::wasm::Anyref::matches(ctx, ty)
    {
        return true;
    }

    // adt.typeref (generic struct references)
    if is_adt_typeref(ctx, ty) {
        return true;
    }

    // Concrete ADT structs
    if is_adt_struct_type(ctx, ty) {
        return true;
    }

    // Structural user struct and variant types
    if trunk_ir::dialect::wasm_gc::Struct::matches(ctx, ty) {
        return true;
    }

    false
}

/// Check if a type is a `closure.closure` type.
fn is_closure_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::closure::Closure::matches(ctx, ty)
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
    let is_i64 = trunk_ir::dialect::core::I64::matches(ctx, from_ty);
    let mut ops: Vec<OpRef> = Vec::new();

    let val = if is_i64 {
        let wrap = wasm_dialect::I32WrapI64::operands(value)
            .results(i32_ty)
            .build(ctx, loc);
        ops.push(wrap.op_ref());
        wrap.result(ctx)
    } else {
        value
    };

    let ref_op = wasm_dialect::RefI31::operands(val)
        .results(i31ref_ty)
        .build(ctx, loc);
    ops.push(ref_op.op_ref());
    let i31_val = ref_op.result(ctx);

    // Upcast i31ref -> anyref (needed for IR type correctness)
    let upcast = wasm_dialect::RefCast::operands(i31_val)
        .target_type(anyref_ty)
        .type_idx(None)
        .results(anyref_ty)
        .build(ctx, loc);
    ops.push(upcast.op_ref());

    Some(MaterializeResult {
        value: upcast.result(ctx),
        ops,
    })
}

/// Helper to generate i31 unboxing operations (ref_cast to i31ref + i31_get_s).
///
/// This is used when converting anyref-typed values back to i32, such as
/// extracting boxed source values from fields with erased reference types.
fn unbox_via_i31(
    ctx: &mut IrContext,
    loc: Location,
    value: ValueRef,
    i31ref_ty: TypeRef,
    i32_ty: TypeRef,
) -> Option<MaterializeResult> {
    // Cast anyref to i31ref
    let cast_op = wasm_dialect::RefCast::operands(value)
        .target_type(i31ref_ty)
        .type_idx(None)
        .results(i31ref_ty)
        .build(ctx, loc);
    let cast_val = cast_op.result(ctx);

    // Extract i32 from i31ref
    let get_op = wasm_dialect::I31GetS::operands(cast_val)
        .results(i32_ty)
        .build(ctx, loc);

    Some(MaterializeResult {
        value: get_op.result(ctx),
        ops: vec![cast_op.op_ref(), get_op.op_ref()],
    })
}

// =============================================================================
// Main entry point
// =============================================================================

/// Replace the shared closure storage, `core.bytes`, and the `tribute_rt`
/// primitives with their Wasm types everywhere they occur, including inside
/// ADT layouts, aliases, signatures, and attributes, before target
/// instructions.
pub(crate) fn convert_builtin_layouts(ctx: &mut IrContext, module: trunk_ir::rewrite::Module) {
    let shared_closure = crate::closure_lower::closure_struct_type_ref(ctx);
    let closure = closure_adt_type(ctx);
    let bytes = super::bytes::bytes_struct_type(ctx);
    let i32_ty = core::I32::type_ref(ctx);
    let f64_ty = core::F64::type_ref(ctx);
    let anyref_ty = wasm_dialect::Anyref::params().intern(ctx).as_type_ref();
    let i31ref_ty = wasm_dialect::I31ref::params().intern(ctx).as_type_ref();
    crate::closure_lower::substitute_module_types(ctx, module, move |ctx, ty| {
        if ty == shared_closure {
            return Some(closure);
        }
        if trunk_ir::dialect::core::Bytes::matches(ctx, ty) {
            return Some(bytes);
        }
        use tribute_ir::dialect::tribute_rt::{Anyref, Bool, Float, Int, Intref, Nat};

        let name = ctx.get_type(ty).name;
        if [Int::NAME, Nat::NAME, Bool::NAME].contains(&name) {
            Some(i32_ty)
        } else if name == Float::NAME {
            Some(f64_ty)
        } else if name == Anyref::NAME {
            Some(anyref_ty)
        } else if name == Intref::NAME {
            Some(i31ref_ty)
        } else {
            None
        }
    });
}

/// Create a TypeConverter configured for WASM backend type conversions.
///
/// This converter handles the IR-level type transformations needed during
/// lowering passes. It complements the emit-phase `gc_types::type_to_field_type`
/// by performing conversions at the IR level.
pub fn wasm_type_converter(ctx: &mut IrContext) -> TypeConverter {
    // Pre-compute commonly used types (TypeRef is Copy)
    let i32_ty = core::I32::type_ref(ctx);
    let f64_ty = core::F64::type_ref(ctx);
    let anyref_ty = wasm_dialect::Anyref::params().intern(ctx).as_type_ref();
    let i31ref_ty = wasm_dialect::I31ref::params().intern(ctx).as_type_ref();
    let structref_ty = wasm_dialect::Structref::params().intern(ctx).as_type_ref();
    let arrayref_ty = wasm_dialect::Arrayref::params().intern(ctx).as_type_ref();
    let closure_ty = closure_adt_type(ctx);
    let shared_closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let bytes_data_ty = super::bytes::bytes_data_type(ctx);
    let bytes_data_gc_ty = super::bytes::bytes_data_gc_type(ctx);

    let mut tc = TypeConverter::new();

    // =========================================================================
    // Type conversions
    // =========================================================================

    // Convert tribute_rt.int -> core.i32 (Phase 1: arbitrary precision as i32)
    tc.add_conversion(move |ctx, ty| {
        if tribute_ir::dialect::tribute_rt::Int::matches(ctx, ty) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.nat -> core.i32 (Phase 1: arbitrary precision as i32)
    tc.add_conversion(move |ctx, ty| {
        if tribute_ir::dialect::tribute_rt::Nat::matches(ctx, ty) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.bool -> core.i32 (boolean as i32)
    tc.add_conversion(move |ctx, ty| {
        if tribute_ir::dialect::tribute_rt::Bool::matches(ctx, ty) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert core.index -> core.i32 (the WasmGC array index width)
    tc.add_conversion(move |ctx, ty| {
        if trunk_ir::dialect::core::Index::matches(ctx, ty) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert core.i1 -> core.i32 (WASM doesn't have i1, use i32)
    tc.add_conversion(move |ctx, ty| {
        if trunk_ir::dialect::core::I1::matches(ctx, ty) {
            Some(i32_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.float -> core.f64 (float as f64)
    tc.add_conversion(move |ctx, ty| {
        if tribute_ir::dialect::tribute_rt::Float::matches(ctx, ty) {
            Some(f64_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.intref -> wasm.i31ref (boxed integer reference)
    tc.add_conversion(move |ctx, ty| {
        if tribute_ir::dialect::tribute_rt::Intref::matches(ctx, ty) {
            Some(i31ref_ty)
        } else {
            None
        }
    });

    // Convert tribute_rt.anyref -> wasm.anyref (any reference type)
    tc.add_conversion(move |ctx, ty| {
        if tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, ty) {
            Some(anyref_ty)
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

    // Exact compiler-owned shared closure storage has the same target contract
    // as a semantic closure. Do not infer this identity from ADT names or shape.
    tc.add_conversion(move |_, ty| (ty == shared_closure_ty).then_some(closure_ty));

    // Convert closure.closure -> adt.struct(name="_closure")
    // This ensures emit can identify closures by ADT name rather than tribute-ir type
    tc.add_conversion(move |ctx, ty| {
        if is_closure_type(ctx, ty) {
            Some(closure_ty)
        } else {
            None
        }
    });

    // Evidence keeps its `adt.array` type until `adt` lowering has consumed
    // every operation that reads it as one; `convert-evidence-arrays` then
    // replaces it with its GC array everywhere, as `convert-struct-layouts`
    // does for user structs.
    tc.add_conversion(move |ctx, ty| is_evidence_type_ref(ctx, ty).then_some(ty));

    // The bytes backing array keeps its layout identifier rather than
    // becoming an erased `wasm.arrayref`.
    tc.add_conversion(move |_, ty| {
        (ty == bytes_data_ty || ty == bytes_data_gc_ty).then_some(bytes_data_gc_ty)
    });

    // Convert any other `adt.array` -> wasm.arrayref. A GC array is already
    // a Wasm type and stays as it is.
    tc.add_conversion(move |ctx, ty| {
        if tribute_ir::dialect::adt::Array::matches(ctx, ty) {
            Some(arrayref_ty)
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

        // Reference upcasts (for example a concrete struct or `adt.typeref` to
        // `structref`/`anyref`) are elided by the Wasm target's upcast elision,
        // not materialized,
        // and values of equal representation get equal types from the target
        // conversion. Only real conversions are materialized below.
        let to_is_structref = trunk_ir::dialect::wasm::Structref::matches(ctx, to_ty);

        // Both are struct-like types but need actual bridging - generate ref_cast
        let from_is_struct_like = is_struct_like(ctx, from_ty);
        let to_is_struct_like = is_struct_like(ctx, to_ty);

        if from_is_struct_like && to_is_struct_like {
            // Upcasts are elided by the target; only downcasts need a ref_cast.
            let to_is_anyref = trunk_ir::dialect::wasm::Anyref::matches(ctx, to_ty);
            let from_is_anyref = trunk_ir::dialect::wasm::Anyref::matches(ctx, from_ty);
            if to_is_anyref || (to_is_structref && !from_is_anyref) {
                return None;
            }

            if trunk_ir::dialect::wasm::Structref::matches(ctx, to_ty) {
                let cast_op = wasm_dialect::RefCast::operands(value)
                    .target_type(to_ty)
                    .type_idx(None)
                    .results(to_ty)
                    .build(ctx, location);
                return Some(MaterializeResult {
                    value: cast_op.result(ctx),
                    ops: vec![cast_op.op_ref()],
                });
            }
            let cast_op = wasm_gc_dialect::RefCast::operands(value)
                .target_type(to_ty)
                .results(to_ty)
                .build(ctx, location);
            return Some(MaterializeResult {
                value: cast_op.result(ctx),
                ops: vec![cast_op.op_ref()],
            });
        }

        // anyref -> concrete struct type: use ref_cast
        // This handles erased struct slots such as closure environments that are
        // passed as anyref for a uniform calling convention.
        // We EXCLUDE wasm.anyref as target since that's handled by primitive equivalence.
        let from_is_anyref = trunk_ir::dialect::wasm::Anyref::matches(ctx, from_ty)
            || tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, from_ty);
        let to_is_abstract_anyref = trunk_ir::dialect::wasm::Anyref::matches(ctx, to_ty);
        if from_is_anyref && to_is_struct_like && !to_is_abstract_anyref {
            let cast_op = wasm_gc_dialect::RefCast::operands(value)
                .target_type(to_ty)
                .results(to_ty)
                .build(ctx, location);
            return Some(MaterializeResult {
                value: cast_op.result(ctx),
                ops: vec![cast_op.op_ref()],
            });
        }

        // -----------------------------------------------------------------
        // Boxing materializations (emit ops)
        // -----------------------------------------------------------------

        // core.i32 -> wasm.anyref (box via i31)
        if trunk_ir::dialect::core::I32::matches(ctx, from_ty)
            && trunk_ir::dialect::wasm::Anyref::matches(ctx, to_ty)
        {
            return box_via_i31(ctx, location, value, from_ty, i31ref_ty, anyref_ty, i32_ty);
        }
        // core.i64 -> wasm.anyref (box via i31, with i32 truncation)
        if trunk_ir::dialect::core::I64::matches(ctx, from_ty)
            && trunk_ir::dialect::wasm::Anyref::matches(ctx, to_ty)
        {
            return box_via_i31(ctx, location, value, from_ty, i31ref_ty, anyref_ty, i32_ty);
        }
        // core.nil -> wasm.anyref (nil is represented as null reference)
        if trunk_ir::dialect::core::Nil::matches(ctx, from_ty)
            && trunk_ir::dialect::wasm::Anyref::matches(ctx, to_ty)
        {
            let null_op = wasm_dialect::RefNull::operands()
                .heap_type("anyref")
                .type_idx(None)
                .results(anyref_ty)
                .build(ctx, location);
            return Some(MaterializeResult {
                value: null_op.result(ctx),
                ops: vec![null_op.op_ref()],
            });
        }

        // -----------------------------------------------------------------
        // Unboxing materializations
        // -----------------------------------------------------------------

        // wasm.anyref -> core.i32 (unbox via i31)
        if trunk_ir::dialect::wasm::Anyref::matches(ctx, from_ty)
            && trunk_ir::dialect::core::I32::matches(ctx, to_ty)
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }
        // tribute_rt.anyref -> core.i32 (unbox via i31, same as wasm.anyref)
        if tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, from_ty)
            && trunk_ir::dialect::core::I32::matches(ctx, to_ty)
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }
        // wasm.anyref -> tribute_rt.int (unbox via i31)
        if trunk_ir::dialect::wasm::Anyref::matches(ctx, from_ty)
            && tribute_ir::dialect::tribute_rt::Int::matches(ctx, to_ty)
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }
        // tribute_rt.anyref -> tribute_rt.int (unbox via i31)
        if tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, from_ty)
            && tribute_ir::dialect::tribute_rt::Int::matches(ctx, to_ty)
        {
            return unbox_via_i31(ctx, location, value, i31ref_ty, i32_ty);
        }
        // anyref -> core.nil: nil carries no data, so produce a fresh nil value
        // the way nil constants lower (`wasm.nop : core.nil`) and discard the
        // reference.
        if (trunk_ir::dialect::wasm::Anyref::matches(ctx, from_ty)
            || tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, from_ty))
            && trunk_ir::dialect::core::Nil::matches(ctx, to_ty)
        {
            let nil_op = wasm_dialect::Nop::operands()
                .results(to_ty)
                .build(ctx, location);
            return Some(MaterializeResult {
                value: nil_op.result(ctx),
                ops: vec![nil_op.op_ref()],
            });
        }

        // -----------------------------------------------------------------
        // anyref -> arrayref materialization (requires ref_cast)
        // -----------------------------------------------------------------

        let from_is_any = trunk_ir::dialect::wasm::Anyref::matches(ctx, from_ty)
            || tribute_ir::dialect::tribute_rt::Anyref::matches(ctx, from_ty);
        if from_is_any && is_evidence_type_ref(ctx, to_ty) {
            let cast_op = wasm_gc_dialect::RefCast::operands(value)
                .target_type(to_ty)
                .results(to_ty)
                .build(ctx, location);
            return Some(MaterializeResult {
                value: cast_op.result(ctx),
                ops: vec![cast_op.op_ref()],
            });
        }
        if from_is_any && trunk_ir::dialect::wasm::Arrayref::matches(ctx, to_ty) {
            let cast_op = wasm_dialect::RefCast::operands(value)
                .target_type(arrayref_ty)
                .type_idx(None)
                .results(arrayref_ty)
                .build(ctx, location);
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

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::ops::DialectType;
    use trunk_ir::types::Attribute;

    #[test]
    fn wasm_type_converter_maps_index_to_the_array_index_width() {
        let mut ctx = IrContext::new();
        let type_converter = wasm_type_converter(&mut ctx);
        let index = trunk_ir::dialect::core::Index::type_ref(&mut ctx);
        let i32_ty = trunk_ir::dialect::core::I32::type_ref(&mut ctx);

        assert_eq!(type_converter.convert_type(&ctx, index), Some(i32_ty));
    }

    #[test]
    fn runtime_layout_identifiers_match_the_wasm_builtin_layouts() {
        assert_eq!(
            tribute_ir::runtime_layout::CLOSURE,
            trunk_ir_wasm_backend::gc_types::CLOSURE_LAYOUT
        );
        assert_eq!(
            tribute_ir::runtime_layout::BYTES,
            trunk_ir_wasm_backend::gc_types::BYTES_LAYOUT
        );
        assert_eq!(
            tribute_ir::runtime_layout::BYTES_DATA,
            trunk_ir_wasm_backend::gc_types::BYTES_DATA_LAYOUT
        );
        assert_eq!(
            tribute_ir::runtime_layout::DESCRIBED,
            trunk_ir_wasm_backend::gc_types::DESCRIBED_LAYOUT
        );
        assert_eq!(
            tribute_ir::runtime_layout::BOXED_F64,
            trunk_ir_wasm_backend::gc_types::BOXED_F64_LAYOUT
        );
        let mut ctx = IrContext::new();
        let closure = closure_adt_type(&mut ctx);
        assert!(tribute_ir::runtime_layout::has_runtime_layout(
            &ctx,
            closure,
            tribute_ir::runtime_layout::CLOSURE
        ));
    }
    use trunk_ir::ops::DialectOp;

    #[test]
    fn cast_legalization_retypes_then_elides_reference_upcasts() {
        use trunk_ir::analysis::AnalysisCache;
        use trunk_ir::conversion::{ReconcileUnrealizedCasts, UnrealizedCastConversionPattern};
        use trunk_ir::pass::Pass;
        use trunk_ir_wasm_backend::passes::reference_upcast_elision::ReferenceUpcastElisionPattern;

        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Closure = adt.struct<_closure(code: core.i32), {layout = "closure"}>
  func.func @f(%c: !Closure) {
    %erased = core.unrealized_conversion_cast %c : tribute_rt.anyref
    func.call %erased {callee = @use}
    func.return
  }
}"#,
        );
        let tc = wasm_type_converter(&mut ctx);

        let result = trunk_ir::rewrite::PatternApplicator::new(tc)
            .add_pattern(UnrealizedCastConversionPattern)
            .add_pattern(ReferenceUpcastElisionPattern)
            .apply_partial(&mut ctx, module);
        let core_module = trunk_ir::dialect::core::Module::from_op(&ctx, module.op()).unwrap();
        ReconcileUnrealizedCasts
            .run(&mut ctx, core_module, &mut AnalysisCache::new())
            .unwrap();

        assert!(result.reached_fixpoint);
        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!printed.contains("unrealized_conversion_cast"), "{printed}");
        assert!(printed.contains("func.call %0"), "{printed}");
    }

    #[test]
    fn reference_upcasts_and_representation_matches_are_not_materializations() {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("test.trb");
        let location = Location::new(path, trunk_ir::location::Span::new(0, 0));
        let tc = wasm_type_converter(&mut ctx);
        let anyref = wasm_dialect::Anyref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let ptr = core::Ptr::params().intern(&mut ctx).as_type_ref();
        let concrete = closure_adt_type(&mut ctx);

        let source_data = trunk_ir::OperationDataBuilder::new(location, "test.source")
            .result(anyref)
            .build(&mut ctx);
        let source = ctx.create_op(source_data);
        let value = ctx.op_result(source, 0);
        // An upcast is elided by the target, and a representation match is not
        // a conversion; neither is materialized.
        assert!(
            tc.materialize(&mut ctx, location, value, concrete, anyref)
                .is_none()
        );
        assert!(
            tc.materialize(&mut ctx, location, value, anyref, ptr)
                .is_none()
        );
    }

    #[test]
    fn anyref_to_nil_materializes_a_fresh_nil_value() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%value: wasm.anyref) -> core.nil {
    %unit = core.unrealized_conversion_cast %value : core.nil
    func.return %unit
  }
}"#,
        );
        let tc = wasm_type_converter(&mut ctx);

        trunk_ir::rewrite::PatternApplicator::new(tc)
            .add_pattern(trunk_ir::conversion::UnrealizedCastConversionPattern)
            .apply_partial(&mut ctx, module);

        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!printed.contains("unrealized_conversion_cast"), "{printed}");
        assert!(printed.contains("wasm.nop"), "{printed}");
    }

    #[test]
    fn canonical_storage_conversion_updates_nested_block_argument_attributes() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
            func.func @test(%value: core.i32) -> core.i32 { func.return %value }
        }"#,
        );
        let function = trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let block = ctx.region(function.body(&ctx)).blocks[0];
        let source = crate::closure_lower::closure_struct_type_ref(&mut ctx);
        let target = closure_adt_type(&mut ctx);
        let nested = |ty| Attribute::List(vec![Attribute::List(vec![Attribute::Type(ty)])]);
        ctx.block_arg_attrs_mut(block, 0)
            .insert("storage", nested(source));
        convert_builtin_layouts(&mut ctx, module);
        assert_eq!(
            ctx.block(block).args[0].attrs.view(&ctx).get("storage"),
            Some(&nested(target))
        );
    }

    #[test]
    fn builtin_layout_conversion_replaces_nested_tribute_rt_primitives() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Option = adt.enum<Option { None(), Some(tribute_rt.anyref) }>
  !Sample = adt.struct<Sample(count: tribute_rt.nat, delta: tribute_rt.int, flag: tribute_rt.bool, ratio: tribute_rt.float, small: tribute_rt.intref, any: tribute_rt.anyref)>
  func.func @f(%sample: !Sample, %callee: core.i32, %value: tribute_rt.anyref) -> tribute_rt.int {
    %null = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %some = adt.variant_new %null {tag = "Some", type = !Option} : !Option
    %result = func.call_indirect %callee, %value {signature = func.func_sig<(tribute_rt.anyref) -> tribute_rt.int>} : tribute_rt.int
    func.return %result
  }
}"#,
        );

        convert_builtin_layouts(&mut ctx, module);

        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!printed.contains("tribute_rt."), "{printed}");
        assert!(
            printed.contains(
                "Sample(count: core.i32, delta: core.i32, flag: core.i32, ratio: core.f64, small: wasm.i31ref, any: wasm.anyref)"
            ),
            "{printed}"
        );
        assert!(printed.contains("Some(wasm.anyref)"), "{printed}");
        assert!(
            printed.contains("func.func_sig<(wasm.anyref) -> core.i32>"),
            "{printed}"
        );
    }

    #[test]
    fn canonical_closure_storage_conversion_requires_exact_shared_identity() {
        let mut ctx = IrContext::new();
        let shared = crate::closure_lower::closure_struct_type_ref(&mut ctx);
        let target = closure_adt_type(&mut ctx);
        let generic = wasm_dialect::Structref::params()
            .intern(&mut ctx)
            .as_type_ref();
        let mut near = ctx.get_type(shared).clone();
        near.attrs
            .view_mut(&mut ctx)
            .insert("unrelated", Attribute::Bool(true));
        let near = ctx.intern_type(near);
        let converter = wasm_type_converter(&mut ctx);
        assert_eq!(converter.convert_type_or_identity(&ctx, shared), target);
        assert_eq!(converter.convert_type_or_identity(&ctx, near), near);
        assert_eq!(converter.convert_type_or_identity(&ctx, generic), generic);
        let i32_ty = core::I32::type_ref(&mut ctx);
        itertools::assert_equal(
            adt::Struct::from_type_ref(&ctx, target)
                .unwrap()
                .fields(&ctx),
            [("code", i32_ty)],
        );
    }
}
