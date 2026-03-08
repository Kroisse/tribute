//! Native type converter for IR-level type transformations.
//!
//! This module provides a `TypeConverter` configuration for converting
//! high-level Tribute types to their native (Cranelift) representations
//! during IR lowering.
//!
//! ## Type Conversion Rules
//!
//! | Source Type         | Target Type | Notes                              |
//! | ------------------- | ----------- | ---------------------------------- |
//! | `tribute_rt.int`    | `core.i32`  | Arbitrary precision -> i32 (Ph. 1) |
//! | `tribute_rt.nat`    | `core.i32`  | Arbitrary precision -> i32 (Ph. 1) |
//! | `tribute_rt.bool`   | `core.i32`  | Boolean as i32                     |
//! | `core.i1`           | `core.i32`  | Native uses i32 for booleans       |
//! | `tribute_rt.float`  | `core.f64`  | Float as f64                       |
//! | `tribute_rt.intref` | `core.ptr`  | Boxed integer as heap pointer      |
//! | `tribute_rt.any`    | `core.ptr`  | Uniform representation as pointer  |
//! | `adt.struct`        | `core.ptr`  | Struct as heap pointer             |
//! | `adt.variant_inst`  | `core.ptr`  | Variant instance as heap pointer   |
//! | `adt.typeref<T>`    | `core.ptr`  | Struct reference as pointer        |
//! | `closure.closure`   | `core.ptr`  | Closure as pointer                 |
//! | `core.array<T>`     | `core.ptr`  | Array as heap pointer              |
//! | `cont.prompt_tag`   | `core.i32`  | Prompt tag as integer              |
//!
//! ## Materializations
//!
//! Unlike the WASM backend which requires `ref_cast` operations between GC
//! types, the native backend uses opaque pointers (`core.ptr`) for all
//! reference types. Most conversions between pointer types are no-ops.

use tribute_ir::dialect::tribute_rt as arena_tribute_rt;
use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif as arena_clif;
use trunk_ir::dialect::cont as arena_cont;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::TypeConverter;
use trunk_ir::types::{Location, TypeDataBuilder};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

// =============================================================================
// Arena versions
// =============================================================================

/// Pre-interned types for the arena native type converter.
///
/// Created once from `&mut IrContext` and then used by conversion closures
/// that only receive `&IrContext`.
pub struct NativeTypeRefs {
    // Primitive source types
    pub tribute_rt_int: TypeRef,
    pub tribute_rt_nat: TypeRef,
    pub tribute_rt_bool: TypeRef,
    pub tribute_rt_float: TypeRef,
    pub tribute_rt_intref: TypeRef,
    pub tribute_rt_any: TypeRef,
    pub core_i1: TypeRef,
    pub cont_prompt_tag: TypeRef,

    // Target types
    pub core_i32: TypeRef,
    pub core_i64: TypeRef,
    pub core_f64: TypeRef,
    pub core_ptr: TypeRef,
    pub core_nil: TypeRef,
    pub core_i8: TypeRef,

    // Special types
    pub evidence_ty: TypeRef,
    pub marker_ty: TypeRef,
}

impl NativeTypeRefs {
    /// Pre-intern all types needed by the native type converter.
    pub fn new(ctx: &mut IrContext) -> Self {
        use tribute_ir::dialect::ability as arena_ability;
        Self {
            tribute_rt_int: arena_tribute_rt::int(ctx).as_type_ref(),
            tribute_rt_nat: arena_tribute_rt::nat(ctx).as_type_ref(),
            tribute_rt_bool: arena_tribute_rt::bool(ctx).as_type_ref(),
            tribute_rt_float: arena_tribute_rt::float(ctx).as_type_ref(),
            tribute_rt_intref: arena_tribute_rt::intref(ctx).as_type_ref(),
            tribute_rt_any: arena_tribute_rt::any(ctx).as_type_ref(),
            core_i1: ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build()),
            cont_prompt_tag: arena_cont::prompt_tag(ctx).as_type_ref(),

            core_i32: ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build()),
            core_i64: ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build()),
            core_f64: ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build()),
            core_ptr: arena_core::ptr(ctx).as_type_ref(),
            core_nil: arena_core::nil(ctx).as_type_ref(),
            core_i8: ctx
                .types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i8")).build()),

            evidence_ty: arena_ability::evidence_adt_type_ref(ctx),
            marker_ty: arena_ability::marker_adt_type_ref(ctx),
        }
    }
}

/// Create an `TypeConverter` configured for native backend type conversions.
///
/// Arena version of `native_type_converter()`. All type checks use pre-interned
/// `TypeRef` comparisons for efficiency.
pub fn native_type_converter(ctx: &mut IrContext) -> (TypeConverter, NativeTypeRefs) {
    let refs = NativeTypeRefs::new(ctx);

    let mut tc = TypeConverter::new();

    // Capture refs for closures (all fields are Copy)
    let r = refs.clone_refs();

    // === Primitive type conversions ===
    tc.add_conversion(move |ctx, ty| {
        if ty == r.tribute_rt_int || ty == r.tribute_rt_nat || ty == r.tribute_rt_bool {
            return Some(r.core_i32);
        }
        if ty == r.core_i1 {
            return Some(r.core_i32);
        }
        if ty == r.tribute_rt_float {
            return Some(r.core_f64);
        }
        if ty == r.tribute_rt_intref || ty == r.tribute_rt_any {
            return Some(r.core_ptr);
        }
        if ty == r.cont_prompt_tag {
            return Some(r.core_i32);
        }
        // Evidence type → i64
        if ty == r.evidence_ty {
            return Some(r.core_i64);
        }
        // ADT struct/enum/variant_instance/typeref → ptr
        if is_adt_ptr_type(ctx, ty) {
            return Some(r.core_ptr);
        }
        // closure.closure → ptr
        if is_closure_type(ctx, ty) {
            return Some(r.core_ptr);
        }
        // core.func → ptr
        if is_func_type(ctx, ty) {
            return Some(r.core_ptr);
        }
        // core.array (non-evidence) → ptr
        if is_array_type(ctx, ty) && ty != r.evidence_ty {
            return Some(r.core_ptr);
        }
        None
    });

    // === Materializations ===
    // Note: TypeConverter uses set_materializer (single function) instead of multiple.
    tc.set_materializer(move |ctx, location, value, from_ty, to_ty| {
        // Same type: no-op
        if from_ty == to_ty {
            return Some(arena_materialize_result_noop(value));
        }

        // Primitive equivalences (NoOp)
        if (from_ty == r.tribute_rt_int
            || from_ty == r.tribute_rt_nat
            || from_ty == r.tribute_rt_bool
            || from_ty == r.core_i1)
            && to_ty == r.core_i32
        {
            return Some(arena_materialize_result_noop(value));
        }
        if from_ty == r.tribute_rt_float && to_ty == r.core_f64 {
            return Some(arena_materialize_result_noop(value));
        }
        if (from_ty == r.tribute_rt_intref || from_ty == r.tribute_rt_any) && to_ty == r.core_ptr {
            return Some(arena_materialize_result_noop(value));
        }
        if from_ty == r.evidence_ty && to_ty == r.core_i64 {
            return Some(arena_materialize_result_noop(value));
        }
        if from_ty == r.core_i64 && to_ty == r.evidence_ty {
            return Some(arena_materialize_result_noop(value));
        }
        if from_ty == r.cont_prompt_tag && to_ty == r.core_i32 {
            return Some(arena_materialize_result_noop(value));
        }

        // Pointer equivalences: ptr-like ↔ ptr
        let from_ptr_like = is_ptr_like(ctx, from_ty, r.evidence_ty, r.core_ptr);
        let to_ptr_like = is_ptr_like(ctx, to_ty, r.evidence_ty, r.core_ptr);
        let from_is_ptr = from_ty == r.core_ptr;
        let to_is_ptr = to_ty == r.core_ptr;

        if (from_ptr_like && to_is_ptr)
            || (from_is_ptr && to_ptr_like)
            || (from_ptr_like && to_ptr_like)
        {
            return Some(arena_materialize_result_noop(value));
        }

        // Boxing: primitive → ptr
        if to_is_ptr {
            if from_ty == r.core_i32 {
                return Some(box_primitive(
                    ctx, location, value, 4, r.core_i64, r.core_i32, r.core_ptr,
                ));
            }
            if from_ty == r.core_i64 {
                return Some(box_primitive(
                    ctx, location, value, 8, r.core_i64, r.core_i32, r.core_ptr,
                ));
            }
            if from_ty == r.core_f64 {
                return Some(box_primitive(
                    ctx, location, value, 8, r.core_i64, r.core_i32, r.core_ptr,
                ));
            }
            if from_ty == r.core_nil {
                let null_op = arena_clif::iconst(ctx, location, r.core_ptr, 0);
                return Some(trunk_ir::rewrite::type_converter::MaterializeResult {
                    value: null_op.result(ctx),
                    ops: vec![null_op.op_ref()],
                });
            }
        }

        // Unboxing: ptr → primitive
        let from_is_ptr_or_any = from_ty == r.core_ptr || from_ty == r.tribute_rt_any;
        if from_is_ptr_or_any {
            if to_ty == r.core_i32 || to_ty == r.tribute_rt_int {
                let load = arena_clif::load(ctx, location, value, r.core_i32, 0);
                return Some(trunk_ir::rewrite::type_converter::MaterializeResult {
                    value: load.result(ctx),
                    ops: vec![load.op_ref()],
                });
            }
            if to_ty == r.core_i64 {
                let load = arena_clif::load(ctx, location, value, r.core_i64, 0);
                return Some(trunk_ir::rewrite::type_converter::MaterializeResult {
                    value: load.result(ctx),
                    ops: vec![load.op_ref()],
                });
            }
            if to_ty == r.core_f64 {
                let load = arena_clif::load(ctx, location, value, r.core_f64, 0);
                return Some(trunk_ir::rewrite::type_converter::MaterializeResult {
                    value: load.result(ctx),
                    ops: vec![load.op_ref()],
                });
            }
            if to_ty == r.core_nil {
                return Some(arena_materialize_result_noop(value));
            }
        }

        None
    });

    (tc, refs)
}

/// NoOp materialization result (pass value through unchanged).
fn arena_materialize_result_noop(
    value: ValueRef,
) -> trunk_ir::rewrite::type_converter::MaterializeResult {
    trunk_ir::rewrite::type_converter::MaterializeResult { value, ops: vec![] }
}

/// Generate boxing operations in arena IR: allocate + store RC header + store value.
fn box_primitive(
    ctx: &mut IrContext,
    location: Location,
    value: ValueRef,
    payload_size: u64,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
    ptr_ty: TypeRef,
) -> trunk_ir::rewrite::type_converter::MaterializeResult {
    let mut ops: Vec<OpRef> = Vec::new();

    // 1. Allocation size (payload + RC header)
    let alloc_size = payload_size + RC_HEADER_SIZE;
    let size_op = arena_clif::iconst(ctx, location, i64_ty, alloc_size as i64);
    ops.push(size_op.op_ref());

    // 2. Allocate heap memory
    let call_op = arena_clif::call(
        ctx,
        location,
        [size_op.result(ctx)],
        ptr_ty,
        Symbol::new(ALLOC_FN),
    );
    ops.push(call_op.op_ref());
    let raw_ptr = call_op.result(ctx);

    // 3. Store refcount = 1 at raw_ptr + 0
    let rc_one = arena_clif::iconst(ctx, location, i32_ty, 1);
    ops.push(rc_one.op_ref());
    let store_rc = arena_clif::store(ctx, location, rc_one.result(ctx), raw_ptr, 0);
    ops.push(store_rc.op_ref());

    // 4. Store rtti_idx = 0 at raw_ptr + 4
    let rtti_zero = arena_clif::iconst(ctx, location, i32_ty, 0);
    ops.push(rtti_zero.op_ref());
    let store_rtti = arena_clif::store(ctx, location, rtti_zero.result(ctx), raw_ptr, 4);
    ops.push(store_rtti.op_ref());

    // 5. Compute payload pointer = raw_ptr + 8
    let hdr_size = arena_clif::iconst(ctx, location, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_size.op_ref());
    let payload_ptr = arena_clif::iadd(ctx, location, raw_ptr, hdr_size.result(ctx), ptr_ty);
    ops.push(payload_ptr.op_ref());

    // 6. Store value at payload offset 0
    let store_val = arena_clif::store(ctx, location, value, payload_ptr.result(ctx), 0);
    ops.push(store_val.op_ref());

    // 7. Identity pass-through so the last op produces the payload ptr result
    let zero_op = arena_clif::iconst(ctx, location, ptr_ty, 0);
    ops.push(zero_op.op_ref());
    let identity_op = arena_clif::iadd(
        ctx,
        location,
        payload_ptr.result(ctx),
        zero_op.result(ctx),
        ptr_ty,
    );
    ops.push(identity_op.op_ref());

    trunk_ir::rewrite::type_converter::MaterializeResult {
        value: identity_op.result(ctx),
        ops,
    }
}

/// Check if a type is a pointer-like reference type in arena native representation.
pub fn is_ptr_like(ctx: &IrContext, ty: TypeRef, evidence_ty: TypeRef, ptr_ty: TypeRef) -> bool {
    if ty == ptr_ty {
        return true;
    }

    let data = ctx.types.get(ty);

    // adt.struct, adt.enum, adt.typeref, or variant instance (adt.*)
    if data.dialect == Symbol::new("adt") {
        // Check for typeref
        if data.name == Symbol::new("typeref") {
            return true;
        }
        // Check for struct or enum types (have "fields" or "variants" attrs)
        if data.attrs.contains_key(&Symbol::new("fields")) {
            return true;
        }
        if data.attrs.contains_key(&Symbol::new("variants")) {
            return true;
        }
        // Check for variant instance (has is_variant=true)
        if let Some(trunk_ir::types::Attribute::Bool(true)) =
            data.attrs.get(&Symbol::new("is_variant"))
        {
            return true;
        }
    }

    // core.array — but NOT evidence arrays
    if data.dialect == Symbol::new("core") && data.name == Symbol::new("array") && ty != evidence_ty
    {
        return true;
    }

    // tribute_rt.any / tribute_rt.intref
    if data.dialect == Symbol::new("tribute_rt")
        && (data.name == Symbol::new("any") || data.name == Symbol::new("intref"))
    {
        return true;
    }

    // closure.closure
    if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
        return true;
    }

    // core.func (function pointers)
    if data.dialect == Symbol::new("core") && data.name == Symbol::new("func") {
        return true;
    }

    false
}

/// Helper: Check if a type is an ADT type that maps to ptr in the native backend.
fn is_adt_ptr_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if data.dialect != Symbol::new("adt") {
        return false;
    }
    // struct, enum, typeref, variant instance
    data.name == Symbol::new("typeref")
        || data.attrs.contains_key(&Symbol::new("fields"))
        || data.attrs.contains_key(&Symbol::new("variants"))
        || matches!(
            data.attrs.get(&Symbol::new("is_variant")),
            Some(trunk_ir::types::Attribute::Bool(true))
        )
}

/// Helper: Check if a type is closure.closure.
fn is_closure_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure")
}

/// Helper: Check if a type is core.func.
fn is_func_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("func")
}

/// Helper: Check if a type is core.array.
fn is_array_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("array")
}

impl NativeTypeRefs {
    /// Clone all TypeRef fields for use in closures.
    /// (TypeRef is Copy, so this is just a convenient struct copy.)
    fn clone_refs(&self) -> NativeTypeRefsCopy {
        NativeTypeRefsCopy {
            tribute_rt_int: self.tribute_rt_int,
            tribute_rt_nat: self.tribute_rt_nat,
            tribute_rt_bool: self.tribute_rt_bool,
            tribute_rt_float: self.tribute_rt_float,
            tribute_rt_intref: self.tribute_rt_intref,
            tribute_rt_any: self.tribute_rt_any,
            core_i1: self.core_i1,
            cont_prompt_tag: self.cont_prompt_tag,
            core_i32: self.core_i32,
            core_i64: self.core_i64,
            core_f64: self.core_f64,
            core_ptr: self.core_ptr,
            core_nil: self.core_nil,
            core_i8: self.core_i8,
            evidence_ty: self.evidence_ty,
            marker_ty: self.marker_ty,
        }
    }
}

/// Copy of NativeTypeRefs that can be moved into closures.
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct NativeTypeRefsCopy {
    tribute_rt_int: TypeRef,
    tribute_rt_nat: TypeRef,
    tribute_rt_bool: TypeRef,
    tribute_rt_float: TypeRef,
    tribute_rt_intref: TypeRef,
    tribute_rt_any: TypeRef,
    core_i1: TypeRef,
    cont_prompt_tag: TypeRef,
    core_i32: TypeRef,
    core_i64: TypeRef,
    core_f64: TypeRef,
    core_ptr: TypeRef,
    core_nil: TypeRef,
    core_i8: TypeRef,
    evidence_ty: TypeRef,
    marker_ty: TypeRef,
}
