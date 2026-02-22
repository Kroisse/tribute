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

use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;
use tribute_ir::dialect::{ability, closure, tribute_rt};
use trunk_ir::dialect::{adt, clif, cont, core};
use trunk_ir::rewrite::{MaterializeResult, TypeConverter};
use trunk_ir::{DialectOp, DialectType, Symbol, Type};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// Generate boxing operations: allocate + store RC header + store value, return pointer.
///
/// The last operation produces a `core.ptr` result (payload pointer) for value mapping.
fn box_primitive<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    value: trunk_ir::Value<'db>,
    payload_size: u64,
) -> Vec<trunk_ir::Operation<'db>> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i64_ty = core::I64::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();

    let mut ops = Vec::new();

    // 1. Allocation size (payload + RC header)
    let alloc_size = payload_size + RC_HEADER_SIZE;
    let size_op = clif::iconst(db, location, i64_ty, alloc_size as i64);
    let size_val = size_op.result(db);
    ops.push(size_op.as_operation());

    // 2. Allocate heap memory
    let call_op = clif::call(db, location, [size_val], ptr_ty, Symbol::new(ALLOC_FN));
    let raw_ptr = call_op.result(db);
    ops.push(call_op.as_operation());

    // 3. Store refcount = 1 at raw_ptr + 0
    let rc_one = clif::iconst(db, location, i32_ty, 1);
    ops.push(rc_one.as_operation());
    let store_rc = clif::store(db, location, rc_one.result(db), raw_ptr, 0);
    ops.push(store_rc.as_operation());

    // 4. Store rtti_idx = 0 at raw_ptr + 4
    let rtti_zero = clif::iconst(db, location, i32_ty, 0);
    ops.push(rtti_zero.as_operation());
    let store_rtti = clif::store(db, location, rtti_zero.result(db), raw_ptr, 4);
    ops.push(store_rtti.as_operation());

    // 5. Compute payload pointer = raw_ptr + 8
    let hdr_size = clif::iconst(db, location, i64_ty, RC_HEADER_SIZE as i64);
    ops.push(hdr_size.as_operation());
    let payload_ptr = clif::iadd(db, location, raw_ptr, hdr_size.result(db), ptr_ty);
    ops.push(payload_ptr.as_operation());

    // 6. Store value at payload offset 0
    let store_val = clif::store(db, location, value, payload_ptr.result(db), 0);
    ops.push(store_val.as_operation());

    // 7. Identity pass-through so the last op produces the payload ptr result.
    let zero_op = clif::iconst(db, location, ptr_ty, 0);
    let zero_val = zero_op.result(db);
    ops.push(zero_op.as_operation());

    let identity_op = clif::iadd(db, location, payload_ptr.result(db), zero_val, ptr_ty);
    ops.push(identity_op.as_operation());

    ops
}

/// Create a TypeConverter configured for native backend type conversions.
///
/// This converter handles the IR-level type transformations needed during
/// lowering passes for the Cranelift native backend. All reference types
/// (structs, arrays, closures, evidence) are represented as `core.ptr`.
///
/// # Example
///
/// ```
/// # use tribute_ir::dialect::tribute_rt;
/// # use trunk_ir::dialect::core;
/// # use trunk_ir::DialectType;
/// use tribute_passes::native::type_converter::native_type_converter;
///
/// # let db = salsa::DatabaseImpl::default();
/// let converter = native_type_converter();
///
/// // Convert tribute_rt.int to core.i32
/// # let int_ty = tribute_rt::Int::new(&db).as_type();
/// let i32_ty = converter.convert_type(&db, int_ty).unwrap();
/// # assert_eq!(i32_ty, core::I32::new(&db).as_type());
/// ```
pub fn native_type_converter() -> TypeConverter {
    TypeConverter::new()
        // === Primitive type conversions ===
        //
        // Convert tribute_rt.int -> core.i32
        .add_conversion(|db, ty| {
            tribute_rt::Int::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert tribute_rt.nat -> core.i32
        .add_conversion(|db, ty| {
            tribute_rt::Nat::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert tribute_rt.bool -> core.i32
        .add_conversion(|db, ty| {
            tribute_rt::Bool::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert core.i1 -> core.i32 (Cranelift uses i8/i32 for booleans)
        .add_conversion(|db, ty| {
            core::I::<1>::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert tribute_rt.float -> core.f64
        .add_conversion(|db, ty| {
            tribute_rt::Float::from_type(db, ty).map(|_| core::F64::new(db).as_type())
        })
        // === Reference type conversions (all -> core.ptr) ===
        //
        // In the native backend, all heap-allocated and reference types
        // are represented as opaque pointers. This is the key difference
        // from WASM where GC types (anyref, structref, i31ref) are used.
        //
        // Convert tribute_rt.intref -> core.ptr (boxed integer)
        .add_conversion(|db, ty| {
            tribute_rt::Intref::from_type(db, ty).map(|_| core::Ptr::new(db).as_type())
        })
        // Convert tribute_rt.any -> core.ptr (uniform representation)
        .add_conversion(|db, ty| {
            tribute_rt::Any::from_type(db, ty).map(|_| core::Ptr::new(db).as_type())
        })
        // Convert adt.typeref -> core.ptr (generic struct reference)
        .add_conversion(|db, ty| {
            if adt::is_typeref(db, ty) {
                Some(core::Ptr::new(db).as_type())
            } else {
                None
            }
        })
        // Convert adt.struct / adt.enum / variant instance -> core.ptr (opaque reference)
        .add_conversion(|db, ty| {
            if adt::is_struct_type(db, ty)
                || adt::is_enum_type(db, ty)
                || adt::is_variant_instance_type(db, ty)
            {
                Some(core::Ptr::new(db).as_type())
            } else {
                None
            }
        })
        // Convert closure.closure -> core.ptr
        .add_conversion(|db, ty| {
            if closure::Closure::from_type(db, ty).is_some() {
                Some(core::Ptr::new(db).as_type())
            } else {
                None
            }
        })
        // Convert core.func -> core.ptr (function pointers are pointers)
        .add_conversion(|db, ty| {
            core::Func::from_type(db, ty).map(|_| core::Ptr::new(db).as_type())
        })
        // Convert evidence type (core.array(Marker)) -> core.i64
        //
        // Evidence pointers are NOT RC-managed (they are allocated via Box
        // in the runtime, not via __tribute_alloc). Using core.i64 instead
        // of core.ptr prevents the RC insertion pass from generating
        // spurious retain/release operations that corrupt memory.
        // At the Cranelift level, i64 and ptr are both I64.
        .add_conversion(|db, ty| {
            if ability::is_evidence_type(db, ty) {
                Some(core::I64::new(db).as_type())
            } else {
                None
            }
        })
        // Convert generic core.array -> core.ptr (arrays are heap-allocated)
        .add_conversion(|db, ty| {
            core::Array::from_type(db, ty).map(|_| core::Ptr::new(db).as_type())
        })
        // Convert cont.prompt_tag -> core.i32
        .add_conversion(|db, ty| {
            cont::PromptTag::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Note: marker ADT type is converted to core.ptr like other structs.
        // But evidence_to_native rewrites evidence_lookup to return prompt_tag
        // directly as core.i32, so marker pointers don't appear in user code.
        // === Materializations ===
        //
        // Primitive type equivalences: same underlying representation, no-op.
        .add_materialization(|db, _location, _value, from_ty, to_ty| {
            if from_ty == to_ty {
                return MaterializeResult::NoOp;
            }

            // tribute_rt.int -> core.i32
            if tribute_rt::Int::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.nat -> core.i32
            if tribute_rt::Nat::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.bool -> core.i32
            if tribute_rt::Bool::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // core.i1 -> core.i32
            if core::I::<1>::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.float -> core.f64
            if tribute_rt::Float::from_type(db, from_ty).is_some()
                && core::F64::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.intref -> core.ptr
            if tribute_rt::Intref::from_type(db, from_ty).is_some()
                && core::Ptr::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.any -> core.ptr
            if tribute_rt::Any::from_type(db, from_ty).is_some()
                && core::Ptr::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // evidence type (core.array(Marker)) -> core.i64
            if ability::is_evidence_type(db, from_ty) && core::I64::from_type(db, to_ty).is_some() {
                return MaterializeResult::NoOp;
            }
            // core.i64 -> evidence type (reverse direction)
            if core::I64::from_type(db, from_ty).is_some() && ability::is_evidence_type(db, to_ty) {
                return MaterializeResult::NoOp;
            }
            // cont.prompt_tag -> core.i32
            if cont::PromptTag::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }

            MaterializeResult::Skip
        })
        // Pointer type equivalences: all reference types share core.ptr representation.
        .add_materialization(|db, _location, _value, from_ty, to_ty| {
            let ptr_ty = core::Ptr::new(db).as_type();

            let from_is_ptr_like = is_ptr_like(db, from_ty);
            let to_is_ptr = to_ty == ptr_ty;

            // Any pointer-like type -> core.ptr: no-op
            if from_is_ptr_like && to_is_ptr {
                return MaterializeResult::NoOp;
            }

            // core.ptr -> any pointer-like type: no-op (implicit cast)
            let from_is_ptr = from_ty == ptr_ty;
            let to_is_ptr_like = is_ptr_like(db, to_ty);
            if from_is_ptr && to_is_ptr_like {
                return MaterializeResult::NoOp;
            }

            // Between two pointer-like types (e.g., adt.struct -> adt.typeref): no-op
            if from_is_ptr_like && to_is_ptr_like {
                return MaterializeResult::NoOp;
            }

            MaterializeResult::Skip
        })
        // Boxing: primitive -> core.ptr (for polymorphic calls)
        // Generates heap allocation + store via clif ops.
        .add_materialization(|db, location, value, from_ty, to_ty| {
            let to_is_ptr = core::Ptr::from_type(db, to_ty).is_some();
            if !to_is_ptr {
                return MaterializeResult::Skip;
            }

            // i32 -> ptr: allocate 4 bytes, store i32
            if core::I32::from_type(db, from_ty).is_some() {
                return MaterializeResult::ops(box_primitive(db, location, value, 4));
            }

            // i64 -> ptr: allocate 8 bytes, store i64
            if core::I64::from_type(db, from_ty).is_some() {
                return MaterializeResult::ops(box_primitive(db, location, value, 8));
            }

            // f64 -> ptr: allocate 8 bytes, store f64
            if core::F64::from_type(db, from_ty).is_some() {
                return MaterializeResult::ops(box_primitive(db, location, value, 8));
            }

            // nil -> ptr: null pointer constant
            if core::Nil::from_type(db, from_ty).is_some() {
                let ptr_ty = core::Ptr::new(db).as_type();
                let null_op = clif::iconst(db, location, ptr_ty, 0);
                return MaterializeResult::single(null_op.as_operation());
            }

            MaterializeResult::Skip
        })
        // Unboxing: core.ptr -> primitive (for extracting values)
        // Generates clif.load from heap pointer.
        .add_materialization(|db, location, value, from_ty, to_ty| {
            let from_is_ptr = core::Ptr::from_type(db, from_ty).is_some()
                || tribute_rt::Any::from_type(db, from_ty).is_some();
            if !from_is_ptr {
                return MaterializeResult::Skip;
            }

            // ptr -> i32: load i32 from offset 0
            if core::I32::from_type(db, to_ty).is_some()
                || tribute_rt::Int::from_type(db, to_ty).is_some()
            {
                let i32_ty = core::I32::new(db).as_type();
                let load_op = clif::load(db, location, value, i32_ty, 0);
                return MaterializeResult::single(load_op.as_operation());
            }

            // ptr -> i64: load i64 from offset 0
            if core::I64::from_type(db, to_ty).is_some() {
                let i64_ty = core::I64::new(db).as_type();
                let load_op = clif::load(db, location, value, i64_ty, 0);
                return MaterializeResult::single(load_op.as_operation());
            }

            // ptr -> f64: load f64 from offset 0
            if core::F64::from_type(db, to_ty).is_some() {
                let f64_ty = core::F64::new(db).as_type();
                let load_op = clif::load(db, location, value, f64_ty, 0);
                return MaterializeResult::single(load_op.as_operation());
            }

            // ptr -> nil: discard the pointer value
            if core::Nil::from_type(db, to_ty).is_some() {
                return MaterializeResult::NoOp;
            }

            MaterializeResult::Skip
        })
}

/// Check if a type is a pointer-like reference type in native representation.
///
/// This includes ADT struct types, typeref, array types, closure types,
/// evidence types, and any/intref types. In the native backend, all of
/// these are represented as `core.ptr`.
fn is_ptr_like(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    // adt.struct, adt.enum, or adt.typeref
    if adt::is_struct_type(db, ty) || adt::is_enum_type(db, ty) || adt::is_typeref(db, ty) {
        return true;
    }

    // adt variant instance types
    if adt::is_variant_instance_type(db, ty) {
        return true;
    }

    // core.ptr
    if core::Ptr::from_type(db, ty).is_some() {
        return true;
    }

    // core.array — but NOT evidence arrays (ability::evidence_adt_type),
    // which are opaque i64 handles in the native backend.
    if core::Array::from_type(db, ty).is_some() {
        let evidence_ty = ability::evidence_adt_type(db);
        if ty != evidence_ty {
            return true;
        }
    }

    // tribute_rt.any / tribute_rt.intref
    if tribute_rt::Any::from_type(db, ty).is_some()
        || tribute_rt::Intref::from_type(db, ty).is_some()
    {
        return true;
    }

    // closure.closure
    if closure::Closure::from_type(db, ty).is_some() {
        return true;
    }

    // core.func (function pointers)
    if core::Func::from_type(db, ty).is_some() {
        return true;
    }

    // Evidence and marker types are NOT ptr-like in the native backend.
    // Evidence is represented as core.i64 (not core.ptr) to avoid RC tracking,
    // since evidence pointers are not RC-managed heap objects.

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::rewrite::MaterializeResult;
    use trunk_ir::{Attribute, Attrs, IdVec, Symbol};

    // === Type conversion tests ===

    #[salsa_test]
    fn test_convert_int_to_i32(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let int_ty = tribute_rt::Int::new(db).as_type();
        let result = converter.convert_type(db, int_ty);
        assert_eq!(result, Some(core::I32::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_nat_to_i32(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let nat_ty = tribute_rt::Nat::new(db).as_type();
        let result = converter.convert_type(db, nat_ty);
        assert_eq!(result, Some(core::I32::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_bool_to_i32(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let bool_ty = tribute_rt::Bool::new(db).as_type();
        let result = converter.convert_type(db, bool_ty);
        assert_eq!(result, Some(core::I32::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_i1_to_i32(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let i1_ty = core::I::<1>::new(db).as_type();
        let result = converter.convert_type(db, i1_ty);
        assert_eq!(result, Some(core::I32::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_float_to_f64(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let float_ty = tribute_rt::Float::new(db).as_type();
        let result = converter.convert_type(db, float_ty);
        assert_eq!(result, Some(core::F64::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_any_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let any_ty = tribute_rt::Any::new(db).as_type();
        let result = converter.convert_type(db, any_ty);
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_intref_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let intref_ty = tribute_rt::Intref::new(db).as_type();
        let result = converter.convert_type(db, intref_ty);
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_typeref_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let typeref_ty = adt::typeref(db, Symbol::new("MyStruct"));
        let result = converter.convert_type(db, typeref_ty);
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_struct_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let i32_ty = core::I32::new(db).as_type();
        let struct_ty = adt::struct_type(db, Symbol::new("Foo"), vec![(Symbol::new("x"), i32_ty)]);
        let result = converter.convert_type(db, struct_ty);
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_variant_instance_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let i32_ty = core::I32::new(db).as_type();
        let base_ty = adt::struct_type(db, Symbol::new("Expr"), vec![(Symbol::new("tag"), i32_ty)]);

        // Build a variant instance type with is_variant=true attribute
        let mut attrs = Attrs::new();
        attrs.insert(adt::ATTR_IS_VARIANT(), Attribute::Bool(true));
        attrs.insert(adt::ATTR_BASE_ENUM(), Attribute::Type(base_ty));
        attrs.insert(
            adt::ATTR_VARIANT_TAG(),
            Attribute::Symbol(Symbol::new("Add")),
        );
        let variant_ty = Type::new(
            db,
            base_ty.dialect(db),
            Symbol::from_dynamic("Expr$Add"),
            IdVec::new(),
            attrs,
        );

        assert!(adt::is_variant_instance_type(db, variant_ty));
        let result = converter.convert_type(db, variant_ty);
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_marker_type_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let marker_ty = ability::marker_adt_type(db);
        let result = converter.convert_type(db, marker_ty);
        // marker type is converted to ptr like other struct types
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_closure_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let func_ty = core::I32::new(db).as_type(); // simplified
        let closure_ty = closure::Closure::new(db, func_ty).as_type();
        let result = converter.convert_type(db, closure_ty);
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_evidence_to_i64(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let evidence_ty = ability::evidence_adt_type(db);
        let result = converter.convert_type(db, evidence_ty);
        assert_eq!(result, Some(core::I64::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_array_to_ptr(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let i32_ty = core::I32::new(db).as_type();
        let array_ty = core::Array::new(db, i32_ty).as_type();
        let result = converter.convert_type(db, array_ty);
        assert_eq!(result, Some(core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_prompt_tag_to_i32(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();
        let tag_ty = cont::PromptTag::new(db).as_type();
        let result = converter.convert_type(db, tag_ty);
        assert_eq!(result, Some(core::I32::new(db).as_type()));
    }

    #[salsa_test]
    fn test_no_conversion_for_core_types(db: &salsa::DatabaseImpl) {
        let converter = native_type_converter();

        assert!(
            converter
                .convert_type(db, core::I32::new(db).as_type())
                .is_none()
        );
        assert!(
            converter
                .convert_type(db, core::I64::new(db).as_type())
                .is_none()
        );
        assert!(
            converter
                .convert_type(db, core::F64::new(db).as_type())
                .is_none()
        );
        assert!(
            converter
                .convert_type(db, core::Ptr::new(db).as_type())
                .is_none()
        );
    }

    // === is_ptr_like tests ===

    #[salsa_test]
    fn test_is_ptr_like_adt_struct(db: &salsa::DatabaseImpl) {
        let i32_ty = core::I32::new(db).as_type();
        let struct_ty = adt::struct_type(db, Symbol::new("Foo"), vec![(Symbol::new("x"), i32_ty)]);
        assert!(is_ptr_like(db, struct_ty));
    }

    #[salsa_test]
    fn test_is_ptr_like_typeref(db: &salsa::DatabaseImpl) {
        let typeref_ty = adt::typeref(db, Symbol::new("T"));
        assert!(is_ptr_like(db, typeref_ty));
    }

    #[salsa_test]
    fn test_is_ptr_like_core_ptr(db: &salsa::DatabaseImpl) {
        assert!(is_ptr_like(db, core::Ptr::new(db).as_type()));
    }

    #[salsa_test]
    fn test_is_ptr_like_closure(db: &salsa::DatabaseImpl) {
        let func_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, func_ty).as_type();
        assert!(is_ptr_like(db, closure_ty));
    }

    #[salsa_test]
    fn test_is_not_ptr_like_primitives(db: &salsa::DatabaseImpl) {
        assert!(!is_ptr_like(db, core::I32::new(db).as_type()));
        assert!(!is_ptr_like(db, core::F64::new(db).as_type()));
        assert!(!is_ptr_like(db, core::Nil::new(db).as_type()));
    }

    // === Materialization tests ===
    //
    // Each tracked function is needed because Salsa materializations
    // must run inside a tracked context.

    // --- Primitive equivalences (NoOp) ---

    #[salsa::tracked]
    fn do_materialize_int_to_i32(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            tribute_rt::Int::new(db).as_type(),
            core::I32::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_int_to_i32_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_int_to_i32(db));
    }

    #[salsa::tracked]
    fn do_materialize_float_to_f64(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            tribute_rt::Float::new(db).as_type(),
            core::F64::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_float_to_f64_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_float_to_f64(db));
    }

    #[salsa::tracked]
    fn do_materialize_i1_to_i32(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            core::I::<1>::new(db).as_type(),
            core::I32::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_i1_to_i32_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_i1_to_i32(db));
    }

    #[salsa::tracked]
    fn do_materialize_prompt_tag_to_i32(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            cont::PromptTag::new(db).as_type(),
            core::I32::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_prompt_tag_to_i32_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_prompt_tag_to_i32(db));
    }

    // --- Pointer equivalences (NoOp) ---

    #[salsa::tracked]
    fn do_materialize_any_to_ptr(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            tribute_rt::Any::new(db).as_type(),
            core::Ptr::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_any_to_ptr_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_any_to_ptr(db));
    }

    #[salsa::tracked]
    fn do_materialize_struct_to_ptr(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let i32_ty = core::I32::new(db).as_type();
        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i32_ty), (Symbol::new("y"), i32_ty)],
        );
        let result =
            converter.materialize(db, location, value, struct_ty, core::Ptr::new(db).as_type());
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_struct_to_ptr_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_struct_to_ptr(db));
    }

    #[salsa::tracked]
    fn do_materialize_ptr_to_struct(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let i32_ty = core::I32::new(db).as_type();
        let struct_ty = adt::struct_type(
            db,
            Symbol::new("Point"),
            vec![(Symbol::new("x"), i32_ty), (Symbol::new("y"), i32_ty)],
        );
        let result =
            converter.materialize(db, location, value, core::Ptr::new(db).as_type(), struct_ty);
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_ptr_to_struct_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_ptr_to_struct(db));
    }

    #[salsa::tracked]
    fn do_materialize_closure_to_ptr(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let func_ty = core::I32::new(db).as_type();
        let closure_ty = closure::Closure::new(db, func_ty).as_type();
        let result = converter.materialize(
            db,
            location,
            value,
            closure_ty,
            core::Ptr::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_closure_to_ptr_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_closure_to_ptr(db));
    }

    #[salsa::tracked]
    fn do_materialize_between_struct_types(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let i32_ty = core::I32::new(db).as_type();
        let from_ty = adt::struct_type(db, Symbol::new("Base"), vec![(Symbol::new("tag"), i32_ty)]);
        let to_ty = adt::struct_type(
            db,
            Symbol::new("Variant"),
            vec![(Symbol::new("tag"), i32_ty), (Symbol::new("val"), i32_ty)],
        );
        let result = converter.materialize(db, location, value, from_ty, to_ty);
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_between_struct_types_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_between_struct_types(db));
    }

    // --- Boxing: i32 -> ptr (generates alloc + store ops) ---

    #[salsa::tracked]
    fn do_materialize_i32_to_ptr(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            core::I32::new(db).as_type(),
            core::Ptr::new(db).as_type(),
        );
        // Should produce Ops (alloc + store + identity)
        matches!(result, Some(MaterializeResult::Ops(_)))
    }

    #[salsa_test]
    fn test_materialize_i32_to_ptr_generates_ops(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_i32_to_ptr(db));
    }

    // --- Unboxing: ptr -> i32 (generates load op) ---

    #[salsa::tracked]
    fn do_materialize_ptr_to_i32(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            core::Ptr::new(db).as_type(),
            core::I32::new(db).as_type(),
        );
        // Should produce Ops (clif.load)
        matches!(result, Some(MaterializeResult::Ops(_)))
    }

    #[salsa_test]
    fn test_materialize_ptr_to_i32_generates_ops(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_ptr_to_i32(db));
    }

    // --- Unboxing: ptr -> i64 (generates load op) ---

    #[salsa::tracked]
    fn do_materialize_ptr_to_i64(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            core::Ptr::new(db).as_type(),
            core::I64::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::Ops(_)))
    }

    #[salsa_test]
    fn test_materialize_ptr_to_i64_generates_ops(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_ptr_to_i64(db));
    }

    // --- Nil conversions ---

    #[salsa::tracked]
    fn do_materialize_nil_to_ptr(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            core::Nil::new(db).as_type(),
            core::Ptr::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::Ops(ref ops)) if ops.len() == 1)
    }

    #[salsa_test]
    fn test_materialize_nil_to_ptr_emits_null_iconst(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_nil_to_ptr(db));
    }

    #[salsa::tracked]
    fn do_materialize_ptr_to_nil(db: &dyn salsa::Database) -> bool {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = native_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let value = Value::new(db, ValueDef::BlockArg(BlockId::fresh()), 0);

        let result = converter.materialize(
            db,
            location,
            value,
            core::Ptr::new(db).as_type(),
            core::Nil::new(db).as_type(),
        );
        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_ptr_to_nil_is_noop(db: &salsa::DatabaseImpl) {
        assert!(do_materialize_ptr_to_nil(db));
    }
}
