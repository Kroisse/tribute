//! WASM type converter for IR-level type transformations.
//!
//! This module provides a `TypeConverter` configuration for converting
//! high-level Tribute types to their WASM representations during IR lowering.
//!
//! ## Type Conversion Rules
//!
//! | Source Type         | Target Type     | Notes                              |
//! |---------------------|-----------------|-------------------------------------|
//! | `tribute_rt.int`    | `core.i32`      | Arbitrary precision → i32 (Phase 1) |
//! | `tribute_rt.nat`    | `core.i32`      | Arbitrary precision → i32 (Phase 1) |
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

use tribute_ir::dialect::{ability, closure, tribute, tribute_rt};
use trunk_ir::dialect::{adt, trampoline};
use trunk_ir::dialect::{core, wasm};
use trunk_ir::rewrite::{MaterializeResult, OpVec, TypeConverter};
use trunk_ir::{Attribute, Symbol};
use trunk_ir::{DialectOp, DialectType, Type};
use trunk_ir_wasm_backend::passes::trampoline_to_wasm::{
    continuation_adt_type, resume_wrapper_adt_type, step_adt_type,
};

#[cfg(test)]
use trunk_ir::dialect::arith;

/// Get the canonical Closure ADT type.
///
/// Layout: (table_idx: i32, env: anyref)
///
/// IMPORTANT: Must use wasm::Anyref (not tribute_rt::Any) to ensure consistent
/// type identity for emit lookups. This matches the pattern used by step_adt_type.
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

/// Helper to generate i31 unboxing operations (ref_cast to i31ref + i31_get_s).
///
/// This is used when converting anyref-typed values back to i32, such as
/// extracting values from Step structs which store all values as anyref.
fn unbox_via_i31<'db>(
    db: &'db dyn salsa::Database,
    location: trunk_ir::Location<'db>,
    value: trunk_ir::Value<'db>,
) -> MaterializeResult<'db> {
    let i31ref_ty = wasm::I31ref::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();

    // Cast anyref to i31ref
    let cast_op = wasm::ref_cast(db, location, value, i31ref_ty, i31ref_ty, None);
    // Extract i32 from i31ref
    let get_op = wasm::i31_get_s(db, location, cast_op.as_operation().result(db, 0), i32_ty);

    let mut ops = OpVec::new();
    ops.push(cast_op.as_operation());
    ops.push(get_op.as_operation());
    MaterializeResult::Ops(ops)
}

/// Create a TypeConverter configured for WASM backend type conversions.
///
/// This converter handles the IR-level type transformations needed during
/// lowering passes. It complements the emit-phase `gc_types::type_to_field_type`
/// by performing conversions at the IR level.
///
/// The returned TypeConverter is `'static` and can be stored in patterns
/// or shared across passes.
///
/// # Example
///
/// ```
/// # use tribute_ir::dialect::tribute_rt;
/// # use trunk_ir::dialect::core;
/// # use trunk_ir::DialectType;
/// use tribute_wasm_backend::type_converter::wasm_type_converter;
///
/// # let db = salsa::DatabaseImpl::default();
/// let converter = wasm_type_converter();
///
/// // Convert tribute_rt.int to core.i32
/// # let int_ty = tribute_rt::Int::new(&db).as_type();
/// let i32_ty = converter.convert_type(&db, int_ty).unwrap();
/// # assert_eq!(i32_ty, core::I32::new(&db).as_type());
/// ```
pub fn wasm_type_converter() -> TypeConverter {
    TypeConverter::new()
        // Convert unresolved tribute.type → concrete types
        // This handles cases where typeck didn't fully resolve type names
        .add_conversion(|db, ty| {
            if !tribute::is_unresolved_type(db, ty) {
                return None;
            }
            // Get the type name from the name attribute
            let name = ty.get_attr(db, Symbol::new("name"))?;
            let Attribute::Symbol(name_sym) = name else {
                return None;
            };
            // Convert known type names to concrete types
            name_sym.with_str(|name_str| match name_str {
                "Int" => Some(core::I32::new(db).as_type()),
                "Nat" => Some(core::I32::new(db).as_type()),
                "Bool" => Some(core::I32::new(db).as_type()),
                "Float" => Some(core::F64::new(db).as_type()),
                "String" => Some(core::String::new(db).as_type()),
                _ => None, // Unknown type - leave as is
            })
        })
        // Convert tribute_rt.int → core.i32 (Phase 1: arbitrary precision as i32)
        .add_conversion(|db, ty| {
            tribute_rt::Int::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert tribute_rt.nat → core.i32 (Phase 1: arbitrary precision as i32)
        .add_conversion(|db, ty| {
            tribute_rt::Nat::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert tribute_rt.bool → core.i32 (boolean as i32)
        .add_conversion(|db, ty| {
            tribute_rt::Bool::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert core.i1 → core.i32 (WASM doesn't have i1, use i32)
        .add_conversion(|db, ty| {
            core::I::<1>::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert tribute_rt.float → core.f64 (float as f64)
        .add_conversion(|db, ty| {
            tribute_rt::Float::from_type(db, ty).map(|_| core::F64::new(db).as_type())
        })
        // Convert tribute_rt.intref → wasm.i31ref (boxed integer reference)
        .add_conversion(|db, ty| {
            tribute_rt::Intref::from_type(db, ty).map(|_| wasm::I31ref::new(db).as_type())
        })
        // Convert tribute_rt.any → wasm.anyref (any reference type)
        .add_conversion(|db, ty| {
            tribute_rt::Any::from_type(db, ty).map(|_| wasm::Anyref::new(db).as_type())
        })
        // Convert adt.typeref → wasm.structref (generic struct reference)
        .add_conversion(|db, ty| {
            if adt::is_typeref(db, ty) {
                Some(wasm::Structref::new(db).as_type())
            } else {
                None
            }
        })
        // Convert closure.closure → adt.struct(name="_closure")
        // This ensures emit can identify closures by ADT name rather than tribute-ir type
        .add_conversion(|db, ty| {
            if closure::Closure::from_type(db, ty).is_some() {
                Some(closure_adt_type(db))
            } else {
                None
            }
        })
        // Convert ability.evidence_ptr → wasm.anyref (evidence is a runtime handle)
        .add_conversion(|db, ty| {
            if ability::EvidencePtr::from_type(db, ty).is_some() {
                Some(wasm::Anyref::new(db).as_type())
            } else {
                None
            }
        })
        // Materialization for struct type bridging
        .add_materialization(|db, location, value, from_ty, to_ty| {
            // Same type - no materialization needed
            if from_ty == to_ty {
                return MaterializeResult::NoOp;
            }

            // adt.typeref ↔ wasm.structref: same runtime representation, no cast needed
            // This covers the common case of passing structs between typed and untyped contexts
            let from_is_typeref = adt::is_typeref(db, from_ty);
            let to_is_typeref = adt::is_typeref(db, to_ty);
            let from_is_structref = wasm::Structref::from_type(db, from_ty).is_some();
            let to_is_structref = wasm::Structref::from_type(db, to_ty).is_some();

            if (from_is_typeref && to_is_structref) || (from_is_structref && to_is_typeref) {
                return MaterializeResult::NoOp;
            }

            // Both are struct-like types but need actual bridging - generate ref_cast
            let from_is_struct_like = is_struct_like(db, from_ty);
            let to_is_struct_like = is_struct_like(db, to_ty);

            if from_is_struct_like && to_is_struct_like {
                let cast_op = wasm::ref_cast(db, location, value, to_ty, to_ty, None);
                let mut ops = OpVec::new();
                ops.push(cast_op.as_operation());
                return MaterializeResult::Ops(ops);
            }

            // anyref → concrete struct type: use ref_cast
            // This handles cases like closure env or resume wrapper parameters
            // that are passed as anyref for uniform calling convention.
            // We EXCLUDE wasm.anyref as target since that's handled by primitive equivalence.
            let from_is_anyref = wasm::Anyref::from_type(db, from_ty).is_some()
                || tribute_rt::Any::from_type(db, from_ty).is_some();
            let to_is_abstract_anyref = wasm::Anyref::from_type(db, to_ty).is_some();
            if from_is_anyref && to_is_struct_like && !to_is_abstract_anyref {
                // Convert trampoline types to their ADT representation for the cast
                let target_ty = if trampoline::ResumeWrapper::from_type(db, to_ty).is_some() {
                    resume_wrapper_adt_type(db)
                } else if trampoline::Step::from_type(db, to_ty).is_some() {
                    step_adt_type(db)
                } else if trampoline::Continuation::from_type(db, to_ty).is_some() {
                    continuation_adt_type(db)
                } else {
                    to_ty
                };
                let cast_op = wasm::ref_cast(db, location, value, target_ty, target_ty, None);
                let mut ops = OpVec::new();
                ops.push(cast_op.as_operation());
                return MaterializeResult::Ops(ops);
            }

            // Cannot materialize this conversion
            MaterializeResult::Skip
        })
        // Primitive type equivalence: tribute_rt types are represented as core types
        // These are no-op conversions (same underlying representation)
        .add_materialization(|db, location, value, from_ty, to_ty| {
            // tribute_rt.int → core.i32 (same representation)
            if tribute_rt::Int::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.nat → core.i32 (same representation)
            if tribute_rt::Nat::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.bool → core.i32 (same representation)
            if tribute_rt::Bool::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // core.i1 → core.i32 (same representation for wasm)
            if core::I::<1>::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.float → core.f64 (same representation)
            if tribute_rt::Float::from_type(db, from_ty).is_some()
                && core::F64::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.intref → wasm.i31ref (same representation)
            if tribute_rt::Intref::from_type(db, from_ty).is_some()
                && wasm::I31ref::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // tribute_rt.any → wasm.anyref (same representation)
            if tribute_rt::Any::from_type(db, from_ty).is_some()
                && wasm::Anyref::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // wasm.i31ref → wasm.anyref (i31ref is a subtype of anyref)
            if wasm::I31ref::from_type(db, from_ty).is_some()
                && wasm::Anyref::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // wasm.structref → wasm.anyref (structref is a subtype of anyref)
            if wasm::Structref::from_type(db, from_ty).is_some()
                && wasm::Anyref::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // closure.closure → adt.struct(name="_closure") (same representation)
            if closure::Closure::from_type(db, from_ty).is_some() && to_ty == closure_adt_type(db) {
                return MaterializeResult::NoOp;
            }
            // closure.closure → wasm.structref (subtype relationship)
            if closure::Closure::from_type(db, from_ty).is_some()
                && wasm::Structref::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // wasm.anyref → core.i32 (unbox via i31)
            // This is used when extracting values from Step (which stores anyref)
            if wasm::Anyref::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return unbox_via_i31(db, location, value);
            }
            // tribute_rt.any → core.i32 (unbox via i31, same as wasm.anyref)
            if tribute_rt::Any::from_type(db, from_ty).is_some()
                && core::I32::from_type(db, to_ty).is_some()
            {
                return unbox_via_i31(db, location, value);
            }
            // wasm.anyref → tribute_rt.int (unbox via i31)
            // Same as anyref -> core.i32, tribute_rt.int is represented as i32
            if wasm::Anyref::from_type(db, from_ty).is_some()
                && tribute_rt::Int::from_type(db, to_ty).is_some()
            {
                return unbox_via_i31(db, location, value);
            }
            // tribute_rt.any → tribute_rt.int (unbox via i31)
            // Used when extracting values from state structs (stored as anyref)
            if tribute_rt::Any::from_type(db, from_ty).is_some()
                && tribute_rt::Int::from_type(db, to_ty).is_some()
            {
                return unbox_via_i31(db, location, value);
            }
            // wasm.anyref → core.ptr (treat pointer as anyref subtype)
            if wasm::Anyref::from_type(db, from_ty).is_some()
                && core::Ptr::from_type(db, to_ty).is_some()
            {
                // Pointers and anyref have the same representation in WasmGC
                return MaterializeResult::NoOp;
            }
            // tribute_rt.any → core.ptr (same representation in WasmGC)
            if tribute_rt::Any::from_type(db, from_ty).is_some()
                && core::Ptr::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // wasm.anyref → core.nil (unit type, value ignored)
            // This is used when functions return Nil but the trampoline stores anyref
            if wasm::Anyref::from_type(db, from_ty).is_some()
                && core::Nil::from_type(db, to_ty).is_some()
            {
                // Nil is a unit type - the anyref value is simply discarded
                return MaterializeResult::NoOp;
            }
            // tribute_rt.any → core.nil (unit type, value ignored)
            if tribute_rt::Any::from_type(db, from_ty).is_some()
                && core::Nil::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            // trampoline.step → _Step ADT (same representation after conversion)
            if trampoline::Step::from_type(db, from_ty).is_some() && to_ty == step_adt_type(db) {
                return MaterializeResult::NoOp;
            }
            // trampoline.continuation → _Continuation ADT (same representation)
            if trampoline::Continuation::from_type(db, from_ty).is_some()
                && to_ty == continuation_adt_type(db)
            {
                return MaterializeResult::NoOp;
            }
            // trampoline.resume_wrapper → _ResumeWrapper ADT (same representation)
            if trampoline::ResumeWrapper::from_type(db, from_ty).is_some()
                && to_ty == resume_wrapper_adt_type(db)
            {
                return MaterializeResult::NoOp;
            }
            // ability.evidence_ptr → wasm.anyref (same representation)
            // Evidence pointers are runtime handles stored as anyref
            if ability::EvidencePtr::from_type(db, from_ty).is_some()
                && wasm::Anyref::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }
            MaterializeResult::Skip
        })
}

/// Check if a type is a struct-like reference type.
///
/// This includes `wasm.structref`, `wasm.anyref`, and ADT typeref types.
fn is_struct_like(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    // wasm.structref or wasm.anyref
    if wasm::Structref::from_type(db, ty).is_some() || wasm::Anyref::from_type(db, ty).is_some() {
        return true;
    }

    // adt.typeref (struct types)
    if adt::is_typeref(db, ty) {
        return true;
    }

    // Check for variant instance types (have is_variant attribute)
    if adt::is_variant_instance_type(db, ty) {
        return true;
    }

    // trampoline types that get lowered to ADT structs
    if trampoline::Step::from_type(db, ty).is_some()
        || trampoline::Continuation::from_type(db, ty).is_some()
        || trampoline::ResumeWrapper::from_type(db, ty).is_some()
    {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

    #[salsa_test]
    fn test_convert_tribute_rt_int(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // Create tribute_rt.int type
        let int_ty = tribute_rt::Int::new(db).as_type();

        // Convert to core.i32
        let result = converter.convert_type(db, int_ty);

        assert!(result.is_some());
        let converted = result.unwrap();
        let expected = core::I32::new(db).as_type();
        assert_eq!(converted, expected);
    }

    #[salsa_test]
    fn test_convert_tribute_rt_nat(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // Create tribute_rt.nat type
        let nat_ty = tribute_rt::Nat::new(db).as_type();

        // Convert to core.i32
        let result = converter.convert_type(db, nat_ty);

        assert!(result.is_some());
        let converted = result.unwrap();
        let expected = core::I32::new(db).as_type();
        assert_eq!(converted, expected);
    }

    #[salsa_test]
    fn test_convert_tribute_rt_bool(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // Create tribute_rt.bool type
        let bool_ty = tribute_rt::Bool::new(db).as_type();

        // Convert to core.i32
        let result = converter.convert_type(db, bool_ty);

        assert!(result.is_some());
        let converted = result.unwrap();
        let expected = core::I32::new(db).as_type();
        assert_eq!(converted, expected);
    }

    #[salsa_test]
    fn test_convert_tribute_rt_float(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // Create tribute_rt.float type
        let float_ty = tribute_rt::Float::new(db).as_type();

        // Convert to core.f64
        let result = converter.convert_type(db, float_ty);

        assert!(result.is_some());
        let converted = result.unwrap();
        let expected = core::F64::new(db).as_type();
        assert_eq!(converted, expected);
    }

    #[salsa_test]
    fn test_convert_tribute_rt_intref(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // Create tribute_rt.intref type
        let intref_ty = tribute_rt::Intref::new(db).as_type();

        // Convert to wasm.i31ref
        let result = converter.convert_type(db, intref_ty);

        assert!(result.is_some());
        let converted = result.unwrap();
        let expected = wasm::I31ref::new(db).as_type();
        assert_eq!(converted, expected);
    }

    #[salsa_test]
    fn test_convert_tribute_rt_any(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // Create tribute_rt.any type
        let any_ty = tribute_rt::Any::new(db).as_type();

        // Convert to wasm.anyref
        let result = converter.convert_type(db, any_ty);

        assert!(result.is_some());
        let converted = result.unwrap();
        let expected = wasm::Anyref::new(db).as_type();
        assert_eq!(converted, expected);
    }

    #[salsa_test]
    fn test_no_conversion_for_core_types(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // core.i32 should not be converted
        let i32_ty = core::I32::new(db).as_type();
        let result = converter.convert_type(db, i32_ty);
        assert!(result.is_none());

        // core.i64 should not be converted
        let i64_ty = core::I64::new(db).as_type();
        let result = converter.convert_type(db, i64_ty);
        assert!(result.is_none());
    }

    #[salsa_test]
    fn test_convert_adt_typeref(db: &salsa::DatabaseImpl) {
        let converter = wasm_type_converter();

        // Create an adt.typeref type
        let typeref_ty = adt::typeref(db, Symbol::new("MyStruct"));

        // Convert to wasm.structref
        let result = converter.convert_type(db, typeref_ty);

        assert!(result.is_some());
        let converted = result.unwrap();
        assert_eq!(converted.dialect(db), wasm::DIALECT_NAME());
        assert_eq!(converted.name(db), Symbol::new("structref"));
    }

    /// Test that tribute_rt.int → core.i32 materialization returns NoOp.
    /// These are the same underlying representation, no conversion needed.
    #[salsa::tracked]
    fn do_materialize_primitive_equivalence_test(db: &dyn salsa::Database) -> bool {
        use trunk_ir::rewrite::MaterializeResult;
        use trunk_ir::{Attribute, Location, PathId, Span, Value, ValueDef};

        let converter = wasm_type_converter();
        let path = PathId::new(db, "test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let int_ty = tribute_rt::Int::new(db).as_type();
        let const_op = arith::r#const(db, location, int_ty, Attribute::IntBits(42));
        let value = Value::new(db, ValueDef::OpResult(const_op.as_operation()), 0);

        let i32_ty = core::I32::new(db).as_type();
        let result = converter.materialize(db, location, value, int_ty, i32_ty);

        matches!(result, Some(MaterializeResult::NoOp))
    }

    #[salsa_test]
    fn test_materialize_tribute_rt_int_to_core_i32_is_noop(db: &salsa::DatabaseImpl) {
        let is_noop = do_materialize_primitive_equivalence_test(db);
        assert!(
            is_noop,
            "tribute_rt.Int → core.I32 should be NoOp (same representation)"
        );
    }
}
