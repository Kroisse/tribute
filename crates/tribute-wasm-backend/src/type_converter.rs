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
//! | `adt.typeref<T>`    | `wasm.structref`| Generic struct reference            |
//!
//! ## Materializations
//!
//! When values need to be bridged between different struct types (e.g., from
//! a base enum type to a variant type), the converter can insert `wasm.ref_cast`
//! operations.

use tribute_ir::dialect::{adt, tribute_rt};
use trunk_ir::dialect::{core, wasm};
use trunk_ir::rewrite::{MaterializeResult, OpVec, TypeConverter};
use trunk_ir::{DialectOp, DialectType, Type};

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
/// ```ignore
/// use tribute_wasm_backend::type_converter::wasm_type_converter;
///
/// let converter = wasm_type_converter();
///
/// // Convert tribute.int to core.i32
/// let i32_ty = converter.convert_type(db, tribute_int_ty);
/// ```
pub fn wasm_type_converter() -> TypeConverter {
    TypeConverter::new()
        // Convert tribute_rt.int → core.i32 (Phase 1: arbitrary precision as i32)
        .add_conversion(|db, ty| {
            tribute_rt::Int::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert tribute_rt.nat → core.i32 (Phase 1: arbitrary precision as i32)
        .add_conversion(|db, ty| {
            tribute_rt::Nat::from_type(db, ty).map(|_| core::I32::new(db).as_type())
        })
        // Convert adt.typeref → wasm.structref (generic struct reference)
        .add_conversion(|db, ty| {
            if adt::is_typeref(db, ty) {
                Some(wasm::Structref::new(db).as_type())
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

            // Both are structref types - generate ref_cast
            let from_is_structref = is_struct_like(db, from_ty);
            let to_is_structref = is_struct_like(db, to_ty);

            if from_is_structref && to_is_structref {
                let cast_op = wasm::ref_cast(db, location, value, to_ty, to_ty);
                let mut ops = OpVec::new();
                ops.push(cast_op.as_operation());
                return MaterializeResult::Ops(ops);
            }

            // Cannot materialize this conversion
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
    adt::is_variant_instance_type(db, ty)
}

/// Convert a function type's parameter and return types.
///
/// This is a utility function for converting function signatures.
/// Each parameter and the return type are converted using the type converter.
pub fn convert_function_type<'db>(
    db: &'db dyn salsa::Database,
    converter: &TypeConverter,
    param_types: &[Type<'db>],
    return_type: Type<'db>,
) -> (Vec<Type<'db>>, Type<'db>) {
    let converted_params: Vec<Type<'db>> = param_types
        .iter()
        .map(|ty| converter.convert_type(db, *ty).unwrap_or(*ty))
        .collect();

    let converted_return = converter
        .convert_type(db, return_type)
        .unwrap_or(return_type);

    (converted_params, converted_return)
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
}
