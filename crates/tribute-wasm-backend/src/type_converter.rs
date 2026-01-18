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

use tribute_ir::dialect::{adt, tribute, tribute_rt};
use trunk_ir::dialect::{core, wasm};
use trunk_ir::rewrite::{MaterializeResult, OpVec, TypeConverter};
use trunk_ir::{Attribute, Symbol};
use trunk_ir::{DialectOp, DialectType, Type};

#[cfg(test)]
use trunk_ir::dialect::arith;

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

            // Cannot materialize this conversion
            MaterializeResult::Skip
        })
        // Primitive type equivalence: tribute_rt types are represented as core types
        // These are no-op conversions (same underlying representation)
        .add_materialization(|db, _location, _value, from_ty, to_ty| {
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
            MaterializeResult::Skip
        })
        // Boxing: primitive types → i31ref/anyref
        // Generate wasm ops directly (wasm.ref_i31) since materialization runs
        // after tribute_rt_to_wasm pass
        .add_materialization(|db, location, value, from_ty, to_ty| {
            let to_i31ref = wasm::I31ref::from_type(db, to_ty).is_some();
            let to_anyref = wasm::Anyref::from_type(db, to_ty).is_some();

            if !to_i31ref && !to_anyref {
                return MaterializeResult::Skip;
            }

            // Int/Nat/I32 → i31ref/anyref: use wasm.ref_i31 directly
            let is_int_like = tribute_rt::Int::from_type(db, from_ty).is_some()
                || tribute_rt::Nat::from_type(db, from_ty).is_some()
                || core::I32::from_type(db, from_ty).is_some();

            if is_int_like {
                let i31ref_ty = wasm::I31ref::new(db).as_type();
                // wasm.ref_i31: i32 -> i31ref
                let ref_i31_op = wasm::ref_i31(db, location, value, i31ref_ty);
                return MaterializeResult::single(ref_i31_op.as_operation());
            }

            MaterializeResult::Skip
        })
        // Boxing: f64 → anyref (via wasm.struct_new for BoxedF64)
        .add_materialization(|db, location, value, from_ty, to_ty| {
            use crate::gc_types::BOXED_F64_IDX;

            let to_anyref = wasm::Anyref::from_type(db, to_ty).is_some();
            if !to_anyref {
                return MaterializeResult::Skip;
            }

            // f64 → anyref: use wasm.struct_new (BoxedF64 struct)
            if core::F64::from_type(db, from_ty).is_some() {
                let anyref_ty = wasm::Anyref::new(db).as_type();
                let struct_new_op =
                    wasm::struct_new(db, location, vec![value], anyref_ty, BOXED_F64_IDX);
                return MaterializeResult::single(struct_new_op.as_operation());
            }

            MaterializeResult::Skip
        })
        // Unboxing: i31ref/anyref → primitive types
        // Generate wasm ops directly (wasm.ref_cast + wasm.i31_get_s)
        .add_materialization(|db, location, value, from_ty, to_ty| {
            let from_i31ref = wasm::I31ref::from_type(db, from_ty).is_some();
            let from_anyref = wasm::Anyref::from_type(db, from_ty).is_some();

            if !from_i31ref && !from_anyref {
                return MaterializeResult::Skip;
            }

            // i31ref/anyref → Int/Nat/I32: use wasm.i31_get_s directly
            let is_int_like = tribute_rt::Int::from_type(db, to_ty).is_some()
                || tribute_rt::Nat::from_type(db, to_ty).is_some()
                || core::I32::from_type(db, to_ty).is_some();

            if is_int_like {
                let i32_ty = core::I32::new(db).as_type();
                let i31ref_ty = wasm::I31ref::new(db).as_type();

                // anyref needs ref_cast to i31ref first (abstract type, no type_idx)
                if from_anyref {
                    let ref_cast = wasm::ref_cast(db, location, value, i31ref_ty, i31ref_ty, None);
                    // wasm.i31_get_s: i31ref -> i32 (signed)
                    let i31_get_op = wasm::i31_get_s(db, location, ref_cast.result(db), i32_ty);
                    return MaterializeResult::ops([
                        ref_cast.as_operation(),
                        i31_get_op.as_operation(),
                    ]);
                }

                // i31ref can be unboxed directly
                // wasm.i31_get_s: i31ref -> i32 (signed)
                let i31_get_op = wasm::i31_get_s(db, location, value, i32_ty);
                return MaterializeResult::single(i31_get_op.as_operation());
            }

            MaterializeResult::Skip
        })
        // Unboxing: anyref → f64 (via wasm.struct_get from BoxedF64)
        .add_materialization(|db, location, value, from_ty, to_ty| {
            use crate::gc_types::BOXED_F64_IDX;

            let from_anyref = wasm::Anyref::from_type(db, from_ty).is_some();
            if !from_anyref {
                return MaterializeResult::Skip;
            }

            // anyref → f64: use wasm.ref_cast + wasm.struct_get
            if core::F64::from_type(db, to_ty).is_some() {
                let anyref_ty = wasm::Anyref::new(db).as_type();
                let f64_ty = core::F64::new(db).as_type();

                // Cast anyref to BoxedF64 struct first
                let cast_op = wasm::ref_cast(
                    db,
                    location,
                    value,
                    anyref_ty,
                    anyref_ty,
                    Some(BOXED_F64_IDX),
                );
                // wasm.struct_get extracts field 0 (the f64 value) from BoxedF64
                let get_op =
                    wasm::struct_get(db, location, cast_op.result(db), f64_ty, BOXED_F64_IDX, 0);
                return MaterializeResult::ops([cast_op.as_operation(), get_op.as_operation()]);
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
    adt::is_variant_instance_type(db, ty)
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

    /// Helper: test boxing materialization (i32 → i31ref)
    #[salsa::tracked]
    fn do_materialize_box_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{Attribute, Location, PathId, Span, Value, ValueDef};

        let converter = wasm_type_converter();
        let path = PathId::new(db, "test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let i32_ty = core::I32::new(db).as_type();
        let const_op = arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        let value = Value::new(db, ValueDef::OpResult(const_op.as_operation()), 0);

        let i31ref_ty = wasm::I31ref::new(db).as_type();
        let result = converter.materialize(db, location, value, i32_ty, i31ref_ty);

        match result {
            Some(trunk_ir::rewrite::MaterializeResult::Ops(ops)) => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_box_int_to_i31ref(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_materialize_box_test(db);
        // Now generates wasm.ref_i31 directly (not tribute_rt.box_int)
        assert_eq!(dialect, wasm::DIALECT_NAME());
        assert_eq!(name, wasm::REF_I31());
    }

    /// Helper: test unboxing materialization (i31ref → i32)
    #[salsa::tracked]
    fn do_materialize_unbox_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{Attribute, Location, PathId, Span, Value, ValueDef};

        let converter = wasm_type_converter();
        let path = PathId::new(db, "test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let i31ref_ty = wasm::I31ref::new(db).as_type();
        // Use arith.const with i31ref type (value doesn't matter for materialize test)
        let const_op = arith::r#const(db, location, i31ref_ty, Attribute::IntBits(42));
        let value = Value::new(db, ValueDef::OpResult(const_op.as_operation()), 0);

        let i32_ty = core::I32::new(db).as_type();
        let result = converter.materialize(db, location, value, i31ref_ty, i32_ty);

        match result {
            Some(trunk_ir::rewrite::MaterializeResult::Ops(ops)) => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_unbox_i31ref_to_int(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_materialize_unbox_test(db);
        // Now generates wasm.i31_get_s directly (not tribute_rt.unbox_int)
        assert_eq!(dialect, wasm::DIALECT_NAME());
        assert_eq!(name, wasm::I31_GET_S());
    }

    /// Helper: test boxing tribute_rt.int → anyref
    #[salsa::tracked]
    fn do_materialize_int_to_anyref_test(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{Attribute, Location, PathId, Span, Value, ValueDef};

        let converter = wasm_type_converter();
        let path = PathId::new(db, "test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let int_ty = tribute_rt::Int::new(db).as_type();
        let const_op = arith::r#const(db, location, int_ty, Attribute::IntBits(42));
        let value = Value::new(db, ValueDef::OpResult(const_op.as_operation()), 0);

        let anyref_ty = wasm::Anyref::new(db).as_type();
        let result = converter.materialize(db, location, value, int_ty, anyref_ty);

        match result {
            Some(trunk_ir::rewrite::MaterializeResult::Ops(ops)) => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_tribute_rt_int_to_anyref(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_materialize_int_to_anyref_test(db);
        // Now generates wasm.ref_i31 directly (not tribute_rt.box_int)
        assert_eq!(dialect, wasm::DIALECT_NAME());
        assert_eq!(name, wasm::REF_I31());
    }

    /// Helper: test unboxing anyref → i32 (should generate ref_cast + unbox_int)
    #[salsa::tracked]
    fn do_materialize_unbox_anyref_test(
        db: &dyn salsa::Database,
    ) -> (usize, Vec<(Symbol, Symbol)>) {
        use trunk_ir::{Attribute, Location, PathId, Span, Value, ValueDef};

        let converter = wasm_type_converter();
        let path = PathId::new(db, "test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let anyref_ty = wasm::Anyref::new(db).as_type();
        let const_op = arith::r#const(db, location, anyref_ty, Attribute::IntBits(42));
        let value = Value::new(db, ValueDef::OpResult(const_op.as_operation()), 0);

        let i32_ty = core::I32::new(db).as_type();
        let result = converter.materialize(db, location, value, anyref_ty, i32_ty);

        match result {
            Some(trunk_ir::rewrite::MaterializeResult::Ops(ops)) => {
                let op_info: Vec<_> = ops.iter().map(|op| (op.dialect(db), op.name(db))).collect();
                (ops.len(), op_info)
            }
            _ => (0, vec![]),
        }
    }

    #[salsa_test]
    fn test_materialize_unbox_anyref_to_i32(db: &salsa::DatabaseImpl) {
        let (count, ops) = do_materialize_unbox_anyref_test(db);
        // Should generate 2 ops: ref_cast + i31_get_s (now generates wasm ops directly)
        assert_eq!(count, 2);
        assert_eq!(ops[0], (wasm::DIALECT_NAME(), wasm::REF_CAST()));
        assert_eq!(ops[1], (wasm::DIALECT_NAME(), wasm::I31_GET_S()));
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
