//! Generic type converter for target-agnostic IR transformations.
//!
//! This module provides a `TypeConverter` configuration for converting
//! high-level Tribute types to their core representations. This converter
//! handles target-agnostic transformations that apply to all backends.
//!
//! ## Type Conversion Rules
//!
//! | Source Type         | Target Type     | Notes                              |
//! |---------------------|-----------------|-------------------------------------|
//! | `tribute_rt.int`    | `core.i32`      | Arbitrary precision → i32 (Phase 1) |
//! | `tribute_rt.nat`    | `core.i32`      | Arbitrary precision → i32 (Phase 1) |
//! | `tribute_rt.bool`   | `core.i32`      | Boolean as i32                      |
//! | `tribute_rt.float`  | `core.f64`      | Float as f64                        |
//!
//! Backend-specific type conversions (e.g., `core.i1 → core.i32`,
//! `tribute_rt.any` → `wasm.anyref`) are handled by backend-specific
//! type converters.

use tribute_ir::dialect::tribute_rt;
use trunk_ir::dialect::{adt, core};
use trunk_ir::rewrite::{MaterializeResult, TypeConverter};
use trunk_ir::{DialectOp, DialectType, Type};

/// Create a TypeConverter configured for target-agnostic type conversions.
///
/// This converter handles the IR-level type transformations that are common
/// across all backends. Backend-specific converters can extend this with
/// additional conversions.
///
/// # Example
///
/// ```
/// # use tribute_ir::dialect::tribute_rt;
/// # use trunk_ir::dialect::core;
/// # use trunk_ir::DialectType;
/// use tribute_passes::type_converter::generic_type_converter;
///
/// # let db = salsa::DatabaseImpl::default();
/// let converter = generic_type_converter();
///
/// // Convert tribute_rt.int to core.i32
/// # let int_ty = tribute_rt::Int::new(&db).as_type();
/// let i32_ty = converter.convert_type(&db, int_ty).unwrap();
/// # assert_eq!(i32_ty, core::I32::new(&db).as_type());
/// ```
pub fn generic_type_converter() -> TypeConverter {
    TypeConverter::new()
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
        // Convert tribute_rt.float → core.f64 (float as f64)
        .add_conversion(|db, ty| {
            tribute_rt::Float::from_type(db, ty).map(|_| core::F64::new(db).as_type())
        })
        // Primitive type equivalence materializations
        // These are no-op conversions (same underlying representation)
        .add_materialization(|db, _location, _value, from_ty, to_ty| {
            // Same type - no materialization needed
            if from_ty == to_ty {
                return MaterializeResult::NoOp;
            }

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
            // tribute_rt.float → core.f64 (same representation)
            if tribute_rt::Float::from_type(db, from_ty).is_some()
                && core::F64::from_type(db, to_ty).is_some()
            {
                return MaterializeResult::NoOp;
            }

            MaterializeResult::Skip
        })
        // Boxing: primitive types → tribute_rt.any
        // Generate tribute_rt.box_* ops which will be lowered by tribute_rt_to_wasm
        .add_materialization(|db, location, value, from_ty, to_ty| {
            // Only handle conversions to tribute_rt.any
            if tribute_rt::Any::from_type(db, to_ty).is_none() {
                return MaterializeResult::Skip;
            }

            let any_ty = tribute_rt::Any::new(db).as_type();

            // Int/Nat/I32/I64 → any: use tribute_rt.box_int
            if tribute_rt::Int::from_type(db, from_ty).is_some()
                || tribute_rt::Nat::from_type(db, from_ty).is_some()
                || core::I32::from_type(db, from_ty).is_some()
                || core::I64::from_type(db, from_ty).is_some()
            {
                let box_op = tribute_rt::box_int(db, location, value, any_ty);
                return MaterializeResult::single(box_op.as_operation());
            }

            // Bool/I1 → any: use tribute_rt.box_bool
            if tribute_rt::Bool::from_type(db, from_ty).is_some()
                || core::I1::from_type(db, from_ty).is_some()
            {
                let box_op = tribute_rt::box_bool(db, location, value, any_ty);
                return MaterializeResult::single(box_op.as_operation());
            }

            // Float/F64 → any: use tribute_rt.box_float
            if tribute_rt::Float::from_type(db, from_ty).is_some()
                || core::F64::from_type(db, from_ty).is_some()
            {
                let box_op = tribute_rt::box_float(db, location, value, any_ty);
                return MaterializeResult::single(box_op.as_operation());
            }

            // adt.struct/adt.typeref → any: no-op (already a reference type)
            if adt::is_struct_type(db, from_ty) || adt::is_typeref(db, from_ty) {
                return MaterializeResult::NoOp;
            }

            MaterializeResult::Skip
        })
        // Unboxing: tribute_rt.any → primitive types
        // Generate tribute_rt.unbox_* ops which will be lowered by tribute_rt_to_wasm
        .add_materialization(|db, location, value, from_ty, to_ty| {
            // Only handle conversions from tribute_rt.any
            if tribute_rt::Any::from_type(db, from_ty).is_none() {
                return MaterializeResult::Skip;
            }

            // any → Int/I32/I64: use tribute_rt.unbox_int
            if tribute_rt::Int::from_type(db, to_ty).is_some()
                || core::I32::from_type(db, to_ty).is_some()
                || core::I64::from_type(db, to_ty).is_some()
            {
                let unbox_op = tribute_rt::unbox_int(db, location, value, to_ty);
                return MaterializeResult::single(unbox_op.as_operation());
            }

            // any → Nat: use tribute_rt.unbox_nat
            if tribute_rt::Nat::from_type(db, to_ty).is_some() {
                let unbox_op = tribute_rt::unbox_nat(db, location, value, to_ty);
                return MaterializeResult::single(unbox_op.as_operation());
            }

            // any → Bool/I1: use tribute_rt.unbox_bool
            if tribute_rt::Bool::from_type(db, to_ty).is_some()
                || core::I1::from_type(db, to_ty).is_some()
            {
                let unbox_op = tribute_rt::unbox_bool(db, location, value, to_ty);
                return MaterializeResult::single(unbox_op.as_operation());
            }

            // any → Float/F64: use tribute_rt.unbox_float
            if tribute_rt::Float::from_type(db, to_ty).is_some()
                || core::F64::from_type(db, to_ty).is_some()
            {
                let unbox_op = tribute_rt::unbox_float(db, location, value, to_ty);
                return MaterializeResult::single(unbox_op.as_operation());
            }

            // any → adt.struct/adt.typeref: no-op (already a reference type)
            if adt::is_struct_type(db, to_ty) || adt::is_typeref(db, to_ty) {
                return MaterializeResult::NoOp;
            }

            // Note: any → trampoline.resume_wrapper and any → core.array conversions
            // are handled by wasm_type_converter, not here, because they require
            // wasm.ref_cast operations that are only available after WASM lowering.

            MaterializeResult::Skip
        })
}

/// Check if two types are equivalent primitives (same underlying representation).
///
/// This is useful for determining if a type conversion requires actual
/// code generation or if it's just a type-level transformation.
pub fn are_equivalent_primitives(db: &dyn salsa::Database, from_ty: Type, to_ty: Type) -> bool {
    if from_ty == to_ty {
        return true;
    }

    // Int-like types (target-agnostic: excludes core.i1 which is backend-specific)
    let from_int_like = tribute_rt::Int::from_type(db, from_ty).is_some()
        || tribute_rt::Nat::from_type(db, from_ty).is_some()
        || tribute_rt::Bool::from_type(db, from_ty).is_some()
        || core::I32::from_type(db, from_ty).is_some();

    let to_int_like = tribute_rt::Int::from_type(db, to_ty).is_some()
        || tribute_rt::Nat::from_type(db, to_ty).is_some()
        || tribute_rt::Bool::from_type(db, to_ty).is_some()
        || core::I32::from_type(db, to_ty).is_some();

    if from_int_like && to_int_like {
        return true;
    }

    // Float-like types
    let from_float_like = tribute_rt::Float::from_type(db, from_ty).is_some()
        || core::F64::from_type(db, from_ty).is_some();

    let to_float_like = tribute_rt::Float::from_type(db, to_ty).is_some()
        || core::F64::from_type(db, to_ty).is_some();

    if from_float_like && to_float_like {
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
    fn test_convert_tribute_rt_int_to_core_i32(db: &salsa::DatabaseImpl) {
        let converter = generic_type_converter();
        let int_ty = tribute_rt::Int::new(db).as_type();
        let result = converter.convert_type(db, int_ty);
        assert_eq!(result, Some(core::I32::new(db).as_type()));
    }

    #[salsa_test]
    fn test_convert_tribute_rt_float_to_core_f64(db: &salsa::DatabaseImpl) {
        let converter = generic_type_converter();
        let float_ty = tribute_rt::Float::new(db).as_type();
        let result = converter.convert_type(db, float_ty);
        assert_eq!(result, Some(core::F64::new(db).as_type()));
    }

    #[salsa_test]
    fn test_materialize_int_to_i32_is_noop(db: &salsa::DatabaseImpl) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        // Create a dummy value
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let int_ty = tribute_rt::Int::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();

        let result = converter.materialize(db, location, value, int_ty, i32_ty);
        assert!(matches!(result, Some(MaterializeResult::NoOp)));
    }

    #[salsa_test]
    fn test_are_equivalent_primitives(db: &salsa::DatabaseImpl) {
        let int_ty = tribute_rt::Int::new(db).as_type();
        let nat_ty = tribute_rt::Nat::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let float_ty = tribute_rt::Float::new(db).as_type();
        let f64_ty = core::F64::new(db).as_type();

        // Int-like types are equivalent
        assert!(are_equivalent_primitives(db, int_ty, i32_ty));
        assert!(are_equivalent_primitives(db, nat_ty, i32_ty));
        assert!(are_equivalent_primitives(db, int_ty, nat_ty));

        // Float-like types are equivalent
        assert!(are_equivalent_primitives(db, float_ty, f64_ty));

        // Int and float are not equivalent
        assert!(!are_equivalent_primitives(db, int_ty, f64_ty));
        assert!(!are_equivalent_primitives(db, float_ty, i32_ty));
    }

    // === Boxing materialization tests ===
    // Each test uses a separate tracked function to satisfy Salsa requirements

    #[salsa::tracked]
    fn do_box_int_to_any(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Int::new(db).as_type();
        let to_ty = tribute_rt::Any::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_int_to_any_generates_box_int(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_box_int_to_any(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::BOX_INT());
    }

    #[salsa::tracked]
    fn do_box_bool_to_any(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Bool::new(db).as_type();
        let to_ty = tribute_rt::Any::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_bool_to_any_generates_box_bool(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_box_bool_to_any(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::BOX_BOOL());
    }

    #[salsa::tracked]
    fn do_box_float_to_any(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Float::new(db).as_type();
        let to_ty = tribute_rt::Any::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_float_to_any_generates_box_float(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_box_float_to_any(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::BOX_FLOAT());
    }

    #[salsa::tracked]
    fn do_box_i32_to_any(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = core::I32::new(db).as_type();
        let to_ty = tribute_rt::Any::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_i32_to_any_generates_box_int(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_box_i32_to_any(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::BOX_INT());
    }

    // === Unboxing materialization tests ===

    #[salsa::tracked]
    fn do_unbox_any_to_int(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Any::new(db).as_type();
        let to_ty = tribute_rt::Int::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_any_to_int_generates_unbox_int(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_unbox_any_to_int(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::UNBOX_INT());
    }

    #[salsa::tracked]
    fn do_unbox_any_to_nat(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Any::new(db).as_type();
        let to_ty = tribute_rt::Nat::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_any_to_nat_generates_unbox_nat(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_unbox_any_to_nat(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::UNBOX_NAT());
    }

    #[salsa::tracked]
    fn do_unbox_any_to_bool(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Any::new(db).as_type();
        let to_ty = tribute_rt::Bool::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_any_to_bool_generates_unbox_bool(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_unbox_any_to_bool(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::UNBOX_BOOL());
    }

    #[salsa::tracked]
    fn do_unbox_any_to_float(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Any::new(db).as_type();
        let to_ty = tribute_rt::Float::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_any_to_float_generates_unbox_float(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_unbox_any_to_float(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::UNBOX_FLOAT());
    }

    #[salsa::tracked]
    fn do_unbox_any_to_i32(db: &dyn salsa::Database) -> (Symbol, Symbol) {
        use trunk_ir::{BlockId, Location, PathId, Span, Value, ValueDef};

        let converter = generic_type_converter();
        let path = PathId::new(db, "test".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let block_id = BlockId::fresh();
        let value = Value::new(db, ValueDef::BlockArg(block_id), 0);

        let from_ty = tribute_rt::Any::new(db).as_type();
        let to_ty = core::I32::new(db).as_type();

        let result = converter.materialize(db, location, value, from_ty, to_ty);
        match result {
            Some(MaterializeResult::Ops(ops)) if !ops.is_empty() => {
                (ops[0].dialect(db), ops[0].name(db))
            }
            _ => (Symbol::new("error"), Symbol::new("error")),
        }
    }

    #[salsa_test]
    fn test_materialize_any_to_i32_generates_unbox_int(db: &salsa::DatabaseImpl) {
        let (dialect, name) = do_unbox_any_to_i32(db);
        assert_eq!(dialect, tribute_rt::DIALECT_NAME());
        assert_eq!(name, tribute_rt::UNBOX_INT());
    }
}
