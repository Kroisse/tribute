//! Tribute runtime dialect for boxing/unboxing and runtime types.
//!
//! This dialect represents runtime type representations that are
//! backend-independent. It provides:
//!
//! ## Types
//!
//! - `tribute_rt.int` - unboxed integer (i32, 31-bit signed)
//! - `tribute_rt.nat` - unboxed natural number (i32, 31-bit unsigned)
//! - `tribute_rt.float` - unboxed float (f64)
//! - `tribute_rt.bool` - unboxed boolean (i32)
//! - `tribute_rt.intref` - boxed integer reference (wasm: i31ref, native: tagged ptr)
//! - `tribute_rt.any` - type-erased value (wasm: anyref, native: universal ref)
//!
//! ## Operations
//!
//! - `tribute_rt.box_int` - box an integer: int → intref
//! - `tribute_rt.unbox_int` - unbox an integer: intref → int
//!
//! ## Type Hierarchy
//!
//! ```text
//! int (unboxed i32)
//!     ↓ box_int
//! intref (boxed)
//!     ↓ (implicit subtyping)
//! any (type-erased)
//! ```
//!
//! ## Backend Mappings
//!
//! | Type    | Wasm        | Native (future)  |
//! |---------|-------------|------------------|
//! | int     | i32         | i32              |
//! | intref  | i31ref      | tagged pointer   |
//! | any     | anyref      | universal ref    |

use trunk_ir::dialect;

dialect! {
    mod tribute_rt {
        // === Types ===

        /// `tribute_rt.int` type: unboxed 31-bit signed integer.
        /// Maps to i32 at runtime, but semantically represents signed integers
        /// in the range [-2^30, 2^30).
        type int;

        /// `tribute_rt.nat` type: unboxed 31-bit unsigned natural number.
        /// Maps to i32 at runtime, but semantically represents unsigned integers
        /// in the range [0, 2^31).
        type nat;

        /// `tribute_rt.float` type: unboxed 64-bit float.
        /// Maps directly to f64.
        type float;

        /// `tribute_rt.bool` type: unboxed boolean.
        /// Maps to i32 (0 = false, 1 = true).
        type bool;

        /// `tribute_rt.intref` type: boxed integer reference.
        /// Wasm: i31ref, Native: tagged pointer.
        /// This is a subtype of `any`.
        type intref;

        /// `tribute_rt.any` type: type-erased reference.
        /// Wasm: anyref, Native: universal reference.
        /// All boxed types (intref, etc.) are subtypes of any.
        type any;

        // === Boxing Operations ===

        /// `tribute_rt.box_int` operation: box an integer.
        /// Converts an unboxed int to a boxed intref.
        fn box_int(value) -> result;

        /// `tribute_rt.unbox_int` operation: unbox an integer.
        /// Converts a boxed intref back to an unboxed int.
        fn unbox_int(value) -> result;
    }
}

// === Pure operation registrations ===
// Boxing operations are pure (no side effects)

trunk_ir::register_pure_op!(BoxInt<'_>);
trunk_ir::register_pure_op!(UnboxInt<'_>);

// === Type constructor helpers ===

/// Create a `tribute_rt.int` type.
pub fn int_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
    *Int::new(db)
}

/// Create a `tribute_rt.nat` type.
pub fn nat_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
    *Nat::new(db)
}

/// Create a `tribute_rt.float` type.
pub fn float_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
    *Float::new(db)
}

/// Create a `tribute_rt.bool` type.
pub fn bool_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
    *Bool::new(db)
}

/// Create a `tribute_rt.intref` type.
pub fn intref_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
    *Intref::new(db)
}

/// Create a `tribute_rt.any` type.
pub fn any_type(db: &dyn salsa::Database) -> trunk_ir::Type<'_> {
    *Any::new(db)
}

// === Type checking helpers ===

/// Check if a type is `tribute_rt.int`.
pub fn is_int(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), INT())
}

/// Check if a type is `tribute_rt.nat`.
pub fn is_nat(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), NAT())
}

/// Check if a type is `tribute_rt.float`.
pub fn is_float(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), FLOAT())
}

/// Check if a type is `tribute_rt.bool`.
pub fn is_bool(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), BOOL())
}

/// Check if a type is `tribute_rt.intref`.
pub fn is_intref(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), INTREF())
}

/// Check if a type is `tribute_rt.any`.
pub fn is_any(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), ANY())
}

/// Check if a type is a primitive (int, nat, float, bool).
pub fn is_primitive(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    is_int(db, ty) || is_nat(db, ty) || is_float(db, ty) || is_bool(db, ty)
}

/// Check if a type is a boxed reference (intref, or any).
pub fn is_boxed_ref(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    is_intref(db, ty) || is_any(db, ty)
}

// === Printable interface registrations ===

inventory::submit! {
    trunk_ir::type_interface::Printable::implement("tribute_rt", "int", |_db, _ty, f| {
        f.write_str("Int")
    })
}

inventory::submit! {
    trunk_ir::type_interface::Printable::implement("tribute_rt", "nat", |_db, _ty, f| {
        f.write_str("Nat")
    })
}

inventory::submit! {
    trunk_ir::type_interface::Printable::implement("tribute_rt", "float", |_db, _ty, f| {
        f.write_str("Float")
    })
}

inventory::submit! {
    trunk_ir::type_interface::Printable::implement("tribute_rt", "bool", |_db, _ty, f| {
        f.write_str("Bool")
    })
}

inventory::submit! {
    trunk_ir::type_interface::Printable::implement("tribute_rt", "intref", |_db, _ty, f| {
        f.write_str("IntRef")
    })
}

inventory::submit! {
    trunk_ir::type_interface::Printable::implement("tribute_rt", "any", |_db, _ty, f| {
        f.write_str("Any")
    })
}
