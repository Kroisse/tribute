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
//! - `tribute_rt.retain` - increment reference count: ptr → ptr
//! - `tribute_rt.release` - decrement reference count (free if zero): ptr → ()
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

use std::fmt::Write;

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

        /// `tribute_rt.box_int` operation: box a signed integer.
        /// Converts an unboxed int to a boxed intref (i31ref).
        fn box_int(value) -> result;

        /// `tribute_rt.unbox_int` operation: unbox a signed integer.
        /// Converts a boxed intref back to an unboxed int (signed).
        fn unbox_int(value) -> result;

        /// `tribute_rt.box_nat` operation: box an unsigned natural number.
        /// Converts an unboxed nat to a boxed intref (i31ref).
        fn box_nat(value) -> result;

        /// `tribute_rt.unbox_nat` operation: unbox an unsigned natural number.
        /// Converts a boxed intref back to an unboxed nat (unsigned).
        fn unbox_nat(value) -> result;

        /// `tribute_rt.box_float` operation: box a float.
        /// Converts an unboxed f64 to a boxed struct (BoxedF64).
        fn box_float(value) -> result;

        /// `tribute_rt.unbox_float` operation: unbox a float.
        /// Converts a boxed struct back to an unboxed f64.
        fn unbox_float(value) -> result;

        /// `tribute_rt.box_bool` operation: box a boolean.
        /// Converts an unboxed bool (i32) to a boxed intref (i31ref).
        fn box_bool(value) -> result;

        /// `tribute_rt.unbox_bool` operation: unbox a boolean.
        /// Converts a boxed intref back to an unboxed bool (i32).
        fn unbox_bool(value) -> result;

        // === RC Operations ===

        /// `tribute_rt.retain` operation: increment reference count.
        /// Takes a pointer and returns the same pointer (for chaining).
        /// Lowered to inline refcount increment in native backend.
        fn retain(ptr) -> result;

        /// `tribute_rt.release` operation: decrement reference count.
        /// If the count reaches zero, the object is freed.
        /// Lowered to inline refcount decrement + conditional free.
        /// `alloc_size` is the total allocation size (payload + 8-byte header),
        /// or 0 if unknown. Used by rc_lowering for dealloc calls.
        #[attr(alloc_size: u64)]
        fn release(ptr);
    }
}

// === RC Header Layout ===

/// RC header size in bytes: 4 bytes refcount + 4 bytes rtti_idx = 8 bytes.
///
/// All heap-allocated objects are prefixed with this header. The allocation
/// functions receive `payload_size + RC_HEADER_SIZE` and return a raw pointer.
/// Callers store the header at the raw pointer and use `raw_ptr + RC_HEADER_SIZE`
/// as the payload pointer.
pub const RC_HEADER_SIZE: u64 = 8;

// === Pure operation registrations ===
// Boxing operations are pure (no side effects)

trunk_ir::register_pure_op!(BoxInt<'_>);
trunk_ir::register_pure_op!(UnboxInt<'_>);
trunk_ir::register_pure_op!(BoxNat<'_>);
trunk_ir::register_pure_op!(UnboxNat<'_>);
trunk_ir::register_pure_op!(BoxFloat<'_>);
trunk_ir::register_pure_op!(UnboxFloat<'_>);
trunk_ir::register_pure_op!(BoxBool<'_>);
trunk_ir::register_pure_op!(UnboxBool<'_>);

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
