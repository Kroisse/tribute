//! Type dialect operations.
//!
//! This dialect defines type and ability declarations at the IR level.
//! These operations represent struct types, enum types, and ability definitions.
//!
//! Additionally, this module provides type constructors for inference-related types:
//! - `type.var` - a type variable to be resolved during type inference
//! - `type.error` - an error type indicating type resolution failed
use std::collections::BTreeMap;

use crate::{Attribute, Attrs, IdVec, Type, dialect};

crate::symbols! {
    ATTR_ID => "id",
    ERROR => "error",
    VAR => "var",
}

dialect! {
    mod r#type {
        // === Type definition operations ===

        /// `type.struct` operation: defines a struct type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the struct type
        /// - `fields`: Field definitions as [(name, type)] pairs
        #[attr(sym_name, fields)]
        fn r#struct() -> result;

        /// `type.enum` operation: defines an enum (sum) type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the enum type
        /// - `variants`: Variant definitions as [(name, fields)] pairs
        #[attr(sym_name, variants)]
        fn r#enum() -> result;

        /// `type.ability` operation: defines an ability (effect) type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the ability
        /// - `operations`: Operation signatures as [(name, signature)] pairs
        #[attr(sym_name, operations)]
        fn ability() -> result;

        // === Tribute language primitive types ===

        /// `type.int` type: arbitrary precision integer (Fixnum/Bignum hybrid).
        /// At runtime, represented as i31ref (fixnum) or BigInt struct (bignum).
        type int;

        /// `type.nat` type: arbitrary precision natural number (non-negative).
        /// Semantically a subset of Int, but may have optimized representation.
        type nat;
    }
}

// === Type constructors for inference-related types ===

/// Create a type variable (`type.var`) to be resolved during type inference.
///
/// The `attrs` can carry metadata such as a unique variable ID or constraints.
pub fn var<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> Type<'db> {
    Type::new(db, DIALECT_NAME(), VAR(), IdVec::new(), attrs)
}

/// Create a type variable with a numeric ID.
pub fn var_with_id<'db>(db: &'db dyn salsa::Database, id: u64) -> Type<'db> {
    var(db, BTreeMap::from([(ATTR_ID(), Attribute::IntBits(id))]))
}

/// Create an error type (`type.error`) indicating type resolution failed.
///
/// The `attrs` can carry error information or source location.
pub fn error<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> Type<'db> {
    Type::new(db, DIALECT_NAME(), ERROR(), IdVec::new(), attrs)
}

/// Check if a type is a type variable (`type.var`).
pub fn is_var(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), VAR())
}

/// Check if a type is an error type (`type.error`).
pub fn is_error(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), ERROR())
}

// === Printable interface registrations ===

use std::fmt::{Formatter, Write};

use crate::type_interface::Printable;

// type.var -> "a", "b", ..., "t0", "t1", ...
inventory::submit! {
    Printable::implement("type", "var", |db, ty, f| {
        if let Some(Attribute::IntBits(id)) = ty.get_attr(db, ATTR_ID()) {
            fmt_var_id(f, *id)
        } else {
            f.write_char('?')
        }
    })
}

// type.error -> "<error>"
inventory::submit! { Printable::implement("type", "error", |_, _, f| f.write_str("<error>")) }

// type.int -> "Int"
inventory::submit! { Printable::implement("type", "int", |_, _, f| f.write_str("Int")) }

// type.nat -> "Nat"
inventory::submit! { Printable::implement("type", "nat", |_, _, f| f.write_str("Nat")) }

/// Convert a variable ID to a readable name (a, b, c, ..., t0, t1, ...).
fn fmt_var_id(f: &mut Formatter<'_>, id: u64) -> std::fmt::Result {
    if id < 26 {
        f.write_char((b'a' + id as u8) as char)
    } else {
        write!(f, "t{}", id - 26)
    }
}
