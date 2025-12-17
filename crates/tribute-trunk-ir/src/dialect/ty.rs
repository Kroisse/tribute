//! Type dialect operations.
//!
//! This dialect defines type and ability declarations at the IR level.
//! These operations represent struct types, enum types, and ability definitions.
//!
//! Additionally, this module provides type constructors for inference-related types:
//! - `type.var` - a type variable to be resolved during type inference
//! - `type.error` - an error type indicating type resolution failed
use std::collections::BTreeMap;

use crate::{Attribute, Attrs, IdVec, Symbol, Type, dialect};

dialect! {
    mod r#type {
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
    }
}

// === Type constructors for inference-related types ===

/// Create a type variable (`type.var`) to be resolved during type inference.
///
/// The `attrs` can carry metadata such as a unique variable ID or constraints.
pub fn var<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> Type<'db> {
    Type::new(
        db,
        Symbol::new(db, "type"),
        Symbol::new(db, "var"),
        IdVec::new(),
        attrs,
    )
}

/// Create a type variable with a numeric ID.
pub fn var_with_id<'db>(db: &'db dyn salsa::Database, id: u64) -> Type<'db> {
    var(
        db,
        BTreeMap::from([(Symbol::new(db, "id"), Attribute::IntBits(id))]),
    )
}

/// Create an error type (`type.error`) indicating type resolution failed.
///
/// The `attrs` can carry error information or source location.
pub fn error<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> Type<'db> {
    Type::new(
        db,
        Symbol::new(db, "type"),
        Symbol::new(db, "error"),
        IdVec::new(),
        attrs,
    )
}

/// Check if a type is a type variable (`type.var`).
pub fn is_var(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.is_dialect(db, "type", "var")
}

/// Check if a type is an error type (`type.error`).
pub fn is_error(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    ty.is_dialect(db, "type", "error")
}
