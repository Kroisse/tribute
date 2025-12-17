//! Type dialect operations.
//!
//! This dialect defines type and ability declarations at the IR level.
//! These operations represent struct types, enum types, and ability definitions.
//!
//! Additionally, this module provides type constructors for inference-related types:
//! - `type.var` - a type variable to be resolved during type inference
//! - `type.error` - an error type indicating type resolution failed

use crate::{Attribute, IdVec, Type, TypeKind, dialect};

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
/// The attribute can carry metadata such as a unique variable ID or constraints.
pub fn var<'db>(db: &'db dyn salsa::Database, attr: Attribute<'db>) -> Type<'db> {
    Type::dialect(db, "type", "var", IdVec::new(), attr)
}

/// Create an error type (`type.error`) indicating type resolution failed.
///
/// The attribute can carry error information or source location.
pub fn error<'db>(db: &'db dyn salsa::Database, attr: Attribute<'db>) -> Type<'db> {
    Type::dialect(db, "type", "error", IdVec::new(), attr)
}

/// Check if a type is a type variable (`type.var`).
pub fn is_var(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    matches!(
        ty.kind(db),
        TypeKind::Dialect { dialect, name, .. }
            if dialect.text(db) == "type" && name.text(db) == "var"
    )
}

/// Check if a type is an error type (`type.error`).
pub fn is_error(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    matches!(
        ty.kind(db),
        TypeKind::Dialect { dialect, name, .. }
            if dialect.text(db) == "type" && name.text(db) == "error"
    )
}
