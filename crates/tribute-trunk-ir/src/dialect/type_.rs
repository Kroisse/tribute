//! Type dialect operations.
//!
//! This dialect defines type and ability declarations at the IR level.
//! These operations represent struct types, enum types, and ability definitions.

use crate::dialect;

dialect! {
    r#type {
        /// `type.struct` operation: defines a struct type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the struct type
        /// - `fields`: Field definitions as [(name, type)] pairs
        op r#struct[sym_name, fields]() -> result;

        /// `type.enum` operation: defines an enum (sum) type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the enum type
        /// - `variants`: Variant definitions as [(name, fields)] pairs
        op r#enum[sym_name, variants]() -> result;

        /// `type.ability` operation: defines an ability (effect) type.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the ability
        /// - `operations`: Operation signatures as [(name, signature)] pairs
        op ability[sym_name, operations]() -> result;
    }
}
