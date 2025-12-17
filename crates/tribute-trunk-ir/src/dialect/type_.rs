//! Type dialect operations.
//!
//! This dialect defines type and ability declarations at the IR level.
//! These operations represent struct types, enum types, and ability definitions.

use crate::dialect;

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
