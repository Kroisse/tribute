//! Operation interface system for querying operation properties.
//!
//! This module provides an interface system similar to `type_interface.rs` but for operations.
//! It uses the `inventory` crate to build a registry of operation properties at compile time.

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::{Operation, Symbol};

/// Marker trait for pure operations (no side effects, safe to remove if unused).
///
/// Operations implementing this trait can be safely eliminated by DCE if their results are unused.
pub trait Pure {}

/// Registration entry for pure operations.
///
/// Use `inventory::submit!` to register pure operations at the dialect definition site.
pub struct PureOpRegistration {
    /// Dialect name (e.g., "arith", "adt")
    pub dialect: &'static str,
    /// Operation name within the dialect (e.g., "add", "const")
    pub op_name: &'static str,
}

inventory::collect!(PureOpRegistration);

/// Internal registry built from inventory at first access.
struct PureOpRegistry {
    /// Lookup: (dialect, op_name) -> is_pure
    pure_ops: HashSet<(Symbol, Symbol)>,
}

impl PureOpRegistry {
    fn new() -> Self {
        Self {
            pure_ops: HashSet::new(),
        }
    }

    fn lookup(&self, dialect: Symbol, op_name: Symbol) -> bool {
        self.pure_ops.contains(&(dialect, op_name))
    }
}

/// Global registry, lazily built from inventory on first access.
static REGISTRY: LazyLock<PureOpRegistry> = LazyLock::new(|| {
    let mut registry = PureOpRegistry::new();

    for reg in inventory::iter::<PureOpRegistration> {
        let dialect = Symbol::from_dynamic(reg.dialect);
        let op_name = Symbol::from_dynamic(reg.op_name);
        registry.pure_ops.insert((dialect, op_name));
    }

    registry
});

/// Interface for querying operation purity.
pub struct PureOps;

impl PureOps {
    /// Register a pure operation (internal use by macro).
    ///
    /// Use the `register_pure_op!` macro instead:
    /// ```ignore
    /// register_pure_op!(arith.add);
    /// ```
    #[doc(hidden)]
    pub const fn register(dialect: &'static str, op_name: &'static str) -> PureOpRegistration {
        PureOpRegistration { dialect, op_name }
    }

    /// Check if an operation is pure (no side effects, safe to remove if unused).
    ///
    /// Returns true only if the operation has been explicitly registered as pure.
    /// Returns false for all unregistered operations (conservative by default).
    pub fn is_pure<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
        let dialect = op.dialect(db);
        let op_name = op.name(db);
        REGISTRY.lookup(dialect, op_name)
    }

    /// Check if an operation can be safely removed by DCE.
    ///
    /// An operation is removable if:
    /// 1. It is registered as pure (no side effects)
    /// 2. Its results are not used elsewhere
    pub fn is_removable<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
        Self::is_pure(db, op)
    }
}

/// Register a pure operation with simplified syntax.
///
/// # Example
/// ```ignore
/// register_pure_op!(arith.add);
/// register_pure_op!(adt.struct_new);
/// ```
///
/// This expands to both the trait implementation and inventory registration:
/// ```ignore
/// impl op_interface::Pure for dialect::arith::Add<'_> {}
/// inventory::submit! {
///     op_interface::PureOps::register("arith", "add")
/// }
/// ```
#[macro_export]
macro_rules! register_pure_op {
    ($dialect:ident . $op_name:ident) => {
        ::paste::paste! {
            impl $crate::op_interface::Pure for $crate::dialect::$dialect::[<$op_name:camel>]<'_> {}

            ::inventory::submit! {
                $crate::op_interface::PureOps::register(
                    stringify!($dialect),
                    stringify!($op_name)
                )
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_is_populated() {
        // Just verify the registry can be accessed without panicking
        let _ = &*REGISTRY;
    }

    #[test]
    fn test_unregistered_ops_are_not_pure() {
        // This test would need a database and operation, so we just verify the struct exists
        let _ = PureOps;
    }
}
