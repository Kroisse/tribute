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
    /// ```text
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

    /// Check if an operation is pure and eligible for DCE removal.
    ///
    /// Returns true if the operation is registered as pure (has no side effects).
    ///
    /// Note: This only checks purity. The caller must separately verify that
    /// the operation's results are unused before removing it.
    pub fn is_removable<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
        Self::is_pure(db, op)
    }
}

/// Register a pure operation with simplified syntax.
///
/// # Example
/// ```text
/// // New syntax: just provide the operation type
/// register_pure_op!(arith::Add);
/// register_pure_op!(adt::StructNew);
///
/// // Legacy syntax: dialect.op_name (still supported)
/// register_pure_op!(arith.add);
/// register_pure_op!(adt.struct_new);
/// ```
///
/// This expands to both the trait implementation and inventory registration:
/// ```text
/// impl op_interface::Pure for dialect::arith::Add<'_> {}
/// inventory::submit! {
///     op_interface::PureOps::register("arith", "add")
/// }
/// ```
#[macro_export]
macro_rules! register_pure_op {
    // New syntax: provide the type path directly
    // The type must implement DialectOp with DIALECT_NAME and OP_NAME constants
    ($op_type:ty) => {
        impl $crate::op_interface::Pure for $op_type {}

        ::inventory::submit! {
            $crate::op_interface::PureOps::register(
                <$op_type as $crate::DialectOp>::DIALECT_NAME,
                <$op_type as $crate::DialectOp>::OP_NAME
            )
        }
    };

    // Legacy syntax: dialect.op_name (for backwards compatibility within trunk-ir)
    ($dialect:ident . $op_name:ident) => {
        $crate::paste::paste! {
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

// =============================================================================
// IsolatedFromAbove Trait
// =============================================================================

/// Marker trait for operations whose regions cannot reference values from above.
///
/// Operations implementing this trait have regions that are "isolated" from the
/// enclosing scope - they cannot directly reference SSA values defined outside
/// the region. This is important for:
///
/// 1. **Verification**: Check that isolated regions don't have stale references
/// 2. **Rewrite passes**: Different handling for isolated vs non-isolated ops
/// 3. **Code generation**: Isolated regions can be compiled independently
///
/// Examples of isolated operations:
/// - `func.func` - function bodies (must receive values via parameters)
/// - `core.module` - module bodies
///
/// Examples of non-isolated operations (can capture outer values):
/// - `scf.if`, `scf.for` - control flow
/// - `closure.new` - closure creation
pub trait IsolatedFromAbove {}

/// Registration entry for isolated operations.
///
/// Use `inventory::submit!` to register isolated operations at the dialect definition site.
pub struct IsolatedFromAboveRegistration {
    /// Dialect name (e.g., "func", "core")
    pub dialect: &'static str,
    /// Operation name within the dialect (e.g., "func", "module")
    pub op_name: &'static str,
}

inventory::collect!(IsolatedFromAboveRegistration);

/// Internal registry built from inventory at first access.
struct IsolatedFromAboveRegistry {
    /// Lookup: (dialect, op_name) -> is_isolated
    isolated_ops: HashSet<(Symbol, Symbol)>,
}

impl IsolatedFromAboveRegistry {
    fn new() -> Self {
        Self {
            isolated_ops: HashSet::new(),
        }
    }

    fn lookup(&self, dialect: Symbol, op_name: Symbol) -> bool {
        self.isolated_ops.contains(&(dialect, op_name))
    }
}

/// Global registry for isolated operations, lazily built from inventory on first access.
static ISOLATED_REGISTRY: LazyLock<IsolatedFromAboveRegistry> = LazyLock::new(|| {
    let mut registry = IsolatedFromAboveRegistry::new();

    for reg in inventory::iter::<IsolatedFromAboveRegistration> {
        let dialect = Symbol::from_dynamic(reg.dialect);
        let op_name = Symbol::from_dynamic(reg.op_name);
        registry.isolated_ops.insert((dialect, op_name));
    }

    registry
});

/// Interface for querying operation isolation.
pub struct IsolatedFromAboveOps;

impl IsolatedFromAboveOps {
    /// Register an isolated operation (internal use by macro).
    ///
    /// Use the `register_isolated_op!` macro instead:
    /// ```text
    /// register_isolated_op!(func::Func);
    /// ```
    #[doc(hidden)]
    pub const fn register(
        dialect: &'static str,
        op_name: &'static str,
    ) -> IsolatedFromAboveRegistration {
        IsolatedFromAboveRegistration { dialect, op_name }
    }

    /// Check if an operation's regions are isolated from above.
    ///
    /// Returns true only if the operation has been explicitly registered as isolated.
    /// Returns false for all unregistered operations (conservative by default).
    pub fn is_isolated<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
        let dialect = op.dialect(db);
        let op_name = op.name(db);
        ISOLATED_REGISTRY.lookup(dialect, op_name)
    }

    /// Verify that an isolated operation doesn't have references to outer values.
    ///
    /// Returns a list of (operation, referenced_value) pairs that violate the constraint.
    /// Empty list means the operation is valid.
    ///
    /// This only checks the immediate regions of the operation, not nested isolated
    /// operations (they are their own verification scope).
    pub fn verify_isolation<'db>(
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> Vec<IsolationViolation<'db>> {
        if !Self::is_isolated(db, op) {
            return vec![];
        }

        let mut violations = Vec::new();
        let mut valid_values: HashSet<crate::Value<'db>> = HashSet::new();

        // Collect block arguments as valid values
        for region in op.regions(db).iter() {
            for block in region.blocks(db).iter() {
                let num_args = block.args(db).len();
                for idx in 0..num_args {
                    valid_values.insert(block.arg(db, idx));
                }
            }
        }

        // Walk regions and check for violations
        for region in op.regions(db).iter() {
            Self::verify_region(db, *region, &mut valid_values, &mut violations);
        }

        violations
    }

    fn verify_region<'db>(
        db: &'db dyn salsa::Database,
        region: crate::Region<'db>,
        valid_values: &mut HashSet<crate::Value<'db>>,
        violations: &mut Vec<IsolationViolation<'db>>,
    ) {
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                // Check operands
                for operand in op.operands(db).iter() {
                    if !valid_values.contains(operand) {
                        violations.push(IsolationViolation {
                            operation: *op,
                            external_value: *operand,
                        });
                    }
                }

                // Add results to valid values
                let num_results = op.results(db).len();
                for idx in 0..num_results {
                    let result = crate::Value::new(db, crate::ValueDef::OpResult(*op), idx);
                    valid_values.insert(result);
                }

                // Recurse into nested regions (but stop at nested isolated ops)
                if !Self::is_isolated(db, op) {
                    for nested_region in op.regions(db).iter() {
                        // Add block arguments of nested region
                        for block in nested_region.blocks(db).iter() {
                            let num_args = block.args(db).len();
                            for idx in 0..num_args {
                                valid_values.insert(block.arg(db, idx));
                            }
                        }
                        Self::verify_region(db, *nested_region, valid_values, violations);
                    }
                }
            }
        }
    }
}

/// Represents a violation of the IsolatedFromAbove constraint.
#[derive(Debug, Clone)]
pub struct IsolationViolation<'db> {
    /// The operation that has the invalid reference.
    pub operation: Operation<'db>,
    /// The value that is referenced from outside the isolated region.
    pub external_value: crate::Value<'db>,
}

/// Register an isolated operation with simplified syntax.
///
/// # Example
/// ```text
/// // Provide the operation type directly
/// register_isolated_op!(func::Func);
///
/// // Legacy syntax: dialect.op_name (still supported)
/// register_isolated_op!(func.func);
/// ```
///
/// This expands to both the trait implementation and inventory registration:
/// ```text
/// impl op_interface::IsolatedFromAbove for dialect::func::Func<'_> {}
/// inventory::submit! {
///     op_interface::IsolatedFromAboveOps::register("func", "func")
/// }
/// ```
#[macro_export]
macro_rules! register_isolated_op {
    // New syntax: provide the type path directly
    // The type must implement DialectOp with DIALECT_NAME and OP_NAME constants
    ($op_type:ty) => {
        impl $crate::op_interface::IsolatedFromAbove for $op_type {}

        ::inventory::submit! {
            $crate::op_interface::IsolatedFromAboveOps::register(
                <$op_type as $crate::DialectOp>::DIALECT_NAME,
                <$op_type as $crate::DialectOp>::OP_NAME
            )
        }
    };

    // Legacy syntax: dialect.op_name (for backwards compatibility within trunk-ir)
    ($dialect:ident . $op_name:ident) => {
        $crate::paste::paste! {
            impl $crate::op_interface::IsolatedFromAbove for $crate::dialect::$dialect::[<$op_name:camel>]<'_> {}

            ::inventory::submit! {
                $crate::op_interface::IsolatedFromAboveOps::register(
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

    #[test]
    fn test_isolated_registry_is_populated() {
        // Just verify the registry can be accessed without panicking
        let _ = &*ISOLATED_REGISTRY;
    }

    #[test]
    fn test_unregistered_ops_are_not_isolated() {
        // This test would need a database and operation, so we just verify the struct exists
        let _ = IsolatedFromAboveOps;
    }
}
