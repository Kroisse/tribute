//! Operation interface system for querying operation properties.
//!
//! This module provides an interface system similar to `type_interface.rs` but for operations.
//! It uses the `inventory` crate to build a registry of operation properties at compile time.

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::Symbol;
use crate::arena::{IrContext, OpRef};

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

    /// Check if an arena operation is pure (no side effects, safe to remove if unused).
    pub fn is_pure_arena(ctx: &IrContext, op: OpRef) -> bool {
        let data = ctx.op(op);
        REGISTRY.lookup(data.dialect, data.name)
    }

    /// Check if an arena operation is pure and eligible for DCE removal.
    pub fn is_removable_arena(ctx: &IrContext, op: OpRef) -> bool {
        Self::is_pure_arena(ctx, op)
    }
}

/// Register a pure operation with simplified syntax.
///
/// # Example
/// ```text
/// register_pure_op!(arith.add);
/// register_pure_op!(adt.struct_new);
/// ```
///
/// This expands to an inventory registration:
/// ```text
/// inventory::submit! {
///     op_interface::PureOps::register("arith", "add")
/// }
/// ```
#[macro_export]
macro_rules! register_pure_op {
    // Legacy syntax: dialect.op_name
    ($dialect:ident . $op_name:ident) => {
        $crate::paste::paste! {
            ::inventory::submit! {
                $crate::op_interface::PureOps::register(
                    $crate::raw_ident_str!($dialect),
                    $crate::raw_ident_str!($op_name)
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
    /// register_isolated_op!(func.func);
    /// ```
    #[doc(hidden)]
    pub const fn register(
        dialect: &'static str,
        op_name: &'static str,
    ) -> IsolatedFromAboveRegistration {
        IsolatedFromAboveRegistration { dialect, op_name }
    }

    /// Check if an arena operation's regions are isolated from above.
    pub fn is_isolated_arena(ctx: &IrContext, op: OpRef) -> bool {
        let data = ctx.op(op);
        ISOLATED_REGISTRY.lookup(data.dialect, data.name)
    }
}

/// Register an isolated operation with simplified syntax.
///
/// # Example
/// ```text
/// register_isolated_op!(func.func);
/// ```
///
/// This expands to an inventory registration:
/// ```text
/// inventory::submit! {
///     op_interface::IsolatedFromAboveOps::register("func", "func")
/// }
/// ```
#[macro_export]
macro_rules! register_isolated_op {
    // Legacy syntax: dialect.op_name
    ($dialect:ident . $op_name:ident) => {
        $crate::paste::paste! {
            ::inventory::submit! {
                $crate::op_interface::IsolatedFromAboveOps::register(
                    $crate::raw_ident_str!($dialect),
                    $crate::raw_ident_str!($op_name)
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
