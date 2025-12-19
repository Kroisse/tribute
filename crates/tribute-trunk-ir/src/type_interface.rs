//! Type interface system for polymorphic type behavior.
//!
//! This module provides an MLIR-style interface system where dialects can register
//! implementations for type-level interfaces (like `Printable`) at their definition site.
//!
//! # Example
//!
//! ```
//! # use tribute_core::TributeDatabaseImpl;
//! # use std::collections::BTreeMap;
//! # use tribute_trunk_ir::{IdVec, Symbol, Type};
//! # use tribute_trunk_ir::type_interface::Printable;
//! #
//! // In dialect/core.rs - register how to print "Lorem" type
//! inventory::submit! {
//!     Printable::implement("foobar", "lorem", |_, _, f| write!(f, "Lorem"))
//! }
//!
//! // For prefix matching (e.g., int1, int32, int64)
//! inventory::submit! {
//!     Printable::implement_prefix("foobar", "int", |db, ty, f| {
//!         let width = &ty.name(db).text(db)[3..];
//!         write!(f, "Int{width}")
//!     })
//! }
//!
//! # let db = TributeDatabaseImpl::default();
//! # let some_type = Type::new(
//! #     &db,
//! #     Symbol::new(&db, "foobar"),
//! #     Symbol::new(&db, "lorem"),
//! #     IdVec::new(),
//! #     BTreeMap::new(),
//! # );
//! // Usage
//! # let printed =
//! tribute_trunk_ir::type_interface::print_type(&db, some_type)
//! # ;
//! # assert_eq!(printed, "Lorem");
//! ```

use std::collections::HashMap;
use std::fmt::{self, Formatter, Write};
use std::sync::LazyLock;

use crate::Type;

/// Type name matching strategy for interface registration.
#[derive(Clone, Copy, Debug)]
pub enum TypeMatcher {
    /// Exact match: `name == expected`
    Exact(&'static str),
    /// Prefix match: `name.starts_with(prefix)`
    Prefix(&'static str),
}

/// Function type for printing a type.
pub type PrintFn = fn(&dyn salsa::Database, Type<'_>, &mut Formatter<'_>) -> fmt::Result;

/// Registration entry for the Printable interface.
///
/// Use `inventory::submit!` to register implementations at the type definition site.
pub struct PrintableRegistration {
    /// Dialect name (e.g., "core", "type", "src")
    pub dialect: &'static str,
    /// Type name matcher
    pub matcher: TypeMatcher,
    /// Print function implementation
    pub print: PrintFn,
}

inventory::collect!(PrintableRegistration);

/// Internal registry built from inventory at first access.
struct PrintableRegistry {
    /// Exact match lookup: (dialect, name) -> PrintFn
    exact: HashMap<(&'static str, &'static str), PrintFn>,
    /// Prefix match entries: (dialect, prefix, PrintFn)
    prefix: Vec<(&'static str, &'static str, PrintFn)>,
}

impl PrintableRegistry {
    fn new() -> Self {
        Self {
            exact: HashMap::new(),
            prefix: Vec::new(),
        }
    }

    fn lookup(&self, dialect: &str, name: &str) -> Option<PrintFn> {
        // Try exact match first
        if let Some(&print_fn) = self.exact.get(&(dialect, name)) {
            return Some(print_fn);
        }

        // Fall back to prefix matching
        for &(d, prefix, print_fn) in &self.prefix {
            if d == dialect && name.starts_with(prefix) {
                return Some(print_fn);
            }
        }

        None
    }
}

/// Global registry, lazily built from inventory on first access.
static REGISTRY: LazyLock<PrintableRegistry> = LazyLock::new(|| {
    let mut registry = PrintableRegistry::new();

    for reg in inventory::iter::<PrintableRegistration> {
        match reg.matcher {
            TypeMatcher::Exact(name) => {
                registry.exact.insert((reg.dialect, name), reg.print);
            }
            TypeMatcher::Prefix(prefix) => {
                registry.prefix.push((reg.dialect, prefix, reg.print));
            }
        }
    }

    registry
});

/// Interface for pretty-printing types.
///
/// This is the main entry point for type printing. Dialects register their
/// implementations using `inventory::submit!` with `Printable::implement()`.
pub struct Printable;

impl Printable {
    /// Register a Printable implementation for an exact type name match.
    ///
    /// Use with `inventory::submit!`:
    /// ```ignore
    /// inventory::submit! {
    ///     Printable::implement("core", "nil", |_, _, f| f.write_str("()"))
    /// }
    /// ```
    pub const fn implement(
        dialect: &'static str,
        name: &'static str,
        print: PrintFn,
    ) -> PrintableRegistration {
        PrintableRegistration {
            dialect,
            matcher: TypeMatcher::Exact(name),
            print,
        }
    }

    /// Register a Printable implementation for a prefix match.
    ///
    /// Use with `inventory::submit!`:
    /// ```ignore
    /// inventory::submit! {
    ///     Printable::implement_prefix("core", "i", |db, ty, f| { ... })
    /// }
    /// ```
    pub const fn implement_prefix(
        dialect: &'static str,
        prefix: &'static str,
        print: PrintFn,
    ) -> PrintableRegistration {
        PrintableRegistration {
            dialect,
            matcher: TypeMatcher::Prefix(prefix),
            print,
        }
    }

    /// Print a type to a formatter.
    ///
    /// Looks up the registered print function for the type's dialect and name.
    /// Falls back to `dialect.name(params...)` format for unregistered types.
    pub fn print_type(
        db: &dyn salsa::Database,
        ty: Type<'_>,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        let dialect = ty.dialect(db).text(db);
        let name = ty.name(db).text(db);

        if let Some(print_fn) = REGISTRY.lookup(dialect, name) {
            print_fn(db, ty, f)
        } else {
            // Fallback: dialect.name or dialect.name(params...)
            Self::print_fallback(db, dialect, name, ty, f)
        }
    }

    /// Fallback printing for unregistered types.
    fn print_fallback(
        db: &dyn salsa::Database,
        dialect: &str,
        name: &str,
        ty: Type<'_>,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        let params = ty.params(db);

        write!(f, "{dialect}.{name}")?;

        if !params.is_empty() {
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                Self::print_type(db, p, f)?;
            }
            f.write_char(')')?;
        }

        Ok(())
    }
}

/// Wrapper for displaying a type with database context.
pub struct TypeDisplay<'a, 'db> {
    db: &'a dyn salsa::Database,
    ty: Type<'db>,
}

impl<'a, 'db> TypeDisplay<'a, 'db>
where
    'a: 'db,
{
    /// Create a new type display wrapper.
    pub fn new(db: &'a dyn salsa::Database, ty: Type<'db>) -> Self {
        Self { db, ty }
    }
}

impl fmt::Display for TypeDisplay<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Printable::print_type(self.db, self.ty, f)
    }
}

/// Pretty-print a type to a user-friendly string.
pub fn print_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    TypeDisplay::new(db, ty).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_is_populated() {
        // Just verify the registry can be accessed without panicking
        let _ = &*REGISTRY;
    }
}
