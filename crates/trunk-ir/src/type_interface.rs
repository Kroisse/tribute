//! Type interface system for polymorphic type behavior.
//!
//! This module provides an MLIR-style interface system where dialects can register
//! implementations for type-level interfaces (like `Printable`) at their definition site.
//!
//! # Example
//!
//! ```
//! # use salsa::DatabaseImpl;
//! # use std::collections::BTreeMap;
//! # use std::fmt::Write;
//! # use trunk_ir::{IdVec, Symbol, Type};
//! # use trunk_ir::type_interface::Printable;
//! #
//! // In dialect/core.rs - register how to print "Lorem" type
//! inventory::submit! {
//!     Printable::implement("foobar", "lorem", |_, _, f| write!(f, "Lorem"))
//! }
//!
//! // For prefix matching (e.g., int1, int32, int64)
//! inventory::submit! {
//!     Printable::implement_prefix("foobar", "int", |db, ty, f| {
//!         let name = ty.name(db).to_string();
//!         let width = &name[3..];
//!         write!(f, "Int{width}")
//!     })
//! }
//!
//! # let db = DatabaseImpl::default();
//! # let some_type = Type::new(
//! #     &db,
//! #     Symbol::new("foobar"),
//! #     Symbol::new("lorem"),
//! #     IdVec::new(),
//! #     BTreeMap::new(),
//! # );
//! // Usage
//! # let printed =
//! trunk_ir::type_interface::print_type(&db, some_type)
//! # ;
//! # assert_eq!(printed, "Lorem");
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Formatter, Write};
use std::sync::LazyLock;

use salsa::plumbing::AsId;

use crate::{Type, ir::Symbol};

/// Context for type printing with cycle detection.
///
/// Tracks types currently being printed to detect and handle recursive types.
pub struct PrintContext<'a, 'f> {
    /// The formatter to write output to.
    pub fmt: &'a mut Formatter<'f>,
    /// Set of type IDs currently being printed (for cycle detection).
    pub visiting: &'a mut HashSet<salsa::Id>,
}

impl<'a, 'f> PrintContext<'a, 'f> {
    /// Create a new print context.
    pub fn new(fmt: &'a mut Formatter<'f>, visiting: &'a mut HashSet<salsa::Id>) -> Self {
        Self { fmt, visiting }
    }

    /// Check if a type is currently being visited (cycle detection).
    pub fn is_visiting(&self, ty: Type<'_>) -> bool {
        self.visiting.contains(&ty.as_id())
    }

    /// Mark a type as being visited. Returns false if already visiting (cycle detected).
    pub fn enter(&mut self, ty: Type<'_>) -> bool {
        self.visiting.insert(ty.as_id())
    }

    /// Mark a type as no longer being visited.
    pub fn exit(&mut self, ty: Type<'_>) {
        self.visiting.remove(&ty.as_id());
    }
}

impl fmt::Write for PrintContext<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.fmt.write_str(s)
    }

    fn write_char(&mut self, c: char) -> fmt::Result {
        self.fmt.write_char(c)
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> fmt::Result {
        self.fmt.write_fmt(args)
    }
}

/// Type name matching strategy for interface registration.
#[derive(Clone, Copy, Debug)]
pub enum TypeMatcher {
    /// Exact match: `name == expected`
    Exact(&'static str),
    /// Prefix match: `name.starts_with(prefix)`
    Prefix(&'static str),
}

/// Function type for printing a type.
///
/// The `PrintContext` provides the formatter and cycle detection state.
pub type PrintFn = fn(&dyn salsa::Database, Type<'_>, &mut PrintContext<'_, '_>) -> fmt::Result;

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
    exact: HashMap<(Symbol, Symbol), PrintFn>,
    /// Prefix match entries: (dialect, prefix_str, PrintFn)
    /// prefix is stored as String to avoid repeated to_string() in lookup
    prefix: Vec<(Symbol, String, PrintFn)>,
}

impl PrintableRegistry {
    fn new() -> Self {
        Self {
            exact: HashMap::new(),
            prefix: Vec::new(),
        }
    }

    fn lookup(&self, dialect: Symbol, name: Symbol) -> Option<PrintFn> {
        // Try exact match first
        if let Some(&print_fn) = self.exact.get(&(dialect, name)) {
            return Some(print_fn);
        }

        // Fall back to prefix matching
        for (d, prefix_str, print_fn) in &self.prefix {
            if *d == dialect && name.with_str(|n| n.starts_with(prefix_str)) {
                return Some(*print_fn);
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
                registry.exact.insert(
                    (
                        Symbol::from_dynamic(reg.dialect),
                        Symbol::from_dynamic(name),
                    ),
                    reg.print,
                );
            }
            TypeMatcher::Prefix(prefix) => {
                // Store prefix as String to avoid repeated to_string() in lookup
                registry.prefix.push((
                    Symbol::from_dynamic(reg.dialect),
                    prefix.to_string(),
                    reg.print,
                ));
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
    /// ```
    /// # use std::fmt::Write;
    /// # use trunk_ir::type_interface::Printable;
    /// inventory::submit! {
    ///     Printable::implement("demo", "unit", |_, _, f| f.write_str("()"))
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
    /// ```
    /// # use std::fmt::Write;
    /// # use trunk_ir::type_interface::Printable;
    /// inventory::submit! {
    ///     Printable::implement_prefix("demo", "t", |db, ty, f| {
    ///         let name = ty.name(db).to_string();
    ///         let suffix = &name[1..];
    ///         write!(f, "T{suffix}")
    ///     })
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

    /// Print a type to a formatter with cycle detection.
    ///
    /// Looks up the registered print function for the type's dialect and name.
    /// Falls back to `dialect.name(params...)` format for unregistered types.
    /// Detects recursive types and prints `...` to avoid infinite loops.
    pub fn print_type(
        db: &dyn salsa::Database,
        ty: Type<'_>,
        f: &mut PrintContext<'_, '_>,
    ) -> fmt::Result {
        // Cycle detection: if we're already printing this type, emit placeholder
        if !f.enter(ty) {
            return f.write_str("...");
        }

        let dialect = ty.dialect(db);
        let name = ty.name(db);

        let result = if let Some(print_fn) = REGISTRY.lookup(dialect, name) {
            print_fn(db, ty, f)
        } else {
            // Fallback: dialect.name or dialect.name(params...)
            Self::print_fallback(db, dialect, name, ty, f)
        };

        f.exit(ty);
        result
    }

    /// Fallback printing for unregistered types.
    fn print_fallback(
        db: &dyn salsa::Database,
        dialect: Symbol,
        name: Symbol,
        ty: Type<'_>,
        f: &mut PrintContext<'_, '_>,
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
        let mut visiting = HashSet::new();
        let mut ctx = PrintContext::new(f, &mut visiting);
        Printable::print_type(self.db, self.ty, &mut ctx)
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
