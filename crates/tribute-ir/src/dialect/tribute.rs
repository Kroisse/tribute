//! Tribute language AST/HIR dialect.
//!
//! This dialect represents Tribute-specific high-level operations.
//!
//! ## Dialect Organization
//!
//! **Data construction**:
//! - `tribute.tuple` (tuple construction)
//!
//! **Effect system** (handler support, kept for reference):
//! - `tribute.ability_def`, `tribute.op_def` (ability declarations)
//! - `tribute.handle`, `tribute.arm` (handler expressions)
//!
//! **Types**:
//! - `tribute.type` (unresolved type reference)
//! - `tribute.tuple_type` (tuple type cons cell)

use std::fmt::Write;

use trunk_ir::type_interface::{PrintContext, Printable};
use trunk_ir::{Attribute, IdVec, Location, Symbol, dialect};

/// Block argument attribute symbols for pattern bindings.
///
/// These attributes are used on block arguments to associate binding names
/// and source locations with SSA values.
pub mod block_arg_attrs {
    trunk_ir::symbols! {
        /// Attribute key for the binding name associated with a block argument.
        /// Used in case arm body blocks to name pattern-bound values.
        BIND_NAME => "bind_name",

        /// Attribute key for the source location of a binding.
        /// Used for LSP features like Go to Definition.
        BIND_LOCATION => "bind_location",
    }
}

dialect! {
    mod tribute {
        // === Data construction ===

        /// `tribute.tuple` operation: tuple construction.
        /// Takes variadic operands (tuple elements) and produces a tuple value.
        fn tuple(#[rest] elements) -> result;

        /// `tribute.ability_def` operation: defines an ability (effect) type.
        ///
        /// The operations region contains `tribute.op_def` operations defining the signatures.
        ///
        /// Attributes:
        /// - `sym_name`: The name of the ability
        #[attr(sym_name: Symbol)]
        fn ability_def() -> result {
            #[region(operations)] {}
        };

        // === Effect definition operations (metadata) ===

        /// `tribute.op_def` operation: declares an operation signature within an ability.
        ///
        /// Used inside `tribute.ability_def` operations region to define what
        /// operations the ability provides.
        ///
        /// Attributes:
        /// - `sym_name`: The operation name
        /// - `type`: The operation's function type (func.Fn)
        #[attr(sym_name: Symbol, r#type: Type)]
        fn op_def();

        /// `tribute.handle` operation: runs body in a delimited context with handler arms.
        ///
        /// Fused handler syntax: `handle expr { handler_arms }`.
        ///
        /// Executes the body region until it either:
        /// - Completes with a value → matches `{ result }` handler pattern
        /// - Performs an ability operation → matches `{ Op(args) -> k }` handler pattern
        ///
        /// The arms region contains `tribute.arm` operations with handler patterns
        /// (`tribute_pat.handler_done` or `tribute_pat.handler_suspend`).
        fn handle() -> result {
            #[region(body)] {}
            #[region(arms)] {}
        };

        /// `tribute.arm` operation: a single pattern-matching arm (used in handler arms).
        /// The pattern region contains a tree of `tribute_pat.*` operations.
        /// The body region contains the arm's expression.
        fn arm() {
            #[region(pattern)] {}
            #[region(body)] {}
        };

        // === Types ===

        /// `tribute.type`: an unresolved type reference that needs name resolution.
        /// The `name` attribute holds the type name (e.g., "Int", "List").
        /// The `params` hold type arguments for generic types (e.g., `List(a)`).
        #[attr(name: Symbol)]
        type r#type(#[rest] params);

        // NOTE: Primitive types (int, nat, float, bool) are now in `tribute_rt` dialect.

        /// `tribute.tuple_type` type: cons cell (head, tail).
        /// Use `core.nil` as the tail terminator.
        /// Example: `(a, b, c)` → `TupleType(a, TupleType(b, TupleType(c, Nil)))`
        type tuple_type(head, tail);
    }
}

/// Check if a type is an unresolved type reference (`tribute.type`).
///
/// These are type names that haven't been resolved to concrete types yet.
pub fn is_unresolved_type(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.dialect(db) == DIALECT_NAME() && ty.name(db) == Symbol::new("type")
}

/// Check if a type is a placeholder that should be resolved before emit.
///
/// This includes `tribute.type` (unresolved type references).
pub fn is_placeholder_type(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    is_unresolved_type(db, ty)
}

// === Convenience function for creating unresolved types ===

/// Create an unresolved type reference (`tribute.type`).
///
/// This is a convenience wrapper around `Type::new` that takes a string name.
pub fn unresolved_type<'db>(
    db: &'db dyn salsa::Database,
    name: Symbol,
    params: IdVec<trunk_ir::Type<'db>>,
) -> trunk_ir::Type<'db> {
    // Use the macro-generated Type struct
    *Type::new(db, params, name)
}

// === Pattern Region Builders ===

/// Re-export pattern helpers from the `tribute_pat` dialect.
pub use super::tribute_pat::helpers as pattern;

impl<'db> Arm<'db> {
    /// Create a wildcard arm that matches anything.
    pub fn wildcard(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        body: trunk_ir::Region<'db>,
    ) -> Self {
        let pattern_region = pattern::wildcard_region(db, location);
        arm(db, location, pattern_region, body)
    }

    /// Create an arm that binds the scrutinee to a name.
    pub fn binding(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        name: Symbol,
        body: trunk_ir::Region<'db>,
    ) -> Self {
        let pattern_region = pattern::bind_region(db, location, name);
        arm(db, location, pattern_region, body)
    }

    /// Create an arm matching a unit variant (e.g., `None`).
    pub fn unit_variant(
        db: &'db dyn salsa::Database,
        location: Location<'db>,
        variant_name: Symbol,
        body: trunk_ir::Region<'db>,
    ) -> Self {
        let fields = pattern::empty_region(db, location);
        let pattern_region = pattern::variant_region(db, location, variant_name, fields);
        arm(db, location, pattern_region, body)
    }
}

// === Printable interface registrations ===

// tribute.type -> "Name" or "Name(params...)"
inventory::submit! {
    Printable::implement("tribute", "type", |db, ty, f| {
        let Some(Attribute::Symbol(name)) = ty.get_attr(db, Type::name_sym()) else {
            return f.write_str("?unresolved");
        };

        let params = ty.params(db);

        // Capitalize first letter
        let name_text = name.to_string();
        let mut chars = name_text.chars();
        if let Some(c) = chars.next() {
            for ch in c.to_uppercase() {
                f.write_char(ch)?;
            }
            f.write_str(chars.as_str())?;
        }

        if !params.is_empty() {
            f.write_char('(')?;
            for (i, &p) in params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                Printable::print_type(db, p, f)?;
            }
            f.write_char(')')?;
        }

        Ok(())
    })
}

// tribute.tuple_type -> "(a, b, c)"
inventory::submit! { Printable::implement("tribute", "tuple_type", print_tuple_type) }

fn print_tuple_type(
    db: &dyn salsa::Database,
    ty: trunk_ir::Type<'_>,
    f: &mut PrintContext<'_, '_>,
) -> std::fmt::Result {
    let params = ty.params(db);
    if params.is_empty() {
        return f.write_str("#()");
    }

    // Flatten cons cells into a list
    let mut elements = Vec::new();
    let mut current = ty;

    while current.is_dialect(db, DIALECT_NAME(), TUPLE_TYPE()) {
        let params = current.params(db);
        if params.len() >= 2 {
            elements.push(params[0]); // head
            current = params[1]; // tail
        } else {
            break;
        }
    }

    // Check if tail is nil (complete tuple)
    let has_tail = !current.is_dialect(
        db,
        trunk_ir::dialect::core::DIALECT_NAME(),
        trunk_ir::dialect::core::NIL(),
    );

    f.write_char('(')?;
    for (i, &elem) in elements.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        Printable::print_type(db, elem, f)?;
    }
    if has_tail {
        if !elements.is_empty() {
            f.write_str(", ")?;
        }
        Printable::print_type(db, current, f)?;
    }
    f.write_char(')')
}

// NOTE: tribute.int and tribute.nat Printable implementations moved to tribute_rt dialect

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::DialectType;
    use trunk_ir::dialect::core;
    use trunk_ir::type_interface::print_type;

    #[salsa_test]
    fn test_unresolved_type(db: &salsa::DatabaseImpl) {
        let ty = unresolved_type(db, Symbol::new("Int"), IdVec::new());
        assert!(is_unresolved_type(db, ty));
        assert_eq!(print_type(db, ty), "Int");

        // With type parameters
        let int_ty = core::I32::new(db).as_type();
        let list_ty = unresolved_type(db, Symbol::new("List"), IdVec::from(vec![int_ty]));
        assert!(is_unresolved_type(db, list_ty));
        assert_eq!(print_type(db, list_ty), "List(I32)");
    }

    #[salsa_test]
    fn test_is_placeholder_type(db: &salsa::DatabaseImpl) {
        let unresolved = unresolved_type(db, Symbol::new("Foo"), IdVec::new());
        let concrete = core::I32::new(db).as_type();

        assert!(is_placeholder_type(db, unresolved));
        assert!(!is_placeholder_type(db, concrete));
    }

    #[salsa_test]
    fn test_tuple_type_printable(db: &salsa::DatabaseImpl) {
        let int_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();

        // (Int, Int) -> TupleType(Int, TupleType(Int, Nil))
        let inner = TupleType::new(db, int_ty, nil_ty);
        let tuple = TupleType::new(db, int_ty, inner.as_type());

        let printed = print_type(db, tuple.as_type());
        assert_eq!(printed, "(I32, I32)");
    }
}
