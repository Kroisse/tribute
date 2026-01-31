//! Tribute language AST/HIR dialect.
//!
//! This dialect represents Tribute-specific high-level operations.
//!
//! ## Dialect Organization
//!
//! **Data construction**:
//! - `tribute.tuple` (tuple construction)
//!
//! **Effect system** (ability declarations):
//! - `tribute.ability_def`, `tribute.op_def` (ability declarations)
//!
//! **Types**:
//! - `tribute.type` (unresolved type reference)
//! - `tribute.type_var`, `tribute.error_type` (type inference)
//! - `tribute.tuple_type` (tuple type cons cell)

use std::collections::BTreeMap;
use std::fmt::Write;

use trunk_ir::type_interface::{PrintContext, Printable};
use trunk_ir::{Attribute, Attrs, IdVec, Symbol, dialect};

trunk_ir::symbols! {
    VAR_ID_ATTR => "id",
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

        // === Types ===

        /// `tribute.type`: an unresolved type reference that needs name resolution.
        /// The `name` attribute holds the type name (e.g., "Int", "List").
        /// The `params` hold type arguments for generic types (e.g., `List(a)`).
        #[attr(name: Symbol)]
        type r#type(#[rest] params);

        // NOTE: Primitive types (int, nat, float, bool) are now in `tribute_rt` dialect.

        /// `tribute.type_var` type: a type variable to be resolved during type inference.
        /// The `id` attribute holds a unique variable ID.
        #[attr(id: any)]
        type type_var;

        /// `tribute.error_type` type: an error type indicating type resolution failed.
        type error_type;

        /// `tribute.tuple_type` type: cons cell (head, tail).
        /// Use `core.nil` as the tail terminator.
        /// Example: `(a, b, c)` → `TupleType(a, TupleType(b, TupleType(c, Nil)))`
        type tuple_type(head, tail);
    }
}

// === Type variable helper functions ===

/// Create a type variable (`tribute.type_var`) to be resolved during type inference.
///
/// The `attrs` can carry metadata such as a unique variable ID or constraints.
pub fn new_type_var<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> trunk_ir::Type<'db> {
    trunk_ir::Type::new(db, DIALECT_NAME(), TYPE_VAR(), IdVec::new(), attrs)
}

/// Create a type variable with a numeric ID.
pub fn type_var_with_id<'db>(db: &'db dyn salsa::Database, id: u64) -> trunk_ir::Type<'db> {
    new_type_var(
        db,
        BTreeMap::from([(VAR_ID_ATTR(), Attribute::IntBits(id))]),
    )
}

/// Create an error type (`tribute.error_type`) indicating type resolution failed.
///
/// The `attrs` can carry error information or source location.
pub fn new_error_type<'db>(db: &'db dyn salsa::Database, attrs: Attrs<'db>) -> trunk_ir::Type<'db> {
    trunk_ir::Type::new(db, DIALECT_NAME(), ERROR_TYPE(), IdVec::new(), attrs)
}

/// Check if a type is a type variable (`tribute.type_var`).
pub fn is_type_var(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), TYPE_VAR())
}

/// Check if a type is an error type (`tribute.error_type`).
pub fn is_error_type(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.is_dialect(db, DIALECT_NAME(), ERROR_TYPE())
}

/// Check if a type is an unresolved type reference (`tribute.type`).
///
/// These are type names that haven't been resolved to concrete types yet.
pub fn is_unresolved_type(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    ty.dialect(db) == DIALECT_NAME() && ty.name(db) == Symbol::new("type")
}

/// Check if a type is a placeholder that should be resolved before emit.
///
/// This includes `tribute.type_var`, `tribute.type`, and `tribute.error_type`.
pub fn is_placeholder_type(db: &dyn salsa::Database, ty: trunk_ir::Type<'_>) -> bool {
    is_type_var(db, ty) || is_unresolved_type(db, ty) || is_error_type(db, ty)
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

// tribute.type_var -> "a", "b", ..., "t0", "t1", ...
inventory::submit! {
    Printable::implement("tribute", "type_var", |db, ty, f| {
        if let Some(Attribute::IntBits(id)) = ty.get_attr(db, VAR_ID_ATTR()) {
            fmt_var_id(f, *id)
        } else {
            f.write_char('?')
        }
    })
}

// tribute.error_type -> "<error>"
inventory::submit! { Printable::implement("tribute", "error_type", |_, _, f| f.write_str("<error>")) }

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

/// Convert a variable ID to a readable name (a, b, c, ..., t0, t1, ...).
fn fmt_var_id(f: &mut PrintContext<'_, '_>, id: u64) -> std::fmt::Result {
    if id < 26 {
        f.write_char((b'a' + id as u8) as char)
    } else {
        write!(f, "t{}", id - 26)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::DialectType;
    use trunk_ir::dialect::core;
    use trunk_ir::type_interface::print_type;

    #[salsa_test]
    fn test_type_var_with_id(db: &salsa::DatabaseImpl) {
        let var_a = type_var_with_id(db, 0);
        assert!(is_type_var(db, var_a));
        assert_eq!(print_type(db, var_a), "a");

        let var_z = type_var_with_id(db, 25);
        assert_eq!(print_type(db, var_z), "z");

        let var_t0 = type_var_with_id(db, 26);
        assert_eq!(print_type(db, var_t0), "t0");
    }

    #[salsa_test]
    fn test_new_type_var(db: &salsa::DatabaseImpl) {
        let var = new_type_var(db, BTreeMap::new());
        assert!(is_type_var(db, var));
        assert!(!is_error_type(db, var));
    }

    #[salsa_test]
    fn test_new_error_type(db: &salsa::DatabaseImpl) {
        let err = new_error_type(db, BTreeMap::new());
        assert!(is_error_type(db, err));
        assert!(!is_type_var(db, err));
        assert_eq!(print_type(db, err), "<error>");
    }

    #[salsa_test]
    fn test_unresolved_type(db: &salsa::DatabaseImpl) {
        let ty = unresolved_type(db, Symbol::new("Int"), IdVec::new());
        assert!(is_unresolved_type(db, ty));
        assert_eq!(print_type(db, ty), "Int");

        // With type parameters
        let inner = type_var_with_id(db, 0);
        let list_ty = unresolved_type(db, Symbol::new("List"), IdVec::from(vec![inner]));
        assert!(is_unresolved_type(db, list_ty));
        assert_eq!(print_type(db, list_ty), "List(a)");
    }

    #[salsa_test]
    fn test_is_placeholder_type(db: &salsa::DatabaseImpl) {
        let type_var = type_var_with_id(db, 0);
        let error_type = new_error_type(db, BTreeMap::new());
        let unresolved = unresolved_type(db, Symbol::new("Foo"), IdVec::new());
        let concrete = core::I32::new(db).as_type();

        assert!(is_placeholder_type(db, type_var));
        assert!(is_placeholder_type(db, error_type));
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
