//! Tribute language AST/HIR dialect.
//!
//! This dialect represents Tribute-specific high-level operations.
//!
//! ## Dialect Organization
//!
//! **Types**:
//! - `tribute.type` (unresolved type reference)

use std::fmt::Write;

use trunk_ir::type_interface::Printable;
use trunk_ir::{Attribute, IdVec, Symbol, dialect};

dialect! {
    mod tribute {
        // === Types ===

        /// `tribute.type`: an unresolved type reference that needs name resolution.
        /// The `name` attribute holds the type name (e.g., "Int", "List").
        /// The `params` hold type arguments for generic types (e.g., `List(a)`).
        #[attr(name: Symbol)]
        type r#type(#[rest] params);

        // NOTE: Primitive types (int, nat, float, bool) are now in `tribute_rt` dialect.
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
}
