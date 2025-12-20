//! Source dialect operations.
//!
//! The `src` dialect represents unresolved AST constructs before name resolution
//! and type inference. All `src.*` operations should be eliminated after resolution.
//!
//! Additionally, this module provides type constructors for unresolved types:
//! - `src.type` - an unresolved type reference that needs name resolution
use crate::{IdVec, Symbol, dialect};

dialect! {
    mod src {
        /// `src.call` operation: unresolved function call.
        /// The callee name will be resolved to a concrete function reference.
        #[attr(name: SymbolRef)]
        fn call(#[rest] args) -> result;

        /// `src.var` operation: unresolved variable reference (single name).
        /// May resolve to local binding or module-level definition.
        #[attr(name: Symbol)]
        fn var() -> result;

        /// `src.path` operation: explicitly qualified path reference.
        /// Always refers to a module-level or type-level definition, never local.
        #[attr(path: SymbolRef)]
        fn path() -> result;

        /// `src.binop` operation: unresolved binary operation.
        /// Used for operators that need type-directed resolution (e.g., `<>` concat).
        /// The `op` attribute holds the operator name.
        #[attr(op: Symbol)]
        fn binop(lhs, rhs) -> result;

        /// `src.block` operation: block expression.
        /// Preserves block structure for source mapping and analysis.
        /// The body region contains the statements, and the result is the block's value.
        fn block() -> result {
            #[region(body)] {}
        };

        /// `src.yield` operation: yields a value from a block.
        /// Used to specify the result value of a `src.block`.
        fn r#yield(value);

        /// `src.lambda` operation: lambda expression.
        /// Represents an anonymous function before capture analysis.
        /// The `type` attribute holds the function type (params -> result).
        /// The body region contains the lambda body, ending with `src.yield`.
        #[attr(r#type: Type)]
        fn lambda() -> result {
            #[region(body)] {}
        };

        /// `src.tuple` operation: tuple construction.
        /// Takes variadic operands (tuple elements) and produces a tuple value.
        fn tuple(#[rest] elements) -> result;

        /// `src.const` operation: constant definition.
        /// Represents a named constant value before resolution.
        /// Unlike functions, constants are evaluated once and their value is inlined at use sites.
        /// The `value` attribute holds the literal value (IntBits, FloatBits, String, etc.).
        #[attr(name: Symbol, value: any)]
        fn r#const() -> result;

        /// `src.use` operation: import declaration.
        /// Carries the fully qualified path and an optional local alias.
        #[attr(path: SymbolRef, alias: Symbol, is_pub: bool)]
        fn r#use();

        /// `src.type`: an unresolved type reference that needs name resolution.
        /// The `name` attribute holds the type name (e.g., "Int", "List").
        /// The `params` hold type arguments for generic types (e.g., `List(a)`).
        #[attr(name: Symbol)]
        type r#type(#[rest] params);
    }
}

// === Convenience function for creating unresolved types ===

/// Create an unresolved type reference (`src.type`).
///
/// This is a convenience wrapper around `Type::new` that takes a string name.
pub fn unresolved_type<'db>(
    db: &'db dyn salsa::Database,
    name: &str,
    params: IdVec<crate::Type<'db>>,
) -> crate::Type<'db> {
    // Use the macro-generated Type struct
    *Type::new(db, params, Symbol::from_dynamic(name))
}

// === Printable interface registrations ===

use std::fmt::Write;

use crate::Attribute;
use crate::type_interface::Printable;

// src.type -> "Name" or "Name(params...)"
inventory::submit! {
    Printable::implement("src", "type", |db, ty, f| {
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
