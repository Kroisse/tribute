//! Abstract Syntax Tree (AST) types for Tribute.
//!
//! This module defines the Salsa-tracked AST representation used throughout
//! the compiler frontend. The AST is designed with several key principles:
//!
//! ## Generic Phase Parameter
//!
//! AST types are parameterized by a "phase" type `V` that represents
//! what information is available about name references:
//!
//! - `UnresolvedName`: After parsing, before name resolution
//! - `ResolvedRef<'db>`: After name resolution
//! - `TypedRef<'db>`: After type checking
//!
//! This allows the same AST structure to be used throughout compilation
//! while the type system ensures correct phase handling.
//!
//! The types themselves place no bound on `V`. Each implements
//! `salsa::SalsaValue` exactly when `V` does, so code that only walks or
//! rebuilds the tree needs no Salsa bound of its own.
//!
//! ## NodeId + SpanMap Pattern
//!
//! Following rust-analyzer's approach, AST nodes don't store spans directly.
//! Instead, each node has a `NodeId` that can be used to look up its span
//! in a separate `SpanMap`. This has several benefits:
//!
//! - AST is purely structural (easier to work with)
//! - Span changes don't invalidate Salsa caches
//! - Additional metadata can be added using the same pattern
//!
//! ## Type vs TypeScheme
//!
//! The type system distinguishes between:
//!
//! - `Type`: Monomorphic types (during inference, may contain type variables)
//! - `TypeScheme`: Polymorphic types with universally quantified parameters
//!
//! This separation is important for proper handling of let-polymorphism
//! and generalization.
//!
//! ## Example Usage
//!
//! ```ignore
//! // After parsing
//! let parsed: ParsedModule = lower_cst_to_ast(db, source, cst);
//!
//! // After name resolution
//! let resolved: ResolvedModule<'db> = resolve_module(db, parsed);
//!
//! // After type checking
//! let typed: TypedModule<'db> = typecheck_module(db, &resolved);
//! ```

mod calling_convention;
mod decl;
mod expr;
pub mod lookup;
pub(crate) mod node_id;
mod pattern;
mod phases;
#[cfg(test)]
pub mod prop;
mod span_map;
mod types;
pub mod visit;

// Re-export core types
pub use calling_convention::*;
pub use decl::*;
pub use expr::*;
pub use node_id::*;
pub use pattern::*;
pub use phases::*;
pub use span_map::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use super::*;

    /// Reports whether `T` implements `salsa::SalsaValue`: the inherent
    /// method exists only under that bound and takes precedence over the
    /// trait fallback.
    struct Probe<T>(PhantomData<T>);

    trait NotSalsaValue {
        fn is_salsa_value(&self) -> bool {
            false
        }
    }

    impl<T> NotSalsaValue for Probe<T> {}

    impl<T: salsa::SalsaValue> Probe<T> {
        fn is_salsa_value(&self) -> bool {
            true
        }
    }

    /// A phase value that Salsa cannot store.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Opaque;

    macro_rules! assert_salsa_value_follows_phase {
        ($($ty:ident),* $(,)?) => {$(
            assert!(
                Probe::<$ty<UnresolvedName>>(PhantomData).is_salsa_value(),
                concat!(stringify!($ty), " stores a Salsa phase value"),
            );
            assert!(
                !Probe::<$ty<Opaque>>(PhantomData).is_salsa_value(),
                concat!(stringify!($ty), " requires a Salsa phase value"),
            );
        )*};
    }

    #[test]
    fn ast_nodes_are_salsa_values_exactly_when_their_phase_is() {
        assert_salsa_value_follows_phase!(
            Module,
            Decl,
            FuncDecl,
            ModuleDecl,
            Expr,
            ExprKind,
            Stmt,
            Arm,
            HandlerArm,
            HandlerKind,
            Pattern,
            PatternKind,
            FieldPattern,
        );
    }
}
