//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate provides IR definitions for the multi-level dialect system.

// === Dialect modules ===
pub mod dialect;

// === IR infrastructure ===
pub mod ir;
pub mod location;
pub mod ops;
pub mod qualified_name;
pub mod rewrite;
pub mod type_interface;
pub mod types;

// Re-export paste for use in macros
#[doc(hidden)]
pub use paste;

// Re-export smallvec for use in macros and external crates
pub use smallvec;

pub use ir::{Block, BlockBuilder, Operation, QualifiedName, Region, Symbol, Value, ValueDef};
pub use location::{Location, PathId, Span, Spanned};
pub use ops::{ConversionError, DialectOp};
pub use qualified_name::QualifiedName;
pub use types::{Attribute, Attrs, DialectType, Type};

/// Small vector for values tracked by Salsa framework.
pub type IdVec<T> = smallvec::SmallVec<[T; 2]>;
pub use smallvec::smallvec as idvec;

/// Small vector for symbols.
pub type SymbolVec = smallvec::SmallVec<[Symbol; 4]>;
