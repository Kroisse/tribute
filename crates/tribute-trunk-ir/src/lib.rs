//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate provides IR definitions for the multi-level dialect system.

// === Dialect modules ===
pub mod dialect;

// === IR infrastructure ===
pub mod ir;
pub mod ops;
pub mod types;

// Re-export paste for use in macros
#[doc(hidden)]
pub use paste;

// Re-export smallvec for use in macros and external crates
pub use smallvec;

// Re-export Location for use in macros
pub use tribute_core::Location;

pub use ir::{Block, BlockBuilder, Operation, Region, Symbol, Value};
pub use ops::{ConversionError, DialectOp};
pub use types::{Attribute, Attrs, DialectType, Type};

/// Small vector for values tracked by Salsa framework.
pub type IdVec<T> = smallvec::SmallVec<[T; 2]>;
pub use smallvec::smallvec as idvec;
