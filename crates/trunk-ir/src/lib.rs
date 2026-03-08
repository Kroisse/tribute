//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate provides IR definitions for the multi-level dialect system.

#![recursion_limit = "512"]

// === Salsa-independent primitives ===
pub mod symbol;

// === ADT layout computation ===
pub mod adt_layout;

// === Arena-based IR ===
pub mod arena;

// === Source location types ===
pub mod location;

// === Operation interface (purity, isolation) ===
pub mod op_interface;

// === Operation utilities (ConversionError, raw_ident_str macro) ===
pub mod ops;

// === Dialect conversion utilities ===
pub mod conversion;

// === IR text format parser ===
pub mod parser;

// Re-export proc macro for arena dialect definitions
pub use trunk_ir_macros::arena_dialect;

// Re-export paste for use in macros
#[doc(hidden)]
pub use paste;

// Re-export smallvec for use in macros and external crates
pub use smallvec;

pub use location::{Span, Spanned};
pub use ops::ConversionError;
pub use symbol::{BlockId, IdVec, Symbol, SymbolVec};

pub use smallvec::smallvec as idvec;
