//! Tribute language-specific IR dialects.
//!
//! This crate provides dialects specific to the Tribute programming language,
//! built on top of the trunk-ir infrastructure.

pub mod callable_abi;
pub mod calling_convention;
pub mod continuation_frame;
pub mod dialect;
pub mod metadata;
pub mod runtime_items;
pub mod runtime_layout;

// Re-export common trunk-ir types for convenience
pub use trunk_ir::{BlockId, ConversionError, IdVec, Span, Symbol, idvec};
