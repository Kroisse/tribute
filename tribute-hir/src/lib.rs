//! Tribute High-level Intermediate Representation (HIR)
//!
//! This crate provides a more structured intermediate representation for Tribute
//! programs, making compilation to MLIR more straightforward.

pub mod hir;
pub mod lower;
pub mod queries;

pub use hir::*;
pub use lower::*;
pub use queries::*;