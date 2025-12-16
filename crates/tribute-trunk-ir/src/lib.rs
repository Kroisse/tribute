//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate is the new home for IR definitions and lowering queries.
//!
//! Note: the current implementation is transitional and still exposes the existing
//! HIR-shaped IR while the dialect-based TrunkIR is introduced incrementally.

pub mod hir;
pub mod lower;
pub mod queries;

pub use hir::*;
pub use lower::{FunctionDef, LowerError, LowerResult};
pub use queries::*;
