//! Tribute TrunkIR crate.
//!
//! `new-plans/ir.md` defines TrunkIR as the compiler's central multi-level dialect IR.
//! This crate is the new home for IR definitions and lowering queries.
//!
//! Note: the current implementation is transitional and still exposes the existing
//! HIR-shaped IR while the dialect-based TrunkIR is introduced incrementally.

// === Dialect modules ===
// Infrastructure
pub mod core;

// Mid-level (target independent)
pub mod adt;
pub mod arith;
pub mod cont;
pub mod func;
pub mod mem;
pub mod scf;

// === Other modules ===
pub mod hir;
pub mod ir;
pub mod lower;
pub mod ops;
pub mod queries;
pub mod types;

// Re-export paste for use in macros
#[doc(hidden)]
pub use paste;

// Re-export Location for use in macros
pub use tribute_core::Location;

pub use hir::*;
pub use ir::*;
pub use lower::{FunctionDef, LowerError, LowerResult};
pub use ops::{ConversionError, DialectOp};
pub use queries::*;
pub use types::*;
