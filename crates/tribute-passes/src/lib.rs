//! Compiler passes for Tribute.
//!
//! This crate contains the various transformation passes used in the Tribute compiler,
//! including CST/AST to TrunkIR lowering, type inference, name resolution, and more.

// === TrunkIR passes ===
pub mod ast_to_tir;
pub mod cst_to_tir;

// === Legacy HIR (transitional) ===
pub mod hir;
pub mod lower;
pub mod queries;

// Re-exports
pub use ast_to_tir::lower_program;
pub use hir::*;
pub use lower::{FunctionDef, LowerError, LowerResult};
pub use queries::*;
