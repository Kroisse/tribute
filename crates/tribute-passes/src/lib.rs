//! Compiler passes for Tribute.
//!
//! This crate contains the various transformation passes used in the Tribute compiler,
//! including CST to TrunkIR lowering, type inference, name resolution, and more.

// === TrunkIR passes ===
pub mod cst_to_tir;

// Re-exports
pub use cst_to_tir::lower_source_file;
