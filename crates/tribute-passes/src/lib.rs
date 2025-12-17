//! Compiler passes for Tribute.
//!
//! This crate contains the various transformation passes used in the Tribute compiler,
//! including CST to TrunkIR lowering, type inference, name resolution, and more.
//!
//! ## Pipeline
//!
//! The compilation pipeline has two main stages:
//!
//! 1. **Parsing**: `parse_cst` - Parse source to CST (Tree-sitter)
//! 2. **Lowering**: `lower_cst` - Lower CST to TrunkIR
//!
//! Both stages are Salsa-tracked for incremental compilation.
//! The convenience function `lower_source_file` combines both stages.

// === TrunkIR passes ===
pub mod cst_to_tir;

// Re-exports
pub use cst_to_tir::{ParsedCst, lower_cst, lower_source_file, parse_cst};
