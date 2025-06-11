//! Tribute MLIR Dialect
//!
//! This crate provides a custom MLIR dialect for the Tribute programming language,
//! enabling direct AST to MLIR lowering and MLIR-based evaluation.

pub mod dialect;
pub mod types;
pub mod ops;
pub mod lowering;
pub mod hir_lowering;
pub mod errors;
pub mod salsa_integration;

// Include generated code from TableGen
include!(concat!(env!("OUT_DIR"), "/tablegen_ops.rs"));

pub use dialect::TributeDialect;
pub use errors::{LoweringError, EvaluationError};
pub use lowering::AstToMLIRLowerer;
pub use hir_lowering::HirToMLIRLowerer;

// Re-export commonly used melior types
pub use melior::{Context, ir::{Module, Operation}};