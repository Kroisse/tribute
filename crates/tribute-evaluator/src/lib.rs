//! Tribute MLIR Evaluator
//!
//! This crate provides MLIR-based evaluation for the Tribute programming language.

pub mod evaluator;

pub use evaluator::{MLIREvaluator, TributeValue};

// Re-export commonly used melior types
pub use melior::{Context, ir::{Module, Operation}};