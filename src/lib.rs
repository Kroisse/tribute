//! Tribute programming language compiler library.
//!
//! This crate provides the main API for compiling Tribute programs.
//!
//! ## Pipeline
//!
//! The compilation pipeline has two Salsa-tracked stages:
//! 1. **Parsing**: `parse_cst` - Parse source to CST (Tree-sitter)
//! 2. **Lowering**: `lower_cst` - Lower CST to TrunkIR
//!
//! For convenience, `lower_source_cst` combines both stages.

pub use crate::database::TributeDatabaseImpl;
pub use ropey::Rope;
pub use tribute_front::SourceCst;
pub use tribute_front::{ParsedCst, lower_cst, lower_source_cst, parse_cst};
pub use tribute_passes::{Diagnostic, DiagnosticSeverity};
pub use trunk_ir::dialect::core::Module;

pub mod database;
pub mod pipeline;

pub use pipeline::{
    CompilationResult, compile, compile_for_lsp, compile_to_wasm_binary, compile_with_diagnostics,
    parse_and_lower,
};
