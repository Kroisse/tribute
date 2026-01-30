//! Tribute programming language compiler library.
//!
//! This crate provides the main API for compiling Tribute programs.
//!
//! ## Pipeline
//!
//! The compilation pipeline flows through:
//! 1. **Parsing**: `parse_cst` - Parse source to CST (Tree-sitter)
//! 2. **AST Lowering**: CST → AST via `astgen`
//! 3. **Name Resolution & Type Checking**: AST passes
//! 4. **IR Generation**: AST → TrunkIR

pub use crate::database::TributeDatabaseImpl;
pub use ropey::Rope;
pub use tribute_front::SourceCst;
pub use tribute_front::{ParsedCst, parse_cst};
pub use tribute_passes::{Diagnostic, DiagnosticSeverity};
pub use trunk_ir::dialect::core::Module;

pub mod database;
pub mod pipeline;

pub use pipeline::{
    CompilationResult, compile_ast, compile_for_lsp, compile_to_wasm_binary,
    compile_with_diagnostics, parse_and_lower_ast,
};
