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
//! 4. **IR Generation**: typed AST → source-logical `tribute_control` IR
//! 5. **Shared Legalization**: callable/control IR → CPS and explicit evidence
//! 6. **Target Lowering**: Native or Wasm ABI, storage, and code generation

pub use crate::database::TributeDatabaseImpl;
pub use ropey::Rope;
pub use tribute_core::diagnostic::{Diagnostic, DiagnosticSeverity};
pub use tribute_front::SourceCst;
pub use tribute_front::{ParsedCst, parse_cst};

pub mod database;
pub mod pipeline;

pub use pipeline::{
    CompilationConfig, CompilationResult, LinkError, compile_ast, compile_frontend,
    compile_to_native_binary, compile_to_wasm_binary, compile_with_diagnostics, dump_ir,
    link_native_binary, parse_and_lower_ast,
};
