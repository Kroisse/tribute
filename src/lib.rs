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

use std::path::Path;

pub use crate::database::TributeDatabaseImpl;
pub use ropey::Rope;
use tree_sitter::Parser;
pub use tribute_front::SourceCst;
use tribute_front::source_file::parse_with_rope;
pub use tribute_front::{ParsedCst, lower_cst, lower_source_cst, parse_cst};
pub use tribute_passes::{Diagnostic, DiagnosticSeverity};
pub use trunk_ir::dialect::core::Module;

pub mod database;
pub mod pipeline;

pub use pipeline::{
    CompilationResult, compile, compile_with_diagnostics, stage_lower_case, stage_lower_to_wasm,
    stage_resolve, stage_tdnr, stage_typecheck,
};

/// Lower a Tribute source string to TrunkIR module.
#[deprecated]
pub fn lower_str<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> Module<'db> {
    let text = Rope::from_str(source);
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None).expect("tree");
    let source = SourceCst::from_path(db, path.as_ref(), text, Some(tree));
    lower_source_cst(db, source)
}
