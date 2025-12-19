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
//! For convenience, `lower_source_file` combines both stages.

use std::path::Path;

pub use tribute_core::{SourceFile, TributeDatabaseImpl};
pub use tribute_passes::{
    Diagnostic, DiagnosticSeverity, ParsedCst, lower_cst, lower_source_file, parse_cst,
};
pub use tribute_trunk_ir::dialect::core::Module;

/// Lower a Tribute source string to TrunkIR module.
pub fn lower_str<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> Module<'db> {
    let source_file = SourceFile::from_path(db, path.as_ref(), source.to_string());
    lower_source_file(db, source_file)
}
