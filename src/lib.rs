//! Tribute programming language compiler library.
//!
//! This crate provides the main API for compiling Tribute programs.
//! The compilation pipeline is: CST (Tree-sitter) → TrunkIR → (backend)

use std::path::Path;

pub use tribute_core::{Diagnostic, DiagnosticSeverity, SourceFile, TributeDatabaseImpl};
pub use tribute_passes::lower_source_file;
pub use tribute_trunk_ir::dialect::core::Module;

/// Lower a Tribute source string to TrunkIR module.
pub fn lower_str<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> Module<'db> {
    let source_file = SourceFile::new(db, path.as_ref().to_path_buf(), source.to_string());
    lower_source_file(db, source_file)
}
