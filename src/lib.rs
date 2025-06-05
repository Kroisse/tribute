pub mod builtins;
pub mod eval;

use salsa::Database as _;
use std::path::Path;

pub use tribute_ast::{
    diagnostics, parse_source_file, Diagnostic, DiagnosticSeverity, Program, SourceFile,
    Item, TributeDatabaseImpl, TributeParser, ast,
};
pub use crate::eval::{eval_expr, Environment, Value};

// Legacy parse function using parse_with_database (kept for compatibility)
pub fn parse(path: &Path, source: &str) -> Vec<(ast::Expr, ast::SimpleSpan)> {
    tribute_ast::parse_with_database(path, source).0
}

// New Salsa-based parse function
pub fn parse_with_database<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> (Program<'db>, Vec<Diagnostic>) {
    let source_file = SourceFile::new(db, path.as_ref().to_path_buf(), source.to_string());
    let program = parse_source_file(db, source_file);
    let diags = diagnostics(db, source_file);
    (program, diags)
}
