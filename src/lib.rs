pub mod ast;
pub mod database;
pub mod eval;
pub mod parser;

use salsa::Database as _;
use std::path::Path;

pub use crate::{
    database::{
        diagnostics, parse_source_file, Diagnostic, DiagnosticSeverity, Program, SourceFile,
        TrackedExpression, TributeDatabaseImpl,
    },
    parser::TributeParser,
};

// Legacy parse function using parse_with_database (kept for compatibility)
pub fn parse(path: &Path, source: &str) -> Vec<(ast::Expr, ast::SimpleSpan)> {
    TributeDatabaseImpl::default().attach(|db| {
        let (program, _diagnostics) = parse_with_database(db, path, source);

        program
            .expressions(db)
            .iter()
            .map(|tracked| (tracked.expr(db).clone(), tracked.span(db)))
            .collect()
    })
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
