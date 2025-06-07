pub mod builtins;
pub mod eval;

use std::path::Path;

pub use tribute_ast::{
    diagnostics, parse_source_file, Diagnostic, DiagnosticSeverity, Program, SourceFile,
    Item, TributeDatabaseImpl, TributeParser, ast,
};
pub use tribute_hir::{lower_source_to_hir, compile_to_hir};
pub use crate::eval::{Environment, Value, eval_hir_program, eval_hir_expr};

// Legacy parse function (kept for compatibility)
pub fn parse(path: &Path, source: &str) -> Vec<(ast::Expr, ast::SimpleSpan)> {
    let db = TributeDatabaseImpl::default();
    let (program, _) = parse_with_database(&db, path, source);
    
    program
        .items(&db)
        .iter()
        .map(|item| item.expr(&db).clone())
        .collect()
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

// HIR-based evaluation function
pub fn eval_with_hir<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> Result<Value, Box<dyn std::error::Error + 'static>> {
    use crate::eval::Environment;
    
    let source_file = SourceFile::new(db, path.as_ref().to_path_buf(), source.to_string());
    let hir_program = lower_source_to_hir(db, source_file)
        .ok_or("Failed to lower AST to HIR")?;
    let mut env = Environment::toplevel();
    eval_hir_program(db, &mut env, hir_program)
}
