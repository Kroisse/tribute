pub mod builtins;
pub mod eval;

use std::path::Path;

pub use crate::eval::{eval_hir_expr, eval_hir_program, Environment, Value};
pub use tribute_ast::{
    ast, parse_source_file, Diagnostic, DiagnosticSeverity, Item, Program, SourceFile,
    TributeDatabaseImpl, TributeParser,
};
pub use tribute_hir::{compile_to_hir, lower_source_to_hir};

// Parse a source file and return the program along with diagnostics
pub fn parse_with_database<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> (Program<'db>, Vec<&'db Diagnostic>) {
    let source_file = SourceFile::new(db, path.as_ref().to_path_buf(), source.to_string());
    let program = parse_source_file(db, source_file);
    let diags = parse_source_file::accumulated::<Diagnostic>(db, source_file);
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
    let hir_program = lower_source_to_hir(db, source_file).ok_or("Failed to lower AST to HIR")?;
    let mut env = Environment::toplevel();
    eval_hir_program(db, &mut env, hir_program)
}
