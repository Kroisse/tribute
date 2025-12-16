#[cfg(feature = "legacy-eval")]
pub mod builtins;

#[cfg(feature = "legacy-eval")]
#[deprecated(
    note = "Legacy HIR evaluator (not compiled by default). Enable feature `legacy-eval` if you still need it."
)]
pub mod eval;

use std::path::Path;

#[cfg(feature = "legacy-eval")]
pub use crate::eval::{Environment, Value, eval_hir_expr, eval_hir_program};
pub use tribute_ast::{Item, Program, TributeParser, ast, parse_source_file};
pub use tribute_core::{Diagnostic, DiagnosticSeverity, SourceFile, TributeDatabaseImpl};
pub use tribute_passes::{compile_to_hir, lower_source_to_hir};

/// Parse a Tribute source string and return the program along with diagnostics
pub fn parse_str<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> (Program<'db>, Vec<&'db Diagnostic>) {
    let source_file = SourceFile::new(db, path.as_ref().to_path_buf(), source.to_string());
    let program = parse_source_file(db, source_file);
    let diags = parse_source_file::accumulated::<Diagnostic>(db, source_file);
    (program, diags)
}

/// Evaluate a Tribute program from a source string
#[cfg(feature = "legacy-eval")]
#[deprecated(
    note = "Legacy evaluation API (not compiled by default). Enable feature `legacy-eval` if you still need it."
)]
pub fn eval_str<'db>(
    db: &'db dyn salsa::Database,
    path: &(impl AsRef<Path> + ?Sized),
    source: &str,
) -> Result<Value, Box<dyn std::error::Error + 'static>> {
    use crate::eval::Environment;

    let source_file = SourceFile::new(db, path.as_ref().to_path_buf(), source.to_string());

    // Check for parse errors first
    let _program = parse_source_file(db, source_file);
    let diagnostics = parse_source_file::accumulated::<Diagnostic>(db, source_file);

    if !diagnostics.is_empty() {
        // Return the first error
        let first_error = &diagnostics[0];
        return Err(format!("{}: {}", first_error.severity, first_error.message).into());
    }

    let hir_program = lower_source_to_hir(db, source_file).ok_or("Failed to lower AST to HIR")?;
    let mut env = Environment::toplevel();
    eval_hir_program(db, &mut env, hir_program)
}
