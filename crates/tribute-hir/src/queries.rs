use crate::{HirExpr, HirFunction, HirProgram};
use salsa::Accumulator;
use std::collections::BTreeMap;
use tribute_ast::{CompilationPhase, Diagnostic, DiagnosticSeverity, Program, SourceFile};

/// Query to lower Program to HIR
#[salsa::tracked]
pub fn lower_program_to_hir<'db>(
    db: &'db dyn salsa::Database,
    program: Program<'db>,
) -> Option<HirProgram<'db>> {
    match crate::lower::lower_program_to_hir(db, program) {
        Ok((functions, main)) => {
            // Convert function definitions to tracked HIR types
            let mut hir_functions = BTreeMap::new();

            for (name, func_def) in functions {
                let body_exprs: Vec<_> = func_def
                    .body
                    .into_iter()
                    .map(|(expr, span)| HirExpr::new(db, expr, span))
                    .collect();

                let hir_func = HirFunction::new(
                    db,
                    func_def.name,
                    func_def.params,
                    body_exprs,
                    func_def.span,
                );
                hir_functions.insert(name, hir_func);
            }

            Some(HirProgram::new(db, hir_functions, main))
        }
        Err(e) => {
            Diagnostic {
                message: format!("HIR lowering error: {}", e),
                severity: DiagnosticSeverity::Error,
                span: tribute_ast::Span::new(0, 0), // Default span
                phase: CompilationPhase::HirLowering,
            }
            .accumulate(db);
            None
        }
    }
}

/// Query to lower AST to HIR for a source file (convenience wrapper)
#[salsa::tracked]
pub fn lower_source_to_hir<'db>(
    db: &'db dyn salsa::Database,
    source: SourceFile,
) -> Option<HirProgram<'db>> {
    let program = tribute_ast::parse_source_file(db, source);
    lower_program_to_hir(db, program)
}

/// Query to get all HIR diagnostics for a program
#[salsa::tracked]
pub fn hir_diagnostics_for_program<'db>(
    db: &'db dyn salsa::Database,
    program: Program<'db>,
) -> Vec<Diagnostic> {
    let _ = lower_program_to_hir(db, program);
    lower_program_to_hir::accumulated::<Diagnostic>(db, program)
        .into_iter()
        .cloned()
        .collect()
}

/// Query to get all HIR diagnostics for a source file
#[salsa::tracked]
pub fn hir_diagnostics<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Vec<Diagnostic> {
    let program = tribute_ast::parse_source_file(db, source);
    hir_diagnostics_for_program(db, program)
}

/// Helper function to compile source to HIR using tribute-ast's database
pub fn compile_to_hir(
    path: &std::path::Path,
    source: &str,
) -> (Option<HirProgram<'static>>, Vec<Diagnostic>) {
    use salsa::Database;

    let db = tribute_ast::TributeDatabaseImpl::default();
    let file = SourceFile::new(&db, path.to_path_buf(), source.to_string());

    // Since we need to return the HIR program, we use the database's attach method
    db.attach(|db| {
        let _hir_program = lower_source_to_hir(db, file);
        let diagnostics = hir_diagnostics(db, file);

        // We can't return the HIR program directly due to lifetime issues
        // This function is mainly for testing/demonstration purposes
        (None, diagnostics)
    })
}
