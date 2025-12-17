//! Compilation pipeline for Tribute.
//!
//! This module defines the compilation stages as Salsa tracked functions.
//! Each stage is independently cacheable for incremental compilation.
//!
//! ## Pipeline Stages
//!
//! ```text
//! SourceFile
//!     │
//!     ▼
//! parse_cst ─► ParsedCst
//!     │
//!     ▼
//! lower_cst ─► Module (src.* ops, type.var)
//!     │
//!     ▼
//! resolve_names ─► Module (resolved references)
//!     │
//!     ▼
//! infer_types ─► Module (concrete types)
//! ```
//!
//! ## Incremental Compilation
//!
//! Salsa tracks dependencies between queries. If a source file changes:
//! - `parse_cst` re-runs for that file
//! - `lower_cst` re-runs if the CST changed
//! - `resolve_names` re-runs if the module changed
//! - `infer_types` re-runs if resolved module changed
//!
//! If only a type annotation changes in file A, but file B's code hasn't changed,
//! file B won't be re-parsed or re-lowered.

use tribute_core::SourceFile;
use tribute_trunk_ir::dialect::core::Module;

use crate::cst_to_tir::{lower_cst, parse_cst};
use crate::typeck::{TypeChecker, TypeSolver};

/// Result of the full compilation pipeline.
pub struct CompilationResult<'db> {
    /// The compiled module with resolved types.
    pub module: Module<'db>,
    /// The type solver with final substitutions.
    pub solver: TypeSolver<'db>,
    /// Diagnostics collected during compilation.
    pub diagnostics: Vec<CompilationDiagnostic>,
}

/// A compilation diagnostic (error or warning).
#[derive(Clone, Debug)]
pub struct CompilationDiagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    // TODO: Add span/location
}

/// Severity of a diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

// =============================================================================
// Pipeline Stages
// =============================================================================

/// Stage 1: Parse source to CST.
///
/// Re-exported from `cst_to_tir` for convenience.
pub use crate::cst_to_tir::parse_cst as stage_parse;

/// Stage 2: Lower CST to TrunkIR.
///
/// Re-exported from `cst_to_tir` for convenience.
pub use crate::cst_to_tir::lower_cst as stage_lower;

/// Stage 3: Resolve names in the module.
///
/// This pass resolves:
/// - `src.var` → concrete variable references or function calls
/// - `src.call` → resolved function references
/// - `src.path` → resolved module paths
/// - `src.type` → concrete type references
///
/// After this pass, all `src.*` operations should be eliminated.
#[salsa::tracked]
pub fn stage_resolve<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Module<'db> {
    // Get the lowered module from the previous stage
    let Some(cst) = parse_cst(db, source) else {
        // Parse failure - return empty module
        let path = tribute_core::PathId::new(db, source.path(db));
        let location = tribute_core::Location::new(path, tribute_core::Span::new(0, 0));
        return Module::build(db, location, "main", |_| {});
    };
    // TODO: Implement name resolution
    // For now, pass through unchanged
    lower_cst(db, source, cst)
}

/// Stage 4: Infer and check types.
///
/// This pass:
/// - Collects type constraints from the module
/// - Solves constraints via unification
/// - Substitutes inferred types back into the module
/// - Reports type errors
#[salsa::tracked]
pub fn stage_typecheck<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Module<'db> {
    // Get the resolved module from the previous stage
    let module = stage_resolve(db, source);

    let mut checker = TypeChecker::new(db);
    checker.check_module(&module);

    // Solve constraints
    match checker.solve() {
        Ok(_solver) => {
            // TODO: Apply substitution back to module
            // For now, return unchanged
            module
        }
        Err(_err) => {
            // TODO: Report type error
            module
        }
    }
}

// =============================================================================
// Full Pipeline
// =============================================================================

/// Run the full compilation pipeline on a source file.
///
/// This combines all stages:
/// 1. Parse CST
/// 2. Lower to TrunkIR
/// 3. Resolve names
/// 4. Infer types
#[salsa::tracked]
pub fn compile<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Module<'db> {
    stage_typecheck(db, source)
}

/// Run compilation and return detailed results including diagnostics.
pub fn compile_with_diagnostics<'db>(
    db: &'db dyn salsa::Database,
    source: SourceFile,
) -> CompilationResult<'db> {
    let mut diagnostics = Vec::new();

    // Stage 1: Parse
    let Some(cst) = parse_cst(db, source) else {
        diagnostics.push(CompilationDiagnostic {
            severity: DiagnosticSeverity::Error,
            message: "Failed to parse source file".to_string(),
        });

        let path = tribute_core::PathId::new(db, source.path(db));
        let location = tribute_core::Location::new(path, tribute_core::Span::new(0, 0));
        let module = Module::build(db, location, "main", |_| {});

        return CompilationResult {
            module,
            solver: TypeSolver::new(db),
            diagnostics,
        };
    };

    // Stage 2: Lower
    let module = lower_cst(db, source, cst);

    // Stage 3: Resolve names (currently passthrough)
    // let module = stage_resolve(db, source);
    // We skip the tracked function here to get more control over diagnostics

    // Stage 4: Type check
    let mut checker = TypeChecker::new(db);
    checker.check_module(&module);

    let solver = match checker.solve() {
        Ok(solver) => solver,
        Err(err) => {
            diagnostics.push(CompilationDiagnostic {
                severity: DiagnosticSeverity::Error,
                message: format!("Type error: {:?}", err),
            });
            TypeSolver::new(db)
        }
    };

    CompilationResult {
        module,
        solver,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::Database;
    use tribute_core::TributeDatabaseImpl;

    #[salsa::tracked]
    fn test_compile(db: &dyn salsa::Database, source: SourceFile) -> Module<'_> {
        compile(db, source)
    }

    #[test]
    fn test_full_pipeline() {
        TributeDatabaseImpl::default().attach(|db| {
            let source = SourceFile::new(
                db,
                std::path::PathBuf::from("test.tr"),
                "fn main() -> Int { 42 }".to_string(),
            );

            let module = test_compile(db, source);
            assert_eq!(module.name(db), "main");
        });
    }

    #[test]
    fn test_compile_with_diagnostics() {
        TributeDatabaseImpl::default().attach(|db| {
            let source = SourceFile::new(
                db,
                std::path::PathBuf::from("test.tr"),
                "fn add(x: Int, y: Int) -> Int { x + y }".to_string(),
            );

            let result = compile_with_diagnostics(db, source);
            // Should compile without errors
            assert!(
                result.diagnostics.is_empty(),
                "Expected no diagnostics, got: {:?}",
                result.diagnostics
            );
        });
    }
}
