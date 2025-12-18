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
//! stage_resolve ─► Module (resolved references, some src.call for UFCS)
//!     │
//!     ▼
//! stage_typecheck ─► Module (concrete types)
//!     │
//!     ▼
//! stage_tdnr ─► Module (UFCS method calls resolved)
//! ```
//!
//! ## Incremental Compilation
//!
//! Salsa tracks dependencies between queries. If a source file changes:
//! - `parse_cst` re-runs for that file
//! - `lower_cst` re-runs if the CST changed
//! - `stage_resolve` re-runs if the module changed
//! - `stage_typecheck` re-runs if resolved module changed
//! - `stage_tdnr` re-runs if typed module changed
//!
//! If only a type annotation changes in file A, but file B's code hasn't changed,
//! file B won't be re-parsed or re-lowered.
//!
//! ## Diagnostics
//!
//! Diagnostics are collected using Salsa accumulators. Each stage can emit
//! diagnostics via `Diagnostic { ... }.accumulate(db)`, which are then
//! collected at the end of compilation.

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity, SourceFile, Span};
use tribute_trunk_ir::dialect::core::Module;

use crate::cst_to_tir::{lower_cst, parse_cst};
use crate::resolve::{Resolver, build_env};
use crate::tdnr::resolve_tdnr;
use crate::typeck::{TypeChecker, TypeSolver, apply_subst_to_module};

// Re-export for convenience
pub use crate::resolve::build_env as build_module_env;

/// Result of the full compilation pipeline.
pub struct CompilationResult<'db> {
    /// The compiled module with resolved types.
    pub module: Module<'db>,
    /// The type solver with final substitutions.
    pub solver: TypeSolver<'db>,
    /// Diagnostics collected during compilation.
    pub diagnostics: Vec<Diagnostic>,
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
/// - `src.var` → `func.constant` or `adt.struct_new`/`adt.variant_new`
/// - `src.call` → `func.call` with resolved callee
/// - `src.path` → resolved module paths
///
/// After this pass, all resolvable `src.*` operations are transformed.
/// Some may remain for type-directed resolution (UFCS).
#[salsa::tracked]
pub fn stage_resolve<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Module<'db> {
    // Get the lowered module from the previous stage
    let Some(cst) = parse_cst(db, source) else {
        // Parse failure - return empty module
        let path = tribute_core::PathId::new(db, source.path(db));
        let location = tribute_core::Location::new(path, tribute_core::Span::new(0, 0));
        return Module::build(db, location, "main", |_| {});
    };

    let module = lower_cst(db, source, cst);

    // Build module environment from declarations
    let env = build_env(db, &module);

    // Resolve names in the module
    let mut resolver = Resolver::new(db, env);
    resolver.resolve_module(&module)
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
        Ok(solver) => {
            // Apply substitution to replace type variables with concrete types
            apply_subst_to_module(db, module, solver.type_subst())
        }
        Err(_err) => {
            // TODO: Report type error
            module
        }
    }
}

/// Stage 5: Type-Directed Name Resolution (TDNR).
///
/// This pass resolves UFCS method calls that couldn't be resolved during
/// initial name resolution because they required type information.
///
/// For example:
/// - `list.len()` → `List::len(list)` (based on list's type being `List(a)`)
/// - `x.map(f)` → `Type::map(x, f)` (based on x's inferred type)
#[salsa::tracked]
pub fn stage_tdnr<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Module<'db> {
    let module = stage_typecheck(db, source);
    resolve_tdnr(db, module)
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
/// 5. TDNR (Type-Directed Name Resolution)
/// 6. Final resolution pass (reports unresolved references)
#[salsa::tracked]
pub fn compile<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Module<'db> {
    let module = stage_tdnr(db, source);

    // Final pass: resolve any remaining unresolved references and emit diagnostics
    let env = build_env(db, &module);
    let mut resolver = Resolver::with_unresolved_reporting(db, env);
    resolver.resolve_module(&module)
}

/// Run compilation and return detailed results including diagnostics.
///
/// Diagnostics are collected using Salsa accumulators from all compilation stages.
pub fn compile_with_diagnostics<'db>(
    db: &'db dyn salsa::Database,
    source: SourceFile,
) -> CompilationResult<'db> {
    // Run the full compilation pipeline (which checks for unresolved references)
    let module = compile(db, source);

    // Re-run type checking to capture the solver for the result
    // (compile already checked types, but we need the solver for diagnostics)
    let resolved_module = stage_resolve(db, source);
    let mut checker = TypeChecker::new(db);
    checker.check_module(&resolved_module);

    let solver = match checker.solve() {
        Ok(solver) => solver,
        Err(err) => {
            // Emit type error diagnostic via accumulator
            Diagnostic {
                message: format!("Type error: {err}"),
                span: Span::new(0, 0), // TODO: Extract span from error
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(db);
            TypeSolver::new(db)
        }
    };

    // Collect all accumulated diagnostics from the compilation
    let diagnostics = compile::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .cloned()
        .collect();

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

    #[test]
    fn test_unresolved_reference_diagnostic() {
        TributeDatabaseImpl::default().attach(|db| {
            let source = SourceFile::new(
                db,
                std::path::PathBuf::from("test.tr"),
                // Reference to undefined variable `undefined_var`
                "fn main() -> Int { undefined_var }".to_string(),
            );

            let result = compile_with_diagnostics(db, source);
            // Should have an unresolved reference error
            assert!(
                !result.diagnostics.is_empty(),
                "Expected diagnostic for unresolved reference"
            );

            // Check that the diagnostic message mentions the unresolved name
            let has_unresolved_error = result.diagnostics.iter().any(|d| {
                d.message.contains("unresolved")
                    && d.severity == DiagnosticSeverity::Error
                    && d.phase == CompilationPhase::NameResolution
            });
            assert!(
                has_unresolved_error,
                "Expected unresolved name error, got: {:?}",
                result.diagnostics
            );
        });
    }
}
