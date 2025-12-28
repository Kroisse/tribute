//! Compilation pipeline for Tribute.
//!
//! This module defines the compilation stages as Salsa tracked functions.
//! Each stage is independently cacheable for incremental compilation.
//!
//! ## Pipeline Stages
//!
//! ```text
//! SourceCst
//!     │
//!     ▼
//! parse_cst ─► ParsedCst
//!     │
//!     ▼
//! lower_cst ─► Module (src.* ops, type.var)
//!     │
//!     ▼
//! stage_resolve ─► Module (resolved references, const refs marked)
//!     │
//!     ▼
//! stage_const_inline ─► Module (const values inlined)
//!     │
//!     ▼
//! stage_typecheck ─► Module (concrete types)
//!     │
//!     ▼
//! stage_lambda_lift ─► Module (lambdas lifted, closure.new created)
//!     │
//!     ▼
//! stage_tdnr ─► Module (UFCS method calls resolved)
//!     │
//!     ▼
//! stage_lower_case ─► Module (case.case lowered to scf.if)
//!     │
//!     ▼
//! stage_dce ─► Module (dead code eliminated)
//!     │
//!     ├─► [No target] ─► Full Tribute module (diagnostic/analysis)
//!     │
//!     └─► [target: wasm] ─► stage_lower_to_wasm ─► WasmBinary (WebAssembly bytes)
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

use crate::SourceCst;
use ropey::Rope;
use salsa::Accumulator;
use tree_sitter::Parser;
use tribute_front::source_file::parse_with_rope;
use tribute_front::{lower_cst, parse_cst};
use tribute_passes::closure_lower::lower_closures;
use tribute_passes::const_inline::inline_module;
use tribute_passes::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_passes::lambda_lift::lift_lambdas;
use tribute_passes::lower_case_to_scf;
use tribute_passes::resolve::{Resolver, build_env};
use tribute_passes::tdnr::resolve_tdnr;
use tribute_passes::typeck::{TypeChecker, TypeSolver, apply_subst_to_module};
use tribute_wasm_backend::{WasmBinary, compile_to_wasm};
use trunk_ir::Span;
use trunk_ir::dialect::core::Module;
use trunk_ir::transforms::eliminate_dead_functions;
use trunk_ir::{Block, BlockId, IdVec, Region, Symbol};

// =============================================================================
// Standard Library Prelude
// =============================================================================

/// The prelude source code, embedded at compile time.
const PRELUDE_SOURCE: &str = include_str!("../lib/std/prelude.trb");

/// Load and cache the prelude module.
///
/// This is a Salsa tracked function, so the prelude is parsed only once
/// and cached for all subsequent compilations.
#[salsa::tracked]
pub fn prelude_module<'db>(db: &'db dyn salsa::Database) -> Option<Module<'db>> {
    let uri = fluent_uri::Uri::parse_from("prelude:///std/prelude".to_owned())
        .expect("valid prelude URI");
    let text: Rope = PRELUDE_SOURCE.into();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None)?;
    let source = SourceCst::new(db, uri, text, Some(tree));
    let cst = parse_cst(db, source)?;
    Some(lower_cst(db, source, cst))
}

/// Merge prelude definitions into a user module.
///
/// This prepends all prelude operations (type definitions, etc.) to the
/// user module's body, making prelude types available for name resolution.
pub fn merge_with_prelude<'db>(
    db: &'db dyn salsa::Database,
    user_module: Module<'db>,
) -> Module<'db> {
    let Some(prelude) = prelude_module(db) else {
        return user_module;
    };

    // Get operations from both modules
    let prelude_body = prelude.body(db);
    let user_body = user_module.body(db);

    // Merge blocks: prepend prelude operations to user operations
    let prelude_blocks = prelude_body.blocks(db);
    let user_blocks = user_body.blocks(db);

    // If both have a single block (common case), merge their operations
    if prelude_blocks.len() == 1 && user_blocks.len() == 1 {
        let prelude_block = &prelude_blocks[0];
        let user_block = &user_blocks[0];

        // Combine operations: prelude first, then user
        let mut combined_ops: IdVec<_> = prelude_block.operations(db).iter().copied().collect();
        combined_ops.extend(user_block.operations(db).iter().copied());

        let merged_block = Block::new(
            db,
            BlockId::fresh(),
            user_block.location(db),
            user_block.args(db).clone(),
            combined_ops,
        );

        let merged_body = Region::new(db, user_body.location(db), IdVec::from(vec![merged_block]));

        return Module::create(
            db,
            user_module.location(db),
            user_module.name(db),
            merged_body,
        );
    }

    // Fallback: just use user module if structure doesn't match
    user_module
}

// Re-export for convenience
pub use tribute_passes::resolve::build_env as build_module_env;

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
/// Re-exported from `tribute-front` for convenience.
pub use tribute_front::parse_cst as stage_parse;

/// Stage 2: Lower CST to TrunkIR.
///
/// Re-exported from `tribute-front` for convenience.
pub use tribute_front::lower_cst as stage_lower;

/// Stage 3: Resolve names in the module.
///
/// This pass resolves:
/// - `src.var` → `func.constant` or `adt.struct_new`/`adt.variant_new`
/// - `src.call` → `func.call` with resolved callee
/// - `src.path` → resolved module paths
///
/// After this pass, all resolvable `src.*` operations are transformed.
/// Some may remain for type-directed resolution (UFCS).
///
/// The prelude is automatically merged into the module before resolution,
/// making standard library types (Option, Result, etc.) available.
#[salsa::tracked]
pub fn stage_resolve<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let Some(cst) = parse_cst(db, source) else {
        let path = trunk_ir::PathId::new(db, source.uri(db).as_str().to_owned());
        let location = trunk_ir::Location::new(path, trunk_ir::Span::new(0, 0));
        return Module::build(db, location, Symbol::new("main"), |_| {});
    };
    emit_parse_errors(db, &cst);
    let user_module = lower_cst(db, source, cst);

    // Merge prelude definitions into the user module
    let module = merge_with_prelude(db, user_module);

    // Build module environment from declarations (including prelude)
    let env = build_env(db, &module);

    // Resolve names in the module
    let mut resolver = Resolver::new(db, env);
    resolver.resolve_module(&module)
}

fn emit_parse_errors(db: &dyn salsa::Database, cst: &tribute_front::ParsedCst) {
    let root = cst.root_node();
    if !root.has_error() {
        return;
    }

    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.is_error() || node.is_missing() {
            let message = if node.is_missing() {
                // For missing nodes, show what kind was expected
                let kind = node.kind();
                let parent_ctx = node
                    .parent()
                    .map(|p| format!(" in {}", format_node_kind(p.kind())))
                    .unwrap_or_default();
                format!("missing {}{}", format_node_kind(kind), parent_ctx)
            } else {
                // For error nodes, try to provide context
                let parent = node.parent();
                let parent_kind = parent.map(|p| p.kind()).unwrap_or("source_file");

                // Check what unexpected content was found
                let content_preview = if node.end_byte() > node.start_byte() {
                    // We don't have direct access to source here, so just note there's content
                    " (unexpected token)".to_string()
                } else {
                    String::new()
                };

                format!(
                    "syntax error{} while parsing {}",
                    content_preview,
                    format_node_kind(parent_kind)
                )
            };
            Diagnostic {
                message,
                span: Span::new(node.start_byte(), node.end_byte()),
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Parsing,
            }
            .accumulate(db);
            continue;
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            stack.push(child);
        }
    }
}

/// Format a tree-sitter node kind into a human-readable name.
fn format_node_kind(kind: &str) -> String {
    // Convert snake_case to space-separated words
    kind.replace('_', " ")
}

/// Stage 3.5: Inline constant values.
///
/// This pass inlines constant values at their use sites:
/// - Finds `src.var` operations marked with `resolved_const=true`
/// - Replaces them with `arith.const` operations containing the inlined value
///
/// This happens after name resolution (which marks const references)
/// but before type checking (which needs the concrete inlined values).
#[salsa::tracked]
pub fn stage_const_inline<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = stage_resolve(db, source);
    inline_module(db, &module)
}

/// Stage 4: Infer and check types.
///
/// This pass:
/// - Collects type constraints from the module
/// - Solves constraints via unification
/// - Substitutes inferred types back into the module
/// - Reports type errors
#[salsa::tracked]
pub fn stage_typecheck<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    // Get the module with inlined constants from the previous stage
    let module = stage_const_inline(db, source);

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

/// Stage 5: Lambda Lifting.
///
/// This pass transforms lambda expressions into:
/// 1. Lifted top-level functions (with captured variables as parameters)
/// 2. `closure.new` operations at the original lambda locations
///
/// This happens after type checking (to know captured variable types)
/// and before closure lowering.
#[salsa::tracked]
pub fn stage_lambda_lift<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = stage_typecheck(db, source);
    lift_lambdas(db, module)
}

/// Stage 6: Closure Lowering.
///
/// This pass transforms `func.call_indirect` operations on closures:
/// - Extracts funcref via `closure.func`
/// - Extracts env via `closure.env`
/// - Passes env as first argument to the call
///
/// This happens after lambda lifting (closures exist) and before TDNR.
#[salsa::tracked]
pub fn stage_closure_lower<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = stage_lambda_lift(db, source);
    lower_closures(db, module)
}

/// Stage 7: Type-Directed Name Resolution (TDNR).
///
/// This pass resolves UFCS method calls that couldn't be resolved during
/// initial name resolution because they required type information.
///
/// For example:
/// - `list.len()` → `List::len(list)` (based on list's type being `List(a)`)
/// - `x.map(f)` → `Type::map(x, f)` (based on x's inferred type)
#[salsa::tracked]
pub fn stage_tdnr<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = stage_closure_lower(db, source);
    resolve_tdnr(db, module)
}

/// Stage 7: Lower `case.case` to `scf.if` chains.
#[salsa::tracked]
pub fn stage_lower_case<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = stage_tdnr(db, source);
    lower_case_to_scf(db, module)
}

/// Stage 8: Dead Code Elimination (DCE).
///
/// This pass removes unreachable function definitions from the module.
/// Entry points include:
/// - Functions named "main" or "_start"
/// - Functions referenced by `wasm.export_func` (for wasm target)
///
/// This should run after all high-level transformations but before
/// target-specific lowering, to avoid emitting unused functions
/// (e.g., unused prelude functions with incomplete lowering).
#[salsa::tracked]
pub fn stage_dce<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = stage_lower_case(db, source);
    let result = eliminate_dead_functions(db, module);
    result.module
}

/// Stage 9: Lower to WebAssembly target.
///
/// This stage compiles the fully-typed, resolved TrunkIR module to WebAssembly binary.
/// It performs:
/// - Lowering mid-level IR (func, scf, arith) to wasm dialect operations
/// - Emission to WebAssembly binary format
/// - Extraction of metadata (exports, imports) for tooling
///
/// The result is a `WasmBinary` artifact containing the compiled WebAssembly bytes.
/// Returns None if compilation fails, with error message accumulated.
#[salsa::tracked]
pub fn stage_lower_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<WasmBinary<'db>> {
    let module = stage_dce(db, source);
    match compile_to_wasm(db, module) {
        Ok(binary) => Some(binary),
        Err(e) => {
            // Accumulate error as diagnostic for reporting
            Diagnostic {
                message: format!("WebAssembly compilation failed: {}", e),
                span: Span::new(0, 0), // TODO: Extract span from error
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::Lowering,
            }
            .accumulate(db);
            None
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
/// 5. TDNR (Type-Directed Name Resolution)
/// 6. Lower case expressions to `scf.if`
/// 7. Dead code elimination
/// 8. Final resolution pass (reports unresolved references)
#[salsa::tracked]
pub fn compile<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    let module = stage_dce(db, source);

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
    source: SourceCst,
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
    use salsa_test_macros::salsa_test;
    use tree_sitter::Parser;

    #[salsa::tracked]
    fn test_compile(db: &dyn salsa::Database, source: SourceCst) -> Module<'_> {
        compile(db, source)
    }

    fn source_from_str(path: &str, text: &str) -> SourceCst {
        salsa::with_attached_database(|db| {
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_tribute::LANGUAGE.into())
                .expect("Failed to set language");
            let tree = parser.parse(text, None).expect("tree");
            SourceCst::from_path(db, path, text.into(), Some(tree))
        })
        .expect("attached db")
    }

    #[salsa_test]
    fn test_full_pipeline(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn main() -> Int { 42 }");

        let module = test_compile(db, source);
        assert_eq!(module.name(db), "main");
    }

    #[salsa_test]
    fn test_compile_with_diagnostics(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn add(x: Int, y: Int) -> Int { x + y }");

        let result = compile_with_diagnostics(db, source);
        // Should compile without errors
        assert!(
            result.diagnostics.is_empty(),
            "Expected no diagnostics, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_unresolved_reference_diagnostic(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn main() -> Int { undefined_var }");

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
    }

    #[salsa_test]
    fn test_parse_error_diagnostic(db: &salsa::DatabaseImpl) {
        let source = source_from_str("test.trb", "fn identity(a)(x: a) -> a { x }");
        let result = compile_with_diagnostics(db, source);

        let has_parse_error = result.diagnostics.iter().any(|d| {
            d.severity == DiagnosticSeverity::Error
                && d.phase == CompilationPhase::Parsing
                && d.message.contains("syntax error")
        });
        assert!(
            has_parse_error,
            "Expected parse error diagnostic, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_prelude_loads(db: &salsa::DatabaseImpl) {
        let prelude = prelude_module(db);
        assert!(prelude.is_some(), "Prelude should load successfully");
    }

    #[salsa_test]
    fn test_prelude_option_type(db: &salsa::DatabaseImpl) {
        // Use Option type from prelude
        let source = source_from_str("test.trb", "fn maybe() -> Option(Int) { None }");

        let result = compile_with_diagnostics(db, source);
        // Should compile without "unresolved" errors for Option or None
        let has_option_error = result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("Option") || d.message.contains("None"));
        assert!(
            !has_option_error,
            "Option and None should be available from prelude, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_prelude_result_type(db: &salsa::DatabaseImpl) {
        // Use Result type from prelude
        let source = source_from_str("test.trb", "fn success() -> Result(Int, String) { Ok(42) }");

        let result = compile_with_diagnostics(db, source);
        // Should compile without "unresolved" errors for Result or Ok
        let has_result_error = result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("Result") || d.message.contains("Ok"));
        assert!(
            !has_result_error,
            "Result and Ok should be available from prelude, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_expression_pattern_binding(db: &salsa::DatabaseImpl) {
        // Simple case expression with identifier pattern binding
        let source = source_from_str(
            "test.trb",
            r#"
            fn test(x: Int) -> Int {
                case x {
                    y -> y
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        // Pattern binding `y` should be resolved in the case arm body
        let has_unresolved_y = result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("unresolved") && d.message.contains("y"));
        assert!(
            !has_unresolved_y,
            "Pattern binding `y` should be resolved, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_lowering_exhaustive(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            fn test(x: Int) -> Int {
                case x {
                    0 -> 1
                    _ -> 2
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        assert!(
            result.diagnostics.is_empty(),
            "Expected no diagnostics, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_case_lowering_non_exhaustive(db: &salsa::DatabaseImpl) {
        let source = source_from_str(
            "test.trb",
            r#"
            fn test(x: Int) -> Int {
                case x {
                    0 -> 1
                }
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);
        let has_non_exhaustive = result.diagnostics.iter().any(|d| {
            d.message.contains("non-exhaustive") && d.severity == DiagnosticSeverity::Error
        });
        assert!(
            has_non_exhaustive,
            "Expected non-exhaustive case diagnostic, got: {:?}",
            result.diagnostics
        );
    }

    #[salsa_test]
    fn test_tdnr_struct_field_access(db: &salsa::DatabaseImpl) {
        // Test that TDNR resolves user.name to User::name(user)
        let source = source_from_str(
            "test.trb",
            r#"
            struct User {
                name: String,
                age: Int,
            }

            fn get_name(user: User) -> String {
                user.name
            }
            "#,
        );

        let result = compile_with_diagnostics(db, source);

        // Should compile without unresolved errors - TDNR should resolve user.name
        let has_unresolved_name = result
            .diagnostics
            .iter()
            .any(|d| d.message.contains("unresolved") && d.message.contains("name"));
        assert!(
            !has_unresolved_name,
            "TDNR should resolve user.name to User::name(user), got: {:?}",
            result.diagnostics
        );
    }
}
