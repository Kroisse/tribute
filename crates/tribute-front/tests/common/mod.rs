//\! Shared test helpers for tribute-front integration tests.

use tribute_core::diagnostic::Diagnostic;
use tribute_front::SourceCst;
use trunk_ir::context::IrContext;
use trunk_ir::printer::print_module;

/// Prelude source, loaded at compile time relative to this file.
const PRELUDE_SOURCE: &str = include_str!("../../../../lib/std/prelude.trb");

/// Result of loading and type-checking the prelude.
struct PreludeData<'db> {
    exports: tribute_front::typeck::PreludeExports<'db>,
    env: tribute_front::resolve::ModuleEnv<'db>,
    typed_module: tribute_front::ast::Module<tribute_front::ast::TypedRef<'db>>,
}

/// Load and type-check the prelude so tests can use arithmetic operators.
///
/// Operators like `+`, `*`, `==` are declared as `extern "intrinsic" fn` in
/// the prelude for Int, Nat, and Float. Without the prelude, TDNR cannot
/// resolve them and the IR lowerer panics on unresolved MethodCall nodes.
fn load_prelude(db: &dyn salsa::Database) -> Option<PreludeData<'_>> {
    let prelude_cst = SourceCst::from_source_str(db, "prelude:///std/prelude", PRELUDE_SOURCE);

    let parsed = tribute_front::query::parsed_ast_with_module_path(
        db,
        prelude_cst,
        trunk_ir::Symbol::new("prelude"),
    )?;

    let prelude_ast = parsed.module(db).clone();
    let prelude_span_map = parsed.span_map(db).clone();

    // Build env for name resolution merging
    let env = tribute_front::resolve::build_env(db, &prelude_ast);

    // Resolve prelude with its own env
    let resolved_prelude = tribute_front::resolve::resolve_with_env(
        db,
        prelude_ast.clone(),
        env.clone(),
        prelude_span_map.clone(),
    );

    // Type-check for PreludeExports (for type injection)
    let checker = tribute_front::typeck::TypeChecker::new(db, prelude_span_map.clone());
    let exports = checker.check_module_for_prelude(resolved_prelude.clone());

    // Type-check to get typed module (for TDNR imports)
    let checker2 = tribute_front::typeck::TypeChecker::new(db, prelude_span_map);
    let result2 = checker2.check_module(resolved_prelude);

    Some(PreludeData {
        exports,
        env,
        typed_module: result2.module,
    })
}

/// Inner tracked function that runs the full pipeline.
///
/// Diagnostics emitted during the pipeline are accumulated as Salsa
/// accumulators and can be collected by the caller.
#[salsa::tracked]
fn run_ast_pipeline_inner(db: &dyn salsa::Database, source: SourceCst) -> String {
    let parsed = tribute_front::query::parsed_ast(db, source);
    assert!(parsed.is_some(), "Should parse successfully");

    let parsed = parsed.unwrap();
    let ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();

    // Load prelude for operator declarations (Int::(+) etc.)
    let prelude = load_prelude(db);

    // Build env, merging prelude bindings so operator names resolve
    let mut env = tribute_front::resolve::build_env(db, &ast);
    if let Some(ref p) = prelude {
        env.merge(&p.env);
    }
    let resolved = tribute_front::resolve::resolve_with_env(db, ast, env, span_map.clone());

    // Type-check with prelude types injected
    let mut checker = tribute_front::typeck::TypeChecker::new(db, span_map.clone());
    if let Some(ref p) = prelude {
        checker.inject_prelude(&p.exports);
    }
    let result = checker.check_module(resolved);

    // TDNR with prelude module as import source for method resolution
    let prelude_modules: Vec<_> = prelude.iter().map(|p| &p.typed_module).collect();
    let tdnr_ast =
        tribute_front::tdnr::resolve_tdnr(db, result.module, prelude_modules.iter().copied());

    let function_types_map: std::collections::HashMap<_, _> =
        result.function_types.into_iter().collect();
    let node_types_map: std::collections::HashMap<_, _> = result.node_types.into_iter().collect();
    let mut ir = IrContext::new();
    let module = tribute_front::ast_to_ir::lower_ast_to_ir(
        db,
        &mut ir,
        tdnr_ast,
        span_map,
        source.uri(db).as_str(),
        function_types_map,
        node_types_map,
    );
    print_module(&ir, module.op())
}

/// Run the full AST pipeline (parse -> resolve -> typecheck -> TDNR -> IR)
/// with the prelude loaded, and return the IR text.
///
/// Panics if any parse error diagnostics were emitted for the test source.
///
/// Uses `parsed_ast` to check only the test source for parse errors,
/// avoiding false positives from prelude diagnostics.
pub fn run_ast_pipeline_with_ir(db: &dyn salsa::Database, source: SourceCst) -> String {
    // Check for parse errors in the test source (not prelude)
    let parse_diagnostics = tribute_front::query::parsed_ast::accumulated::<Diagnostic>(db, source);
    let parse_errors: Vec<_> = parse_diagnostics
        .iter()
        .filter(|d| {
            d.inner.severity == tribute_core::diagnostic::DiagnosticSeverity::Error
                && matches!(
                    d.phase,
                    tribute_core::diagnostic::CompilationPhase::Parsing
                        | tribute_core::diagnostic::CompilationPhase::AstGeneration
                )
        })
        .collect();
    assert!(
        parse_errors.is_empty(),
        "Expected no parse errors, but found {}:\n{}",
        parse_errors.len(),
        parse_errors
            .iter()
            .map(|d| {
                format!(
                    "  - [{}] {} (span: {}..{})",
                    d.phase, d.inner.message, d.inner.span.start, d.inner.span.end
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    );

    run_ast_pipeline_inner(db, source)
}

/// Run the full AST pipeline without returning IR text.
///
/// Panics if any error diagnostics were emitted during the pipeline.
pub fn run_ast_pipeline(db: &dyn salsa::Database, source: SourceCst) {
    let _ = run_ast_pipeline_with_ir(db, source);
}
