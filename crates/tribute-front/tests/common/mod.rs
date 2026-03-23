//! Shared test helpers for tribute-front integration tests.

use ropey::Rope;
use tree_sitter::Parser;
use tribute_front::SourceCst;
use trunk_ir::context::IrContext;
use trunk_ir::printer::print_module;

pub fn source_from_str(db: &dyn salsa::Database, path: &str, text: &str) -> SourceCst {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parser.parse(text, None).expect("Failed to parse");
    SourceCst::from_path(db, path, Rope::from_str(text), Some(tree))
}

/// Run the full AST pipeline (parse → resolve → typecheck → TDNR → IR)
/// and return the IR text.
#[salsa::tracked]
pub fn run_ast_pipeline_with_ir(db: &dyn salsa::Database, source: SourceCst) -> String {
    let parsed = tribute_front::query::parsed_ast(db, source);
    assert!(parsed.is_some(), "Should parse successfully");

    let parsed = parsed.unwrap();
    let ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();

    let env = tribute_front::resolve::build_env(db, &ast);
    let resolved = tribute_front::resolve::resolve_with_env(db, ast, env, span_map.clone());

    let checker = tribute_front::typeck::TypeChecker::new(db, span_map.clone());
    let result = checker.check_module(resolved);

    let tdnr_ast = tribute_front::tdnr::resolve_tdnr(db, result.module);

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

/// Run the full AST pipeline without returning IR text.
#[salsa::tracked]
pub fn run_ast_pipeline(db: &dyn salsa::Database, source: SourceCst) {
    let _ = run_ast_pipeline_with_ir(db, source);
}
