//! Tests for tuple pattern type inference.
//!
//! These tests verify correct UniVar resolution in tuple patterns.

use ropey::Rope;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn source_from_str(path: &str, text: &str) -> SourceCst {
    use tree_sitter::Parser;
    salsa::with_attached_database(|db| {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let text: Rope = text.into();
        let tree = parser.parse(text.to_string().as_str(), None);
        SourceCst::new(
            db,
            fluent_uri::Uri::parse_from(format!("test:///{}", path)).unwrap(),
            text,
            tree,
        )
    })
    .expect("attached db")
}

/// Helper tracked function to run the AST pipeline.
#[salsa::tracked]
fn run_ast_pipeline<'db>(db: &'db dyn salsa::Database, source: SourceCst) {
    // Parse and process through AST pipeline
    let parsed = tribute_front::query::parsed_ast(db, source);
    assert!(parsed.is_some(), "Should parse successfully");

    let parsed = parsed.unwrap();
    let ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();

    // Build env and resolve
    let env = tribute_front::resolve::build_env(db, &ast);
    let resolved = tribute_front::resolve::resolve_with_env(db, ast, env, span_map.clone());

    // Type check - this is where UniVar issues manifest
    let checker = tribute_front::typeck::TypeChecker::new(db);
    let (typed_ast, function_types) = checker.check_module(resolved);

    // TDNR
    let tdnr_ast = tribute_front::tdnr::resolve_tdnr(db, typed_ast);

    // Lower to IR - this will panic if UniVars survive
    let function_types_map: std::collections::HashMap<_, _> = function_types.into_iter().collect();
    let _module = tribute_front::ast_to_ir::lower_ast_to_ir(
        db,
        tdnr_ast,
        span_map,
        source.uri(db).as_str(),
        function_types_map,
    );
}

/// Test basic tuple pattern matching in case expression.
/// This should infer types correctly without UniVar leakage.
#[salsa_test]
fn test_tuple_pattern_basic(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn test(x: Bool, y: Bool) -> Int {
    case #(x, y) {
        #(True, True) -> 1
        #(True, False) -> 2
        #(False, _) -> 3
    }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test tuple pattern with nested structure.
#[salsa_test]
fn test_tuple_pattern_nested(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn test(a: Bool, b: Bool, c: Bool) -> Int {
    case #(a, #(b, c)) {
        #(True, #(True, True)) -> 1
        #(True, #(_, False)) -> 2
        #(False, _) -> 3
    }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test tuple pattern in generic function context.
#[salsa_test]
fn test_tuple_pattern_generic(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn first(pair: #(a, b)) -> a {
    case pair {
        #(x, _) -> x
    }
}

fn test() -> Int {
    first(#(42, True))
}
"#,
    );

    run_ast_pipeline(db, source);
}
