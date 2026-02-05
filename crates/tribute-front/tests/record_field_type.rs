//! Tests for record field type checking.
//!
//! These tests verify that record construction properly validates
//! field expression types against declared struct field types.

use insta::assert_debug_snapshot;
use ropey::Rope;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
use trunk_ir::dialect::core::Module;

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
    let _ = run_ast_pipeline_with_ir(db, source);
}

/// Helper tracked function to run the AST pipeline and return the IR module.
#[salsa::tracked]
fn run_ast_pipeline_with_ir<'db>(db: &'db dyn salsa::Database, source: SourceCst) -> Module<'db> {
    // Parse and process through AST pipeline
    let parsed = tribute_front::query::parsed_ast(db, source);
    assert!(parsed.is_some(), "Should parse successfully");

    let parsed = parsed.unwrap();
    let ast = parsed.module(db).clone();
    let span_map = parsed.span_map(db).clone();

    // Build env and resolve
    let env = tribute_front::resolve::build_env(db, &ast);
    let resolved = tribute_front::resolve::resolve_with_env(db, ast, env, span_map.clone());

    // Type check - this is where field type constraints are applied
    let checker = tribute_front::typeck::TypeChecker::new(db, span_map.clone());
    let (typed_ast, function_types, node_types) = checker.check_module(resolved);

    // TDNR
    let tdnr_ast = tribute_front::tdnr::resolve_tdnr(db, typed_ast);

    // Lower to IR
    let function_types_map: std::collections::HashMap<_, _> = function_types.into_iter().collect();
    let node_types_map: std::collections::HashMap<_, _> = node_types.into_iter().collect();
    tribute_front::ast_to_ir::lower_ast_to_ir(
        db,
        tdnr_ast,
        span_map,
        source.uri(db).as_str(),
        function_types_map,
        node_types_map,
    )
}

/// Test basic record construction with correct field types.
/// This should compile successfully.
#[salsa_test]
fn test_record_field_type_correct(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn make_point() -> Point {
    Point { x: 10, y: 20 }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with multiple field types.
#[salsa_test]
fn test_record_mixed_field_types(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Person { name: String, age: Int, active: Bool }

fn make_person() -> Person {
    Person { name: "Alice", age: 30, active: True }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with spread operator.
/// The spread expression should be constrained to the struct type.
#[salsa_test]
fn test_record_spread_same_type(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn update_x(p: Point) -> Point {
    Point { x: 100, ..p }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record construction with only spread (no explicit fields).
#[salsa_test]
fn test_record_spread_only(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Config { debug: Bool, verbose: Bool }

fn copy_config(c: Config) -> Config {
    Config { ..c }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record with generic type parameter.
#[salsa_test]
fn test_record_generic_type(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn make_pair() -> Pair(Int, Bool) {
    Pair { first: 42, second: True }
}
"#,
    );

    run_ast_pipeline(db, source);
}

/// Test record field type inference in let binding.
#[salsa_test]
fn test_record_field_type_inference(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Int {
    let p = Point { x: 1, y: 2 };
    p.x
}
"#,
    );

    run_ast_pipeline(db, source);
}

// ========================================================================
// Snapshot Tests
// ========================================================================

/// Snapshot test for basic record construction IR.
#[salsa_test]
fn test_snapshot_record_construction(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn make_point() -> Point {
    Point { x: 10, y: 20 }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Snapshot test for record with spread operator.
#[salsa_test]
fn test_snapshot_record_spread(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn update_x(p: Point) -> Point {
    Point { x: 100, ..p }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Snapshot test for generic record construction.
#[salsa_test]
fn test_snapshot_record_generic(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn make_pair() -> Pair(Int, Bool) {
    Pair { first: 42, second: True }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

// ========================================================================
// Forward Reference Tests
// ========================================================================

/// Test record construction where the function using the record appears
/// before the struct definition (forward reference).
///
/// This tests that prescan_struct_fields correctly registers field orders
/// before lowering, regardless of declaration order in the source.
#[salsa_test]
fn test_record_forward_reference(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn make_point() -> Point {
    Point { x: 1, y: 2 }
}

struct Point { x: Int, y: Int }
"#,
    );

    // Should compile without ICE, emitting adt.struct_new
    run_ast_pipeline(db, source);
}
