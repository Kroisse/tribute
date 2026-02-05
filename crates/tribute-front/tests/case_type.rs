//! Tests for case expression type checking.
//!
//! These tests verify that case expressions properly unify
//! scrutinee types with pattern types and arm body types.

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

    // Type check - this is where case expression types are unified
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

// ========================================================================
// Basic Case Expression Tests
// ========================================================================

/// Test case expression with Nat literals.
/// Pattern type (Nat) should unify with scrutinee type (Nat).
#[salsa_test]
fn test_case_nat_literal(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn classify(x: Nat) -> Nat {
    case x {
        0 -> 100
        1 -> 200
        _ -> 300
    }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test case expression with Int literals (negative numbers).
/// Pattern type (Int) should unify with scrutinee type (Int).
#[salsa_test]
fn test_case_int_literal(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn sign(x: Int) -> Int {
    case x {
        -1 -> -100
        0 -> 0
        _ -> 100
    }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test case expression with Bool patterns.
#[salsa_test]
fn test_case_bool_literal(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn invert(x: Bool) -> Bool {
    case x {
        True -> False
        False -> True
    }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test case expression with enum variant patterns.
#[salsa_test]
fn test_case_enum_variant(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
enum Option(a) {
    Some(a),
    None,
}

fn unwrap_or(opt: Option(Nat), default: Nat) -> Nat {
    case opt {
        Some(x) -> x
        None -> default
    }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test case expression result type unification.
/// All arm body types should unify with the case expression's result type.
#[salsa_test]
fn test_case_result_type_unification(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn to_nat(b: Bool) -> Nat {
    case b {
        True -> 1
        False -> 0
    }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}

/// Test nested case expressions.
#[salsa_test]
fn test_case_nested(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn nested(x: Nat, y: Bool) -> Nat {
    case x {
        0 -> case y {
            True -> 10
            False -> 20
        }
        _ -> 30
    }
}
"#,
    );

    let ir_module = run_ast_pipeline_with_ir(db, source);
    assert_debug_snapshot!(ir_module);
}
