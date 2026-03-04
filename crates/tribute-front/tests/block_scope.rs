//! Tests for block scope handling in type checking.
//!
//! These tests verify that let-pattern bindings inside blocks
//! do not escape to outer scopes.

use insta::assert_snapshot;
use ropey::Rope;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::printer::print_module;

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

/// Helper tracked function to run the AST pipeline and return the IR text.
#[salsa::tracked]
fn run_ast_pipeline_with_ir(db: &dyn salsa::Database, source: SourceCst) -> String {
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
    let arena_module = tribute_front::ast_to_ir::lower_ast_to_ir(
        db,
        &mut ir,
        tdnr_ast,
        span_map,
        source.uri(db).as_str(),
        function_types_map,
        node_types_map,
    );
    print_module(&ir, arena_module.op())
}

// ========================================================================
// Block Scope Tests - Success Cases
// ========================================================================

/// Test that let bindings inside a block are accessible within the block.
#[salsa_test]
fn test_block_let_binding_accessible_inside(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn example() -> Nat {
    {
        let x = 42;
        x
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test nested blocks with let bindings at different levels.
#[salsa_test]
fn test_nested_blocks_separate_scopes(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn example() -> Nat {
    let outer = 1;
    {
        let inner = 2;
        inner
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test that outer scope variables are accessible in inner blocks.
#[salsa_test]
fn test_outer_scope_accessible_in_inner_block(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn example() -> Nat {
    let outer = 10;
    {
        let inner = outer;
        inner
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test multiple sequential blocks with independent scopes.
#[salsa_test]
fn test_sequential_blocks_independent_scopes(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
fn example() -> Nat {
    let a = {
        let x = 1;
        x
    };
    let b = {
        let x = 2;
        x
    };
    a
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test block with case expression containing let bindings in patterns.
#[salsa_test]
fn test_block_with_case_pattern_binding(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
enum Option(a) {
    Some(a),
    None,
}

fn example(opt: Option(Nat)) -> Nat {
    {
        case opt {
            Some(x) -> x
            None -> 0
        }
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
