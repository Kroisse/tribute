//! Tests for block scope handling in type checking.
//!
//! These tests verify that let-pattern bindings inside blocks
//! do not escape to outer scopes.

mod common;

use insta::assert_snapshot;
use salsa_test_macros::salsa_test;

// ========================================================================
// Block Scope Tests - Success Cases
// ========================================================================

/// Test that let bindings inside a block are accessible within the block.
#[salsa_test]
fn test_block_let_binding_accessible_inside(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    let ir_text = common::run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test nested blocks with let bindings at different levels.
#[salsa_test]
fn test_nested_blocks_separate_scopes(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    let ir_text = common::run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test that outer scope variables are accessible in inner blocks.
#[salsa_test]
fn test_outer_scope_accessible_in_inner_block(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    let ir_text = common::run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test multiple sequential blocks with independent scopes.
#[salsa_test]
fn test_sequential_blocks_independent_scopes(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    let ir_text = common::run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test block with case expression containing let bindings in patterns.
#[salsa_test]
fn test_block_with_case_pattern_binding(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    let ir_text = common::run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
