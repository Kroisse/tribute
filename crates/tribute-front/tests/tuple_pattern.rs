//! Tests for tuple pattern type inference and IR lowering.
//!
//! These tests verify correct UniVar resolution in tuple patterns,
//! and that tuple destructuring in `let` bindings produces correct IR.

mod common;

use self::common::{run_ast_pipeline, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// Test basic tuple pattern matching in case expression.
/// This should infer types correctly without UniVar leakage.
#[salsa_test]
fn test_tuple_pattern_basic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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

// ========================================================================
// Tuple Let Destructuring Tests
// ========================================================================

/// Test basic tuple let destructuring lowers to adt.struct_get with correct types.
#[salsa_test]
fn test_tuple_let_destructure_basic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test() -> Nat {
    let t = #(1, 2)
    let #(a, b) = t
    a + b
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test tuple let destructuring with nested tuples.
#[salsa_test]
fn test_tuple_let_destructure_nested(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test() -> Nat {
    let t = #(1, #(2, 3))
    let #(a, #(b, c)) = t
    a + b + c
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test tuple let destructuring with wildcard elements.
#[salsa_test]
fn test_tuple_let_destructure_wildcard(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn test() -> Nat {
    let t = #(1, 2, 3)
    let #(a, _, c) = t
    a + c
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test tuple let destructuring used in function arguments.
#[salsa_test]
#[ignore = "operator TDNR fails with pattern bindings (#617)"]
fn test_tuple_let_destructure_from_function(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn make_pair(x: Nat, y: Nat) -> #(Nat, Nat) {
    #(x, y)
}

fn test() -> Nat {
    let #(a, b) = make_pair(10, 20)
    a + b
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
