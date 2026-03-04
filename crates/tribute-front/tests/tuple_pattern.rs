//! Tests for tuple pattern type inference.
//!
//! These tests verify correct UniVar resolution in tuple patterns.

mod common;

use salsa_test_macros::salsa_test;

/// Test basic tuple pattern matching in case expression.
/// This should infer types correctly without UniVar leakage.
#[salsa_test]
fn test_tuple_pattern_basic(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    common::run_ast_pipeline(db, source);
}

/// Test tuple pattern with nested structure.
#[salsa_test]
fn test_tuple_pattern_nested(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    common::run_ast_pipeline(db, source);
}

/// Test tuple pattern in generic function context.
#[salsa_test]
fn test_tuple_pattern_generic(db: &salsa::DatabaseImpl) {
    let source = common::source_from_str(
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

    common::run_ast_pipeline(db, source);
}
