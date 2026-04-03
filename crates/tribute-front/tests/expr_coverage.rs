//! Additional expression coverage tests.
//!
//! These tests exercise literal types, tuple construction, boolean operators,
//! and higher-order function patterns through the full pipeline to improve
//! code coverage across astgen, typeck, and ast_to_ir.

mod common;

use self::common::run_ast_pipeline_with_ir;
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

// ========================================================================
// Literal Expression Tests
// ========================================================================

#[salsa_test]
fn test_string_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn greeting() -> String {
    "hello"
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_bytes_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn data() -> Bytes {
    b"payload"
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_float_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pi() -> Float {
    3.14
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_rune_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn letter() -> Rune {
    ?a
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_bool_literal_true(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn yes() -> Bool {
    True
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_bool_literal_false(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn no() -> Bool {
    False
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_nil_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn nothing() -> Nil {
    Nil
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Compound Expression Tests
// ========================================================================

#[salsa_test]
fn test_tuple_construction(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn pair() -> #(Nat, Bool) {
    #(42, True)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_boolean_operators(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn logic(a: Bool, b: Bool) -> Bool {
    a || b && False
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Higher-Order Function Tests
// ========================================================================

#[salsa_test]
fn test_lambda_as_argument(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn apply(f: fn(Nat) -> Nat, x: Nat) -> Nat {
    f(x)
}

fn test_lambda() -> Nat {
    apply(fn(x) { x }, 1)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_function_reference_as_value(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn id(x: Nat) -> Nat {
    x
}

fn test_ref() -> Nat {
    let f = id
    f(1)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
