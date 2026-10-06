//! Additional expression coverage tests.
//!
//! These tests exercise literal types, tuple construction, boolean operators,
//! and higher-order function patterns through the full pipeline to improve
//! code coverage across astgen, typeck, and ast_to_ir.

mod common;

use self::common::{ast_pipeline_diagnostics, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::DiagnosticSeverity;
use tribute_front::SourceCst;

// ========================================================================
// Literal Expression Tests
// ========================================================================

#[salsa_test]
fn test_literal_expressions(db: &salsa::DatabaseImpl) {
    for (name, text) in [
        (
            "string_literal",
            r#"
fn greeting() -> String {
    "hello"
}
"#,
        ),
        (
            "bytes_literal",
            r#"
fn data() -> Bytes {
    b"payload"
}
"#,
        ),
        (
            "float_literal",
            r#"
fn pi() -> Float {
    3.14
}
"#,
        ),
        (
            "rune_literal",
            r#"
fn letter() -> Rune {
    ?a
}
"#,
        ),
        (
            "bool_literal_true",
            r#"
fn yes() -> Bool {
    True
}
"#,
        ),
        (
            "bool_literal_false",
            r#"
fn no() -> Bool {
    False
}
"#,
        ),
        (
            "nil_literal",
            r#"
fn nothing() -> Nil {
    Nil
}
"#,
        ),
    ] {
        let source = SourceCst::from_source_str(db, "test.trb", text);
        let ir_text = run_ast_pipeline_with_ir(db, source);
        assert_snapshot!(name, ir_text);
    }
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

#[salsa_test]
fn test_list_literals_use_shared_sequence_ops(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn empty() -> List(a) {
    []
}

fn numbers() -> List(Nat) {
    [1, 2, 3]
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert!(ir_text.contains("list.empty"));
    assert!(ir_text.contains("list.prepend"));
    assert!(!ir_text.contains("Empty"));
    assert!(!ir_text.contains("Cons"));
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_list_sequence_patterns_use_shared_observation_ops(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn observe(xs: List(Nat)) -> Nat {
    case xs {
        [] -> 0
        [only] -> only
        [first, second] -> second
        [head, ..tail] -> head
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert!(ir_text.contains("list.is_empty"));
    assert!(ir_text.contains("list.head"));
    assert!(ir_text.contains("list.tail"));
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_float_comparison_operators(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn comparisons(a: Float, b: Float) -> #(Bool, Bool, Bool, Bool, Bool, Bool) {
    #(a == b, a != b, a < b, a <= b, a > b, a >= b)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn test_string_equality_operators(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn comparisons(a: String, b: String) -> #(Bool, Bool) {
    #(a == b, a != b)
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
fn apply(f: fn(Nat) ->{e} Nat, x: Nat) ->{e} Nat {
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

// ========================================================================
// Binary Operator Operand Types
// ========================================================================

/// Binary operators require operands of matching, operator-appropriate types.
/// `+1` is Int (explicit sign), `1` is Nat (no sign).
#[salsa_test]
fn test_binop_operand_types(db: &salsa::DatabaseImpl) {
    // (operator expression, result type, accepted)
    for (expr, result_ty, accepted) in [
        ("+1 + 2", "Int", false),
        ("+1 < 2.0", "Bool", false),
        ("+1 && +2", "Bool", false),
        ("+1 + +2", "Int", true),
        ("1 + 2", "Nat", true),
        ("1.5 + 2.5", "Float", true),
        ("True && False", "Bool", true),
    ] {
        let text =
            format!("fn compute() ->{{}} {result_ty} {{ {expr} }}\nfn main() -> Nil {{ }}\n");
        let source = SourceCst::from_source_str(db, "binop.trb", &text);
        let diagnostics = ast_pipeline_diagnostics(db, source);
        if accepted {
            assert!(
                diagnostics.is_empty(),
                "expected no diagnostics for `{expr}`, got {diagnostics:?}"
            );
        } else {
            assert!(
                diagnostics
                    .iter()
                    .any(|d| d.inner.severity == DiagnosticSeverity::Error),
                "expected a type error for `{expr}`, got {diagnostics:?}"
            );
        }
    }
}
