//! Tests for record field type checking.
//!
//! These tests verify that record construction properly validates
//! field expression types against declared struct field types.

mod common;

use self::common::{ast_pipeline_diagnostics, run_ast_pipeline, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::SourceCst;
use trunk_ir::Span;

fn diagnostics(db: &dyn salsa::Database, text: &str) -> Vec<Diagnostic> {
    let source = SourceCst::from_source_str(db, "record_shape.trb", text);
    ast_pipeline_diagnostics(db, source)
}

fn shape_error(text: &str, record: &str, occurrence: usize, message: &str) -> Diagnostic {
    let start = text.match_indices(record).nth(occurrence).unwrap().0;
    Diagnostic::new(
        message,
        Span::new(start, start + record.len()),
        DiagnosticSeverity::Error,
        CompilationPhase::TypeChecking,
    )
}

#[salsa_test]
fn record_shape_diagnostic_matrix(db: &salsa::DatabaseImpl) {
    for (fields, messages) in [
        (
            "x: +1, y: +2, z: +3",
            vec!["unknown field `z` for struct `Point`"],
        ),
        ("x: +1, x: +2, y: +3", vec!["duplicate field `x`"]),
        (
            "x: +1, x: +2, x: +3, y: +4",
            vec!["duplicate field `x`", "duplicate field `x`"],
        ),
        (
            "x: +1, y: +2, z: +3, z: +4",
            vec![
                "unknown field `z` for struct `Point`",
                "unknown field `z` for struct `Point`",
            ],
        ),
        ("x: +1", vec!["missing field: y"]),
        ("", vec!["missing field: x"]),
        (
            "x: +1, x: +2, z: +3",
            vec![
                "duplicate field `x`",
                "unknown field `z` for struct `Point`",
                "missing field: y",
            ],
        ),
        ("x: +1, ..base", vec![]),
        ("x: +1, x: +2, ..base", vec!["duplicate field `x`"]),
        (
            "z: +1, z: +2, ..base",
            vec![
                "unknown field `z` for struct `Point`",
                "unknown field `z` for struct `Point`",
            ],
        ),
    ] {
        let record = format!("Point {{ {fields} }}");
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nfn make(base: Point) -> Point {{ {record} }}"
        );
        let expected: Vec<_> = messages
            .into_iter()
            .map(|message| shape_error(&text, &record, 0, message))
            .collect();
        assert_eq!(diagnostics(db, &text), expected, "fields: {fields}");
    }
}

#[salsa_test]
fn record_shape_reports_the_first_missing_field_in_declaration_order(db: &salsa::DatabaseImpl) {
    let record = "Ordered { }";
    let text =
        format!("struct Ordered {{ z: Int, a: Int, m: Int }}\nfn make() -> Ordered {{ {record} }}");
    assert_eq!(
        diagnostics(db, &text),
        vec![shape_error(&text, record, 0, "missing field: z")]
    );
}

#[salsa_test]
fn record_shape_empty_struct_has_registered_fields(db: &salsa::DatabaseImpl) {
    let valid = "Empty { }";
    let invalid = "Empty { extra: +1 }";
    let text = format!(
        "struct Empty {{}}\nfn valid() -> Empty {{ {valid} }}\nfn invalid() -> Empty {{ {invalid} }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![shape_error(
            &text,
            invalid,
            0,
            "unknown field `extra` for struct `Empty`"
        )]
    );
}

#[salsa_test]
fn record_shape_argument_revisits_preserve_distinct_occurrences(db: &salsa::DatabaseImpl) {
    let record = "Point { x: +1, y: +2, z: +3 }";
    for occurrences in [1, 2] {
        let calls = format!("take({record})\n").repeat(occurrences);
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nfn take(p: Point) {{}}\nfn run() {{ {calls} }}"
        );
        let expected: Vec<_> = (0..occurrences)
            .map(|occurrence| {
                shape_error(
                    &text,
                    record,
                    occurrence,
                    "unknown field `z` for struct `Point`",
                )
            })
            .collect();
        assert_eq!(diagnostics(db, &text), expected);
    }
}

#[salsa_test]
fn record_shape_nested_errors_survive_argument_revisits(db: &salsa::DatabaseImpl) {
    let inner = "Point { x: +1 }";
    let outer = format!("Wrapper {{ point: {inner}, extra: +2 }}");
    let text = format!(
        "struct Point {{ x: Int, y: Int }}\nstruct Wrapper {{ point: Point }}\n\
         fn take(value: Wrapper) {{}}\nfn run() {{ take({outer}) }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![
            shape_error(
                &text,
                &outer,
                0,
                "unknown field `extra` for struct `Wrapper`"
            ),
            shape_error(&text, inner, 0, "missing field: y"),
        ]
    );
}

#[salsa_test]
fn record_shape_unknown_and_duplicate_fields_still_check_nested_rhs(db: &salsa::DatabaseImpl) {
    for (field, message) in [
        ("extra", "unknown field `extra` for struct `Wrapper`"),
        ("point", "duplicate field `point`"),
    ] {
        let inner = "Point { x: +1 }";
        let outer = format!("Wrapper {{ point: valid, {field}: {inner} }}");
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nstruct Wrapper {{ point: Point }}\n\
             fn make(valid: Point) -> Wrapper {{ {outer} }}"
        );
        assert_eq!(
            diagnostics(db, &text),
            vec![
                shape_error(&text, &outer, 0, message),
                shape_error(&text, inner, 0, "missing field: y"),
            ]
        );
    }
}

#[salsa_test]
fn record_shape_errors_preserve_rhs_type_errors(db: &salsa::DatabaseImpl) {
    for (field, rhs, shape_message) in [
        (
            "extra",
            "need_int(True)",
            "unknown field `extra` for struct `Point`",
        ),
        ("x", "True", "duplicate field `x`"),
    ] {
        let record = format!("Point {{ x: +1, y: +2, {field}: {rhs} }}");
        let text = format!(
            "struct Point {{ x: Int, y: Int }}\nfn need_int(value: Int) -> Int {{ value }}\n\
             fn make() -> Point {{ {record} }}"
        );
        let errors = diagnostics(db, &text);
        assert_eq!(errors.len(), 2, "{errors:#?}");
        assert_eq!(errors[0], shape_error(&text, &record, 0, shape_message));
        assert_eq!(errors[1].phase, CompilationPhase::TypeChecking);
        assert_eq!(errors[1].inner.severity, DiagnosticSeverity::Error);
        assert!(
            errors[1]
                .inner
                .message
                .contains("expected `Int`, found `Bool`"),
            "{errors:#?}"
        );
    }
}

#[salsa_test]
fn record_shape_unknown_field_preserves_nominal_spread_error(db: &salsa::DatabaseImpl) {
    let record = "Point { z: +1, ..base }";
    let text = format!(
        "struct Point {{ x: Int, y: Int }}\nstruct Other {{ x: Int, y: Int }}\n\
         fn make(base: Other) -> Point {{ {record} }}"
    );
    let errors = diagnostics(db, &text);
    assert_eq!(errors.len(), 2, "{errors:#?}");
    assert_eq!(
        errors[0],
        shape_error(&text, record, 0, "unknown field `z` for struct `Point`")
    );
    assert_eq!(errors[1].phase, CompilationPhase::TypeChecking);
    assert!(
        errors[1]
            .inner
            .message
            .contains("expected `Point`, found `Other`"),
        "{errors:#?}"
    );
}

#[salsa_test]
fn record_shape_uses_generic_declaration_fields(db: &salsa::DatabaseImpl) {
    let record = "Pair { first: +1, extra: True }";
    let text = format!(
        "struct Pair(a, b) {{ first: a, second: b }}\n\
         fn make() -> Pair(Int, Bool) {{ {record} }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![
            shape_error(&text, record, 0, "unknown field `extra` for struct `Pair`"),
            shape_error(&text, record, 0, "missing field: second"),
        ]
    );
}

#[salsa_test]
fn record_shape_uses_qualified_declaration_identity(db: &salsa::DatabaseImpl) {
    let a_record = "A::Point { y: +1 }";
    let b_record = "B::Point { x: +2 }";
    let text = format!(
        "pub mod A {{ pub struct Point {{ x: Int }} }}\n\
         pub mod B {{ pub struct Point {{ y: Int }} }}\n\
         fn make_a() -> A::Point {{ {a_record} }}\n\
         fn make_b() -> B::Point {{ {b_record} }}"
    );
    assert_eq!(
        diagnostics(db, &text),
        vec![
            shape_error(
                &text,
                a_record,
                0,
                "unknown field `y` for struct `A::Point`"
            ),
            shape_error(&text, a_record, 0, "missing field: x"),
            shape_error(
                &text,
                b_record,
                0,
                "unknown field `x` for struct `B::Point`"
            ),
            shape_error(&text, b_record, 0, "missing field: y"),
        ]
    );
}

#[salsa_test]
fn valid_record_shapes_keep_field_and_callable_inference(db: &salsa::DatabaseImpl) {
    let text = r#"
struct Point { x: Int, y: Int }
struct Pair(a, b) { first: a, second: b }
struct Callback { run: fn(Int) -> Int }
pub mod A { pub struct Point { x: Int } }
pub mod B { pub struct Point { y: Int } }

fn complete() -> Point { Point { y: +2, x: +1 } }
fn spread_only(base: Point) -> Point { Point { ..base } }
fn partial_override(base: Point) -> Point { Point { y: +3, ..base } }
fn full_override(base: Point) -> Point { Point { x: +3, y: +4, ..base } }
fn generic() -> Pair(Int, Bool) { Pair { first: +1, second: True } }
fn callable() -> Callback { Callback { run: fn(value) { value } } }
fn make_a() -> A::Point { A::Point { x: +1 } }
fn make_b() -> B::Point { B::Point { y: +2 } }
"#;
    assert_eq!(diagnostics(db, text), vec![]);
}

/// Test basic record construction with correct field types.
/// This should compile successfully.
#[salsa_test]
fn test_record_field_type_correct(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn test() -> Int {
    let p = Point { x: 1, y: 2 }
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
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn make_point() -> Point {
    Point { x: 10, y: 20 }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with spread operator.
#[salsa_test]
fn test_snapshot_record_spread(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn update_x(p: Point) -> Point {
    Point { x: 100, ..p }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for generic record construction.
#[salsa_test]
fn test_snapshot_record_generic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Pair(a, b) { first: a, second: b }

fn make_pair() -> Pair(Int, Bool) {
    Pair { first: 42, second: True }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with spread only (no explicit fields).
/// All fields should be extracted via `adt.struct_get` from the base.
#[salsa_test]
fn test_snapshot_record_spread_only(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Config { debug: Bool, verbose: Bool }

fn copy_config(c: Config) -> Config {
    Config { ..c }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Snapshot test for record with all fields explicit plus spread.
/// Explicit fields should take priority over spread values.
#[salsa_test]
fn test_snapshot_record_spread_all_fields_explicit(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn replace_all(p: Point) -> Point {
    Point { x: 1, y: 2, ..p }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test record spread with a function call as the spread expression.
#[salsa_test]
fn test_record_spread_complex_expr(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
struct Point { x: Int, y: Int }

fn origin() -> Point {
    Point { x: 0, y: 0 }
}

fn shift_x() -> Point {
    Point { x: 10, ..origin() }
}
"#,
    );

    run_ast_pipeline(db, source);
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
    let source = SourceCst::from_source_str(
        db,
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
