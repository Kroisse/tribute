//! Tests for brace-form constructor patterns (`Name { field: pattern, .. }`),
//! which match fields by name, and for constructor pattern field counts.

mod common;

use self::common::ast_pipeline_diagnostics;
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::DiagnosticSeverity;
use tribute_front::SourceCst;

/// Messages of `severity` from checking `text`.
fn messages(db: &dyn salsa::Database, text: &str, severity: DiagnosticSeverity) -> Vec<String> {
    let source = SourceCst::from_source_str(db, "record_pattern.trb", text);
    ast_pipeline_diagnostics(db, source)
        .into_iter()
        .filter(|diagnostic| diagnostic.inner.severity == severity)
        .map(|diagnostic| diagnostic.inner.message)
        .collect()
}

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    messages(db, text, DiagnosticSeverity::Error)
}

const POINT: &str = "struct Point { x: Nat, y: Bool }\n";

#[salsa_test]
fn fields_are_matched_by_name_in_any_order(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{POINT}{}",
        r#"
fn pick(p: Point) -> Nat {
    case p {
        Point { y: flag, x: value } -> case flag {
            True -> value
            False -> 0
        }
    }
}

fn first(p: Point) -> Nat {
    let Point { y: _, x } = p
    x
}

fn flag(p: Point) -> Bool {
    case p {
        Point { y: True, .. } -> True
        Point { y, .. } -> y
    }
}
"#
    );
    let errors = errors(db, &source);
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn named_variant_fields_are_matched_by_name(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
enum Shape {
    Circle { radius: Nat },
    Rect { width: Nat, tall: Bool },
}

fn area(shape: Shape) -> Nat {
    case shape {
        Circle { radius } -> radius
        Rect { tall: True, width } -> width
        Rect { width: _, .. } -> 0
    }
}
"#,
    );
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn generic_struct_fields_take_the_instance_types(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
struct Pair(a, b) { left: a, right: b }

fn swap(pair: Pair(Nat, Bool)) -> Bool {
    let Pair { right, left: _ } = pair
    right
}

fn left(pair: Pair(Nat, Bool)) -> Nat {
    case pair {
        Pair { right: _, left } -> left
    }
}
"#,
    );
    assert!(errors.is_empty(), "{errors:?}");
}

#[salsa_test]
fn field_names_must_fit_the_declaration(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{POINT}{}",
        r#"
fn unknown(p: Point) -> Nat {
    case p {
        Point { x, z, .. } -> x
    }
}

fn duplicate(p: Point) -> Nat {
    case p {
        Point { x, x: _, .. } -> x
    }
}

fn missing(p: Point) -> Nat {
    let Point { x } = p
    x
}
"#
    );
    assert_eq!(
        errors(db, &source),
        [
            "unknown field `z` for struct `Point`",
            "duplicate field `x`",
            "missing field: y",
        ]
    );
}

#[salsa_test]
fn brace_form_needs_named_fields(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
enum Shape {
    Circle { radius: Nat },
    Dot(Nat),
}

fn size(shape: Shape) -> Nat {
    case shape {
        Circle { size, .. } -> size
        Dot { value } -> value
    }
}
"#,
    );
    assert_eq!(
        errors,
        [
            "unknown field `size` for variant `Circle`",
            "`Dot` has positional fields; match it with `Dot(...)`",
        ]
    );
}

#[salsa_test]
fn positional_patterns_need_one_pattern_per_field(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
enum Shape {
    Dot(Nat),
    Line(Nat, Nat),
    Empty,
}

fn size(shape: Shape) -> Nat {
    case shape {
        Dot(a, b) -> a
        Line(a) -> a
        Empty(a) -> a
        Dot -> 0
        _ -> 0
    }
}
"#,
    );
    assert_eq!(
        errors,
        [
            "constructor `Dot` expects 1 field, but the pattern has 2",
            "constructor `Line` expects 2 fields, but the pattern has 1",
            "constructor `Empty` expects 0 fields, but the pattern has 1",
            "constructor `Dot` expects 1 field, but the pattern has 0",
        ]
    );
}

#[salsa_test]
fn type_names_are_not_constructors(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
enum Shape {
    Dot(Nat),
}

fn size(shape: Shape) -> Nat {
    case shape {
        Shape { size: _, .. } -> 0
    }
}
"#,
    );
    assert!(
        errors.contains(&"`Shape` is not a constructor".to_owned()),
        "{errors:?}"
    );
}

#[salsa_test]
fn named_patterns_are_checked_for_exhaustiveness(db: &salsa::DatabaseImpl) {
    let shapes = r#"
enum Shape {
    Circle { radius: Nat },
    Rect { width: Nat, tall: Bool },
}
"#;
    let covered = format!(
        "{shapes}{}",
        r#"
fn area(shape: Shape) -> Nat {
    case shape {
        Circle { .. } -> 0
        Rect { tall: True, width } -> width
        Rect { width: _, tall: False } -> 1
    }
}
"#
    );
    let errors = errors(db, &covered);
    assert!(errors.is_empty(), "{errors:?}");

    let partial = format!(
        "{shapes}{}",
        r#"
fn area(shape: Shape) -> Nat {
    case shape {
        Circle { .. } -> 0
        Rect { tall: True, .. } -> 1
    }
}
"#
    );
    assert_eq!(
        messages(db, &partial, DiagnosticSeverity::Error),
        ["non-exhaustive case expression: missing patterns: Rect(_, False)"]
    );

    let repeated = format!(
        "{shapes}{}",
        r#"
fn area(shape: Shape) -> Nat {
    case shape {
        Rect { tall: _, .. } -> 1
        Rect { width, tall: _ } -> width
        Circle { radius } -> radius
    }
}
"#
    );
    assert_eq!(
        messages(db, &repeated, DiagnosticSeverity::Warning),
        ["unreachable pattern"]
    );
}

#[salsa_test]
fn lone_spread_matches_every_field(db: &salsa::DatabaseImpl) {
    let source = format!(
        "{POINT}{}",
        r#"
enum Shape {
    Circle { radius: Nat },
    Rect { width: Nat, tall: Bool },
    Empty,
}

fn kind(shape: Shape) -> Nat {
    case shape {
        Circle { .. } -> 0
        Rect { .., } -> 1
        Empty { .. } -> 2
    }
}

fn ignore(p: Point) -> Nat {
    let Point { .. } = p
    0
}
"#
    );
    let errors = errors(db, &source);
    assert!(errors.is_empty(), "{errors:?}");
}
