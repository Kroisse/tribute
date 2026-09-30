//! Diagnostics for Unicode escapes in string and rune literals.

use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::SourceCst;

/// Error diagnostics produced while parsing and lowering `text`.
fn lowering_errors(db: &dyn salsa::Database, text: &str) -> Vec<Diagnostic> {
    let source = SourceCst::from_source_str(db, "literal_escapes.trb", text);
    tribute_front::query::parsed_ast(db, source);
    tribute_front::query::parsed_ast::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .filter(|d| d.inner.severity == DiagnosticSeverity::Error)
        .cloned()
        .collect()
}

/// Assert exactly one AST generation error whose span covers `escape` in `text`.
fn assert_escape_error(db: &dyn salsa::Database, text: &str, escape: &str, message_part: &str) {
    let errors = lowering_errors(db, text);
    let [error] = errors.as_slice() else {
        panic!("expected exactly one error for {text:?}, got {errors:?}");
    };
    assert_eq!(error.phase, CompilationPhase::AstGeneration);
    let start = text.find(escape).expect("escape must occur in the source");
    assert_eq!(
        (error.inner.span.start, error.inner.span.end),
        (start, start + escape.len()),
        "span must cover the escape sequence"
    );
    assert!(
        error.inner.message.contains(message_part),
        "unexpected message: {}",
        error.inner.message
    );
}

#[salsa_test]
fn valid_unicode_escapes_are_accepted(db: &salsa::DatabaseImpl) {
    let errors = lowering_errors(
        db,
        r#"fn main() -> Nil {
    let s = "caf\u{E9} \u{1F600} \u{0}\u{D7FF}\u{E000}\u{10FFFF}"
    let r = ?\u{1F600}
    case s { "\u{41}" -> r, _ -> ?\u{41} }
}"#,
    );
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
}

#[salsa_test]
fn surrogate_escape_in_string_is_rejected(db: &salsa::DatabaseImpl) {
    assert_escape_error(
        db,
        r#"fn main() -> Nil { "a\u{D800}b" }"#,
        r"\u{D800}",
        "U+D800 is a surrogate code point",
    );
}

#[salsa_test]
fn out_of_range_escape_in_string_is_rejected(db: &salsa::DatabaseImpl) {
    assert_escape_error(
        db,
        r#"fn main() -> Nil { s"x\u{110000}" }"#,
        r"\u{110000}",
        "U+110000 exceeds the maximum U+10FFFF",
    );
}

#[salsa_test]
fn surrogate_escape_in_rune_is_rejected(db: &salsa::DatabaseImpl) {
    assert_escape_error(
        db,
        r"fn main() -> Nil { ?\u{dfff} }",
        r"\u{dfff}",
        "U+DFFF is a surrogate code point",
    );
}

#[salsa_test]
fn surrogate_escape_in_string_pattern_is_rejected(db: &salsa::DatabaseImpl) {
    assert_escape_error(
        db,
        r#"fn f(s: String) -> Nat { case s { "\u{DBFF}" -> 1, _ -> 0 } }"#,
        r"\u{DBFF}",
        "U+DBFF is a surrogate code point",
    );
}

#[salsa_test]
fn fixed_width_unicode_escape_is_a_parse_error(db: &salsa::DatabaseImpl) {
    // Built at runtime: the removed form is `\u` followed by four hex digits.
    let text = format!(r#"fn main() -> Nil {{ "{}u0041" }}"#, '\\');
    let errors = lowering_errors(db, &text);
    assert!(
        errors.iter().any(|d| d.phase == CompilationPhase::Parsing),
        "expected a parse error, got {errors:?}"
    );
}

#[salsa_test]
fn unicode_escape_in_bytes_is_a_parse_error(db: &salsa::DatabaseImpl) {
    let errors = lowering_errors(db, r#"fn main() -> Nil { b"\u{41}" }"#);
    assert!(
        errors.iter().any(|d| d.phase == CompilationPhase::Parsing),
        "expected a parse error, got {errors:?}"
    );
}
