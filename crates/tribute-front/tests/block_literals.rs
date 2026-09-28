//! Diagnostics for block literal indentation and escapes in `#`-delimited
//! string and bytes literals.

use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::SourceCst;

/// Error diagnostics produced while parsing and lowering `text`.
fn lowering_errors(db: &dyn salsa::Database, text: &str) -> Vec<Diagnostic> {
    let source = SourceCst::from_source_str(db, "block_literals.trb", text);
    tribute_front::query::parsed_ast(db, source);
    tribute_front::query::parsed_ast::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .filter(|d| d.inner.severity == DiagnosticSeverity::Error)
        .cloned()
        .collect()
}

/// Assert exactly one AST generation error whose span covers the first
/// occurrence of `part` at or after `after` in `text`.
fn assert_error(db: &dyn salsa::Database, text: &str, after: &str, part: &str, message: &str) {
    let errors = lowering_errors(db, text);
    let [error] = errors.as_slice() else {
        panic!("expected exactly one error for {text:?}, got {errors:?}");
    };
    assert_eq!(error.phase, CompilationPhase::AstGeneration);
    let from = text.find(after).expect("anchor must occur");
    let start = from + text[from..].find(part).expect("part must occur");
    assert_eq!(
        (error.inner.span.start, error.inner.span.end),
        (start, start + part.len()),
        "span must cover {part:?}"
    );
    assert!(
        error.inner.message.contains(message),
        "unexpected message: {}",
        error.inner.message
    );
}

#[salsa_test]
fn under_indented_line_is_rejected(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn main() -> String {\n    #\"\n        a\n      b\n        \"#\n}\n",
        "a\n",
        "      ",
        "line is not indented with the block literal's indentation (8 spaces)",
    );
}

#[salsa_test]
fn closing_delimiter_after_content_is_rejected(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn main() -> String {\n    #\"\n        a\n        b\"#\n}\n",
        "b",
        "\"#",
        "closing delimiter of a block literal must be on its own line",
    );
}

#[salsa_test]
fn line_continuation_is_reserved(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn main() -> String {\n    #\"\n        a\\\n        b\n        \"#\n}\n",
        "a",
        "\\\n",
        "before a line break is not an escape sequence",
    );
}

#[salsa_test]
fn unknown_escape_in_multiline_string_is_rejected(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        r##"fn main() -> String { #"a\qb"# }"##,
        "a",
        r"\q",
        r"unknown escape sequence `\q`",
    );
}

#[salsa_test]
fn unicode_escape_in_multiline_bytes_is_rejected(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        r##"fn main() -> Bytes { b#"\u{41}"# }"##,
        "b#",
        r"\u{41}",
        "bytes literals cannot contain",
    );
}

#[salsa_test]
fn interpolation_is_reported_as_unimplemented(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn main() -> String {\n    let x = \"b\"\n    \"a\\{x}c\"\n}\n",
        "\"a",
        r"\{x}",
        "interpolation is not implemented yet",
    );
}
