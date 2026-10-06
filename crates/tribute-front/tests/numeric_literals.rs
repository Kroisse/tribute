//! Diagnostics for numeric literal separators, exponents, and suffixes.

use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_front::SourceCst;

/// Error diagnostics produced while parsing and lowering `text`.
fn lowering_errors(db: &dyn salsa::Database, text: &str) -> Vec<Diagnostic> {
    let source = SourceCst::from_source_str(db, "numeric_literals.trb", text);
    tribute_front::query::parsed_ast(db, source);
    tribute_front::query::parsed_ast::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .filter(|d| d.inner.severity == DiagnosticSeverity::Error)
        .cloned()
        .collect()
}

/// Assert exactly one AST generation error whose span covers `part` of the
/// first occurrence of `literal` in `text`.
fn assert_literal_error(
    db: &dyn salsa::Database,
    text: &str,
    literal: &str,
    part: &str,
    message_part: &str,
) {
    let errors = lowering_errors(db, text);
    let [error] = errors.as_slice() else {
        panic!("expected exactly one error for {text:?}, got {errors:?}");
    };
    assert_eq!(error.phase, CompilationPhase::AstGeneration, "{text:?}");
    let start = text
        .find(literal)
        .expect("literal must occur in the source")
        + literal.find(part).expect("part must occur in the literal");
    assert_eq!(
        (error.inner.span.start, error.inner.span.end),
        (start, start + part.len()),
        "{text:?}: span must cover {part:?}"
    );
    assert!(
        error.inner.message.contains(message_part),
        "{text:?}: unexpected message: {}",
        error.inner.message
    );
}

#[salsa_test]
fn valid_numeric_literals_are_accepted(db: &salsa::DatabaseImpl) {
    let errors = lowering_errors(
        db,
        r#"fn main() -> Nil {
    let a = 1_000_000
    let b = 0x_FF_FF
    let c = 1e10
    let d = -1e3
    let e = 1.5e-3
    let f = 42i
    let g = 1e-3f
    let h = +0xFF
    case a { 1_000n -> b, _ -> 0 }
}"#,
    );
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
}

#[salsa_test]
fn invalid_numeric_literals_are_rejected(db: &salsa::DatabaseImpl) {
    for (text, literal, part, message_part) in [
        // unknown suffix is rejected
        (
            "fn main() -> Nil { 42u8 }",
            "42u8",
            "u8",
            "unknown numeric literal suffix `u8`",
        ),
        // negative exponent on integer suggests float
        (
            "fn main() -> Nil { 1e-3 }",
            "1e-3",
            "e-3",
            "write `1.0e-3` or `1e-3f` for a Float",
        ),
        // invalid radix digit is rejected
        (
            "fn main() -> Nil { 0b102 }",
            "0b102",
            "2",
            "invalid digit `2` in binary literal",
        ),
        // suffix conflict in pattern is rejected
        (
            "fn f(x: Float) -> Nat { case x { 1.5i -> 1, _ -> 0 } }",
            "1.5i",
            "i",
            "suffix `i` cannot be used on a literal with a decimal point",
        ),
        // overflowing literal is rejected
        (
            "fn main() -> Nil { 1e20 }",
            "1e20",
            "1e20",
            "exceeds the current implementation limit for `Nat`",
        ),
        // suffix on hexadecimal literal is rejected
        (
            "fn main() -> Nil { 0xFFi }",
            "0xFFi",
            "i",
            "write `+0xFF` for an Int",
        ),
    ] {
        assert_literal_error(db, text, literal, part, message_part);
    }
}
