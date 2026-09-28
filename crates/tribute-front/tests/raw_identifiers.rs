//! Strict keywords and raw identifiers.

mod common;

use self::common::{ast_pipeline_diagnostics, ast_pipeline_error_messages};
use salsa_test_macros::salsa_test;
use tribute_core::diagnostic::{CompilationPhase, DiagnosticSeverity};
use tribute_front::SourceCst;

/// Assert exactly one parsing or AST generation error whose span covers the
/// first occurrence of `part` in `text` and whose message contains `message`.
fn assert_error(
    db: &dyn salsa::Database,
    text: &str,
    part: &str,
    phase: CompilationPhase,
    message: &str,
) {
    let source = SourceCst::from_source_str(db, "raw_identifiers.trb", text);
    let errors: Vec<_> = ast_pipeline_diagnostics(db, source)
        .into_iter()
        .filter(|d| {
            d.inner.severity == DiagnosticSeverity::Error
                && matches!(
                    d.phase,
                    CompilationPhase::Parsing | CompilationPhase::AstGeneration
                )
        })
        .collect();
    let [error] = errors.as_slice() else {
        panic!("expected exactly one error for {text:?}, got {errors:?}");
    };
    assert_eq!(error.phase, phase);
    let start = text.find(part).expect("part must occur");
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
fn raw_identifiers_name_keywords(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "raw_identifiers.trb",
        r#"
struct Token {
    r#type: Nat
}

fn r#case(r#op: Nat) -> Nat {
    let r#in = r#op
    r#in
}

fn kind(token: Token) -> Nat {
    token.r#type
}

fn run() -> Nat {
    let token = Token { r#type: 1 }
    r#case(kind(token))
}
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
}

#[salsa_test]
fn raw_and_bare_spellings_name_the_same_binding(db: &salsa::DatabaseImpl) {
    let source =
        SourceCst::from_source_str(db, "raw_identifiers.trb", "fn f(r#x: Nat) -> Nat { x }\n");
    let errors = ast_pipeline_error_messages(db, source);
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
}

#[salsa_test]
fn keyword_binding_suggests_raw_identifier(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn f() -> Nat {\n    let op = 1\n    1\n}\n",
        "op",
        CompilationPhase::Parsing,
        "`op` is a keyword; write `r#op` to use it as a name",
    );
}

#[salsa_test]
fn keyword_field_suggests_raw_identifier(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "struct T { case: Nat }\n",
        "case",
        CompilationPhase::Parsing,
        "`case` is a keyword; write `r#case` to use it as a name",
    );
}

#[salsa_test]
fn keyword_after_dot_suggests_raw_identifier(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn f(x: Nat) -> Nat { x.as }\n",
        "as",
        CompilationPhase::Parsing,
        "`as` is a keyword; write `r#as` to use it as a name",
    );
}

#[salsa_test]
fn reserved_word_is_rejected(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn f() -> Nat {\n    let where = 1\n    1\n}\n",
        "where",
        CompilationPhase::AstGeneration,
        "`where` is reserved for future use; write `r#where` to use it as a name",
    );
}

#[salsa_test]
fn raw_path_keyword_is_rejected(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn f() -> Nat {\n    let r#self = 1\n    1\n}\n",
        "r#self",
        CompilationPhase::AstGeneration,
        "`self` cannot be a raw identifier",
    );
}

#[salsa_test]
fn path_keyword_binding_is_rejected(db: &salsa::DatabaseImpl) {
    assert_error(
        db,
        "fn f() -> Nat {\n    let self = 1\n    1\n}\n",
        "self",
        CompilationPhase::Parsing,
        "`self` is a keyword and cannot be used as a name",
    );
}
