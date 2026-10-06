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
    assert_eq!(error.phase, phase, "{text:?}");
    let start = text.find(part).expect("part must occur");
    assert_eq!(
        (error.inner.span.start, error.inner.span.end),
        (start, start + part.len()),
        "{text:?}: span must cover {part:?}"
    );
    assert!(
        error.inner.message.contains(message),
        "{text:?}: unexpected message: {}",
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
fn keyword_misuse_is_rejected(db: &salsa::DatabaseImpl) {
    for (text, part, phase, message) in [
        // keyword binding suggests raw identifier
        (
            "fn f() -> Nat {\n    let op = 1\n    1\n}\n",
            "op",
            CompilationPhase::Parsing,
            "`op` is a keyword; write `r#op` to use it as a name",
        ),
        // keyword field suggests raw identifier
        (
            "struct T { case: Nat }\n",
            "case",
            CompilationPhase::Parsing,
            "`case` is a keyword; write `r#case` to use it as a name",
        ),
        // keyword after dot suggests raw identifier
        (
            "fn f(x: Nat) -> Nat { x.as }\n",
            "as",
            CompilationPhase::Parsing,
            "`as` is a keyword; write `r#as` to use it as a name",
        ),
        // reserved word is rejected
        (
            "fn f() -> Nat {\n    let where = 1\n    1\n}\n",
            "where",
            CompilationPhase::AstGeneration,
            "`where` is reserved for future use; write `r#where` to use it as a name",
        ),
        // raw path keyword is rejected
        (
            "fn f() -> Nat {\n    let r#self = 1\n    1\n}\n",
            "r#self",
            CompilationPhase::AstGeneration,
            "`self` cannot be a raw identifier",
        ),
        // path keyword binding is rejected
        (
            "fn f() -> Nat {\n    let self = 1\n    1\n}\n",
            "self",
            CompilationPhase::Parsing,
            "`self` is a keyword and cannot be used as a name",
        ),
        // reserved word path segment is rejected
        (
            "fn f() -> Nat { foo::where::bar }\n",
            "where",
            CompilationPhase::AstGeneration,
            "`where` is reserved for future use",
        ),
        // raw path keyword path segment is rejected
        (
            "fn f() -> Nat { foo::r#self::bar }\n",
            "r#self",
            CompilationPhase::AstGeneration,
            "`self` cannot be a raw identifier",
        ),
    ] {
        assert_error(db, text, part, phase, message);
    }
}
