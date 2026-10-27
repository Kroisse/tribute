//! Only `extern "intrinsic"` functions may be generic: an external symbol
//! cannot serve instances of different representations.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

const GENERIC_EXTERN: &str = "only `extern \"intrinsic\"` functions may have type parameters";

/// Explicit and implicit type variables are both type parameters, and an
/// unused declaration is reported too.
#[salsa_test]
fn generic_c_externs_are_type_errors(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
extern "C" fn identity(value: a) -> a
extern "C" fn untyped(value) -> Nat
extern "C" fn abs(value: Int) -> Int

fn main() -> Nil {
    let _ = abs(-3)
}
"#,
    );
    let errors: Vec<_> = ast_pipeline_error_messages(db, source)
        .into_iter()
        .filter(|message| message.contains(GENERIC_EXTERN))
        .collect();
    assert_eq!(
        errors,
        [
            format!("extern function `identity` cannot be generic; {GENERIC_EXTERN}"),
            format!("extern function `untyped` cannot be generic; {GENERIC_EXTERN}"),
        ]
    );
}

#[salsa_test]
fn generic_intrinsics_are_accepted(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        "extern \"intrinsic\" fn identity(value: a) -> a\nfn main() -> Nil { }",
    );
    assert_eq!(
        ast_pipeline_error_messages(db, source),
        Vec::<String>::new()
    );
}
