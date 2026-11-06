//! A lambda's written result type and ability row take part in inference
//! (`new-plans/type-inference.md`, lambda annotations).

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

const STATE: &str = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}
"#;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(
        db,
        SourceCst::from_source_str(db, "lambda_annotations.trb", text),
    )
}

fn with_state(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    errors(db, &format!("{STATE}{text}"))
}

#[salsa_test]
fn mismatched_result_annotation_is_rejected(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn main() -> Nil {
    let f = fn() -> Int { "x" }
    let s = f()
}
"#,
    );
    assert_eq!(
        errors,
        ["type error in function 'main': expected `Int`, found `std::String`"]
    );
}

#[salsa_test]
fn matching_result_annotation_is_accepted(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn main() -> Nil {
    let positive = fn(x: Int) -> Bool { x > +0 }
    let b = positive(+1)
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// The annotation fixes the ability argument the body leaves open, so the
/// lambda is not generalized over it and its call performs `State(Int)`.
#[salsa_test]
fn result_annotation_fixes_a_type_the_body_leaves_open(db: &salsa::DatabaseImpl) {
    let errors = with_state(
        db,
        r#"
fn run() ->{State(String)} Nat {
    let read = fn() -> Int { State::get() }
    let _ = read()
    0
}
"#,
    );
    assert_eq!(
        errors,
        [
            "type error in function 'run': ability row mismatch: expected `{State(std::String)}`, found `{State(Int), State(std::String)}`"
        ]
    );
}

#[salsa_test]
fn result_annotation_types_an_unannotated_parameter(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn main() -> Nil {
    let f = fn(x) -> Int { x }
    let s = f("x")
}
"#,
    );
    assert_eq!(
        errors,
        ["type error at call site in function 'main': expected `Int`, found `std::String`"]
    );
}

#[salsa_test]
fn matching_row_annotation_is_accepted(db: &salsa::DatabaseImpl) {
    let errors = with_state(
        db,
        r#"
fn read() ->{State(Int)} Int {
    let f = fn() ->{State(Int)} Int { State::get() }
    f()
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn pure_row_annotation_rejects_an_effect(db: &salsa::DatabaseImpl) {
    let errors = with_state(
        db,
        r#"
fn read() ->{State(Int)} Int {
    let f = fn() ->{} Int { State::get() }
    f()
}
"#,
    );
    assert_eq!(
        errors,
        [
            "type error in lambda in function 'read': ability row mismatch: expected `{}`, found `{State(_)}`"
        ]
    );
}

#[salsa_test]
fn row_annotation_is_matched_with_the_context(db: &salsa::DatabaseImpl) {
    let errors = with_state(
        db,
        r#"
fn run(f: fn() ->{State(Int)} Int) ->{State(Int)} Int { f() }

fn read() ->{State(Int)} Int {
    run(fn() ->{} Int { +1 })
}
"#,
    );
    assert_eq!(
        errors,
        [
            "type error in lambda in function 'read': ability row mismatch: expected `{State(Int)}`, found `{}`"
        ]
    );
}
