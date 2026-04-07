//! Tests for evidence parameter insertion during AST-to-IR lowering.
//!
//! Verifies that:
//! - Effectful functions receive `(evidence, done_k, params...)` in their signature
//! - Effectful call sites pass evidence as the first argument
//! - Pure functions do not have evidence parameters
//! - Pure contexts create null evidence when calling effectful functions

mod common;

use self::common::run_ast_pipeline_with_ir;
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// Effectful function should have evidence as first block arg, before done_k.
#[salsa_test]
fn test_effectful_func_has_evidence_param(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Foo {
    op bar() -> Nat
}

fn effectful() ->{Foo} Nat {
    Foo::bar()
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Direct call to an effectful function from another effectful function
/// should pass the caller's evidence as the first argument.
#[salsa_test]
fn test_effectful_call_passes_evidence(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Counter {
    op inc() -> Nat
}

fn get_count() ->{Counter} Nat {
    Counter::inc()
}

fn use_counter() ->{Counter} Nat {
    let a = get_count()
    let b = get_count()
    a + b
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Pure function calling an effectful function through a handler should
/// create null evidence for the outer call context.
#[salsa_test]
fn test_pure_context_creates_null_evidence(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Ask {
    op ask() -> Nat
}

fn use_ask() ->{Ask} Nat {
    Ask::ask()
}

fn main() {
    let r = handle use_ask() {
        do result { result }
        op Ask::ask() { resume 42 }
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
