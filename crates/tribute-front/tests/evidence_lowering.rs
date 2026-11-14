//! Tests for source-logical effectful callables during AST-to-IR lowering.
//!
//! Verifies that:
//! - Effectful functions expose only their source-visible signature
//! - Effectful calls remain logical `tribute_control.call` operations
//! - A pure source context installs a logical handler without frontend evidence

mod common;

use self::common::run_ast_pipeline_with_ir;
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// Effectful functions retain only source-visible parameters and results.
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

fn main() -> Nil { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Direct calls between effectful source functions remain logical calls.
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

fn main() -> Nil { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// A pure function handling an effectful call introduces no frontend evidence.
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

fn main() -> Nil {
    let r = handle use_ask() {
        do result { result }
        op Ask::ask() { 42 }
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// A `resume` under a handler the resumed body does not name masks it, so
/// operations reaching the body's row tail pass that handler.
#[salsa_test]
fn resume_under_another_handler_masks_it(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Ping {
    op ping() -> Nat
}

ability Other {
    op other() -> Nil
}

fn handled(comp: fn() ->{e, Ping} Nat) ->{e, Ping} Nat {
    handle comp() {
        op Ping::ping() {
            handle {
                Other::other()
                resume 0
            } {
                op Other::other() { 0 }
            }
        }
    }
}

fn main() -> Nil { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let resume = ir_text
        .lines()
        .find(|line| line.contains("= tribute_control.resume "))
        .expect("source-logical resume");
    assert!(
        resume.contains("{evidence_plan = [{mask = @\"ability_") && resume.contains("Other\"}]}"),
        "{resume}"
    );
}
