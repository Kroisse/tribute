//! Local callables in source-logical lowering: a let-bound lambda is one
//! lambda, and a named-function alias references its function at each use.
mod common;

use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

#[salsa_test]
fn fixed_captured_local_reaches_pure_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "local_capture.trb",
        r#"
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn captured(k: Int) ->{} Int {
    let action = fn(x: Int) k
    apply(action, +3)
}

fn entry() ->{} Int { captured(+9) }
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(ir.contains("tribute_control.lambda"));
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn a_lambda_and_its_alias_are_one_lambda(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "alias.trb",
        r#"
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn entry() ->{} Int {
    let action = fn(x: Int) x
    let alias = action
    let first = pure(alias, +3)
    pure(action, first)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert_eq!(ir.matches("tribute_control.lambda").count(), 1, "{ir}");
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

/// A lambda that uses a named-function alias references the function in its
/// body, so it captures nothing.
#[salsa_test]
fn a_named_alias_inside_a_lambda_captures_nothing(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "named_capture.trb",
        r#"
fn identity(x: Int) ->{} Int { x }
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn entry() ->{} Int {
    let named = identity
    let alias = named
    pure(fn(x: Int) open(alias, x), +3)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(ir.contains("captures []"), "{ir}");
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn a_named_alias_passed_to_an_open_consumer_needs_no_cast(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "indirect_consumer.trb",
        r#"
fn identity(x: Int) ->{} Int { x }
fn call(consumer: fn(Int) ->{e} Int, x: Int) ->{e} Int { consumer(x) }
fn entry() ->{} Int {
    let named = identity
    let alias = named
    call(alias, +3)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn shadowed_lambda_bindings_are_separate_lambdas(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "shadowed.trb",
        r#"
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn entry() ->{} Int {
    let action = fn(x: Int) x
    let first = pure(action, +3)
    let action = fn(x: Int) first
    pure(action, +9)
}
"#,
    );
    let ir = common::run_ast_pipeline_with_ir(db, source);
    assert_eq!(ir.matches("tribute_control.lambda").count(), 2, "{ir}");
    assert!(!ir.contains("unrealized_conversion_cast"), "{ir}");
}

#[salsa_test]
fn effectful_local_is_rejected_by_pure_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "effectful_local.trb",
        r#"
ability State(s) { fn get() -> s }
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn effectful(x: Int) ->{State(Int)} Int { State::get() + x }
fn invalid() ->{State(Int)} Int {
    let action = effectful
    apply(action, +3)
}
fn main() -> Nil {}
"#,
    );
    let errors = common::ast_pipeline_error_messages(db, source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("ability row mismatch")),
        "{errors:?}"
    );
}
