//! Fixed-type local callable behavior on the production route.
//!
//! These cases exercise fixed-data-type local lambdas, `fn`-declared named
//! callables, and captures under managed ownership on the source-logical
//! route. Named-operator callables used directly or through a local alias are
//! deferred to issue #1007 and are intentionally absent here.
mod common;

use salsa_test_macros::salsa_test;
use tribute::pipeline::parse_and_lower_ast;
use tribute_core::{Diagnostic, diagnostic::DiagnosticSeverity};
use tribute_front::SourceCst;

fn assert_native(body: &str, expected: &[u8]) {
    let source = format!("{}\n{body}", common::PRINT_EXTERNS);
    let output = common::compile_and_run_native("local_callable.trb", &source);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, expected);
}

#[test]
fn fixed_local_identity_prints_three() {
    assert_native(
        r#"
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn main() {
    let identity = fn(x: Int) x
    __tribute_print_int(apply(identity, +3))
}
"#,
        b"3\n",
    );
}

#[test]
fn captured_alias_prints_nine() {
    assert_native(
        r#"
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn captured(k: Int) ->{} Int {
    let action = fn(x: Int) k
    let alias = action
    apply(alias, +3)
}
fn main() { __tribute_print_int(captured(+9)) }
"#,
        b"9\n",
    );
}

#[test]
fn factory_rhs_is_evaluated_once() {
    assert_native(
        r#"
fn make() -> fn(Int) ->{} Int {
    __tribute_print_int(+1)
    fn(x: Int) ->{} Int { x }
}
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn main() {
    let action = make()
    let alias = action
    __tribute_print_int(apply(action, +3))
    __tribute_print_int(apply(alias, +9))
}
"#,
        b"1\n3\n9\n",
    );
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
fn main() {}
"#,
    );
    let _ = parse_and_lower_ast(db, source);
    let errors: Vec<_> = parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
        .into_iter()
        .filter(|diagnostic| diagnostic.inner.severity == DiagnosticSeverity::Error)
        .collect();
    assert!(
        !errors.is_empty(),
        "effectful callable must not become pure"
    );
    assert!(
        errors
            .iter()
            .any(|diagnostic| diagnostic.inner.message.contains("effect")),
        "{errors:?}"
    );
}

#[test]
fn named_direct_pure_and_open_consumers() {
    assert_native(
        r#"
fn identity(x: Int) ->{} Int { x }
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() {
    let first = pure(identity, +3)
    __tribute_print_int(open(identity, first))
}
"#,
        b"3\n",
    );
}

#[test]
fn local_pure_and_open_consumers() {
    for calls in [
        "let first = pure(action, +3)\nopen(action, first)",
        "let first = open(action, +3)\npure(action, first)",
    ] {
        assert_native(
            &format!(
                r#"
fn pure(f: fn(Int) ->{{}} Int, x: Int) ->{{}} Int {{ f(x) }}
fn open(f: fn(Int) ->{{e}} Int, x: Int) ->{{e}} Int {{ f(x) }}
fn run() ->{{}} Int {{
    let action = fn(x: Int) x
    {calls}
}}
fn main() {{ __tribute_print_int(run()) }}
"#
            ),
            b"3\n",
        );
    }
}

#[test]
fn existing_generalized_local_direct_calls() {
    assert_native(
        r#"
fn main() {
    let identity = fn(x) x
    __tribute_print_int(identity(+3))
    case identity(True) {
        True -> __tribute_print_int(+4)
        False -> __tribute_print_int(+0)
    }
}
"#,
        b"3\n4\n",
    );
}

#[test]
fn managed_capture_retains_its_field() {
    assert_native(
        r#"
struct Box { value: Int }
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn captured(k: Box) ->{} Int {
    let action = fn(x: Int) k.value
    apply(action, +3)
}
fn main() { __tribute_print_int(captured(Box { value: +9 })) }
"#,
        b"9\n",
    );
}

#[test]
fn local_open_then_pure_consumers() {
    assert_native(
        r#"
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn run() ->{} Int {
    let action = fn(x: Int) x
    let first = open(action, +3)
    pure(action, first)
}
fn main() { __tribute_print_int(run()) }
"#,
        b"3\n",
    );
}
