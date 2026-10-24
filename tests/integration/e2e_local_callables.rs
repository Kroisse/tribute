//! Fixed-type local callable behavior on the production route.
//!
//! These cases exercise fixed-data-type local lambdas, `fn`-declared named
//! callables, and captures under managed ownership on the source-logical
//! route. Named-operator callables used directly or through a local alias are
//! deferred to issue #1007 and are intentionally absent here.
use crate::common;

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
use std::io::{Io, print_line}
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn main() ->{Io} Nil {
    let identity = fn(x: Int) x
    print_line(apply(identity, +3).to_string)
}
"#,
        b"3\n",
    );
}

#[test]
fn captured_alias_prints_nine() {
    assert_native(
        r#"
use std::io::{Io, print_line}
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn captured(k: Int) ->{} Int {
    let action = fn(x: Int) k
    let alias = action
    apply(alias, +3)
}
fn main() ->{Io} Nil { print_line(captured(+9).to_string) }
"#,
        b"9\n",
    );
}

#[test]
fn factory_rhs_is_evaluated_once() {
    assert_native(
        r#"
use std::io::{Io, print_line}
fn make() ->{Io} fn(Int) ->{} Int {
    print_line(+1.to_string)
    fn(x: Int) ->{} Int { x }
}
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn main() ->{Io} Nil {
    let action = make()
    let alias = action
    print_line(apply(action, +3).to_string)
    print_line(apply(alias, +9).to_string)
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
fn main() -> Nil {}
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
            .any(|diagnostic| diagnostic.inner.message.contains("ability row mismatch")),
        "{errors:?}"
    );
}

#[test]
fn named_direct_pure_and_open_consumers() {
    assert_native(
        r#"
use std::io::{Io, print_line}
fn identity(x: Int) ->{} Int { x }
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() ->{Io} Nil {
    let first = pure(identity, +3)
    print_line(open(identity, first).to_string)
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
use std::io::{{Io, print_line}}
fn pure(f: fn(Int) ->{{}} Int, x: Int) ->{{}} Int {{ f(x) }}
fn open(f: fn(Int) ->{{e}} Int, x: Int) ->{{e}} Int {{ f(x) }}
fn run() ->{{}} Int {{
    let action = fn(x: Int) x
    {calls}
}}
fn main() ->{{Io}} Nil {{ print_line(run().to_string) }}
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
use std::io::{Io, print_line}
fn main() ->{Io} Nil {
    let identity = fn(x) x
    print_line(identity(+3).to_string)
    case identity(True) {
        True -> print_line({{ +4 }}.to_string)
        False -> print_line({{ +0 }}.to_string)
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
use std::io::{Io, print_line}
struct Box { value: Int }
fn apply(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn captured(k: Box) ->{} Int {
    let action = fn(x: Int) k.value
    apply(action, +3)
}
fn main() ->{Io} Nil { print_line(captured(Box { value: +9 }).to_string) }
"#,
        b"9\n",
    );
}

#[test]
fn local_open_then_pure_consumers() {
    assert_native(
        r#"
use std::io::{Io, print_line}
fn pure(f: fn(Int) ->{} Int, x: Int) ->{} Int { f(x) }
fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn run() ->{} Int {
    let action = fn(x: Int) x
    let first = open(action, +3)
    pure(action, first)
}
fn main() ->{Io} Nil { print_line(run().to_string) }
"#,
        b"3\n",
    );
}
