//! Fixed-type local callable behavior on the production route.
//!
//! These cases exercise fixed-data-type local lambdas, `fn`-declared named
//! callables, and captures under managed ownership on the source-logical
//! route, and fixed-data-type local helpers whose callbacks have different
//! ability rows. Named-operator callables used directly or through a local
//! alias are deferred to issue #1007 and are intentionally absent here.
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

const STATE_PRELUDE: &str = r#"use std::io::{Io, print_line}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() ->{Io} Nil { print_line(run_state(fn() body(), +10).to_string) }
"#;

/// Runs `body() ->{State(Int)} Int` under a `State` handler starting at 10.
fn assert_state_program(body: &str, expected: &str) {
    let source = format!("{STATE_PRELUDE}\n{body}");
    common::assert_output_on_both_targets("local_row_polymorphic.trb", &source, expected);
}

#[test]
fn local_helper_takes_callbacks_with_different_rows_in_either_order() {
    for (calls, expected) in [
        (
            "let a = twice(fn(x: Int) ->{} Int { x + +1 }, +1)
    twice(fn(x: Int) { State::set(State::get() + x)
                       State::get() }, a)",
            "26",
        ),
        (
            "let a = twice(fn(x: Int) { State::set(State::get() + x)
                               State::get() }, +3)
    twice(fn(x: Int) ->{} Int { x + +1 }, a)",
            "28",
        ),
    ] {
        assert_state_program(
            &format!(
                "fn body() ->{{State(Int)}} Int {{
    let twice = fn(f: fn(Int) ->{{e}} Int, x: Int) ->{{e}} Int {{ f(f(x)) }}
    {calls}
}}"
            ),
            expected,
        );
    }
}

#[test]
fn local_helper_with_a_nested_callable_signature() {
    assert_state_program(
        r#"fn body() ->{State(Int)} Int {
    let with_inc = fn(k: fn(fn(Int) ->{e} Int) ->{e} Int) ->{e} Int { k(fn(x: Int) x + +1) }
    let a = with_inc(fn(g: fn(Int) ->{} Int) ->{} Int { g(+1) })
    with_inc(fn(g: fn(Int) ->{State(Int)} Int) {
        State::set(a)
        g(State::get())
    })
}"#,
        "3",
    );
}

#[test]
fn local_helper_alias_and_capture_keep_their_rows() {
    assert_state_program(
        r#"fn body() ->{State(Int)} Int {
    let k = +100
    let twice = fn(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(f(x + k)) }
    let alias = twice
    let a = alias(fn(x: Int) ->{} Int { x + +1 }, +1)
    alias(fn(x: Int) { State::set(State::get() + x)
                       State::get() }, a)
}"#,
        "426",
    );
}

#[test]
fn local_helper_passed_to_consumers_with_different_rows() {
    assert_state_program(
        r#"fn use_pure(h: fn(fn(Int) ->{} Int, Int) ->{} Int) ->{} Int { h(fn(x: Int) x + +1, +1) }
fn use_state(h: fn(fn(Int) ->{State(Int)} Int, Int) ->{State(Int)} Int) ->{State(Int)} Int {
    h(fn(x: Int) { State::set(State::get() + x)
                   State::get() }, +2)
}
fn body() ->{State(Int)} Int {
    let twice = fn(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(f(x)) }
    let a = use_pure(twice)
    a + use_state(twice)
}"#,
        "27",
    );
}

/// One use of the helper is inside another lambda, whose captures must name
/// the copy that use selects.
#[test]
fn local_helper_used_inside_another_lambda_at_a_second_row() {
    assert_state_program(
        r#"fn body() ->{State(Int)} Int {
    let twice = fn(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(f(x)) }
    let a = twice(fn(x: Int) ->{} Int { x + +1 }, +1)
    let go = fn() ->{State(Int)} Int {
        twice(fn(x: Int) { State::set(State::get() + x)
                           State::get() }, a)
    }
    go()
}"#,
        "26",
    );
}

#[test]
fn local_helper_used_by_a_row_polymorphic_local_helper() {
    assert_state_program(
        r#"fn body() ->{State(Int)} Int {
    let twice = fn(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(f(x)) }
    let quad = fn(f: fn(Int) ->{r} Int, x: Int) ->{r} Int { twice(f, twice(f, x)) }
    let a = quad(fn(x: Int) ->{} Int { x + +1 }, +1)
    quad(fn(x: Int) { State::set(State::get() + x)
                      State::get() }, a)
}"#,
        "120",
    );
}

#[test]
fn local_helper_forwards_the_enclosing_open_row() {
    common::assert_output_on_both_targets(
        "local_row_forwarding.trb",
        &format!(
            "{}\n{}",
            STATE_PRELUDE.replace(
                "fn main() ->{Io} Nil { print_line(run_state(fn() body(), +10).to_string) }",
                ""
            ),
            r#"fn open(f: fn(Int) ->{e} Int, x: Int) ->{e} Int {
    let twice = fn(g: fn(Int) ->{r} Int, y: Int) ->{r} Int { g(g(y)) }
    let a = twice(fn(y: Int) ->{} Int { y + +1 }, x)
    twice(f, a)
}
fn main() ->{Io} Nil {
    print_line(open(fn(x: Int) ->{} Int { x * +2 }, +1).to_string)
    print_line(run_state(fn() open(fn(x: Int) {
        State::set(x)
        State::get() + +1
    }, +1), +0).to_string)
}"#
        ),
        "12\n5",
    );
}

/// A pure factory's result is generalized like a lambda but stays one value.
#[test]
fn local_helper_from_a_factory_takes_callbacks_with_different_rows() {
    assert_state_program(
        r#"fn make() ->{} fn(fn(Int) ->{e} Int, Int) ->{e} Int {
    fn(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(f(x)) }
}
fn body() ->{State(Int)} Int {
    let twice = make()
    let a = twice(fn(x: Int) ->{} Int { x + +1 }, +1)
    twice(fn(x: Int) { State::set(State::get() + x)
                       State::get() }, a)
}"#,
        "26",
    );
}
