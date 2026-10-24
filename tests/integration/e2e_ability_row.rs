//! End-to-end tests for ability row polymorphism runtime behavior (#502).
//!
//! These tests verify that row-polymorphic row variables are correctly
//! instantiated and propagated at runtime — higher-order functions accepting
//! effectful callbacks, pure-for-effectful substitution, and unification
//! across multiple call sites.

use crate::common;

use common::{assert_native_output, assert_output_on_both_targets};

// =============================================================================
// Ability Row Polymorphism Tests
// =============================================================================

/// Test row-polymorphic higher-order function with effectful callback.
///
/// `apply` accepts `fn(Nat) ->{e} Nat` and calls it. When called inside
/// a `State` handler, `e` is instantiated to `{State(Nat)}`, so the callback
/// can perform State operations.
#[test]
fn test_ability_row_poly_higher_order_function_slow() {
    let code = r#"use std::io::{Io, print_line}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn apply(f: fn(Nat) ->{e} Nat, x: Nat) ->{e} Nat {
    f(x)
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() ->{Io} Nil {
    let result = run_state(fn() {
        apply(fn(x: Nat) {
            State::set(x)
            State::get()
        }, 7)
    }, 0)
    print_line(Nat::to_string(result))
}
"#;
    // apply calls the callback with x=7, which does set(7), get() → 7
    assert_native_output("ability_row_poly_higher_order.trb", code, "7");
}

/// Test row variable instantiated to multiple concrete abilities.
///
/// `apply` accepts `fn() ->{e} Nat`. The callback uses both State and Reader,
/// so `e` is instantiated to `{State(Nat), Reader(Nat)}`.
#[test]
fn test_ability_row_poly_multiple_abilities() {
    let code = r#"use std::io::{Io, print_line}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Reader(r) {
    op ask() -> r
}

fn apply(f: fn() ->{e} Nat) ->{e} Nat {
    f()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn run_reader(comp: fn() ->{e, Reader(r)} a, value: r) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { run_reader(fn() { resume value }, value) }
    }
}

fn main() ->{Io} Nil {
    let result = run_reader(fn() {
        run_state(fn() {
            apply(fn() {
                let config = Reader::ask()
                State::set(config)
                State::get()
            })
        }, 0)
    }, 42)
    print_line(Nat::to_string(result))
}
"#;
    // callback: ask() → 42, set(42), get() → 42
    assert_native_output("ability_row_poly_multiple_abilities.trb", code, "42");
}

/// Test pure function passed to a row-polymorphic parameter.
///
/// `apply` accepts `fn(Nat) ->{e} Nat`. When called with a pure lambda,
/// `e` is instantiated to `{}` (empty row), so no handler is needed.
#[test]
fn test_ability_row_poly_pure_for_effectful() {
    let code = r#"use std::io::{Io, print_line}

fn apply(f: fn(Nat) ->{e} Nat, x: Nat) ->{e} Nat {
    f(x)
}

fn main() ->{Io} Nil {
    let result = apply(fn(x: Nat) { x + 10 }, 5)
    print_line(Nat::to_string(result))
}
"#;
    // pure lambda: 5 + 10 = 15, no handler needed
    assert_native_output("ability_row_poly_pure.trb", code, "15");
}

/// Test same row-polymorphic function called with different effect instantiations.
///
/// `apply` is called twice: once where `e = {State(Nat)}` and once where
/// `e = {Reader(Nat)}`. Each call site independently instantiates the row variable.
#[test]
fn test_ability_row_poly_unification_across_call_sites_slow() {
    let code = r#"use std::io::{Io, print_line}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Reader(r) {
    op ask() -> r
}

fn apply(f: fn() ->{e} Nat) ->{e} Nat {
    f()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn run_reader(comp: fn() ->{e, Reader(r)} a, value: r) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { run_reader(fn() { resume value }, value) }
    }
}

fn main() ->{Io} Nil {
    // call 1: e = {State(Nat)}
    let a = run_state(fn() {
        apply(fn() { State::get() })
    }, 10)
    print_line(Nat::to_string(a))

    // call 2: e = {Reader(Nat)}
    let b = run_reader(fn() {
        apply(fn() { Reader::ask() })
    }, 20)
    print_line(Nat::to_string(b))
}
"#;
    // call 1: get() with State(init=10) → 10
    // call 2: ask() with Reader(value=20) → 20
    assert_native_output("ability_row_poly_unification.trb", code, "10\n20");
}

/// Test closure returned from a function and called via local variable.
///
/// `identity` returns the closure as-is. The returned closure is bound to
/// a local variable and called, exercising the Local closure CPS call path
/// with a closure obtained from a function return value.
#[test]
fn test_returned_closure_call() {
    let code = r#"use std::io::{Io, print_line}

fn identity(f: fn(Nat) ->{e} Nat) -> fn(Nat) ->{e} Nat {
    f
}

fn main() ->{Io} Nil {
    let f = identity(fn(x: Nat) { x + 10 })
    let result = f(5)
    print_line(Nat::to_string(result))
}
"#;
    // identity returns the lambda, f(5) = 5 + 10 = 15
    assert_native_output("returned_closure_call.trb", code, "15");
}

/// Test a generic function specialized at function types that differ only in
/// their effect rows.
///
/// `keep` is used at `a = fn() ->{Ask} Nat` and `a = fn() ->{Tell} Nat`. Each
/// function type selects its own specialization.
#[test]
fn test_specializations_distinguish_function_argument_ability_rows() {
    let code = r#"use std::io::{Io, print_line}

ability Ask {
    op ask() -> Nat
}

ability Tell {
    op tell() -> Nat
}

fn keep(value: a) -> a { value }

fn ask() ->{Ask} Nat { Ask::ask() }

fn tell() ->{Tell} Nat { Tell::tell() }

fn asked() -> Nat {
    let held = keep(ask)
    handle held() {
        do result { result }
        op Ask::ask() { resume 40 }
    }
}

fn told() -> Nat {
    let held = keep(tell)
    handle held() {
        do result { result }
        op Tell::tell() { resume 2 }
    }
}

fn main() ->{Io} Nil {
    print_line(Nat::to_string(asked() + told()))
}
"#;
    // ask() → 40, tell() → 2
    assert_native_output("ability_row_specializations.trb", code, "42");
}

/// A method whose receiver is typed after solving runs under the handler of
/// the effects it performs, with the callback it is given.
#[test]
fn test_ability_row_method_resolved_after_solving() {
    let code = r#"
use std::io::{Io, print_line}

ability Ask {
    op ask() -> Nat
}

struct Box { value: Nat }

fn bump(n: Nat) ->{Ask} Nat { n + Ask::ask() }

fn each(b: Box, g: fn(Nat) ->{e} Nat) ->{e} Nat { g(b.value) }

fn asks(b: Box) ->{Ask} Nat { b.value + Ask::ask() }

fn run() ->{Ask} Nat {
    let callback = fn(c) { c.each(bump) }
    let direct = fn(c) { c.asks() }
    callback(Box { value: 1 }) + direct(Box { value: 1 })
}

fn main() ->{Io} Nil {
    let total = handle run() {
        do result { result }
        op Ask::ask() { resume 20 }
    }
    print_line(Nat::to_string(total))
}
"#;
    // each(bump) → 1 + 20, asks() → 1 + 20
    assert_native_output("ability_row_late_method.trb", code, "42");
}

/// Passing a named effectful function to a callee with a single row tail and
/// then performing the same ability in the handled body (#1332).
#[test]
fn test_named_effectful_function_then_same_ability_in_handled_body() {
    let code = r#"use std::io::{Io, print_line}

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

fn apply(f: fn() ->{e} a) ->{e} a { f() }

fn inc() ->{State(Int)} Nil { State::set(State::get() + +1) }

fn main() ->{Io} Nil {
    let n = run_state(fn() {
        apply(inc)
        State::get()
    }, +0)
    print_line(Int::to_string(n))
}
"#;
    assert_output_on_both_targets("ability_row_named_effectful_argument.trb", code, "1");
}

/// A method call deferred until its receiver's type is solved keeps the
/// callee's effect row (#1337).
#[test]
fn test_deferred_method_call_keeps_callee_effect() {
    let code = r#"use std::io::{Io, print_line}

struct Name { text: String }

fn id(x: a) -> a { x }

fn shout(n: Name) ->{Io} Name {
    print_line("shout")
    n
}

fn main() ->{Io} Nil {
    let name = id(Name { text: "io" }).shout()
    print_line(name.text)
}
"#;
    assert_output_on_both_targets("ability_row_deferred_method_effect.trb", code, "shout\nio");
}
