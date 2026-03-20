//! End-to-end tests for effect row polymorphism runtime behavior (#502).
//!
//! These tests verify that row-polymorphic effect variables are correctly
//! instantiated and propagated at runtime — higher-order functions accepting
//! effectful callbacks, pure-for-effectful substitution, and unification
//! across multiple call sites.

mod common;

use common::assert_native_output;

// =============================================================================
// Effect Row Polymorphism Tests
// =============================================================================

/// Test row-polymorphic higher-order function with effectful callback.
///
/// `apply` accepts `fn(Nat) ->{e} Nat` and calls it. When called inside
/// a `State` handler, `e` is instantiated to `{State(Nat)}`, so the callback
/// can perform State operations.
#[test]
#[ignore = "row-polymorphic effectful callbacks crash at runtime (#502)"]
fn test_effect_row_poly_higher_order_function() {
    let code = r#"ability State(s) {
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

fn main() {
    let result = run_state(fn() {
        apply(fn(x: Nat) {
            State::set(x)
            State::get()
        }, 7)
    }, 0)
    __tribute_print_nat(result)
}
"#;
    // apply calls the callback with x=7, which does set(7), get() → 7
    assert_native_output("effect_row_poly_higher_order.trb", code, "7");
}

/// Test row variable instantiated to multiple concrete abilities.
///
/// `apply` accepts `fn() ->{e} Nat`. The callback uses both State and Reader,
/// so `e` is instantiated to `{State(Nat), Reader(Nat)}`.
#[test]
#[ignore = "row-polymorphic effectful callbacks crash at runtime (#502)"]
fn test_effect_row_poly_multiple_abilities() {
    let code = r#"ability State(s) {
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

fn main() {
    let result = run_reader(fn() {
        run_state(fn() {
            apply(fn() {
                let config = Reader::ask()
                State::set(config)
                State::get()
            })
        }, 0)
    }, 42)
    __tribute_print_nat(result)
}
"#;
    // callback: ask() → 42, set(42), get() → 42
    assert_native_output("effect_row_poly_multiple_abilities.trb", code, "42");
}

/// Test pure function passed to a row-polymorphic parameter.
///
/// `apply` accepts `fn(Nat) ->{e} Nat`. When called with a pure lambda,
/// `e` is instantiated to `{}` (empty row), so no handler is needed.
#[test]
fn test_effect_row_poly_pure_for_effectful() {
    let code = r#"fn apply(f: fn(Nat) ->{e} Nat, x: Nat) ->{e} Nat {
    f(x)
}

fn main() {
    let result = apply(fn(x: Nat) { x + 10 }, 5)
    __tribute_print_nat(result)
}
"#;
    // pure lambda: 5 + 10 = 15, no handler needed
    assert_native_output("effect_row_poly_pure.trb", code, "15");
}

/// Test same row-polymorphic function called with different effect instantiations.
///
/// `apply` is called twice: once where `e = {State(Nat)}` and once where
/// `e = {Reader(Nat)}`. Each call site independently instantiates the row variable.
#[test]
#[ignore = "row-polymorphic effectful callbacks crash at runtime (#502)"]
fn test_effect_row_poly_unification_across_call_sites() {
    let code = r#"ability State(s) {
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

fn main() {
    // call 1: e = {State(Nat)}
    let a = run_state(fn() {
        apply(fn() { State::get() })
    }, 10)
    __tribute_print_nat(a)

    // call 2: e = {Reader(Nat)}
    let b = run_reader(fn() {
        apply(fn() { Reader::ask() })
    }, 20)
    __tribute_print_nat(b)
}
"#;
    // call 1: get() with State(init=10) → 10
    // call 2: ask() with Reader(value=20) → 20
    assert_native_output("effect_row_poly_unification.trb", code, "10\n20");
}
