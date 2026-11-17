//! End-to-end tests for proper tail calls with `become`.
//!
//! Each program recurses far deeper than a native or Wasm stack could hold
//! if every call kept a frame, so it only finishes when `become` transfers
//! the caller's frame. They print only through `std::io` and run on both
//! the native and Wasm targets.

use crate::common;

use common::assert_output_on_both_targets;

/// Deep enough that a frame per call overflows any default stack.
const DEPTH: &str = "10000000";

/// A self tail call in a case arm, between Direct callables.
#[test]
fn test_become_self_recursion_runs_in_constant_stack() {
    let code = r#"
fn count(n: Nat, acc: Nat) -> Nat {
    case n {
        0 -> acc
        _ -> become count(n - 1, acc + 1)
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(count(DEPTH, 0).to_string)
}
"#
    .replace("DEPTH", DEPTH);
    assert_output_on_both_targets("become_self.trb", &code, DEPTH);
}

/// Mutual tail calls through a nested block in a case arm.
#[test]
fn test_become_mutual_recursion_runs_in_constant_stack() {
    let code = r#"
fn is_even(n: Nat) -> Nat {
    case n {
        0 -> 1
        _ -> { become is_odd(n - 1) }
    }
}

fn is_odd(n: Nat) -> Nat {
    case n {
        0 -> 0
        _ -> become is_even(n - 1)
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(is_even(DEPTH).to_string)
}
"#
    .replace("DEPTH", DEPTH);
    assert_output_on_both_targets("become_mutual.trb", &code, "1");
}

/// A tail call of a callable value, and a lambda whose body tail calls a
/// named function.
#[test]
fn test_become_callable_value_runs_in_constant_stack() {
    let code = r#"
fn step(f: fn(Nat, fn(Nat) ->{} Nat) ->{} Nat, n: Nat) -> Nat {
    case n {
        0 -> 7
        _ -> become f(n - 1, fn(m: Nat) -> Nat { become step(f, m) })
    }
}

fn apply(n: Nat, k: fn(Nat) ->{} Nat) -> Nat {
    become k(n)
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(step(apply, DEPTH).to_string)
}
"#
    .replace("DEPTH", DEPTH);
    assert_output_on_both_targets("become_indirect.trb", &code, "7");
}

/// Tail calls inside an effectful callable keep the caller's continuation,
/// and a tail call to a value-returning callee completes through it.
#[test]
fn test_become_under_a_handler_runs_in_constant_stack() {
    let code = r#"
ability Tick {
    op tick() -> Nil
}

fn finish(acc: Nat) -> Nat { acc }

fn loop(n: Nat, acc: Nat) ->{Tick} Nat {
    case n {
        0 -> become finish(acc)
        50000 -> {
            Tick::tick()
            become loop(n - 1, acc + 1)
        }
        _ -> become loop(n - 1, acc + 1)
    }
}

fn count_ticks(comp: fn() ->{e, Tick} Nat) ->{e} Nat {
    handle comp() {
        do result { result }
        op Tick::tick() { count_ticks(fn() { resume Nil }) }
    }
}

fn main() ->{std::io::Io} Nil {
    let total = count_ticks(fn() loop(100000, 0))
    std::io::print_line(total.to_string)
}
"#;
    assert_output_on_both_targets("become_handler.trb", code, "100000");
}

/// A handler that installs itself again around each `resume` handles every
/// operation of a loop of tail calls.
#[test]
fn test_reinstalled_handler_handles_each_operation_of_a_tail_loop() {
    let code = r#"
ability Tick {
    op tick() -> Nil
}

fn loop(n: Nat, acc: Nat) ->{Tick} Nat {
    case n {
        0 -> acc
        _ -> {
            Tick::tick()
            become loop(n - 1, acc + 1)
        }
    }
}

fn count_ticks(comp: fn() ->{e, Tick} Nat) ->{e} Nat {
    handle comp() {
        do result { result }
        op Tick::tick() { count_ticks(fn() { resume Nil }) }
    }
}

fn main() ->{std::io::Io} Nil {
    let total = count_ticks(fn() loop(2000, 0))
    std::io::print_line(total.to_string)
}
"#;
    assert_output_on_both_targets("become_reinstalled_handler.trb", code, "2000");
}

const STATE_HANDLER: &str = r#"
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
"#;

/// A callee that takes each of the caller's row tails in the same order
/// keeps the evidence unchanged, so a self tail call loops without a frame.
#[test]
fn test_become_with_the_callers_own_tails_loops() {
    let code = format!(
        "{STATE_HANDLER}{}",
        r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil, n: Int) ->{e1, e2} Nil {
    f()
    g()
    case n == +0 {
        True -> Nil
        False -> become both(f, g, n - +1)
    }
}

fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        both(fn() { State::set(State::get() + +1) }, h, +999)
        State::get()
    }, +0)
}

fn main() ->{std::io::Io} Nil {
    let outer = run_state(fn() {
        let calls = count_calls(fn() { State::set(State::get() + +10) })
        std::io::print_line(calls.to_string)
        State::get()
    }, +0)
    std::io::print_line(outer.to_string)
}
"#
    );
    assert_output_on_both_targets("become_own_tails.trb", &code, "1000\n10000");
}

/// With an ordinary last call, the callee's handlers are those the call
/// selects.
#[test]
fn test_last_call_selects_the_handlers_of_each_tail() {
    let code = format!(
        "{STATE_HANDLER}{}",
        r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    g()
    f()
    g()
}

fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        both(fn() { State::set(State::get() + +1) }, h)
        State::get()
    }, +0)
}

fn main() ->{std::io::Io} Nil {
    let total = run_state(fn() {
        let calls = count_calls(fn() { State::set(State::get() + +10) })
        calls * +100 + State::get()
    }, +0)
    std::io::print_line(total.to_string)
}
"#
    );
    assert_output_on_both_targets("become_plain_last_call.trb", &code, "220");
}

/// An `fn` arm answers an `op` operation performed in every iteration of a
/// tail loop. The loop runs in constant stack, both directly under the
/// handle and under an effect-polymorphic handler beneath another ability's
/// handler.
#[test]
fn test_fn_arm_for_an_op_operation_answers_a_tail_loop_in_constant_stack() {
    let code = r#"
ability Tick {
    op next() -> Int
}

ability Ping {
    op ping() -> Nil
}

fn loop_(n: Nat, acc: Int) ->{Tick} Int {
    case n {
        0 -> acc
        _ -> become loop_(n - 1, acc + Tick::next())
    }
}

fn start(n: Nat) ->{Tick, Ping} Int {
    Ping::ping()
    loop_(n, +0)
}

fn run_direct(n: Nat) ->{} Int {
    handle loop_(n, +0) {
        do result { result }
        fn Tick::next() { +1 }
    }
}

fn run(comp: fn() ->{e, Tick} Int) ->{e} Int {
    handle comp() {
        do result { result }
        fn Tick::next() { +1 }
    }
}

fn pinged(comp: fn() ->{e, Ping} Int) ->{e} Int {
    handle comp() {
        do result { result }
        op Ping::ping() { pinged(fn() { resume Nil }) }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(run_direct(100000).to_string)
    std::io::print_line(pinged(fn() { run(fn() { start(100000) }) }).to_string)
}
"#;
    assert_output_on_both_targets("become_fn_arm_for_op.trb", code, "100000\n100000");
}
