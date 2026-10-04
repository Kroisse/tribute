//! Handlers are selected by row position (`new-plans/cps-effects.md`,
//! "Row 위치에 따른 evidence 선택"): a handler handles only the instances
//! explicit in the row at its handle site, handler arms run outside their
//! handle, and a resumed computation keeps those selections. Each program
//! runs on the native and Wasm targets.

use crate::common::assert_output_on_both_targets;

const PRELUDE: &str = r#"use std::io::{Io, print_line}

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

fn show(n: Int) ->{Io} Nil {
    print_line(Int::to_string(n))
}
"#;

fn assert_program(name: &str, body: &str, expected: &str) {
    assert_output_on_both_targets(name, &format!("{PRELUDE}{body}"), expected);
}

/// A handler installed by an effect-polymorphic function does not handle the
/// operations its callback performs through the row tail.
#[test]
fn test_callback_operations_pass_an_internal_handler() {
    let code = r#"
fn twice_counted(f: fn() ->{e} Nil) ->{e} Int {
    run_state(fn() {
        f()
        State::set(State::get() + +1)
        f()
        State::set(State::get() + +1)
        State::get()
    }, +0)
}

fn main() ->{Io} Nil {
    let outer = run_state(fn() {
        let calls = twice_counted(fn() { State::set(State::get() + +10) })
        calls * +100 + State::get()
    }, +0)
    show(outer)
}
"#;
    // The internal handler counts 2 calls; the callback adds 10 twice to the
    // caller's state.
    assert_program("callback_passes_internal_handler.trb", code, "220");
}

/// A callee that names an instance and also takes it through its row tail
/// receives the caller's handler in both positions.
#[test]
fn test_callee_takes_one_handler_in_two_positions() {
    let code = r#"
fn both(g: fn() ->{e} Nil) ->{e, State(Int)} Nil {
    g()
    State::set(State::get() + +1)
}

fn caller() ->{State(Int)} Nil {
    both(fn() { State::set(State::get() + +10) })
}

fn main() ->{Io} Nil {
    let result = run_state(fn() {
        caller()
        State::get()
    }, +0)
    show(result)
}
"#;
    assert_program("handler_in_two_positions.trb", code, "11");
}

/// An `op` arm's own operations go to the handler outside its handle, and the
/// computation it resumes stays under the arm's handle. The outer handler
/// reinstalls itself while the arm runs.
#[test]
fn test_op_arm_operations_reach_the_outer_handler() {
    let code = r#"
fn counter(comp: fn() ->{e, State(Int)} a) ->{e, State(Int)} a {
    handle comp() {
        do result { result }
        op State::get() { resume State::get() + +1 }
        op State::set(v) { resume State::set(v * +2) }
    }
}

fn main() ->{Io} Nil {
    let result = run_state(fn() {
        let seen = counter(fn() {
            State::set(+5)
            State::get() * +10 + State::get()
        })
        seen * +1000 + State::get()
    }, +0)
    show(result)
}
"#;
    // set(5) stores 10 outside; each get() reads 10 and adds 1.
    assert_program("op_arm_reaches_outer_handler.trb", code, "121010");
}

/// A `do` arm runs outside its handle: an operation of the handled instance
/// reaches the outer handler.
#[test]
fn test_do_arm_operations_reach_the_outer_handler() {
    let code = r#"
fn finishing(comp: fn() ->{e, State(Int)} Int) ->{e, State(Int)} Int {
    handle comp() {
        do result {
            State::set(result)
            State::get() + +1
        }
        op State::get() { resume +7 }
        op State::set(v) { resume Nil }
    }
}

fn main() ->{Io} Nil {
    let result = run_state(fn() {
        let inner = finishing(fn() {
            State::set(+100)
            State::get() * +3
        })
        inner * +100 + State::get()
    }, +0)
    show(result)
}
"#;
    // The body yields 21; the `do` arm stores it outside and returns 22.
    assert_program("do_arm_reaches_outer_handler.trb", code, "2221");
}

/// A `fn` arm's own operations go to the handler outside its handle.
#[test]
fn test_fn_arm_operations_reach_the_outer_handler() {
    let code = r#"
ability Logger {
    fn log(n: Int) -> Int
}

fn outer_logger(comp: fn() ->{e, Logger} a) ->{e} a {
    handle comp() {
        do result { result }
        fn Logger::log(n) { n + +1 }
    }
}

fn doubling(comp: fn() ->{e, Logger} a) ->{e, Logger} a {
    handle comp() {
        do result { result }
        fn Logger::log(n) { Logger::log(n * +2) }
    }
}

fn main() ->{Io} Nil {
    show(outer_logger(fn() { doubling(fn() { Logger::log(+5) }) }))
}
"#;
    assert_program("fn_arm_reaches_outer_handler.trb", code, "11");
}

/// A resume from an arm body rebuilds each handle between the operation and
/// its handler, in both nesting orders.
#[test]
fn test_arm_body_resume_keeps_nested_handlers() {
    let code = r#"
ability Reader {
    op ask() -> Int
}

fn with_reader(comp: fn() ->{e, Reader} a, value: Int) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { resume value }
    }
}

fn with_state(comp: fn() ->{e, State(Int)} a) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { resume +3 }
        op State::set(v) { resume Nil }
    }
}

fn body() ->{State(Int), Reader} Int {
    let a = Reader::ask()
    let b = State::get()
    let c = Reader::ask()
    State::set(c)
    a * +100 + b * +10 + State::get()
}

fn main() ->{Io} Nil {
    show(with_reader(fn() { with_state(fn() { body() }) }, +4))
    show(with_state(fn() { with_reader(fn() { body() }, +4) }))
}
"#;
    assert_program("arm_body_resume_nested.trb", code, "433\n433");
}

/// A handler that reinstalls itself from an arm leaves the handle whose arm
/// ran with the continuation of the resumed computation, so a handler
/// reinstalled further out is still reachable from it.
#[test]
fn test_alternating_operations_of_two_reinstalling_handlers() {
    let code = r#"
ability Reader(r) {
    op ask() -> r
}

fn run_reader(comp: fn() ->{e, Reader(r)} a, value: r) ->{e} a {
    handle comp() {
        op Reader::ask() { run_reader(fn() { resume value }, value) }
    }
}

fn use_both() ->{State(Int), Reader(Int)} Int {
    let n = State::get()
    State::set(n + Reader::ask())
    State::get() * +100 + Reader::ask()
}

fn main() ->{Io} Nil {
    show(run_state(fn() { run_reader(fn() { use_both() }, +7) }, +5))
    show(run_reader(fn() { run_state(fn() { use_both() }, +5) }, +7))
}
"#;
    assert_program("alternating_two_abilities.trb", code, "1207\n1207");
}

/// An arm that does not resume leaves its handle through the continuation of
/// the layer it ran in, after an outer handler reinstalled itself.
#[test]
fn test_abort_after_an_outer_handler_reinstalled_itself() {
    let code = r#"
ability Abort {
    op abort(code: Int) -> Never
}

fn run_abort(comp: fn() ->{e, Abort} Int) ->{e} Int {
    handle comp() {
        do result { result }
        op Abort::abort(code) { code }
    }
}

fn main() ->{Io} Nil {
    show(run_state(fn() {
        let r = run_abort(fn() {
            State::set(+5)
            Abort::abort(State::get() + +1)
        })
        r * +10 + State::get()
    }, +0))
    show(run_abort(fn() {
        run_state(fn() {
            State::set(+5)
            Abort::abort(State::get() + +1)
        }, +0)
    }))
}
"#;
    assert_program("abort_after_reinstall.trb", code, "65\n6");
}

/// A resume written in a handle body nested in an arm resumes under the
/// arm's own handle: the nested handler handles the nested body's operations
/// but not those of the resumed computation.
#[test]
fn test_resume_in_a_handle_nested_in_an_arm() {
    let code = r#"
ability Reader {
    op ask() -> Int
}

fn with_reader(comp: fn() ->{e, Reader} a, value: Int) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { resume value }
    }
}

fn with_state(comp: fn() ->{e, State(Int), Reader} Int) ->{e, Reader} Int {
    handle comp() {
        do result { result }
        op State::get() {
            handle resume Reader::ask() {
                do inner { inner + +1000 }
                op Reader::ask() { resume +9 }
            }
        }
        op State::set(v) { resume Nil }
    }
}

fn main() ->{Io} Nil {
    show(with_reader(fn() {
        with_state(fn() { State::get() * +10 + Reader::ask() })
    }, +4))
}
"#;
    // The nested handler answers the resume argument (9); the resumed
    // computation's `ask` reaches the outer handler (4).
    assert_program("resume_in_nested_handle.trb", code, "1094");
}
