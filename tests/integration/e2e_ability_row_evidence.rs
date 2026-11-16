//! Handlers are selected by row position (`design/cps-effects.md`,
//! "Row 위치에 따른 evidence 선택"): a handler handles only the instances
//! explicit in the row at its handle site, handler arms run outside their
//! handle, and a resumed computation keeps the selections of its other
//! handles but runs without the handle whose arm resumed it. Each program
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

/// A callback's explicit instance that the caller's row argument also names
/// goes to the handler of the callee whose row names it, for a lambda and for
/// a named function.
#[test]
fn test_callback_instance_also_named_by_the_row_argument() {
    let code = r#"
ability Logger {
    fn log(n: Int) -> Int
}

fn use_cb(f: fn() ->{e, Logger} Int) ->{e} Int {
    handle f() {
        do r { r }
        fn Logger::log(n) { n + +1 }
    }
}

fn outer_logger(comp: fn() ->{e, Logger} a) ->{e} a {
    handle comp() {
        do r { r }
        fn Logger::log(n) { n + +100 }
    }
}

fn five() ->{Logger} Int { Logger::log(+5) }

fn main() ->{Io} Nil {
    show(outer_logger(fn() { use_cb(fn() { Logger::log(+5) }) }))
    show(outer_logger(fn() { use_cb(five) }))
}
"#;
    assert_program("instance_named_by_row_argument.trb", code, "6\n6");
}

/// An `op` arm's own operations go to the handler outside its handle, and so do
/// those of the computation it resumes: `resume` leaves the arm's handle, so
/// only the first operation of the body reaches it. The outer handler
/// reinstalls itself while the arm runs.
#[test]
fn test_op_arm_and_resumed_operations_reach_the_outer_handler() {
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
    // set(5) reaches the arm, which stores 10 outside and resumes. Both get()s
    // of the resumed body then reach the outer handler, which reads 10 each:
    // 10 * 10 + 10 = 110, and the state stays 10.
    assert_program("op_arm_reaches_outer_handler.trb", code, "110010");
}

/// A `do` arm runs outside its handle: an operation of the handled instance
/// reaches the outer handler. It applies only when the body returns under the
/// handle, not to the result of a computation an `op` arm resumed.
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
    let returned = run_state(fn() {
        let inner = finishing(fn() { +21 })
        inner * +100 + State::get()
    }, +0)
    show(returned)
    let resumed = run_state(fn() {
        let inner = finishing(fn() { State::get() * +3 })
        inner * +100 + State::get()
    }, +0)
    show(resumed)
}
"#;
    // The first body returns 21 under the handle: the `do` arm stores it
    // outside and returns 22, so 22 * 100 + 21. The second body performs
    // get(), whose arm resumes with 7; the body then returns 21 outside the
    // handle, so the `do` arm does not run: 21 * 100 + 0.
    assert_program("do_arm_reaches_outer_handler.trb", code, "2221\n2100");
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

/// A resumed computation keeps the handles between the operation and its
/// handler, in both nesting orders: a handler that reinstalls itself resumes
/// the body with the other handler still around the operation.
#[test]
fn test_arm_body_resume_keeps_nested_handlers() {
    let code = r#"
ability Reader {
    op ask() -> Int
}

fn with_reader(comp: fn() ->{e, Reader} a, value: Int) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { with_reader(fn() { resume value }, value) }
    }
}

fn with_state(comp: fn() ->{e, State(Int)} a) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { with_state(fn() { resume +3 }) }
        op State::set(v) { with_state(fn() { resume Nil }) }
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

/// A resume written in a handle body nested in an arm runs the resumed
/// computation where the resume is written, but the nested handler's own
/// `resume` leaves that handle. The nested handler answers the resume argument
/// only; the resumed computation's operations reach the handler around the
/// nested handle, and the nested `do` arm does not run.
#[test]
fn test_resume_in_a_handle_nested_in_an_arm() {
    let code = r#"
ability Reader {
    op ask() -> Int
}

fn with_reader(comp: fn() ->{e, Reader} a, value: Int) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { with_reader(fn() { resume value }, value) }
    }
}

fn with_state(
    comp: fn() ->{e, State(Int), Reader} Int
) ->{e, State(Int), Reader} Int {
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
    show(run_state(fn() {
        with_reader(fn() {
            with_state(fn() { State::get() * +10 + Reader::ask() })
        }, +4)
    }, +0))
}
"#;
    // get() reaches the arm, whose nested body performs ask(): the nested arm
    // resumes with 9 and leaves the nested handle, so `resume 9` of the outer
    // arm runs outside it, and `do inner` never applies. The resumed
    // computation's ask() reaches `with_reader` (4), and its raw result is
    // the result: 9 * 10 + 4.
    assert_program("resume_in_nested_handle.trb", code, "94");
}

/// A handler that reinstalls itself from an arm reaches an outer handler, which
/// reinstalls itself too, from the arm and from the resumed body in turn.
#[test]
fn test_handler_arm_follows_a_reinstalled_outer_handler() {
    let code = r#"
ability Reader {
    op ask() -> Int
}

fn with_ask(comp: fn() ->{e, State(Int), Reader} Int) ->{e, State(Int)} Int {
    handle comp() {
        do result { result }
        op Reader::ask() { with_ask(fn() { resume State::get() + +1 }) }
    }
}

fn inline() ->{State(Int)} Int {
    with_ask(fn() {
        let a = Reader::ask()
        State::set(a)
        let b = Reader::ask()
        State::set(State::get() + b)
        State::get() + Reader::ask()
    })
}

fn main() ->{Io} Nil {
    show(run_state(fn() { inline() }, +1))
}
"#;
    // ask() reads the state and adds 1: 2, then 3 (state 5), then 6.
    assert_program("inline_handle_body.trb", code, "11");
}

/// A resume in a handle body nested in an arm uses the arm's evidence as it
/// is at the resume, after an operation of the nested body made an outer
/// handler reinstall itself. The arm reinstalls its own handler around the
/// nested handle, so the next `tick` reaches that handler.
#[test]
fn test_nested_resume_after_an_outer_handler_reinstalled_itself() {
    let code = r#"
ability Tick {
    op tick() -> Int
}

ability Reader {
    op ask() -> Int
}

fn with_reader(comp: fn() ->{e, Reader} a, value: Int) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { with_reader(fn() { resume value }, value) }
    }
}

fn with_tick(
    comp: fn() ->{e, Tick, State(Int), Reader} Int
) ->{e, State(Int), Reader} Int {
    handle comp() {
        do result { result }
        op Tick::tick() {
            with_tick(fn() {
                handle {
                    let s = State::get()
                    resume s + Reader::ask()
                } {
                    do inner { inner }
                    op Reader::ask() { resume +9 }
                }
            })
        }
    }
}

fn body() ->{Tick, State(Int), Reader} Int {
    let t = Tick::tick()
    State::set(t)
    State::get() + Tick::tick()
}

fn main() ->{Io} Nil {
    show(run_state(fn() {
        with_reader(fn() { with_tick(body) }, +100)
    }, +1))
}
"#;
    // The first tick() reads the state (1), the nested arm answers ask() with 9
    // and leaves the nested handle: tick() is 10, stored. The resumed body runs
    // under the reinstalled `with_tick`, so the second tick() reaches it: the
    // state read there is 10, giving 19, and the result is 10 + 19. No ask()
    // reaches `with_reader`.
    assert_program("nested_resume_after_reinstall.trb", code, "29");
}

/// A handle nested in an arm may handle an instance the arm's row also names.
/// The nested handler's own `resume` leaves that handle, so the resumed
/// computation's operations of the instance reach the handler the nested
/// handle hid, also after an outer handler reinstalled itself in between.
#[test]
fn test_nested_handler_does_not_handle_the_resumed_computation() {
    let code = r#"
ability Tick {
    op tick() -> Int
}

ability Cell(c) {
    op read() -> c
    op write(value: c) -> Nil
}

fn run_cell(comp: fn() ->{e, Cell(c)} a, init: c) ->{e} a {
    handle comp() {
        do result { result }
        op Cell::read() { run_cell(fn() { resume init }, init) }
        op Cell::write(v) { run_cell(fn() { resume Nil }, v) }
    }
}

fn with_tick(
    comp: fn() ->{e, Tick, State(Int), Cell(Int)} Int
) ->{e, State(Int), Cell(Int)} Int {
    handle comp() {
        do result { result }
        op Tick::tick() {
            with_tick(fn() {
                handle {
                    let c = Cell::read()
                    let s = State::get()
                    resume s + c
                } {
                    do inner { inner }
                    op State::get() { resume +100 }
                    op State::set(v) { resume Nil }
                }
            })
        }
    }
}

fn main() ->{Io} Nil {
    show(run_cell(fn() {
        run_state(fn() {
            with_tick(fn() {
                let t = Tick::tick()
                t + Cell::read() + State::get()
            })
        }, +1)
    }, +5))
}
"#;
    // tick() is the nested state (100) plus the cell (5). The nested handler
    // is gone from the resumed computation, which reads the cell (5) and the
    // outer state (1), not the nested one (100): 105 + 5 + 1.
    assert_program("nested_resume_restores_hidden_handler.trb", code, "111");
}

/// A callee whose row is a union of tails gives each tail its own handlers.
/// The handler a caller installs for one tail does not handle the operations
/// of a callback that takes the instance through another tail.
#[test]
fn test_each_tail_of_a_callee_takes_its_own_handler() {
    let code = r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    g()
}

fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        both(fn() { State::set(State::get() + +1) }, h)
        State::get()
    }, +0)
}

fn main() ->{Io} Nil {
    let outer = run_state(fn() {
        let calls = count_calls(fn() { State::set(State::get() + +10) })
        calls * +100 + State::get()
    }, +0)
    show(outer)
}
"#;
    // The internal handler counts the 1 call of the first callback; the
    // second callback adds 10 to the caller's state.
    assert_program("each_tail_takes_its_own_handler.trb", code, "110");
}

/// The same selection when the multi-tail call is made by a function that
/// names the instance in its own row beside its tail.
#[test]
fn test_each_tail_takes_its_own_handler_under_an_explicit_row() {
    let code = r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    g()
}

fn inc() ->{State(Int)} Nil {
    State::set(State::get() + +1)
}

fn counted(h: fn() ->{t} Nil) ->{t, State(Int)} Int {
    both(inc, h)
    State::get()
}

fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() { counted(h) }, +0)
}

fn main() ->{Io} Nil {
    let outer = run_state(fn() {
        let calls = count_calls(fn() { State::set(State::get() + +10) })
        calls * +100 + State::get()
    }, +0)
    show(outer)
}
"#;
    assert_program("each_tail_under_explicit_row.trb", code, "110");
}

/// Tails that are filled with different instances share one evidence: each
/// callback reaches the handler of its own instance.
#[test]
fn test_tails_with_different_instances_keep_their_handlers() {
    let code = r#"
ability Reader {
    op ask() -> Int
}

fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Int) ->{e1, e2} Int {
    f()
    g()
}

fn inc() ->{State(Int)} Nil {
    State::set(State::get() + +1)
}

fn with_reader(comp: fn() ->{e, Reader} a) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { with_reader(fn() { resume +7 }) }
    }
}

fn counted(h: fn() ->{t} Int) ->{t, State(Int)} Int {
    let asked = both(inc, h)
    asked * +10 + State::get()
}

fn count_calls(h: fn() ->{t} Int) ->{t} Int {
    run_state(fn() { counted(h) }, +0)
}

fn main() ->{Io} Nil {
    show(with_reader(fn() { count_calls(fn() { Reader::ask() }) }))
}
"#;
    assert_program("tails_with_different_instances.trb", code, "71");
}

/// Each call a multi-tail callee makes selects its tail again, also after the
/// handlers of both tails resumed the callee.
#[test]
fn test_tails_keep_their_handlers_across_resumes() {
    let code = r#"
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

fn main() ->{Io} Nil {
    let outer = run_state(fn() {
        let calls = count_calls(fn() { State::set(State::get() + +10) })
        calls * +100 + State::get()
    }, +0)
    show(outer)
}
"#;
    assert_program("tails_across_resumes.trb", code, "220");
}

/// A multi-tail callee that calls another one passes each of its tails on in
/// the position the inner callee declares.
#[test]
fn test_tails_are_forwarded_to_another_multi_tail_callee() {
    let code = r#"
fn both(f: fn() ->{e1} Nil, g: fn() ->{e2} Nil) ->{e1, e2} Nil {
    f()
    g()
}

fn swapped(f: fn() ->{a} Nil, g: fn() ->{b} Nil) ->{a, b} Nil {
    both(g, f)
}

fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        swapped(h, fn() { State::set(State::get() + +1) })
        State::get()
    }, +0)
}

fn main() ->{Io} Nil {
    let outer = run_state(fn() {
        let calls = count_calls(fn() { State::set(State::get() + +10) })
        calls * +100 + State::get()
    }, +0)
    show(outer)
}
"#;
    assert_program("tails_forwarded.trb", code, "110");
}

/// A callback whose row names an instance beside a tail takes the callee's
/// own handler of that instance on top of the tail's handlers.
#[test]
fn test_selected_tail_takes_the_callees_explicit_handler() {
    let code = r#"
fn stateful(
    f: fn() ->{e1, State(Int)} Nil,
    g: fn() ->{e2} Nil
) ->{e1, e2, State(Int)} Nil {
    f()
    g()
}

fn count_calls(h: fn() ->{t} Nil) ->{t} Int {
    run_state(fn() {
        stateful(fn() { State::set(State::get() + +1) }, h)
        State::get()
    }, +0)
}

fn main() ->{Io} Nil {
    let outer = run_state(fn() {
        let calls = count_calls(fn() { State::set(State::get() + +10) })
        calls * +100 + State::get()
    }, +0)
    show(outer)
}
"#;
    assert_program("selected_tail_explicit_handler.trb", code, "110");
}
