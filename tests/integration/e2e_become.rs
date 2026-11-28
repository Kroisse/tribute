//! End-to-end tests for proper tail calls with `become`.
//!
//! Each program recurses far deeper than a native or Wasm stack could hold
//! if every call kept a frame, so it only finishes when `become` transfers
//! the caller's frame. They print only through `std::io` and run on both
//! the native and Wasm targets.

use crate::common;

use common::{
    assert_output_on_both_targets, assert_target_outputs, compile_and_run_native_asan,
    compile_and_run_wasm,
};

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

/// A lambda and a named function value whose bodies loop through `become`,
/// in a program where no transfer hands a frame to a callable value: their
/// `Cps` entries are adapters, and their loops still run in constant stack.
#[test]
fn test_become_in_callable_values_that_take_no_frame_runs_in_constant_stack() {
    let code = r#"
fn count(n: Nat, acc: Nat) -> Nat {
    case n {
        0 -> acc
        _ -> become count(n - 1, acc + 1)
    }
}

fn start(n: Nat) -> Nat {
    become count(n, 0)
}

fn apply(f: fn(Nat) ->{} Nat, n: Nat) -> Nat {
    f(n)
}

fn main() ->{std::io::Io} Nil {
    let a = apply(fn(n: Nat) -> Nat { become count(n, 0) }, DEPTH)
    std::io::print_line(apply(start, a).to_string)
}
"#
    .replace("DEPTH", DEPTH);
    assert_output_on_both_targets("become_values_without_frames.trb", &code, DEPTH);
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

/// A tail whose row is closed takes a fresh evidence, and the callee pushes
/// the handler it names on the tail it selects: each `State` operation of
/// the loop reaches the one handler, from a closed row and from an open one.
#[test]
fn test_closed_tails_take_the_handlers_their_rows_name() {
    let code = format!(
        "{STATE_HANDLER}{}",
        r#"
fn two(f: fn(Int) ->{State(Int), e1} Int, g: fn() ->{e2} Nil, n: Int) ->{State(Int), e1, e2} Int {
    g()
    State::set(State::get() + +1)
    f(n)
}

fn closed(n: Int) ->{State(Int)} Int {
    case n == +0 {
        True -> State::get()
        False -> two(closed, fn() { Nil }, n - +1)
    }
}

fn open(n: Int, h: fn() ->{r} Nil) ->{State(Int), r} Int {
    case n == +0 {
        True -> State::get()
        False -> two(fn(m) { open(m, h) }, h, n - +1)
    }
}

fn main() ->{std::io::Io} Nil {
    let outer = run_state(fn() {
        let inner = run_state(fn() {
            closed(+50) * +1000 + open(+50, fn() { State::set(State::get() + +1) })
        }, +0)
        std::io::print_line(inner.to_string)
        State::get()
    }, +7)
    std::io::print_line(outer.to_string)
}
"#
    );
    assert_output_on_both_targets("closed_tails.trb", &code, "50150\n7");
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

/// Operations a loop performs in the arms of a handler that installs itself
/// again with `become`: far more than a frame per operation could hold.
const HANDLER_LOOP: &str = "100000";

/// `State` handled by a handler that installs itself again with `become`,
/// in tail position of `run_state`, and a loop that performs `State`.
const STATE_LOOP: &str = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { become run_state(fn() { resume init }, init) }
        op State::set(v) { become run_state(fn() { resume Nil }, v) }
    }
}

fn count(n: Int) ->{State(Int)} Int {
    case n == +0 {
        True -> State::get()
        False -> {
            State::set(State::get() + +1)
            become count(n - +1)
        }
    }
}
"#;

/// A handler that installs itself again with `become` from its `op` arms
/// leaves neither a handler nor a frame per operation (issue #1963). Its
/// handle is in tail position, so the arms hand on the frame `run_state`
/// received; a resume in a lambda's tail position and the call that makes
/// the handle body's result pass their frames on as well.
#[test]
fn test_become_in_op_arms_runs_a_state_loop_in_constant_space() {
    let code = format!(
        "{STATE_LOOP}{}",
        r#"
fn main() ->{std::io::Io} Nil {
    std::io::print_line(run_state(fn() count(+N), +0).to_string)
}
"#
        .replace("+N", &format!("+{HANDLER_LOOP}"))
    );
    assert_output_on_both_targets("become_state_loop.trb", &code, HANDLER_LOOP);
}

/// `become handle ..` and `become resume v` assert their tail positions and
/// run as the handle and the resume alone.
#[test]
fn test_become_handle_and_become_resume_run_the_state_loop() {
    let code = format!(
        "{}{}",
        STATE_LOOP
            .replace("    handle comp() {", "    become handle comp() {")
            .replace("{ resume init }", "{ become resume init }"),
        r#"
fn main() ->{std::io::Io} Nil {
    std::io::print_line(run_state(fn() count(+N), +0).to_string)
}
"#
        .replace("+N", &format!("+{HANDLER_LOOP}"))
    );
    assert!(code.contains("become handle comp()") && code.contains("become resume init"));
    assert_output_on_both_targets("become_handle_resume.trb", &code, HANDLER_LOOP);
}

/// The arms of a tail handle in each kind of callable: `EvidenceDirect`
/// and `Cps` callables, whose callee is a `Cps` instance, and a lambda.
#[test]
fn test_become_in_op_arms_of_each_enclosing_convention() {
    let evidence_direct = format!(
        "{STATE_LOOP}{}",
        r#"
fn count_io(n: Int) ->{State(Int), std::io::Io} Int {
    case n == +0 {
        True -> State::get()
        False -> {
            State::set(State::get() + +1)
            become count_io(n - +1)
        }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(run_state(fn() count_io(+N), +0).to_string)
}
"#
        .replace("+N", &format!("+{HANDLER_LOOP}"))
    );
    assert_output_on_both_targets(
        "become_arm_evidence_direct.trb",
        &evidence_direct,
        HANDLER_LOOP,
    );

    // `run_state` runs at a `Cps` instance under `Log`, an `op` ability.
    let cps = format!(
        "{STATE_LOOP}{}",
        r#"
ability Log {
    op log() -> Nil
}

fn with_log(comp: fn() ->{Log} Int) ->{} Int {
    handle comp() {
        do r { r }
        op Log::log() { +0 }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(with_log(fn() run_state(fn() count(+N), +0)).to_string)
}
"#
        .replace("+N", &format!("+{HANDLER_LOOP}"))
    );
    assert_output_on_both_targets("become_arm_cps.trb", &cps, HANDLER_LOOP);

    let lambda = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{State(Int)} Int, init: Int) ->{} Int {
    let go = fn(c: fn() ->{State(Int)} Int, i: Int) -> Int {
        handle c() {
            do result { result }
            op State::get() { become run_state(fn() { resume i }, i) }
            op State::set(v) { become run_state(fn() { resume Nil }, v) }
        }
    }
    become go(comp, init)
}

fn count(n: Int) ->{State(Int)} Int {
    case n == +0 {
        True -> State::get()
        False -> {
            State::set(State::get() + +1)
            become count(n - +1)
        }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(run_state(fn() count(+N), +0).to_string)
}
"#
    .replace("+N", &format!("+{HANDLER_LOOP}"));
    assert_output_on_both_targets("become_arm_lambda.trb", &lambda, HANDLER_LOOP);
}

/// A `do` arm that loops with `become` in `Direct`, `EvidenceDirect`, and
/// `Cps` callables.
#[test]
fn test_become_in_do_arms_runs_in_constant_space() {
    let program = |row: &str, main: &str| {
        format!(
            r#"
ability Tick {{
    fn tick() -> Nil
}}

ability Log {{
    op log() -> Nil
}}

fn work() ->{{Tick}} Int {{
    Tick::tick()
    +1
}}

fn fin(acc: Int) -> Int {{ acc }}

fn rep(n: Int, acc: Int) ->{{{row}}} Int {{
    handle work() {{
        do r {{
            case n == +0 {{
                True -> become fin(acc)
                False -> become rep(n - +1, acc + r)
            }}
        }}
        fn Tick::tick() {{ Nil }}
    }}
}}

fn main() ->{{std::io::Io}} Nil {{
    {main}
}}
"#
        )
    };
    let direct = format!("std::io::print_line(rep(+{HANDLER_LOOP}, +0).to_string)");
    assert_output_on_both_targets(
        "become_do_arm_direct.trb",
        &program("", &direct),
        HANDLER_LOOP,
    );
    assert_output_on_both_targets(
        "become_do_arm_evidence_direct.trb",
        &program("std::io::Io", &direct),
        HANDLER_LOOP,
    );
    let cps = format!(
        "let v = handle rep(+{HANDLER_LOOP}, +0) {{\n        do r {{ r }}\n        op Log::log() {{ +0 }}\n    }}\n    std::io::print_line(v.to_string)"
    );
    assert_output_on_both_targets("become_do_arm_cps.trb", &program("Log", &cps), HANDLER_LOOP);
}

/// `Step` drops its resumption: each arm hands the frame on to a callable
/// value, `tick` as a named function and as a lambda, which enters it at its
/// `Cps` entry body.
const STEP_LOOP: &str = r#"
ability Step {
    op step() -> Nil
}

fn ping(n: Int) ->{Step} Int {
    case n == +0 {
        True -> +0
        False -> {
            Step::step()
            +1
        }
    }
}

fn drive(k: fn(Int) ->{} Int, n: Int) ->{} Int {
    handle ping(n) {
        do r { r }
        op Step::step() { become k(n - +1) }
    }
}
"#;

/// A `become` of a callable value in an arm enters the value's `Cps` entry
/// with the arm's frame, whatever class the callee type reads.
#[test]
fn test_become_of_a_callable_value_in_an_op_arm_runs_in_constant_space() {
    let named = format!(
        "{STEP_LOOP}{}",
        r#"
fn tick(n: Int) ->{} Int {
    become drive(tick, n)
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(tick(+N).to_string)
}
"#
        .replace("+N", &format!("+{HANDLER_LOOP}"))
    );
    assert_output_on_both_targets("become_arm_func_value.trb", &named, "0");
    let lambda = format!(
        "{STEP_LOOP}{}",
        r#"
fn tick(n: Int) ->{} Int {
    become drive(fn(m: Int) -> Int { become tick(m) }, n)
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(tick(+N).to_string)
}
"#
        .replace("+N", &format!("+{HANDLER_LOOP}"))
    );
    assert_output_on_both_targets("become_arm_lambda_value.trb", &lambda, "0");
}

/// A handle in tail position of `f` whose body result calls `g`, which
/// loops back with `become f`, as a named callee, a named function value,
/// and a lambda. `ROW` is the row of both, so `g` is `Direct` or
/// `EvidenceDirect`.
const BODY_RESULT_LOOPS: [(&str, &str); 3] = [
    (
        "named",
        r#"
fn g(n: Int, acc: Int) ->{ROW} Int {
    case n == +0 {
        True -> acc
        False -> become f(n - +1, acc + +1)
    }
}

fn f(n: Int, acc: Int) ->{ROW} Int {
    handle g(n, acc) {
        do x { x }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(f(+N, +0).to_string)
}
"#,
    ),
    (
        "func_value",
        r#"
fn g(n: Int, acc: Int) ->{ROW} Int {
    case n == +0 {
        True -> acc
        False -> become f(g, n - +1, acc + +1)
    }
}

fn f(k: fn(Int, Int) ->{ROW} Int, n: Int, acc: Int) ->{ROW} Int {
    handle k(n, acc) {
        do x { x }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(f(g, +N, +0).to_string)
}
"#,
    ),
    (
        "lambda",
        r#"
fn g(n: Int, acc: Int) ->{ROW} Int {
    case n == +0 {
        True -> acc
        False -> become f(fn(m: Int, a: Int) ->{ROW} Int { become g(m, a) }, n - +1, acc + +1)
    }
}

fn f(k: fn(Int, Int) ->{ROW} Int, n: Int, acc: Int) ->{ROW} Int {
    handle k(n, acc) {
        do x { x }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(f(g, +N, +0).to_string)
}
"#,
    ),
];

/// The call that makes the body result of a tail handle enters a `Direct`
/// or `EvidenceDirect` callee at its `Cps` entry with the body's frame, so
/// a loop back through `become` leaves no frame.
#[test]
fn test_a_body_result_call_of_a_weaker_callee_leaves_no_frame() {
    for (row_name, row) in [("direct", ""), ("evidence_direct", "std::io::Io")] {
        for (shape, program) in BODY_RESULT_LOOPS {
            let code = program
                .replace("ROW", row)
                .replace("+N", &format!("+{HANDLER_LOOP}"));
            assert_output_on_both_targets(
                &format!("body_result_{row_name}_{shape}.trb"),
                &code,
                HANDLER_LOOP,
            );
        }
    }
}

/// A lambda whose own flow hands no frame on is entered at the body result
/// through its `Cps` adapter.
#[test]
fn test_a_body_result_call_enters_a_value_returning_lambda() {
    let code = r#"
fn f(k: fn(Int) ->{} Int, n: Int) ->{} Int {
    handle k(n) {
        do x { x + +1 }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(f(fn(m: Int) -> Int { m + m }, +20).to_string)
}
"#;
    assert_output_on_both_targets("body_result_adapter.trb", code, "41");
}

/// The call that makes a handle body's result passes the body's frame on
/// when it is the last expression of a block or a `case` arm too.
#[test]
fn test_a_call_in_a_handle_body_result_position_leaves_no_frame() {
    for (name, body) in [
        (
            "block",
            "{\n        let unused = init\n        comp()\n    }",
        ),
        (
            "case",
            "case flag {\n        True -> comp()\n        False -> comp()\n    }",
        ),
    ] {
        let code = format!(
            "{}{}",
            STATE_LOOP.replace(
                "    handle comp() {",
                &format!("    let flag = True\n    handle {body} {{")
            ),
            r#"
fn main() ->{std::io::Io} Nil {
    std::io::print_line(run_state(fn() count(+N), +0).to_string)
}
"#
            .replace("+N", &format!("+{HANDLER_LOOP}"))
        );
        assert_output_on_both_targets(&format!("become_body_{name}.trb"), &code, HANDLER_LOOP);
    }
}

/// A `resume` whose resumed computation takes another selection of
/// handlers keeps its frame, which applies the selection again on the
/// resumed path: the computation's `Other` reaches the handler in `main`,
/// not the one around the `resume`.
#[test]
fn test_a_resume_with_a_selection_keeps_its_frame() {
    let code = r#"
ability Ping {
    op ping() -> Int
}

ability Other {
    op other() -> Int
}

fn handled(comp: fn() ->{e, Ping} Int) ->{e, Ping} Int {
    handle comp() {
        do n { n }
        op Ping::ping() {
            handle {
                let inner = Other::other()
                resume inner
            } {
                fn Other::other() { +1 }
            }
        }
    }
}

fn main() ->{std::io::Io} Nil {
    let v = handle handled(fn() { Ping::ping() + Other::other() }) {
        do r { r }
        fn Other::other() { +100 }
        fn Ping::ping() { +1000 }
    }
    std::io::print_line(v.to_string)
}
"#;
    assert_output_on_both_targets("resume_with_selection.trb", code, "101");
}

/// The resumptions an arm drops, captures in the lambda it hands on, or
/// never has (a `do` arm) are released exactly once.
#[test]
fn test_become_in_arms_releases_resumptions_under_asan() {
    let loops = "1000";
    let dropped = format!(
        "{STEP_LOOP}{}",
        r#"
fn tick(n: Int) ->{} Int {
    become drive(tick, n)
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(tick(+N).to_string)
}
"#
        .replace("+N", &format!("+{loops}"))
    );
    assert_released_on_both_targets("become_arm_dropped.trb", &dropped, "0");
    let captured = format!(
        "{STATE_LOOP}{}",
        r#"
fn main() ->{std::io::Io} Nil {
    std::io::print_line(run_state(fn() count(+N), +0).to_string)
}
"#
        .replace("+N", &format!("+{loops}"))
    );
    assert_released_on_both_targets("become_arm_captured.trb", &captured, loops);
    let do_arm = r#"
ability Tick {
    fn tick() -> Nil
}

fn work() ->{Tick} Int {
    Tick::tick()
    +1
}

fn rep(n: Int, acc: Int) ->{} Int {
    handle work() {
        do r {
            case n == +0 {
                True -> acc
                False -> become rep(n - +1, acc + r)
            }
        }
        fn Tick::tick() { Nil }
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(rep(+N, +0).to_string)
}
"#
    .replace("+N", &format!("+{loops}"));
    assert_released_on_both_targets("become_do_arm.trb", &do_arm, loops);
}

/// A lambda whose own flow runs a resumption-carrying lambda in a handler
/// arm is converted twice, once for each of its entries: `l(+1)` runs it
/// directly, and `via_arm` enters its `Cps` entry through a `become` in an
/// `op` arm. Both print their answers on each target, and the native run
/// leaks nothing.
#[test]
fn test_cps_entry_lambda_with_a_resumption_in_an_arm_releases_under_asan() {
    let code = r#"
use std::io::{Io, print_line}

ability Ask {
    op ask() -> Int
}

fn via_arm(k: fn(Int) ->{} Int, n: Int) ->{} Int {
    handle Ask::ask() {
        do x { x }
        op Ask::ask() { become k(n) }
    }
}

fn main() ->{Io} Nil {
    let l = fn(n: Int) ->{} Int {
        let r = handle {
            handle Ask::ask() + n {
                do x { x }
                op Ask::ask() {
                    let k = fn(v: Int) { resume v }
                    k(+10)
                }
            }
        } {
            do y { y + +100 }
            op Ask::ask() { +0 }
        }
        r + +1
    }
    print_line(l(+1).to_string)
    print_line(via_arm(l, +2).to_string)
}
"#;
    assert_released_on_both_targets("cps_entry_arm_resumption.trb", code, "112\n113");
}

/// Like [`common::assert_output_on_both_targets_with_native_asan`], and the
/// native run reports no leak: every allocation is released exactly once.
fn assert_released_on_both_targets(source_name: &str, source_code: &str, expected_stdout: &str) {
    let native = compile_and_run_native_asan(source_name, source_code);
    let stderr = String::from_utf8_lossy(&native.stderr);
    assert!(
        !stderr.contains("detected memory leaks"),
        "native leaks for {source_name}: {stderr}"
    );
    assert_target_outputs(
        source_name,
        expected_stdout,
        native,
        compile_and_run_wasm(source_name, source_code),
    );
}
