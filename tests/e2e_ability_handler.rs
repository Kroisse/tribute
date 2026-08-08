//! End-to-end tests for single ability handler execution.
//!
//! These tests verify that handler expressions compile to native binaries
//! and execute correctly with proper effect handling semantics.

mod common;

use common::{assert_native_output, compile_and_run_native};

fn cps_control_read_outcome_program() -> &'static str {
    r#"
use abilities::Throw
use std::io::{Error, Io, print_line, read_line}

// `Line` intentionally owns public enum tag zero. The normal path pattern
// matches and transforms its payload, so treating this value as a private
// carrier cannot accidentally produce the expected result.
enum ReadOutcome {
    Line(String),
    EndOfInput,
}

ability Gate {
    op decide() -> ReadOutcome
}

fn read_outcome() ->{Io} ReadOutcome {
    handle read_line() {
        do line { ReadOutcome::Line(line) }
        op Throw::throw(error) {
            case error {
                Error::EndOfFile -> ReadOutcome::EndOfInput
                _ -> ReadOutcome::EndOfInput
            }
        }
    }
}

fn gate_body() ->{Gate} ReadOutcome {
    let value = Gate::decide()
    case value {
        ReadOutcome::Line(text) -> ReadOutcome::Line(text <> "-suffix")
        ReadOutcome::EndOfInput -> ReadOutcome::EndOfInput
    }
}

fn nested_gate(stop: Bool) -> ReadOutcome {
    let inner = handle gate_body() {
        do value {
            case value {
                ReadOutcome::Line(text) -> ReadOutcome::Line(text <> "-inner-do")
                ReadOutcome::EndOfInput -> ReadOutcome::Line("inner-do-eof")
            }
        }
        // This is an ordinary `op`, not a `Never` operation. The branch
        // dynamically completes without resume and must skip the `do` arm.
        op Gate::decide() {
            case stop {
                True -> ReadOutcome::EndOfInput
                False -> resume ReadOutcome::Line("resumed")
            }
        }
    }
    handle inner {
        do value {
            case value {
                ReadOutcome::Line(text) -> ReadOutcome::Line(text <> "-outer-do")
                ReadOutcome::EndOfInput -> ReadOutcome::Line("outer-do-eof")
            }
        }
    }
}

fn print_outcome(outcome: ReadOutcome) {
    case outcome {
        ReadOutcome::Line(text) -> print_line("line:" <> text)
        ReadOutcome::EndOfInput -> print_line("eof")
    }
}

fn main() {
    print_outcome(read_outcome())
    print_outcome(nested_gate(False))
    print_outcome(nested_gate(True))
}
"#
}

fn read_outcome_cases() -> [(&'static str, &'static [u8], &'static str); 2] {
    [
        (
            "input",
            b"typed\n",
            "line:typed\nline:resumed-suffix-inner-do-outer-do\nline:outer-do-eof\n",
        ),
        (
            "eof",
            b"",
            "eof\nline:resumed-suffix-inner-do-outer-do\nline:outer-do-eof\n",
        ),
    ]
}

fn assert_read_outcome_cases(profile: &str, outputs: Vec<std::process::Output>) {
    let cases = read_outcome_cases();
    assert_eq!(outputs.len(), cases.len());
    for ((name, _, expected), output) in cases.iter().zip(outputs) {
        assert!(
            output.status.success(),
            "{name}/{profile}: exit={:?}, stderr='{}'",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            *expected,
            "{name}/{profile}"
        );
    }
}

#[test]
fn test_cps_control_carrier_preserves_read_outcome_and_nested_zero_resume_production() {
    let cases = read_outcome_cases();
    let inputs: Vec<_> = cases.iter().map(|(_, stdin, _)| *stdin).collect();
    let outputs = common::compile_and_run_native_with_stdin_cases(
        "cps_control_read_outcome_production.trb",
        cps_control_read_outcome_program(),
        &inputs,
    );
    assert_read_outcome_cases("production", outputs);
}

#[test]
fn test_cps_control_carrier_preserves_read_outcome_and_nested_zero_resume_baseline() {
    let cases = read_outcome_cases();
    let inputs: Vec<_> = cases.iter().map(|(_, stdin, _)| *stdin).collect();
    let outputs = common::compile_and_run_native_with_stdin_cases_baseline_optimizations(
        "cps_control_read_outcome_baseline.trb",
        cps_control_read_outcome_program(),
        &inputs,
    );
    assert_read_outcome_cases("baseline", outputs);
}

#[test]
fn test_cps_control_carrier_preserves_read_outcome_and_nested_zero_resume_asan() {
    let cases = read_outcome_cases();
    let inputs: Vec<_> = cases.iter().map(|(_, stdin, _)| *stdin).collect();
    let outputs = common::compile_and_run_native_with_stdin_cases_asan(
        "cps_control_read_outcome_asan.trb",
        cps_control_read_outcome_program(),
        &inputs,
    );
    assert_read_outcome_cases("asan", outputs);
}

fn cps_control_outer_nonresumptive_crosses_inner_handle_program() -> &'static str {
    r#"
use std::io::print_line

ability Stop {
    op stop() -> String
}

// The effect is performed while this inner handler is active, but the inner
// handler has no `Stop` arm. Its `do` marker must therefore not run when the
// outer handler completes the operation without resume.
fn perform_inside_inner() ->{Stop} String {
    Stop::stop()
}

fn inner_boundary() ->{Stop} String {
    handle perform_inside_inner() {
        do value { "inner-do:" <> value }
    }
}

fn outer_boundary() -> String {
    handle inner_boundary() {
        do value { "outer-do:" <> value }
        op Stop::stop() { "outer-handler" }
    }
}

fn main() {
    print_line(outer_boundary())
}
"#
}

/// A non-resumptive outer handler must cross an inner, non-matching handler
/// boundary without invoking either `do` arm.
#[test]
fn test_cps_control_outer_nonresumptive_bypasses_nested_do_arms() {
    let source = cps_control_outer_nonresumptive_crosses_inner_handle_program();
    let profiles = [
        (
            "production",
            common::compile_and_run_native_with_stdin(
                "cps_control_outer_nonresumptive_nested_handle_production.trb",
                source,
                b"",
            ),
        ),
        (
            "baseline",
            common::compile_and_run_native_with_stdin_baseline_optimizations(
                "cps_control_outer_nonresumptive_nested_handle_baseline.trb",
                source,
                b"",
            ),
        ),
        (
            "asan",
            common::compile_and_run_native_with_stdin_asan(
                "cps_control_outer_nonresumptive_nested_handle_asan.trb",
                source,
                b"",
            ),
        ),
    ];
    for (profile, output) in profiles {
        assert!(
            output.status.success(),
            "{profile}: exit={:?}, stderr='{}'",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "outer-handler\n",
            "{profile}"
        );
    }
}

/// The nearest dynamically installed handler owns its Escape even when an
/// outer handle installs an arm for the same ability. The inner `do` is
/// bypassed, while the outer receives an ordinary source completion and runs
/// its own `do` exactly once.
#[test]
fn test_cps_control_same_ability_nested_owner_is_nearest() {
    assert_native_output(
        "cps_control_same_ability_dynamic_owner.trb",
        r#"
use std::io::print_line

ability Stop {
    op stop() -> String
}

fn inner() -> String {
    handle Stop::stop() {
        do value { "inner-do:" <> value }
        op Stop::stop() { "inner-handler" }
    }
}

fn main() {
    let value = handle inner() {
        do result { "outer-do:" <> result }
        op Stop::stop() { "outer-handler" }
    }
    print_line(value)
}
"#,
        "outer-do:inner-handler",
    );
}

/// Recursive re-entry executes the same syntactic `handle` site twice. Each
/// dynamic activation receives its own runtime owner tag: the inner Escape is
/// consumed by the inner activation, then the outer arm completes its own
/// activation without either `do` marker running.
#[test]
fn test_cps_control_recursive_same_handle_site_uses_distinct_dynamic_owners() {
    assert_native_output(
        "cps_control_recursive_dynamic_owner.trb",
        r#"
use std::io::print_line

ability Ping {
    op ping() -> String
}

fn run(next: fn() -> String) -> String {
    handle Ping::ping() {
        do value { "do:" <> value }
        op Ping::ping() { next() }
    }
}

fn base() -> String { "base" }

fn main() {
    print_line(run(fn() { run(fn() { base() }) }))
}
"#,
        "base",
    );
}

/// A foreign Escape produced while resuming must not execute the resumed
/// handler arm's strict suffix, its inner `do`, or the owning outer `do`.
#[test]
fn test_cps_control_foreign_escape_bypasses_resume_suffix() {
    assert_native_output(
        "cps_control_foreign_escape_resume_suffix.trb",
        r#"
use std::io::print_line

ability First {
    op first() -> String
}

ability Stop {
    op stop() -> String
}

fn inner() ->{Stop} String {
    handle {
        let _ = First::first()
        Stop::stop()
    } {
        do value { "inner-do:" <> value }
        op First::first() {
            let resumed = resume "first-value"
            "resume-suffix:" <> resumed
        }
    }
}

fn main() {
    let value = handle inner() {
        do result { "outer-do:" <> result }
        op Stop::stop() { "stop-handler" }
    }
    print_line(value)
}
"#,
        "stop-handler",
    );
}

// =============================================================================
// Native Execution Tests
// =============================================================================

/// Test ability_core.trb compiles and executes correctly.
///
/// The program calls counter() three times starting from 0:
/// - counter() returns 0, state becomes 1
/// - counter() returns 1, state becomes 2
/// - counter() returns 2, state becomes 3
///
/// The final return value is 2 (the last counter() call's return).
#[test]
fn test_ability_core_execution() {
    let code = include_str!("../lang-examples/ability_core.trb");
    let output = compile_and_run_native("ability_core.trb", code);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        stderr
    );
}

/// Test simple State::get handler that returns a constant.
#[test]
fn test_state_get_simple() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn main() {
    let _ = handle get_state() {
        do result { result }
        op State::get() { resume +42 }
        op State::set(v) { resume Nil }
    }
}
"#;
    let output = compile_and_run_native("state_get_simple.trb", code);
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test State::set followed by State::get.
#[test]
fn test_state_set_then_get() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn set_then_get() ->{State(Int)} Int {
    State::set(+100)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let _ = run_state(fn() { set_then_get() }, +0)
}
"#;
    let output = compile_and_run_native("state_set_then_get.trb", code);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        stderr
    );
}

/// Test nested handler calls.
#[test]
fn test_nested_state_calls() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn increment() ->{State(Nat)} Nil {
    let n = State::get()
    State::set(n + 1)
}

fn double_increment() ->{State(Nat)} Nat {
    increment()
    increment()
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let _ = run_state(fn() { double_increment() }, 5)
}
"#;
    let output = compile_and_run_native("nested_state_calls.trb", code);
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test deeply nested recursive handler invocations.
///
/// Performs three increments starting from 10, resulting in state 13.
/// Stresses the runtime tag uniqueness mechanism more than
/// `test_nested_state_calls` (5 yields × 3 increments = 15+ prompt frames).
#[test]
fn test_nested_state_triple_increment() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn increment() ->{State(Nat)} Nil {
    let n = State::get()
    State::set(n + 1)
}

fn triple_increment() ->{State(Nat)} Nat {
    increment()
    increment()
    increment()
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let _ = run_state(fn() { triple_increment() }, 10)
}
"#;
    let output = compile_and_run_native("nested_state_triple.trb", code);
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test direct result path (no effect operations).
#[test]
fn test_handler_direct_result() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn no_effects() ->{State(Int)} Int {
    +42
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let _ = run_state(fn() { no_effects() }, +0)
}
"#;
    let output = compile_and_run_native("handler_direct_result.trb", code);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        stderr
    );
}

// =============================================================================
// Handler Early Return Tests
// =============================================================================

/// Test handler that discards the continuation (early return).
///
/// `might_fail()` calls Fail::fail(), but the handler doesn't call resume —
/// it returns a default value directly, short-circuiting the computation.
#[test]
fn test_handler_early_return() {
    let code = r#"ability Fail {
    op fail() -> Nat
}

fn might_fail() ->{Fail} Nat {
    let x = Fail::fail()
    x + 100
}

fn main() {
    let result = handle might_fail() {
        do result { result }
        op Fail::fail() { 99 }
    }
    __tribute_print_nat(result)
}
"#;
    // Handler returns 99 directly without calling resume, so x + 100 is never reached
    assert_native_output("handler_early_return.trb", code, "99");
}

/// Test handler with `op -> Never` (abort pattern).
///
/// When an operation is declared as `op fail() -> Never`, the handler cannot
/// call `resume` and no continuation is captured — this exercises the
/// non-resuming type-checking and lowering path.
#[test]
fn test_handler_op_never_abort() {
    let code = r#"ability FailNever {
    op fail() -> Never
}

fn might_fail() ->{FailNever} Nat {
    FailNever::fail()
}

fn main() {
    let result = handle might_fail() {
        do result { result }
        op FailNever::fail() { 99 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_op_never_abort.trb", code, "99");
}

/// Test Abort ability with a parameterized abort operation.
///
/// The abort operation carries a Nat payload, and the handler uses it
/// as an alternative result.
#[test]
fn test_abort_with_payload() {
    let code = r#"ability Abort {
    op abort(code: Nat) -> Never
}

fn do_abort() ->{Abort} Nat {
    Abort::abort(99)
}

fn main() {
    let result = handle do_abort() {
        do result { result }
        op Abort::abort(code) { code }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("abort_with_payload.trb", code, "99");
}

/// Test Abort ability: handler provides fallback for multiple abort calls.
///
/// Two separate handle expressions each handle their own abort independently.
#[test]
fn test_abort_multiple_handles() {
    let code = r#"ability Abort {
    op abort() -> Never
}

fn always_abort() ->{Abort} Nat {
    Abort::abort()
}

fn main() {
    let a = handle always_abort() {
        do result { result }
        op Abort::abort() { 10 }
    }
    let b = handle always_abort() {
        do result { result }
        op Abort::abort() { 20 }
    }
    __tribute_print_nat(a + b)
}
"#;
    assert_native_output("abort_multiple_handles.trb", code, "30");
}

/// Test conditional abort with case branches returning `Never` and `Nat`.
#[test]
fn test_abort_conditional() {
    let code = r#"ability Abort {
    op abort() -> Never
}

fn might_abort(should_abort: Bool) ->{Abort} Nat {
    case should_abort {
        True -> Abort::abort()
        False -> 42
    }
}

fn main() {
    let a = handle might_abort(True) {
        do result { result }
        op Abort::abort() { 0 }
    }
    let b = handle might_abort(False) {
        do result { result }
        op Abort::abort() { 0 }
    }
    __tribute_print_nat(a)
    __tribute_print_nat(b)
}
"#;
    assert_native_output("abort_conditional.trb", code, "0\n42");
}

/// Test handling an ability declared inside a module.
///
/// The handler arm uses a module-qualified path (MyMod::Counter::inc)
/// to reference the operation.
#[test]
fn test_handler_ability_in_module() {
    let code = r#"mod MyMod {
    pub ability Counter {
        op inc() -> Nat
    }
}

fn count() ->{MyMod::Counter} Nat {
    let a = MyMod::Counter::inc()
    let b = MyMod::Counter::inc()
    a + b
}

fn main() {
    let result = handle count() {
        do result { result }
        op MyMod::Counter::inc() { resume 1 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_ability_in_module.trb", code, "2");
}

// =============================================================================
// Tail-Resumptive (fn) Handler Arm Tests
// =============================================================================

/// Test `fn` handler arm (tail-resumptive) compiles and runs.
///
/// `Ask::ask()` is declared as `fn`, so the handler arm uses `fn` keyword.
/// Currently `fn` arms are lowered identically to `op` arms (no automatic
/// resume from body return value yet). This test verifies the `fn` handler
/// arm path through parsing, resolution, and lowering.
///
/// Note: When tail-resumptive optimization is implemented, the body's return
/// value will automatically become the resume value without explicit `resume`.
#[test]
fn test_fn_handler_arm() {
    let code = r#"ability Ask {
    fn ask() -> Nat
}

fn use_ask() ->{Ask} Nat {
    Ask::ask()
}

fn main() {
    let result = handle use_ask() {
        do result { result }
        fn Ask::ask() { 42 }
    }
    __tribute_print_nat(result)
}
"#;
    // fn arm body returns 42; since tail-resumptive auto-resume is not yet
    // implemented, this acts like an early return with value 42.
    assert_native_output("fn_handler_arm.trb", code, "42");
}

/// Tail-resumptive `fn` arms return their own operation result, not the
/// enclosing handle answer. A `Nil` operation therefore cannot overwrite the
/// preceding `Nat` call's source value.
#[test]
fn test_fn_handler_arm_preserves_each_operation_result_type() {
    let code = r#"ability Console {
    fn read() -> Nat
    fn print(value: Nat) -> Nil
}

fn use_console() ->{Console} Nat {
    let value = Console::read()
    Console::print(value)
    value
}

fn main() {
    let result = handle use_console() {
        do value { value }
        fn Console::read() { 42 }
        fn Console::print(value) { Nil }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("fn_handler_arm_result_types.trb", code, "42");
}

// =============================================================================
// Handler Result Transformation Tests
// =============================================================================

/// Test handler result arm with identity (pass-through).
///
/// `pure_value()` returns 10 with no effects. The handler's result arm
/// just returns result unchanged.
#[test]
fn test_handler_result_identity() {
    let code = r#"ability Ask {
    op ask() -> Nat
}

fn pure_value() ->{Ask} Nat {
    10
}

fn main() {
    let result = handle pure_value() {
        do result { result }
        op Ask::ask() { resume 0 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_result_identity.trb", code, "10");
}

/// Test handler result arm that returns a constant.
#[test]
fn test_handler_result_constant() {
    let code = r#"ability Ask {
    op ask() -> Nat
}

fn pure_value() ->{Ask} Nat {
    10
}

fn main() {
    let result = handle pure_value() {
        do result { 42 }
        op Ask::ask() { resume 0 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_result_constant.trb", code, "42");
}

/// Test handler result arm that transforms the body's return value.
///
/// `pure_value()` returns 10 with no effects. The handler's result arm
/// doubles it: result + result = 20.
#[test]
fn test_handler_transforms_result() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn pure_value() ->{State(Nat)} Nat {
    10
}

fn main() {
    let result = handle pure_value() {
        do result { result + result }
        op State::get() { resume 0 }
        op State::set(v) { resume Nil }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_transforms_result.trb", code, "20");
}

/// Test a result transformation after resuming an effectful operation.
///
/// `effectful_value()` resumes twice with 10, producing 21, and the result arm
/// doubles it. This exercises nested ability operations in both a let initializer
/// and the final value, then verifies result TDNR and native execution.
#[test]
fn test_handler_transforms_resumed_result() {
    let code = r#"ability State(s) {
    op get() -> s
}

fn effectful_value() ->{State(Nat)} Nat {
    let first = State::get() + 1
    first + State::get()
}

fn main() {
    let result = handle effectful_value() {
        do result { result + result }
        op State::get() { resume 10 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_transforms_resumed_result.trb", code, "42");
}

/// A resumed body reaches the completion exactly once before its answer is
/// returned to the outer continuation.
#[test]
fn test_handler_one_resume_runs_nonidentity_completion_once() {
    let code = r#"ability Ask {
    op ask() -> Nat
}

fn body() ->{Ask} Nat {
    Ask::ask() + 1
}

fn main() {
    let result = handle body() {
        do value { value + value }
        op Ask::ask() { resume 10 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_one_resume_completion_once.trb", code, "22");
}

/// The handle body and answer types are independent. The effectful callee is
/// separately declared, so the CPS call boundary must retain the answer-indexed
/// Done/Dispatch ABI instead of equating it with the body's String result.
#[test]
fn test_handler_resumed_callee_can_change_answer_type() {
    let code = r#"ability Ask {
    op ask() -> String
}

fn separately_compiled() ->{Ask} String {
    Ask::ask()
}

fn main() {
    let answer = handle separately_compiled() {
        do _body { 7 }
        op Ask::ask() { resume "payload" }
    }
    __tribute_print_nat(answer)
}
"#;
    assert_native_output("handler_resumed_answer_type_change.trb", code, "7");
}

/// Each re-entrant resume owns an immutable arm-local suffix.  The second
/// perform completes and runs its suffix before the first resumed suffix is
/// re-entered; no mutable delimiter slot may overwrite the outer suffix.
#[test]
fn test_handler_reentrant_resumes_preserve_suffix_stack() {
    let code = r#"ability Ask {
    op ask() -> Nat
}

fn twice() ->{Ask} Nat {
    let first = Ask::ask()
    first + Ask::ask()
}

fn main() {
    let result = handle twice() {
        do value { value + 1 }
        op Ask::ask() {
            let resumed = resume 10
            resumed + 100
        }
    }
    __tribute_print_nat(result)
}
"#;
    // The inner resume completes to 21, then its arm-local suffix yields
    // 121.  That is the resumed result observed by the suspended outer arm,
    // whose own suffix must still run and yield 221.
    assert_native_output("handler_reentrant_resume_suffix_stack.trb", code, "221");
}

/// A foreign operation reached by a resumed inner body must use the Parent
/// captured by that resume.  Its outer handler returns to the suspended inner
/// arm suffix rather than the initial handle entry flow.
#[test]
fn test_handler_resumed_foreign_dispatch_preserves_arm_suffix() {
    let code = r#"ability Inner {
    op ask() -> Nat
}

ability Outer {
    op ask() -> Nat
}

fn body() ->{Inner, Outer} Nat {
    let inner = Inner::ask()
    inner + Outer::ask()
}

fn run_inner() ->{Inner, Outer} Nat {
    handle body() {
        do value { value + 1 }
        op Inner::ask() {
            let resumed = resume 10
            resumed + 100
        }
    }
}

fn main() {
    let result = handle run_inner() {
        do value { value }
        op Outer::ask() { resume 5 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_resumed_foreign_dispatch_suffix.trb", code, "116");
}

/// The call-boundary adapter must rebuild its dispatcher from the dynamic
/// Parent supplied to the resumed callback, not from the call site's initial
/// dispatch capture. This exercises the same re-entrant suffix stack through
/// a first-class CPS call.
#[test]
fn test_handler_indirect_reentrant_resumes_preserve_suffix_stack() {
    let code = r#"ability Ask {
    op ask() -> Nat
}

fn twice() ->{Ask} Nat {
    let first = Ask::ask()
    first + Ask::ask()
}

fn apply(f: fn() ->{Ask} Nat) ->{Ask} Nat {
    f()
}

fn main() {
    let result = handle apply(twice) {
        do value { value + 1 }
        op Ask::ask() {
            let resumed = resume 10
            resumed + 100
        }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output(
        "handler_indirect_reentrant_resume_suffix_stack.trb",
        code,
        "221",
    );
}

/// A re-entrant body can produce Nat while the handle answer and every
/// arm-local resume suffix produce String. Parent<A> therefore cannot be
/// conflated with the body's result index.
#[test]
fn test_handler_answer_changing_reentrant_resumes_preserve_suffix_stack() {
    let code = r#"use std::io::print_line

ability Ask {
    op ask() -> Nat
}

fn twice() ->{Ask} Nat {
    let first = Ask::ask()
    first + Ask::ask()
}

fn main() {
    let result = handle twice() {
        do _value { "done" }
        op Ask::ask() {
            let resumed = resume 10
            resumed <> "!"
        }
    }
    print_line(result)
}
"#;
    assert_native_output(
        "handler_answer_changing_reentrant_resume_suffix_stack.trb",
        code,
        "done!!",
    );
}

// =============================================================================
// Counter Output Verification Tests
// =============================================================================

/// Test classic counter pattern: get, set(n+1), return n.
///
/// counter() does get → set(n+1) → return n.
/// Starting from 0: counter()=0 (state→1). Returns 0.
#[test]
fn test_counter_returns_correct_value() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn counter() ->{State(Nat)} Nat {
    let n = State::get()
    State::set(n + 1)
    n
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let result = run_state(fn() { counter() }, 0)
    __tribute_print_nat(result)
}
"#;
    assert_native_output("counter_value.trb", code, "0");
}

/// Test counter starting from a non-zero initial state.
///
/// Starting from 10: counter()=10 (state→11). Returns 10.
#[test]
fn test_counter_nonzero_initial() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn counter() ->{State(Nat)} Nat {
    let n = State::get()
    State::set(n + 1)
    n
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let result = run_state(fn() { counter() }, 10)
    __tribute_print_nat(result)
}
"#;
    assert_native_output("counter_nonzero.trb", code, "10");
}

// =============================================================================
// State Final Value Tests
// =============================================================================

/// Test reading final state after multiple mutations.
///
/// Performs: set(3), set(get()+10) → set(13), get() → 13.
#[test]
fn test_state_multiple_mutations() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn mutate() ->{State(Nat)} Nat {
    State::set(3)
    let n = State::get()
    State::set(n + 10)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let result = run_state(fn() { mutate() }, 0)
    __tribute_print_nat(result)
}
"#;
    assert_native_output("state_multiple_mutations.trb", code, "13");
}

/// Test that handler result arm receives the final computed value.
///
/// `compute()` does set(5), get()+get() → 10.
/// Handler result arm adds 1: 10 + 1 = 11.
#[test]
fn test_handler_result_receives_body_value() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn compute() ->{State(Nat)} Nat {
    State::set(5)
    let a = State::get()
    let b = State::get()
    a + b
}

fn main() {
    let result = handle compute() {
        do result { result + 1 }
        op State::get() { resume 5 }
        op State::set(v) { resume Nil }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_result_body_value.trb", code, "11");
}

// =============================================================================
// Closure / call_indirect with Effects Execution Tests
// =============================================================================

/// Test that multiple effectful calls followed by a non-effectful call
/// execute correctly.
///
/// Exercises the `needs_rebuild` fix: after effectful calls are expanded into
/// Done/Shift branches, subsequent ops that reference remapped call results
/// must have their operands correctly updated.
#[test]
fn test_multiple_effectful_calls_then_pure_call() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn add(a: Nat, b: Nat) -> Nat { a + b }

fn compute() ->{State(Nat)} Nat {
    let a = State::get()
    let b = State::get()
    add(a, b)
}

fn main() {
    let result = handle compute() {
        do result { result }
        op State::get() { resume 5 }
        op State::set(v) { resume Nil }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("multiple_effectful_then_pure_call.trb", code, "10");
}

/// Test that a direct non-effectful function call after an effectful call
/// is not incorrectly truncated as dead code.
///
/// Exercises the `remaining_are_dead_code` check on a direct call.
#[test]
fn test_non_effectful_call_in_nested_region() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn identity(x: Nat) -> Nat { x }

fn compute() ->{State(Nat)} Nat {
    let n = State::get()
    identity(n)
}

fn main() {
    let result = handle compute() {
        do result { result }
        op State::get() { resume 7 }
        op State::set(v) { resume Nil }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("non_effectful_call_in_nested_region.trb", code, "7");
}

// =============================================================================
// Edge Case: Ability Operation with Multiple Parameters
// =============================================================================

/// Test that a handler correctly receives multiple parameters from an ability op.
///
/// `Multi::combine(10, 20, 30)` yields three arguments to the handler arm.
/// The handler sums them and resumes with the result.
///
/// Currently segfaults at runtime: handler unpack produces anyref values that
/// are used directly in arith.add without unbox_int, causing type mismatch.
#[test]
fn test_handler_multi_param_op() {
    let code = r#"ability Multi {
    op combine(x: Nat, y: Nat, z: Nat) -> Nat
}

fn use_multi() ->{Multi} Nat {
    Multi::combine(10, 20, 30)
}

fn main() {
    let result = handle use_multi() {
        do result { result }
        op Multi::combine(x, y, z) { resume x + y + z }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_multi_param_op.trb", code, "60");
}

/// Test two-parameter ability op handler (simplest multi-param case).
#[test]
fn test_handler_two_param_op() {
    let code = r#"ability Pair {
    op make(a: Nat, b: Nat) -> Nat
}

fn use_pair() ->{Pair} Nat {
    Pair::make(3, 7)
}

fn main() {
    let result = handle use_pair() {
        do result { result }
        op Pair::make(a, b) { resume a + b }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_two_param_op.trb", code, "10");
}

/// Test multi-param op where handler only uses one parameter.
#[test]
fn test_handler_multi_param_partial_use() {
    let code = r#"ability Pick {
    op choose(a: Nat, b: Nat, c: Nat) -> Nat
}

fn use_pick() ->{Pick} Nat {
    Pick::choose(100, 200, 300)
}

fn main() {
    let result = handle use_pick() {
        do result { result }
        op Pick::choose(a, _, c) { resume a + c }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_multi_param_partial.trb", code, "400");
}

/// Test multi-param ability op with `fn` (tail-resumptive) handler.
#[test]
fn test_handler_multi_param_fn_arm() {
    let code = r#"ability Arith {
    fn add(a: Nat, b: Nat) -> Nat
}

fn use_arith() ->{Arith} Nat {
    Arith::add(15, 27)
}

fn main() {
    let result = handle use_arith() {
        do result { result }
        fn Arith::add(a, b) { a + b }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_multi_param_fn.trb", code, "42");
}

/// Test multi-param op called multiple times within a handler.
#[test]
fn test_handler_multi_param_repeated_calls() {
    let code = r#"ability Math {
    op mul(a: Nat, b: Nat) -> Nat
}

fn computation() ->{Math} Nat {
    let x = Math::mul(3, 4)
    let y = Math::mul(x, 5)
    y
}

fn main() {
    let result = handle computation() {
        do result { result }
        op Math::mul(a, b) { resume a * b }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("handler_multi_param_repeated.trb", code, "60");
}

// =============================================================================
// Throw(e) Ability Tests (#193)
//
// These tests use the prelude-defined abilities (abilities::Abort, abilities::Throw).
// =============================================================================

/// Test basic Throw(e) ability from prelude.
///
/// Throw(e) combines parameterized ability (like State(s)) with Never return
/// type (like Abort). The handler catches the thrown error value.
#[test]
fn test_throw_basic() {
    let code = r#"fn do_throw() ->{abilities::Throw(Nat)} Nat {
    abilities::Throw::throw(42)
}

fn main() {
    let result = handle do_throw() {
        do result { result }
        op abilities::Throw::throw(error) { error }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("throw_basic.trb", code, "42");
}

/// Test Throw(e) with error payload used in handler computation.
///
/// The handler receives the thrown error value and uses it to compute
/// an alternative result.
#[test]
fn test_throw_with_payload() {
    let code = r#"fn throw_with_offset(x: Nat) ->{abilities::Throw(Nat)} Nat {
    abilities::Throw::throw(x + 100)
}

fn main() {
    let result = handle throw_with_offset(5) {
        do result { result }
        op abilities::Throw::throw(error) { error }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("throw_with_payload.trb", code, "105");
}

/// Test Throw(e) with multiple throwing operations composed sequentially.
///
/// Two functions share the same Throw(Nat) effect. The first succeeds,
/// the second throws, and the handler catches the error.
#[test]
fn test_throw_multiple_operations() {
    let code = r#"fn no_throw() ->{abilities::Throw(Nat)} Nat {
    10
}

fn must_throw() ->{abilities::Throw(Nat)} Nat {
    abilities::Throw::throw(77)
}

fn do_work() ->{abilities::Throw(Nat)} Nat {
    let a = no_throw()
    let b = must_throw()
    a + b
}

fn main() {
    let result = handle do_work() {
        do result { result }
        op abilities::Throw::throw(error) { error }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("throw_multiple_ops.trb", code, "77");
}

/// Test Abort ability from prelude.
#[test]
fn test_prelude_abort() {
    let code = r#"fn do_abort() ->{abilities::Abort} Nat {
    abilities::Abort::abort()
}

fn main() {
    let result = handle do_abort() {
        do result { result }
        op abilities::Abort::abort() { 99 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("prelude_abort.trb", code, "99");
}

// =============================================================================
// Effect-Directed Name Resolution Tests (#629)
// =============================================================================

/// Test effect-directed resolution: `abort()` resolves from effect row.
#[test]
fn test_effect_directed_abort() {
    let code = r#"fn do_abort() ->{abilities::Abort} Nat {
    abort()
}

fn main() {
    let result = handle do_abort() {
        do result { result }
        op abilities::Abort::abort() { 99 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("effect_directed_abort.trb", code, "99");
}

/// Test effect-directed resolution: `throw(x)` resolves from effect row.
#[test]
fn test_effect_directed_throw() {
    let code = r#"fn do_throw() ->{abilities::Throw(Nat)} Nat {
    throw(42)
}

fn main() {
    let result = handle do_throw() {
        do result { result }
        op abilities::Throw::throw(error) { error }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("effect_directed_throw.trb", code, "42");
}

/// Test that a local variable shadows an effect-injected operation.
///
/// The parameter `abort` shadows `abilities::Abort::abort`, so calling
/// `abort()` invokes the parameter (a function), not the ability operation.
#[test]
fn test_effect_directed_local_shadows_op() {
    let code = r#"fn use_local(abort: fn() -> Nat) ->{abilities::Abort} Nat {
    abort()
}

fn main() {
    let result = handle use_local(fn() 77) {
        do result { result }
        op abilities::Abort::abort() { 0 }
    }
    __tribute_print_nat(result)
}
"#;
    // abort() calls the parameter (returns 77), not the ability op (would return 0)
    assert_native_output("effect_directed_shadow.trb", code, "77");
}

// =============================================================================
// Use Import Tests (#631)
// =============================================================================

/// Test `use abilities::Abort` makes Abort usable without qualification.
#[test]
fn test_use_import_abort() {
    let code = r#"use abilities::Abort

fn do_abort() ->{Abort} Nat {
    abort()
}

fn main() {
    let result = handle do_abort() {
        do result { result }
        op Abort::abort() { 99 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("use_import_abort.trb", code, "99");
}

/// Test `use abilities::Throw` makes Throw(e) usable without qualification.
#[test]
fn test_use_import_throw() {
    let code = r#"use abilities::Throw

fn do_throw() ->{Throw(Nat)} Nat {
    throw(42)
}

fn main() {
    let result = handle do_throw() {
        do result { result }
        op Throw::throw(error) { error }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("use_import_throw.trb", code, "42");
}
