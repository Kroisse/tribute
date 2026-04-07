//! End-to-end tests for single ability handler execution.
//!
//! These tests verify that handler expressions compile to native binaries
//! and execute correctly with proper effect handling semantics.

mod common;

use common::{assert_native_output, compile_and_run_native};

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
    State::set(100)
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
    let _ = run_state(fn() { set_then_get() }, 0)
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
    42
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() {
    let _ = run_state(fn() { no_effects() }, 0)
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

/// Test conditional abort with case expression (codegen limitation).
///
/// When `case` branches have different types (Never vs Nat), Cranelift
/// codegen currently has a type mismatch. This test is ignored until fixed.
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
/// to reference the operation. Currently module-qualified ability paths
/// are not yet supported in name resolution (#530).
#[test]
#[ignore = "module-qualified ability paths not yet supported in name resolution"]
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
#[ignore = "MethodCall in do handler arm not desugared before IR lowering"]
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
///
/// Currently fails because CPS transformation applies the continuation
/// to the error value for non-resumptive operations in sequential let
/// bindings (produces 87 = 10 + 77 instead of 77).
#[test]
#[ignore = "CPS continuation applied to non-resumptive op result in sequential let bindings (#624)"]
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
