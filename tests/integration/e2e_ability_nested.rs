//! End-to-end tests for nested and multi-ability handler execution.
//!
//! These tests verify correct behavior when multiple abilities are composed
//! through nested handler expressions, including shadowing, cross-ability
//! interactions, and deep nesting.

use crate::common;

use common::assert_native_output;

// =============================================================================
// Multi-Ability Execution Tests (#499)
// =============================================================================

/// Test two different abilities (State + Reader) with nested handlers.
///
/// `use_both()` performs Reader::ask() then State::set/get.
/// Outer handler provides Reader(42), inner handler runs State starting at 0.
/// Expected: Reader::ask() returns 42, State::set(42), State::get() returns 42.
#[test]
fn test_two_abilities_nested_handlers() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Reader(r) {
    op ask() -> r
}

fn use_both() ->{State(Nat), Reader(Nat)} Nat {
    let config = Reader::ask()
    State::set(config)
    State::get()
}

fn run_reader(comp: fn() ->{e, Reader(r)} a, value: r) ->{e} a {
    handle comp() {
        do result { result }
        op Reader::ask() { run_reader(fn() { resume value }, value) }
    }
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() -> Nil {
    let result = run_reader(fn() { run_state(fn() { use_both() }, 0) }, 42)
    __tribute_print_nat(result)
}
"#;
    assert_native_output("two_abilities_nested.trb", code, "42");
}

/// Test same ability with different type parameter instances nested (State inside State).
///
/// `inner()` uses `State(Bool)`: get() → True, set(False), get() → False.
/// `outer()` uses `State(Nat)`: set(7), delegates to a nested `run_state` for inner
/// with a Bool initial value, then get() → 7. Verifies each handler dispatches to
/// the correct prompt with distinct type parameters.
#[test]
fn test_same_ability_different_type_params_nested() {
    let code = r#"ability State(s) {
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

fn inner() ->{State(Bool)} Bool {
    let b = State::get()
    State::set(False)
    State::get()
}

fn outer() ->{State(Nat)} Nat {
    State::set(7)
    let _ = run_state(fn() { inner() }, True)
    State::get()
}

fn main() -> Nil {
    let result = run_state(fn() { outer() }, 0)
    __tribute_print_nat(result)
}
"#;
    // outer: set(7), run inner with State(Bool) (doesn't affect outer State(Nat)), get() → 7
    assert_native_output("same_ability_nested.trb", code, "7");
}

/// Test a single handle expression that handles multiple abilities at once.
///
/// `use_both()` calls Reader::ask() → 10, State::set(10) (discarded by stateless handler),
/// State::get() → 0 (stateless handler returns +0), returns 0+1=1.
#[test]
fn test_multiple_abilities_single_handler() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Reader(r) {
    op ask() -> r
}

fn use_both() ->{State(Nat), Reader(Nat)} Nat {
    let base = Reader::ask()
    State::set(base)
    let n = State::get()
    n + 1
}

fn main() -> Nil {
    let result = handle use_both() {
        do result { result }
        op State::get() { resume 0 }
        op State::set(v) { resume Nil }
        op Reader::ask() { resume 10 }
    }
    __tribute_print_nat(result)
}
"#;
    assert_native_output("multi_ability_single_handler.trb", code, "1");
}

// =============================================================================
// Triple Nested Handler Tests
// =============================================================================

/// Test three different abilities with nested handlers.
///
/// `use_all()` calls Reader::ask() → 5, Writer::tell(5), State::set(5), State::get() → 5.
/// Handlers: Reader provides 5, Writer is no-op, State starts at 0.
/// Expected: 5.
#[test]
fn test_three_abilities_nested_handlers_slow() {
    let code = include_str!("../fixtures/three_abilities_nested.trb");
    assert_native_output("three_abilities_nested.trb", code, "5");
}

// =============================================================================
// Nested Handler Semantics Tests (#500)
// =============================================================================

/// Test inner handler shadowing outer handler with same ability and same type.
///
/// Both handlers handle `State(Nat)`. The inner handler (init=0) should shadow
/// the outer handler (init=100). After inner completes, outer's state remains 100.
#[test]
fn test_nested_handler_same_ability_same_type_shadowing() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn inner_comp() ->{State(Nat)} Nat {
    let x = State::get()
    State::set(x + 1)
    State::get()
}

fn main() -> Nil {
    let result = run_state(fn() {
        let inner_result = run_state(fn() { inner_comp() }, 0)
        __tribute_print_nat(inner_result)
        let outer_val = State::get()
        __tribute_print_nat(outer_val)
        outer_val
    }, 100)
    __tribute_print_nat(result)
}
"#;
    // inner: get()→0, set(1), get()→1 → inner_result=1
    // outer: get()→100 (unchanged) → outer_val=100
    // final result=100
    assert_native_output("nested_handler_shadowing.trb", code, "1\n100\n100");
}

/// Test calling a different ability operation inside a handler arm.
///
/// The State handler's `set` arm calls `Logger::log()` to record the value.
/// Logger handler wraps State handler from outside.
#[test]
fn test_nested_handler_cross_ability_in_handler_arm() {
    let code = r#"ability Logger {
    op log(msg: Nat) -> Nil
}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_logging_state(comp: fn() ->{e, State(Nat), Logger} a, init: Nat) ->{e, Logger} a {
    handle comp() {
        op State::get() { run_logging_state(fn() { resume init }, init) }
        op State::set(v) {
            Logger::log(v)
            run_logging_state(fn() { resume Nil }, v)
        }
    }
}

fn run_logger(comp: fn() ->{e, Logger} a) ->{e} a {
    handle comp() {
        op Logger::log(msg) {
            __tribute_print_nat(msg)
            run_logger(fn() { resume Nil })
        }
    }
}

fn computation() ->{State(Nat), Logger} Nat {
    State::set(10)
    State::set(20)
    State::get()
}

fn main() -> Nil {
    let result = run_logger(fn() {
        run_logging_state(fn() { computation() }, 0)
    })
    __tribute_print_nat(result)
}
"#;
    // set(10) → Logger::log(10) prints 10
    // set(20) → Logger::log(20) prints 20
    // get() → 20
    // result = 20
    assert_native_output("nested_handler_cross_ability.trb", code, "10\n20\n20");
}

/// Test resuming a continuation that triggers another effect (re-entrant yield).
///
/// Computation calls do_a() then do_b(), summing results.
/// A handler resumes with 10, B handler resumes with 32.
/// Expected: 10 + 32 = 42.
#[test]
fn test_nested_handler_resume_triggers_different_effect() {
    let code = r#"ability A {
    op do_a() -> Nat
}

ability B {
    op do_b() -> Nat
}

fn run_a(comp: fn() ->{e, A} a) ->{e} a {
    handle comp() {
        op A::do_a() { run_a(fn() { resume 10 }) }
    }
}

fn run_b(comp: fn() ->{e, B} a) ->{e} a {
    handle comp() {
        op B::do_b() { run_b(fn() { resume 32 }) }
    }
}

fn computation() ->{A, B} Nat {
    let a = A::do_a()
    let b = B::do_b()
    a + b
}

fn main() -> Nil {
    let result = run_a(fn() { run_b(fn() { computation() }) })
    __tribute_print_nat(result)
}
"#;
    assert_native_output("nested_handler_reentrant_yield.trb", code, "42");
}

/// Test deep nesting (4 levels) of the same handler.
///
/// Four nested `run_state` handlers with init values 1, 2, 3, 4 (innermost first).
/// Each level performs get() to read its own state. The innermost computation
/// reads its state (init=1) and returns it.
#[test]
fn test_nested_handler_deep_four_levels_same_ability() {
    let code = r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn level4() ->{State(Nat)} Nat {
    let v = State::get()
    __tribute_print_nat(v)
    State::set(v + 1)
    State::get()
}

fn main() -> Nil {
    let result = run_state(fn() {
        let v4 = State::get()
        __tribute_print_nat(v4)
        run_state(fn() {
            let v3 = State::get()
            __tribute_print_nat(v3)
            run_state(fn() {
                let v2 = State::get()
                __tribute_print_nat(v2)
                run_state(fn() { level4() }, 1)
            }, 2)
        }, 3)
    }, 4)
    __tribute_print_nat(result)
}
"#;
    // level 4 (outermost): get()→4
    // level 3: get()→3
    // level 2: get()→2
    // level 1 (innermost): get()→1, set(2), get()→2
    // result = 2
    assert_native_output("nested_handler_deep_four_levels.trb", code, "4\n3\n2\n1\n2");
}

/// Two instances of one ability whose arguments share a representation
/// (`Nat` and `Int` are both `core.i32`) remain distinct evidence entries.
///
/// The inner handler runs `State(Int)`. Inside it, `get_nat()` performs the
/// outer `State(Nat)` and must reach the outer handler (7), not the inner one.
#[test]
fn test_same_ability_instances_with_shared_representation() {
    let code = r#"ability State(s) {
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

fn get_nat() ->{State(Nat)} Nat {
    State::get()
}

fn set_int(value: Int) ->{State(Int)} Nil {
    State::set(value)
}

fn both() ->{State(Int), State(Nat)} Nat {
    set_int(+5)
    get_nat()
}

fn main() -> Nil {
    let result = run_state(fn() { run_state(fn() { both() }, +1) }, 7)
    __tribute_print_nat(result)
}
"#;
    assert_native_output("same_ability_shared_representation.trb", code, "7");
}

/// Two instances of one ability whose arguments are distinct nominal types
/// with the same shape remain distinct evidence entries.
#[test]
fn test_same_ability_instances_with_distinct_nominal_arguments() {
    let code = r#"struct Point { value: Nat }
struct Color { value: Nat }

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

fn get_point() ->{State(Point)} Point {
    State::get()
}

fn set_color(value: Color) ->{State(Color)} Nil {
    State::set(value)
}

fn both() ->{State(Color), State(Point)} Nat {
    set_color(Color { value: 5 })
    let Point { value } = get_point()
    value
}

fn main() -> Nil {
    let result = run_state(fn() { run_state(fn() { both() }, Color { value: 1 }) }, Point { value: 7 })
    __tribute_print_nat(result)
}
"#;
    assert_native_output("same_ability_distinct_nominals.trb", code, "7");
}

/// Same-spelled nominal arguments declared in different modules are distinct
/// ability instances.
#[test]
fn test_same_ability_instances_with_same_spelled_nominal_arguments() {
    let code = r#"pub mod A {
    pub struct Color { value: Nat }
}

pub mod B {
    pub struct Color { value: Nat }
}

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

fn get_a() ->{State(A::Color)} A::Color {
    State::get()
}

fn set_b(value: B::Color) ->{State(B::Color)} Nil {
    State::set(value)
}

fn both() ->{State(B::Color), State(A::Color)} Nat {
    set_b(B::Color { value: 5 })
    let A::Color { value } = get_a()
    value
}

fn main() -> Nil {
    let result = run_state(fn() { run_state(fn() { both() }, B::Color { value: 1 }) }, A::Color { value: 7 })
    __tribute_print_nat(result)
}
"#;
    assert_native_output("same_ability_same_spelled_nominals.trb", code, "7");
}

// =============================================================================
// Ability-heavy sanitizer smoke test (#407)
// =============================================================================

fn asan_ability_smoke_program() -> String {
    format!(
        "{}\n{}",
        common::PRINT_EXTERNS,
        r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

ability Scale {
    op scale(x: Float) -> Float
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

// The handler arm captures `factor` (closure environment) and calls the
// function value `adjust` indirectly before resuming once.
fn run_scale(
    comp: fn() ->{e, Scale} a,
    factor: Float,
    adjust: fn(Float) ->{} Float,
) ->{e} a {
    handle comp() {
        do result { result }
        op Scale::scale(x) {
            let y = adjust(x * factor)
            run_scale(fn() { resume y }, factor, adjust)
        }
    }
}

fn step(n: Int) ->{State(Int), Scale} Float {
    let current = State::get()
    State::set(current + n)
    Scale::scale(1.5)
}

fn body() ->{State(Int), Scale} Float {
    let a = step(+10)
    let b = step(-3)
    let c = step(+100)
    let total = State::get()
    __tribute_print_int(total)
    a + b + c
}

fn main() -> Nil {
    let offset = 0.25
    let result = run_scale(
        fn() { run_state(fn() { body() }, +5) },
        2.0,
        fn(v) { v + offset }
    )
    __tribute_print_float(result)
}
"#
    )
}

/// Two abilities nested as handlers (multiple evidence entries/markers), a
/// handler arm capturing a local and calling a closure indirectly, one-shot
/// `resume`, and boxed `Int`/`Float` operation arguments and results. The
/// native binary is run under the production, baseline and ASan profiles, and
/// every profile must produce the same output.
#[test]
fn test_asan_ability_smoke_nested_handlers_closures_and_boxed_values() {
    let source = asan_ability_smoke_program();
    let profiles = [
        (
            "production",
            "asan_ability_smoke_production.trb",
            common::NativeTestProfile::Production,
        ),
        (
            "baseline",
            "asan_ability_smoke_baseline.trb",
            common::NativeTestProfile::Baseline,
        ),
        (
            "asan",
            "asan_ability_smoke_asan.trb",
            common::NativeTestProfile::Asan,
        ),
    ];
    let mut outputs = Vec::new();
    for (profile, source_name, kind) in profiles {
        let output =
            common::compile_native_test_binary(source_name, &source, kind).run_with_stdin(b"");
        assert!(
            output.status.success(),
            "{profile}: exit={:?}, stderr='{}'",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        outputs.push((
            profile,
            String::from_utf8_lossy(&output.stdout).into_owned(),
        ));
    }
    let (_, reference) = &outputs[0];
    for (profile, stdout) in &outputs {
        assert_eq!(
            stdout, reference,
            "{profile} output differs from production"
        );
    }
    assert_eq!(reference.trim(), "112\n9.75");
}
