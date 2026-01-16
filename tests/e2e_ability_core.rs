//! End-to-end tests for Ability System (Core) milestone.
//!
//! These tests verify the target code from issue #100 passes the full compilation
//! pipeline and executes correctly with wasmtime CLI.
//!
//! ## Milestone Target
//!
//! The ability system should support:
//! - Ability definitions with type parameters (`ability State(s) { ... }`)
//! - Effect annotations in function signatures (`->{State(Int)}`)
//! - Handler expressions (`handle ... { ... }`)
//! - Handler patterns (`{ State::get() -> k }`)
//!
//! ## Test Strategy
//!
//! Tests are organized in two categories:
//! 1. **Frontend tests**: Verify parsing, name resolution, and type checking
//! 2. **Execution tests**: Compile to WASM and run with wasmtime CLI

mod common;

use common::run_wasm_main;
use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{compile_to_wasm_binary, compile_with_diagnostics};
use tribute_front::SourceCst;

/// Helper to parse source code
fn parse_source(db: &dyn salsa::Database, name: &str, code: &str) -> SourceCst {
    let source_code = Rope::from_str(code);
    let tree = parse_with_thread_local(&source_code, None);
    SourceCst::from_path(db, name, source_code, tree)
}

/// Helper to compile and run code, returning the i32 result from main()
fn compile_and_run(code: &str, name: &str) -> i32 {
    let source_code = Rope::from_str(code);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, name, source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        run_wasm_main::<i32>(wasm_binary.bytes(db))
    })
}

// =============================================================================
// Basic Ability Definition Tests
// =============================================================================

/// Test that ability definitions parse and typecheck.
#[test]
fn test_ability_definition() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "ability_def.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test ability operations with effect annotations.
#[test]
fn test_ability_operation_with_effect() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "ability_effect.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Handler Expression Tests
// =============================================================================

/// Test basic handle expression parsing and typechecking.
#[test]
fn test_handle_expression() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn run() -> Int {
    handle get_state() {
        { result } -> result
        { State::get() -> k } -> 42
        { State::set(v) -> k } -> 0
    }
}

fn main() -> Int { run() }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "handle_expr.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Milestone Target Code Test (Issue #100)
// =============================================================================

/// Test the complete milestone target code from issue #100.
///
/// This is the main acceptance test for the Ability System (Core) milestone.
/// The code should:
/// 1. Parse correctly
/// 2. Pass name resolution
/// 3. Pass type checking with effect inference
///
/// Note: Full execution requires backend support (issues #112-#114).
#[test]
fn test_milestone_target_code() {
    // This is the target code from issue #100
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Int {
    run_state(fn() {
        counter()
        counter()
        counter()
    }, 0)
}
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "milestone_100.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Milestone target code should compile without errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Effect Row Tests
// =============================================================================

/// Test that effect row polymorphism works correctly.
/// The function `run_state` should handle `State(s)` and propagate remaining effects `e`.
#[test]
fn test_effect_row_polymorphism() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

ability Console {
    fn print(msg: Text) -> Nil
}

fn stateful_print() ->{State(Int), Console} Int {
    Console::print("hello")
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "effect_row.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test that multiple abilities can be combined in effect rows.
#[test]
fn test_multiple_abilities() {
    let code = r#"ability Reader(r) {
    fn ask() -> r
}

ability Writer(w) {
    fn tell(value: w) -> Nil
}

fn copy() ->{Reader(Int), Writer(Int)} Nil {
    let x = Reader::ask()
    Writer::tell(x)
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "multiple_abilities.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Expected no errors, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Let Binding Effect Propagation Tests (Issue #200)
// =============================================================================

/// Test that effects propagate correctly through let bindings.
///
/// When a let binding initializes from an effectful expression,
/// the effect must propagate to the enclosing function.
#[test]
fn test_let_binding_effect_propagation() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

// Effect from State::get() in let binding propagates to function signature
fn read_state() ->{State(Int)} Int {
    let x = State::get()
    x
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "let_effect_propagation.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Effect should propagate through let binding, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test that multiple let bindings accumulate effects correctly.
#[test]
fn test_multiple_let_bindings_accumulate_effects() {
    let code = r#"ability Reader(r) {
    fn ask() -> r
}

ability Writer(w) {
    fn tell(value: w) -> Nil
}

// Both Reader and Writer effects from let bindings propagate
fn copy_value() ->{Reader(Int), Writer(Int)} Nil {
    let x = Reader::ask()
    let _ = Writer::tell(x)
    Nil
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "multiple_let_effects.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Multiple effects should accumulate from let bindings, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test that sequential let bindings with effects work correctly (Phase 1-2 compatible).
#[test]
fn test_sequential_let_bindings_with_effects() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn sequential_state() ->{State(Int)} Int {
    let a = State::get()
    let b = State::get()
    let c = a + b
    c + 1
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "sequential_let_effects.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Sequential let bindings should propagate effects, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test that nested block let bindings with effects work correctly.
/// Currently ignored: Phase 1-2 only supports sequential code without nested blocks containing shifts.
/// TODO: Enable in Phase 3 when nested control flow with shifts is supported.
#[test]
#[ignore = "Phase 3: nested blocks with shifts not yet supported"]
fn test_nested_let_bindings_with_effects() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn nested_state() ->{State(Int)} Int {
    let a = State::get()
    let b = {
        let c = State::get()
        c + 1
    }
    a + b
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "nested_let_effects.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Nested let bindings should propagate effects, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

/// Test that let binding with pure expression doesn't introduce spurious effects.
#[test]
fn test_pure_let_binding_no_spurious_effects() {
    // This function has no effect annotation and uses only pure let bindings
    let code = r#"fn pure_computation() -> Int {
    let x = 1
    let y = 2
    x + y
}

fn main() -> Int { pure_computation() }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "pure_let_binding.trb", code);
        let result = compile_with_diagnostics(db, source);

        for diag in &result.diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }

        assert!(
            result.diagnostics.is_empty(),
            "Pure let bindings should not introduce effects, got {} diagnostics",
            result.diagnostics.len()
        );
    });
}

// =============================================================================
// Edge Cases and Error Detection
// =============================================================================

/// Test that unhandled effects are properly tracked.
/// A function using State without declaring it in its effect row should error.
#[test]
#[ignore = "Effect checking not yet enforced - requires #112"]
fn test_unhandled_effect_error() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

// Missing ->{State(Int)} annotation
fn bad_counter() -> Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Int { 0 }
"#;

    TributeDatabaseImpl::default().attach(|db| {
        let source = parse_source(db, "unhandled_effect.trb", code);
        let result = compile_with_diagnostics(db, source);

        // Should have an error about unhandled effect
        assert!(
            !result.diagnostics.is_empty(),
            "Expected error for unhandled effect"
        );
    });
}

// =============================================================================
// WASM Execution Tests
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
#[ignore = "Handler dispatch type mismatch - see plan file for details"]
fn test_ability_core_execution() {
    let code = include_str!("../lang-examples/ability_core.trb");
    let result = compile_and_run(code, "ability_core.trb");
    assert_eq!(result, 2, "Expected main to return 2, got {}", result);
}

/// Test simple State::get handler that returns a constant.
#[test]
#[ignore = "Handler dispatch type mismatch - see plan file for details"]
fn test_state_get_simple() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn main() -> Int {
    handle get_state() {
        { result } -> result
        { State::get() -> k } -> 42
        { State::set(v) -> k } -> 0
    }
}
"#;
    let result = compile_and_run(code, "state_get_simple.trb");
    assert_eq!(result, 42, "Expected main to return 42, got {}", result);
}

/// Test State::set followed by State::get.
#[test]
#[ignore = "Handler dispatch type mismatch - see plan file for details"]
fn test_state_set_then_get() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn set_then_get() ->{State(Int)} Int {
    State::set(100)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Int {
    run_state(fn() { set_then_get() }, 0)
}
"#;
    let result = compile_and_run(code, "state_set_then_get.trb");
    assert_eq!(result, 100, "Expected main to return 100, got {}", result);
}

/// Test nested handler calls.
#[test]
#[ignore = "Handler dispatch type mismatch - see plan file for details"]
fn test_nested_state_calls() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn increment() ->{State(Int)} Nil {
    let n = State::get()
    State::set(n + 1)
}

fn double_increment() ->{State(Int)} Int {
    increment()
    increment()
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Int {
    run_state(fn() { double_increment() }, 5)
}
"#;
    let result = compile_and_run(code, "nested_state_calls.trb");
    assert_eq!(result, 7, "Expected main to return 7, got {}", result);
}

/// Test direct result path (no effect operations).
#[test]
#[ignore = "Handler dispatch type mismatch - see plan file for details"]
fn test_handler_direct_result() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn no_effects() ->{State(Int)} Int {
    42
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() -> Int {
    run_state(fn() { no_effects() }, 0)
}
"#;
    let result = compile_and_run(code, "handler_direct_result.trb");
    assert_eq!(result, 42, "Expected main to return 42, got {}", result);
}
