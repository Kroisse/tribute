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
//!
//! ## Blocking Issues
//!
//! Many tests are blocked by:
//! - **WASM backend: unrealized_conversion_cast failures**: The WASM lowering
//!   pipeline cannot resolve certain type casts (especially `core.array` types).
//! - **Type validation not enforced**: Parameterized ability type argument
//!   validation (e.g., `State(Int)` vs `State(Bool)`) is not yet implemented.
//! - **Effect checking not enforced**: Missing effect annotations don't produce
//!   errors yet (see #112).

mod common;

// TODO: Re-enable once WASM backend issues are resolved
#[allow(unused_imports)]
use common::run_wasm;
use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::compile_to_wasm_binary;
use tribute_front::SourceCst;

/// Helper to compile code and collect diagnostics using CLI pipeline.
fn compile_and_check(code: &str, name: &str) -> Vec<tribute_passes::diagnostic::Diagnostic> {
    use tribute_passes::diagnostic::Diagnostic;

    let source_code = Rope::from_str(code);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, name, source_code.clone(), tree);

        compile_to_wasm_binary::accumulated::<Diagnostic>(db, source_file)
            .into_iter()
            .cloned()
            .collect()
    })
}

/// Helper to print diagnostics concisely (truncating long messages).
fn print_diagnostics(diagnostics: &[tribute_passes::diagnostic::Diagnostic]) {
    for diag in diagnostics {
        let msg = if diag.message.len() > 200 {
            format!(
                "{}... [truncated {} chars]",
                &diag.message[..200],
                diag.message.len() - 200
            )
        } else {
            diag.message.clone()
        };
        eprintln!("[{:?}] {}: {}", diag.severity, diag.phase, msg);
    }
}

/// Helper to compile code to WASM and run it.
fn compile_and_run(code: &str, name: &str) -> i32 {
    let source_code = Rope::from_str(code);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, name, source_code.clone(), tree);

        let wasm_binary = compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");

        run_wasm::<i32>(wasm_binary.bytes(db))
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

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "ability_def.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Expected no errors, got {} diagnostics",
        diagnostics.len()
    );
}

/// Test ability operations with effect annotations.
#[test]
fn test_ability_operation_with_effect() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Nat)} Nat {
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "ability_effect.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Expected no errors, got {} diagnostics",
        diagnostics.len()
    );
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

    let diagnostics = compile_and_check(code, "handle_expr.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Expected no errors, got {} diagnostics",
        diagnostics.len()
    );
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

fn counter() ->{State(Nat)} Nat {
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

fn main() -> Nat {
    run_state(fn() {
        counter()
        counter()
        counter()
    }, 0)
}
"#;

    let diagnostics = compile_and_check(code, "milestone_100.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Milestone target code should compile without errors, got {} diagnostics",
        diagnostics.len()
    );
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
    fn print(msg: String) -> Nil
}

fn stateful_print() ->{State(Nat), Console} Nat {
    Console::print("hello")
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "effect_row.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Expected no errors, got {} diagnostics",
        diagnostics.len()
    );
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

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "multiple_abilities.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Expected no errors, got {} diagnostics",
        diagnostics.len()
    );
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

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "let_effect_propagation.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Effect should propagate through let binding, got {} diagnostics",
        diagnostics.len()
    );
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

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "multiple_let_effects.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Multiple effects should accumulate from let bindings, got {} diagnostics",
        diagnostics.len()
    );
}

/// Test that sequential let bindings with effects work correctly (Phase 1-2 compatible).
#[test]
fn test_sequential_let_bindings_with_effects() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn sequential_state() ->{State(Nat)} Nat {
    let a = State::get()
    let b = State::get()
    let c = a + b
    c + 1
}

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "sequential_let_effects.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Sequential let bindings should propagate effects, got {} diagnostics",
        diagnostics.len()
    );
}

/// Test that nested block let bindings with effects work correctly.
/// Note: This now compiles successfully with the current pipeline.
#[test]
fn test_nested_let_bindings_with_effects() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn nested_state() ->{State(Nat)} Nat {
    let a = State::get()
    let b = {
        let c = State::get()
        c + 1
    }
    a + b
}

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "nested_let_effects.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Nested let bindings should propagate effects, got {} diagnostics",
        diagnostics.len()
    );
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

    let diagnostics = compile_and_check(code, "pure_let_binding.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Pure let bindings should not introduce effects, got {} diagnostics",
        diagnostics.len()
    );
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

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "unhandled_effect.trb");

    // Should have an error about unhandled effect
    assert!(
        !diagnostics.is_empty(),
        "Expected error for unhandled effect"
    );
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
#[ignore = "WASM backend: ability runtime not yet functional"]
fn test_ability_core_execution() {
    let code = include_str!("../lang-examples/ability_core.trb");
    let result = compile_and_run(code, "ability_core.trb");
    assert_eq!(result, 2, "Expected main to return 2, got {}", result);
}

/// Test simple State::get handler that returns a constant.
#[test]
#[ignore = "WASM backend: ability runtime not yet functional"]
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
#[ignore = "WASM backend: ability runtime not yet functional"]
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
#[ignore = "WASM backend: ability runtime not yet functional"]
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
#[ignore = "WASM backend: ability runtime not yet functional"]
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

// =============================================================================
// Duplicate Handler Tests (Code Review Fix)
// =============================================================================

/// Test that duplicate effect handlers for the same ability compile correctly.
///
/// When a handler has multiple arms handling the same ability (e.g., State::get
/// and State::set both from State), the handled_abilities list may contain
/// duplicates. The deduplication fix ensures constraint generation doesn't fail.
#[test]
fn test_duplicate_ability_handlers_compile() {
    // This code has two handlers for the same ability (State)
    // Previously, this could cause constraint issues due to duplicate entries
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn use_state() ->{State(Nat)} Nat {
    let n = State::get()
    State::set(n + 1)
    n
}

fn main() -> Nat {
    handle use_state() {
        { result } -> result
        { State::get() -> k } -> k(42)
        { State::set(v) -> k } -> k(Nil)
    }
}
"#;

    let diagnostics = compile_and_check(code, "duplicate_handlers.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Duplicate ability handlers should compile without errors, got {} diagnostics",
        diagnostics.len()
    );
}

// =============================================================================
// Parameterized Ability Type Distinction Tests
// =============================================================================

/// Test that State(Int) and State(Bool) are treated as distinct abilities.
/// A function with State(Int) effect cannot be called where State(Bool) is expected.
#[test]
#[ignore = "Type validation: State(Int) vs State(Bool) not yet enforced"]
fn test_parameterized_ability_distinct_types() {
    // This should produce a type error: State(Int) is not State(Bool)
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn use_int_state() ->{State(Int)} Int {
    State::get()
}

// This should fail: use_int_state has State(Int), but we're in State(Bool) context
fn wrapper() ->{State(Bool)} Int {
    use_int_state()
}

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "param_ability_distinct.trb");

    // Should have a type error due to State(Int) != State(Bool)
    assert!(
        !diagnostics.is_empty(),
        "Expected type error for State(Int) vs State(Bool) mismatch"
    );

    // Verify it's a row mismatch or type mismatch error
    let has_type_error = diagnostics.iter().any(|d| {
        let msg = format!("{:?}", d);
        msg.contains("Mismatch") || msg.contains("mismatch")
    });
    assert!(
        has_type_error,
        "Expected row/type mismatch error, got: {:?}",
        diagnostics
    );
}

/// Test that State(Int) and State(Int) are the same ability and unify correctly.
#[test]
#[ignore = "WASM backend: ability runtime not yet functional"]
fn test_parameterized_ability_same_type_unifies() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn use_state_int() ->{State(Int)} Int {
    State::get()
}

fn wrapper() ->{State(Int)} Int {
    use_state_int()
}

fn main() -> Int {
    handle wrapper() {
        { result } -> result
        { State::get() -> k } -> k(42)
        { State::set(v) -> k } -> k(Nil)
    }
}
"#;

    let diagnostics = compile_and_check(code, "param_ability_same.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Same parameterized abilities should unify without errors, got {} diagnostics",
        diagnostics.len()
    );
}

/// Test type variable unification in ability args: State(?a) unifies with State(Int).
#[test]
#[ignore = "WASM backend: ability runtime not yet functional"]
fn test_parameterized_ability_type_var_unification() {
    // Generic function with State(s) should unify with concrete State(Int)
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn use_state_generic() ->{State(s)} s {
    State::get()
}

fn use_state_int() ->{State(Int)} Int {
    use_state_generic()
}

fn main() -> Int {
    handle use_state_int() {
        { result } -> result
        { State::get() -> k } -> k(100)
        { State::set(v) -> k } -> k(Nil)
    }
}
"#;

    let diagnostics = compile_and_check(code, "param_ability_typevar.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Type variable in ability arg should unify, got {} diagnostics",
        diagnostics.len()
    );
}

/// Test arity mismatch: State(Int) vs State() should be an error.
#[test]
#[ignore = "Type validation: arity mismatch not yet enforced"]
fn test_parameterized_ability_arity_mismatch() {
    // This is invalid: ability State(s) requires one type argument
    let code = r#"ability State(s) {
    fn get() -> s
}

// Missing type argument - should be State(Int) or similar
fn bad_func() ->{State()} Int {
    State::get()
}

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "param_ability_arity.trb");

    // Should have an arity mismatch error
    assert!(
        !diagnostics.is_empty(),
        "Expected arity mismatch error for State() vs State(s)"
    );
}

// =============================================================================
// Parameterized Ability Type Argument Preservation Tests
// =============================================================================

/// Test that handle expressions preserve parameterized ability type arguments.
///
/// When handling State(Int), the type argument Int should be preserved in the
/// effect row constraint, not lost by creating Effect entries with empty args.
#[test]
fn test_handle_preserves_parameterized_ability_type_args() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn use_state() ->{State(Nat)} Nat {
    State::set(10)
    State::get()
}

fn main() -> Nat {
    handle use_state() {
        { result } -> result
        { State::get() -> k } -> k(42)
        { State::set(v) -> k } -> k(Nil)
    }
}
"#;

    let diagnostics = compile_and_check(code, "handle_param_ability.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Handle should preserve parameterized ability type args, got {} diagnostics",
        diagnostics.len()
    );
}

/// Test that ability operations substitute type parameters into their signature.
///
/// When calling State::get() with State(Int), the return type should be Int,
/// not the unsubstituted type parameter `s`.
#[test]
fn test_ability_op_substitutes_type_params() {
    let code = r#"ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn use_state() ->{State(Nat)} Nat {
    let x = State::get()
    x + 1
}

fn main() -> Nat { 0 }
"#;

    let diagnostics = compile_and_check(code, "ability_op_subst.trb");

    print_diagnostics(&diagnostics);

    assert!(
        diagnostics.is_empty(),
        "Ability op should substitute type params into signature, got {} diagnostics",
        diagnostics.len()
    );
}
