//! Tests for lambda effect type propagation to IR.
//!
//! These tests verify that lambda expressions with effects have their
//! effect types correctly propagated to the lifted IR functions.
//!
//! This is critical for the evidence pass to correctly identify which
//! lifted lambdas need evidence parameters.

mod common;

use self::common::run_ast_pipeline_with_ir;
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn assert_shared_identity_done_k(ir_text: &str, expected_references: usize) {
    let lines: Vec<_> = ir_text.lines().collect();
    let identity_done_k_headers: Vec<_> = lines
        .windows(2)
        .filter_map(|pair| {
            let header = pair[0].trim_start();
            (header.starts_with("func.func ")
                && header.contains("%2: tribute_rt.anyref) -> tribute_rt.anyref")
                && pair[1].trim() == "func.return %2")
                .then_some(header)
        })
        .collect();
    assert_eq!(
        identity_done_k_headers.len(),
        1,
        "identity done_k should have one function definition per compilation unit"
    );

    let identity_done_k_symbol = identity_done_k_headers[0]
        .strip_prefix("func.func ")
        .expect("identity done_k header should start with func.func")
        .split('(')
        .next()
        .expect("identity done_k header should contain a parameter list");
    let identity_done_k_ref = format!("func_ref = {identity_done_k_symbol}");
    assert_eq!(
        ir_text.matches(&identity_done_k_ref).count(),
        expected_references,
        "all identity done_k closures should reference the shared function"
    );
}

// ========================================================================
// Pure Lambda Tests - No Effect Expected
// ========================================================================

/// Test that a pure lambda (no effects) is lifted without effect type.
#[salsa_test]
fn test_pure_lambda_no_effect(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }

fn main() -> Int {
    apply(fn(n) { n + 1 }, 41)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test that a pure lambda capturing a variable has no effect type.
#[salsa_test]
fn test_pure_lambda_with_capture_no_effect(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn apply(f: fn(Int) -> Int, x: Int) -> Int { f(x) }

fn main() -> Int {
    let offset = 10
    apply(fn(n) { n + offset }, 32)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Effectful Lambda Tests - Effect Type Expected
// ========================================================================

/// Test that a lambda directly calling an ability operation has effect type.
///
/// The lambda `fn() { State::get() }` should have `State(Int)` effect
/// in its lifted function type.
#[salsa_test]
fn test_effectful_lambda_direct_ability_call(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        op State::get() { resume 42 }
        op State::set(v) { resume Nil }
    }
}

fn main() -> Int {
    run_with_state(fn() { State::get() })
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test that a lambda calling an effectful function inherits the effect.
///
/// The lambda `fn() { counter() }` should have `State(Int)` effect
/// because `counter` has that effect.
#[salsa_test]
fn test_effectful_lambda_indirect_effect_call(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn counter() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + 1)
    n
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        op State::get() { resume 0 }
        op State::set(v) { resume Nil }
    }
}

fn main() -> Int {
    run_with_state(fn() { counter() })
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test that multiple ability operations in lambda accumulate effects.
#[salsa_test]
fn test_effectful_lambda_multiple_operations(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        op State::get() { resume 0 }
        op State::set(v) { resume Nil }
    }
}

fn main() -> Int {
    run_with_state(fn() {
        let n = State::get()
        State::set(n + 1)
        n
    })
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Handler Arm Lambda Tests - Core Ability Pattern
// ========================================================================

/// Test handler arm lambdas that call continuations.
///
/// This is the core pattern from ability_core.trb:
/// `op State::get() { run_state(fn() { resume init }, init) }`
///
/// The lambda `fn() { resume init }` should preserve the effect row variable `e`
/// from the outer handler context.
#[salsa_test]
fn test_handler_arm_continuation_lambda(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() -> Int { 0 }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Test the full ability_core pattern with multiple counter calls.
#[salsa_test]
fn test_ability_core_full_pattern(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
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
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() -> Int {
    run_state(fn() {
        counter()
        counter()
        counter()
    }, 0)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_shared_identity_done_k(&ir_text, 6);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Nested Lambda Tests
// ========================================================================

/// Test nested lambdas where inner lambda has effect.
#[salsa_test]
fn test_nested_lambda_inner_effectful(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
}

fn run_with_state(f: fn() ->{State(Int)} Int) -> Int {
    handle f() {
        do result { result }
        op State::get() { resume 99 }
    }
}

fn apply_thunk(f: fn() -> Int) -> Int { f() }

fn main() -> Int {
    apply_thunk(fn() {
        run_with_state(fn() { State::get() })
    })
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Effect Row Variable Tests
// ========================================================================

/// Test that effect row variables are properly unified.
///
/// When a lambda is passed to a function expecting `fn() ->{e, State(s)} a`,
/// the lambda's effect should include State(s) with the row variable e.
#[salsa_test]
fn test_lambda_effect_row_unification(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn with_state(f: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle f() {
        do result { result }
        op State::get() { with_state(fn() { resume init }, init) }
        op State::set(v) { with_state(fn() { resume Nil }, v) }
    }
}

fn main() -> Nat {
    with_state(fn() {
        State::get()
    }, 42)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
