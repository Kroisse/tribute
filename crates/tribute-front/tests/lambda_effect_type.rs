//! Tests for lambda effect type propagation to IR.
//!
//! These tests verify that lambda expressions with effects have their
//! effect types correctly propagated to the lifted IR functions.
//!
//! This ensures the source-logical callable and lambda forms preserve the
//! checker-selected calling convention without frontend evidence parameters.

mod common;

use self::common::run_ast_pipeline_with_ir;
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

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

/// A pure body remains Direct even when root promotion makes the receiving
/// callback slot Cps; lowering inserts an explicit logical wrapper instead of
/// a control callable conversion cast.
#[salsa_test]
fn test_pure_lambda_contextualized_to_cps_slot_stays_direct(db: &salsa::DatabaseImpl) {
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
    let main = ir_text
        .split("tribute_control.func @main")
        .nth(1)
        .expect("fixture must lower main");
    assert!(
        main.matches("tribute_control.lambda(").count() == 2
            && main.contains("convention(direct)")
            && main.matches("convention(cps)").count() >= 2,
        "the pure closure and its structural Cps wrapper must remain distinct:\n{main}"
    );
    assert!(
        !main.contains("core.unrealized_conversion_cast"),
        "the Direct-to-Cps boundary must not use a control callable cast:\n{main}"
    );
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
    op get() -> s
    op set(value: s) -> Nil
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
    op get() -> s
    op set(value: s) -> Nil
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
    op get() -> s
    op set(value: s) -> Nil
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
    op get() -> s
    op set(value: s) -> Nil
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
    assert!(ir_text.contains("tribute_control.lambda"));
    assert!(!ir_text.contains("__tribute_cps_control"));
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
    op get() -> s
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
    op get() -> s
    op set(value: s) -> Nil
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

/// Contextual generic rows must retain one solved logical signature regardless
/// of harmless body wrapping.  In particular, lowering must not recover a
/// lambda result by recognizing a perform-shaped AST body.
#[salsa_test]
fn test_generic_effect_lambda_wrappers_share_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn with_state(f: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle f() {
        do result { result }
        op State::get() { with_state(fn() { resume init }, init) }
    }
}

fn main() -> Nat {
    let direct = with_state(fn() { State::get() }, 42)
    let empty_block = with_state(fn() { { State::get() } }, 42)
    let let_wrapped = with_state(fn() {
        let value = State::get()
        value
    }, 42)
    let case_wrapped = with_state(fn() {
        case State::get() {
            value -> value
        }
    }, 42)
    direct + empty_block + let_wrapped + case_wrapped
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_eq!(
        ir_text
            .match_indices("tribute_control.lambda() -> core.i32 convention(cps)")
            .count(),
        4,
        "every direct or wrapped generic State operation lambda must retain the same solved signature:\n{ir_text}"
    );
    let main = ir_text
        .split("tribute_control.func @main")
        .nth(1)
        .expect("fixture must lower root main");
    assert!(
        !main.contains("tribute_control.lambda() -> tribute_rt.anyref"),
        "the four main lambdas must not erase their result:\n{main}"
    );
}
