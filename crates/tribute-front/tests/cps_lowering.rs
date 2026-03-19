//! Tests for CPS-based effect handling in AST-to-IR lowering.
//!
//! Verifies that:
//! - `resume` expressions lower to `func.call_indirect` on the continuation
//! - Ability op calls in blocks produce `ability.perform` with CPS continuations
//! - Handle expressions produce `ability.handle_dispatch` with handler closures
//! - Nested ability op calls chain continuations correctly

mod common;

use self::common::{run_ast_pipeline_with_ir, source_from_str};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;

// ========================================================================
// Resume Expression Tests
// ========================================================================

/// `resume value` in an `op` handler arm should lower to `func.call_indirect`
/// on the continuation closure.
#[salsa_test]
fn test_resume_in_op_handler(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn run() -> Int {
    handle 42 {
        do result { result }
        op State::get() { resume +0 }
        op State::set(v) { resume Nil }
    }
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// CPS Block Lowering Tests
// ========================================================================

/// A single ability op call in a block should produce `ability.perform`
/// with a trivial identity continuation.
#[salsa_test]
fn test_single_ability_op_in_block(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// An ability op call followed by a pure expression should produce
/// `ability.perform` with a continuation that evaluates the remaining code.
#[salsa_test]
fn test_ability_op_then_pure_expr(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn increment() ->{State(Int)} Int {
    let n = State::get()
    n + 1
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Two sequential ability op calls should chain continuations:
/// the first continuation contains the second `ability.perform`.
#[salsa_test]
fn test_sequential_ability_ops(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn set_and_get() ->{State(Int)} Int {
    State::set(+42)
    State::get()
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Handle Expression Tests
// ========================================================================

/// A handle expression should produce `ability.handle_dispatch` with:
/// - A body closure wrapping the handled computation
/// - A handler dispatch closure with per-arm dispatch
/// - A dispatch body region with `ability.done` and `ability.suspend` ops
#[salsa_test]
fn test_handle_with_do_and_op_arms(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn run() -> Int {
    handle get_state() {
        do result { result }
        op State::get() { resume +42 }
        op State::set(v) { resume Nil }
    }
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// A handle expression with an `fn` (tail-resumptive) handler arm
/// should work without explicit `resume`.
#[salsa_test]
fn test_handle_with_fn_handler(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn run() -> Int {
    handle get_state() {
        do result { result }
        fn State::get() { +42 }
        fn State::set(v) { Nil }
    }
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

// ========================================================================
// Multi-arg Ability Op Tests
// ========================================================================

/// An ability op with multiple arguments should pack them into a tuple
/// before passing to `ability.perform`.
#[salsa_test]
fn test_multi_arg_ability_op(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
ability KV(k, v) {
    fn put(key: k, value: v) -> Nil
}

fn store() ->{KV(Int, Int)} Nil {
    KV::put(+1, +2)
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}
