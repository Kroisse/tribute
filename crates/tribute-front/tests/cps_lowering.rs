//! Tests for CPS-based effect handling in AST-to-IR lowering.
//!
//! Verifies that:
//! - `resume` expressions lower to `func.call_indirect` on the continuation
//! - Ability op calls in blocks produce `ability.perform` with CPS continuations
//! - Handle expressions produce `ability.handle_dispatch` with handler closures
//! - Nested ability op calls chain continuations correctly

mod common;

use self::common::run_ast_pipeline_with_ir;
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

// ========================================================================
// Resume Expression Tests
// ========================================================================

/// `resume value` in an `op` handler arm should lower to `func.call_indirect`
/// on the continuation closure.
#[salsa_test]
fn test_resume_in_op_handler(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_value() ->{State(Int)} Int {
    let n = State::get()
    n
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
    let source = SourceCst::from_source_str(
        db,
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

/// An effectful named call nested inside another call must be lifted into a
/// continuation before the outer expression is evaluated.
#[salsa_test]
fn test_nested_effectful_call_in_argument(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn read() ->{State(Int)} Int {
    State::get()
}

fn add_one(value: Int) -> Int {
    value + 1
}

fn run() ->{State(Int)} Int {
    add_one(read())
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Local closures use the CPS calling convention even when the call is nested
/// inside a larger expression.
#[salsa_test]
fn test_nested_effectful_closure_call(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn run() ->{State(Int)} Int {
    let read = fn() { State::get() }
    read() + 1
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// CPS lifting in a case arm stays inside that arm rather than executing
/// before the branch is selected.
#[salsa_test]
fn test_nested_effectful_call_in_case_arm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn read() ->{State(Int)} Int {
    State::get()
}

fn run(flag: Bool) ->{State(Int)} Int {
    case flag {
        True -> read() + 1
        False -> 0
    }
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Short-circuit RHS lowering must keep the effectful call inside the selected
/// region.
#[salsa_test]
fn test_nested_effectful_call_in_short_circuit_rhs(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Flag {
    op get() -> Bool
}

fn read() ->{Flag} Bool {
    Flag::get()
}

fn run() ->{Flag} Bool {
    False && read()
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// A nested effectful call in a handle body must use the handle body's local
/// continuation and stay inside the installed handler boundary.
#[salsa_test]
fn test_nested_effectful_call_in_handle_body(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn read() ->{State(Int)} Int {
    State::get()
}

fn run() -> Int {
    handle read() + 1 {
        do result { result }
        op State::get() { resume +41 }
    }
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_snapshot!(ir_text);
}

/// Handler operation arms use a region-local identity continuation for nested
/// effectful calls before resuming the captured continuation.
#[salsa_test]
fn test_nested_effectful_call_in_handler_arm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

ability Log {
    op value() -> Int
}

fn read_log() ->{Log} Int {
    Log::value()
}

fn run() ->{Log} Int {
    handle State::get() {
        do result { result }
        op State::get() {
            let value = read_log() + 1
            resume value
        }
    }
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
    let source = SourceCst::from_source_str(
        db,
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
