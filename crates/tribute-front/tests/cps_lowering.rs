//! Tests for CPS-based effect handling in AST-to-IR lowering.
//!
//! Verifies that:
//! - `resume` expressions lower to `func.call_indirect` on the continuation
//! - Ability op calls in blocks produce `ability.perform` with CPS continuations
//! - Handle expressions produce `ability.handle_dispatch` with handler closures
//! - Nested ability op calls chain continuations correctly

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline_with_ir};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// A diagnosed CPS root must still leave valid IR: failed CPS lowering uses
/// the same Nil fallback as other direct-returning functions.
#[salsa_test]
fn test_root_main_cps_lowering_failure_still_returns(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn apply_open(value: Int, callback: fn(Int) -> Nil) -> Nil {
    callback(value)
}

fn main() {
    apply_open(State::get, fn(_) { Nil })
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let main = ir_text
        .split("func.func @main")
        .nth(1)
        .expect("root main must be lowered");
    assert!(
        main.contains("func.return"),
        "root main must retain a terminator after CPS lowering fails:\n{main}"
    );
}

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

/// Worker conventions start with concrete effects, then monotonically promote
/// bodies whose open callback evaluation requires CPS.  `forward_open` appears
/// before `apply_open` so this also proves named-call propagation needs a
/// fixed-point pass rather than a declaration-order scan.
#[salsa_test]
fn test_open_callback_workers_promote_to_cps(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn forward_open(value: a, callback: fn(a) -> b) -> b {
    apply_open(value, callback)
}

fn apply_open(value: a, callback: fn(a) -> b) -> b {
    callback(value)
}

fn pure(value: Int) -> Int { value }

fn apply_closed(value: Int, callback: fn(Int) ->{} Int) ->{} Int {
    callback(value)
}

fn main() {
    let _ = forward_open(+41, fn(value) { value })
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    for name in ["forward_open", "apply_open"] {
        let header = ir_text
            .lines()
            .find(|line| {
                line.trim_start()
                    .starts_with(&format!("func.func @{name}("))
            })
            .unwrap_or_else(|| panic!("missing lowered worker {name}"));
        assert!(
            header.contains("tribute.calling_convention = 2"),
            "{name} must be promoted to Cps:\n{header}"
        );
    }
    let pure_header = ir_text
        .lines()
        .find(|line| line.trim_start().starts_with("func.func @pure("))
        .expect("missing lowered pure worker");
    assert!(
        pure_header.contains("tribute.calling_convention = 0"),
        "pure worker must remain Direct:\n{pure_header}"
    );
    for name in ["apply_closed", "main"] {
        let header = ir_text
            .lines()
            .find(|line| {
                line.trim_start()
                    .starts_with(&format!("func.func @{name}("))
            })
            .unwrap_or_else(|| panic!("missing lowered worker {name}"));
        assert!(
            header.contains("tribute.calling_convention = 0"),
            "{name} must stay Direct:\n{header}"
        );
    }
}

/// An `Io` root keeps its EvidenceDirect ABI while the frontend delimiter
/// closes the CPS implementation convention of an open callback worker.
#[salsa_test]
fn test_open_callback_evidence_root_main_stays_evidence_direct(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
fn apply_open(value: a, callback: fn(a) -> b) -> b {
    callback(value)
}

fn main() ->{std::io::Io} Nil {
    let _ = apply_open(+41, fn(value) { value })
    Nil
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let main_header = ir_text
        .lines()
        .find(|line| line.trim_start().starts_with("func.func @main("))
        .expect("missing lowered root main");
    assert!(
        main_header.contains("tribute.calling_convention = 1"),
        "Io root main must remain EvidenceDirect, not Cps:\n{main_header}"
    );
}

/// A nested module's `main` is an ordinary function: it may have both a
/// source result and a non-Io residual effect without root-entry diagnostics.
#[salsa_test]
fn test_nested_main_is_not_subject_to_entrypoint_diagnostics(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

mod Nested {
    fn main() -> Int { State::get() }
}

fn main() { }
"#,
    );

    let diagnostics = ast_pipeline_error_messages(db, source);
    assert!(
        diagnostics.is_empty(),
        "nested main must not receive root-entry diagnostics: {diagnostics:?}"
    );
}

/// A nested open-callback `main` is promoted as an ordinary worker; only the
/// exact root entrypoint receives the frontend delimiter exemption.
#[salsa_test]
fn test_nested_open_callback_main_promotes_to_cps(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
mod Nested {
    fn apply_open(value: a, callback: fn(a) -> b) -> b {
        callback(value)
    }

    fn main() -> Int {
        apply_open(+41, fn(value) { value })
    }
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let nested_main = ir_text
        .lines()
        .find(|line| {
            line.trim_start()
                .starts_with("func.func @\"Nested::main\"(")
        })
        .expect("missing lowered Nested::main worker");
    assert!(
        nested_main.contains("tribute.calling_convention = 2"),
        "nested open-callback main must be promoted to Cps:\n{nested_main}"
    );
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
// Strict consumer and local-region regressions (#816)
// ========================================================================

/// A CPS producer in the scrutinee must resume the strict consumer outside
/// the selected arm; it must not be hoisted past `after`.
#[salsa_test]
fn test_cps_case_scrutinee_keeps_outer_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Flag {
    op read() -> Bool
}

fn effectful() ->{Flag} Bool { Flag::read() }
fn after(value: Nat) -> Nat { value + 1 }

fn run() ->{Flag} Nat {
    after(case effectful() {
        True -> 40
        False -> 0
    })
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// The right side of `||` is a selected strict region, but its resulting value
/// still has to flow into the surrounding strict consumer.
#[salsa_test]
fn test_cps_short_circuit_rhs_keeps_outer_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Flag {
    op read() -> Bool
}

fn effectful() ->{Flag} Bool { Flag::read() }
fn after(value: Bool) -> Bool { value }

fn run() ->{Flag} Bool { after(False || effectful()) }

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// Guards are evaluated only after their pattern matches and inside that arm's
/// local region.
#[salsa_test]
fn test_cps_case_guard_stays_in_matched_arm(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Flag {
    op read() -> Bool
}

fn effectful() ->{Flag} Bool { Flag::read() }

fn run(value: Bool) ->{Flag} Nat {
    case value {
        True if effectful() -> 1
        False -> 0
        True -> 2
    }
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// A computation inside a selected case arm of an `op` handler must preserve
/// the arm-local resume continuation.
#[salsa_test]
fn test_cps_handler_case_arm_resumes_local_continuation(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State {
    op get() -> Nat
}

ability Log {
    op value() -> Nat
}

fn read_log() ->{Log} Nat { Log::value() }

fn run() ->{Log} Nat {
    handle State::get() {
        do result { result }
        op State::get() {
            case True {
                True -> resume read_log()
                False -> resume 0
            }
        }
    }
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
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

/// A Direct parent consumes a handled result as an ordinary source value, even
/// when it is nested first in a constructor and then as a call argument.
/// The handle delimiter owns normalization of its body and arms exactly once.
#[salsa_test]
fn test_handle_is_source_value_in_strict_constructor_and_call_contexts(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

enum Boxed(a) {
    Boxed(a)
}

fn read() ->{State(Int)} Int { State::get() }

fn consume(value: Boxed(Int)) -> Int {
    case value { Boxed(result) -> result }
}

fn run() -> Int {
    consume(Boxed(handle read() {
        do result { result }
        op State::get() { resume +41 }
    }))
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// A normalized CPS parent does not normalize an already isolated handle
/// again; the continuation for `before` consumes the handle as a source value.
#[salsa_test]
fn test_handle_in_cps_parent_is_not_renormalized(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

ability Trace {
    op before() -> Nil
}

fn read() ->{State(Int)} Int { State::get() }

fn run() ->{Trace} Int {
    Trace::before()
    handle read() {
        do result { result }
        op State::get() { resume +41 }
    }
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// A handler arm may continue after `resume`: the delimited answer becomes the
/// resume expression's source value before the arm-local strict consumer runs.
#[salsa_test]
fn test_strict_consumer_after_resume_uses_arm_continuation(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn run() -> Int {
    handle State::get() {
        do result { result }
        op State::get() {
            let resumed = resume +41
            resumed + 1
        }
    }
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// A lambda created in an `op` arm captures the tagged resume local and still
/// invokes the arm-local continuation after the resume answer is recovered.
#[salsa_test]
fn test_handler_lambda_captures_resume_for_strict_consumer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn run() -> Int {
    handle State::get() {
        do result { result }
        op State::get() {
            let later = fn() { resume +41 }
            later() + 1
        }
    }
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// A CPS parent normalizes the nested direct case, short-circuit RHS, and
/// lambda body once. Their region helpers must retain that ownership instead
/// of re-entering raw value lowering.
#[salsa_test]
fn test_normalized_direct_regions_do_not_reenter_raw_lowering(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Trace {
    op before() -> Nil
}

ability Flag {
    op read() -> Bool
}

fn read() ->{Flag} Bool { Flag::read() }

enum Boxed(a) {
    Boxed(a)
}

fn run() ->{Trace} Boxed(fn() -> Bool) {
    Trace::before()
    // `fn` is atomic to normalization, so this constructor argument exercises
    // the mode-preserving Cons strict-child path from a normalized CPS parent.
    Boxed(fn() {
        case True {
            True -> False || handle read() {
                do result { result }
                op Flag::read() { resume True }
            }
            False -> False
        }
    })
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// Closure capture analysis must include a record spread base after a CPS
/// boundary. Downstream closure-lowering compatibility is covered in the root
/// integration suite.
#[salsa_test]
fn test_record_spread_capture_survives_cps_continuation(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability Trace {
    op before() -> Nil
}

struct Point { x: Int, y: Int }

fn run(point: Point) ->{Trace} fn() -> Point {
    Trace::before()
    fn() { Point { x: 1, ..point } }
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
}

/// One compact aggregate regression covers CPS values in tuple, record, and
/// constructor strict consumers without duplicating end-to-end fixtures.
#[salsa_test]
fn test_cps_values_flow_through_mixed_strict_aggregates(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
}

struct Point { x: Int, y: Int }

enum Boxed(a) {
    Boxed(a)
}

fn read() ->{State(Int)} Int { State::get() }

fn run() ->{State(Int)} Int {
    let base = Point { x: 0, y: 0 }
    let pair = #(read(), read())
    let point = Point { x: read(), ..base }
    let boxed = Boxed(read())
    0
}

fn main() { }
"#,
    );

    assert_snapshot!(run_ast_pipeline_with_ir(db, source));
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
