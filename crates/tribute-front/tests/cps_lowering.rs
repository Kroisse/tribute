//! Tests for source-logical control lowering in AST-to-IR.
//!
//! Verifies that:
//! - source calls, performs, handlers, and resumptions retain their logical forms;
//! - ordinary branches remain structured `scf` value control;
//! - strict source evaluation is ordered exactly once around logical control.

mod common;

use self::common::{ast_pipeline_error_messages, run_ast_pipeline_with_ir};
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

/// The former snapshots described frontend-owned physical CPS. The logical
/// boundary is asserted through canonical printed TrunkIR, while each fixture
/// below additionally asserts its own source-level ordering or region shape.
fn assert_logical_boundary(ir: &str) {
    assert!(
        ir.contains("tribute_control.func"),
        "source-logical lowering must define control callables:\n{ir}"
    );
    for forbidden in [
        "func.func",
        "func.call",
        "func.return",
        "func.unreachable",
        "closure.",
        "ability.legacy",
        "ability.handle",
        "effect.",
        "__tribute_cps_control",
        "core.func",
    ] {
        assert!(
            !ir.contains(forbidden),
            "source-logical lowering must not contain `{forbidden}`:\n{ir}"
        );
    }
}

fn assert_in_order(ir: &str, needles: &[&str]) {
    let mut start = 0;
    for needle in needles {
        let offset = ir[start..]
            .find(needle)
            .unwrap_or_else(|| panic!("missing `{needle}` in canonical IR:\n{ir}"));
        start += offset + needle.len();
    }
}

fn assert_occurrences(ir: &str, needle: &str, expected: usize) {
    assert_eq!(
        ir.match_indices(needle).count(),
        expected,
        "expected {expected} occurrence(s) of `{needle}`:\n{ir}"
    );
}

fn logical_function<'a>(ir: &'a str, name: &str) -> &'a str {
    let unquoted = format!("tribute_control.func @{name}(");
    let quoted = format!("tribute_control.func @\"{name}\"(");
    let start = ir
        .find(&unquoted)
        .or_else(|| ir.find(&quoted))
        .unwrap_or_else(|| panic!("missing logical function `{name}` in canonical IR:\n{ir}"));
    let function_and_following = &ir[start..];
    let end = function_and_following[1..]
        .find("\n  tribute_control.func ")
        .map_or(function_and_following.len(), |offset| offset + 1);
    &function_and_following[..end]
}

fn checked_logical_function<'a>(ir: &'a str, name: &str) -> &'a str {
    assert_logical_boundary(ir);
    logical_function(ir, name)
}

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

    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.iter().any(|error| error
            == "type error at call site in function 'main': expected `Int`, found `fn() -> _`"),
        "fixture must reach the diagnosed CPS-lowering recovery path: {errors:?}"
    );
    let ir_text = run_ast_pipeline_with_ir(db, source);
    let main = ir_text
        .split("tribute_control.func @main")
        .nth(1)
        .expect("root main must be lowered");
    assert!(
        main.contains("tribute_control.return"),
        "root main must retain a terminator after CPS lowering fails:\n{main}"
    );
}

// ========================================================================
// Resume Expression Tests
// ========================================================================

/// `resume value` in an `op` handler arm must remain a logical resume followed
/// by the handler-region yield.
#[salsa_test]
fn test_resume_in_op_handler(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run() ->{State(Int)} Int {
    handle State::get() {
        do result { result }
        op State::get() { resume +0 }
        op State::set(v) { resume Nil }
    }
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_resume_shape(checked_logical_function(&ir_text, "run"));
}

/// Resume materializes an erased literal to the token input and returns the
/// token answer, rather than using the literal or source-expression type.
#[salsa_test]
fn resume_uses_exact_token_input_and_answer_types(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
enum String {
    Leaf(Bytes),
    Branch(String, String, Nat),
}

ability Read {
    op read() -> String
}

fn run() -> Int {
    handle Read::read() {
        do result { 0 }
        op Read::read() {
            let answer = resume "resumed"
            answer + 1
        }
    }
}

fn main() { }
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let run = checked_logical_function(&ir, "run");
    assert!(
        run.contains("resume_token(!String, core.i32)"),
        "handler token must retain its exact input and answer types:\n{run}"
    );
    assert_in_order(
        run,
        &[
            "adt.string_const",
            "core.unrealized_conversion_cast",
            "tribute_control.resume",
        ],
    );
    let cast = run
        .lines()
        .find(|line| line.contains("core.unrealized_conversion_cast"))
        .expect("resume input materialization");
    assert!(cast.contains(": !String"), "{run}");
    let resume = run
        .lines()
        .find(|line| line.contains("= tribute_control.resume "))
        .expect("source-logical resume");
    assert!(resume.contains(": core.i32"), "{run}");
}

#[salsa_test]
fn handler_kind_mismatch_is_a_type_diagnostic_not_a_lowering_panic(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    fn get() -> s
}

fn run() -> Int {
    handle 42 {
        op State::get() { resume 0 }
    }
}
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("handler arm uses @Op")),
        "expected an operation-kind diagnostic, got {errors:#?}"
    );
}

/// Handler-operation metadata is fail-closed: a generic handled ability must
/// have exactly one body-effect instance, and the arm still diagnoses unknown
/// operations and incorrect source arity before logical lowering can run.
#[salsa_test]
fn handler_metadata_failures_remain_type_diagnostics(db: &salsa::DatabaseImpl) {
    let missing_instance = SourceCst::from_source_str(
        db,
        "missing_handler_instance.trb",
        r#"
ability State(s) {
    op get() -> s
}

fn run() -> Int {
    handle 0 {
        op State::get() { resume 0 }
    }
}
"#,
    );
    assert!(
        ast_pipeline_error_messages(db, missing_instance)
            .iter()
            .any(|error| error.contains("cannot determine the instantiated ability")),
        "a generic handler with no matching effect must be diagnosed"
    );

    let unknown_operation = SourceCst::from_source_str(
        db,
        "unknown_handler_operation.trb",
        r#"
ability State {
    op get() -> Int
}

fn run() -> Int {
    handle State::get() {
        op State::missing() { resume 0 }
    }
}
"#,
    );
    assert!(
        ast_pipeline_error_messages(db, unknown_operation)
            .iter()
            .any(|error| error.contains("unknown handler operation 'missing'")),
        "unknown handler operations must be diagnosed before lowering"
    );

    let wrong_arity = SourceCst::from_source_str(
        db,
        "handler_arity.trb",
        r#"
ability State {
    op get(value: Int) -> Int
}

fn run() -> Int {
    handle State::get(0) {
        op State::get() { resume 0 }
    }
}
"#,
    );
    assert!(
        ast_pipeline_error_messages(db, wrong_arity)
            .iter()
            .any(|error| error.contains("handler arm has 0 parameter(s)")),
        "handler parameter arity must be diagnosed before logical metadata is emitted"
    );
}

#[salsa_test]
fn logical_root_main_preserves_pure_and_io_conventions(db: &salsa::DatabaseImpl) {
    let pure = SourceCst::from_source_str(db, "pure.trb", "fn main() { }");
    let pure_ir = run_ast_pipeline_with_ir(db, pure);
    assert!(
        pure_ir.contains("tribute_control.func @main() -> core.nil convention(direct)"),
        "pure root main must be direct:\n{pure_ir}"
    );
    let io = SourceCst::from_source_str(db, "io.trb", "fn main() ->{std::io::Io} Nil { Nil }");
    let io_ir = run_ast_pipeline_with_ir(db, io);
    assert!(
        io_ir.contains("tribute_control.func @main() -> core.nil convention(evidence_direct)"),
        "Io root main must preserve EvidenceDirect:\n{io_ir}"
    );
}

#[salsa_test]
fn nested_main_is_not_a_root_convention_special_case(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State {
    op get() -> Int
}
mod Nested {
    fn main() ->{State} Int { State::get() }
}
fn main() { }
"#,
    );
    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert!(
        ir_text.contains("tribute_control.func @\"Nested::main\"() -> core.i32 convention(cps)"),
        "nested main must preserve its ordinary checker-selected CPS convention:\n{ir_text}"
    );
}

/// Source extern declarations stay source-logical callables, and direct calls
/// to them retain the declared symbol rather than introducing a physical ABI.
#[salsa_test]
fn logical_extern_declaration_and_call_preserve_source_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_extern.trb",
        r#"
extern "intrinsic" fn identity(value: Int) -> Int

fn call(value: Int) -> Int { identity(value) }

mod Native {
    extern "intrinsic" fn identity(value: Int) -> Int

    fn call(value: Int) -> Int { identity(value) }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    assert_logical_boundary(&ir);
    assert!(
        ir.contains(
            "tribute_control.func @identity(%arg0: core.i32) -> core.i32 convention(direct)"
        ),
        "extern declaration must retain its source-visible logical signature:\n{ir}"
    );
    let call = checked_logical_function(&ir, "call");
    assert!(
        call.contains("tribute_control.call %0 {callee = @identity} : core.i32"),
        "direct source call must target the logical extern declaration:\n{call}"
    );
    let nested_call = checked_logical_function(&ir, "Native::call");
    assert!(
        nested_call.contains("callee = @\"Native::identity\"")
            && ir.contains(
                "tribute_control.func @\"Native::identity\"(%arg0: core.i32) -> core.i32 convention(direct)"
            ),
        "nested extern calls must retain their qualified logical declaration:\n{nested_call}"
    );
}

/// Source value layouts stay logical even when they contain the primitive
/// values that are not represented by the old physical function carrier.
#[salsa_test]
fn logical_scalar_tuple_preserves_all_element_types(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_scalar_tuple.trb",
        r#"
fn scalars() -> #(Float, Bytes, Rune, Nil) {
    #(3.25, b"x", ?x, Nil)
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let scalars = checked_logical_function(&ir_text, "scalars");
    assert!(
        scalars.contains("adt.struct_new")
            && scalars.contains("core.f64")
            && scalars.contains("core.bytes")
            && scalars.contains("value = 120")
            && scalars.contains("rune")
            && scalars.contains("core.nil"),
        "scalar tuple layout must preserve each source value type:\n{scalars}"
    );
}

/// A `Never` operation result is the logical bottom type, rather than an
/// erased control carrier or frontend-owned completion value.
#[salsa_test]
fn logical_never_perform_preserves_bottom_result_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_never.trb",
        r#"
ability Abort {
    op abort() -> Never
}

fn terminate() ->{Abort} Never {
    Abort::abort()
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let terminate = checked_logical_function(&ir_text, "terminate");
    assert!(
        terminate.contains("tribute_control.perform") && terminate.contains(": core.never"),
        "a Never perform must retain its logical result type:\n{terminate}"
    );
}

/// Nested declarations keep their fully-qualified logical accessor symbols,
/// including callable field types, rather than relying on physical closure
/// extraction to recover a getter later.
#[salsa_test]
fn logical_nested_accessor_preserves_qualified_callable_signature(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_nested_accessor.trb",
        r#"
pub mod Nested {
    pub struct Holder { callback: fn(Int) -> Int }

    pub fn read(holder: Holder) -> fn(Int) -> Int {
        holder.callback
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let accessor = checked_logical_function(&ir_text, "Nested::Holder::callback");
    assert!(
        accessor.contains("tribute_control.func @\"Nested::Holder::callback\"")
            && accessor.contains("tribute_control.return"),
        "nested accessor must use its qualified logical symbol:\n{accessor}"
    );
    let read = checked_logical_function(&ir_text, "Nested::read");
    assert!(
        read.contains("callee = @\"Nested::Holder::callback\"")
            && read.contains("tribute_control.call"),
        "field access must call the matching logical accessor:\n{read}"
    );
    assert!(
        ir_text.contains("tribute_control.callable(core.i32, core.i32)")
            && !ir_text.contains("core.func"),
        "nested callable fields must remain source-logical:\n{ir_text}"
    );
}

/// A handler without a `do` arm yields the handled computation directly; the
/// missing completion arm is ordinary logical control, not a CPS fallback.
#[salsa_test]
fn logical_handle_without_do_arm_yields_handled_answer(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_handle_without_do.trb",
        r#"
ability State {
    fn get() -> Int
}

fn run() -> Int {
    handle State::get() {
        fn State::get() { +42 }
    }
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    let run = checked_logical_function(&ir_text, "run");
    assert!(
        run.contains("tribute_control.handle")
            && run.contains("kind = @fn")
            && run.contains("tribute_control.yield %2"),
        "a no-do handler must yield the logical handled answer:\n{run}"
    );
}

#[salsa_test]
fn perform_metadata_preserves_phantom_args(db: &salsa::DatabaseImpl) {
    let phantom = SourceCst::from_source_str(
        db,
        "phantom.trb",
        r#"
ability Phantom(p) {
    fn get() -> Int
}
fn read() ->{Phantom(Bool)} Int { Phantom::get() }
fn main() { }
"#,
    );
    let phantom_ir = run_ast_pipeline_with_ir(db, phantom);
    assert!(
        phantom_ir.contains("core.ability_ref(core.i1) {name = @Phantom}"),
        "perform must retain the phantom ability argument from its typed effect:\n{phantom_ir}"
    );
    let conflicting = SourceCst::from_source_str(
        db,
        "conflicting.trb",
        r#"
ability Same(a) {
    fn pair(a, a) -> Nil
}
fn bad() ->{Same(Int)} Nil { Same::pair(+1, true) }
"#,
    );
    assert!(
        !ast_pipeline_error_messages(db, conflicting).is_empty(),
        "repeated ability BoundVars with conflicting actual arguments must be rejected"
    );
}

// ========================================================================
// CPS Block Lowering Tests
// ========================================================================

/// A single ability operation must retain its exact logical operation kind.
#[salsa_test]
fn test_single_ability_op_in_block(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn get_state() ->{State(Int)} Int {
    State::get()
}

fn main() { }
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_single_perform_shape(checked_logical_function(&ir_text, "get_state"));
}

/// An ability call followed by a pure expression must return only after the
/// logical perform has produced its source value.
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
    assert_perform_then_value_shape(checked_logical_function(&ir_text, "get_value"));
}

/// Two sequential ability calls must stay in source evaluation order.
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
    assert_sequential_performs_shape(checked_logical_function(&ir_text, "set_and_get"));
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
    assert_nested_argument_shape(checked_logical_function(&ir_text, "run"));
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
    assert_indirect_lambda_shape(checked_logical_function(&ir_text, "run"));
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

fn pure(value: Int) ->{} Int { value }

fn apply_closed(value: Int, callback: fn(Int) ->{} Int) ->{} Int {
    callback(value)
}

fn evidence() ->{std::io::Io} Nil { Nil }

struct Box(a) { value: a }
fn unwrap(value: Box(a)) ->{} a { value.value }

mod Nested {
    struct Box(a) { value: a }
    fn unwrap(value: Box(a)) ->{} a { value.value }
    fn use_nested() ->{} Int { unwrap(Box { value: +41 }) }
}

fn main() {
    let _ = forward_open(+41, fn(value) { value })
    let _ = unwrap(Box { value: +1 })
    let _ = Nested::use_nested()
}
"#,
    );

    let ir_text = run_ast_pipeline_with_ir(db, source);
    for name in ["forward_open", "apply_open"] {
        let header = ir_text
            .lines()
            .find(|line| {
                line.trim_start()
                    .starts_with(&format!("tribute_control.func @{name}("))
            })
            .unwrap_or_else(|| panic!("missing lowered worker {name}"));
        assert!(
            header.contains("convention(cps)"),
            "{name} must be promoted to Cps:\n{header}"
        );
    }
    let pure_header = ir_text
        .lines()
        .find(|line| line.trim_start().starts_with("tribute_control.func @pure("))
        .expect("missing lowered pure worker");
    assert!(
        pure_header.contains("convention(direct)"),
        "pure worker must remain Direct:\n{pure_header}"
    );
    let apply_closed_header = ir_text
        .lines()
        .find(|line| {
            line.trim_start()
                .starts_with("tribute_control.func @apply_closed(")
        })
        .expect("missing lowered apply_closed worker");
    assert!(
        apply_closed_header.contains("convention(direct)"),
        "apply_closed must stay Direct:\n{apply_closed_header}"
    );
    let evidence_header = ir_text
        .lines()
        .find(|line| {
            line.trim_start()
                .starts_with("tribute_control.func @evidence(")
        })
        .expect("missing lowered evidence worker");
    assert!(
        evidence_header.contains("convention(evidence_direct)"),
        "evidence worker must stay EvidenceDirect:\n{evidence_header}"
    );
    let main = checked_logical_function(&ir_text, "main");
    let main_header = main.lines().next().expect("logical function has a header");
    assert!(
        main_header.contains("convention(cps)"),
        "root main worker must be promoted to Cps:\n{main_header}"
    );
    assert!(
        main_header.contains("tribute.root_export_convention = 0"),
        "pure root export must stay Direct:\n{main_header}"
    );
    assert!(
        main_header.contains("tribute.root_source_result = core.nil"),
        "root source result must stay core.nil:\n{main_header}"
    );
    assert!(
        main.contains("tribute_control.call"),
        "root main must contain the direct worker call:\n{main}"
    );
    assert!(
        main.contains("callee = @forward_open"),
        "root main must call the promoted forward_open worker:\n{main}"
    );
    assert!(
        !ir_text.contains("Nested::Nested::"),
        "nested workers must not be requalified:\n{ir_text}"
    );
    assert!(
        logical_function(&ir_text, "Nested::unwrap").contains("callee = @\"Nested::Box::value\""),
        "nested worker must call the exact qualified accessor:\n{ir_text}"
    );
    assert!(
        ir_text.lines().any(|line| line
            .trim_start()
            .starts_with("tribute_control.func @\"Box::value\"")),
        "specialized accessor must keep its exact root identity:\n{ir_text}"
    );
}

/// An `Io` root retains EvidenceDirect export metadata while its worker is
/// promoted for an open-callback call.
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
        .find(|line| line.trim_start().starts_with("tribute_control.func @main("))
        .expect("missing lowered root main");
    assert!(
        main_header.contains("convention(cps)")
            && main_header.contains("tribute.root_export_convention = 1"),
        "Io root export must remain EvidenceDirect while its worker is Cps:\n{main_header}"
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
                .starts_with("tribute_control.func @\"Nested::main\"(")
        })
        .expect("missing lowered Nested::main worker");
    assert!(
        nested_main.contains("convention(cps)"),
        "nested open-callback main must be promoted to Cps:\n{nested_main}"
    );
}

/// Nested function references must be resolved before lowering; a same-named
/// local shadows the nested definition and therefore remains an indirect call.
#[salsa_test]
fn nested_function_resolution_respects_local_shadowing(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "shadowing.trb",
        r#"
mod Nested {
    fn apply(value: Int) ->{} Int { value }

    fn main() -> Int {
        let apply = fn(value) { value }
        apply(+41)
    }
}

fn main() { }
"#,
    );
    let ir = run_ast_pipeline_with_ir(db, source);
    assert_logical_boundary(&ir);
    assert!(
        ir.contains("tribute_control.func @\"Nested::apply\""),
        "nested declaration must be present:\n{ir}"
    );
    assert!(
        ir.contains("tribute_control.call_indirect"),
        "shadowing local must stay an indirect call:\n{ir}"
    );
    assert!(
        !ir.contains("tribute_control.call {callee = @\"Nested::apply\"}"),
        "lowering must not reinterpret the local as the nested function:\n{ir}"
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
    assert_case_arm_shape(checked_logical_function(&ir_text, "run"));
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
    assert_short_circuit_shape(checked_logical_function(&ir_text, "run"));
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
    assert_handle_body_shape(checked_logical_function(&ir_text, "run"));
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
    assert_handler_arm_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_case_scrutinee_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_short_circuit_consumer_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_case_guard_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_handler_case_resume_shape(checked_logical_function(&ir_text, "run"));
}

// ========================================================================
// Handle Expression Tests
// ========================================================================

/// A handle expression must contain a logical handler with `@op` arm metadata
/// and a region-local resumption.
#[salsa_test]
fn test_handle_with_do_and_op_arms(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "test.trb",
        r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
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
    assert_op_handler_shape(checked_logical_function(&ir_text, "run"));
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
    assert_fn_handler_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_handle_strict_value_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_nested_handle_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_resume_then_suffix_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_resume_capture_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_normalized_region_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_record_spread_shape(checked_logical_function(&ir_text, "run"));
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

    let ir_text = run_ast_pipeline_with_ir(db, source);
    assert_aggregate_shape(checked_logical_function(&ir_text, "run"));
}

/// Record spreads evaluate their base before explicit fields, even though the
/// final `adt.struct_new` operands are arranged in declaration layout order.
#[salsa_test]
fn logical_record_spread_evaluates_base_before_explicit_fields(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "record_spread_order.trb",
        r#"
ability Trace {
    op spread() -> Pair
    op field() -> Int
}

struct Pair { left: Int, right: Int }

fn run() ->{Trace} Pair {
    Pair { right: Trace::field(), ..Trace::spread() }
}
"#,
    );
    let ir = run_ast_pipeline_with_ir(db, source);
    let run = checked_logical_function(&ir, "run");
    assert_in_order(
        run,
        &["op_name = @spread", "op_name = @field", "adt.struct_new"],
    );
    assert_occurrences(run, "op_name = @spread", 1);
    assert_occurrences(run, "op_name = @field", 1);
}

/// Logical aggregate construction and matching must preserve callable fields
/// recursively; no physical `core.func` carrier may enter tuple/list layouts.
#[salsa_test]
fn logical_callable_tuple_and_list_layouts_remain_distinct(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "callable_aggregates.trb",
        r#"
fn tuple_int(callback: fn(Int) -> Int) {
    let tuple = #(callback, True)
}
fn tuple_bool(callback: fn(Bool) -> Bool) {
    let tuple = #(callback, 0)
}

struct A {}
struct B {}
struct A__named_B {}

fn delimiter_pair(left: A, right: B) {
    let pair = #(left, right)
}
fn delimiter_spoof(value: A__named_B) {
    let singleton = #(value)
}

fn from_list(callback: fn(Int) -> Int) -> Int {
    case [callback] {
        [callback] -> callback(+1)
        _ -> +0
    }
}
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.is_empty(),
        "callable aggregate fixture must resolve: {errors:?}"
    );
    let ir = run_ast_pipeline_with_ir(db, source);
    assert_logical_boundary(&ir);
    assert!(ir.contains("tribute_control.callable(core.i32, core.i32)"));
    assert!(ir.contains("tribute_control.callable(core.i1, core.i1)"));
    let tuple_layout = |function: &str| {
        logical_function(&ir, function)
            .split("adt.struct_new")
            .nth(1)
            .and_then(|line| line.split("{type = !").nth(1))
            .and_then(|suffix| suffix.split('}').next())
            .expect("tuple fixture must construct a logical tuple")
    };
    let tuple_int_layout = tuple_layout("tuple_int");
    let tuple_bool_layout = tuple_layout("tuple_bool");
    assert_ne!(
        tuple_int_layout, tuple_bool_layout,
        "distinct callable tuple shapes must retain distinct layouts:\n{ir}"
    );
    for layout in [tuple_int_layout, tuple_bool_layout] {
        assert!(
            layout.starts_with("__logical_tuple_tuple_"),
            "callable tuple layout must retain the logical tuple prefix: {layout}"
        );
        assert!(
            !layout.contains('"') && !layout.contains("{:?}"),
            "callable tuple layout must not contain debug-formatted fragments: {layout}"
        );
    }
    assert_ne!(
        tuple_layout("delimiter_pair"),
        tuple_layout("delimiter_spoof"),
        "length-delimited logical keys must not alias delimiter-sensitive tuple shapes:\n{ir}"
    );
    assert!(
        !ir.contains("core.func"),
        "callable aggregate layouts must not use physical core.func:\n{ir}"
    );
}

/// Nominal logical layouts must share the same recursive callable conversion:
/// generated field accessors, enum construction, and enum pattern extraction
/// cannot reintroduce a physical `core.func` field.
#[salsa_test]
fn logical_nominal_callable_fields_use_control_callables(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "callable_nominals.trb",
        r#"
struct Holder { callback: fn(Int) -> Int }
enum Choice { Callback(#(fn(Int) -> Int, List(fn(Int) -> Int))), Other }

fn make(callback: fn(Int) -> Int) -> Choice { Callback(#(callback, [callback])) }
fn inspect(choice: Choice, fallback: fn(Int) -> Int) -> fn(Int) -> Int {
    case choice {
        Callback(#(callback, [nested])) -> nested
        Other -> fallback
    }
}
"#,
    );
    let ir = run_ast_pipeline_with_ir(db, source);
    assert_logical_boundary(&ir);
    let accessor = checked_logical_function(&ir, "Holder::callback");
    assert!(
        accessor.contains("-> !t") && ir.contains("= tribute_control.callable(core.i32, core.i32)"),
        "generated accessor must retain its callable field type:\n{accessor}"
    );
    let inspect = checked_logical_function(&ir, "inspect");
    assert!(
        inspect.contains("adt.variant_get")
            && inspect.contains("list.head")
            && inspect.contains("adt.struct_get"),
        "nested callable variant patterns must stay in the logical aggregate dialect:\n{inspect}"
    );
    assert!(inspect.contains("-> !t"));
    assert!(!ir.contains("core.func"));
}

/// Callable list patterns use the source-logical list layout through exact,
/// rest, and `as` branches; extracting a callable must not reconstruct a
/// physical function carrier.
#[salsa_test]
fn logical_callable_list_rest_pattern_preserves_element_type(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_callable_list_pattern.trb",
        r#"
fn first_or(values: List(fn(Int) -> Int), fallback: fn(Int) -> Int) -> fn(Int) -> Int {
    case values {
        [] as empty -> fallback
        [head, ..tail] as whole -> head
    }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let first_or = checked_logical_function(&ir, "first_or");
    assert!(
        first_or.contains("list.is_empty")
            && first_or.contains("list.head")
            && first_or.contains("list.tail")
            && ir.contains("tribute_control.callable(core.i32, core.i32)"),
        "callable list pattern extraction must retain its logical element type:\n{first_or}"
    );
    assert!(!first_or.contains("core.func"));
}

/// Nonempty list and tuple pattern checks both extract callable values through
/// the logical aggregate layouts before choosing the corresponding `scf` arm.
#[salsa_test]
fn logical_callable_aggregate_checks_preserve_pattern_order(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_callable_aggregate_checks.trb",
        r#"
fn first_nonempty(values: List(fn(Int) -> Int), fallback: fn(Int) -> Int) -> fn(Int) -> Int {
    case values {
        [head, ..tail] -> head
        [] -> fallback
    }
}

fn choose_pair(pair: #(fn(Int) -> Int, Bool), fallback: fn(Int) -> Int) -> fn(Int) -> Int {
    case pair {
        #(callback, True) -> callback
        #(callback, False) -> callback
    }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let list_case = checked_logical_function(&ir, "first_nonempty");
    assert!(
        list_case.contains("list.is_empty")
            && list_case.contains("list.head")
            && list_case.contains("list.tail")
            && list_case.contains("scf.if"),
        "nonempty callable list check must stay structured and evaluate once:\n{list_case}"
    );
    let tuple_case = checked_logical_function(&ir, "choose_pair");
    assert!(
        tuple_case.contains("adt.struct_get")
            && tuple_case.contains("arith.and")
            && tuple_case.contains("scf.if"),
        "callable tuple pattern check must stay in logical aggregate IR:\n{tuple_case}"
    );
    assert!(!ir.contains("core.func"));
}

/// Multi-field variants and exact list literals retain each nested pattern
/// check in ordinary structured control before their final exhaustive arms.
#[salsa_test]
fn logical_nested_pattern_checks_remain_structured(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_nested_pattern_checks.trb",
        r#"
enum Pair { Pair(Int, Int), None }

fn classify_pair(value: Pair) -> Int {
    case value {
        Pair(1, 2) -> 1
        Pair(_, _) -> 2
        None -> 0
    }
}

fn classify_list(values: List(Int)) -> Int {
    case values {
        [1, 2] -> 1
        _ -> 0
    }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let pair = checked_logical_function(&ir, "classify_pair");
    assert!(
        pair.contains("adt.variant_is")
            && pair.contains("adt.variant_cast")
            && pair.contains("arith.and"),
        "two-field variant checks must combine both source patterns:\n{pair}"
    );
    assert_occurrences(pair, "adt.variant_get", 8);

    let list = checked_logical_function(&ir, "classify_list");
    assert!(
        list.contains("list.is_empty")
            && list.contains("list.head")
            && list.contains("list.tail")
            && list.contains("scf.if"),
        "exact list patterns must remain nested structured checks:\n{list}"
    );
    assert_occurrences(list, "list.head", 4);
    assert_occurrences(list, "list.tail", 4);
}

/// A guarded handler arm resumes only in its selected case region; later
/// source arms remain distinct structured fallbacks.
#[salsa_test]
fn logical_guarded_handler_resume_stays_in_selected_region(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "logical_guarded_handler_resume.trb",
        r#"
ability State {
    op get() -> Nat
    op choose(pair: #(Nat, Nat), values: List(Nat)) -> Nat
}

ability Flag {
    op read() -> Bool
}

fn allowed() ->{Flag} Bool { Flag::read() }

fn run() ->{Flag} Nat {
    handle State::get() {
        do result { result }
        op State::get() {
            case True {
                True if allowed() -> resume 1
                False -> resume 2
                True -> resume 3
            }
        }
        op State::choose(pair, values) {
            case #(pair, values) {
                #(#(left, right), [head, ..tail]) if allowed() -> resume left
                _ -> resume 0
            }
        }
    }
}
"#,
    );

    let ir = run_ast_pipeline_with_ir(db, source);
    let run = checked_logical_function(&ir, "run");
    assert_in_order(
        run,
        &[
            "tribute_control.handler",
            "scf.if",
            "callee = @allowed",
            "tribute_control.resume",
        ],
    );
    assert_occurrences(run, "tribute_control.resume %", 8);
    assert_occurrences(run, "scf.if", 8);
    assert_in_order(
        run,
        &[
            "op_name = @choose",
            "adt.struct_get",
            "list.head",
            "callee = @allowed",
            "tribute_control.resume",
        ],
    );
}

/// Logical nominal layouts are collected before fields are converted, so a
/// recursive or forward field retains its nominal typeref instead of erasing
/// to `anyref` while the target layout is still incomplete.
#[salsa_test]
fn logical_nominal_layouts_preserve_recursive_and_forward_fields(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "recursive_nominals.trb",
        r#"
struct Node { next: Node }
struct First { second: Second }
struct Second { first: First }

fn keep_node(node: Node) -> Node { node }
fn keep_first(first: First) -> First { first }
"#,
    );
    let errors = ast_pipeline_error_messages(db, source);
    assert!(
        errors.is_empty(),
        "recursive nominal fixture must typecheck: {errors:?}"
    );
    let ir = run_ast_pipeline_with_ir(db, source);
    assert_logical_boundary(&ir);
    for field in [
        "!Node_1 = adt.struct() {fields = [[@next, !Node]], name = @Node}",
        "!First_1 = adt.struct() {fields = [[@second, !Second]], name = @First}",
        "!Second_1 = adt.struct() {fields = [[@first, !First]], name = @Second}",
    ] {
        assert!(
            ir.contains(field),
            "recursive/forward nominal field must retain `{field}`:\n{ir}"
        );
    }
    assert!(
        !ir.contains("[@next, tribute_rt.anyref]")
            && !ir.contains("[@second, tribute_rt.anyref]")
            && !ir.contains("[@first, tribute_rt.anyref]"),
        "recursive/forward nominal fields must not erase to anyref:\n{ir}"
    );
}

// ========================================================================
// Multi-arg Ability Op Tests
// ========================================================================

/// An ability operation with multiple arguments must retain each source value
/// as a separate logical perform operand.
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
    assert_multi_argument_shape(checked_logical_function(&ir_text, "store"));
}

/// Named semantic assertions replace the former physical-CPS snapshots. Each
/// test selects one directly, so its source-level contract stays visible.
fn assert_resume_shape(function: &str) {
    assert_in_order(
        function,
        &["tribute_control.resume", "tribute_control.yield"],
    );
}
fn assert_single_perform_shape(function: &str) {
    assert!(function.contains("operation_kind = @op"));
}
fn assert_perform_then_value_shape(function: &str) {
    assert_in_order(function, &["op_name = @get", "tribute_control.return"]);
}
fn assert_sequential_performs_shape(function: &str) {
    assert_in_order(function, &["op_name = @set", "op_name = @get"]);
}
fn assert_nested_argument_shape(function: &str) {
    assert_in_order(function, &["callee = @read", "callee = @add_one"]);
}
fn assert_exact_integer_addition(function: &str) -> &'static str {
    let callee = ["callee = @\"Int::+\"", "callee = @\"Nat::+\""]
        .into_iter()
        .find(|callee| function.contains(callee))
        .unwrap_or_else(|| {
            panic!("missing exact integer intrinsic call in canonical IR:\n{function}")
        });
    assert_in_order(function, &["tribute_control.call", callee]);
    assert!(
        !function.contains("arith.addi"),
        "source-logical lowering must preserve the exact intrinsic call instead of lowering by name:\n{function}"
    );
    callee
}
fn assert_indirect_lambda_shape(function: &str) {
    let callee = assert_exact_integer_addition(function);
    assert_in_order(
        function,
        &[
            "tribute_control.lambda",
            "tribute_control.call_indirect",
            callee,
        ],
    );
}
fn assert_case_arm_shape(function: &str) {
    let callee = assert_exact_integer_addition(function);
    assert_in_order(function, &["scf.if", "callee = @read", callee]);
}
fn assert_short_circuit_shape(function: &str) {
    assert_in_order(function, &["scf.if", "callee = @read"]);
}
fn assert_handle_body_shape(function: &str) {
    assert_in_order(
        function,
        &[
            "tribute_control.handle",
            "callee = @read",
            "tribute_control.resume",
        ],
    );
}
fn assert_handler_arm_shape(function: &str) {
    assert_in_order(
        function,
        &[
            "tribute_control.handler",
            "callee = @read_log",
            "tribute_control.resume",
        ],
    );
}
fn assert_case_scrutinee_shape(function: &str) {
    assert_in_order(
        function,
        &["callee = @effectful", "scf.if", "callee = @after"],
    );
}
fn assert_short_circuit_consumer_shape(function: &str) {
    assert_in_order(
        function,
        &["scf.if", "callee = @effectful", "callee = @after"],
    );
}
fn assert_case_guard_shape(function: &str) {
    assert_in_order(function, &["scf.if", "callee = @effectful"]);
}
fn assert_handler_case_resume_shape(function: &str) {
    assert_occurrences(function, "tribute_control.resume %", 2);
    assert!(function.contains("scf.if"));
}
fn assert_op_handler_shape(function: &str) {
    assert_in_order(
        function,
        &[
            "tribute_control.handle",
            "kind = @op",
            "tribute_control.resume",
        ],
    );
}
fn assert_fn_handler_shape(function: &str) {
    assert!(function.contains("kind = @fn"));
}
fn assert_handle_strict_value_shape(function: &str) {
    assert_in_order(
        function,
        &[
            "tribute_control.handle",
            "adt.variant_new",
            "tribute_control.call",
        ],
    );
}
fn assert_nested_handle_shape(function: &str) {
    assert_in_order(
        function,
        &["tribute_control.perform", "tribute_control.handle"],
    );
}
fn assert_resume_then_suffix_shape(function: &str) {
    let callee = assert_exact_integer_addition(function);
    assert_in_order(function, &["tribute_control.resume", callee]);
}
fn assert_resume_capture_shape(function: &str) {
    let callee = assert_exact_integer_addition(function);
    assert_in_order(
        function,
        &["tribute_control.lambda", "tribute_control.resume", callee],
    );
}
fn assert_normalized_region_shape(function: &str) {
    assert_in_order(
        function,
        &[
            "tribute_control.lambda",
            "scf.if",
            "tribute_control.handle",
            "adt.variant_new",
        ],
    );
}
fn assert_record_spread_shape(function: &str) {
    assert_in_order(
        function,
        &["tribute_control.lambda", "adt.struct_get", "adt.struct_new"],
    );
}
fn assert_aggregate_shape(function: &str) {
    assert_occurrences(function, "callee = @read", 4);
    assert_in_order(
        function,
        &["adt.struct_new", "adt.struct_new", "adt.variant_new"],
    );
}
fn assert_multi_argument_shape(function: &str) {
    assert!(function.contains("tribute_control.perform %0, %1"));
}
