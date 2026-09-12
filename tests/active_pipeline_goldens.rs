//! Structural tests for the active ability lowering pipeline.
//!
//! The source-logical CPS route intentionally changes generated IR shape, so
//! these tests assert its stable contracts instead of legacy textual snapshots.

use itertools::Itertools;
use salsa_test_macros::salsa_test;
use tribute::Diagnostic;
use tribute::pipeline::{
    NativePipelineStage, OptimizationOptions, compile_with_diagnostics, dump_ir,
    dump_native_ir_at_stage,
};
use tribute_front::SourceCst;
use trunk_ir::printer::print_module;

fn assert_no_diagnostics(stage: &str, diagnostics: &[Diagnostic]) {
    assert!(
        diagnostics.is_empty(),
        "{stage} emitted diagnostics:\n{}",
        diagnostics.iter().format_with("\n", |diagnostic, f| {
            f(&format_args!(
                "  - [{}] {}",
                diagnostic.phase, diagnostic.inner.message
            ))
        })
    );
}

fn shared_pipeline_ir(db: &dyn salsa::Database, name: &str, code: &str) -> String {
    let source = SourceCst::from_source_str(db, name, code);
    let result = compile_with_diagnostics(db, source);
    assert_no_diagnostics("shared pipeline", &result.diagnostics);

    let (ctx, module) = result
        .module
        .expect("shared pipeline should produce a module when diagnostics are empty");
    print_module(&ctx, module.op())
}

fn native_pipeline_ir(db: &dyn salsa::Database, name: &str, code: &str) -> String {
    let source = SourceCst::from_source_str(db, name, code);
    let ir_text = dump_ir(db, source, true).expect("native pipeline dump should succeed");
    let diagnostics: Vec<Diagnostic> = dump_ir::accumulated::<Diagnostic>(db, source, true)
        .into_iter()
        .cloned()
        .collect();
    assert_no_diagnostics("native pipeline", &diagnostics);
    ir_text
}

fn assert_shared_cps_contract(ir_text: &str) {
    for required in [
        "__tribute_continuation_frame_",
        "func.func @main",
        "func.tail_call_indirect",
    ] {
        assert!(
            ir_text.contains(required),
            "shared source-logical route must contain `{required}`:\n{ir_text}"
        );
    }
    for forbidden in [
        "__tribute_cps_control",
        "ability.legacy_",
        "tribute_control.",
    ] {
        assert!(
            !ir_text.contains(forbidden),
            "shared source-logical route must not retain `{forbidden}`:\n{ir_text}"
        );
    }
}

fn assert_native_cps_root_contract(ir_text: &str) {
    for required in [
        "__tribute_continuation_frame_",
        "func.func @__tribute_cps_main",
        "func.func @__tribute_root_done_k",
        "func.func @__tribute_root_dispatch",
        "func.tail_call_indirect",
        "tribute.root_cps_call = true",
    ] {
        assert!(
            ir_text.contains(required),
            "native source-logical route must contain `{required}`:\n{ir_text}"
        );
    }
    for forbidden in [
        "__tribute_cps_control",
        "ability.legacy_",
        "tribute_control.",
    ] {
        assert!(
            !ir_text.contains(forbidden),
            "native source-logical route must not retain `{forbidden}`:\n{ir_text}"
        );
    }
}

const DIRECT_FN_SOURCE: &str = r#"
ability Console {
    fn read() -> Int
    fn print(value: Int) -> Nil
}

fn use_console() ->{Console} Int {
    let n = Console::read()
    Console::print(n)
    n
}

fn run() -> Int {
    handle use_console() {
        do result { result }
        fn Console::read() { +41 }
        fn Console::print(value) { Nil }
    }
}

fn main() {
    let _ = run()
}
"#;

const RESUMPTIVE_OP_SOURCE: &str = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn bump() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + +1)
    n
}

fn run_state() -> Int {
    handle bump() {
        do result { result }
        op State::get() { resume +10 }
        op State::set(value) { resume Nil }
    }
}

fn main() {
    let _ = run_state()
}
"#;

const MIXED_NESTED_SOURCE: &str = r#"
ability Console {
    fn read() -> Int
    fn print(value: Int) -> Nil
}

ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn step() ->{Console, State(Int)} Int {
    let base = Console::read()
    let current = State::get()
    State::set(current + base)
    Console::print(current)
    current + base
}

fn run_state_with_console() ->{Console} Int {
    handle step() {
        do result { result }
        op State::get() { resume +7 }
        op State::set(value) { resume Nil }
    }
}

fn run_all() -> Int {
    handle run_state_with_console() {
        do result { result }
        fn Console::read() { +3 }
        fn Console::print(value) { Nil }
    }
}

fn main() {
    let _ = run_all()
}
"#;

const FLOAT_COMPARISON_SOURCE: &str = r#"
fn main() {
    let a = 1.0
    let b = 2.0
    let _ = #(a == b, a != b, a < b, a <= b, a > b, a >= b)
}
"#;

#[salsa_test]
fn shared_pipeline_direct_fn_ability_call(db: &salsa::DatabaseImpl) {
    let ir_text = shared_pipeline_ir(db, "direct_fn.trb", DIRECT_FN_SOURCE);
    assert_shared_cps_contract(&ir_text);
    assert!(ir_text.contains("func.func @run"), "{ir_text}");
}

#[salsa_test]
fn shared_pipeline_resumptive_op_continuation(db: &salsa::DatabaseImpl) {
    let ir_text = shared_pipeline_ir(db, "resumptive_op.trb", RESUMPTIVE_OP_SOURCE);
    assert_shared_cps_contract(&ir_text);
    assert!(ir_text.contains("func.func @run_state"), "{ir_text}");
    assert!(ir_text.contains("__tribute_one_shot_state_"), "{ir_text}");
}

#[salsa_test]
fn shared_pipeline_mixed_nested_handler_boundary(db: &salsa::DatabaseImpl) {
    let ir_text = shared_pipeline_ir(db, "mixed_nested.trb", MIXED_NESTED_SOURCE);
    assert_shared_cps_contract(&ir_text);
    assert!(ir_text.contains("func.func @run_all"), "{ir_text}");
}

#[salsa_test]
fn shared_pipeline_float_comparison_predicates(db: &salsa::DatabaseImpl) {
    let ir_text = shared_pipeline_ir(db, "float_comparisons.trb", FLOAT_COMPARISON_SOURCE);
    assert_shared_cps_contract(&ir_text);
    for predicate in ["@oeq", "@une", "@olt", "@ole", "@ogt", "@oge"] {
        assert!(
            ir_text.contains(predicate),
            "missing {predicate}:\n{ir_text}"
        );
    }
}

#[salsa_test]
fn native_pipeline_direct_fn_ability_call_uses_cps_root_contract(db: &salsa::DatabaseImpl) {
    let ir_text = native_pipeline_ir(db, "direct_fn_native.trb", DIRECT_FN_SOURCE);
    assert_native_cps_root_contract(&ir_text);
    assert!(ir_text.contains("func.func @run"), "{ir_text}");
}

#[salsa_test]
fn native_pipeline_resumptive_op_continuation(db: &salsa::DatabaseImpl) {
    let ir_text = native_pipeline_ir(db, "resumptive_op_native.trb", RESUMPTIVE_OP_SOURCE);
    assert_native_cps_root_contract(&ir_text);
    assert!(ir_text.contains("__tribute_one_shot_state_"), "{ir_text}");
}

#[salsa_test]
fn native_pipeline_mixed_nested_handler_boundary(db: &salsa::DatabaseImpl) {
    let ir_text = native_pipeline_ir(db, "mixed_nested_native.trb", MIXED_NESTED_SOURCE);
    assert_native_cps_root_contract(&ir_text);
    assert!(ir_text.contains("func.func @run_all"), "{ir_text}");
}

#[salsa_test]
fn native_pre_backend_lowering_consumes_into_raw_without_call_provenance(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "into_raw_boundary.trb", DIRECT_FN_SOURCE);
    let ir = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterRcInsertion,
        OptimizationOptions::production(),
    )
    .expect("native pre-backend IR should be available");
    assert!(!ir.contains("tribute_rt.into_raw"), "{ir}");
    assert!(
        !ir.contains("native_evidence_closure_transfer_destinations"),
        "{ir}"
    );
}
