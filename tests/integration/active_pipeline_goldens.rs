//! Structural tests for the active ability lowering pipeline.
//!
//! These tests assert source-logical CPS signatures, transfer structure, and
//! target-owned root and closure-storage contracts.

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

// Snapshot semantic signature shapes and transfer counts, without SSA IDs,
// generated symbol numbering, or unrelated allocation/layout operations.
fn pipeline_contract_summary(ir_text: &str, native: bool) -> String {
    use std::collections::BTreeMap;
    use std::ops::ControlFlow;
    use tribute_core::calling_convention::{
        CPS_CONTINUATION_FRAME_RESULT_ATTR, CallingConvention, get_calling_convention,
    };
    use trunk_ir::dialect::func;
    use trunk_ir::ops::{DialectOp, DialectType};
    use trunk_ir::walk::{WalkAction, walk_op};
    use trunk_ir::{IrContext, TypeRef};

    fn type_shape(ctx: &IrContext, ty: TypeRef) -> String {
        let data = ctx.get_type(ty);
        if let Some(result) = data.attrs.get_type(CPS_CONTINUATION_FRAME_RESULT_ATTR) {
            return format!("Frame<{}>", type_shape(ctx, result));
        }
        let mut shape = format!("{}.{}", data.dialect, data.name);
        if let Some(name) = data.attrs.get_symbol("name") {
            shape.push_str(&format!("<{name}>"));
        }
        if !data.params.is_empty() {
            shape.push_str(&format!(
                "<{}>",
                data.params.iter().map(|&ty| type_shape(ctx, ty)).join(", ")
            ));
        }
        shape
    }

    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_module(&mut ctx, ir_text).expect("pipeline IR round trip");
    let mut summary = BTreeMap::new();
    let mut cps_functions = 0;
    let _ = walk_op::<()>(&ctx, module, &mut |op| {
        let Ok(function) = func::Func::from_op(&ctx, op) else {
            return ControlFlow::Continue(WalkAction::Advance);
        };
        let signature = func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx))
            .expect("exact callable signature");
        let convention = get_calling_convention(&ctx, op);
        if convention == Some(CallingConvention::Cps) {
            cps_functions += 1;
            let results = signature.results(&ctx);
            assert!(
                if native {
                    results.is_empty()
                } else {
                    results.len() == 1 && type_shape(&ctx, results[0]) == "core.never"
                },
                "CPS functions must have logical Never or empty physical results"
            );
        }
        let name = function.sym_name(&ctx).to_string();
        if !matches!(
            name.as_str(),
            "main"
                | "use_console"
                | "run"
                | "bump"
                | "run_state"
                | "step"
                | "run_state_with_console"
                | "run_all"
                | "__tribute_cps_main"
                | "__tribute_root_done_k"
                | "__tribute_root_dispatch"
        ) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        let inputs = signature
            .inputs(&ctx)
            .iter()
            .map(|&ty| type_shape(&ctx, ty))
            .join(", ");
        let results = signature
            .results(&ctx)
            .iter()
            .map(|&ty| type_shape(&ctx, ty))
            .join(", ");
        let mut transfers = BTreeMap::<String, usize>::new();
        let _ = walk_op::<()>(&ctx, op, &mut |nested| {
            let data = ctx.op(nested);
            if data.dialect == "func"
                && matches!(
                    data.name.to_string().as_str(),
                    "tail_call"
                        | "tail_call_indirect"
                        | "call"
                        | "call_indirect"
                        | "return"
                        | "unreachable"
                )
                || data.dialect == "effect"
            {
                if matches!(
                    data.name.to_string().as_str(),
                    "tail_call" | "tail_call_indirect"
                ) {
                    assert!(
                        ctx.op_result_types(nested).is_empty(),
                        "tail transfer has SSA results"
                    );
                }
                let key = format!(
                    "{}.{} operands={} results={}",
                    data.dialect,
                    data.name,
                    ctx.op_operands(nested).len(),
                    ctx.op_result_types(nested).len()
                );
                *transfers.entry(key).or_default() += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        summary.insert(
            name,
            format!(
                "{convention:?} ({inputs}) -> ({results})\n{}",
                transfers
                    .iter()
                    .map(|(shape, count)| format!("  {shape}: {count}"))
                    .join("\n")
            ),
        );
        ControlFlow::Continue(WalkAction::Skip)
    });
    assert!(cps_functions > 0, "fixture must exercise CPS");
    summary
        .iter()
        .map(|(name, contract)| format!("{name}: {contract}"))
        .join("\n")
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
    insta::assert_snapshot!(pipeline_contract_summary(&ir_text, false));
    assert!(ir_text.contains("func.func @run"), "{ir_text}");
}

#[salsa_test]
fn shared_pipeline_resumptive_op_continuation(db: &salsa::DatabaseImpl) {
    let ir_text = shared_pipeline_ir(db, "resumptive_op.trb", RESUMPTIVE_OP_SOURCE);
    assert_shared_cps_contract(&ir_text);
    insta::assert_snapshot!(pipeline_contract_summary(&ir_text, false));
    assert!(ir_text.contains("func.func @run_state"), "{ir_text}");
    assert!(ir_text.contains("__tribute_one_shot_state_"), "{ir_text}");
}

#[salsa_test]
fn shared_pipeline_mixed_nested_handler_boundary(db: &salsa::DatabaseImpl) {
    let ir_text = shared_pipeline_ir(db, "mixed_nested.trb", MIXED_NESTED_SOURCE);
    assert_shared_cps_contract(&ir_text);
    insta::assert_snapshot!(pipeline_contract_summary(&ir_text, false));
    assert!(ir_text.contains("func.func @run_all"), "{ir_text}");
}

#[salsa_test]
fn shared_pipeline_float_comparison_predicates(db: &salsa::DatabaseImpl) {
    let ir_text = shared_pipeline_ir(db, "float_comparisons.trb", FLOAT_COMPARISON_SOURCE);
    assert_shared_cps_contract(&ir_text);
    insta::assert_snapshot!(pipeline_contract_summary(&ir_text, false));
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
    insta::assert_snapshot!(pipeline_contract_summary(&ir_text, true));
    assert!(ir_text.contains("func.func @run"), "{ir_text}");
}

#[salsa_test]
fn native_pipeline_resumptive_op_continuation(db: &salsa::DatabaseImpl) {
    let ir_text = native_pipeline_ir(db, "resumptive_op_native.trb", RESUMPTIVE_OP_SOURCE);
    assert_native_cps_root_contract(&ir_text);
    insta::assert_snapshot!(pipeline_contract_summary(&ir_text, true));
    assert!(ir_text.contains("__tribute_one_shot_state_"), "{ir_text}");
}

#[salsa_test]
fn native_pipeline_mixed_nested_handler_boundary(db: &salsa::DatabaseImpl) {
    let ir_text = native_pipeline_ir(db, "mixed_nested_native.trb", MIXED_NESTED_SOURCE);
    assert_native_cps_root_contract(&ir_text);
    insta::assert_snapshot!(pipeline_contract_summary(&ir_text, true));
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
