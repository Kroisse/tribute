//! Semantic and structural gates for independently selectable optimizations.

mod common;

use common::{
    compile_and_run_native_with_borrowed_parameters,
    compile_and_run_native_with_done_continuation_dedup,
    compile_and_run_native_with_paired_rc_elimination,
};
use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute::pipeline::{
    BorrowedParameterPolicy, NativeOptimizationOptions, NativePipelineStage, OptimizationOptions,
    PairedRcEliminationPolicy, SharedPipelineStage, dump_native_ir_at_stage,
    dump_shared_ir_at_stage,
};
use tribute_front::SourceCst;
use tribute_front::ast_to_ir::DoneContinuationPolicy;

const DONE_CONTINUATION_DEDUP_STATE: &str =
    include_str!("fixtures/optimizations/done_continuation_dedup_state.trb");
const PAIRED_RC_ELIMINATION: &str =
    include_str!("fixtures/optimizations/paired_rc_elimination.trb");
const BORROWED_PARAMETERS: &str = include_str!("fixtures/optimizations/borrowed_parameters.trb");

fn native_optimization_options(
    paired_rc_elimination: PairedRcEliminationPolicy,
    borrowed_parameters: BorrowedParameterPolicy,
) -> OptimizationOptions {
    OptimizationOptions {
        native: NativeOptimizationOptions {
            paired_rc_elimination,
            borrowed_parameters,
        },
        ..OptimizationOptions::production()
    }
}

fn focused_rc_ops(ir: &str) -> String {
    ir.lines()
        .filter(|line| line.contains("tribute_rt.retain") || line.contains("tribute_rt.release"))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
}

fn identity_done_symbols(ir: &str) -> Vec<&str> {
    let lines: Vec<_> = ir.lines().collect();
    lines
        .windows(2)
        .filter_map(|pair| {
            let header = pair[0].trim_start();
            (header.starts_with("func.func ")
                && header.contains("%2: tribute_rt.anyref) -> tribute_rt.anyref")
                && pair[1].trim() == "func.return %2")
                .then(|| {
                    header
                        .strip_prefix("func.func ")
                        .expect("checked prefix")
                        .split('(')
                        .next()
                        .expect("function header has parameters")
                })
        })
        .collect()
}

fn focused_identity_done_ir(ir: &str) -> String {
    let symbols = identity_done_symbols(ir);
    ir.lines()
        .filter(|line| symbols.iter().any(|symbol| line.contains(symbol)))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
}

fn lambda_function_count(ir: &str) -> usize {
    ir.lines()
        .filter(|line| {
            let line = line.trim_start();
            line.starts_with("func.func ") && line.contains("::__lambda_")
        })
        .count()
}

#[test]
fn done_continuation_dedup_preserves_native_execution() {
    let disabled = compile_and_run_native_with_done_continuation_dedup(
        "done_continuation_dedup_disabled.trb",
        DONE_CONTINUATION_DEDUP_STATE,
        DoneContinuationPolicy::PerUse,
    );
    let enabled = compile_and_run_native_with_done_continuation_dedup(
        "done_continuation_dedup_enabled.trb",
        DONE_CONTINUATION_DEDUP_STATE,
        DoneContinuationPolicy::PerCompilationUnit,
    );

    assert!(
        disabled.status.success(),
        "disabled pipeline failed: {}",
        String::from_utf8_lossy(&disabled.stderr)
    );
    assert!(
        enabled.status.success(),
        "enabled pipeline failed: {}",
        String::from_utf8_lossy(&enabled.stderr)
    );
    assert_eq!(disabled.stdout, enabled.stdout);
    assert_eq!(String::from_utf8_lossy(&enabled.stdout).trim(), "10");
}

#[test]
fn paired_rc_elimination_preserves_native_execution() {
    let disabled = compile_and_run_native_with_paired_rc_elimination(
        "paired_rc_elimination_disabled.trb",
        PAIRED_RC_ELIMINATION,
        PairedRcEliminationPolicy::Disabled,
    );
    let enabled = compile_and_run_native_with_paired_rc_elimination(
        "paired_rc_elimination_enabled.trb",
        PAIRED_RC_ELIMINATION,
        PairedRcEliminationPolicy::Enabled,
    );

    assert!(
        disabled.status.success(),
        "disabled pipeline failed: {}",
        String::from_utf8_lossy(&disabled.stderr)
    );
    assert!(
        enabled.status.success(),
        "enabled pipeline failed: {}",
        String::from_utf8_lossy(&enabled.stderr)
    );
    assert_eq!(disabled.stdout, enabled.stdout);
    assert_eq!(String::from_utf8_lossy(&enabled.stdout).trim(), "10");
}

#[test]
fn borrowed_parameters_preserve_native_execution() {
    for sanitize_address in [false, true] {
        let preserved = compile_and_run_native_with_borrowed_parameters(
            "borrowed_parameters_preserved.trb",
            BORROWED_PARAMETERS,
            BorrowedParameterPolicy::Preserve,
            sanitize_address,
        );
        let elided = compile_and_run_native_with_borrowed_parameters(
            "borrowed_parameters_elided.trb",
            BORROWED_PARAMETERS,
            BorrowedParameterPolicy::ElideProvenBorrowed,
            sanitize_address,
        );

        assert!(
            preserved.status.success(),
            "preserved pipeline failed with sanitize_address={sanitize_address}: {}",
            String::from_utf8_lossy(&preserved.stderr)
        );
        assert!(
            elided.status.success(),
            "elided pipeline failed with sanitize_address={sanitize_address}: {}",
            String::from_utf8_lossy(&elided.stderr)
        );
        assert_eq!(preserved.stdout, elided.stdout);
        assert_eq!(String::from_utf8_lossy(&elided.stdout).trim(), "10");
    }
}

#[salsa_test]
fn paired_rc_elimination_has_focused_before_after_ir(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "paired_rc_elimination_snapshot.trb",
        PAIRED_RC_ELIMINATION,
    );
    let before = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterRcInsertion,
        OptimizationOptions::production(),
    )
    .expect("RC insertion IR should be available");
    let after = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterRcOptimization,
        native_optimization_options(
            PairedRcEliminationPolicy::Enabled,
            BorrowedParameterPolicy::Preserve,
        ),
    )
    .expect("RC optimization IR should be available");
    let disabled_after = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterRcOptimization,
        native_optimization_options(
            PairedRcEliminationPolicy::Disabled,
            BorrowedParameterPolicy::Preserve,
        ),
    )
    .expect("disabled RC optimization IR should be available");

    let before = focused_rc_ops(&before);
    let after = focused_rc_ops(&after);
    let disabled_after = focused_rc_ops(&disabled_after);

    assert_eq!(before, disabled_after);
    let before_retain = before.matches("tribute_rt.retain").count();
    let before_release = before.matches("tribute_rt.release").count();
    let after_retain = after.matches("tribute_rt.retain").count();
    let after_release = after.matches("tribute_rt.release").count();
    assert!(before_retain > after_retain, "before RC ops:\n{before}");
    assert!(before_release > after_release, "before RC ops:\n{before}");
    assert_eq!(before_retain - after_retain, before_release - after_release);

    assert_snapshot!("paired_rc_elimination_before", before);
    assert_snapshot!("paired_rc_elimination_after", after);
}

#[salsa_test]
fn borrowed_parameters_have_focused_before_after_ir(db: &salsa::DatabaseImpl) {
    let source =
        SourceCst::from_source_str(db, "borrowed_parameters_snapshot.trb", BORROWED_PARAMETERS);
    let before = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterRcInsertion,
        native_optimization_options(
            PairedRcEliminationPolicy::Disabled,
            BorrowedParameterPolicy::Preserve,
        ),
    )
    .expect("owned-parameter RC insertion IR should be available");
    let after = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterBorrowedParameterOptimization,
        native_optimization_options(
            PairedRcEliminationPolicy::Disabled,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        ),
    )
    .expect("borrowed-parameter IR should be available");
    let preserved_after = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterBorrowedParameterOptimization,
        native_optimization_options(
            PairedRcEliminationPolicy::Disabled,
            BorrowedParameterPolicy::Preserve,
        ),
    )
    .expect("preserved parameter RC IR should be available");

    let before = focused_rc_ops(&before);
    let after = focused_rc_ops(&after);
    let preserved_after = focused_rc_ops(&preserved_after);

    assert_eq!(before, preserved_after);
    let before_retain = before.matches("tribute_rt.retain").count();
    let before_release = before.matches("tribute_rt.release").count();
    let after_retain = after.matches("tribute_rt.retain").count();
    let after_release = after.matches("tribute_rt.release").count();
    assert!(before_retain > after_retain, "before RC ops:\n{before}");
    assert!(before_release > after_release, "before RC ops:\n{before}");
    assert_eq!(before_retain - after_retain, 1);
    assert_eq!(before_release - after_release, 2);
    assert!(
        after_retain > 0,
        "escaping parameters must retain ownership"
    );
    assert!(
        after_release > 0,
        "escaping parameters must still be released"
    );

    assert_snapshot!("borrowed_parameters_before", before);
    assert_snapshot!("borrowed_parameters_after", after);
}

#[salsa_test]
fn done_continuation_dedup_has_focused_before_after_ir(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "done_continuation_dedup_snapshot.trb",
        DONE_CONTINUATION_DEDUP_STATE,
    );
    let before = dump_shared_ir_at_stage(
        db,
        source,
        SharedPipelineStage::AfterFrontend,
        OptimizationOptions::baseline(),
    )
    .expect("unoptimized frontend IR should be available");
    let after = dump_shared_ir_at_stage(
        db,
        source,
        SharedPipelineStage::AfterFrontend,
        OptimizationOptions::production(),
    )
    .expect("optimized frontend IR should be available");

    let before_symbols = identity_done_symbols(&before);
    let after_symbols = identity_done_symbols(&after);
    assert!(
        before_symbols.len() > 1,
        "fixture must generate duplicate identity done continuations"
    );
    assert_eq!(after_symbols.len(), 1);
    assert_eq!(
        lambda_function_count(&before) - before_symbols.len(),
        lambda_function_count(&after) - after_symbols.len(),
        "capturing and user-authored lambdas must remain distinct"
    );

    let before_references: usize = before_symbols
        .iter()
        .map(|symbol| before.matches(&format!("func_ref = {symbol}")).count())
        .sum();
    let after_references = after
        .matches(&format!("func_ref = {}", after_symbols[0]))
        .count();
    assert_eq!(before_references, after_references);

    assert_snapshot!(
        "done_continuation_dedup_before",
        focused_identity_done_ir(&before)
    );
    assert_snapshot!(
        "done_continuation_dedup_after",
        focused_identity_done_ir(&after)
    );
}
