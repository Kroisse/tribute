//! Semantic and structural gates for independently selectable optimizations.

mod common;

use std::fmt::Write as _;

use common::{
    compile_and_run_native, compile_and_run_native_asan,
    compile_and_run_native_with_borrowed_parameters,
    compile_and_run_native_with_paired_rc_elimination,
    compile_and_run_native_with_temporary_borrows,
};
use salsa_test_macros::salsa_test;
use tribute::pipeline::{
    BorrowedParameterPolicy, NativeOptimizationOptions, NativePipelineStage, OptimizationOptions,
    PairedRcEliminationPolicy, TemporaryBorrowPolicy, dump_native_ir_at_stage,
};
use tribute_front::SourceCst;

const STATE_HANDLERS: &str = include_str!("fixtures/optimizations/state_handlers.trb");
const PAIRED_RC_ELIMINATION: &str =
    include_str!("fixtures/optimizations/paired_rc_elimination.trb");
const BORROWED_PARAMETERS: &str = include_str!("fixtures/optimizations/borrowed_parameters.trb");
const TRUSTED_OWNERSHIP_FORWARDING: &str =
    include_str!("fixtures/optimizations/trusted_ownership_forwarding.trb");
const TEMPORARY_FIELD_BORROWS: &str =
    include_str!("fixtures/optimizations/temporary_field_borrows.trb");
const BOXED_DYNAMIC_PRIMITIVES: &str = r#"
extern "C" fn __tribute_print_int(value: Int) -> Nil
extern "C" fn __tribute_print_float(value: Float) -> Nil

fn identity(value: a) -> a { value }

fn main() {
    __tribute_print_int(identity(+7))
    __tribute_print_float(identity(3.5))
}
"#;

fn native_optimization_options(
    paired_rc_elimination: PairedRcEliminationPolicy,
    borrowed_parameters: BorrowedParameterPolicy,
) -> OptimizationOptions {
    OptimizationOptions {
        native: NativeOptimizationOptions {
            paired_rc_elimination,
            borrowed_parameters,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
        },
    }
}

fn temporary_borrow_options(temporary_borrows: TemporaryBorrowPolicy) -> OptimizationOptions {
    OptimizationOptions {
        native: NativeOptimizationOptions {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows,
        },
    }
}

fn focused_rc_ops(ir: &str) -> String {
    let mut function = "<outside function>";
    let mut emitted_function = None;
    let mut output = String::new();
    for line in ir.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("clif.func ") {
            function = trimmed
                .split_once("sym_name = ")
                .and_then(|(_, rest)| rest.split([',', '}']).next())
                .unwrap_or("<unknown function>");
            continue;
        }
        if function.starts_with("@__tribute_release_")
            || !(trimmed.contains("tribute_rt.retain") || trimmed.contains("tribute_rt.release"))
        {
            continue;
        }
        if emitted_function != Some(function) {
            writeln!(&mut output, "{function}:").expect("writing to a String cannot fail");
            emitted_function = Some(function);
        }
        writeln!(&mut output, "{trimmed}").expect("writing to a String cannot fail");
    }
    output.truncate(output.len().saturating_sub(1));
    output
}

fn generated_rtti_field_releases(ir: &str) -> String {
    let mut function = None;
    let mut count = 0;
    let mut releases = String::new();
    for line in ir.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("clif.func ") {
            if let Some(function) = function.take()
                && count > 0
            {
                if !releases.is_empty() {
                    releases.push(',');
                }
                write!(&mut releases, "{function}={count}")
                    .expect("writing to a String cannot fail");
            }
            count = 0;
            function = trimmed
                .split_once("sym_name = ")
                .and_then(|(_, rest)| rest.split([',', '}']).next())
                .filter(|symbol| symbol.starts_with("@__tribute_release_"));
        } else if function.is_some() && trimmed.contains("tribute_rt.release") {
            count += 1;
        }
    }
    if let Some(function) = function
        && count > 0
    {
        if !releases.is_empty() {
            releases.push(',');
        }
        write!(&mut releases, "{function}={count}").expect("writing to a String cannot fail");
    }
    releases
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

#[test]
fn trusted_ownership_forwarding_preserves_native_execution() {
    for sanitize_address in [false, true] {
        let preserved = compile_and_run_native_with_borrowed_parameters(
            "trusted_ownership_forwarding_preserved.trb",
            TRUSTED_OWNERSHIP_FORWARDING,
            BorrowedParameterPolicy::Preserve,
            sanitize_address,
        );
        let elided = compile_and_run_native_with_borrowed_parameters(
            "trusted_ownership_forwarding_elided.trb",
            TRUSTED_OWNERSHIP_FORWARDING,
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
        assert_eq!(String::from_utf8_lossy(&elided.stdout).trim(), "40");
    }
}

#[test]
fn temporary_field_borrows_preserve_native_execution_with_sanitizer() {
    for sanitize_address in [false, true] {
        let preserved = compile_and_run_native_with_temporary_borrows(
            "temporary_field_borrows_preserved.trb",
            TEMPORARY_FIELD_BORROWS,
            TemporaryBorrowPolicy::Preserve,
            sanitize_address,
        );
        let elided = compile_and_run_native_with_temporary_borrows(
            "temporary_field_borrows_elided.trb",
            TEMPORARY_FIELD_BORROWS,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows,
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
        let preserved_stdout = String::from_utf8_lossy(&preserved.stdout);
        let elided_stdout = String::from_utf8_lossy(&elided.stdout);
        assert_eq!(preserved_stdout.trim(), "20");
        assert_eq!(elided_stdout.trim(), "20");
        assert_eq!(preserved.stdout, elided.stdout);
    }
}

#[test]
fn boxed_dynamic_primitives_release_through_exact_rtti_entries() {
    for sanitize_address in [false, true] {
        let output = if sanitize_address {
            compile_and_run_native_asan("boxed_dynamic_primitives.trb", BOXED_DYNAMIC_PRIMITIVES)
        } else {
            compile_and_run_native("boxed_dynamic_primitives.trb", BOXED_DYNAMIC_PRIMITIVES)
        };
        assert!(
            output.status.success(),
            "boxed primitive pipeline failed with sanitize_address={sanitize_address}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), "7\n3.5\n");
    }
}

#[salsa_test]
fn trusted_ownership_forwarding_has_focused_ir(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "trusted_ownership_forwarding_snapshot.trb",
        TRUSTED_OWNERSHIP_FORWARDING,
    );
    let preserved = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterBorrowedParameterOptimization,
        native_optimization_options(
            PairedRcEliminationPolicy::Disabled,
            BorrowedParameterPolicy::Preserve,
        ),
    )
    .expect("preserved forwarding IR should be available");
    let elided = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterBorrowedParameterOptimization,
        native_optimization_options(
            PairedRcEliminationPolicy::Disabled,
            BorrowedParameterPolicy::ElideProvenBorrowed,
        ),
    )
    .expect("elided forwarding IR should be available");
    assert_eq!(
        generated_rtti_field_releases(&preserved),
        "@__tribute_release_32=1"
    );
    assert_eq!(
        generated_rtti_field_releases(&elided),
        "@__tribute_release_32=1"
    );
    let preserved = focused_rc_ops(&preserved);
    let elided = focused_rc_ops(&elided);
    assert_eq!(
        preserved, elided,
        "physical CPS functions are conservatively consumed; typed-plan policy coverage lives with the pre-erasure action planner"
    );
}
/// Follow typed source allocations to the RTTI index stored in their header,
/// then verify the deep-release loads. Generated numbering is not a contract.
fn assert_source_allocation_field_releases(ir: &str) {
    use std::collections::BTreeSet;
    use std::ops::ControlFlow;
    use tribute_ir::dialect::tribute_rt;
    use trunk_ir::dialect::{clif, core};
    use trunk_ir::ops::DialectOp;
    use trunk_ir::walk::{WalkAction, walk_op};
    use trunk_ir::{IrContext, ValueDef};

    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_module(&mut ctx, ir).expect("native stage IR round trip");
    let mut ops = Vec::new();
    let _ = walk_op::<()>(&ctx, module, &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    let mut checked = BTreeSet::new();
    for &op in &ops {
        let Ok(retain) = tribute_rt::Retain::from_op(&ctx, op) else {
            continue;
        };
        let ty = ctx.types.get(ctx.value_ty(retain.result(&ctx)));
        let kind = if ty
            .attrs
            .get_symbol("name")
            .is_some_and(|name| name == "_closure")
        {
            "closure"
        } else if ty
            .attrs
            .get_type(tribute_core::calling_convention::CPS_CONTINUATION_FRAME_RESULT_ATTR)
            .is_some_and(|result| ctx.types.get(result).name == "i32")
        {
            "frame"
        } else {
            continue;
        };
        let mut value = retain.ptr(&ctx);
        // Native materialization casts and payload offsets retain the allocation
        // base as operand zero. Stop at block args rather than guessing ownership.
        let allocation = loop {
            let ValueDef::OpResult(def, _) = ctx.value_def(value) else {
                break None;
            };
            if let Ok(call) = clif::Call::from_op(&ctx, def) {
                break (call.callee(&ctx) == "__tribute_alloc").then_some(value);
            }
            if core::UnrealizedConversionCast::matches(&ctx, def) || clif::Iadd::matches(&ctx, def)
            {
                value = ctx.op_operands(def)[0];
            } else {
                break None;
            }
        };
        let Some(allocation) = allocation else {
            continue;
        };
        let header = ops
            .iter()
            .filter_map(|&op| clif::Store::from_op(&ctx, op).ok())
            .find(|store| store.addr(&ctx) == allocation && store.offset(&ctx) == 4)
            .expect("allocation RTTI header");
        let ValueDef::OpResult(index, _) = ctx.value_def(header.value(&ctx)) else {
            panic!("constant RTTI index");
        };
        let index = clif::Iconst::from_op(&ctx, index).unwrap().value(&ctx);
        let symbol = format!("{}{index}", tribute_passes::native::rtti::RELEASE_FN_PREFIX);
        if !checked.insert((kind, symbol.clone())) {
            continue;
        }
        let release = ops
            .iter()
            .filter_map(|&op| clif::Func::from_op(&ctx, op).ok())
            .find(|function| function.sym_name(&ctx) == symbol.as_str())
            .expect("allocation release function");
        let entry = ctx.region(release.body(&ctx)).blocks[0];
        let payload = ctx.block_args(entry)[0];
        let mut offsets = Vec::new();
        let _ = walk_op::<()>(&ctx, release.op_ref(), &mut |op| {
            if let Ok(release) = tribute_rt::Release::from_op(&ctx, op) {
                let ValueDef::OpResult(load, _) = ctx.value_def(release.ptr(&ctx)) else {
                    panic!("release of loaded field");
                };
                let load =
                    clif::Load::from_op(&ctx, load).expect("release must load a managed field");
                assert_eq!(load.addr(&ctx), payload);
                offsets.push(load.offset(&ctx));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        offsets.sort();
        assert_eq!(
            offsets,
            if kind == "closure" {
                vec![8]
            } else {
                vec![0, 8]
            },
            "{kind} must release exactly its managed fields"
        );
    }
    assert!(
        checked.iter().any(|(kind, _)| *kind == "closure"),
        "fixture must allocate closure storage"
    );
    assert!(
        checked.iter().any(|(kind, _)| *kind == "frame"),
        "fixture must allocate Int continuation frames"
    );
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

    for ir in [&before, &after, &disabled_after] {
        assert_source_allocation_field_releases(ir);
    }

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

    assert_eq!(
        generated_rtti_field_releases(&before),
        "@__tribute_release_32=1"
    );
    assert_eq!(
        generated_rtti_field_releases(&after),
        "@__tribute_release_32=1"
    );
    assert_eq!(
        generated_rtti_field_releases(&preserved_after),
        "@__tribute_release_32=1"
    );

    let before = focused_rc_ops(&before);
    let after = focused_rc_ops(&after);
    let preserved_after = focused_rc_ops(&preserved_after);

    assert_eq!(before, preserved_after);
    assert_eq!(before, after, "recursive summaries must fail closed");
}

#[salsa_test]
fn temporary_field_borrows_have_focused_before_after_ir(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "temporary_field_borrows_snapshot.trb",
        TEMPORARY_FIELD_BORROWS,
    );
    let before = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterBorrowedParameterOptimization,
        temporary_borrow_options(TemporaryBorrowPolicy::Preserve),
    )
    .expect("preserved temporary RC insertion IR should be available");
    let after = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterTemporaryBorrowOptimization,
        temporary_borrow_options(TemporaryBorrowPolicy::ElideProvenFieldBorrows),
    )
    .expect("temporary-borrow IR should be available");
    let preserved_after = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterTemporaryBorrowOptimization,
        temporary_borrow_options(TemporaryBorrowPolicy::Preserve),
    )
    .expect("preserved temporary-borrow stage IR should be available");
    assert_eq!(
        generated_rtti_field_releases(&before),
        "@__tribute_release_33=1"
    );
    assert_eq!(
        generated_rtti_field_releases(&after),
        "@__tribute_release_33=1"
    );
    assert_eq!(
        generated_rtti_field_releases(&preserved_after),
        "@__tribute_release_33=1"
    );
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
    assert_eq!(before_retain - after_retain, before_release - after_release);
}

#[test]
fn state_handlers_preserve_native_execution() {
    let output = compile_and_run_native("state_handlers.trb", STATE_HANDLERS);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "10");
}
