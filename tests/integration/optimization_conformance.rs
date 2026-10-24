//! Semantic and structural gates for independently selectable optimizations.

use crate::common;

use std::fmt::Write as _;

use common::{
    compile_and_run_native, compile_and_run_native_asan,
    compile_and_run_native_with_borrowed_parameters, compile_and_run_native_with_field_moves,
    compile_and_run_native_with_last_use_moves, compile_and_run_native_with_paired_rc_elimination,
    compile_and_run_native_with_temporary_borrows,
};
use salsa_test_macros::salsa_test;
use tribute::pipeline::{
    BorrowedParameterPolicy, FieldMovePolicy, LastUseMovePolicy, NativeOptimizationOptions,
    NativePipelineStage, OptimizationOptions, PairedRcEliminationPolicy, TemporaryBorrowPolicy,
    dump_native_ir_at_stage,
};
use tribute_front::SourceCst;

const STATE_HANDLERS: &str = include_str!("../fixtures/optimizations/state_handlers.trb");
const PAIRED_RC_ELIMINATION: &str =
    include_str!("../fixtures/optimizations/paired_rc_elimination.trb");
const BORROWED_PARAMETERS: &str = include_str!("../fixtures/optimizations/borrowed_parameters.trb");
const TRUSTED_OWNERSHIP_FORWARDING: &str =
    include_str!("../fixtures/optimizations/trusted_ownership_forwarding.trb");
const TEMPORARY_FIELD_BORROWS: &str =
    include_str!("../fixtures/optimizations/temporary_field_borrows.trb");
const TEMPORARY_FIELD_BORROWS_EFFECTS: &str =
    include_str!("../fixtures/optimizations/temporary_field_borrows_effects.trb");
const BOXED_DYNAMIC_PRIMITIVES: &str = r#"
extern "C" fn __tribute_print_int(value: Int) -> Nil
extern "C" fn __tribute_print_float(value: Float) -> Nil

fn identity(value: a) -> a { value }

fn main() -> Nil {
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
            last_use_moves: LastUseMovePolicy::Acquire,
            field_moves: FieldMovePolicy::Acquire,
        },
    }
}

fn temporary_borrow_options(temporary_borrows: TemporaryBorrowPolicy) -> OptimizationOptions {
    OptimizationOptions {
        native: NativeOptimizationOptions {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows,
            last_use_moves: LastUseMovePolicy::Acquire,
            field_moves: FieldMovePolicy::Acquire,
        },
    }
}

/// The `@name` of a generic `clif.func` line, whose name prints as `sym_name = "name"`.
fn clif_func_symbol(line: &str) -> Option<String> {
    line.split_once("sym_name = \"")
        .and_then(|(_, rest)| rest.split('"').next())
        .map(|name| format!("@{name}"))
}

fn focused_rc_ops(ir: &str) -> String {
    let mut function = "<outside function>".to_owned();
    let mut emitted_function = None;
    let mut output = String::new();
    for line in ir.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("clif.func ") {
            function = clif_func_symbol(trimmed).unwrap_or_else(|| "<unknown function>".to_owned());
            continue;
        }
        if function.starts_with("@__tribute_release_")
            || !(trimmed.contains("tribute_rt.retain") || trimmed.contains("tribute_rt.release"))
        {
            continue;
        }
        if emitted_function.as_ref() != Some(&function) {
            writeln!(&mut output, "{function}:").expect("writing to a String cannot fail");
            emitted_function = Some(function.clone());
        }
        writeln!(&mut output, "{trimmed}").expect("writing to a String cannot fail");
    }
    output.truncate(output.len().saturating_sub(1));
    output
}

/// Whether a `clif.func` belongs to the fixture rather than to the prelude
/// (`std::`), the generated native entry `main`, or the RTTI releases.
fn is_fixture_function(symbol: &str) -> bool {
    symbol != "@main" && !symbol.starts_with("@std::") && !symbol.starts_with("@__tribute_release_")
}

/// [`focused_rc_ops`] restricted to the fixture's own functions, so the
/// prelude code that formats the printed result does not enter the
/// comparison.
fn fixture_rc_ops(ir: &str) -> String {
    let mut keep = false;
    let mut output = String::new();
    for line in focused_rc_ops(ir).lines() {
        if !line.starts_with("tribute_rt.") {
            keep = is_fixture_function(line.trim_end_matches(':'));
        }
        if keep {
            writeln!(&mut output, "{line}").expect("writing to a String cannot fail");
        }
    }
    output.truncate(output.len().saturating_sub(1));
    output
}

/// For each distinct RTTI index that the fixture's own functions store in an
/// allocation header, the number of fields its release function releases,
/// sorted. Generated numbering is not a contract.
fn fixture_allocation_field_releases(ir: &str) -> Vec<usize> {
    use std::collections::BTreeSet;
    use std::ops::ControlFlow;
    use tribute_ir::dialect::tribute_rt;
    use trunk_ir::IrContext;
    use trunk_ir::ValueDef;
    use trunk_ir::dialect::clif;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::walk::{WalkAction, walk_op};

    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_module(&mut ctx, ir).expect("native stage IR round trip");
    let mut functions = Vec::new();
    let _ = walk_op::<()>(&ctx, module, &mut |op| {
        if let Ok(function) = clif::Func::from_op(&ctx, op) {
            functions.push(function);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    let mut indices = BTreeSet::new();
    for function in &functions {
        if !is_fixture_function(&format!("@{}", function.sym_name(&ctx))) {
            continue;
        }
        let mut ops = Vec::new();
        let _ = walk_op::<()>(&ctx, function.op_ref(), &mut |op| {
            ops.push(op);
            ControlFlow::Continue(WalkAction::Advance)
        });
        for &op in &ops {
            let Ok(call) = clif::Call::from_op(&ctx, op) else {
                continue;
            };
            if call.callee(&ctx) != ["__tribute_alloc"] {
                continue;
            }
            let allocation = ctx.op_results(op)[0];
            let header = ops
                .iter()
                .filter_map(|&op| clif::Store::from_op(&ctx, op).ok())
                .find(|store| store.addr(&ctx) == allocation && store.offset(&ctx) == 4)
                .expect("allocation RTTI header");
            let ValueDef::OpResult(index, _) = ctx.value_def(header.value(&ctx)) else {
                panic!("constant RTTI index");
            };
            indices.insert(clif::Iconst::from_op(&ctx, index).unwrap().value(&ctx));
        }
    }
    let mut releases: Vec<usize> = indices
        .into_iter()
        .map(|index| {
            let symbol = format!("{}{index}", tribute_passes::native::rtti::RELEASE_FN_PREFIX);
            let release = functions
                .iter()
                .find(|function| function.sym_name(&ctx) == symbol.as_str())
                .unwrap_or_else(|| panic!("missing release function for RTTI index {index}"));
            let mut count = 0;
            let _ = walk_op::<()>(&ctx, release.op_ref(), &mut |op| {
                count += usize::from(tribute_rt::Release::matches(&ctx, op));
                ControlFlow::Continue(WalkAction::Advance)
            });
            count
        })
        .collect();
    releases.sort();
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
fn temporary_field_borrows_across_handlers_preserve_native_execution_with_sanitizer() {
    for sanitize_address in [false, true] {
        for policy in [
            TemporaryBorrowPolicy::Preserve,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        ] {
            let output = compile_and_run_native_with_temporary_borrows(
                "temporary_field_borrows_effects.trb",
                TEMPORARY_FIELD_BORROWS_EFFECTS,
                policy,
                sanitize_address,
            );
            assert!(
                output.status.success(),
                "{policy:?} failed with sanitize_address={sanitize_address}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "150");
        }
    }
}

#[test]
fn last_use_moves_preserve_native_execution_with_sanitizer() {
    let fixtures = [
        ("state_handlers.trb", STATE_HANDLERS, "10"),
        ("borrowed_parameters.trb", BORROWED_PARAMETERS, "10"),
        ("temporary_field_borrows.trb", TEMPORARY_FIELD_BORROWS, "20"),
        (
            "temporary_field_borrows_effects.trb",
            TEMPORARY_FIELD_BORROWS_EFFECTS,
            "150",
        ),
    ];
    for (name, source, expected) in fixtures {
        for sanitize_address in [false, true] {
            for policy in [LastUseMovePolicy::Acquire, LastUseMovePolicy::Move] {
                let output = compile_and_run_native_with_last_use_moves(
                    name,
                    source,
                    policy,
                    sanitize_address,
                );
                assert!(
                    output.status.success(),
                    "{name} {policy:?} failed with sanitize_address={sanitize_address}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(
                    String::from_utf8_lossy(&output.stdout).trim(),
                    expected,
                    "{name} {policy:?}"
                );
            }
        }
    }
}

#[test]
fn field_moves_preserve_native_execution_with_sanitizer() {
    let fixtures = [
        ("state_handlers.trb", STATE_HANDLERS, "10"),
        ("borrowed_parameters.trb", BORROWED_PARAMETERS, "10"),
        ("temporary_field_borrows.trb", TEMPORARY_FIELD_BORROWS, "20"),
        (
            "temporary_field_borrows_effects.trb",
            TEMPORARY_FIELD_BORROWS_EFFECTS,
            "150",
        ),
    ];
    for (name, source, expected) in fixtures {
        for sanitize_address in [false, true] {
            for policy in [FieldMovePolicy::Acquire, FieldMovePolicy::Move] {
                let output =
                    compile_and_run_native_with_field_moves(name, source, policy, sanitize_address);
                assert!(
                    output.status.success(),
                    "{name} {policy:?} failed with sanitize_address={sanitize_address}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(
                    String::from_utf8_lossy(&output.stdout).trim(),
                    expected,
                    "{name} {policy:?}"
                );
            }
        }
    }
}

const FIELD_MOVES_FROM_SHARED_AND_UNIQUE_OWNERS: &str = r#"
use std::io::{Io, print_line}

struct Holder { name: String, count: Int }
struct Pair { left: Holder, right: Holder }

fn shout(text: String) -> String { text }

// Recursion keeps the pair, and so the holders' other references, opaque.
fn keep(pair: Pair, n: Int) -> Pair {
    case n == +0 {
        True -> pair
        False -> keep(pair, n - +1)
    }
}

fn first(holder: Holder) -> String { shout(holder.name) }

fn main() ->{Io} Nil {
    print_line(first(Holder { name: "unique", count: +1 }))
    let holder = Holder { name: "shared", count: +2 }
    let pair = keep(Pair { left: holder, right: holder }, +3)
    print_line(first(holder))
    print_line(pair.left.name)
    print_line(pair.right.name)
}
"#;

#[test]
fn field_moves_from_shared_and_unique_owners_preserve_native_execution() {
    for sanitize_address in [false, true] {
        for policy in [FieldMovePolicy::Acquire, FieldMovePolicy::Move] {
            let output = compile_and_run_native_with_field_moves(
                "field_moves_from_shared_and_unique_owners.trb",
                FIELD_MOVES_FROM_SHARED_AND_UNIQUE_OWNERS,
                policy,
                sanitize_address,
            );
            assert!(
                output.status.success(),
                "{policy:?} failed with sanitize_address={sanitize_address}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                "unique\nshared\nshared\nshared\n",
                "{policy:?}"
            );
        }
    }
}

#[salsa_test]
fn field_moves_from_shared_and_unique_owners_use_moving_releases(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "field_moves_from_shared_and_unique_owners.trb",
        FIELD_MOVES_FROM_SHARED_AND_UNIQUE_OWNERS,
    );
    let ir = dump_native_ir_at_stage(
        db,
        source,
        NativePipelineStage::AfterRcOptimization,
        OptimizationOptions::production(),
    )
    .expect("RC optimization IR should be available");
    // The inlined `first` moves the name out of the shared holder, whose
    // pair keeps two more references, and `pair.right.name` moves out of
    // the unique pair and then out of its holder.
    let (_, main) = ir
        .split_once("sym_name = \"__tribute_main\"")
        .expect("main in the module");
    let main = main.split("clif.func ").next().unwrap();
    assert!(
        main.matches("tribute_rt.release_moving").count() >= 3,
        "{main}"
    );
}

#[salsa_test]
fn field_moves_remove_retains_and_keep_every_release(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "state_handlers.trb", STATE_HANDLERS);
    let dump = |field_moves| {
        let mut options = OptimizationOptions::production();
        options.native.field_moves = field_moves;
        let ir = dump_native_ir_at_stage(
            db,
            source,
            NativePipelineStage::AfterRcOptimization,
            options,
        )
        .expect("RC optimization IR should be available");
        assert_source_allocation_field_releases(ir);
        let ops = focused_rc_ops(ir);
        (
            ops.matches("tribute_rt.retain").count(),
            ir.matches("tribute_rt.release_moving").count(),
        )
    };
    let (acquire_retains, acquire_moving) = dump(FieldMovePolicy::Acquire);
    let (move_retains, move_moving) = dump(FieldMovePolicy::Move);
    assert_eq!(acquire_moving, 0);
    assert!(move_moving > 0);
    assert!(move_retains < acquire_retains);
}

#[salsa_test]
fn last_use_moves_remove_whole_retain_release_pairs(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "state_handlers.trb", STATE_HANDLERS);
    let dump = |last_use_moves| {
        let mut options = OptimizationOptions::production();
        options.native.last_use_moves = last_use_moves;
        let ir = dump_native_ir_at_stage(
            db,
            source,
            NativePipelineStage::AfterRcOptimization,
            options,
        )
        .expect("RC optimization IR should be available");
        assert_source_allocation_field_releases(ir);
        let ops = focused_rc_ops(ir);
        (
            ops.matches("tribute_rt.retain").count(),
            ops.matches("tribute_rt.release").count(),
        )
    };
    let (acquire_retains, acquire_releases) = dump(LastUseMovePolicy::Acquire);
    let (move_retains, move_releases) = dump(LastUseMovePolicy::Move);
    // A move drops the acquire at the destination and the final release
    // of the value together.
    assert!(move_retains < acquire_retains);
    assert_eq!(
        acquire_retains - move_retains,
        acquire_releases - move_releases
    );
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
    // The fixture's own functions allocate only `Boxed(Nat)`, whose
    // descriptor has no managed field to release.
    assert_eq!(fixture_allocation_field_releases(preserved), [0]);
    assert_eq!(fixture_allocation_field_releases(elided), [0]);
    let preserved = fixture_rc_ops(preserved);
    let elided = fixture_rc_ops(elided);
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
    let mut released_capture = false;
    for &op in &ops {
        let Ok(retain) = tribute_rt::Retain::from_op(&ctx, op) else {
            continue;
        };
        let ty = ctx.get_type(ctx.value_ty(retain.result(&ctx)));
        let kind = if ty
            .attrs
            .view(&ctx)
            .get_str("name")
            .is_some_and(|name| name == "_closure")
        {
            "closure"
        } else if ty
            .attrs
            .view(&ctx)
            .get_str("name")
            .is_some_and(|name| name.starts_with(tribute_ir::continuation_frame::NAME_PREFIX))
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
                break (call.callee(&ctx) == ["__tribute_alloc"]).then_some(value);
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
        if kind == "closure" {
            // A closure releases only its captures, after the code pointer at
            // offset 0. A capture-free closure is the RC header and the code
            // pointer alone and releases nothing.
            let ValueDef::OpResult(alloc, _) = ctx.value_def(allocation) else {
                panic!("allocation call result");
            };
            let ValueDef::OpResult(size, _) = ctx.value_def(ctx.op_operands(alloc)[0]) else {
                panic!("constant allocation size");
            };
            let size = clif::Iconst::from_op(&ctx, size).unwrap().value(&ctx);
            if size == 16 {
                assert!(
                    offsets.is_empty(),
                    "capture-free closure releases {offsets:?}"
                );
            } else {
                assert!(
                    offsets
                        .iter()
                        .all(|&offset| offset >= 8 && i64::from(offset) < size - 8),
                    "closure of {size} bytes must release only its captures: {offsets:?}"
                );
                released_capture |= !offsets.is_empty();
            }
        } else {
            assert_eq!(
                offsets,
                vec![0, 8],
                "{kind} must release exactly its managed fields"
            );
        }
    }
    assert!(
        checked.iter().any(|(kind, _)| *kind == "closure"),
        "fixture must allocate closure storage"
    );
    assert!(
        released_capture,
        "fixture must release a managed closure capture"
    );
    assert!(
        checked.iter().any(|(kind, _)| *kind == "frame"),
        "fixture must allocate continuation frames"
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

    let before = focused_rc_ops(before);
    let after = focused_rc_ops(after);
    let disabled_after = focused_rc_ops(disabled_after);

    assert_eq!(before, disabled_after);
    let before_retain = before.matches("tribute_rt.retain").count();
    let before_release = before.matches("tribute_rt.release").count();
    let after_retain = after.matches("tribute_rt.retain").count();
    let after_release = after.matches("tribute_rt.release").count();
    // Module-internal callables consume their parameters, so this fixture
    // no longer acquires a parameter at entry only to release it. The pass's
    // own pair matching is covered by its unit tests; here it must keep the
    // releases of every allocation and remove only whole pairs.
    assert!(before_retain >= after_retain, "before RC ops:\n{before}");
    assert!(before_release >= after_release, "before RC ops:\n{before}");
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

    // The fixture's own functions allocate only `Boxed(Nat)`, whose
    // descriptor has no managed field to release.
    assert_eq!(fixture_allocation_field_releases(before), [0]);
    assert_eq!(fixture_allocation_field_releases(after), [0]);
    assert_eq!(fixture_allocation_field_releases(preserved_after), [0]);

    let before = fixture_rc_ops(before);
    let after = fixture_rc_ops(after);
    let preserved_after = fixture_rc_ops(preserved_after);

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
    // `Value(Nat)` releases nothing and `Wrapped(Inner)` releases its one
    // managed field.
    assert_eq!(fixture_allocation_field_releases(before), [0, 1]);
    assert_eq!(fixture_allocation_field_releases(after), [0, 1]);
    assert_eq!(fixture_allocation_field_releases(preserved_after), [0, 1]);
    let before = fixture_rc_ops(before);
    let after = fixture_rc_ops(after);
    let preserved_after = fixture_rc_ops(preserved_after);
    assert_eq!(before, preserved_after);
    let before_retain = before.matches("tribute_rt.retain").count();
    let before_release = before.matches("tribute_rt.release").count();
    let after_retain = after.matches("tribute_rt.retain").count();
    let after_release = after.matches("tribute_rt.release").count();
    assert!(before_retain > after_retain, "before RC ops:\n{before}");
    assert!(before_release > after_release, "before RC ops:\n{before}");
    assert_eq!(before_retain - after_retain, before_release - after_release);
}

#[salsa_test]
fn temporary_field_borrows_across_handlers_elide_rc_operations(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "temporary_field_borrows_effects_snapshot.trb",
        TEMPORARY_FIELD_BORROWS_EFFECTS,
    );
    let rc_ops = |stage, policy| {
        let ir = dump_native_ir_at_stage(db, source, stage, temporary_borrow_options(policy))
            .expect("temporary-borrow IR should be available");
        focused_rc_ops(ir)
    };
    let before = rc_ops(
        NativePipelineStage::AfterBorrowedParameterOptimization,
        TemporaryBorrowPolicy::Preserve,
    );
    let after = rc_ops(
        NativePipelineStage::AfterTemporaryBorrowOptimization,
        TemporaryBorrowPolicy::ElideProvenFieldBorrows,
    );
    let count = |ir: &str, operation: &str| ir.matches(operation).count();
    let retains = count(&before, "tribute_rt.retain") - count(&after, "tribute_rt.retain");
    let releases = count(&before, "tribute_rt.release") - count(&after, "tribute_rt.release");
    assert!(retains > 0, "before RC ops:\n{before}");
    // A projection that keeps its own unit is acquired once and released on
    // each path that drops it, so more releases than retains disappear.
    assert!(releases >= retains, "before RC ops:\n{before}");
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
