use super::*;
use crate::frontend::merge_and_lower_to_ir;
use crate::link::link_native_binary;
use crate::prelude::{prelude_dependencies, prelude_exports, prelude_module};
use itertools::Itertools;
use rustc_hash::FxHashSet as HashSet;
use salsa_test_macros::salsa_test;
use std::ops::ControlFlow;
use tribute_front::monomorphize::SpecializationOutput;
use tribute_front::resolve as ast_resolve;
use tribute_front::resolve::PackageEnv;
use tribute_front::typeck as ast_typeck;
use trunk_ir::SymbolPath;
use trunk_ir::dialect::clif;
use trunk_ir::ops::DialectType;
use trunk_ir::walk::{WalkAction, walk_region};

/// The shared middle-end output for `source`.
fn run_shared_pipeline(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> PassResult<Option<(IrContext, Module)>> {
    compile_frontend_for_shared_route(db, source)
        .map(run_shared_middle_end)
        .transpose()
}

/// The root `core.module` of `module`.
fn core_module(ctx: &IrContext, module: Module) -> core_dialect::Module {
    core_dialect::Module::from_op(ctx, module.op()).expect("core.module")
}

/// Run one module pass on `module`, without a verifier.
fn run_pass(
    ctx: &mut IrContext,
    module: Module,
    pass: impl trunk_ir::pass::Pass<Target = core_dialect::Module> + 'static,
) -> PassResult {
    let root = core_module(ctx, module);
    let mut pm = PassManager::new();
    pm.add_pass(pass);
    pm.run(ctx, root, &mut AnalysisCache::new())
}

/// The declaration of `entry` with its qualified name, if it has one.
fn declaration_with_name<'a, 'db>(
    entry: &'a tribute_front::ast::DeclEntry<tribute_front::ast::TypedRef<'db>>,
) -> Option<(
    trunk_ir::Symbol,
    &'a tribute_front::ast::Decl<tribute_front::ast::TypedRef<'db>>,
)> {
    use tribute_front::ast::Decl;
    let name = match &entry.decl {
        Decl::Function(function) => &function.name,
        Decl::ExternFunction(function) => &function.name,
        Decl::Struct(structure) => &structure.name,
        Decl::Enum(enumeration) => &enumeration.name,
        _ => return None,
    };
    Some((entry.module.child(name.clone()).to_symbol(), &entry.decl))
}

fn source_logical_cps_root_module(body: &str) -> (IrContext, Module) {
    let mut ctx = IrContext::new();
    let source = r#"core.module @test {
            !"std::runtime::evidence::Marker" = adt.struct<"std::runtime::evidence::Marker"(ability_id: core.i32, prompt_tag: core.i32)>
            !Evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
            !Frame = adt.typeref<{name = "__tribute_continuation_frame_root_nil", tribute.cps_continuation_frame_result = core.nil}>
            !Done = closure.closure<func.func_sig<(core.nil) -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 0}>
            !Resume = closure.closure<func.func_sig<(!Evidence, !Frame, tribute_rt.anyref) -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 0}>
            !Dispatch = closure.closure<func.func_sig<(!Evidence, !Resume, core.i32, core.i32, core.i32, tribute_rt.anyref) -> core.never>, {tribute.calling_convention = "cps", tribute.closure_environment_index = 1}>
            !__tribute_continuation_frame_root_nil = adt.struct<__tribute_continuation_frame_root_nil(done: !Done, dispatch: !Dispatch), {tribute.cps_continuation_frame_result = core.nil}>
            !Payload = adt.struct<__tribute_ability_payload_7590c57e()>
            func.func @main(%evidence: !Evidence, %frame: !Frame) -> core.never attributes {tribute.calling_convention = "cps"} {
                BODY
            }
            func.func @"std::runtime::evidence::lookup"(%ev: !Evidence, %id: core.i32) -> core.i32 {
                func.return %id
            }
        }"#.replace("BODY", body);
    let module = trunk_ir::parser::parse_test_module(&mut ctx, &source);
    (ctx, module)
}

#[test]
fn source_logical_root_defers_closure_storage_until_target_finalization() {
    let (mut ctx, module) = source_logical_cps_root_module("func.unreachable");
    let root = core_module(&ctx, module);
    let mut pm = PassManager::new();
    pm.add_pass(tribute_passes::target_abi::PhysicalizeCpsCallables)
        .add_pass(tribute_passes::closure_lower::LowerPreparedClosures);
    pm.run(&mut ctx, root, &mut Default::default()).unwrap();
    let after_abi = trunk_ir::printer::print_module(&ctx, module.op());
    assert!(after_abi.contains("closure.closure"), "{after_abi}");

    run_pass(
        &mut ctx,
        module,
        tribute_passes::closure_lower::FinalizeClosureStorageLayout,
    )
    .unwrap();

    let physical = trunk_ir::printer::print_module(&ctx, module.op());
    assert!(!physical.contains("closure.closure"), "{physical}");
}

#[test]
fn empty_result_cps_root_executes_native_done_continuation() {
    let (mut ctx, module) = source_logical_cps_root_module(
        r#"
            %done = adt.struct_get %frame {field = 0, type = !__tribute_continuation_frame_root_nil} : !Done
            %nil = core.nil_value : core.nil
            func.tail_call_indirect %done, %nil {signature = func.func_sig<(core.nil) -> core.never>, tribute.calling_convention = "cps"}
        "#,
    );

    run_target_to_boundary_exit(
        &mut ctx,
        module,
        tribute_passes::abi_boundary::TargetKind::Native,
    )
    .expect("logical CPS root crosses native boundary");
    for name in ["__tribute_main", "__tribute_done_k", "__tribute_unhandled"] {
        let function = module
            .ops(&ctx)
            .iter()
            .copied()
            .find_map(|op| {
                let function = func_dialect::Func::from_op(&ctx, op).ok()?;
                (function.sym_name(&ctx) == name).then_some(function)
            })
            .expect("root bridge function remains present");
        // The convention is consumed inside the boundary; the empty
        // physical result list is what identifies a Cps callable here.
        assert_eq!(
            tribute_ir::calling_convention::DynConventionBearing::of(&ctx, function.op_ref()),
            None
        );
        assert!(
            func_dialect::FuncSig::from_type_ref(&ctx, function.r#type(&ctx))
                .unwrap()
                .results(&ctx)
                .is_empty()
        );
    }
    let object = lower_and_emit_native(
        &mut ctx,
        module,
        false,
        NativeOptimizationOptions::production(),
    )
    .expect("empty-result root emits native code");
    let temp = tempfile::tempdir().unwrap();
    let executable = temp.path().join("empty-result-cps-root");
    link_native_binary(&object, &executable, None).expect("empty-result root links");
    let output = std::process::Command::new(executable)
        .output()
        .expect("empty-result root starts");
    assert!(
        output.status.success(),
        "native root failed: {:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn root_dispatch_definition_matches_fixed_wasm_tail_signature_and_validates_binary() {
    use trunk_ir::dialect::wasm;
    let (mut ctx, module) = source_logical_cps_root_module(
        r#"
            %dispatch = adt.struct_get %frame {field = 1, type = !__tribute_continuation_frame_root_nil} : !Dispatch
            %resume = adt.ref_null {type = !Resume} : !Resume
            %product = adt.struct_new {type = !Payload} : !Payload
            %payload = core.unrealized_conversion_cast %product : tribute_rt.anyref
            effect.dispatch_cps %evidence, %dispatch, %resume, %payload {ability = @State, op_name = "get", answer_type = core.nil}
        "#,
    );
    run_target_to_boundary_exit(
        &mut ctx,
        module,
        tribute_passes::abi_boundary::TargetKind::Wasm,
    )
    .unwrap();
    let binary = lower_and_emit_wasm(&mut ctx, module).unwrap_or_else(|error| {
        panic!(
            "{error}\n{}",
            trunk_ir::printer::print_module(&ctx, module.op())
        )
    });
    let dispatch_definition = module
        .ops(&ctx)
        .iter()
        .copied()
        .find_map(|op| {
            let function = wasm::Func::from_op(&ctx, op).ok()?;
            (function.sym_name(&ctx) == "__tribute_unhandled").then_some(function.r#type(&ctx))
        })
        .unwrap();
    let mut signatures = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
        if wasm::ReturnCallIndirect::matches(&ctx, op) {
            signatures.push(
                trunk_ir::op_interface::DynIndirectCallLike::get(&ctx, op)
                    .and_then(|call| call.exact_signature(&ctx))
                    .unwrap(),
            );
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    assert!(signatures.contains(&dispatch_definition));
    let canonical_closure = tribute_passes::wasm::type_converter::closure_adt_type(&mut ctx);
    let signature = wasm::FuncSig::from_type_ref(&ctx, dispatch_definition).unwrap();
    assert_eq!(signature.inputs(&ctx)[2], canonical_closure);
    assert!(signature.results(&ctx).is_empty());
    let mut binary_dispatch_signatures = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&binary.bytes) {
        if let wasmparser::Payload::TypeSection(types) = payload.unwrap() {
            for group in types {
                for ty in group.unwrap().into_types() {
                    if let wasmparser::CompositeInnerType::Func(signature) = ty.composite_type.inner
                        && signature.params().len() == 7
                    {
                        use wasmparser::{RefType, ValType};
                        let concrete = |index| {
                            ValType::Ref(
                                RefType::new(
                                    true,
                                    wasmparser::HeapType::Concrete(
                                        wasmparser::UnpackedIndex::Module(index),
                                    ),
                                )
                                .unwrap(),
                            )
                        };
                        // The evidence is the GC array of markers.
                        assert!(
                            matches!(
                                signature.params()[0],
                                ValType::Ref(evidence)
                                    if matches!(evidence.heap_type(), wasmparser::HeapType::Concrete(_))
                            ),
                            "{:?}",
                            signature.params()
                        );
                        assert_eq!(
                            signature.params()[1..],
                            [
                                concrete(trunk_ir_wasm_backend::gc_types::CLOSURE_STRUCT_IDX),
                                concrete(trunk_ir_wasm_backend::gc_types::CLOSURE_STRUCT_IDX),
                                ValType::I32,
                                ValType::I32,
                                ValType::I32,
                                ValType::Ref(RefType::ANYREF)
                            ]
                        );
                        assert!(signature.results().is_empty());
                        binary_dispatch_signatures += 1;
                    }
                }
            }
        }
    }
    assert_eq!(binary_dispatch_signatures, 1);
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(&binary.bytes)
        .expect("canonical root dispatch binary validates");
}

fn source_from_str(path: &str, text: &str) -> SourceCst {
    salsa::with_attached_database(|db| SourceCst::from_source_str(db, path, text))
        .expect("attached db")
}

/// Programs whose boundary exit is observed on both targets. They cover
/// Direct and CPS callables, closures with captures, tail-resumptive and
/// general handlers, the CPS root bridge, and standard I/O.
const BOUNDARY_EXIT_PROGRAMS: &[(&str, &str)] = &[
    (
        "native_calculator.trb",
        include_str!("../../lang-examples/native_calculator.trb"),
    ),
    (
        "native_effects.trb",
        include_str!("../../lang-examples/native_effects.trb"),
    ),
    (
        "wasm_dynamic_output.trb",
        include_str!("../../lang-examples/wasm_dynamic_output.trb"),
    ),
    (
        "closure_capture.trb",
        r#"fn apply(f: fn(Int) ->{e} Int, x: Int) ->{e} Int { f(x) }
fn main() -> Nil {
    let a = +1
    let _ = apply(fn(n) { n + a }, +41)
}
"#,
    ),
    (
        "tail_resumptive_handler.trb",
        r#"ability Ask {
    fn ask() -> Nat
}

fn use_ask() ->{Ask} Nat {
    Ask::ask()
}

fn main() -> Nil {
    let _ = handle use_ask() {
        do result { result }
        fn Ask::ask() { 42 }
    }
}
"#,
    ),
    (
        "state_handler.trb",
        r#"ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn set_then_get() ->{State(Int)} Int {
    State::set(+100)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(v) { run_state(fn() { resume Nil }, v) }
    }
}

fn main() -> Nil {
    let _ = run_state(fn() { set_then_get() }, +0)
}
"#,
    ),
];

/// Every program reaches `target`'s boundary exit, whose enforcement
/// rejects any violation.
fn assert_boundary_exit_programs_meet_the_contract(
    db: &crate::TributeDatabaseImpl,
    target: tribute_passes::abi_boundary::TargetKind,
) {
    let (mut declared, mut called) = (false, false);
    for (path, text) in BOUNDARY_EXIT_PROGRAMS {
        let source = source_from_str(path, text);
        let (mut ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .unwrap_or_else(|| panic!("{path} must lower"));
        let shared = trunk_ir::printer::print_module(&ctx, module.op());
        run_target_to_boundary_exit(&mut ctx, module, target)
            .unwrap_or_else(|error| panic!("{path}: target boundary failed: {error}"));
        let exit = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(
            !exit.contains("ability.decl"),
            "{path}: evidence lowering removes ability declarations:\n{exit}"
        );
        assert!(
            !exit.contains("__tribute_evidence_"),
            "{path}: evidence runs on the source runtime:\n{exit}"
        );
        declared |= shared.contains("ability.decl");
        called |= exit.contains("@\"std::runtime::evidence::");
    }
    assert!(declared, "a boundary program declares an ability instance");
    assert!(called, "a boundary program calls the evidence runtime");
}

#[salsa_test]
fn native_boundary_exit_programs_meet_the_contract(db: &crate::TributeDatabaseImpl) {
    assert_boundary_exit_programs_meet_the_contract(
        db,
        tribute_passes::abi_boundary::TargetKind::Native,
    );
}

#[salsa_test]
fn wasm_boundary_exit_programs_meet_the_contract(db: &crate::TributeDatabaseImpl) {
    assert_boundary_exit_programs_meet_the_contract(
        db,
        tribute_passes::abi_boundary::TargetKind::Wasm,
    );
}

#[test]
fn boundary_exit_enforcement_rejects_any_violation() {
    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_test_module(
        &mut ctx,
        r#"core.module @test {
  func.func @main() -> core.nil attributes {tribute.calling_convention = "direct"} {
    %nil = core.nil_value : core.nil
    func.return %nil
  }
}"#,
    );
    for target in [
        tribute_passes::abi_boundary::TargetKind::Native,
        tribute_passes::abi_boundary::TargetKind::Wasm,
    ] {
        let error = run_pass(
            &mut ctx,
            module,
            tribute_passes::abi_boundary::CheckBoundaryExit { target },
        )
        .expect_err("a forbidden attribute must fail the boundary exit");
        assert!(
            error
                .to_string()
                .contains("forbidden attribute tribute.calling_convention"),
            "{error}"
        );
    }
}

/// Print `module` and parse the text into a fresh context.
fn reparse_module(ctx: &IrContext, module: Module, path: &str) -> (IrContext, Module) {
    let text = trunk_ir::printer::print_module(ctx, module.op());
    let mut reparsed = IrContext::new();
    let op = trunk_ir::parser::parse_module(&mut reparsed, &text).unwrap_or_else(|error| {
        panic!(
            "{path}: printed boundary-exit IR must parse at offset {}: {}",
            error.offset, error.message
        )
    });
    let module = Module::new(&reparsed, op)
        .unwrap_or_else(|| panic!("{path}: parsed boundary-exit IR must be a module"));
    (reparsed, module)
}

/// Lowering and emission after the boundary exit need nothing but the
/// printed IR: a program parsed back from its boundary-exit text emits
/// the same binary as the original.
fn assert_boundary_exit_round_trips(
    db: &crate::TributeDatabaseImpl,
    target: tribute_passes::abi_boundary::TargetKind,
) {
    for (path, text) in BOUNDARY_EXIT_PROGRAMS {
        let source = source_from_str(path, text);
        let (mut ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .unwrap_or_else(|| panic!("{path} must lower"));
        run_target_to_boundary_exit(&mut ctx, module, target)
            .unwrap_or_else(|error| panic!("{path}: target boundary failed: {error}"));
        let (mut reparsed, reparsed_module) = reparse_module(&ctx, module, path);

        // Parsing registers every alias explicitly, which can change how
        // the first reprint spells types; the text is stable from then on.
        let reprinted = trunk_ir::printer::print_module(&reparsed, reparsed_module.op());
        let (again, again_module) = reparse_module(&reparsed, reparsed_module, path);
        assert_eq!(
            trunk_ir::printer::print_module(&again, again_module.op()),
            reprinted,
            "{path}: reparsed boundary-exit IR must print stably"
        );

        let direct = emit_from_boundary_exit(&mut ctx, module, target);
        let round_trip = emit_from_boundary_exit(&mut reparsed, reparsed_module, target);
        match (direct, round_trip) {
            (Ok(direct), Ok(round_trip)) => assert!(
                round_trip == direct,
                "{path}: the round-tripped IR must emit the same binary"
            ),
            // A program the target cannot emit yet, such as `read_line` on
            // Wasm, fails either way.
            (Err(_), Err(_)) => {}
            (direct, round_trip) => panic!(
                "{path}: emission outcome changed through text: direct {:?}, round trip {:?}",
                direct.map(|bytes| bytes.len()),
                round_trip.map(|bytes| bytes.len())
            ),
        }
    }
}

/// Every physical tail signature at the boundary exit states that the
/// callee consumes each input: definitions, function references, and
/// indirect calls, including the bridge and dispatch signatures the
/// boundary builds itself.
fn assert_tail_signatures_consume_every_input(
    db: &crate::TributeDatabaseImpl,
    target: tribute_passes::abi_boundary::TargetKind,
) {
    use trunk_ir::op_interface::DynIndirectCallLike;
    let consumed = |ctx: &IrContext, signature: func_dialect::FuncSig| {
        signature
            .input_attrs(ctx)
            .all(|attrs| attrs.view(ctx).get_str("tribute.ownership") == Some("consumed"))
    };
    let mut tail_signatures = 0;
    for (path, text) in BOUNDARY_EXIT_PROGRAMS {
        let source = source_from_str(path, text);
        let (mut ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .unwrap_or_else(|| panic!("{path} must lower"));
        run_target_to_boundary_exit(&mut ctx, module, target)
            .unwrap_or_else(|error| panic!("{path}: target boundary failed: {error}"));
        let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            let signature = if let Ok(function) = func_dialect::Func::from_op(&ctx, op) {
                Some(function.r#type(&ctx))
            } else if func_dialect::Constant::matches(&ctx, op) {
                ctx.op_result_types(op).first().copied()
            } else {
                DynIndirectCallLike::get(&ctx, op).and_then(|call| call.exact_signature(&ctx))
            };
            if let Some(signature) =
                signature.and_then(|ty| func_dialect::FuncSig::from_type_ref(&ctx, ty))
                && signature.call_conv(&ctx) == Some(func_dialect::CallConv::Tail)
            {
                tail_signatures += 1;
                assert!(
                    consumed(&ctx, signature),
                    "{path}: {target:?} tail signature lacks the consumed contract: {}",
                    trunk_ir::printer::print_type(&ctx, signature.as_type_ref())
                );
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    assert!(tail_signatures > 0, "no physical tail signature observed");
}

#[salsa_test]
fn native_tail_signatures_consume_every_input(db: &crate::TributeDatabaseImpl) {
    assert_tail_signatures_consume_every_input(
        db,
        tribute_passes::abi_boundary::TargetKind::Native,
    );
}

#[salsa_test]
fn wasm_tail_signatures_consume_every_input(db: &crate::TributeDatabaseImpl) {
    assert_tail_signatures_consume_every_input(db, tribute_passes::abi_boundary::TargetKind::Wasm);
}

#[salsa_test]
fn native_boundary_exit_round_trips_through_text(db: &crate::TributeDatabaseImpl) {
    assert_boundary_exit_round_trips(db, tribute_passes::abi_boundary::TargetKind::Native);
}

#[salsa_test]
fn wasm_boundary_exit_round_trips_through_text(db: &crate::TributeDatabaseImpl) {
    assert_boundary_exit_round_trips(db, tribute_passes::abi_boundary::TargetKind::Wasm);
}

fn prepare_native_fixture(
    db: &crate::TributeDatabaseImpl,
    path: &str,
    text: &str,
) -> (IrContext, Module) {
    let source = source_from_str(path, text);
    let (mut ctx, module) = run_shared_pipeline(db, source)
        .expect("shared pipeline must succeed")
        .expect("fixture must lower");
    run_target_to_boundary_exit(
        &mut ctx,
        module,
        tribute_passes::abi_boundary::TargetKind::Native,
    )
    .expect("native target lowering must succeed");
    run_stages(
        &mut ctx,
        module,
        StageConfig::target(tribute_passes::abi_boundary::TargetKind::Native),
        Stage::LowerTarget..=Stage::LowerTarget,
    )
    .expect("native preparation must succeed");
    trunk_ir_cranelift_backend::validate_clif_ir(&ctx, module)
        .expect("prepared native IR must satisfy exact callable slot contracts");
    (ctx, module)
}

/// Indirect calls of the compiled program, excluding the RTTI release
/// dispatch that native lowering generates.
fn clif_indirect_calls(ctx: &IrContext, module: Module) -> Vec<trunk_ir::OpRef> {
    let body = module.body(ctx).expect("module must have a body");
    let mut calls = Vec::new();
    let _ = walk_region::<()>(ctx, body, &mut |op| {
        if clif::Func::from_op(ctx, op).is_ok_and(|function| {
            function.sym_name(ctx) == tribute_passes::native::rtti::DEEP_RELEASE_FN
        }) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        if clif::CallIndirect::matches(ctx, op) || clif::ReturnCallIndirect::matches(ctx, op) {
            calls.push(op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    calls
}

fn clif_indirect_signature(ctx: &IrContext, call: trunk_ir::OpRef) -> clif::FuncSig {
    let signature = ctx
        .op_attrs(call)
        .get_type("sig")
        .and_then(|ty| clif::FuncSig::from_type_ref(ctx, ty))
        .expect("indirect call must have an exact signature");
    let operands = ctx.op_operands(call);
    assert_eq!(
        operands[1..]
            .iter()
            .map(|&operand| ctx.value_ty(operand))
            .collect::<Vec<_>>(),
        signature.inputs(ctx),
        "indirect call operands must match its exact signature"
    );
    assert_eq!(
        ctx.op_result_types(call),
        signature.results(ctx),
        "indirect call results must match its exact signature"
    );
    signature
}

#[salsa_test]
fn native_preparation_direct_closure_call_slots_are_exact(db: &crate::TributeDatabaseImpl) {
    let (mut ctx, module) = prepare_native_fixture(
        db,
        "closure_exec_simple.trb",
        r#"
use std::io::{Io, print_line}

fn main() ->{Io} Nil {
    let f = fn(x) { x + 1 }
    print_line(f(41).to_string)
}
"#,
    );
    let pointer_type = core_dialect::Ptr::params().intern(&mut ctx).as_type_ref();
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .expect("fixture must contain core.i32");
    let calls = clif_indirect_calls(&ctx, module);
    assert_eq!(
        calls.len(),
        1,
        "pure closure fixture must have one Direct indirect call"
    );
    let call = calls[0];
    let signature = clif_indirect_signature(&ctx, call);
    assert_eq!(signature.inputs(&ctx), [pointer_type, i32_type]);
    assert_eq!(signature.results(&ctx), [i32_type]);
}

#[salsa_test]
fn native_preparation_cps_closure_call_slots_are_exact(db: &crate::TributeDatabaseImpl) {
    let (mut ctx, module) = prepare_native_fixture(
        db,
        "closure_exec_callback.trb",
        r#"
use std::io::{Io, print_line}

ability Tick {
    op tick() -> Int
}

fn apply(f: fn(Int) ->{e} Int, value: Int) ->{e} Int { f(value) }

fn run(seed: fn() ->{r} Int) ->{r} Int { apply(fn(x) { x + +1 }, seed()) }

fn main() ->{Io} Nil {
    let result = handle run(fn() { Tick::tick() }) {
        do value { value }
        op Tick::tick() { resume +41 }
    }
    print_line(Int::to_string(result))
}
"#,
    );
    let pointer_type = core_dialect::Ptr::params().intern(&mut ctx).as_type_ref();
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .expect("fixture must contain core.i32");
    let calls: Vec<_> = clif_indirect_calls(&ctx, module)
        .into_iter()
        .filter(|&call| {
            clif::ReturnCallIndirect::matches(&ctx, call)
                && ctx.op_operands(call).len() == 3
                && ctx.op_operands(call).last().is_some_and(|&argument| {
                    matches!(
                        ctx.value_def(argument),
                        trunk_ir::refs::ValueDef::OpResult(producer, _)
                            if clif::Iadd::matches(&ctx, producer)
                    )
                })
        })
        .collect();
    assert_eq!(
        calls.len(),
        1,
        "the Cps callback fixture must have one CPS call with the addition result"
    );
    let call = calls[0];
    let [callee, continuation_environment, value] = ctx.op_operands(call) else {
        panic!("CPS Done tail must have one callee, its environment, and the value");
    };
    assert!(clif::ReturnCallIndirect::matches(&ctx, call));
    assert_eq!(ctx.op_attrs(call).get("tribute.calling_convention"), None);
    // The Done closure is its own environment.
    let done = *continuation_environment;
    let trunk_ir::refs::ValueDef::OpResult(done_load, 0) = ctx.value_def(done) else {
        panic!("Done closure must be loaded from its continuation frame");
    };
    let done_load = clif::Load::from_op(&ctx, done_load).expect("Done must come from clif.load");
    assert_eq!(done_load.offset(&ctx), 0);
    let frame = ctx.op_operands(done_load.op_ref())[0];
    assert!(matches!(
        ctx.value_def(frame),
        trunk_ir::refs::ValueDef::BlockArg(_, 2)
    ));
    let trunk_ir::refs::ValueDef::OpResult(callee_load, 0) = ctx.value_def(*callee) else {
        panic!("Done callee must be loaded from the same closure");
    };
    let callee_load = clif::Load::from_op(&ctx, callee_load).expect("callee load");
    assert_eq!(callee_load.offset(&ctx), 0);
    assert_eq!(ctx.op_operands(callee_load.op_ref()), [done]);
    assert!(matches!(
        ctx.value_def(*value),
        trunk_ir::refs::ValueDef::OpResult(producer, 0)
            if clif::Iadd::matches(&ctx, producer)
    ));
    let signature = clif_indirect_signature(&ctx, call);
    assert_eq!(signature.inputs(&ctx), [pointer_type, i32_type]);
    assert!(signature.results(&ctx).is_empty());
}

#[cfg(unix)]
#[test]
fn native_one_shot_wrapper_traps_on_second_invocation() {
    use std::os::unix::process::ExitStatusExt;

    let input = r#"core.module @one_shot {
  !state = adt.struct<OneShotState(consumed: core.i1)>

  func.func @one_shot_wrapper(%state: !state, %value: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %consumed = adt.struct_get %state {field = 0, type = !state} : core.i1
    %answer = scf.if %consumed : core.i32 {
      func.unreachable
    } {
      %consumed_true = arith.const {value = 1} : core.i1
      adt.struct_set %state, %consumed_true {field = 0, type = !state}
      scf.yield %value
    }
    func.return %answer
  }

  func.func @main() -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %not_consumed = arith.const {value = 0} : core.i1
    %state = adt.struct_new %not_consumed {type = !state} : !state
    %input = arith.const {value = 0} : core.i32
    %first = func.call %state, %input {callee = @one_shot_wrapper, tribute.calling_convention = "direct"} : core.i32
    %second = func.call %state, %first {callee = @one_shot_wrapper, tribute.calling_convention = "direct"} : core.i32
    func.return %second
  }
}"#;
    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_test_module(&mut ctx, input);

    let object = lower_and_emit_native(
        &mut ctx,
        module,
        false,
        NativeOptimizationOptions::production(),
    )
    .expect("one-shot wrapper must compile to native code");
    let temp = tempfile::tempdir().expect("temporary executable directory");
    let executable = temp.path().join("one-shot-wrapper");
    link_native_binary(&object, &executable, None).expect("one-shot wrapper must link");

    let status = std::process::Command::new(executable)
        .status()
        .expect("one-shot wrapper executable must start");
    assert!(
        status.signal().is_some(),
        "the second invocation of the same wrapper must terminate by signal, got {status}"
    );
}

/// A physical module carries its callable contract in signatures alone:
/// no semantic calling convention, root, or frame metadata is present.
fn physical_tail_chain_module(done_call_conv: &str, reference_call_conv: &str) -> String {
    format!(
        r#"core.module @physical {{
  func.func @fail() {{
    func.unreachable
  }}

  func.func @done(%value: core.i32) attributes {{type = func.func_sig<(core.i32) -> (){done_call_conv}>}} {{
    %expected = arith.const {{value = 3}} : core.i32
    %ok = arith.cmpi %value, %expected {{predicate = "eq"}} : core.i1
    scf.if %ok {{
      scf.yield
    }} {{
      func.call {{callee = @fail}}
      scf.yield
    }}
    func.return
  }}

  func.func @step(%value: core.i32) attributes {{type = func.func_sig<(core.i32) -> (), {{call_conv = "tail"}}>}} {{
    %one = arith.const {{value = 1}} : core.i32
    %next = arith.addi %value, %one : core.i32
    %done = func.constant {{func_ref = @done}} : func.func_sig<(core.i32) -> (){reference_call_conv}>
    func.tail_call_indirect %done, %next {{signature = func.func_sig<(core.i32) -> (), {{call_conv = "tail"}}>}}
  }}

  func.func @start(%value: core.i32) attributes {{type = func.func_sig<(core.i32) -> (), {{call_conv = "tail"}}>}} {{
    %one = arith.const {{value = 1}} : core.i32
    %next = arith.addi %value, %one : core.i32
    func.tail_call %next {{callee = @step}}
  }}

  func.func @main() -> core.i32 {{
    %input = arith.const {{value = 1}} : core.i32
    func.call %input {{callee = @start}}
    %exit = arith.const {{value = 0}} : core.i32
    func.return %exit
  }}
}}"#
    )
}

#[test]
fn physical_module_without_control_metadata_runs_native_tail_calls() {
    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_test_module(
        &mut ctx,
        &physical_tail_chain_module(", {call_conv = \"tail\"}", ", {call_conv = \"tail\"}"),
    );
    let verified = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
    assert!(verified.is_ok(), "{verified}");

    let object = lower_and_emit_native(
        &mut ctx,
        module,
        false,
        NativeOptimizationOptions::production(),
    )
    .expect("a physical module must lower without semantic control metadata");
    let temp = tempfile::tempdir().expect("temporary executable directory");
    let executable = temp.path().join("physical-tail-chain");
    link_native_binary(&object, &executable, None).expect("physical module must link");

    let status = std::process::Command::new(executable)
        .status()
        .expect("physical module executable must start");
    assert!(
        status.success(),
        "direct and indirect tail transfers must deliver the value, got {status}"
    );
}

#[test]
fn physical_module_with_mismatched_tail_convention_is_rejected() {
    // A reference typed like its platform target cannot flow into a
    // tail-convention indirect call: the typed-callee verifier rejects it.
    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_test_module(&mut ctx, &physical_tail_chain_module("", ""));
    let verified = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
    assert!(
        verified
            .to_string()
            .contains("exact indirect signature differs from typed callee"),
        "{verified}"
    );

    // A reference typed with the wrong convention for its target passes
    // the typed-callee check, so lowering must refuse to erase it.
    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_test_module(
        &mut ctx,
        &physical_tail_chain_module("", ", {call_conv = \"tail\"}"),
    );
    let verified = trunk_ir::validation::validate_operation_verifiers(&ctx, module);
    assert!(verified.is_ok(), "{verified}");
    let error = lower_and_emit_native(
        &mut ctx,
        module,
        false,
        NativeOptimizationOptions::production(),
    )
    .expect_err("a tail transfer to a platform-convention function must not be emitted");
    assert!(error.to_string().contains("func.constant"), "{error}");
}

#[cfg(debug_assertions)]
#[test]
fn debug_use_chain_verifier_reports_offending_pass() {
    let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %y = arith.addi %x, %x : core.i32
    func.return %y
  }
}"#;
    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_test_module(&mut ctx, input);
    let core_module = core_dialect::Module::from_op(&ctx, module.op())
        .expect("test input must parse a core.module");

    let mut pm = PassManager::new();
    struct BreakUseChain;
    impl trunk_ir::pass::Pass for BreakUseChain {
        type Target = core_dialect::Module;
        fn name(&self) -> &'static str {
            "break-use-chain"
        }
        fn run(
            &mut self,
            ctx: &mut IrContext,
            module: core_dialect::Module,
            _analyses: &mut AnalysisCache,
        ) -> trunk_ir::pass::PassRunResult {
            let module_block = ctx.region(module.body(ctx)).blocks[0];
            let func_op = ctx.block(module_block).ops[0];
            let func_region = ctx.op_region(func_op, 0).unwrap();
            let func_block = ctx.region(func_region).blocks[0];
            let add_op = ctx
                .block(func_block)
                .ops
                .iter()
                .copied()
                .find(|&op| {
                    let data = ctx.op(op);
                    data.name.is(ctx, "arith.addi")
                })
                .expect("test input must contain arith.addi");

            // Deliberately mutate operands without updating use-chains to
            // exercise the graph-wide pass-manager verifier.
            let _old_operands = std::mem::take(&mut ctx.op_mut(add_op).operands);
            Ok(())
        }
    }

    pm.add_pass(BreakUseChain);
    pm.with_debug_verifier();

    let error = pm
        .run(&mut ctx, core_module, &mut Default::default())
        .unwrap_err();

    assert_eq!(error.pass_name(), "break-use-chain");
    assert!(
        error
            .to_string()
            .contains("pass `break-use-chain` broke an IR invariant: use-chain regression"),
        "{error}"
    );
    assert!(
        error.to_string().contains("no such operand exists"),
        "{error}"
    );
}

#[cfg(debug_assertions)]
#[test]
fn debug_verifier_reports_schema_violations_after_a_pass() {
    let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %y = arith.addi %x, %x : core.i32
    func.return %y
  }
}"#;
    let mut ctx = IrContext::new();
    let module = trunk_ir::parser::parse_test_module(&mut ctx, input);
    let core_module = core_dialect::Module::from_op(&ctx, module.op())
        .expect("test input must parse a core.module");

    let mut pm = PassManager::new();
    struct BreakSchema;
    impl trunk_ir::pass::Pass for BreakSchema {
        type Target = core_dialect::Module;
        fn name(&self) -> &'static str {
            "break-schema"
        }
        fn run(
            &mut self,
            ctx: &mut IrContext,
            module: core_dialect::Module,
            _analyses: &mut AnalysisCache,
        ) -> trunk_ir::pass::PassRunResult {
            let module_block = ctx.region(module.body(ctx)).blocks[0];
            let func_op = ctx.block(module_block).ops[0];
            let func_block = ctx.region(ctx.op_region(func_op, 0).unwrap()).blocks[0];
            let add_op = ctx.block(func_block).ops[0];
            // Retype the result without touching use-chains, so only the
            // `T` binding of `arith.addi` is violated.
            let i64_ty = trunk_ir::types::TypeDataBuilder::new("core.i64").intern(ctx);
            ctx.set_op_result_type(add_op, 0, i64_ty);
            Ok(())
        }
    }

    pm.add_pass(BreakSchema);
    pm.with_debug_verifier();

    let error = pm
        .run(&mut ctx, core_module, &mut Default::default())
        .unwrap_err();

    assert_eq!(error.pass_name(), "break-schema");
    let message = error.to_string();
    assert!(message.contains("schema regression"), "{message}");
    assert!(message.contains("arith.addi"), "{message}");
}

#[salsa_test]
fn test_full_pipeline(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        "fn compute() -> Int { +42 }\nfn main() -> Nil { }",
    );

    let result = compile_ast(db, source).expect("pipeline should not fail");
    assert!(result.is_some(), "Should compile successfully");
    let (ctx, m) = result.unwrap();
    assert_eq!(m.name(&ctx), Some("test"));
}

#[salsa_test]
fn production_route_legalizes_source_control_with_exact_declarations(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "production-route.trb",
        r#"
ability Counter {
    op next() -> Int
}

fn count() ->{Counter} Int {
    Counter::next()
}

fn main() -> Nil {
    let _ = handle count() {
        do result { result }
        op Counter::next() { resume +1 }
    }
}
"#,
    );
    let frontend = compile_frontend_for_shared_route(db, source)
        .expect("production frontend route should lower");
    let logical = trunk_ir::printer::print_module(&frontend.context, frontend.module.op());
    assert!(
        logical.contains("tribute_control.perform"),
        "frontend must retain source logical perform:\n{logical}"
    );
    let declared = |operation: &str| {
        logical
            .find(&format!(
                "ability.operation {{kind = \"op\", op_name = \"{operation}\""
            ))
            .unwrap_or_else(|| panic!("missing declaration of `{operation}`:\n{logical}"))
    };
    assert!(
        declared("throw") < declared("next"),
        "ability declarations must preserve prelude-before-source order"
    );
    let (ctx, module) = compile_ast(db, source)
        .expect("shared lowering should succeed")
        .expect("shared lowering should produce a module");
    let output = trunk_ir::printer::print_module(&ctx, module.op());
    for forbidden in [
        "tribute_control.",
        "tribute_control.func_sig",
        "resume_token",
        "ability.legacy_",
    ] {
        assert!(
            !output.contains(forbidden),
            "shared route leaked `{forbidden}`:\n{output}"
        );
    }
}

#[salsa_test]
fn test_std_io_lowers_to_shared_dialect(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "io.trb",
        r#"
use std::io::print
use std::io::print_line
use std::io::read_line

fn input() ->{std::io::Io, std::abilities::Throw(std::io::Error)} String {
    read_line()
}

fn main() ->{std::io::Io} Nil {
    print("hello")
    print_line("hello")
}
"#,
    );

    let (ctx, module) = compile_ast(db, source)
        .expect("pipeline should not fail")
        .expect("pipeline should produce a module");
    let output = trunk_ir::printer::print_module(&ctx, module.op());

    assert!(output.contains("tribute_io.write"), "{output}");
    assert!(output.contains("tribute_io.read_line"), "{output}");
    assert!(!output.contains("__tribute_io_"), "{output}");

    let mut newline_values = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
        if let Ok(write) = tribute_ir::dialect::tribute_io::Write::from_op(&ctx, op)
            && let trunk_ir::refs::ValueDef::OpResult(producer, _) =
                ctx.value_def(write.newline(&ctx))
            && let Ok(constant) = trunk_ir::dialect::arith::Const::from_op(&ctx, producer)
            && let trunk_ir::Attribute::Bool(value) = constant.value(&ctx)
        {
            newline_values.push(value);
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    newline_values.sort_unstable();
    assert_eq!(newline_values, [false, true], "{output}");
}

#[salsa_test]
fn test_compile_with_diagnostics(db: &salsa::DatabaseImpl) {
    let source = source_from_str("test.trb", "fn add(x: Int, y: Int) -> Int { x + y }");

    let result = compile_with_diagnostics(db, source);
    // Should compile without errors
    assert!(
        result.diagnostics.is_empty(),
        "Expected no diagnostics, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_unresolved_reference_diagnostic(db: &salsa::DatabaseImpl) {
    let source = source_from_str("test.trb", "fn main() -> Int { undefined_var }");

    let result = compile_with_diagnostics(db, source);
    // Should have an unresolved reference error
    assert!(
        !result.diagnostics.is_empty(),
        "Expected diagnostic for unresolved reference"
    );

    // Check that the diagnostic message mentions the unresolved name
    let has_unresolved_error = result.diagnostics.iter().any(|d| {
        d.inner.message.contains("unresolved")
            && d.inner.severity == DiagnosticSeverity::Error
            && d.phase == CompilationPhase::NameResolution
    });
    assert!(
        has_unresolved_error,
        "Expected unresolved name error, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_parse_error_diagnostic(db: &salsa::DatabaseImpl) {
    let source = source_from_str("test.trb", "fn identity(a)(x: a) -> a { x }");
    let result = compile_with_diagnostics(db, source);

    let has_parse_error = result.diagnostics.iter().any(|d| {
        d.inner.severity == DiagnosticSeverity::Error
            && d.phase == CompilationPhase::Parsing
            && d.inner.message.contains("syntax error")
    });
    assert!(
        has_parse_error,
        "Expected parse error diagnostic, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_prelude_loads(db: &salsa::DatabaseImpl) {
    let prelude = prelude_module(db);
    assert!(prelude.is_some(), "Prelude should load successfully");
}

#[salsa_test]
fn test_prelude_exports_share_typed_declaration_identity(db: &salsa::DatabaseImpl) {
    let exports = prelude_exports(db).expect("prelude exports");
    let typed = prelude_module(db).expect("typed prelude");
    for (id, exported) in exports.function_types(db) {
        let (_, definition) = typed
            .declarations(db)
            .function_types
            .iter()
            .find(|(path, _)| path == id.path(db))
            .expect("exported function has a typed declaration");
        assert_eq!(exported, definition, "{}", id.qualified(db));
    }
}

#[salsa_test]
fn test_prelude_option_type(db: &salsa::DatabaseImpl) {
    // Use Option type from prelude
    let source = source_from_str("test.trb", "fn maybe() -> Option(Int) { .None }");

    let result = compile_with_diagnostics(db, source);
    // Should compile without "unresolved" errors for Option or None
    let has_option_error = result
        .diagnostics
        .iter()
        .any(|d| d.inner.message.contains("Option") || d.inner.message.contains("None"));
    assert!(
        !has_option_error,
        "Option and None should be available from prelude, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_prelude_result_type(db: &salsa::DatabaseImpl) {
    // Use Result type from prelude
    let source = source_from_str(
        "test.trb",
        "fn success() -> Result(Int, String) { .Ok(42) }",
    );

    let result = compile_with_diagnostics(db, source);
    // Should compile without "unresolved" errors for Result or Ok
    let has_result_error = result
        .diagnostics
        .iter()
        .any(|d| d.inner.message.contains("Result") || d.inner.message.contains("Ok"));
    assert!(
        !has_result_error,
        "Result and Ok should be available from prelude, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_case_expression_pattern_binding(db: &salsa::DatabaseImpl) {
    // Simple case expression with identifier pattern binding
    let source = source_from_str(
        "test.trb",
        r#"
            fn test(x: Int) -> Int {
                case x {
                    y -> y
                }
            }
            "#,
    );

    let result = compile_with_diagnostics(db, source);
    // Pattern binding `y` should be resolved in the case arm body
    let has_unresolved_y = result
        .diagnostics
        .iter()
        .any(|d| d.inner.message.contains("unresolved") && d.inner.message.contains("y"));
    assert!(
        !has_unresolved_y,
        "Pattern binding `y` should be resolved, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_case_lowering_exhaustive(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
            fn test(x: Nat) -> Nat {
                case x {
                    0 -> 1
                    _ -> 2
                }
            }
            "#,
    );

    let result = compile_with_diagnostics(db, source);
    assert!(
        result.diagnostics.is_empty(),
        "Expected no diagnostics, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_case_lowering_non_exhaustive(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
            fn test(x: Nat) -> Nat {
                case x {
                    0 -> 1
                }
            }
            "#,
    );

    let result = compile_with_diagnostics(db, source);
    let has_non_exhaustive = result.diagnostics.iter().any(|d| {
        d.inner.message.contains("non-exhaustive") && d.inner.severity == DiagnosticSeverity::Error
    });
    assert!(
        has_non_exhaustive,
        "Expected non-exhaustive case diagnostic, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_case_lowering_nested_non_exhaustive(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
            fn pick(o: Option(Bool)) -> Nat {
                case o {
                    .None -> 0
                    .Some(True) -> 1
                }
            }
            "#,
    );

    let result = compile_with_diagnostics(db, source);
    let has_non_exhaustive = result.diagnostics.iter().any(|d| {
        d.inner.message.contains("missing patterns: .Some(False)")
            && d.inner.severity == DiagnosticSeverity::Error
    });
    assert!(
        has_non_exhaustive,
        "Expected non-exhaustive case diagnostic, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_case_lowering_bool_exhaustive(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
            fn test(x: Bool) -> Nat {
                case x {
                    True -> 1
                    False -> 0
                }
            }
            "#,
    );

    let result = compile_with_diagnostics(db, source);
    assert!(
        result.diagnostics.is_empty(),
        "Expected no diagnostics, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_tdnr_struct_field_access(db: &salsa::DatabaseImpl) {
    // Test that TDNR resolves user.name to User::name(user)
    let source = source_from_str(
        "test.trb",
        r#"
            struct User {
                name: String,
                age: Int,
            }

            fn get_name(user: User) -> String {
                user.name
            }
            "#,
    );

    let result = compile_with_diagnostics(db, source);

    // Should compile without unresolved errors - TDNR should resolve user.name
    let has_unresolved_name = result
        .diagnostics
        .iter()
        .any(|d| d.inner.message.contains("unresolved") && d.inner.message.contains("name"));
    assert!(
        !has_unresolved_name,
        "TDNR should resolve user.name to User::name(user), got: {:?}",
        result.diagnostics
    );
}

// =========================================================================
// AST-based Pipeline Tests
// =========================================================================

#[salsa_test]
fn test_ast_pipeline_simple_function(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        "fn compute() -> Int { +42 }\nfn main() -> Nil { }",
    );

    let result = compile_frontend(db, source);
    assert!(result.is_some());
    let (ctx, m) = result.unwrap();
    assert_eq!(m.name(&ctx), Some("test"));
}

#[salsa_test]
fn test_ast_pipeline_with_params(db: &salsa::DatabaseImpl) {
    let source = source_from_str("test.trb", "fn add(x: Int, y: Int) -> Int { x + y }");

    let result = compile_frontend(db, source);
    assert!(result.is_some());
    let (ctx, m) = result.unwrap();
    assert_eq!(m.name(&ctx), Some("test"));
}

#[test]
fn generic_intrinsic_specialization_reaches_concrete_lowering() {
    salsa::Database::attach(&salsa::DatabaseImpl::default(), |db| {
        let source = source_from_str(
            "prelude_list_prepend.trb",
            r#"
fn main() -> Nil {
    let _ = List::prepend("token", [])
}
"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        let monomorphized =
            prepare_frontend_for_lowering(db, typed, source).expect("specialized frontend");
        let ast = format!("{:#?}", monomorphized.module(db));
        let concrete = "std::collections::List::__tribute_list_prepend_intrinsic$std::String";

        assert!(
            has_function_type(
                db,
                monomorphized,
                &SymbolPath::new([
                    "std",
                    "collections",
                    "List",
                    "__tribute_list_prepend_intrinsic$std::String",
                ])
            ),
            "monomorphization must retain the concrete intrinsic scheme"
        );
        assert!(
            ast.contains("__tribute_list_prepend_intrinsic$std::String"),
            "specialized List body must call the concrete intrinsic:\n{ast}"
        );
        assert!(
            !ast.contains("std::collections::List::__tribute_list_prepend_intrinsic$T"),
            "specialized List body must not retain generic intrinsic binders:\n{ast}"
        );

        let frontend = compile_frontend_for_shared_route(db, source)
            .expect("source-logical lowering must retain the specialization");
        let concrete_symbol = trunk_ir::Symbol::new(concrete);
        assert!(
            frontend.compiler_intrinsics.iter().any(|declaration| {
                declaration.symbol == SymbolPath::new([concrete_symbol.clone()])
                    && declaration.identity
                        == trunk_ir::Symbol::new(
                            "std::collections::List::__tribute_list_prepend_intrinsic",
                        )
            }),
            "the concrete specialization must retain its base intrinsic identity"
        );
        let logical = trunk_ir::printer::print_module(&frontend.context, frontend.module.op());
        assert!(
            logical.contains(
                r#"tribute.compiler_intrinsic = "std::collections::List::__tribute_list_prepend_intrinsic""#
            ),
            "the concrete declaration must carry its exact intrinsic identity:\n{logical}"
        );

        let (ctx, module) = compile_ast(db, source)
            .expect("shared pipeline must lower the concrete intrinsic specialization")
            .expect("shared pipeline must produce a module");
        let lowered = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(lowered.contains("list.prepend"), "{lowered}");
        assert!(
            !lowered.contains("std::collections::List::__tribute_list_prepend_intrinsic"),
            "list intrinsic lowering must consume the concrete declaration:\n{lowered}"
        );
    });
}

#[salsa_test]
fn list_import_paths_share_canonical_specialization(db: &salsa::DatabaseImpl) {
    use tribute_front::ast::Decl;
    use trunk_ir::Symbol;

    let source = source_from_str(
        "list_import_paths.trb",
        r#"
use std::collections::List as Sequence
use std::collections::List::prepend as push

fn main() -> Nil {
    let first = List::prepend(1, [])
    let second = std::collections::List::prepend(2, first)
    let third = Sequence::prepend(3, second)
    let _ = push(4, third)
}
"#,
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    let diagnostics = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let instances = &typed.expression_types(db).function_instances;
    let canonical = Symbol::new("std::collections::List::prepend");
    assert_eq!(instances.len(), 4);
    let declaration = instances[0].1.function;
    assert_eq!(declaration.qualified(db), canonical);
    assert!(
        instances
            .iter()
            .all(|(_, instance)| instance.function == declaration),
        "{instances:#?}"
    );

    let prepared = prepare_frontend_for_lowering(db, typed, source).expect("specialized frontend");
    let specialized = Symbol::new("std::collections::List::prepend$Nat");
    assert_eq!(
        prepared
            .module(db)
            .entries
            .iter()
            .filter_map(declaration_with_name)
            .filter(|(name, decl)| matches!(decl, Decl::Function(_)) && *name == specialized)
            .count(),
        1,
        "all import paths must share one specialized definition"
    );
    assert!(has_function_type(
        db,
        prepared,
        &SymbolPath::new(["std", "collections", "List", "prepend$Nat"])
    ));
    assert!(!has_function_type(
        db,
        prepared,
        &SymbolPath::new(["List", "prepend$Nat"])
    ));
}

/// Whether specialization left a function scheme at `path`.
fn has_function_type(
    db: &dyn salsa::Database,
    specialized: SpecializationOutput<'_>,
    path: &SymbolPath,
) -> bool {
    specialized
        .declarations(db)
        .function_types
        .iter()
        .any(|(function, _)| function == path)
}

#[salsa_test]
fn specialized_enum_schemas_and_dependencies_reach_logical_cps(db: &salsa::DatabaseImpl) {
    use tribute_front::ast::{Decl, TypeKind};
    use tribute_ir::dialect::adt::layout::{get_enum_variants, get_struct_fields};
    use trunk_ir::{Symbol, TypeRef};

    fn alias(ir: &IrContext, name: &str) -> TypeRef {
        let name = Symbol::new(name);
        let ty = ir
            .type_alias_by_name(&name)
            .expect("dependency layout is published");
        assert!(
            ir.type_attrs(ty)
                .get_str("name")
                .is_some_and(|text| name == text)
        );
        ty
    }
    fn target(ir: &IrContext, ty: TypeRef) -> TypeRef {
        assert!(tribute_ir::dialect::adt::Typeref::matches(ir, ty));
        ir.type_alias_by_text(ir.type_attrs(ty).get_str("name").unwrap())
            .expect("referenced layout")
    }
    let source = source_from_str(
        "specialized_enum_dependencies.trb",
        r#"
struct Inner(a) { value: a }
enum Outer(a) { Wrap(Inner(a)), Pair(#(a, fn(a) -> a)), Empty, Both(a, Bool), Label(String) }
struct Envelope(a) { outer: Outer(a) }
struct Link(a) { tail: Loop(a) }
enum Loop(a) { Next(Link(a)), Done(a) }
pub mod A { pub enum Token(a) { Item(a), Empty } }
pub mod B { pub enum Token(a) { Item(a), Empty } }
enum Boxed(a) { Box(a), EmptyBox }
fn keep(value: Envelope(Int)) -> Envelope(Int) { value }
fn keep_bool(value: Outer(Bool)) -> Outer(Bool) { value }
fn keep_loop(value: Loop(Int)) -> Loop(Int) { value }
fn keep_a(value: A::Token(Int)) -> A::Token(Int) { value }
fn keep_b(value: B::Token(Bool)) -> B::Token(Bool) { value }
fn int_payload(value: Boxed(Int)) -> Int { case value { .Box(x) -> x, .EmptyBox -> +0 } }
fn bool_payload(value: Boxed(Bool)) -> Bool { case value { .Box(x) -> x, .EmptyBox -> False } }
fn main() -> Nil {}
"#,
    );
    let typed = parse_and_lower_ast(db, source).expect("typed fixture");
    let prepared =
        prepare_frontend_for_lowering(db, typed, source).expect("closed nominal dependencies");
    let schemas = &prepared
        .declarations(db)
        .constructor_types
        .specialized_enum_variants;
    let mut seen = HashSet::default();
    for (name, declaration) in prepared
        .module(db)
        .entries
        .iter()
        .filter_map(declaration_with_name)
    {
        if let Decl::Enum(e) = declaration
            && e.id.variant().is_some()
        {
            for v in &e.variants {
                assert!(seen.insert(v.id));
                let matching: Vec<_> = schemas.iter().filter(|(id, _)| *id == v.id).collect();
                assert_eq!(matching.len(), 1);
                let scheme = matching[0].1;
                assert!(scheme.type_params(db).is_empty());
                let result = match scheme.body(db).kind(db) {
                    TypeKind::Func { params, result, .. } => {
                        assert_eq!(params.len(), v.fields.len());
                        *result
                    }
                    _ => {
                        assert!(v.fields.is_empty());
                        scheme.body(db)
                    }
                };
                assert!(matches!(result.kind(db), TypeKind::Named { id, args, .. }
                    if id.qualified(db) == name && args.is_empty()));
            }
        }
    }
    assert_eq!(seen.len(), schemas.len());
    for (id, source_scheme) in &typed.declarations(db).constructor_types.schemes {
        if id.qualified(db) == "Box" {
            let retained = prepared
                .declarations(db)
                .constructor_types
                .schemes
                .iter()
                .find(|(key, _)| key == id)
                .unwrap()
                .1;
            assert_eq!(
                retained, *source_scheme,
                "source generic schema is preserved"
            );
        }
    }
    let FrontendCompilation {
        context: mut ir,
        module,
        compiler_intrinsics,
    } = merge_and_lower_to_ir(db, &typed, source);
    for after_cps in [false, true] {
        if after_cps {
            tribute_passes::specialize_convention_classes::specialize(&mut ir, module)
                .expect("class specialization accepts frontend output");
            tribute_passes::tribute_control_to_cps::tribute_control_to_cps(
                &mut ir,
                module,
                &compiler_intrinsics,
                &mut Default::default(),
            )
            .unwrap();
        }
        use trunk_ir::dialect::core::{I1, I32};
        for (suffix, primitive) in [("Int", I32::NAME), ("Bool", I1::NAME)] {
            let inner = alias(&ir, &format!("Inner${suffix}"));
            let value = get_struct_fields(&ir, inner).unwrap()[0].1;
            assert_eq!(ir.get_type(value).name, primitive);
            let outer = get_enum_variants(&ir, alias(&ir, &format!("Outer${suffix}"))).unwrap();
            assert_eq!(ir.str(outer[0].0), "Wrap");
            assert_eq!(target(&ir, outer[0].1[0]), inner);
            assert!(outer[2].1.is_empty());
            assert_eq!(outer[3].1.len(), 2);
            assert_eq!(outer[3].1[0], value);
            assert_eq!(ir.get_type(outer[3].1[1]).name, I1::NAME);
            assert_eq!(target(&ir, outer[4].1[0]), alias(&ir, "std::String"));
            let tuple = target(&ir, outer[1].1[0]);
            let fields = get_struct_fields(&ir, tuple).unwrap();
            assert_eq!(fields[0].1, value);
            let callback = ir.get_type(fields[1].1);
            assert_eq!(
                callback.name.dialect(&ir),
                if after_cps {
                    "closure"
                } else {
                    "tribute_control"
                }
            );
            let boxed = get_enum_variants(&ir, alias(&ir, &format!("Boxed${suffix}"))).unwrap();
            assert_eq!(boxed[0].1, vec![value]);
        }
        let envelope = get_struct_fields(&ir, alias(&ir, "Envelope$Int")).unwrap();
        assert_eq!(target(&ir, envelope[0].1), alias(&ir, "Outer$Int"));
        let link = get_struct_fields(&ir, alias(&ir, "Link$Int")).unwrap();
        assert_eq!(target(&ir, link[0].1), alias(&ir, "Loop$Int"));
        let recursive = get_enum_variants(&ir, alias(&ir, "Loop$Int")).unwrap();
        assert_eq!(target(&ir, recursive[0].1[0]), alias(&ir, "Link$Int"));
        for (name, primitive) in [("A::Token$Int", I32::NAME), ("B::Token$Bool", I1::NAME)] {
            let variants = get_enum_variants(&ir, alias(&ir, name)).unwrap();
            assert_eq!(ir.str(variants[0].0), "Item");
            assert_eq!(ir.get_type(variants[0].1[0]).name, primitive);
        }
        let output = trunk_ir::printer::print_module(&ir, module.op());
        for (name, primitive) in [("Boxed$Int", "core.i32"), ("Boxed$Bool", "core.i1")] {
            assert!(
                output.lines().any(|line| line.contains("adt.variant_get")
                    && line.contains(name)
                    && line.ends_with(primitive)),
                "{output}"
            );
        }
    }
}

#[salsa_test]
fn logical_nominal_layouts_are_published_through_cps(db: &salsa::DatabaseImpl) {
    use tribute_ir::dialect::adt::layout::get_struct_fields;
    use tribute_ir::dialect::tribute_control;
    use trunk_ir::{Symbol, TypeRef};

    const STRUCT: trunk_ir::TypeName =
        <tribute_ir::dialect::adt::Struct as trunk_ir::ops::DialectType>::NAME;
    const ENUM: trunk_ir::TypeName =
        <tribute_ir::dialect::adt::Enum as trunk_ir::ops::DialectType>::NAME;

    fn layout(ir: &IrContext, name: &Symbol, kind: trunk_ir::TypeName) -> TypeRef {
        let ty = ir
            .type_alias_by_name(name)
            .expect("published nominal layout");
        assert_eq!(ir.get_type(ty).name, kind);
        assert!(
            ir.type_attrs(ty)
                .get_str("name")
                .is_some_and(|text| name == text)
        );
        assert_eq!(
            ir.type_aliases()
                .iter()
                .filter(|(_, ty)| {
                    ir.type_attrs(*ty)
                        .get_str("name")
                        .is_some_and(|text| name == text)
                })
                .count(),
            1,
            "one published layout for {name}"
        );
        ty
    }

    fn reference_name(ir: &IrContext, ty: TypeRef) -> Symbol {
        let data = ir.get_type(ty);
        assert!(data.name == tribute_ir::dialect::adt::Typeref::NAME);
        Symbol::new(ir.type_attrs(ty).get_str("name").expect("nominal identity"))
    }

    // These specializations occur only in signatures, never in allocations.
    let source = source_from_str(
        "published_nominal_layouts.trb",
        r#"
pub mod A { pub struct Token(a) {} }
pub mod B { pub struct Token(a) {} }
enum Choice(a) { Item(a), Empty }
struct Holder { pair: #(Nat, fn(Nat) -> Nat) }
struct Node { next: Node }
struct First { second: Second }
struct Second { first: First }

fn keep_a(value: A::Token(Nat)) -> A::Token(Nat) { value }
fn keep_b(value: B::Token(Nat)) -> B::Token(Nat) { value }
fn keep_choice(value: Choice(Nat)) -> Choice(Nat) { value }
fn keep_holder(value: Holder) -> Holder { value }
fn keep_node(value: Node) -> Node { value }
fn keep_first(value: First) -> First { value }
fn main() -> Nil {}
"#,
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    let diagnostics = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let FrontendCompilation {
        context: mut ir,
        module,
        compiler_intrinsics,
    } = merge_and_lower_to_ir(db, &typed, source);
    let mut source_tuple = None;
    for after_cps in [false, true] {
        if after_cps {
            tribute_passes::specialize_convention_classes::specialize(&mut ir, module)
                .expect("class specialization accepts frontend output");
            tribute_passes::tribute_control_to_cps::tribute_control_to_cps(
                &mut ir,
                module,
                &compiler_intrinsics,
                &mut Default::default(),
            )
            .expect("CPS converts published layout fields");
        }
        for (function, nominal, kind) in [
            ("keep_a", "A::Token$Nat", STRUCT),
            ("keep_b", "B::Token$Nat", STRUCT),
            ("keep_choice", "Choice$Nat", ENUM),
        ] {
            let parameter = module
                .ops(&ir)
                .iter()
                .copied()
                .find_map(|op| {
                    if after_cps {
                        let f = func_dialect::Func::from_op(&ir, op).ok()?;
                        (f.sym_name(&ir) == function).then(|| {
                            func_dialect::FuncSig::from_type_ref(&ir, f.r#type(&ir))
                                .unwrap()
                                .inputs(&ir)[0]
                        })
                    } else {
                        let f = tribute_control::Func::from_op(&ir, op).ok()?;
                        (f.sym_name(&ir) == function).then(|| {
                            tribute_control::FuncSig::from_type_ref(&ir, f.r#type(&ir))
                                .unwrap()
                                .inputs(&ir)[0]
                        })
                    }
                })
                .expect("signature-only specialization");
            let name = reference_name(&ir, parameter);
            assert_eq!(name, Symbol::new(nominal));
            layout(&ir, &name, kind);
        }
        assert_ne!(
            layout(&ir, &Symbol::new("A::Token$Nat"), STRUCT),
            layout(&ir, &Symbol::new("B::Token$Nat"), STRUCT)
        );
        for (owner, target) in [("Node", "Node"), ("First", "Second"), ("Second", "First")] {
            let owner = layout(&ir, &Symbol::new(owner), STRUCT);
            let field = get_struct_fields(&ir, owner).unwrap()[0].1;
            let name = reference_name(&ir, field);
            assert_eq!(name, Symbol::new(target));
            layout(&ir, &name, STRUCT);
        }
        let holder = layout(&ir, &Symbol::new("Holder"), STRUCT);
        let pair = get_struct_fields(&ir, holder).unwrap()[0].1;
        let tuple = layout(&ir, &reference_name(&ir, pair), STRUCT);
        let fields = get_struct_fields(&ir, tuple).unwrap();
        let callback = ir.get_type(fields[1].1);
        if after_cps {
            assert_ne!(Some(tuple), source_tuple);
            assert!(callback.name.is(&ir, "closure.closure"));
        } else {
            source_tuple = Some(tuple);
            assert!(callback.name.is(&ir, "tribute_control.func_sig"));
        }
    }
}

#[salsa_test]
fn source_list_does_not_change_logical_prelude_signatures(db: &salsa::DatabaseImpl) {
    use tribute_ir::dialect::tribute_control;
    use trunk_ir::ops::DialectOp;

    let source = source_from_str(
        "unused_source_list.trb",
        "enum List(a) { SourceList(a), }\nfn main() -> Nil {}",
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    let FrontendCompilation {
        context: ir,
        module,
        compiler_intrinsics,
    } = merge_and_lower_to_ir(db, &typed, source);
    let mut checked = 0;
    for &operation in module.ops(&ir) {
        let Ok(function) = tribute_control::Func::from_op(&ir, operation) else {
            continue;
        };
        if function.sym_name(&ir) == "std::collections::List::prepend" {
            let signature = tribute_control::FuncSig::from_type_ref(&ir, function.r#type(&ir))
                .expect("logical signature");
            let result = ir.get_type(signature.result(&ir));
            assert!(result.name.is(&ir, "tribute_rt.anyref"));
            assert_eq!(signature.inputs(&ir)[1], signature.result(&ir));
            checked += 1;
        }
    }
    assert_eq!(checked, 1);
    let validation =
        tribute_control::validate(&ir, module, &compiler_intrinsics, &mut Default::default());
    assert!(validation.is_ok(), "{validation}");
}

#[test]
fn recursive_generic_handler_collects_a_concrete_specialization() {
    salsa::Database::attach(&salsa::DatabaseImpl::default(), |db| {
        let source = source_from_str(
            "recursive_generic_handler.trb",
            r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        do result { result }
        op State::get() { run_state(fn() { resume init }, init) }
        op State::set(value) { run_state(fn() { resume Nil }, value) }
    }
}

fn consume(value: Nat) -> Nil { Nil }

fn main() -> Nil {
    consume(run_state(fn() { State::get() }, 0))
}
"#,
        );
        let typed = parse_and_lower_ast(db, source).expect("frontend output");
        let diagnostics = parse_and_lower_ast::accumulated::<Diagnostic>(db, source);
        assert!(
            diagnostics.is_empty(),
            "recursive handler fixture must type-check without diagnostics: {diagnostics:#?}"
        );
        let monomorphized =
            prepare_frontend_for_lowering(db, typed, source).expect("specialized frontend");
        let specialized = SymbolPath::new(["run_state$Nat$Nat"]);
        let scheme = monomorphized
            .declarations(db)
            .function_types
            .iter()
            .find_map(|(path, scheme)| (*path == specialized).then_some(*scheme))
            .expect("recursive handler call must collect a concrete run_state specialization");

        assert!(
            scheme.type_params(db).is_empty(),
            "specialized run_state must not retain generic binders: {scheme:?}"
        );
        let specialized_body = monomorphized
            .module(db)
            .decls()
            .find_map(|decl| match decl {
                tribute_front::ast::Decl::Function(function) if specialized == [&function.name] => {
                    Some(function)
                }
                _ => None,
            })
            .expect("monomorphization must materialize the concrete run_state body");
        assert!(
            !format!("{specialized_body:#?}").contains("BoundVar"),
            "specialized handler body must not retain BoundVar metadata"
        );
    });
}

#[salsa_test]
fn wasm_managed_c_ffi_omits_unreferenced_declaration(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "unused_c_ffi.trb",
        r#"extern "C" fn user_bridge(value: String) -> String
fn main() -> Nil { Nil }"#,
    );
    let binary =
        compile_to_wasm_binary(db, source).expect("unused managed C declaration may be omitted");
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(binary)
        .expect("unused C declaration must not leave an invalid Wasm definition");
}

#[salsa_test]
fn wasm_managed_c_ffi_requires_a_binding_when_referenced(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "used_c_ffi.trb",
        r#"extern "C" fn user_bridge(value: String) -> String
fn main() -> Nil {
    let _ = user_bridge("hello")
    Nil
}"#,
    );
    let errors =
        compile_to_wasm_binary(db, source).expect_err("C ABI alone must not create a Wasm import");
    // Reaching the boundary exit proves shared lowering accepted the
    // managed FFI signature, without running the shared pipeline again
    // just to check it.
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].phase, CompilationPhase::Lowering);
    let message = &errors[0].inner.message;
    assert!(
        message.contains("unsatisfiable runtime binding user_bridge"),
        "{message}"
    );
}

#[salsa_test]
fn unknown_intrinsic_directives_are_rejected_before_specialization(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "unknown_intrinsics.trb",
        r#"extern "intrinsic" fn user_intrinsic(value: Int) -> Int
mod Nested {
    extern "intrinsic" fn unused_generic(value: a) -> a
}"#,
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    assert!(parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty());
    assert!(prepare_frontend_for_lowering(db, typed, source).is_none());
    let diagnostics = prepare_frontend_for_lowering::accumulated::<Diagnostic>(db, typed, source);
    assert_eq!(diagnostics.len(), 2, "{diagnostics:?}");
    for (diagnostic, identity) in diagnostics
        .iter()
        .zip(["user_intrinsic", "Nested::unused_generic"])
    {
        assert_eq!(
            diagnostic.inner.message,
            format!("unsupported compiler intrinsic directive `{identity}`")
        );
        assert_eq!(diagnostic.inner.severity, DiagnosticSeverity::Error);
        assert_eq!(diagnostic.phase, CompilationPhase::Lowering);
        assert_ne!(diagnostic.inner.span, trunk_ir::Span::default());
    }
}

/// An intrinsic identity is the declaration's package path, so a user
/// declaration cannot claim one the `std` package owns.
#[salsa_test]
fn user_declaration_cannot_claim_a_library_intrinsic(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "library_intrinsic.trb",
        r#"extern "intrinsic" fn __bytes_get_or_panic(bytes: Bytes, index: Nat) -> Nat"#,
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    assert!(parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty());
    assert!(prepare_frontend_for_lowering(db, typed, source).is_none());
    let diagnostics = prepare_frontend_for_lowering::accumulated::<Diagnostic>(db, typed, source);
    itertools::assert_equal(
        diagnostics
            .iter()
            .map(|diagnostic| diagnostic.inner.message.as_str()),
        ["unsupported compiler intrinsic directive `__bytes_get_or_panic`"],
    );
}

#[salsa_test]
fn arithmetic_intrinsic_lowering_consumes_its_identities(db: &salsa::DatabaseImpl) {
    use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
    use trunk_ir::dialect::func;

    let source = source_from_str(
        "native_calculator.trb",
        include_str!("../../lang-examples/native_calculator.trb"),
    );
    let (ctx, module) = run_shared_pipeline(db, source)
        .expect("shared pipeline must succeed")
        .expect("fixture must lower");
    let identity = |name: &'static str| {
        module.ops(&ctx).iter().copied().find_map(|op| {
            let function = func::Func::from_op(&ctx, op).ok()?;
            (function.sym_name(&ctx) == name).then(|| {
                assert_eq!(ctx.op_attrs(op).get_str("abi"), Some("intrinsic"));
                ctx.op_attrs(op).get_str(COMPILER_INTRINSIC_ATTR)
            })
        })
    };
    assert_eq!(
        identity("std::Int::+"),
        None,
        "arithmetic lowering is the last reader of its identities and removes \
             declarations nothing references"
    );
    assert_eq!(
        identity("std::__bytes_get_or_panic"),
        Some(Some("std::__bytes_get_or_panic")),
        "bytes lowering consumes its identity inside the target boundary"
    );
}

#[salsa_test]
fn both_targets_consume_the_bytes_intrinsic_inside_the_boundary(db: &salsa::DatabaseImpl) {
    use tribute_ir::dialect::tribute_control::COMPILER_INTRINSIC_ATTR;
    use tribute_passes::abi_boundary::TargetKind;
    use trunk_ir::dialect::func;

    for target in [TargetKind::Native, TargetKind::Wasm] {
        let source = source_from_str(
            "native_calculator.trb",
            include_str!("../../lang-examples/native_calculator.trb"),
        );
        let (mut ctx, module) = run_shared_pipeline(db, source)
            .expect("shared pipeline must succeed")
            .expect("fixture must lower");
        match target {
            TargetKind::Native => run_target_to_boundary_exit(
                &mut ctx,
                module,
                tribute_passes::abi_boundary::TargetKind::Native,
            ),
            TargetKind::Wasm => run_target_to_boundary_exit(
                &mut ctx,
                module,
                tribute_passes::abi_boundary::TargetKind::Wasm,
            ),
        }
        .unwrap_or_else(|error| panic!("{target:?} boundary failed: {error}"));

        for &op in module.ops(&ctx) {
            assert!(
                ctx.op_attrs(op).get(COMPILER_INTRINSIC_ATTR).is_none(),
                "{target:?}: the identity is consumed at the boundary exit"
            );
            let declares_intrinsic = func::Func::from_op(&ctx, op)
                .is_ok_and(|function| function.sym_name(&ctx) == "__bytes_get_or_panic");
            assert!(
                !declares_intrinsic,
                "{target:?}: the intrinsic declaration is removed"
            );
        }
    }
}

/// Type checking reads package std through the dependency table: the
/// user's output holds only the user's declarations, and preparation
/// takes std's with them as one table.
#[salsa_test]
fn type_check_output_holds_only_its_package_declarations(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "own_declarations.trb",
        "fn helper(x: Int) -> Option(Int) { Some(x + +1) }\nfn main() -> Nil { let _ = helper(+1) }",
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    let own = typed.declarations(db);
    let is_std = |path: &SymbolPath| path.first().is_some_and(|first| first == "std");
    assert!(
        own.function_types.iter().all(|(path, _)| !is_std(path)),
        "{:?}",
        own.function_types
    );
    assert!(
        own.function_types
            .iter()
            .any(|(path, _)| *path == ["helper"])
    );
    assert!(
        own.constructor_types
            .schemes
            .iter()
            .all(|(id, _)| !is_std(id.path(db))),
        "{:?}",
        own.constructor_types.schemes
    );

    let prepared = prepare_frontend_for_lowering(db, typed, source).expect("prepared");
    let merged = prepared.declarations(db);
    assert!(
        merged
            .function_types
            .iter()
            .any(|(path, _)| *path == ["std", "Int", "+"])
    );
    assert!(
        merged
            .constructor_types
            .schemes
            .iter()
            .any(|(id, _)| *id.path(db) == ["std", "Option", "Some"])
    );
}

/// Literal patterns compare through the prelude's `==` for their type,
/// resolved by receiver type, even beside a user `String` lookalike with
/// its own `==`.
#[salsa_test]
fn literal_pattern_equalities_are_the_prelude_methods(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "lookalike_equality.trb",
        r#"
mod shadow {
    pub enum String { Leaf(Bytes) }

    pub mod String {
        pub fn (==)(left: shadow::String, right: shadow::String) -> Bool { True }
    }
}

fn main() -> Nil { }
"#,
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    // The user's check output holds only its own declarations; std's
    // identities join them in preparation.
    let prepared = prepare_frontend_for_lowering(db, typed, source).expect("prepared");
    assert_eq!(
        typed.declarations(db).well_known_types,
        ast_typeck::WellKnownTypes::empty()
    );
    let well_known = prepared.declarations(db).well_known_types;
    let qualified =
        |id: Option<tribute_front::ast::FuncDefId<'_>>| id.map(|id| id.qualified(db).to_string());
    assert_eq!(
        qualified(well_known.string_equality).as_deref(),
        Some("std::String::==")
    );
    assert_eq!(
        qualified(well_known.bytes_equality).as_deref(),
        Some("std::Bytes::==")
    );
}

/// Every type reachable from `ty`, through type parameters and
/// type-bearing attributes, satisfies `ok`.
fn type_tree_all(
    ctx: &IrContext,
    ty: trunk_ir::TypeRef,
    ok: &impl Fn(&IrContext, trunk_ir::TypeRef) -> bool,
) -> bool {
    let mut pending = vec![ty];
    let mut seen = HashSet::default();
    while let Some(ty) = pending.pop() {
        if !seen.insert(ty) {
            continue;
        }
        if !ok(ctx, ty) {
            return false;
        }
        let data = ctx.get_type(ty);
        pending.extend(data.params.iter().copied());
        for value in data.attrs.values() {
            value.visit_types(&mut |nested| pending.push(nested));
        }
    }
    true
}

/// The places where a type failing `ok` survives Wasm lowering of `text`:
/// type aliases and operations, including nested type parameters,
/// attributes, results, and block arguments.
fn types_surviving_wasm_lowering(
    db: &salsa::DatabaseImpl,
    path: &str,
    text: &str,
    ok: &impl Fn(&IrContext, trunk_ir::TypeRef) -> bool,
) -> std::collections::BTreeSet<String> {
    let source = source_from_str(path, text);
    let (mut ctx, module) = run_shared_pipeline(db, source)
        .expect("shared pipeline must succeed")
        .expect("fixture must lower");
    run_target_to_boundary_exit(
        &mut ctx,
        module,
        tribute_passes::abi_boundary::TargetKind::Wasm,
    )
    .expect("Wasm boundary");
    run_pass(&mut ctx, module, tribute_passes::wasm::lower::LowerToWasm).expect("Wasm lowering");

    let mut sites = std::collections::BTreeSet::new();
    for (name, ty) in ctx.type_aliases().iter().cloned() {
        if !type_tree_all(&ctx, ty, ok) {
            sites.insert(format!("alias !{name}"));
        }
    }
    let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
        let data = ctx.op(op);
        let site = format!("{}", data.name.display(&ctx));
        let mut types: Vec<_> = ctx.op_result_types(op).to_vec();
        for value in data.attributes.values() {
            value.visit_types(&mut |ty| types.push(ty));
        }
        for region in ctx.op_regions(op) {
            for &block in &ctx.region(region).blocks {
                for argument in ctx.block(block).args.iter() {
                    types.push(argument.ty);
                    for value in argument.attrs.values() {
                        value.visit_types(&mut |ty| types.push(ty));
                    }
                }
            }
        }
        if types.into_iter().any(|ty| !type_tree_all(&ctx, ty, ok)) {
            sites.insert(site);
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    sites
}

#[salsa_test]
fn wasm_lowering_leaves_no_core_bytes(db: &salsa::DatabaseImpl) {
    let not_core_bytes = |ctx: &IrContext, ty| {
        let data = ctx.get_type(ty);
        !(data.name.is(ctx, "core.bytes"))
    };
    for (path, text) in [
        (
            "wasm_dynamic_output.trb",
            include_str!("../../lang-examples/wasm_dynamic_output.trb"),
        ),
        (
            "case_bytes.trb",
            r#"fn pick(x: Bytes, y: Bytes, f: Bool) -> Bytes {
    case f {
        True -> x
        False -> y
    }
}

fn main() ->{std::io::Io} Nil {
    std::io::print_line(String::from_bytes(pick(b"ab", b"cd", True)))
}
"#,
        ),
    ] {
        let sites = types_surviving_wasm_lowering(db, path, text, &not_core_bytes);
        assert!(
            sites.is_empty(),
            "{path}: core.bytes survives Wasm lowering at {sites:#?}"
        );
    }
}

#[salsa_test]
fn wasm_lowering_leaves_no_tribute_rt_types(db: &salsa::DatabaseImpl) {
    let not_tribute_rt = |ctx: &IrContext, ty| {
        ctx.get_type(ty).name.dialect(ctx) != tribute_ir::dialect::tribute_rt::DIALECT_NAME
    };
    for (path, text) in [
        (
            "native_effects.trb",
            include_str!("../../lang-examples/native_effects.trb"),
        ),
        ("lambda.trb", include_str!("../../lang-examples/lambda.trb")),
        ("float.trb", include_str!("../../lang-examples/float.trb")),
        (
            "record-patterns.trb",
            include_str!("../../lang-examples/record-patterns.trb"),
        ),
    ] {
        let sites = types_surviving_wasm_lowering(db, path, text, &not_tribute_rt);
        assert!(
            sites.is_empty(),
            "{path}: a tribute_rt type survives Wasm lowering at {sites:#?}"
        );
    }
}

#[salsa_test]
fn wasm_lowering_leaves_adt_types_only_as_builtin_layouts(db: &salsa::DatabaseImpl) {
    // The backend identifies a builtin layout by its `layout` attribute
    // and knows no other `adt` type.
    let builtin_layout_or_not_adt = |ctx: &IrContext, ty| {
        let data = ctx.get_type(ty);
        data.name.dialect(ctx) != tribute_ir::dialect::adt::DIALECT_NAME
            || ctx
                .type_attrs(ty)
                .get(trunk_ir::types::LAYOUT_ATTR)
                .is_some()
    };
    for (path, text) in [
        (
            "native_effects.trb",
            include_str!("../../lang-examples/native_effects.trb"),
        ),
        ("lambda.trb", include_str!("../../lang-examples/lambda.trb")),
        ("float.trb", include_str!("../../lang-examples/float.trb")),
        ("tuples.trb", include_str!("../../lang-examples/tuples.trb")),
        (
            "record-patterns.trb",
            include_str!("../../lang-examples/record-patterns.trb"),
        ),
    ] {
        let mut sites = types_surviving_wasm_lowering(db, path, text, &builtin_layout_or_not_adt);
        // Source layouts stay declared as aliases and module metadata,
        // which no operation or value refers to.
        sites.retain(|site| !site.starts_with("alias !") && site != "core.module");
        assert!(
            sites.is_empty(),
            "{path}: a source adt type survives Wasm lowering at {sites:#?}"
        );
    }
}

#[salsa_test]
fn wasm_lowering_carries_only_preserved_language_metadata(db: &salsa::DatabaseImpl) {
    use tribute_passes::abi_boundary::is_preserved_attribute;

    let source = source_from_str(
        "wasm_dynamic_output.trb",
        include_str!("../../lang-examples/wasm_dynamic_output.trb"),
    );
    let (mut ctx, module) = run_shared_pipeline(db, source)
        .expect("shared pipeline must succeed")
        .expect("fixture must lower");
    run_target_to_boundary_exit(
        &mut ctx,
        module,
        tribute_passes::abi_boundary::TargetKind::Wasm,
    )
    .expect("Wasm boundary");
    run_pass(&mut ctx, module, tribute_passes::wasm::lower::LowerToWasm).expect("Wasm lowering");

    // Wasm lowering copies operation attributes without interpreting
    // them, so what it carries is exactly what the boundary let through.
    let mut unexpected = std::collections::BTreeSet::new();
    let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
        for name in ctx.op(op).attributes.keys() {
            let name = ctx.str(name);
            if name.starts_with("tribute.") && !is_preserved_attribute(name) {
                unexpected.insert(format!("{} {name}", ctx.op(op).name.display(&ctx)));
            }
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    assert!(unexpected.is_empty(), "{unexpected:#?}");
}

#[salsa_test]
fn well_known_string_metadata_uses_prelude_declaration_identity(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "lookalike.trb",
        r#"
enum String {
    Leaf(Bytes),
    Branch(String, String, Nat),
}

fn main() -> String { "hello" }
"#,
    );
    let typed = parse_and_lower_ast(db, source).expect("frontend output");
    let canonical = prepare_frontend_for_lowering(db, typed, source)
        .expect("prepared")
        .declarations(db)
        .well_known_types
        .string
        .expect("prelude String identity");
    let FrontendCompilation {
        context: ctx,
        module,
        ..
    } = merge_and_lower_to_ir(db, &typed, source);
    let string_ty = tribute_ir::metadata::WellKnownTypes::from_module(&ctx, module.op())
        .string
        .expect("String IR metadata");

    let Some(trunk_ir::Attribute::Location(definition)) =
        ctx.type_attrs(string_ty).get("tribute.definition")
    else {
        panic!("String IR type records its definition location");
    };
    assert_eq!(
        ctx.paths().get(definition.path),
        crate::prelude::PRELUDE_URI
    );
    assert_eq!(
        canonical.definition,
        tribute_front::ast::TypeDefId::source(db, trunk_ir::SymbolPath::new(["std", "String"]))
    );

    let user_lookalike = ctx.types().iter().find_map(|(ty, data)| {
        (ty != string_ty
            && data.name.is(&ctx, "adt.enum")
            && data.attrs.view(&ctx).get_str("name") == Some("String"))
        .then_some(ty)
    });
    assert!(
        user_lookalike.is_some(),
        "compatible user String must remain distinct from prelude String"
    );
}

#[salsa_test]
fn test_ast_pipeline_let_binding(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
            fn main() -> Nil {
                let x = 10
                let y = 20
                let _ = x + y
            }
            "#,
    );

    let result = compile_frontend(db, source);
    assert!(
        result.is_some(),
        "{:?}",
        parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
    );
    let (ctx, m) = result.unwrap();
    assert_eq!(m.name(&ctx), Some("test"));
}

#[salsa_test]
fn test_ast_pipeline_struct(db: &salsa::DatabaseImpl) {
    let source = source_from_str(
        "test.trb",
        r#"
            struct Point {
                x: Int,
                y: Int,
            }

            fn main() -> Nil { }
            "#,
    );

    let result = compile_frontend(db, source);
    assert!(result.is_some());
    let (ctx, m) = result.unwrap();
    assert_eq!(m.name(&ctx), Some("test"));
}

// =========================================================================
// main function return type validation
// =========================================================================

#[salsa_test]
fn test_main_returns_nil_ok(db: &salsa::DatabaseImpl) {
    let source = source_from_str("test.trb", "fn main() -> Nil { }");

    let result = compile_with_diagnostics(db, source);
    let has_main_error = result
        .diagnostics
        .iter()
        .any(|d| d.inner.message.contains("must return Nil"));
    assert!(
        !has_main_error,
        "main() returning Nil should not produce an error, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_main_returns_int_error(db: &salsa::DatabaseImpl) {
    let source = source_from_str("test.trb", "fn main() -> Int { 42 }");

    let result = compile_with_diagnostics(db, source);
    let has_main_error = result.diagnostics.iter().any(|d| {
        d.inner.message.contains("must return Nil")
            && d.inner.severity == DiagnosticSeverity::Error
            && d.phase == CompilationPhase::TypeChecking
    });
    assert!(
        has_main_error,
        "main() returning Int should produce 'must return Nil' error, got: {:?}",
        result.diagnostics
    );
}

#[salsa_test]
fn test_non_main_returns_int_ok(db: &salsa::DatabaseImpl) {
    let source = source_from_str("test.trb", "fn foo() -> Int { 42 }");

    let result = compile_with_diagnostics(db, source);
    let has_main_error = result
        .diagnostics
        .iter()
        .any(|d| d.inner.message.contains("must return Nil"));
    assert!(
        !has_main_error,
        "non-main function returning Int should not produce main-specific error, got: {:?}",
        result.diagnostics
    );
}

/// The short names every module of a user package sees, the public
/// items of `std::prelude::v0`, with the namespace a path through each
/// continues in and the package path it names.
#[salsa_test]
fn library_names_are_the_prelude_short_names(db: &crate::TributeDatabaseImpl) {
    let dependencies = prelude_dependencies(db);
    let own = PackageEnv::new();
    let scope = ast_resolve::Scope::new(&own, &dependencies);
    let mut names: Vec<String> = scope
        .iter_library_names()
        .into_iter()
        .filter_map(|name| Some((name.clone(), scope.lookup_library(&name)?)))
        .filter(|(_, binding)| scope.restriction(binding).is_none())
        .map(|(name, binding)| {
            let kind = match binding {
                ast_resolve::Binding::Function { .. } => "fn",
                ast_resolve::Binding::Constructor { .. } => "ctor",
                ast_resolve::Binding::TypeDef { .. } => "type",
                ast_resolve::Binding::Module { .. } => "mod",
                ast_resolve::Binding::Import { .. } => "use",
                ast_resolve::Binding::AbilityOp { .. } => "op",
                ast_resolve::Binding::Ability { .. } => "ability",
            };
            format!("{kind} {name}")
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "mod Bytes",
            "mod Float",
            "mod Int",
            "mod Nat",
            "type List",
            "type Option",
            "type Result",
            "type String",
        ]
    );

    let paths: Vec<String> = scope
        .iter_library_names()
        .into_iter()
        .map(|name| {
            let path = scope
                .library_path(&name)
                .expect("a prelude name has a path");
            let path = path.iter().format("::");
            match scope.library_namespace(&name) {
                Some(namespace) => format!(
                    "{name} -> {path} ({})",
                    namespace.identity().iter().format("::")
                ),
                None => format!("{name} -> {path}"),
            }
        })
        .collect();
    assert_eq!(
        paths,
        [
            "Bytes -> std::Bytes (std::Bytes)",
            "Float -> std::Float (std::Float)",
            "Int -> std::Int (std::Int)",
            "List -> std::collections::List (std::collections::List)",
            "Nat -> std::Nat (std::Nat)",
            "Option -> std::Option (std::Option)",
            "Result -> std::Result (std::Result)",
            "String -> std::String (std::String)",
        ]
    );
}
