//! Golden tests for the active ability lowering pipeline.
//!
//! These snapshots intentionally target textual IR at named pipeline stages:
//! shared middle-end IR and native-target IR.

use std::ops::ControlFlow;

use insta::assert_snapshot;
use itertools::Itertools;
use salsa_test_macros::salsa_test;
use tribute::Diagnostic;
use tribute::pipeline::{compile_with_diagnostics, dump_ir};
use tribute_front::SourceCst;
use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::{print_module, print_op};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{IrContext, Module, Symbol};

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

fn is_active_pipeline_function(ctx: &IrContext, op: trunk_ir::OpRef) -> bool {
    let data = ctx.op(op);
    if data.dialect != Symbol::new("func") || data.name != Symbol::new("func") {
        return false;
    }
    let Some(symbol) = data.attributes.get_symbol("sym_name") else {
        return false;
    };
    let name = symbol.to_string();
    let selected_names = [
        "__tribute_evidence_",
        "__tribute_next_tag",
        "main",
        "use_console",
        "run",
        "run_state",
        "run_state_with_console",
        "run_all",
        "bump",
        "step",
        "run::",
        "run_state::",
        "run_state_with_console::",
        "run_all::",
        "bump::",
        "step::",
        "direct_fn::__lambda",
        "direct_fn_native::__lambda",
        "resumptive_op::__lambda",
        "resumptive_op_native::__lambda",
        "mixed_nested::__lambda",
        "mixed_nested_native::__lambda",
    ];

    selected_names.iter().any(|prefix| name.starts_with(prefix))
}

fn parse_ir(ir_text: &str) -> (IrContext, Module) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir_text);
    (ctx, module)
}

fn filter_ir_for_active_pipeline(ir_text: &str) -> String {
    let (ctx, module) = parse_ir(ir_text);
    let selected: Vec<_> = module
        .ops(&ctx)
        .into_iter()
        .filter(|&op| is_active_pipeline_function(&ctx, op))
        .collect();
    let full = print_module(&ctx, module.op());
    let preamble = full
        .lines()
        .take_while(|line| !line.starts_with("  func.func "))
        .collect::<Vec<_>>()
        .join("\n");
    let functions = selected
        .into_iter()
        .map(|op| {
            print_op(&ctx, op)
                .lines()
                .map(|line| format!("  {line}"))
                .join("\n")
        })
        .join("\n");
    let separator = if preamble.ends_with('\n') { "" } else { "\n" };
    format!("{preamble}{separator}{functions}\n}}")
}

fn snapshot_shared_pipeline_ir(db: &dyn salsa::Database, name: &str, code: &str) -> String {
    let source = SourceCst::from_source_str(db, name, code);
    let result = compile_with_diagnostics(db, source);
    assert_no_diagnostics("shared pipeline", &result.diagnostics);

    let (ctx, module) = result
        .module
        .expect("shared pipeline should produce a module when diagnostics are empty");
    filter_ir_for_active_pipeline(&print_module(&ctx, module.op()))
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

fn assert_no_production_control_artifacts(ir_text: &str) {
    for forbidden in [
        "__tribute_cps_control",
        "@Normal",
        "@Escape",
        "tribute_control.",
        "ability.legacy_",
    ] {
        assert!(
            !ir_text.contains(forbidden),
            "production native fixture contains forbidden `{forbidden}` artifact:\n{ir_text}"
        );
    }

    let (ctx, module) = parse_ir(ir_text);
    assert_no_dead_int_unbox_in_module(&ctx, module);

    for op in module.ops(&ctx) {
        let data = ctx.op(op);
        if data.dialect != Symbol::new("func")
            || data.name != Symbol::new("func")
            || data.attributes.get_i128("tribute.calling_convention") != Some(2)
        {
            continue;
        }
        assert!(
            function_returns_core_nil(&ctx, op),
            "CPS function must have a physically empty result:\n{}",
            print_module(&ctx, module.op())
        );
    }
}

/// Source operation inputs may cross the erased payload boundary and therefore
/// legitimately unbox.  The retired carrier path instead left an unconsumed
/// `unbox_int` after its placeholder re-erasure was resolved.
fn assert_no_dead_int_unbox_in_module(ctx: &IrContext, module: Module) {
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        let data = ctx.op(op);
        if data.dialect == Symbol::new("tribute_rt") && data.name == Symbol::new("unbox_int") {
            let results = ctx.op_results(op);
            assert_eq!(results.len(), 1, "unbox_int must have one SSA result");
            assert!(
                ctx.has_uses(results[0]),
                "production native fixture contains a dead placeholder `unbox_int`:\n{}",
                print_module(ctx, module.op())
            );
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
}

fn is_core_nil(ctx: &IrContext, ty: trunk_ir::TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("core") && data.name == Symbol::new("nil")
}

fn function_returns_core_nil(ctx: &IrContext, op: trunk_ir::OpRef) -> bool {
    let Some(function_type) = ctx.op(op).attributes.get_type("type") else {
        return false;
    };
    let data = ctx.types.get(function_type);
    data.dialect == Symbol::new("core")
        && data.name == Symbol::new("func")
        && data
            .params
            .first()
            .is_some_and(|&result| is_core_nil(ctx, result))
}

fn exact_marker(ctx: &IrContext, op: trunk_ir::OpRef, marker: &str) -> bool {
    ctx.op(op).attributes.get_bool(marker) == Some(true)
}

fn unique_marked_function(ctx: &IrContext, module: Module, marker: &str) -> trunk_ir::OpRef {
    let matches: Vec<_> = module
        .ops(ctx)
        .into_iter()
        .filter(|&op| {
            let data = ctx.op(op);
            data.dialect == Symbol::new("func")
                && data.name == Symbol::new("func")
                && exact_marker(ctx, op, marker)
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one `{marker}` in full native IR:\n{}",
        print_module(ctx, module.op())
    );
    matches[0]
}

fn function_symbol(ctx: &IrContext, op: trunk_ir::OpRef) -> Symbol {
    ctx.op(op)
        .attributes
        .get_symbol("sym_name")
        .expect("function must have a symbol")
}

fn assert_root_cps_topology(ir_text: &str) {
    let (ctx, module) = parse_ir(ir_text);
    let worker = unique_marked_function(&ctx, module, "tribute.root_cps_worker");
    let wrapper = unique_marked_function(&ctx, module, "tribute.root_wrapper");
    let done = unique_marked_function(&ctx, module, "tribute.root_done_k");
    let worker_symbol = function_symbol(&ctx, worker);
    let wrapper_symbol = function_symbol(&ctx, wrapper);
    let done_symbol = function_symbol(&ctx, done);

    assert_eq!(
        ctx.op(worker)
            .attributes
            .get_i128("tribute.calling_convention"),
        Some(2)
    );
    assert!(function_returns_core_nil(&ctx, worker));
    assert!(matches!(
        ctx.op(wrapper)
            .attributes
            .get_i128("tribute.calling_convention"),
        Some(0 | 1)
    ));
    assert_eq!(
        ctx.op(done)
            .attributes
            .get_i128("tribute.calling_convention"),
        Some(2)
    );
    assert!(function_returns_core_nil(&ctx, done));
    assert_ne!(worker_symbol, wrapper_symbol);
    assert_ne!(worker_symbol, done_symbol);
    assert_ne!(wrapper_symbol, done_symbol);

    let mut root_calls = Vec::new();
    let mut done_constants = Vec::new();
    let _ = walk_op::<()>(&ctx, wrapper, &mut |op| {
        let data = ctx.op(op);
        if exact_marker(&ctx, op, "tribute.root_cps_call") {
            root_calls.push(op);
        }
        if data.dialect == Symbol::new("func")
            && data.name == Symbol::new("constant")
            && data.attributes.get_symbol("func_ref") == Some(done_symbol)
        {
            done_constants.push(op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    assert_eq!(
        root_calls.len(),
        1,
        "root wrapper must have one marked CPS call"
    );
    assert_eq!(
        done_constants.len(),
        1,
        "root wrapper must capture its marked done continuation"
    );

    let root_call = root_calls[0];
    let data = ctx.op(root_call);
    assert_eq!(data.dialect, Symbol::new("func"));
    assert_eq!(data.name, Symbol::new("call"));
    assert_eq!(data.attributes.get_symbol("callee"), Some(worker_symbol));
    assert_eq!(
        data.attributes.get_i128("tribute.calling_convention"),
        Some(2)
    );
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

#[test]
#[should_panic(expected = "dead placeholder `unbox_int`")]
fn dead_int_unbox_cannot_borrow_a_use_from_a_later_function() {
    let (ctx, module) = parse_ir(
        r#"
core.module @test {
  func.func @first(%0: tribute_rt.anyref) -> core.nil {
    %6 = tribute_rt.unbox_int %0 : core.i32
    func.return
  }
  func.func @second(%0: core.i32) -> core.nil {
    %6 = arith.addi %0, %0 : core.i32
    %7 = arith.addi %6, %0 : core.i32
    func.return
  }
}
"#,
    );
    assert_no_dead_int_unbox_in_module(&ctx, module);
}

#[test]
fn state_get_unbox_used_by_its_immediate_tail_transfer_is_allowed() {
    let (ctx, module) = parse_ir(
        r#"
core.module @test {
  func.func @state_get_resume(%0: tribute_rt.anyref, %1: core.ptr, %2: core.ptr) -> core.nil {
    %6 = tribute_rt.unbox_int %0 : core.i32
    func.tail_call_indirect %1, %2, %6 { type = core.func(core.nil, core.ptr, core.i32) }
  }
}
"#,
    );
    assert_no_dead_int_unbox_in_module(&ctx, module);
}

#[salsa_test]
fn shared_pipeline_direct_fn_ability_call(db: &salsa::DatabaseImpl) {
    let ir_text = snapshot_shared_pipeline_ir(db, "direct_fn.trb", DIRECT_FN_SOURCE);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn shared_pipeline_resumptive_op_continuation(db: &salsa::DatabaseImpl) {
    let ir_text = snapshot_shared_pipeline_ir(db, "resumptive_op.trb", RESUMPTIVE_OP_SOURCE);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn shared_pipeline_mixed_nested_handler_boundary(db: &salsa::DatabaseImpl) {
    let ir_text = snapshot_shared_pipeline_ir(db, "mixed_nested.trb", MIXED_NESTED_SOURCE);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn shared_pipeline_float_comparison_predicates(db: &salsa::DatabaseImpl) {
    let ir_text = snapshot_shared_pipeline_ir(db, "float_comparisons.trb", FLOAT_COMPARISON_SOURCE);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn native_pipeline_direct_fn_ability_call(db: &salsa::DatabaseImpl) {
    let full_ir = native_pipeline_ir(db, "direct_fn_native.trb", DIRECT_FN_SOURCE);
    assert_no_production_control_artifacts(&full_ir);
    assert_root_cps_topology(&full_ir);
    assert_snapshot!(filter_ir_for_active_pipeline(&full_ir));
}

#[salsa_test]
fn native_pipeline_resumptive_op_continuation(db: &salsa::DatabaseImpl) {
    let full_ir = native_pipeline_ir(db, "resumptive_op_native.trb", RESUMPTIVE_OP_SOURCE);
    assert_no_production_control_artifacts(&full_ir);
    assert_root_cps_topology(&full_ir);
    assert_snapshot!(filter_ir_for_active_pipeline(&full_ir));
}

#[salsa_test]
fn native_pipeline_mixed_nested_handler_boundary(db: &salsa::DatabaseImpl) {
    let full_ir = native_pipeline_ir(db, "mixed_nested_native.trb", MIXED_NESTED_SOURCE);
    assert_no_production_control_artifacts(&full_ir);
    assert_root_cps_topology(&full_ir);
    assert_snapshot!(filter_ir_for_active_pipeline(&full_ir));
}
