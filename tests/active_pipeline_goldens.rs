//! Golden tests for the active ability lowering pipeline.
//!
//! These snapshots intentionally target textual IR at named pipeline stages:
//! shared middle-end IR and native-target IR.

use insta::assert_snapshot;
use salsa_test_macros::salsa_test;
use tribute::Diagnostic;
use tribute::pipeline::{compile_with_diagnostics, dump_ir};
use tribute_front::SourceCst;
use trunk_ir::printer::print_module;

fn assert_no_diagnostics(stage: &str, diagnostics: &[Diagnostic]) {
    assert!(
        diagnostics.is_empty(),
        "{stage} emitted diagnostics:\n{}",
        diagnostics
            .iter()
            .map(|diagnostic| format!("  - [{}] {}", diagnostic.phase, diagnostic.inner.message))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn is_active_pipeline_function(header: &str) -> bool {
    let selected_names = [
        "@__tribute_evidence_",
        "@__tribute_next_tag",
        "@main",
        "@use_console",
        "@run",
        "@run_state",
        "@run_state_with_console",
        "@run_all",
        "@bump",
        "@step",
        "\"run::",
        "\"run_state::",
        "\"run_state_with_console::",
        "\"run_all::",
        "\"bump::",
        "\"step::",
        "\"direct_fn::__lambda",
        "\"direct_fn_native::__lambda",
        "\"resumptive_op::__lambda",
        "\"resumptive_op_native::__lambda",
        "\"mixed_nested_native::__lambda",
    ];

    selected_names.iter().any(|name| header.contains(name))
}

fn brace_delta(line: &str) -> isize {
    line.chars().fold(0, |delta, ch| match ch {
        '{' => delta + 1,
        '}' => delta - 1,
        _ => delta,
    })
}

fn filter_ir_for_active_pipeline(ir_text: &str) -> String {
    let mut output = Vec::new();
    let mut lines = ir_text.lines().peekable();

    while let Some(line) = lines.next() {
        if line.starts_with("core.module ") || line.trim_start().starts_with('!') {
            output.push(line.to_owned());
            continue;
        }

        if !line.starts_with("  func.func ") {
            continue;
        }

        let include_function = is_active_pipeline_function(line);
        let mut function_lines = vec![line.to_owned()];
        let mut depth = brace_delta(line);

        while depth > 0 {
            let body_line = lines
                .next()
                .expect("function body should close before the module ends");
            depth += brace_delta(body_line);
            function_lines.push(body_line.to_owned());
        }

        if include_function {
            if !output.last().is_some_and(|line| line.is_empty()) {
                output.push(String::new());
            }
            output.extend(function_lines);
        }
    }

    output.push("}".to_owned());
    output.join("\n")
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

fn snapshot_native_pipeline_ir(db: &dyn salsa::Database, name: &str, code: &str) -> String {
    let source = SourceCst::from_source_str(db, name, code);
    let ir_text = dump_ir(db, source, true).expect("native pipeline dump should succeed");
    let diagnostics: Vec<Diagnostic> = dump_ir::accumulated::<Diagnostic>(db, source, true)
        .into_iter()
        .cloned()
        .collect();
    assert_no_diagnostics("native pipeline", &diagnostics);
    filter_ir_for_active_pipeline(&ir_text)
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
fn native_pipeline_direct_fn_ability_call(db: &salsa::DatabaseImpl) {
    let ir_text = snapshot_native_pipeline_ir(db, "direct_fn_native.trb", DIRECT_FN_SOURCE);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn native_pipeline_resumptive_op_continuation(db: &salsa::DatabaseImpl) {
    let ir_text = snapshot_native_pipeline_ir(db, "resumptive_op_native.trb", RESUMPTIVE_OP_SOURCE);
    assert_snapshot!(ir_text);
}

#[salsa_test]
fn native_pipeline_mixed_nested_handler_boundary(db: &salsa::DatabaseImpl) {
    let ir_text = snapshot_native_pipeline_ir(db, "mixed_nested_native.trb", MIXED_NESTED_SOURCE);
    assert_snapshot!(ir_text);
}
