//! End-to-end WebAssembly compilation tests.
//!
//! These tests validate the full source code → WASM compilation pipeline.
//!
//! ## Current Status
//!
//! Most basic compilation scenarios work:
//! - Simple literals and arithmetic expressions
//! - Functions with parameters
//! - Local variables (let bindings)
//! - Intrinsics like print_line
//!
//! ## Remaining Work
//!
//! The following features need additional lowering passes:
//! - `tribute.block` → block expressions in case branches
//! - String literals in pattern matching (case expressions)

mod common;

use std::io::Write as _;
use std::process::Command;

use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_to_wasm_binary;
use tribute_front::SourceCst;

fn expect_wasm_compilation_success(
    db: &dyn salsa::Database,
    source: SourceCst,
    message: &str,
) -> Vec<u8> {
    compile_to_wasm_binary(db, source)
        .unwrap_or_else(|diagnostics| panic!("{message}: {diagnostics:?}"))
}

#[salsa_test]
fn test_compile_failure_returns_diagnostics(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(db, "invalid.trb", "fn main() -> Int { true }");
    let Err(diagnostics) = compile_to_wasm_binary(db, source) else {
        panic!("invalid source should fail WebAssembly compilation");
    };
    assert!(!diagnostics.is_empty());
}

// =============================================================================
// Passing end-to-end tests
// =============================================================================

#[salsa_test]
fn test_compile_builtin_io_entrypoint(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "builtin_io_main.trb",
        r#"use std::io::Io

fn touch_world() ->{Io} Nil { Nil }

fn main() ->{Io} Nil {
    touch_world()
}
"#,
    );
    let bytes = expect_wasm_compilation_success(db, source, "Should compile builtin Io main");
    assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
}

#[salsa_test]
fn test_compile_simple_literal(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "literal.trb",
        r#"
extern "intrinsic" fn __print_line(message: String) -> Nil
fn main() {
    case 42 {
        42 -> __print_line("ok")
        _ -> __print_line("unexpected")
    }
}
"#,
    );
    let bytes = expect_wasm_compilation_success(db, source, "Should compile literal return");
    assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
}

#[salsa_test]
fn test_compile_arithmetic_expr(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "arith.trb",
        r#"
extern "intrinsic" fn __print_line(message: String) -> Nil
fn main() {
    case 1 + 2 * 3 {
        7 -> __print_line("ok")
        _ -> __print_line("unexpected")
    }
}
"#,
    );
    expect_wasm_compilation_success(db, source, "Should compile arithmetic expression");
}

#[salsa_test]
fn test_compile_function_with_params(db: &salsa::DatabaseImpl) {
    let code = r#"
extern "intrinsic" fn __print_line(message: String) -> Nil
fn add(a: Nat, b: Nat) -> Nat { a + b }
fn main() {
    case add(1, 2) {
        3 -> __print_line("ok")
        _ -> __print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "params.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile function with params");
}

#[salsa_test]
fn test_compile_print_line(db: &salsa::DatabaseImpl) {
    let code = r#"
fn main() ->{std::io::Io} Nil {
    let left = b"Hello, "
    let right = b"World!"
    std::io::print_line(String::from_bytes(left <> right))
}
"#;
    let source = SourceCst::from_source_str(db, "hello.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile dynamic print_line");
}

#[test]
fn test_execute_dynamic_bytes_write_boundary() {
    let mut ctx = trunk_ir::IrContext::new();
    let left = "x".repeat(35_000);
    let right = "y".repeat(35_000);
    let ir = format!(
        r#"core.module @test {{
  func.func @main() -> core.nil {{
    %before = adt.string_const {{value = "before|"}} : core.string
    wasm.call %before {{callee = @__print_line}}
    %left = adt.bytes_const {{value = b"{left}"}} : core.bytes
    %right = adt.bytes_const {{value = b"{right}"}} : core.bytes
    %joined = func.call %left, %right {{callee = @__bytes_concat}} : core.bytes
    %newline = arith.const {{value = 1}} : core.i1
    %result = tribute_io.write %joined, %newline : core.nil
    %after = adt.string_const {{value = "|after"}} : core.string
    wasm.call %after {{callee = @__print_line}}
    func.return
  }}
}}"#
    );
    let module = trunk_ir::parser::parse_test_module(&mut ctx, &ir);
    tribute_passes::wasm::lower::lower_to_wasm(&mut ctx, module)
        .expect("lower dynamic output to Wasm");
    tribute_passes::wasm::lower::finalize_wasm_gc_types(&mut ctx, module)
        .expect("finalize semantic WasmGC types");
    let binary = trunk_ir_wasm_backend::emit_module_to_wasm(&mut ctx, module)
        .expect("emit dynamic output Wasm");
    let mut wasm = tempfile::NamedTempFile::new().expect("temporary Wasm file");
    wasm.write_all(&binary.bytes).expect("write Wasm module");
    let output = Command::new("wasmtime")
        .arg("-Wgc=y,function-references=y")
        .arg(wasm.path())
        .output()
        .expect("run Wasm module with wasmtime");
    assert!(
        output.status.success(),
        "wasmtime failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut expected = b"before|".to_vec();
    expected.extend_from_slice(left.as_bytes());
    expected.extend_from_slice(right.as_bytes());
    expected.push(b'\n');
    expected.extend_from_slice(b"|after");
    assert_eq!(output.stdout, expected);
}

#[salsa_test]
fn test_compile_local_variables(db: &salsa::DatabaseImpl) {
    let code = r#"
fn test_ops() -> Nat {
    let a = 10
    let b = 3
    a + b
}
extern "intrinsic" fn __print_line(message: String) -> Nil
fn main() {
    case test_ops() {
        13 -> __print_line("ok")
        _ -> __print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "locals.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile local variables");
}

// =============================================================================
// Tests requiring additional lowering passes
// =============================================================================

// Note: Tribute does not have if-else expressions; control flow uses
// pattern matching (case) and algebraic effects.

#[salsa_test]
fn test_compile_case_expression(db: &salsa::DatabaseImpl) {
    let code = r#"
fn classify(n: Nat) -> String {
    case n {
        0 -> "zero"
        1 -> "one"
        _ -> "other"
    }
}
extern "intrinsic" fn __print_line(message: String) -> Nil
fn main() { __print_line(classify(1)) }
"#;
    let source = SourceCst::from_source_str(db, "case_expr.trb", code);
    expect_wasm_compilation_success(db, source, "Should compile case expression");
}

#[salsa_test]
fn test_compile_tail_dispatch_ability(db: &salsa::DatabaseImpl) {
    let code = r#"
ability Console {
    fn read() -> Int
    fn print(value: Int) -> Nil
}

extern "intrinsic" fn __print_line(message: String) -> Nil

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
    case run() {
        +41 -> __print_line("ok")
        _ -> __print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "tail_dispatch_ability.trb", code);
    expect_wasm_compilation_success(
        db,
        source,
        "Should compile tail-dispatch ability through wasm effect ABI lowering",
    );
}

#[salsa_test]
fn test_compile_cps_dispatch_ability(db: &salsa::DatabaseImpl) {
    let code = r#"
ability State(s) {
    op get() -> s
    op set(value: s) -> Nil
}

extern "intrinsic" fn __print_line(message: String) -> Nil

fn bump() ->{State(Int)} Int {
    let n = State::get()
    State::set(n + +1)
    n
}

fn run_state() -> Int {
    handle bump() {
        do result { result }
        op State::get() { resume +41 }
        op State::set(value) { resume Nil }
    }
}

fn main() {
    case run_state() {
        +41 -> __print_line("ok")
        _ -> __print_line("unexpected")
    }
}
"#;
    let source = SourceCst::from_source_str(db, "cps_dispatch_ability.trb", code);
    expect_wasm_compilation_success(
        db,
        source,
        "Should compile CPS ability dispatch through wasm effect ABI lowering",
    );
}
