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

use salsa_test_macros::salsa_test;
use tribute::pipeline::compile_to_wasm_binary;
use tribute_front::SourceCst;
use tribute_passes::diagnostic::Diagnostic;

// =============================================================================
// Passing end-to-end tests
// =============================================================================

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
    let binary = compile_to_wasm_binary(db, source);
    if binary.is_none() {
        for diagnostic in compile_to_wasm_binary::accumulated::<Diagnostic>(db, source) {
            eprintln!("Diagnostic: {diagnostic:?}");
        }
    }
    assert!(binary.is_some(), "Should compile literal return");

    let bytes = binary.unwrap();
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
    let binary = compile_to_wasm_binary(db, source);
    if binary.is_none() {
        for diagnostic in compile_to_wasm_binary::accumulated::<Diagnostic>(db, source) {
            eprintln!("Diagnostic: {diagnostic:?}");
        }
    }
    assert!(binary.is_some(), "Should compile arithmetic expression");
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
    let binary = compile_to_wasm_binary(db, source);
    if binary.is_none() {
        for diagnostic in compile_to_wasm_binary::accumulated::<Diagnostic>(db, source) {
            eprintln!("Diagnostic: {diagnostic:?}");
        }
    }
    assert!(binary.is_some(), "Should compile function with params");
}

// Note: String literals work as intrinsic arguments (e.g., __print_line)
// but require additional lowering for case branch return values.
// This test calls the intrinsic directly (not through prelude's print_line,
// which now depends on native-only extern "C" functions).
#[salsa_test]
fn test_compile_print_line(db: &salsa::DatabaseImpl) {
    let code = r#"
extern "intrinsic" fn __print_line(message: String) -> Nil
fn my_print(message: String) -> Nil { __print_line(message) }
fn main() { my_print("Hello, World!") }
"#;
    let source = SourceCst::from_source_str(db, "hello.trb", code);
    let binary = compile_to_wasm_binary(db, source);

    if binary.is_none() {
        // Collect and print diagnostics
        let diagnostics = compile_to_wasm_binary::accumulated::<Diagnostic>(db, source);
        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }
    }

    assert!(binary.is_some(), "Should compile print_line");
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
    let binary = compile_to_wasm_binary(db, source);
    if binary.is_none() {
        for diagnostic in compile_to_wasm_binary::accumulated::<Diagnostic>(db, source) {
            eprintln!("Diagnostic: {diagnostic:?}");
        }
    }
    assert!(binary.is_some(), "Should compile local variables");
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
    let binary = compile_to_wasm_binary(db, source);
    if binary.is_none() {
        for diagnostic in compile_to_wasm_binary::accumulated::<Diagnostic>(db, source) {
            eprintln!("Diagnostic: {diagnostic:?}");
        }
    }
    assert!(binary.is_some(), "Should compile case expression");
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
    let binary = compile_to_wasm_binary(db, source);

    if binary.is_none() {
        let diagnostics = compile_to_wasm_binary::accumulated::<Diagnostic>(db, source);
        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }
    }

    assert!(
        binary.is_some(),
        "Should compile tail-dispatch ability through wasm effect ABI lowering"
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
    let binary = compile_to_wasm_binary(db, source);

    if binary.is_none() {
        let diagnostics = compile_to_wasm_binary::accumulated::<Diagnostic>(db, source);
        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }
    }

    assert!(
        binary.is_some(),
        "Should compile CPS ability dispatch through wasm effect ABI lowering"
    );
}
