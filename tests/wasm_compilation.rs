//! End-to-end WebAssembly compilation tests.
//!
//! These tests document the expected behavior of the full compilation pipeline.
//!
//! ## Current Status
//!
//! The wasm backend currently handles lowering of these dialects:
//! - `arith.*` → `wasm.*` (arithmetic operations)
//! - `scf.*` → `wasm.*` (structured control flow)
//! - `func.*` → `wasm.*` (function definitions and calls)
//! - `adt.*` → `wasm.*` (struct/variant operations)
//! - `intrinsic.*` → `wasm.*` (print_line, etc.)
//!
//! However, the full source→wasm pipeline requires additional passes:
//! - `src.var` → variable references (needs closure/local variable lowering)
//! - `src.block` → block expressions
//! - `case.case` → pattern matching
//! - Type resolution from `src.type` to concrete wasm types
//!
//! These tests are marked `#[ignore]` until those passes are implemented.

use salsa_test_macros::salsa_test;
use tree_sitter::Parser;
use tribute::{SourceCst, stage_lower_to_wasm};

/// Helper to create a source file from code
fn source_from_code(db: &dyn salsa::Database, name: &str, code: &str) -> SourceCst {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parser.parse(code, None).expect("Failed to parse");
    SourceCst::from_path(db, name, code.into(), Some(tree))
}

// =============================================================================
// End-to-end tests (require full pipeline completion)
// =============================================================================

#[salsa_test]
#[ignore = "requires src.type resolution to core types"]
fn test_compile_simple_literal(db: &salsa::DatabaseImpl) {
    let source = source_from_code(db, "literal.trb", "fn main() { 42 }");
    let binary = stage_lower_to_wasm(db, source);
    assert!(binary.is_some(), "Should compile literal return");

    let bytes = binary.unwrap().bytes(db);
    assert_eq!(&bytes[0..4], b"\x00asm", "Should have wasm magic number");
}

#[salsa_test]
#[ignore = "requires src.type resolution to core types"]
fn test_compile_arithmetic_expr(db: &salsa::DatabaseImpl) {
    let source = source_from_code(db, "arith.trb", "fn main() { 1 + 2 * 3 }");
    let binary = stage_lower_to_wasm(db, source);
    assert!(binary.is_some(), "Should compile arithmetic expression");
}

#[salsa_test]
#[ignore = "requires src.var lowering"]
fn test_compile_function_with_params(db: &salsa::DatabaseImpl) {
    let code = r#"
fn add(a, b) { a + b }
fn main() { add(1, 2) }
"#;
    let source = source_from_code(db, "params.trb", code);
    let binary = stage_lower_to_wasm(db, source);
    assert!(binary.is_some(), "Should compile function with params");
}

#[salsa_test]
#[ignore = "requires src.var and src.block lowering"]
fn test_compile_if_expression(db: &salsa::DatabaseImpl) {
    let code = r#"
fn max(a, b) {
    if a > b { a } else { b }
}
fn main() { max(3, 5) }
"#;
    let source = source_from_code(db, "if_expr.trb", code);
    let binary = stage_lower_to_wasm(db, source);
    assert!(binary.is_some(), "Should compile if expression");
}

#[salsa_test]
#[ignore = "requires case.case lowering"]
fn test_compile_case_expression(db: &salsa::DatabaseImpl) {
    let code = r#"
fn classify(n) {
    case n {
        0 { "zero" }
        1 { "one" }
        _ { "other" }
    }
}
fn main() { classify(1) }
"#;
    let source = source_from_code(db, "case_expr.trb", code);
    let binary = stage_lower_to_wasm(db, source);
    assert!(binary.is_some(), "Should compile case expression");
}

#[salsa_test]
#[ignore = "requires src.type resolution to core types"]
fn test_compile_print_line(db: &salsa::DatabaseImpl) {
    let code = r#"fn main() { print_line("Hello, World!") }"#;
    let source = source_from_code(db, "hello.trb", code);
    let binary = stage_lower_to_wasm(db, source);
    assert!(binary.is_some(), "Should compile print_line");
}

#[salsa_test]
#[ignore = "requires src.var lowering"]
fn test_compile_local_variables(db: &salsa::DatabaseImpl) {
    let code = r#"
fn test_ops() {
    let a = 10;
    let b = 3;
    a + b
}
fn main() { test_ops() }
"#;
    let source = source_from_code(db, "locals.trb", code);
    let binary = stage_lower_to_wasm(db, source);
    assert!(binary.is_some(), "Should compile local variables");
}
