//! End-to-end tests for Float type operations.
//!
//! These tests verify Float type functionality including:
//! - Arithmetic operations (+, -, *, /)
//! - Comparison operations (<, >, <=, >=, ==)
//! - Native compilation

use salsa_test_macros::salsa_test;
use tree_sitter::Parser;
use tribute::SourceCst;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::{CompilationConfig, compile_to_native_binary};

/// Helper to create a source file from code.
fn source_from_code(db: &TributeDatabaseImpl, name: &str, code: &str) -> SourceCst {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parser.parse(code, None).expect("Failed to parse");
    SourceCst::from_path(db, name, code.into(), Some(tree))
}

/// Test Float multiplication operation.
#[salsa_test]
fn test_float_multiply(db: &TributeDatabaseImpl) {
    // 3.0 * 2.0 = 6.0
    let source = source_from_code(
        db,
        "float_multiply.trb",
        r#"
fn main() {
    let _ = 3.0 * 2.0
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float subtraction operation.
#[salsa_test]
fn test_float_subtraction(db: &TributeDatabaseImpl) {
    // 5.5 - 2.5 = 3.0
    let source = source_from_code(
        db,
        "float_subtraction.trb",
        r#"
fn main() {
    let _ = 5.5 - 2.5
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float division operation.
#[salsa_test]
fn test_float_division(db: &TributeDatabaseImpl) {
    // 10.0 / 4.0 = 2.5
    let source = source_from_code(
        db,
        "float_division.trb",
        r#"
fn main() {
    let _ = 10.0 / 4.0
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float less-than comparison.
#[salsa_test]
fn test_float_comparison_less_than(db: &TributeDatabaseImpl) {
    // 1.5 < 2.5 == True
    let source = source_from_code(
        db,
        "float_less_than.trb",
        r#"
fn main() {
    let _ = 1.5 < 2.5
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float greater-than comparison.
#[salsa_test]
fn test_float_comparison_greater_than(db: &TributeDatabaseImpl) {
    // 2.5 > 1.5 == True
    let source = source_from_code(
        db,
        "float_greater_than.trb",
        r#"
fn main() {
    let _ = 2.5 > 1.5
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float equality comparison.
#[salsa_test]
fn test_float_comparison_equality(db: &TributeDatabaseImpl) {
    // 1.5 == 1.5 == True
    let source = source_from_code(
        db,
        "float_equality.trb",
        r#"
fn main() {
    let _ = 1.5 == 1.5
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float less-than-or-equal comparison.
#[salsa_test]
fn test_float_comparison_lte(db: &TributeDatabaseImpl) {
    let source = source_from_code(
        db,
        "float_lte.trb",
        r#"
fn main() {
    let _ = 1.5 <= 2.5
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float greater-than-or-equal comparison.
#[salsa_test]
fn test_float_comparison_gte(db: &TributeDatabaseImpl) {
    let source = source_from_code(
        db,
        "float_gte.trb",
        r#"
fn main() {
    let _ = 2.5 >= 1.5
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float combined arithmetic operations.
#[salsa_test]
fn test_float_combined_arithmetic(db: &TributeDatabaseImpl) {
    // (3.0 * 2.0) + (10.0 / 2.0) - 1.0 = 6.0 + 5.0 - 1.0 = 10.0
    let source = source_from_code(
        db,
        "float_combined.trb",
        r#"
fn main() {
    let a = 3.0 * 2.0
    let b = 10.0 / 2.0
    let _ = a + b - 1.0
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float literal compilation.
#[salsa_test]
fn test_float_literal_compiles(db: &TributeDatabaseImpl) {
    let source = source_from_code(
        db,
        "float_literal.trb",
        r#"
fn main() {
    let _ = 3.14159
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test negative Float literal.
#[salsa_test]
fn test_negative_float_literal(db: &TributeDatabaseImpl) {
    let source = source_from_code(
        db,
        "negative_float.trb",
        r#"
fn main() {
    let _ = -3.14
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}

/// Test Float in function parameter and return.
#[salsa_test]
fn test_float_function_param_and_return(db: &TributeDatabaseImpl) {
    let source = source_from_code(
        db,
        "float_function.trb",
        r#"
fn double(x: Float) ->{} Float {
    x * 2.0
}

fn main() {
    let _ = double(3.5)
}
"#,
    );
    let config = CompilationConfig::new(db, false);
    compile_to_native_binary(db, source, config).expect("Native compilation failed");
}
