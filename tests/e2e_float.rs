//! End-to-end tests for Float type operations.
//!
//! These tests verify Float type functionality including:
//! - Arithmetic operations (*, /, -)
//! - Comparison operations (<, >, <=, >=, ==)
//! - WASM compilation

mod common;

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::compile_to_wasm_binary;
use tribute_front::SourceCst;

/// Test Float multiplication operation.
#[test]
fn test_float_multiply() {
    use tribute::database::parse_with_thread_local;

    // 3.0 * 2.0 = 6.0
    let source_code = Rope::from_str(
        r#"
fn main() ->{} Float {
    3.0 * 2.0
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_multiply.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float subtraction operation.
#[test]
fn test_float_subtraction() {
    use tribute::database::parse_with_thread_local;

    // 5.5 - 2.5 = 3.0
    let source_code = Rope::from_str(
        r#"
fn main() ->{} Float {
    5.5 - 2.5
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "float_subtraction.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float division operation.
#[test]
fn test_float_division() {
    use tribute::database::parse_with_thread_local;

    // 10.0 / 4.0 = 2.5
    let source_code = Rope::from_str(
        r#"
fn main() ->{} Float {
    10.0 / 4.0
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_division.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float comparison operations (<, >, ==).
#[test]
fn test_float_comparison_less_than() {
    use tribute::database::parse_with_thread_local;

    // 1.5 < 2.5 == True
    let source_code = Rope::from_str(
        r#"
fn main() ->{} Bool {
    1.5 < 2.5
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "float_less_than.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float greater-than comparison.
#[test]
fn test_float_comparison_greater_than() {
    use tribute::database::parse_with_thread_local;

    // 2.5 > 1.5 == True
    let source_code = Rope::from_str(
        r#"
fn main() ->{} Bool {
    2.5 > 1.5
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file =
            SourceCst::from_path(db, "float_greater_than.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float equality comparison.
#[test]
fn test_float_comparison_equality() {
    use tribute::database::parse_with_thread_local;

    // 1.5 == 1.5 == True
    let source_code = Rope::from_str(
        r#"
fn main() ->{} Bool {
    1.5 == 1.5
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_equality.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float less-than-or-equal comparison.
#[test]
fn test_float_comparison_lte() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn main() ->{} Bool {
    1.5 <= 2.5
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_lte.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float greater-than-or-equal comparison.
#[test]
fn test_float_comparison_gte() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn main() ->{} Bool {
    2.5 >= 1.5
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_gte.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float combined arithmetic operations.
#[test]
fn test_float_combined_arithmetic() {
    use tribute::database::parse_with_thread_local;

    // (3.0 * 2.0) + (10.0 / 2.0) - 1.0 = 6.0 + 5.0 - 1.0 = 10.0
    let source_code = Rope::from_str(
        r#"
fn main() ->{} Float {
    let a = 3.0 * 2.0
    let b = 10.0 / 2.0
    a + b - 1.0
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_combined.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float literal WASM compilation.
#[test]
fn test_float_literal_compiles() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn main() ->{} Float {
    3.14159
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_literal.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test negative Float literal.
#[test]
fn test_negative_float_literal() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn main() ->{} Float {
    -3.14
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "negative_float.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}

/// Test Float in function parameter and return.
#[test]
fn test_float_function_param_and_return() {
    use tribute::database::parse_with_thread_local;

    let source_code = Rope::from_str(
        r#"
fn double(x: Float) ->{} Float {
    x * 2.0
}

fn main() ->{} Float {
    double(3.5)
}
"#,
    );

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source_file = SourceCst::from_path(db, "float_function.trb", source_code.clone(), tree);

        let _wasm_binary =
            compile_to_wasm_binary(db, source_file).expect("WASM compilation failed");
    });
}
