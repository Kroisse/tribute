//! End-to-end tests for Float type operations.
//!
//! These tests verify Float type functionality including:
//! - Arithmetic operations (+, -, *, /)
//! - Comparison operations (<, >, <=, >=, ==)
//! - Native compilation and execution with stdout verification

mod common;

use common::assert_native_output;

#[test]
fn test_float_multiply() {
    // 3.0 * 2.0 = 6.0
    assert_native_output(
        "float_multiply.trb",
        r#"
fn main() {
    __tribute_print_float(3.0 * 2.0)
}
"#,
        "6.0",
    );
}

#[test]
fn test_float_subtraction() {
    // 5.5 - 2.5 = 3.0
    assert_native_output(
        "float_subtraction.trb",
        r#"
fn main() {
    __tribute_print_float(5.5 - 2.5)
}
"#,
        "3.0",
    );
}

#[test]
fn test_float_division() {
    // 10.0 / 4.0 = 2.5
    assert_native_output(
        "float_division.trb",
        r#"
fn main() {
    __tribute_print_float(10.0 / 4.0)
}
"#,
        "2.5",
    );
}

#[test]
fn test_float_literal_compiles() {
    assert_native_output(
        "float_literal.trb",
        r#"
fn main() {
    __tribute_print_float(3.14159)
}
"#,
        "3.14159",
    );
}

#[test]
fn test_negative_float_literal() {
    assert_native_output(
        "negative_float.trb",
        r#"
fn main() {
    __tribute_print_float(-3.14)
}
"#,
        "-3.14",
    );
}

#[test]
fn test_float_function_param_and_return() {
    assert_native_output(
        "float_function.trb",
        r#"
fn double(x: Float) -> Float {
    x * 2.0
}

fn main() {
    __tribute_print_float(double(3.5))
}
"#,
        "7.0",
    );
}

#[test]
fn test_float_combined_arithmetic() {
    // (3.0 * 2.0) + (10.0 / 2.0) - 1.0 = 6.0 + 5.0 - 1.0 = 10.0
    assert_native_output(
        "float_combined.trb",
        r#"
fn main() {
    let a = 3.0 * 2.0
    let b = 10.0 / 2.0
    __tribute_print_float(a + b - 1.0)
}
"#,
        "10.0",
    );
}

// =============================================================================
// Comparison Tests (compile-only — Bool print not yet available)
// =============================================================================

#[test]
fn test_float_comparison_less_than() {
    assert_native_output(
        "float_less_than.trb",
        r#"
fn main() {
    let _ = 1.5 < 2.5
}
"#,
        "",
    );
}

#[test]
fn test_float_comparison_greater_than() {
    assert_native_output(
        "float_greater_than.trb",
        r#"
fn main() {
    let _ = 2.5 > 1.5
}
"#,
        "",
    );
}

#[test]
fn test_float_comparison_equality() {
    assert_native_output(
        "float_equality.trb",
        r#"
fn main() {
    let _ = 1.5 == 1.5
}
"#,
        "",
    );
}

#[test]
fn test_float_comparison_lte() {
    assert_native_output(
        "float_lte.trb",
        r#"
fn main() {
    let _ = 1.5 <= 2.5
}
"#,
        "",
    );
}

#[test]
fn test_float_comparison_gte() {
    assert_native_output(
        "float_gte.trb",
        r#"
fn main() {
    let _ = 2.5 >= 1.5
}
"#,
        "",
    );
}

// =============================================================================
// Float comparison with branching (regression test for is_float operand type check)
// =============================================================================

#[test]
fn test_float_comparison_branch_eq() {
    assert_native_output(
        "float_cmp_branch_eq.trb",
        r#"
fn main() {
    let a = 1.5
    let b = 1.5
    case a == b {
        true -> __tribute_print_int(1)
        false -> __tribute_print_int(0)
    }
}
"#,
        "1",
    );
}

#[test]
fn test_float_comparison_branch_lt() {
    assert_native_output(
        "float_cmp_branch_lt.trb",
        r#"
fn main() {
    let a = 1.0
    let b = 2.0
    case a < b {
        true -> __tribute_print_int(1)
        false -> __tribute_print_int(0)
    }
}
"#,
        "1",
    );
}

#[test]
fn test_float_comparison_branch_ne() {
    assert_native_output(
        "float_cmp_branch_ne.trb",
        r#"
fn main() {
    let a = 1.0
    let b = 2.0
    case a != b {
        true -> __tribute_print_int(1)
        false -> __tribute_print_int(0)
    }
}
"#,
        "1",
    );
}

#[test]
fn test_float_comparison_branch_gt() {
    assert_native_output(
        "float_cmp_branch_gt.trb",
        r#"
fn main() {
    let a = 2.0
    let b = 1.0
    case a > b {
        true -> __tribute_print_int(1)
        false -> __tribute_print_int(0)
    }
}
"#,
        "1",
    );
}

#[test]
fn test_float_comparison_branch_le() {
    assert_native_output(
        "float_cmp_branch_le.trb",
        r#"
fn main() {
    let a = 1.0
    let b = 2.0
    case a <= b {
        true -> __tribute_print_int(1)
        false -> __tribute_print_int(0)
    }
}
"#,
        "1",
    );
}

#[test]
fn test_float_comparison_branch_ge() {
    assert_native_output(
        "float_cmp_branch_ge.trb",
        r#"
fn main() {
    let a = 2.0
    let b = 2.0
    case a >= b {
        true -> __tribute_print_int(1)
        false -> __tribute_print_int(0)
    }
}
"#,
        "1",
    );
}
