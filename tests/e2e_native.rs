//! End-to-end tests for the native (Cranelift) compilation pipeline.
//!
//! These tests compile `.trb` source to native binaries, link them,
//! and run the resulting executables to verify the full pipeline works.
//!
//! Tests that overlap with other e2e test files (e2e_add, e2e_ability_core)
//! are kept there; this file contains native-specific tests for features
//! like tuples, enums, pattern matching, and recursion.

mod common;

use common::{assert_native_output, compile_and_run_native};

#[test]
fn test_native_simple_literal() {
    let output = compile_and_run_native("simple_literal.trb", "fn main() { }");
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_native_arithmetic() {
    assert_native_output(
        "arithmetic.trb",
        r#"
fn main() {
    __tribute_print_nat(10 + 20 + 3)
}
"#,
        "33",
    );
}

#[test]
fn test_native_function_call() {
    assert_native_output(
        "function_call.trb",
        r#"
fn add(a: Nat, b: Nat) -> Nat {
    a + b
}

fn main() {
    __tribute_print_nat(add(10, 20))
}
"#,
        "30",
    );
}

#[test]
fn test_native_let_binding() {
    assert_native_output(
        "let_binding.trb",
        r#"
fn main() {
    let a = 10
    let b = 20
    __tribute_print_nat(a + b)
}
"#,
        "30",
    );
}

// =============================================================================
// Intermediate Feature Tests
// =============================================================================

#[test]
fn test_native_case_expression() {
    assert_native_output(
        "case_expression.trb",
        r#"
fn classify(n: Nat) -> Nat {
    case n {
        0 -> 0
        1 -> 1
        _ -> 2
    }
}

fn main() {
    __tribute_print_nat(classify(5))
}
"#,
        "2",
    );
}

#[test]
fn test_native_struct() {
    assert_native_output(
        "struct.trb",
        r#"
struct Point { x: Nat, y: Nat }

fn main() {
    let p = Point { x: 10, y: 20 }
    __tribute_print_nat(p.x())
}
"#,
        "10",
    );
}

#[test]
fn test_native_closure() {
    assert_native_output(
        "closure.trb",
        r#"
fn main() {
    let a = 10
    let f = fn(x) { x + a }
    __tribute_print_nat(f(32))
}
"#,
        "42",
    );
}

#[test]
fn test_native_enum_case() {
    assert_native_output(
        "enum_case.trb",
        r#"
enum Shape {
    Circle(Nat),
    Square(Nat),
}

fn area(s: Shape) -> Nat {
    case s {
        Circle(r) -> r * r
        Square(side) -> side * side
    }
}

fn main() {
    __tribute_print_nat(area(Circle(5)))
}
"#,
        "25",
    );
}

/// Test enum with empty variants (no fields).
#[test]
fn test_native_enum_empty_variants() {
    assert_native_output(
        "enum_empty.trb",
        r#"
enum Color {
    Red,
    Green,
    Blue,
}

fn to_num(c: Color) -> Nat {
    case c {
        Red -> 1
        Green -> 2
        Blue -> 3
    }
}

fn main() {
    __tribute_print_nat(to_num(Green))
}
"#,
        "2",
    );
}

/// Test enum with mixed variant arities (Option-like).
#[test]
fn test_native_enum_option_like() {
    assert_native_output(
        "enum_option.trb",
        r#"
enum Maybe {
    Just(Nat),
    Nothing,
}

fn maybe_unwrap(m: Maybe, default: Nat) -> Nat {
    case m {
        Just(x) -> x
        Nothing -> default
    }
}

fn main() {
    __tribute_print_nat(maybe_unwrap(Just(42), 0))
    __tribute_print_nat(maybe_unwrap(Nothing, 99))
}
"#,
        "42\n99",
    );
}

#[test]
fn test_native_recursion() {
    assert_native_output(
        "recursion.trb",
        r#"
fn fibonacci(n: Nat) -> Nat {
    case n {
        0 -> 0
        1 -> 1
        _ -> fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() {
    __tribute_print_nat(fibonacci(10))
}
"#,
        "55",
    );
}

#[test]
fn test_native_tuple_create_and_match() {
    assert_native_output(
        "tuple_create_match.trb",
        r#"
fn main() {
    let t = #(1, 2)
    let #(a, b) = t
    __tribute_print_nat(a + b)
}
"#,
        "3",
    );
}

/// Regression test for #548: Bool case expression always took the first branch.
#[test]
fn test_native_bool_case() {
    assert_native_output(
        "bool_case.trb",
        r#"
fn pick(b: Bool) -> Nat {
    case b {
        True -> 10
        False -> 42
    }
}

fn main() {
    __tribute_print_nat(pick(True))
    __tribute_print_nat(pick(False))
}
"#,
        "10\n42",
    );
}

/// Test enum destructuring with mixed-type fields.
/// Regression test: each pattern binding must get a fresh type variable.
#[test]
fn test_native_enum_mixed_type_fields() {
    assert_native_output(
        "enum_mixed_fields.trb",
        r#"
enum Tree {
    Leaf(Nat),
    Branch(Tree, Tree, Nat),
}

fn sum(t: Tree) -> Nat {
    case t {
        Leaf(n) -> n
        Branch(left, right, len) -> sum(left) + sum(right) + len
    }
}

fn main() {
    let t = Branch(Leaf(1), Branch(Leaf(2), Leaf(3), 0), 0)
    __tribute_print_nat(sum(t))
}
"#,
        "6",
    );
}

// =============================================================================
// String / print_line Tests
// =============================================================================

#[test]
fn test_native_print_line() {
    let output = compile_and_run_native(
        "print_line.trb",
        r#"
fn main() {
    print_line("Hello, World!")
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit={:?}, stdout='{}', stderr='{}'",
        output.status,
        stdout,
        stderr,
    );
    assert_eq!(stdout.trim(), "Hello, World!");
}

#[test]
fn test_native_print_line_empty() {
    let output = compile_and_run_native(
        "print_line_empty.trb",
        r#"
fn main() {
    print_line("")
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit={:?}, stdout='{}', stderr='{}'",
        output.status,
        stdout,
        stderr,
    );
    // Empty string + newline
    assert_eq!(stdout, "\n");
}
