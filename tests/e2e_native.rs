//! End-to-end tests for the native (Cranelift) compilation pipeline.
//!
//! These tests compile `.trb` source to native binaries, link them,
//! and run the resulting executables to verify the full pipeline works.
//!
//! Tests that overlap with other e2e test files (e2e_add, e2e_ability_core)
//! are kept there; this file contains native-specific tests for features
//! like tuples, enums, pattern matching, and recursion.

mod common;

use common::compile_and_run_native;

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
    let output = compile_and_run_native(
        "arithmetic.trb",
        r#"
fn main() {
    let _ = 10 + 20 + 3
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_native_function_call() {
    let output = compile_and_run_native(
        "function_call.trb",
        r#"
fn add(a: Nat, b: Nat) -> Nat {
    a + b
}

fn main() {
    let _ = add(10, 20)
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_native_let_binding() {
    let output = compile_and_run_native(
        "let_binding.trb",
        r#"
fn main() {
    let a = 10
    let b = 20
    let _ = a + b
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

// =============================================================================
// Intermediate Feature Tests
// =============================================================================

#[test]
fn test_native_case_expression() {
    let output = compile_and_run_native(
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
    let _ = classify(5)
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_native_struct() {
    let output = compile_and_run_native(
        "struct.trb",
        r#"
struct Point { x: Nat, y: Nat }

fn main() {
    let p = Point { x: 10, y: 20 }
    let _ = p.x()
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "native backend: closure codegen causes linker crash (needs investigation)"]
fn test_native_closure() {
    let output = compile_and_run_native(
        "closure.trb",
        r#"
fn main() {
    let a = 10
    let f = fn(x) { x + a }
    let _ = f(32)
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_native_enum_case() {
    let output = compile_and_run_native(
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
    let _ = area(Circle(5))
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test enum with empty variants (no fields).
#[test]
fn test_native_enum_empty_variants() {
    let output = compile_and_run_native(
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
    let _ = to_num(Green)
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test enum with mixed variant arities (Option-like).
#[test]
fn test_native_enum_option_like() {
    let output = compile_and_run_native(
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
    let _ = maybe_unwrap(Just(42), 0)
    let _ = maybe_unwrap(Nothing, 99)
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_native_recursion() {
    let output = compile_and_run_native(
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
    let _ = fibonacci(10)
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_native_tuple_create_and_match() {
    let output = compile_and_run_native(
        "tuple_create_match.trb",
        r#"
fn main() {
    let t = (1, 2)
    let (a, b) = t
    let _ = a + b
}
"#,
    );
    assert!(
        output.status.success(),
        "Native binary exited with non-zero status: {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}
