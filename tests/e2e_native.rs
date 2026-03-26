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

#[test]
fn test_native_print_line_multiple() {
    let output = compile_and_run_native(
        "print_line_multi.trb",
        r#"
fn main() {
    print_line("Hello")
    print_line("World")
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
    assert_eq!(stdout, "Hello\nWorld\n");
}

#[test]
fn test_native_string_as_function_arg() {
    let output = compile_and_run_native(
        "string_arg.trb",
        r#"
fn greet(name: String) -> Nil {
    print_line(name)
}

fn main() {
    greet("Tribute")
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
    assert_eq!(stdout.trim(), "Tribute");
}

#[test]
fn test_native_string_dedup_rodata() {
    // Same string literal used twice should work (rodata deduplication)
    let output = compile_and_run_native(
        "string_dedup.trb",
        r#"
fn main() {
    print_line("echo")
    print_line("echo")
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
    assert_eq!(stdout, "echo\necho\n");
}

// =============================================================================
// UFCS Tests — prelude module methods resolvable from user code (#577)
// =============================================================================

#[test]
fn test_native_ufcs_string_len() {
    // s.len() should resolve to String::len via TDNR
    let output = compile_and_run_native(
        "ufcs_string_len.trb",
        r#"
fn main() {
    let s = "Hello"
    let _ = s.len()
}
"#,
    );
    assert!(
        output.status.success(),
        "s.len() UFCS should compile and run; exit={:?}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn test_native_ufcs_method_disambiguation() {
    // String::len and Bytes::len both register under "len" — receiver type disambiguates.
    // to_bytes is called via fully-qualified name to keep the test focused on len disambiguation.
    let output = compile_and_run_native(
        "ufcs_method_disambiguation.trb",
        r#"
fn main() {
    let s = "hello"
    let bs = String::to_bytes(s)
    let _ = s.len()
    let _ = bs.len()
}
"#,
    );
    assert!(
        output.status.success(),
        "String::len and Bytes::len should disambiguate by receiver type; exit={:?}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn test_native_ufcs_chained() {
    // Chained UFCS: s.to_bytes().len() requires post-solve deferred method resolution
    // because to_bytes() return type is inferred after constraint solving
    let output = compile_and_run_native(
        "ufcs_chained.trb",
        r#"
fn main() {
    let s = "hello"
    let _ = s.to_bytes().len()
}
"#,
    );
    assert!(
        output.status.success(),
        "chained UFCS s.to_bytes().len() should compile and run; exit={:?}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn test_native_ufcs_triple_chain() {
    // 3-level chained UFCS with user-defined types: each step's return type must be
    // inferred via post-solve deferred resolution before the next step can resolve.
    let output = compile_and_run_native(
        "ufcs_triple_chain.trb",
        r#"
struct Foo { value: Nat }
struct Bar { value: Nat }
struct Baz { value: Nat }

pub mod Foo {
    pub fn to_bar(f: Foo) -> Bar { Bar { value: f.value + 1 } }
}

pub mod Bar {
    pub fn to_baz(b: Bar) -> Baz { Baz { value: b.value + 1 } }
}

pub mod Baz {
    pub fn get_value(b: Baz) -> Nat { b.value }
}

fn main() {
    let f = Foo { value: 10 }
    let _ = f.to_bar().to_baz().get_value()
}
"#,
    );
    assert!(
        output.status.success(),
        "3-level chained UFCS should compile and run; exit={:?}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn test_native_ufcs_chained_multi_arg() {
    // Chained UFCS where an intermediate method takes multiple arguments.
    // Regression test for #582: extra arguments were dropped during codegen.
    let output = compile_and_run_native(
        "ufcs_chained_multi_arg.trb",
        r#"
struct Pair { x: Nat, y: Nat }
struct Triple { x: Nat, y: Nat, z: Nat }

pub mod Pair {
    pub fn extend(p: Pair, z: Nat) -> Triple {
        Triple { x: p.x, y: p.y, z: z }
    }
}

pub mod Triple {
    pub fn sum(t: Triple) -> Nat { t.x + t.y + t.z }
    pub fn scale(t: Triple, factor: Nat) -> Triple {
        Triple { x: t.x * factor, y: t.y * factor, z: t.z * factor }
    }
}

fn main() {
    let p = Pair { x: 1, y: 2 }
    let _ = p.extend(3).scale(10).sum()
}
"#,
    );
    assert!(
        output.status.success(),
        "chained UFCS with multi-arg method should compile and run; exit={:?}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn test_native_ufcs_chained_multi_arg_user_defined() {
    // Chained UFCS with user-defined types where intermediate methods have extra args.
    let output = compile_and_run_native(
        "ufcs_chained_multi_arg_user.trb",
        r#"
struct Pair { x: Nat, y: Nat }
struct Triple { x: Nat, y: Nat, z: Nat }

pub mod Pair {
    pub fn extend(p: Pair, z: Nat) -> Triple {
        Triple { x: p.x, y: p.y, z: z }
    }
}

pub mod Triple {
    pub fn sum(t: Triple) -> Nat { t.x + t.y + t.z }
}

fn main() {
    let p = Pair { x: 1, y: 2 }
    let _ = p.extend(3).sum()
}
"#,
    );
    assert!(
        output.status.success(),
        "chained UFCS with multi-arg user-defined method should compile and run; exit={:?}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}
