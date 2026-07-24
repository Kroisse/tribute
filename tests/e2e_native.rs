//! End-to-end tests for the native (Cranelift) compilation pipeline.
//!
//! These tests compile `.trb` source to native binaries, link them,
//! and run the resulting executables to verify the full pipeline works.
//!
//! Tests that overlap with other e2e test files (e2e_add, e2e_ability_core)
//! are kept there; this file contains native-specific tests for features
//! like tuples, enums, pattern matching, and recursion.

mod common;

#[cfg(unix)]
use common::compile_and_run_native_with_closed_stdin;
use common::{
    assert_native_output, compile_and_run_native, compile_and_run_native_asan,
    compile_and_run_native_with_stdin,
};

fn std_io_read_line_program() -> &'static str {
    r#"
use abilities::Throw
use std::io::{Error, Io, print_line, read_line}

fn read_or_error() ->{Io} String {
    handle read_line() {
        do line { line }
        op Throw::throw(error) {
            case error {
                Error::EndOfFile -> "eof"
                Error::InvalidEncoding -> "invalid"
                Error::System(_) -> "system"
            }
        }
    }
}

fn main() ->{Io} Nil {
    print_line(read_or_error())
}
"#
}

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
fn test_native_generic_specializations_distinguish_same_spelled_source_types() {
    assert_native_output(
        "generic_same_spelled_source_types.trb",
        r#"
pub mod A {
    pub struct Thing { value: Nat }
}

pub mod B {
    pub struct Thing { value: Nat }
}

fn keep(value: a) -> a {
    value
}

fn accept_a(_value: A::Thing) {}
fn accept_b(_value: B::Thing) {}

fn main() {
    accept_a(keep(A::Thing { value: 1 }))
    accept_b(keep(B::Thing { value: 2 }))
    __tribute_print_nat(3)
}
"#,
        "3",
    );
}

#[test]
fn test_native_nested_generic_types_keep_declaration_identity() {
    assert_native_output(
        "nested_generic_nominal_identity.trb",
        r#"
pub mod A {
    pub struct Token(a) {}

    pub fn tag(_value: Token(Nat)) -> Nat {
        1
    }
}

pub mod B {
    pub struct Token(a) {}

    pub fn tag(_value: Token(Nat)) -> Nat {
        12
    }
}

fn main() {
    let a = A::Token {}
    let b = B::Token {}
    __tribute_print_nat(a.tag())
    __tribute_print_nat(b.tag())
}
"#,
        "1\n12",
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

#[test]
fn test_native_list_literal_evaluates_left_to_right_once() {
    assert_native_output(
        "list_literal_order.trb",
        r#"
fn mark(value: Nat) -> Nat {
    __tribute_print_nat(value)
    value
}

fn main() {
    let values = [mark(1), mark(2), mark(3)]
    case values {
        [] -> Nil
        [head, ..tail] -> __tribute_print_nat(head)
    }
}
"#,
        "1\n2\n3\n1",
    );
}

#[test]
fn test_native_list_empty_exact_and_rest_patterns() {
    assert_native_output(
        "list_patterns.trb",
        r#"
fn classify(values: List(Nat)) -> Nat {
    case values {
        [] -> 0
        [only] -> only
        [first, second] -> first + second
        [head, ..tail] -> head
    }
}

fn main() {
    __tribute_print_nat(classify([]))
    __tribute_print_nat(classify([4]))
    __tribute_print_nat(classify([5, 6]))
    __tribute_print_nat(classify([7, 8, 9]))
}
"#,
        "0\n4\n11\n7",
    );
}

#[test]
fn test_native_short_lists_skip_nested_pattern_observation() {
    assert_native_output(
        "list_nested_pattern_safety.trb",
        r#"
enum Item {
    Number(Nat),
}

fn exact(values: List(Item)) -> Nat {
    case values {
        [Number(first), Number(second)] -> first + second
        _ -> 9
    }
}

fn prefix(values: List(Item)) -> Nat {
    case values {
        [Number(first), Number(second), ..tail] -> first + second
        _ -> 8
    }
}

fn main() {
    __tribute_print_nat(exact([]))
    __tribute_print_nat(exact([Number(1)]))
    __tribute_print_nat(prefix([]))
    __tribute_print_nat(prefix([Number(1)]))
}
"#,
        "9\n9\n8\n8",
    );
}

#[test]
fn test_native_list_tail_order_and_persistence() {
    assert_native_output(
        "list_tail_persistence.trb",
        r#"
fn tail(values: List(Nat)) -> List(Nat) {
    case values {
        [] -> []
        [head, ..tail] -> tail
    }
}

fn head_or_zero(values: List(Nat)) -> Nat {
    case values {
        [] -> 0
        [head, ..tail] -> head
    }
}

fn main() {
    let original = [4, 5, 6]
    let rest = tail(original)
    let last = tail(rest)
    __tribute_print_nat(head_or_zero(original))
    __tribute_print_nat(head_or_zero(rest))
    __tribute_print_nat(head_or_zero(last))
}
"#,
        "4\n5\n6",
    );
}

#[test]
fn test_native_list_retains_reference_elements_and_shared_tails() {
    let output = compile_and_run_native_asan(
        "list_reference_ownership.trb",
        r#"
use std::io::{Io, print_line}

fn tail(values: List(String)) -> List(String) {
    case values {
        [] -> []
        [head, ..tail] -> tail
    }
}

fn head_or_empty(values: List(String)) -> String {
    case values {
        [] -> ""
        [head, ..tail] -> head
    }
}

fn main() ->{Io} Nil {
    let original = ["first", "second"]
    let rest = tail(original)
    print_line(head_or_empty(rest))
    print_line(head_or_empty(original))
}
"#,
    );
    assert!(
        output.status.success(),
        "ASan list ownership run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "second\nfirst"
    );
}

#[test]
fn test_native_list_prepend_builds_dynamic_strings_recursively() {
    assert_native_output(
        "list_dynamic_string_prepend.trb",
        r#"
use std::io::{Io, print_line}

fn build(token: String, remaining: Nat) -> List(String) {
    case remaining {
        0 -> []
        _ -> List::prepend(token, build(token <> "!", remaining - 1))
    }
}

fn join(values: List(String)) -> String {
    case values {
        [] -> ""
        [head, ..tail] -> head <> join(tail)
    }
}

fn main() ->{Io} Nil {
    print_line(join(build("x", 3)))
}
"#,
        "xx!x!!",
    );
}

#[test]
fn test_native_list_float_exact_pattern_has_typed_empty_fallback() {
    assert_native_output(
        "list_float_pattern.trb",
        r#"
fn only_or_default(values: List(Float)) -> Float {
    case values {
        [value] -> value
        _ -> 9.5
    }
}

fn main() {
    __tribute_print_float(only_or_default([]))
    __tribute_print_float(only_or_default([1.25]))
}
"#,
        "9.5\n1.25",
    );
}

#[test]
fn test_native_generic_specializations_distinguish_builtin_and_source_list() {
    assert_native_output(
        "generic_builtin_and_source_list.trb",
        r#"
enum List(a) {
    SourceList(a),
}

fn keep(value: a) -> a {
    value
}

fn builtin_value() -> Nat {
    case keep([1]) {
        [value] -> value
        _ -> 0
    }
}

fn source_value(value: List(Nat)) -> Nat {
    2
}

fn main() {
    let source = source_value(keep(SourceList(2)))
    __tribute_print_nat(builtin_value() + source)
}
"#,
        "3",
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

/// Test short-circuit evaluation for &&.
#[test]
fn test_native_short_circuit_and() {
    assert_native_output(
        "short_circuit_and.trb",
        r#"
fn bool_to_nat(b: Bool) -> Nat {
    case b {
        True -> 1
        False -> 0
    }
}

fn main() {
    __tribute_print_nat(bool_to_nat(False && True))
    __tribute_print_nat(bool_to_nat(True && True))
    __tribute_print_nat(bool_to_nat(True && False))
}
"#,
        "0\n1\n0",
    );
}

/// Test short-circuit evaluation for ||.
#[test]
fn test_native_short_circuit_or() {
    assert_native_output(
        "short_circuit_or.trb",
        r#"
fn bool_to_nat(b: Bool) -> Nat {
    case b {
        True -> 1
        False -> 0
    }
}

fn main() {
    __tribute_print_nat(bool_to_nat(True || False))
    __tribute_print_nat(bool_to_nat(False || True))
    __tribute_print_nat(bool_to_nat(False || False))
}
"#,
        "1\n1\n0",
    );
}

/// Test that short-circuit && does not evaluate rhs when lhs is false.
/// If rhs were evaluated, the side effect would appear in the output.
#[test]
fn test_native_short_circuit_and_skips_rhs() {
    assert_native_output(
        "short_circuit_and_side_effect.trb",
        r#"
fn side_effect() -> Bool {
    __tribute_print_nat(99)
    True
}

fn main() {
    case False && side_effect() {
        True -> __tribute_print_nat(1)
        False -> __tribute_print_nat(0)
    }
}
"#,
        "0",
    );
}

/// Test that short-circuit || does not evaluate rhs when lhs is true.
#[test]
fn test_native_short_circuit_or_skips_rhs() {
    assert_native_output(
        "short_circuit_or_side_effect.trb",
        r#"
fn side_effect() -> Bool {
    __tribute_print_nat(99)
    False
}

fn main() {
    case True || side_effect() {
        True -> __tribute_print_nat(1)
        False -> __tribute_print_nat(0)
    }
}
"#,
        "1",
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
fn test_native_string_escape_sequences() {
    let output = compile_and_run_native(
        "string_escape.trb",
        r#"
fn main() {
    print_line("a\tb\nc")
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout, "a\tb\nc\n");
}

#[test]
fn test_native_string_escape_hex() {
    // \x41 = 'A', \x42 = 'B'
    let output = compile_and_run_native(
        "string_escape_hex.trb",
        r#"
fn main() {
    print_line("\x41\x42\x43")
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout.trim(), "ABC");
}

#[test]
fn test_native_string_escape_unicode() {
    // \u0041 = 'A', \u00E9 = 'é'
    let output = compile_and_run_native(
        "string_escape_unicode.trb",
        r#"
fn main() {
    print_line("\u0041\u00E9")
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout.trim(), "Aé");
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
fn test_native_std_io_prints_dynamic_strings() {
    let output = compile_and_run_native(
        "std_io_dynamic_print.trb",
        r#"
use std::io::{print, print_line}

fn message() -> String {
    "Hello" <> ", World"
}

fn main() ->{std::io::Io} Nil {
    print(message())
    print_line("!")
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
    assert_eq!(stdout, "Hello, World!\n");
}

#[test]
fn test_native_string_equality() {
    assert_native_output(
        "string_equality.trb",
        r#"
fn bool_to_nat(value: Bool) -> Nat {
    case value {
        True -> 1
        False -> 0
    }
}

fn main() {
    __tribute_print_nat(bool_to_nat("same" == "same"))
    __tribute_print_nat(bool_to_nat("same" != "different"))
    __tribute_print_nat(bool_to_nat("" == String::empty()))
    __tribute_print_nat(bool_to_nat("" != "x"))
    __tribute_print_nat(bool_to_nat("안녕🌍" == "안녕🌍"))
    __tribute_print_nat(bool_to_nat("안녕🌍" != "안녕🌎"))
    let ab = "a" <> "b"
    __tribute_print_nat(bool_to_nat("ab" == ab))

    let left_prefix = "a" <> "b"
    let left_shape = left_prefix <> "c"
    let right_suffix = "b" <> "c"
    let right_shape = "a" <> right_suffix
    __tribute_print_nat(bool_to_nat(left_shape == right_shape))

    let shared = "shared"
    let shared_twice = shared <> shared
    __tribute_print_nat(bool_to_nat(shared_twice == "sharedshared"))

    let flat = "abcd"
    let two_leaves = "ab" <> "cd"
    let three_leaf_suffix = "bc" <> "d"
    let three_leaves = "a" <> three_leaf_suffix
    __tribute_print_nat(bool_to_nat(flat == two_leaves))
    __tribute_print_nat(bool_to_nat(flat == three_leaves))

    let boundaries_left = "ab" <> "cdef"
    let boundaries_prefix = "a" <> "bc"
    let boundaries_right = boundaries_prefix <> "def"
    __tribute_print_nat(bool_to_nat(boundaries_left == boundaries_right))

    let empty_end_suffix = "abcd" <> ""
    let empty_ends = "" <> empty_end_suffix
    let empty_middle_prefix = "ab" <> ""
    let empty_middle = empty_middle_prefix <> "cd"
    __tribute_print_nat(bool_to_nat(empty_ends == empty_middle))

    let shared_part = "xy"
    let repeated_suffix = shared_part <> shared_part
    let repeated_left = shared_part <> repeated_suffix
    let repeated_prefix = "x" <> "yxy"
    let repeated_right = repeated_prefix <> "xy"
    __tribute_print_nat(bool_to_nat(repeated_left == repeated_right))

    let unicode_suffix = "녕" <> "🌍"
    let unicode_rope = "안" <> unicode_suffix
    __tribute_print_nat(bool_to_nat("안녕🌍" == unicode_rope))

    __tribute_print_nat(bool_to_nat("short" != "shorter"))
    __tribute_print_nat(bool_to_nat("xbc" != "abc"))
    __tribute_print_nat(bool_to_nat("axc" != "abc"))
    __tribute_print_nat(bool_to_nat("abx" != "abc"))

    __tribute_print_nat(bool_to_nat("abc" != "axc"))
    __tribute_print_nat(bool_to_nat("a" != "a"))
    __tribute_print_nat(bool_to_nat("a" == "b"))
}
"#,
        "1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n0\n0",
    );
}

#[test]
fn test_string_equality_prelude_uses_direct_leaf_and_cursor_spans() {
    let prelude = include_str!("../lib/std/prelude.trb");
    let equality_start = prelude
        .find("fn cursors_equal")
        .expect("String equality cursor helper");
    let equality_end = prelude[equality_start..]
        .find("// =============================================================================")
        .map(|offset| equality_start + offset)
        .expect("end of String module");
    let equality = &prelude[equality_start..equality_end];

    assert!(equality.contains("__tribute_bytes_range_equal"));
    assert!(!equality.contains("bytes_equal_at"));
    assert!(!equality.contains(".get_or_panic("));
    assert!(!equality.contains(".to_bytes()"));

    let operator_start = equality
        .find("pub fn (==)")
        .expect("String equality operator");
    let operator_end = equality[operator_start..]
        .find("/// Return the logical complement")
        .map(|offset| operator_start + offset)
        .expect("end of String equality operator");
    let operator = &equality[operator_start..operator_end];
    assert!(operator.contains("Leaf(left_bytes)"));
    assert!(operator.contains("Leaf(right_bytes)"));
    let compact_operator: String = operator.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        compact_operator.contains("__tribute_bytes_range_equal(left_bytes,0,right_bytes,0,len)")
    );
    assert_eq!(
        operator.matches("__tribute_bytes_range_equal").count(),
        1,
        "Leaf/Leaf should make exactly one full-span comparison"
    );
    assert!(
        !operator.contains("Cursor"),
        "Leaf/Leaf direct path must not construct cursors"
    );
    assert_eq!(
        operator.matches("String::rope_equal(left, right)").count(),
        2,
        "only Branch-containing cases should use the cursor fallback"
    );
}

#[test]
fn test_native_string_equality_does_not_collide_with_user_string_cursor() {
    assert_native_output(
        "string_cursor_name.trb",
        r#"
enum StringCursor {
    UserCursor(Nat),
    EmptyCursor,
}

fn bool_to_nat(value: Bool) -> Nat {
    case value {
        True -> 1
        False -> 0
    }
}

fn main() {
    let cursor = UserCursor(7)
    case cursor {
        UserCursor(value) -> __tribute_print_nat(value)
        EmptyCursor -> __tribute_print_nat(0)
    }
    __tribute_print_nat(bool_to_nat("leaf" == "leaf"))
    let rope = "ro" <> "pe"
    __tribute_print_nat(bool_to_nat(rope == "rope"))
}
"#,
        "7\n1\n1",
    );
}

#[test]
fn test_native_std_io_read_line_contract() {
    for (name, stdin, expected) in [
        ("empty", b"\n".as_slice(), "\n"),
        ("crlf", b"hello\r\n".as_slice(), "hello\n"),
        ("partial", b"partial".as_slice(), "partial\n"),
    ] {
        let output = compile_and_run_native_with_stdin(
            &format!("std_io_read_line_{name}.trb"),
            std_io_read_line_program(),
            stdin,
        );
        assert!(
            output.status.success(),
            "{name}: exit={:?}, stderr='{}'",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), expected, "{name}");
    }
}

#[test]
fn test_native_std_io_read_line_preserves_buffered_input() {
    let output = compile_and_run_native_with_stdin(
        "std_io_read_line_buffered.trb",
        r#"
use abilities::Throw
use std::io::{Error, Io, print_line, read_line}

fn read_or_error() ->{Io} String {
    handle read_line() {
        do line { line }
        op Throw::throw(error) {
            case error {
                Error::EndOfFile -> "eof"
                Error::InvalidEncoding -> "invalid"
                Error::System(_) -> "system"
            }
        }
    }
}

fn main() ->{Io} Nil {
    print_line(read_or_error())
    print_line(read_or_error())
}
"#,
        b"first\nsecond\n",
    );
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "first\nsecond\n");
}

#[test]
fn test_native_std_io_accepts_large_stdin() {
    let mut stdin = vec![b'x'; 256 * 1024];
    stdin.push(b'\n');
    let output = compile_and_run_native_with_stdin(
        "std_io_read_line_large.trb",
        std_io_read_line_program(),
        &stdin,
    );
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(output.stdout, stdin);
}

#[test]
fn test_native_std_io_read_line_errors() {
    for (name, stdin, expected) in [
        ("eof", b"".as_slice(), "eof\n"),
        ("invalid", b"\xff\n".as_slice(), "invalid\n"),
    ] {
        let output = compile_and_run_native_with_stdin(
            &format!("std_io_read_line_{name}.trb"),
            std_io_read_line_program(),
            stdin,
        );
        assert!(
            output.status.success(),
            "{name}: exit={:?}, stderr='{}'",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), expected, "{name}");
    }
}

#[cfg(unix)]
#[test]
fn test_native_std_io_read_line_system_error() {
    let output = compile_and_run_native_with_closed_stdin(
        "std_io_read_line_system.trb",
        std_io_read_line_program(),
    );
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "system\n");
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

// =========================================================================
// Bytes literal tests
// =========================================================================

#[test]
fn test_native_bytes_literal_basic() {
    let output = compile_and_run_native(
        "bytes_lit_basic.trb",
        r#"
fn main() {
    let bs = b"hello"
    print_line(String::from_bytes(bs))
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout.trim(), "hello");
}

#[test]
fn test_native_bytes_literal_empty() {
    let output = compile_and_run_native(
        "bytes_lit_empty.trb",
        r#"
fn main() {
    let bs = b""
    print_line(String::from_bytes(bs))
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout, "\n");
}

#[test]
fn test_native_bytes_literal_escape_sequences() {
    let output = compile_and_run_native(
        "bytes_lit_escape.trb",
        r#"
fn main() {
    let bs = b"a\tb\nc"
    print_line(String::from_bytes(bs))
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout, "a\tb\nc\n");
}

#[test]
fn test_native_bytes_literal_raw() {
    // Raw bytes: backslash is literal, not an escape
    let output = compile_and_run_native(
        "bytes_lit_raw.trb",
        r#"
fn main() {
    let bs = rb"\n\t"
    print_line(String::from_bytes(bs))
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout.trim(), r"\n\t");
}

#[test]
fn test_native_bytes_literal_len() {
    assert_native_output(
        "bytes_lit_len.trb",
        r#"
fn main() {
    let bs = b"hello"
    __tribute_print_nat(bs.len())
}
"#,
        "5",
    );
}

#[test]
fn test_native_bytes_literal_concat() {
    let output = compile_and_run_native(
        "bytes_lit_concat.trb",
        r#"
fn main() {
    let a = b"Hello, "
    let b = b"World!"
    print_line(String::from_bytes(a <> b))
}
"#,
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit={:?}, stderr='{}'",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(stdout.trim(), "Hello, World!");
}

// =========================================================================
// Bytes operations tests
// =========================================================================

#[test]
fn test_native_bytes_get_or_panic() {
    assert_native_output(
        "bytes_get_or_panic.trb",
        r#"
fn main() {
    let bs = b"abc"
    __tribute_print_nat(bs.get_or_panic(0))
    __tribute_print_nat(bs.get_or_panic(1))
    __tribute_print_nat(bs.get_or_panic(2))
}
"#,
        "97\n98\n99",
    );
}

#[test]
fn test_native_bytes_get_safe() {
    assert_native_output(
        "bytes_get_safe.trb",
        r#"
fn main() ->{std::io::Io} Nil {
    let bs = b"hi"
    let a = bs.get(0)
    let b = bs.get(2)
    case a {
        Some(_) -> std::io::print_line("some")
        None -> std::io::print_line("none")
    }
    case b {
        Some(_) -> std::io::print_line("some")
        None -> std::io::print_line("none")
    }
}
"#,
        "some\nnone",
    );
}

#[test]
fn test_native_bytes_slice_or_panic() {
    assert_native_output(
        "bytes_slice_or_panic.trb",
        r#"
fn main() {
    let bs = b"hello world"
    let sl = bs.slice_or_panic(0, 5)
    print(String::from_bytes(sl))
}
"#,
        "hello",
    );
}

#[test]
fn test_native_bytes_slice_safe() {
    assert_native_output(
        "bytes_slice_safe.trb",
        r#"
fn main() {
    let bs = b"hello world"
    let sl = bs.slice(6, 11)
    print(String::from_bytes(sl))
}
"#,
        "world",
    );
}

#[test]
fn test_native_bytes_slice_clamping() {
    assert_native_output(
        "bytes_slice_clamp.trb",
        r#"
fn main() {
    let bs = b"hello"
    let sl = bs.slice(3, 100)
    __tribute_print_nat(sl.len())
}
"#,
        "2",
    );
}

#[test]
fn test_native_bytes_as_function_arg() {
    assert_native_output(
        "bytes_fn_arg.trb",
        r#"
fn print_bytes_len(bs: Bytes) -> Nil {
    __tribute_print_nat(bs.len())
}

fn main() {
    let bs = b"test"
    print_bytes_len(bs)
}
"#,
        "4",
    );
}

// =========================================================================
// String::empty() and Bytes::empty() tests
// =========================================================================

#[test]
fn test_native_string_empty() {
    assert_native_output(
        "string_empty.trb",
        r#"
fn main() {
    let s = String::empty()
    __tribute_print_nat(s.len())
}
"#,
        "0",
    );
}

#[test]
fn test_native_bytes_empty() {
    assert_native_output(
        "bytes_empty.trb",
        r#"
fn main() {
    let bs = Bytes::empty()
    __tribute_print_nat(bs.len())
}
"#,
        "0",
    );
}
