//! End-to-end tests for the native (Cranelift) compilation pipeline.
//!
//! These tests compile `.trb` source to native binaries, link them,
//! and run the resulting executables to verify the full pipeline works.

use std::process::{Command, ExitStatus};

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::{compile_to_native_binary, link_native_binary};
use tribute_front::SourceCst;
use tribute_passes::Diagnostic;

/// Compile Tribute source code to a native binary, link it, and run it.
///
/// Returns the exit status of the executed binary.
/// Panics if compilation, linking, or execution fails.
fn compile_and_run_native(source_name: &str, source_code: &str) -> ExitStatus {
    use tribute::database::parse_with_thread_local;

    let source_rope = Rope::from_str(source_code);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_rope, None);
        let source_file = SourceCst::from_path(db, source_name, source_rope.clone(), tree);

        let object_bytes = compile_to_native_binary(db, source_file).unwrap_or_else(|| {
            let diagnostics: Vec<_> =
                compile_to_native_binary::accumulated::<Diagnostic>(db, source_file);
            for diag in &diagnostics {
                eprintln!("Diagnostic: {:?}", diag);
            }
            panic!(
                "Native compilation failed with {} diagnostics",
                diagnostics.len()
            );
        });

        // Link into executable
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let exec_path = temp_dir.path().join("tribute_test_bin");

        link_native_binary(&object_bytes, &exec_path).unwrap_or_else(|e| {
            panic!("Linking failed: {e}");
        });

        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o755);
            std::fs::set_permissions(&exec_path, perms).expect("Failed to set permissions");
        }

        // Run the executable
        Command::new(&exec_path)
            .status()
            .unwrap_or_else(|e| panic!("Failed to execute native binary: {e}"))
    })
}

#[test]
fn test_native_simple_literal() {
    let status = compile_and_run_native("simple_literal.trb", "fn main() { }");
    assert!(
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
fn test_native_arithmetic() {
    let status = compile_and_run_native(
        "arithmetic.trb",
        r#"
fn main() {
    let _ = 10 + 20 + 3
}
"#,
    );
    assert!(
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
fn test_native_function_call() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
fn test_native_let_binding() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

// =============================================================================
// Intermediate Feature Tests
// =============================================================================

#[test]
fn test_native_case_expression() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
fn test_native_struct() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
#[ignore = "native backend: closure codegen causes linker crash (needs investigation)"]
fn test_native_closure() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
#[ignore = "native backend: clif.iconst value mapping issue in case expression codegen"]
fn test_native_enum_case() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

/// Test enum with empty variants (no fields).
#[test]
fn test_native_enum_empty_variants() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

/// Test enum with mixed variant arities (Option-like).
#[test]
#[ignore = "native backend: duplicate function definition with mixed-arity enum variants"]
fn test_native_enum_option_like() {
    let status = compile_and_run_native(
        "enum_option.trb",
        r#"
enum Maybe {
    Just(Nat),
    Nothing,
}

fn unwrap_or(m: Maybe, default: Nat) -> Nat {
    case m {
        Just(x) -> x
        Nothing -> default
    }
}

fn main() {
    let _ = unwrap_or(Just(42), 0)
    let _ = unwrap_or(Nothing, 99)
}
"#,
    );
    assert!(
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
fn test_native_recursion() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

// =============================================================================
// Ability E2E Tests
// =============================================================================

/// Test handler direct result path (no effect operations).
/// Mirrors WASM test `test_handler_direct_result`.
#[test]
#[ignore = "native backend: ability handlers require RTTI/continuation support"]
fn test_native_ability_handler_direct_result() {
    let status = compile_and_run_native(
        "ability_direct_result.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn no_effects() ->{State(Nat)} Nat {
    42
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() {
    let _ = run_state(fn() { no_effects() }, 0)
}
"#,
    );
    assert!(
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

/// Test simple State::get handler.
/// Mirrors WASM test `test_state_get_simple`.
#[test]
#[ignore = "native backend: ability handlers require RTTI/continuation support"]
fn test_native_state_get_simple() {
    let status = compile_and_run_native(
        "state_get_simple.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn get_state() ->{State(Nat)} Nat {
    State::get()
}

fn main() {
    let _ = handle get_state() {
        { result } -> result
        { State::get() -> k } -> 42
        { State::set(v) -> k } -> 0
    }
}
"#,
    );
    assert!(
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

/// Test State::set followed by State::get with run_state handler.
/// Mirrors WASM test `test_state_set_then_get`.
#[test]
#[ignore = "native backend: ability handlers require RTTI/continuation support"]
fn test_native_state_set_then_get() {
    let status = compile_and_run_native(
        "state_set_then_get.trb",
        r#"
ability State(s) {
    fn get() -> s
    fn set(value: s) -> Nil
}

fn set_then_get() ->{State(Nat)} Nat {
    State::set(100)
    State::get()
}

fn run_state(comp: fn() ->{e, State(s)} a, init: s) ->{e} a {
    handle comp() {
        { result } -> result
        { State::get() -> k } -> run_state(fn() { k(init) }, init)
        { State::set(v) -> k } -> run_state(fn() { k(Nil) }, v)
    }
}

fn main() {
    let _ = run_state(fn() { set_then_get() }, 0)
}
"#,
    );
    assert!(
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}

#[test]
fn test_native_tuple_create_and_match() {
    let status = compile_and_run_native(
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
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}
