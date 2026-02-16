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
    let status = compile_and_run_native("simple_literal.trb", "fn main() -> Int { 42 }");
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
fn main() -> Nat {
    10 + 20 + 3
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

fn main() -> Nat {
    add(10, 20)
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
fn main() -> Nat {
    let a = 10;
    let b = 20;
    a + b
}
"#,
    );
    assert!(
        status.success(),
        "Native binary exited with non-zero status: {:?}",
        status
    );
}
