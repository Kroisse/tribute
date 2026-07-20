//! Common test utilities for e2e tests.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::pipeline::{
    BorrowedParameterPolicy, CompilationConfig, NativeOptimizationOptions, OptimizationOptions,
    PairedRcEliminationPolicy, compile_to_native_binary, link_native_binary,
};
use tribute_front::SourceCst;
use tribute_front::ast_to_ir::{AstToIrOptions, DoneContinuationPolicy};
use tribute_passes::Diagnostic;

#[cfg(unix)]
unsafe extern "C" {
    fn close(fd: i32) -> i32;
}

/// Compile source to a native object file, panicking with diagnostics on failure.
#[allow(dead_code)]
pub fn compile_native_or_panic(db: &dyn salsa::Database, source_file: SourceCst) -> Vec<u8> {
    compile_native_or_panic_with(db, source_file, false)
}

/// Compile source to a native object file with optional ASan, panicking with diagnostics on failure.
#[allow(dead_code)]
pub fn compile_native_or_panic_with(
    db: &dyn salsa::Database,
    source_file: SourceCst,
    sanitize_address: bool,
) -> Vec<u8> {
    compile_native_or_panic_with_options(
        db,
        source_file,
        sanitize_address,
        OptimizationOptions::production(),
    )
}

/// Compile source with explicit optimization selection.
#[allow(dead_code)]
pub fn compile_native_or_panic_with_options(
    db: &dyn salsa::Database,
    source_file: SourceCst,
    sanitize_address: bool,
    optimizations: OptimizationOptions,
) -> Vec<u8> {
    let config = CompilationConfig::new(db, sanitize_address, optimizations);
    compile_to_native_binary(db, source_file, config).unwrap_or_else(|| {
        let diagnostics: Vec<_> =
            compile_to_native_binary::accumulated::<Diagnostic>(db, source_file, config);
        for diag in &diagnostics {
            eprintln!("Diagnostic: {:?}", diag);
        }
        panic!(
            "Native compilation failed with {} diagnostics",
            diagnostics.len()
        );
    })
}

/// Compile Tribute source code to a native binary, link it, and run it.
///
/// Returns the [`Output`] (status, stdout, stderr) of the executed binary.
/// Panics if compilation, linking, or execution fails.
#[allow(dead_code)]
pub fn compile_and_run_native(source_name: &str, source_code: &str) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        false,
        DoneContinuationPolicy::PerCompilationUnit,
        PairedRcEliminationPolicy::Enabled,
        BorrowedParameterPolicy::ElideProvenBorrowed,
        NativeStdin::Null,
    )
}

/// Compile and run with explicit done-continuation deduplication selection.
#[allow(dead_code)]
pub fn compile_and_run_native_with_done_continuation_dedup(
    source_name: &str,
    source_code: &str,
    policy: DoneContinuationPolicy,
) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        false,
        policy,
        PairedRcEliminationPolicy::Enabled,
        BorrowedParameterPolicy::ElideProvenBorrowed,
        NativeStdin::Null,
    )
}

/// Compile and run with explicit paired RC elimination selection.
#[allow(dead_code)]
pub fn compile_and_run_native_with_paired_rc_elimination(
    source_name: &str,
    source_code: &str,
    policy: PairedRcEliminationPolicy,
) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        false,
        DoneContinuationPolicy::PerCompilationUnit,
        policy,
        BorrowedParameterPolicy::Preserve,
        NativeStdin::Null,
    )
}

/// Compile and run with explicit borrowed-parameter RC selection.
#[allow(dead_code)]
pub fn compile_and_run_native_with_borrowed_parameters(
    source_name: &str,
    source_code: &str,
    policy: BorrowedParameterPolicy,
    sanitize_address: bool,
) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        sanitize_address,
        DoneContinuationPolicy::PerCompilationUnit,
        PairedRcEliminationPolicy::Disabled,
        policy,
        NativeStdin::Null,
    )
}

/// Compile and run Tribute source with raw bytes supplied to native stdin.
#[allow(dead_code)]
pub fn compile_and_run_native_with_stdin(
    source_name: &str,
    source_code: &str,
    stdin: &[u8],
) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        false,
        DoneContinuationPolicy::PerCompilationUnit,
        PairedRcEliminationPolicy::Enabled,
        BorrowedParameterPolicy::ElideProvenBorrowed,
        NativeStdin::Bytes(stdin),
    )
}

/// Compile and run Tribute source after closing native stdin.
#[cfg(unix)]
#[allow(dead_code)]
pub fn compile_and_run_native_with_closed_stdin(source_name: &str, source_code: &str) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        false,
        DoneContinuationPolicy::PerCompilationUnit,
        PairedRcEliminationPolicy::Enabled,
        BorrowedParameterPolicy::ElideProvenBorrowed,
        NativeStdin::Closed,
    )
}

/// Extern declarations for print intrinsics, prepended to test source code.
pub const PRINT_EXTERNS: &str = "\
extern \"C\" fn __tribute_print_nat(value: Nat) -> Nil
extern \"C\" fn __tribute_print_int(value: Int) -> Nil
extern \"C\" fn __tribute_print_float(value: Float) -> Nil
";

/// Run a native test and assert that stdout matches the expected output.
///
/// Automatically prepends extern declarations for `__tribute_print_nat`
/// and `__tribute_print_int`.
#[allow(dead_code)]
pub fn assert_native_output(source_name: &str, source_code: &str, expected_stdout: &str) {
    let full_source = format!("{PRINT_EXTERNS}\n{source_code}");
    let output = compile_and_run_native(source_name, &full_source);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit={:?}, stdout='{}', stderr='{}'",
        output.status,
        stdout,
        stderr,
    );
    assert_eq!(
        stdout.trim(),
        expected_stdout,
        "stdout mismatch for {source_name}"
    );
}

/// Compile Tribute source code to a native binary with ASan enabled, link it, and run it.
///
/// Returns the [`Output`] (status, stdout, stderr) of the executed binary.
/// Panics if compilation, linking, or execution fails.
#[allow(dead_code)]
pub fn compile_and_run_native_asan(source_name: &str, source_code: &str) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        true,
        DoneContinuationPolicy::PerCompilationUnit,
        PairedRcEliminationPolicy::Enabled,
        BorrowedParameterPolicy::ElideProvenBorrowed,
        NativeStdin::Null,
    )
}

enum NativeStdin<'a> {
    Null,
    Bytes(&'a [u8]),
    #[cfg(unix)]
    Closed,
}

fn compile_and_run_native_impl(
    source_name: &str,
    source_code: &str,
    sanitize_address: bool,
    done_continuation: DoneContinuationPolicy,
    paired_rc_elimination: PairedRcEliminationPolicy,
    borrowed_parameters: BorrowedParameterPolicy,
    stdin: NativeStdin<'_>,
) -> Output {
    use tribute::database::parse_with_thread_local;

    let source_rope = Rope::from_str(source_code);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_rope, None);
        let source_file = SourceCst::from_path(db, source_name, source_rope.clone(), tree);

        let optimizations = OptimizationOptions {
            ast_to_ir: AstToIrOptions { done_continuation },
            native: NativeOptimizationOptions {
                paired_rc_elimination,
                borrowed_parameters,
            },
        };
        let object_bytes =
            compile_native_or_panic_with_options(db, source_file, sanitize_address, optimizations);

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
        let mut command = Command::new(&exec_path);
        match stdin {
            NativeStdin::Bytes(input) => {
                let mut child = command
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap_or_else(|e| panic!("Failed to execute native binary: {e}"));
                let mut child_stdin = child.stdin.take().expect("piped stdin");
                let input = input.to_vec();
                let writer = std::thread::spawn(move || child_stdin.write_all(&input));
                let output = child.wait_with_output().expect("wait for native binary");
                writer
                    .join()
                    .expect("native stdin writer panicked")
                    .expect("write native stdin");
                output
            }
            #[cfg(unix)]
            NativeStdin::Closed => {
                use std::os::unix::process::CommandExt;

                unsafe {
                    command.pre_exec(|| {
                        if close(0) == 0 {
                            Ok(())
                        } else {
                            Err(std::io::Error::last_os_error())
                        }
                    });
                }
                command
                    .output()
                    .unwrap_or_else(|e| panic!("Failed to execute native binary: {e}"))
            }
            NativeStdin::Null => command
                .output()
                .unwrap_or_else(|e| panic!("Failed to execute native binary: {e}")),
        }
    })
}
