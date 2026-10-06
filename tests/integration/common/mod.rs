//! Common test utilities for e2e tests.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use ropey::Rope;
use salsa::Database;
use tribute::TributeDatabaseImpl;
use tribute::link::link_native_binary;
use tribute::pipeline::{
    BorrowedParameterPolicy, CompilationConfig, NativeOptimizationOptions, OptimizationOptions,
    PairedRcEliminationPolicy, TemporaryBorrowPolicy, compile_to_native_binary,
};
use tribute_core::diagnostic::Diagnostic;
use tribute_front::SourceCst;

#[cfg(unix)]
unsafe extern "C" {
    fn close(fd: i32) -> i32;
}

/// Compile source to a native object file, panicking with diagnostics on failure.
#[allow(dead_code)]
pub fn compile_native_or_panic(db: &dyn salsa::Database, source_file: SourceCst) -> &[u8] {
    compile_native_or_panic_with(db, source_file, false)
}

/// Compile source to a native object file with optional ASan, panicking with diagnostics on failure.
#[allow(dead_code)]
pub fn compile_native_or_panic_with(
    db: &dyn salsa::Database,
    source_file: SourceCst,
    sanitize_address: bool,
) -> &[u8] {
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
) -> &[u8] {
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
        NativeTestOptimizations::production(),
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
        NativeTestOptimizations {
            paired_rc_elimination: policy,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
        },
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
        NativeTestOptimizations {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: policy,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
        },
        NativeStdin::Null,
    )
}

/// Compile and run with explicit temporary field-borrow selection.
#[allow(dead_code)]
pub fn compile_and_run_native_with_temporary_borrows(
    source_name: &str,
    source_code: &str,
    policy: TemporaryBorrowPolicy,
    sanitize_address: bool,
) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        sanitize_address,
        NativeTestOptimizations {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows: policy,
        },
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
    compile_native_test_binary(source_name, source_code, NativeTestProfile::Production)
        .run_with_stdin(stdin)
}

/// Run with the baseline optimization profile and supplied stdin.
#[allow(dead_code)]
pub fn compile_and_run_native_with_stdin_baseline_optimizations(
    source_name: &str,
    source_code: &str,
    stdin: &[u8],
) -> Output {
    compile_native_test_binary(source_name, source_code, NativeTestProfile::Baseline)
        .run_with_stdin(stdin)
}

/// Run the native binary with ASan and supplied stdin.
#[allow(dead_code)]
pub fn compile_and_run_native_with_stdin_asan(
    source_name: &str,
    source_code: &str,
    stdin: &[u8],
) -> Output {
    compile_native_test_binary(source_name, source_code, NativeTestProfile::Asan)
        .run_with_stdin(stdin)
}

/// Compile and run Tribute source after closing native stdin.
#[cfg(unix)]
#[allow(dead_code)]
pub fn compile_and_run_native_with_closed_stdin(source_name: &str, source_code: &str) -> Output {
    compile_and_run_native_impl(
        source_name,
        source_code,
        false,
        NativeTestOptimizations::production(),
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
        NativeTestOptimizations::production(),
        NativeStdin::Null,
    )
}

pub enum NativeTestProfile {
    Production,
    Baseline,
    Asan,
}
pub fn compile_native_test_binary(
    source_name: &str,
    source_code: &str,
    profile: NativeTestProfile,
) -> NativeTestBinary {
    let (sanitize_address, test_optimizations) = match profile {
        NativeTestProfile::Production => (false, NativeTestOptimizations::production()),
        NativeTestProfile::Baseline => (false, NativeTestOptimizations::baseline()),
        NativeTestProfile::Asan => (true, NativeTestOptimizations::production()),
    };
    compile_native_test_binary_impl(
        source_name,
        source_code,
        sanitize_address,
        test_optimizations,
    )
}
pub struct NativeTestBinary {
    temp_dir: tempfile::TempDir,
}
impl NativeTestBinary {
    fn from_object_bytes(object_bytes: &[u8]) -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let exec_path = temp_dir.path().join("tribute_test_bin");
        link_native_binary(object_bytes, &exec_path, None).unwrap_or_else(|e| {
            panic!("Linking failed: {e}");
        });
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o755);
            std::fs::set_permissions(&exec_path, perms).expect("Failed to set permissions");
        }
        Self { temp_dir }
    }
    pub fn run_with_stdin(&self, input: &[u8]) -> Output {
        self.run(NativeStdin::Bytes(input))
    }
    fn run(&self, stdin: NativeStdin<'_>) -> Output {
        let mut command = Command::new(self.temp_dir.path().join("tribute_test_bin"));
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
    }
}
enum NativeStdin<'a> {
    Null,
    Bytes(&'a [u8]),
    #[cfg(unix)]
    Closed,
}
#[derive(Clone, Copy)]
struct NativeTestOptimizations {
    paired_rc_elimination: PairedRcEliminationPolicy,
    borrowed_parameters: BorrowedParameterPolicy,
    temporary_borrows: TemporaryBorrowPolicy,
}
impl NativeTestOptimizations {
    const fn baseline() -> Self {
        Self {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
        }
    }

    const fn production() -> Self {
        Self {
            paired_rc_elimination: PairedRcEliminationPolicy::Enabled,
            borrowed_parameters: BorrowedParameterPolicy::ElideProvenBorrowed,
            temporary_borrows: TemporaryBorrowPolicy::ElideProvenFieldBorrows,
        }
    }
}
fn compile_and_run_native_impl(
    source_name: &str,
    source_code: &str,
    sanitize_address: bool,
    test_optimizations: NativeTestOptimizations,
    stdin: NativeStdin<'_>,
) -> Output {
    let binary = compile_native_test_binary_impl(
        source_name,
        source_code,
        sanitize_address,
        test_optimizations,
    );
    binary.run(stdin)
}

fn compile_native_test_binary_impl(
    source_name: &str,
    source_code: &str,
    sanitize_address: bool,
    test_optimizations: NativeTestOptimizations,
) -> NativeTestBinary {
    use tribute::database::parse_with_thread_local;

    let source_rope = Rope::from_str(source_code);

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_rope, None);
        let source_file = SourceCst::from_path(db, source_name, source_rope.clone(), tree);

        let optimizations = OptimizationOptions {
            native: NativeOptimizationOptions {
                paired_rc_elimination: test_optimizations.paired_rc_elimination,
                borrowed_parameters: test_optimizations.borrowed_parameters,
                temporary_borrows: test_optimizations.temporary_borrows,
            },
        };
        let object_bytes =
            compile_native_or_panic_with_options(db, source_file, sanitize_address, optimizations);
        NativeTestBinary::from_object_bytes(object_bytes)
    })
}

/// Tribute source defining `print_nat`, which prints a `Nat` in decimal
/// followed by a newline on every target, unlike the native-only
/// `__tribute_print_nat` intrinsic.
#[allow(dead_code)]
pub const PRINT_NAT: &str = r#"
fn nat_text(n: Nat) -> String {
    let digit = case n % 10 {
        0 -> "0"
        1 -> "1"
        2 -> "2"
        3 -> "3"
        4 -> "4"
        5 -> "5"
        6 -> "6"
        7 -> "7"
        8 -> "8"
        _ -> "9"
    }
    case n < 10 {
        True -> digit
        False -> nat_text(n / 10) <> digit
    }
}

fn print_nat(n: Nat) ->{std::io::Io} Nil {
    std::io::print_line(nat_text(n))
}
"#;

/// Compile Tribute source code to a Wasm module and run it with wasmtime.
///
/// Panics if compilation fails or the module is not valid Wasm.
#[allow(dead_code)]
pub fn compile_and_run_wasm(source_name: &str, source_code: &str) -> Output {
    let binary = TributeDatabaseImpl::default().attach(|db| {
        let source = SourceCst::from_source_str(db, source_name, source_code);
        tribute::pipeline::compile_to_wasm_binary(db, source)
            .unwrap_or_else(|diagnostics| panic!("Wasm compilation failed: {diagnostics:?}"))
            .to_vec()
    });
    run_wasm(&binary)
}

/// Validate a Wasm module and run its start function with wasmtime.
#[allow(dead_code)]
pub fn run_wasm(binary: &[u8]) -> Output {
    run_wasm_impl(binary, None)
}

/// Validate a Wasm module and invoke one of its exports with wasmtime.
#[allow(dead_code)]
pub fn run_wasm_invoking(binary: &[u8], export: &str) -> Output {
    run_wasm_impl(binary, Some(export))
}

/// Run a Wasm module with wasmtime and assert that it succeeds and prints
/// exactly `expected_stdout`.
#[allow(dead_code)]
pub fn assert_wasm_output(binary: &[u8], expected_stdout: impl AsRef<[u8]>) {
    let output = run_wasm(binary);
    assert!(
        output.status.success(),
        "wasmtime failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected_stdout = expected_stdout.as_ref();
    assert_eq!(
        output.stdout,
        expected_stdout,
        "stdout mismatch:\n  actual: {:?}\nexpected: {:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(expected_stdout)
    );
}

fn run_wasm_impl(binary: &[u8], invoke: Option<&str>) -> Output {
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(binary)
        .expect("compiled source must produce a valid Wasm binary");
    let mut wasm = tempfile::NamedTempFile::new().expect("temporary Wasm file");
    wasm.write_all(binary).expect("write Wasm module");
    let mut command = Command::new("wasmtime");
    command.arg("-Wgc=y,function-references=y");
    if let Some(export) = invoke {
        command.arg("--invoke").arg(export);
    }
    command
        .arg(wasm.path())
        .output()
        .expect("run Wasm module with wasmtime")
}

/// Run a program on the native and Wasm targets and assert that both print
/// the expected output.
#[allow(dead_code)]
pub fn assert_output_on_both_targets(source_name: &str, source_code: &str, expected_stdout: &str) {
    assert_target_outputs(
        source_name,
        expected_stdout,
        compile_and_run_native(source_name, source_code),
        compile_and_run_wasm(source_name, source_code),
    );
}

/// Like [`assert_output_on_both_targets`], with the native binary built
/// under AddressSanitizer.
#[allow(dead_code)]
pub fn assert_output_on_both_targets_with_native_asan(
    source_name: &str,
    source_code: &str,
    expected_stdout: &str,
) {
    assert_target_outputs(
        source_name,
        expected_stdout,
        compile_and_run_native_asan(source_name, source_code),
        compile_and_run_wasm(source_name, source_code),
    );
}

fn assert_target_outputs(source_name: &str, expected_stdout: &str, native: Output, wasm: Output) {
    for (target, output) in [("native", native), ("wasm", wasm)] {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{target}: exit={:?}, stdout='{stdout}', stderr='{}'",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
        assert_eq!(
            stdout.trim(),
            expected_stdout,
            "{target}: stdout mismatch for {source_name}"
        );
    }
}
