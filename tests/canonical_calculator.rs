//! Product-level regression coverage for the canonical native calculator.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

const CALCULATOR_SOURCE: &str = include_str!("../lang-examples/native_calculator.trb");
const SCRIPTED_STDIN: &[u8] = include_bytes!("fixtures/native_calculator_scripted.stdin");
const SCRIPTED_STDOUT: &[u8] = include_bytes!("expected/native_calculator_scripted.stdout");
const EOF_STDIN: &[u8] = include_bytes!("fixtures/native_calculator_eof.stdin");
const EOF_STDOUT: &[u8] = include_bytes!("expected/native_calculator_eof.stdout");
const SENTINEL_LIKE_STDIN: &[u8] = b"\0eof\nquit\n";
const SENTINEL_LIKE_STDOUT: &[u8] = b"error: expected 'add|sub|mul <int> <int>' or 'quit'\n";
const INVALID_ENCODING_STDIN: &[u8] = b"\xff\n";
const INPUT_FAILURE_STDOUT: &[u8] = b"error: input failure\n";

fn compile_calculator(output: &Path, sanitize_address: bool) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tribute"));
    command.arg("compile");
    if sanitize_address {
        command.arg("--sanitize=address");
    }
    let output = command
        .arg("lang-examples/native_calculator.trb")
        .arg("-o")
        .arg(output)
        .output()
        .expect("invoke tribute compile");

    assert!(
        output.status.success(),
        "calculator compilation failed: exit={:?}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim(),
    );
}

fn execute(binary: &Path, stdin: &[u8]) -> Output {
    let mut child = Command::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run calculator binary");
    let mut child_stdin = child.stdin.take().expect("calculator stdin");
    let input = stdin.to_vec();
    let writer = std::thread::spawn(move || child_stdin.write_all(&input));
    let output = child.wait_with_output().expect("wait for calculator");
    writer
        .join()
        .expect("calculator stdin writer panicked")
        .expect("write calculator stdin");
    output
}

fn assert_fixture(binary: &Path, stdin: &[u8], expected_stdout: &[u8]) {
    let output = execute(binary, stdin);
    assert!(
        output.status.success(),
        "calculator execution failed: exit={:?}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim(),
    );
    assert_eq!(output.stdout, expected_stdout, "calculator stdout");
}

#[test]
fn canonical_calculator_cli_matches_all_input_contracts() {
    let temp_dir = tempfile::tempdir().expect("create calculator test directory");
    let binary = temp_dir.path().join("native-calculator");
    compile_calculator(&binary, false);

    assert_fixture(&binary, SCRIPTED_STDIN, SCRIPTED_STDOUT);
    assert_fixture(&binary, EOF_STDIN, EOF_STDOUT);
    assert_fixture(&binary, SENTINEL_LIKE_STDIN, SENTINEL_LIKE_STDOUT);
    assert_fixture(&binary, INVALID_ENCODING_STDIN, INPUT_FAILURE_STDOUT);
}

#[test]
fn canonical_calculator_uses_only_public_source_apis() {
    for marker in ["__tribute_", "extern "] {
        assert!(
            !CALCULATOR_SOURCE.contains(marker),
            "canonical calculator must not contain private marker {marker:?}",
        );
    }
}

#[test]
fn canonical_calculator_asan_matches_scripted_contract() {
    let temp_dir = tempfile::tempdir().expect("create calculator ASan test directory");
    let binary = temp_dir.path().join("native-calculator-asan");
    compile_calculator(&binary, true);

    assert_fixture(&binary, SCRIPTED_STDIN, SCRIPTED_STDOUT);
}
