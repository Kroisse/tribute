//! Common test utilities for e2e tests.

use std::io::Write;
use std::process::Command;
use std::str::FromStr;
use tempfile::NamedTempFile;

/// Run WASM binary with wasmtime CLI and parse the result
#[allow(dead_code)]
pub fn run_wasm<T>(wasm_bytes: &[u8]) -> T
where
    T: FromStr,
    T::Err: std::fmt::Debug,
{
    let mut temp_file = NamedTempFile::with_suffix(".wasm").expect("Failed to create temp file");
    temp_file
        .write_all(wasm_bytes)
        .expect("Failed to write WASM");

    let output = Command::new("wasmtime")
        .args(["run", "--wasm", "gc"])
        .arg(temp_file.path())
        .output()
        .expect("Failed to execute wasmtime");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("wasmtime execution failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse().expect("Failed to parse result")
}
