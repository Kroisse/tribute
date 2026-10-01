//! The benchmarks skip programs that fail to compile, so these tests pin
//! which programs each target is expected to compile and run.

use std::process::Command;

use tribute_bench::programs::{PROGRAMS, WasmSupport};
use tribute_bench::stages;
use tribute_passes::abi_boundary::TargetKind;

#[test]
fn every_program_compiles_and_runs_natively() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let failures: Vec<String> = PROGRAMS
        .iter()
        .filter_map(|program| {
            let executable = dir.path().join(program.name);
            let result = stages::emitted(program, TargetKind::Native)
                .and_then(|object| stages::link(&object, &executable))
                .and_then(|()| {
                    stages::run(&mut Command::new(&executable), program.stdin)
                        .map_err(|error| error.to_string())
                })
                .and_then(|output| {
                    output.status.success().then_some(()).ok_or_else(|| {
                        format!(
                            "exited with {}: {}",
                            output.status,
                            String::from_utf8_lossy(&output.stderr)
                        )
                    })
                });
            result
                .err()
                .map(|error| format!("{}: {error}", program.name))
        })
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn wasm_support_matches_each_program() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let failures: Vec<String> = PROGRAMS
        .iter()
        .filter_map(|program| {
            let outcome = stages::emitted(program, TargetKind::Wasm)
                .and_then(|bytes| run_wasm(&bytes, &dir, program.name));
            let result = match (program.wasm, outcome) {
                (WasmSupport::Supported, outcome) => outcome,
                (WasmSupport::Unsupported(reason), Ok(())) => Err(format!(
                    "now runs on Wasm; mark it supported (expected failure: {reason})"
                )),
                (WasmSupport::Unsupported(_), Err(_)) => Ok(()),
            };
            result
                .err()
                .map(|error| format!("{}: {error}", program.name))
        })
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

fn run_wasm(bytes: &[u8], dir: &tempfile::TempDir, name: &str) -> Result<(), String> {
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(bytes)
        .map_err(|error| format!("invalid Wasm module: {error}"))?;
    let module = dir.path().join(format!("{name}.wasm"));
    std::fs::write(&module, bytes).map_err(|error| error.to_string())?;
    let output = Command::new("wasmtime")
        .arg("-Wgc=y,function-references=y")
        .arg(&module)
        .output()
        .map_err(|error| format!("wasmtime did not start: {error}"))?;
    output.status.success().then_some(()).ok_or_else(|| {
        format!(
            "wasmtime exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    })
}
