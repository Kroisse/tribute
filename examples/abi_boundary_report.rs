//! Allocation, code size, and execution report for the representation/ABI
//! boundary benchmark programs.
//!
//! ```text
//! cargo run --release --example abi_boundary_report > report.json
//! ```
//!
//! Code sizes are deterministic. Allocation counts vary slightly between runs
//! (typically well under 1%), so one run per program suffices for comparing
//! commits. Compile and run times are measured by `cargo bench --bench
//! abi_boundary`. Native execution needs the development sysroot (`cargo
//! xtask runtime`, found through the `TRIBUTE_SYSROOT` that Cargo sets);
//! Wasm execution needs `wasmtime` on `PATH`. Each is reported as skipped
//! when unavailable.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use tribute_passes::abi_boundary::TargetKind;

#[path = "../benches/support/programs.rs"]
mod programs;
// The benchmark uses the stage helpers this report does not.
#[allow(dead_code)]
#[path = "../benches/support/stages.rs"]
mod stages;

use programs::{PROGRAMS, Program};
use stages::{TARGETS, target_name};

/// Counts allocations and allocated bytes; a reallocation counts as one
/// allocation of its new size.
struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn record(size: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

/// Allocations made while running `stage`.
fn counted<T>(stage: impl FnOnce() -> T) -> (T, Value) {
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let result = stage();
    let counts = json!({
        "allocations": ALLOCATIONS.load(Ordering::Relaxed) - allocations,
        "bytes": ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes,
    });
    (result, counts)
}

fn main() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let programs: Vec<Value> = PROGRAMS
        .iter()
        .map(|program| report_program(program, dir.path()))
        .collect();
    let report = json!({
        "commit": commit(),
        "programs": programs,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
}

fn report_program(program: &Program, dir: &std::path::Path) -> Value {
    let (frontend, frontend_allocs) = counted(|| stages::frontend(program));
    let frontend = match frontend {
        Ok(frontend) => frontend,
        Err(error) => return json!({ "name": program.name, "error": error }),
    };
    let (shared, shared_allocs) = counted(|| stages::shared_middle_end(frontend));
    if let Err(error) = shared {
        return json!({ "name": program.name, "error": error });
    }
    let targets: serde_json::Map<String, Value> = TARGETS
        .into_iter()
        .map(|target| {
            (
                target_name(target).to_owned(),
                report_target(program, target, dir),
            )
        })
        .collect();
    json!({
        "name": program.name,
        "allocations": {
            "frontend": frontend_allocs,
            "shared_middle_end": shared_allocs,
        },
        "targets": targets,
    })
}

fn report_target(program: &Program, target: TargetKind, dir: &std::path::Path) -> Value {
    let (mut ctx, module) = stages::through_shared(program).expect("shared middle-end");
    let (exit, to_exit_allocs) = counted(|| stages::to_boundary_exit(&mut ctx, module, target));
    if let Err(error) = exit {
        return json!({ "error": error });
    }
    let (emitted, after_exit_allocs) =
        counted(|| stages::after_boundary_exit(&mut ctx, module, target));
    drop(ctx);
    let allocations = json!({
        "to_boundary_exit": to_exit_allocs,
        "after_boundary_exit": after_exit_allocs,
    });
    let bytes = match emitted {
        Ok(bytes) => bytes,
        Err(error) => return json!({ "allocations": allocations, "error": error }),
    };
    match target {
        TargetKind::Native => {
            let executable = dir.join(program.name);
            let linked = stages::link(&bytes, &executable).map(|()| {
                std::fs::metadata(&executable)
                    .expect("linked executable")
                    .len()
            });
            let (executable_bytes, run) = match linked {
                Ok(size) => (
                    json!(size),
                    execution(stages::run(&mut Command::new(&executable), program.stdin)),
                ),
                Err(error) => (Value::Null, json!({ "skipped": error })),
            };
            json!({
                "allocations": allocations,
                "object_bytes": bytes.len(),
                "executable_bytes": executable_bytes,
                "run": run,
            })
        }
        TargetKind::Wasm => {
            let validation =
                wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
                    .validate_all(&bytes)
                    .map(|_| "valid".to_owned())
                    .unwrap_or_else(|error| error.to_string());
            let module_path = dir.join(format!("{}.wasm", program.name));
            std::fs::write(&module_path, &bytes).expect("write Wasm module");
            let mut wasmtime = Command::new("wasmtime");
            wasmtime
                .arg("-Wgc=y,function-references=y")
                .arg(&module_path);
            let run = match stages::run(&mut wasmtime, program.stdin) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    json!({ "skipped": "wasmtime not found" })
                }
                result => execution(result),
            };
            json!({
                "allocations": allocations,
                "module_bytes": bytes.len(),
                "validation": validation,
                "run": run,
            })
        }
    }
}

fn execution(result: std::io::Result<std::process::Output>) -> Value {
    match result {
        Ok(output) => json!({
            "exit_code": output.status.code(),
            "stdout_bytes": output.stdout.len(),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }),
        Err(error) => json!({ "error": error.to_string() }),
    }
}

fn commit() -> Value {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| json!(String::from_utf8_lossy(&output.stdout).trim()))
        .unwrap_or(Value::Null)
}
