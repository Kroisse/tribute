use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Build tribute-runtime as a lean no_std staticlib using the `runtime` profile.
    // Uses a dedicated target directory to avoid Cargo lock contention
    // (the parent cargo process holds a lock on the main target directory).
    let runtime_target_dir = out_dir.join("runtime-target");
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());

    let status = Command::new(&cargo)
        .arg("rustc")
        .args(["-p", "tribute-runtime"])
        .arg("--lib")
        .args(["--profile", "runtime"])
        .args(["--crate-type", "staticlib"])
        .args(["--target-dir", runtime_target_dir.to_str().unwrap()])
        .current_dir(&manifest_dir)
        // Strip coverage/instrumentation flags from the environment so that
        // the runtime staticlib is built without instrumentation.
        //
        // `cargo-llvm-cov` injects `-C instrument-coverage` via RUSTC_WRAPPER
        // (and possibly RUSTFLAGS). The runtime is a no_std staticlib and
        // should not carry profiling instrumentation.
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .status()
        .expect("failed to invoke cargo to build tribute-runtime");

    assert!(
        status.success(),
        "failed to build tribute-runtime staticlib"
    );

    // The staticlib is at {target_dir}/runtime/libtribute_runtime.a
    // It bundles the Rust runtime code as a native static lib.
    let static_lib_dir = runtime_target_dir.join("runtime");
    println!(
        "cargo:rustc-env=TRIBUTE_RUNTIME_STATIC_LIB_DIR={}",
        static_lib_dir.display()
    );

    // Rerun when tribute-runtime sources change.
    println!("cargo:rerun-if-changed=crates/tribute-runtime/src");
    println!("cargo:rerun-if-changed=crates/tribute-runtime/Cargo.toml");

    // Rerun when instrumentation-related env vars change (e.g., switching
    // between coverage and normal builds) so the cached staticlib doesn't go stale.
    println!("cargo:rerun-if-env-changed=RUSTC_WRAPPER");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
