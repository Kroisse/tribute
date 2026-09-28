//! Development tasks for the Tribute workspace, run as `cargo xtask <task>`.

use clap::{Parser, Subcommand};
use std::error::Error;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Development tasks for the Tribute workspace.
#[derive(Parser)]
#[command(bin_name = "cargo xtask")]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Build tribute-runtime into the development sysroot (target/sysroot)
    Runtime,
}

fn main() -> ExitCode {
    let result = match Cli::parse().task {
        Task::Runtime => runtime(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Build the runtime staticlib and install it at
/// `target/sysroot/lib/tribute/<host-triple>/`, the layout the compiler
/// searches (see `new-plans/linking.md`).
fn runtime() -> Result<()> {
    let root = workspace_root()?;
    let built = build_runtime_staticlib(&root)?;

    let dest_dir = root
        .join("target/sysroot/lib/tribute")
        .join(target_lexicon::HOST.to_string());
    std::fs::create_dir_all(&dest_dir)?;
    let dest = dest_dir.join(runtime_library_name());
    std::fs::copy(&built, &dest)?;
    eprintln!("installed {}", dest.display());
    Ok(())
}

/// File name rustc gives the runtime staticlib on the host target; the
/// compiler's `tribute::link` looks the library up by the same name.
fn runtime_library_name() -> &'static str {
    if target_lexicon::HOST.environment == target_lexicon::Environment::Msvc {
        "tribute_runtime.lib"
    } else {
        "libtribute_runtime.a"
    }
}

fn workspace_root() -> Result<PathBuf> {
    // Cargo sets this for `cargo run`; `cargo xtask` is an alias for it.
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .ok_or("CARGO_MANIFEST_DIR is not set; run this through `cargo xtask`")?;
    Path::new(&manifest_dir)
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask manifest directory has no parent".into())
}

/// Build `tribute-runtime` as a staticlib with the `runtime` profile and
/// return the path of the produced archive.
fn build_runtime_staticlib(root: &Path) -> Result<PathBuf> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut child = Command::new(cargo)
        .current_dir(root)
        .args(["rustc", "--package", "tribute-runtime", "--lib"])
        .args(["--profile", "runtime", "--crate-type", "staticlib"])
        .arg("--message-format=json-render-diagnostics")
        .stdout(Stdio::piped())
        .spawn()?;

    let mut staticlib = None;
    let stdout = child.stdout.take().ok_or("cargo stdout was not captured")?;
    for line in BufReader::new(stdout).lines() {
        let message: serde_json::Value = serde_json::from_str(&line?)?;
        if message["reason"] != "compiler-artifact"
            || message["target"]["name"] != "tribute_runtime"
        {
            continue;
        }
        let filenames = message["filenames"].as_array().into_iter().flatten();
        if let Some(path) = filenames
            .filter_map(serde_json::Value::as_str)
            .find(|path| Path::new(path).file_name() == Some(runtime_library_name().as_ref()))
        {
            staticlib = Some(PathBuf::from(path));
        }
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(format!("building tribute-runtime failed: {status}").into());
    }
    staticlib.ok_or_else(|| "cargo reported no tribute-runtime staticlib".into())
}
