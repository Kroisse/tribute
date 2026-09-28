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

/// Build the runtime staticlib and install it into the development sysroot
/// `target/sysroot`, at the path `tribute_sysroot` gives the compiler.
fn runtime() -> Result<()> {
    let root = workspace_root()?;
    let built = build_runtime_staticlib(&root)?;

    let dest = tribute_sysroot::runtime_library_path(&root.join("target/sysroot"));
    std::fs::create_dir_all(dest.parent().ok_or("sysroot runtime path has no parent")?)?;
    std::fs::copy(&built, &dest)?;
    eprintln!("installed {}", dest.display());
    Ok(())
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
            .find(|path| {
                Path::new(path).file_name()
                    == Some(tribute_sysroot::runtime_library_name().as_ref())
            })
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
