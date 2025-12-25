//! Tribute compiler CLI entry point.

mod cli;
mod lsp;

use clap::Parser;
use cli::{Cli, Command};
use ropey::Rope;
use salsa::Database;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{compile_with_diagnostics, stage_lower_to_wasm, stage_resolve};
use tribute_passes::diagnostic::Diagnostic;
use tribute::{SourceCst, TributeDatabaseImpl};
use tribute_passes::resolve::build_env;

fn main() {
    // Initialize tracing with env_logger-style filtering via RUST_LOG
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Serve => {
            if let Err(e) = lsp::serve() {
                eprintln!("LSP server error: {e}");
                std::process::exit(1);
            }
        }
        Command::Compile {
            file,
            output,
            target,
        } => {
            compile_file(file, output, &target);
        }
        Command::Debug { file, show_env } => {
            debug_file(file, show_env);
        }
    }
}

fn compile_file(input_path: PathBuf, output_path: Option<PathBuf>, target: &str) {
    let source_code = {
        match std::fs::File::open(&input_path).and_then(Rope::from_reader) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading file {}: {}", input_path.display(), e);
                std::process::exit(1);
            }
        }
    };

    TributeDatabaseImpl::default().attach(|db| {
        let tree = parse_with_thread_local(&source_code, None);
        let source = SourceCst::from_path(db, &input_path, source_code, tree);

        match target {
            "wasm" => {
                println!("Compiling {} to WebAssembly...", input_path.display());

                if let Some(wasm_binary) = stage_lower_to_wasm(db, source) {
                    let bytes = wasm_binary.bytes(db);
                    let output = output_path.unwrap_or_else(|| input_path.with_extension("wasm"));

                    match std::fs::write(&output, bytes) {
                        Ok(_) => {
                            println!(
                                "✓ Successfully wrote WebAssembly binary to {}",
                                output.display()
                            );
                        }
                        Err(e) => {
                            eprintln!("Error writing output file {}: {}", output.display(), e);
                            std::process::exit(1);
                        }
                    }
                } else {
                    // Collect diagnostics from wasm lowering
                    let wasm_diags: Vec<_> = stage_lower_to_wasm::accumulated::<Diagnostic>(db, source);
                    if !wasm_diags.is_empty() {
                        println!("WebAssembly compilation errors:");
                        for diag in &wasm_diags {
                            println!("  [{:?}] {}", diag.phase, diag.message);
                        }
                    } else {
                        // Try getting frontend diagnostics
                        let result = compile_with_diagnostics(db, source);
                        if !result.diagnostics.is_empty() {
                            println!("Diagnostics ({} total):", result.diagnostics.len());
                            for diag in &result.diagnostics {
                                println!("  [{:?}] {}", diag.phase, diag.message);
                            }
                        } else {
                            eprintln!("✗ WebAssembly compilation failed (unknown error)");
                        }
                    }
                    std::process::exit(1);
                }
            }
            "none" => {
                println!("Compiling {}...", input_path.display());
                let result = compile_with_diagnostics(db, source);

                if result.diagnostics.is_empty() {
                    println!("✓ Compiled successfully");
                } else {
                    println!("Diagnostics ({} total):", result.diagnostics.len());
                    for diag in &result.diagnostics {
                        println!("  [{:?}] {}", diag.phase, diag.message);
                    }
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown target: {}. Use 'wasm' or 'none'", target);
                std::process::exit(1);
            }
        }
    });
}

fn debug_file(path: std::path::PathBuf, show_env: bool) {
    let source_code = {
        match std::fs::File::open(&path).and_then(Rope::from_reader) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading file: {e}");
                std::process::exit(1);
            }
        }
    };

    TributeDatabaseImpl::default().attach(|db| {
        println!("=== Compiling: {} ===\n", path.display());

        let tree = parse_with_thread_local(&source_code, None);
        let source = SourceCst::from_path(db, &path, source_code, tree);
        let result = compile_with_diagnostics(db, source);

        // Show diagnostics
        if result.diagnostics.is_empty() {
            println!("✓ No errors");
        } else {
            println!("Diagnostics ({} total):", result.diagnostics.len());
            for diag in &result.diagnostics {
                println!("  [{:?}] {}", diag.phase, diag.message);
            }
        }

        // Show environment if requested
        if show_env {
            println!("\n=== Module Environment ===");
            let resolved = stage_resolve(db, source);
            let env = build_env(db, &resolved);
            println!("{:#?}", env);
        }

        println!();
    });
}
