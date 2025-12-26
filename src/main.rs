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
use tribute::{SourceCst, TributeDatabaseImpl};
use tribute_passes::diagnostic::Diagnostic;
use tribute_passes::resolve::build_env;

/// Format a diagnostic with source location and context.
fn format_diagnostic(diag: &Diagnostic, source: &Rope, file_path: &str) -> String {
    let start = diag.span.start;
    let end = diag.span.end;

    // Convert byte offset to line/column (0-indexed)
    let line = source.byte_to_line(start.min(source.len_bytes().saturating_sub(1)));
    let line_start = source.line_to_byte(line);
    let col = start.saturating_sub(line_start);

    // Get the source line
    let line_text = source.line(line).to_string();
    let line_text = line_text.trim_end(); // Remove trailing newline

    // Calculate end column for underline
    let end_col = if end > start {
        let end_line = source.byte_to_line(end.min(source.len_bytes().saturating_sub(1)));
        if end_line == line {
            end.saturating_sub(line_start)
        } else {
            line_text.len()
        }
    } else {
        col + 1
    };

    // Build the underline with carets
    let underline_len = end_col.saturating_sub(col).max(1);
    let underline = format!("{}{}", " ".repeat(col), "^".repeat(underline_len));

    format!(
        "error[{:?}]: {}\n  --> {}:{}:{}\n   |\n{:>3}| {}\n   | {}",
        diag.phase,
        diag.message,
        file_path,
        line + 1, // 1-indexed for display
        col + 1,  // 1-indexed for display
        line + 1,
        line_text,
        underline
    )
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve => {
            // LSP server initializes its own tracing with LspLayer
            if let Err(e) = lsp::serve(&cli.log) {
                eprintln!("LSP server error: {e}");
                std::process::exit(1);
            }
        }
        Command::Compile {
            file,
            output,
            target,
        } => {
            init_tracing(&cli.log);
            compile_file(file, output, &target);
        }
        Command::Debug { file, show_env } => {
            init_tracing(&cli.log);
            debug_file(file, show_env);
        }
    }
}

fn init_tracing(log_filter: &str) {
    let env_filter = EnvFilter::try_new(log_filter).unwrap_or_else(|e| {
        eprintln!("Invalid log filter '{}': {}", log_filter, e);
        EnvFilter::new("warn")
    });
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .init();
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
                    let file_path = input_path.display().to_string();
                    let source_text = source.text(db);

                    // Collect diagnostics from wasm lowering
                    let wasm_diags: Vec<_> =
                        stage_lower_to_wasm::accumulated::<Diagnostic>(db, source);
                    if !wasm_diags.is_empty() {
                        eprintln!();
                        for diag in &wasm_diags {
                            eprintln!("{}\n", format_diagnostic(diag, source_text, &file_path));
                        }
                    } else {
                        // Try getting frontend diagnostics
                        let result = compile_with_diagnostics(db, source);
                        if !result.diagnostics.is_empty() {
                            eprintln!();
                            for diag in &result.diagnostics {
                                eprintln!("{}\n", format_diagnostic(diag, source_text, &file_path));
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
                    let file_path = input_path.display().to_string();
                    let source_text = source.text(db);
                    eprintln!();
                    for diag in &result.diagnostics {
                        eprintln!("{}\n", format_diagnostic(diag, source_text, &file_path));
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
