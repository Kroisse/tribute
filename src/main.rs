//! Tribute compiler CLI entry point.

mod cli;
mod diagnostics;
mod lsp;

use clap::Parser;
use cli::{Cli, Command};
use diagnostics::print_diagnostic;
use ropey::Rope;
use salsa::Database;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{
    compile_to_native_binary, compile_to_wasm_binary, compile_with_diagnostics,
};
use tribute::{SourceCst, TributeDatabaseImpl};
use tribute_front::query::parsed_ast;
use tribute_front::resolve::build_env;
use tribute_passes::diagnostic::Diagnostic;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve => {
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
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
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
            "native" => {
                println!("Compiling {} to native executable...", input_path.display());

                if let Some(object_bytes) = compile_to_native_binary(db, source) {
                    // Write object file to a temp file
                    let obj_file = tempfile::Builder::new()
                        .suffix(".o")
                        .tempfile()
                        .unwrap_or_else(|e| {
                            eprintln!("Error creating temp file: {}", e);
                            std::process::exit(1);
                        });
                    std::fs::write(obj_file.path(), &object_bytes).unwrap_or_else(|e| {
                        eprintln!("Error writing object file: {}", e);
                        std::process::exit(1);
                    });

                    // Determine output path
                    let output = output_path.unwrap_or_else(|| input_path.with_extension(""));

                    // Link with system cc
                    let status = std::process::Command::new("cc")
                        .arg(obj_file.path())
                        .arg("-o")
                        .arg(&output)
                        .status();

                    match status {
                        Ok(s) if s.success() => {
                            println!(
                                "Successfully compiled native executable: {}",
                                output.display()
                            );
                        }
                        Ok(s) => {
                            eprintln!("Linker failed with exit code: {}", s.code().unwrap_or(-1));
                            std::process::exit(1);
                        }
                        Err(e) => {
                            eprintln!("Failed to run linker (cc): {}", e);
                            eprintln!("Make sure a C compiler is installed and available in PATH.");
                            std::process::exit(1);
                        }
                    }
                } else {
                    let file_path = input_path.display().to_string();
                    let source_text = source.text(db);

                    let native_diags: Vec<_> =
                        compile_to_native_binary::accumulated::<Diagnostic>(db, source);
                    if !native_diags.is_empty() {
                        for diag in &native_diags {
                            print_diagnostic(diag, source_text, &file_path);
                        }
                    } else {
                        let result = compile_with_diagnostics(db, source);
                        if !result.diagnostics.is_empty() {
                            for diag in &result.diagnostics {
                                print_diagnostic(diag, source_text, &file_path);
                            }
                        } else {
                            eprintln!("Native compilation failed (unknown error)");
                        }
                    }
                    std::process::exit(1);
                }
            }
            "wasm" => {
                println!("Compiling {} to WebAssembly...", input_path.display());

                if let Some(wasm_binary) = compile_to_wasm_binary(db, source) {
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
                        compile_to_wasm_binary::accumulated::<Diagnostic>(db, source);
                    if !wasm_diags.is_empty() {
                        for diag in &wasm_diags {
                            print_diagnostic(diag, source_text, &file_path);
                        }
                    } else {
                        // Try getting frontend diagnostics
                        let result = compile_with_diagnostics(db, source);
                        if !result.diagnostics.is_empty() {
                            for diag in &result.diagnostics {
                                print_diagnostic(diag, source_text, &file_path);
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
                    for diag in &result.diagnostics {
                        print_diagnostic(diag, source_text, &file_path);
                    }
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!(
                    "Unknown target: {}. Use 'native', 'wasm', or 'none'",
                    target
                );
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
            if let Some(parsed) = parsed_ast(db, source) {
                let env = build_env(db, &parsed.module(db));
                println!("{:#?}", env);
            } else {
                println!("(Failed to parse AST)");
            }
        }

        println!();
    });
}
