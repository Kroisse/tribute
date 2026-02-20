//! Tribute compiler CLI entry point.

mod cli;
mod diagnostics;
mod lsp;

use clap::Parser;
use cli::{Cli, Command};
use diagnostics::{print_diagnostic, report_diagnostics};
use ropey::Rope;
use salsa::Database;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{
    compile_to_native_binary, compile_to_wasm_binary, compile_with_diagnostics, link_native_binary,
    run_native_pipeline, run_wasm_pipeline,
};
use tribute::{SourceCst, TributeDatabaseImpl};
use tribute_front::query::parsed_ast;
use tribute_front::resolve::build_env;
use tribute_passes::{Diagnostic, DiagnosticSeverity};
use trunk_ir::DialectOp;

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
            dump_ir,
        } => {
            init_tracing(&cli.log);
            compile_file(file, output, &target, dump_ir);
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

fn compile_file(input_path: PathBuf, output_path: Option<PathBuf>, target: &str, dump_ir: bool) {
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

        if dump_ir {
            let result = match target {
                "native" => run_native_pipeline(db, source),
                _ => run_wasm_pipeline(db, source),
            };

            // Collect diagnostics from the target-specific pipeline
            let diagnostics: Vec<Diagnostic> = match target {
                "native" => run_native_pipeline::accumulated::<Diagnostic>(db, source),
                _ => run_wasm_pipeline::accumulated::<Diagnostic>(db, source),
            }
            .into_iter()
            .cloned()
            .collect();

            if !diagnostics.is_empty() {
                let file_path = input_path.display().to_string();
                let source_text = source.text(db);
                for diag in &diagnostics {
                    print_diagnostic(diag, source_text, &file_path);
                }
                if diagnostics
                    .iter()
                    .any(|d| d.severity == DiagnosticSeverity::Error)
                {
                    std::process::exit(1);
                }
            }

            match result {
                Ok(module) => {
                    let ir_text = trunk_ir::printer::print_op(db, module.as_operation());
                    println!("{}", ir_text);
                }
                Err(e) => {
                    eprintln!("Pipeline failed: {e}");
                    std::process::exit(1);
                }
            }
            return;
        }

        match target {
            "native" => {
                println!("Compiling {} to native executable...", input_path.display());

                if let Some(object_bytes) = compile_to_native_binary(db, source) {
                    let output = output_path.unwrap_or_else(|| input_path.with_extension(""));
                    if let Err(e) = link_native_binary(&object_bytes, &output) {
                        eprintln!("Linking failed: {e}");
                        std::process::exit(1);
                    }
                    println!(
                        "Successfully compiled native executable: {}",
                        output.display()
                    );
                } else {
                    let file_path = input_path.display().to_string();
                    let diags = compile_to_native_binary::accumulated::<Diagnostic>(db, source);
                    report_diagnostics(db, source, &file_path, diags);
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
                                "Successfully wrote WebAssembly binary to {}",
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
                    let diags = compile_to_wasm_binary::accumulated::<Diagnostic>(db, source);
                    report_diagnostics(db, source, &file_path, diags);
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
