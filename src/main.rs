//! Tribute compiler CLI entry point.

mod cli;
mod lsp;

use clap::Parser;
use cli::{Cli, Command};
use ropey::Rope;
use salsa::Database;
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{compile_with_diagnostics, stage_resolve};
use tribute::{SourceCst, TributeDatabaseImpl};
use tribute_passes::resolve::build_env;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve => {
            if let Err(e) = lsp::serve() {
                eprintln!("LSP server error: {e}");
                std::process::exit(1);
            }
        }
        Command::Debug { file, show_env } => {
            debug_file(file, show_env);
        }
    }
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
