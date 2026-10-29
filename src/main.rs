//! Tribute compiler CLI entry point.

mod cli;
mod diagnostics;
mod lsp;

use clap::Parser;
use cli::{Cli, Command};
use diagnostics::{print_diagnostic, print_diagnostics, report_diagnostics};
use salsa::Database;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;
use tribute::link::link_native_binary;
use tribute::pipeline::{
    CompilationConfig, compile_to_native_binary, compile_to_wasm_binary, compile_with_diagnostics,
    wasm_binary_diagnostics,
};
use tribute::{TributeDatabaseImpl, check_library};
use tribute_core::diagnostic::{Diagnostic, DiagnosticSeverity};
use tribute_front::manifest::load_package;
use tribute_front::resolve::{Dependencies, build_env};
use tribute_front::source_tree::source_tree_of;

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
            sanitize,
            sysroot,
        } => {
            init_tracing(&cli.log);
            let sanitize_address = sanitize.as_deref() == Some("address");
            compile_file(
                file,
                output,
                &target,
                dump_ir,
                sanitize_address,
                sysroot.as_deref(),
            );
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

fn compile_file(
    input_path: PathBuf,
    output_path: Option<PathBuf>,
    target: &str,
    dump_ir: bool,
    sanitize_address: bool,
    sysroot: Option<&Path>,
) {
    let db = TributeDatabaseImpl::default();
    // The default outputs of the native and the wasm target.
    let (source, input_path, (default_output, default_wasm)) = if input_path.is_dir() {
        match load_package_dir(&db, &input_path, target) {
            PackageInput::Executable { root, output } => {
                let wasm = output.with_extension("wasm");
                (root, input_path.join("src/main.trb"), (output, wasm))
            }
            PackageInput::Checked => return,
        }
    } else {
        let source = match db.input(input_path.clone()) {
            Ok(source) => source,
            Err(e) => {
                eprintln!("Error reading file {}: {}", input_path.display(), e);
                std::process::exit(1);
            }
        };
        let outputs = (
            input_path.with_extension(""),
            input_path.with_extension("wasm"),
        );
        (source, input_path, outputs)
    };

    db.attach(|db| {
        if dump_ir {
            if sanitize_address {
                eprintln!("warning: --sanitize has no effect with --dump-ir (sanitizer ops are added during backend lowering)");
            }
            let is_native = target == "native";
            let result = tribute::pipeline::dump_ir(db, source, is_native);

            // Collect diagnostics
            let diagnostics: Vec<Diagnostic> =
                tribute::pipeline::dump_ir::accumulated::<Diagnostic>(db, source, is_native)
                    .into_iter()
                    .cloned()
                    .collect();

            if !diagnostics.is_empty() {
                let file_path = input_path.display().to_string();
                for diag in &diagnostics {
                    print_diagnostic(db, source, &file_path, diag);
                }
                if diagnostics
                    .iter()
                    .any(|d| d.inner.severity == DiagnosticSeverity::Error)
                {
                    std::process::exit(1);
                }
            }

            match result {
                Ok(ir_text) => {
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

                let config = CompilationConfig::new(
                    db,
                    sanitize_address,
                    tribute::pipeline::OptimizationOptions::production(),
                );
                if let Some(object_bytes) = compile_to_native_binary(db, source, config) {
                    let mut diags =
                        compile_to_native_binary::accumulated::<Diagnostic>(db, source, config);
                    print_diagnostics(db, source, &input_path.display().to_string(), &mut diags);
                    let output = output_path.unwrap_or(default_output);
                    if let Err(e) = link_native_binary(object_bytes, &output, sysroot) {
                        eprintln!("Linking failed: {e}");
                        std::process::exit(1);
                    }
                    println!(
                        "Successfully compiled native executable: {}",
                        output.display()
                    );
                } else {
                    let file_path = input_path.display().to_string();
                    let diags =
                        compile_to_native_binary::accumulated::<Diagnostic>(db, source, config);
                    report_diagnostics(db, source, &file_path, diags);
                    std::process::exit(1);
                }
            }
            "wasm" => {
                println!("Compiling {} to WebAssembly...", input_path.display());

                match compile_to_wasm_binary(db, source) {
                    Ok(wasm_bytes) => {
                        let mut diags = wasm_binary_diagnostics(db, source);
                        print_diagnostics(
                            db,
                            source,
                            &input_path.display().to_string(),
                            &mut diags,
                        );
                        let output =
                            output_path.unwrap_or(default_wasm);

                        if let Err(error) = std::fs::write(&output, wasm_bytes) {
                            eprintln!(
                                "Error writing output file {}: {}",
                                output.display(),
                                error
                            );
                            std::process::exit(1);
                        }

                        println!(
                            "Successfully wrote WebAssembly binary to {}",
                            output.display()
                        );
                    }
                    Err(diagnostics) => {
                        let file_path = input_path.display().to_string();
                        report_diagnostics(db, source, &file_path, diagnostics);
                        std::process::exit(1);
                    }
                }
            }
            "none" => {
                println!("Compiling {}...", input_path.display());
                let result = compile_with_diagnostics(db, source);

                let file_path = input_path.display().to_string();
                for diag in &result.diagnostics {
                    print_diagnostic(db, source, &file_path, diag);
                }
                if result
                    .diagnostics
                    .iter()
                    .any(|diag| diag.inner.severity == DiagnosticSeverity::Error)
                {
                    std::process::exit(1);
                }
                println!("✓ Compiled successfully");
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

/// What compiling a package directory continues with.
enum PackageInput {
    /// Build the executable whose root file is `root`, by default into
    /// `output`.
    Executable {
        root: tribute::SourceCst,
        output: PathBuf,
    },
    /// The package has no executable, and its library checked.
    Checked,
}

/// Load the package in `dir` from its manifest, reporting the manifest's
/// errors. A package with `src/main.trb` builds that executable; a package
/// with only `src/lib.trb` is checked as a library, for target `none` only.
fn load_package_dir(db: &TributeDatabaseImpl, dir: &Path, target: &str) -> PackageInput {
    let manifest = match db.manifest(dir) {
        Ok(manifest) => manifest,
        Err(e) => {
            eprintln!("Error loading package {}: {e}", dir.display());
            std::process::exit(1);
        }
    };
    db.attach(|db| {
        let implicit = *tribute::frontend_environment(db);
        let source = manifest.source(db).expect("a manifest that exists");
        let manifest_path = dir.join(tribute_front::manifest::MANIFEST_FILE);
        let manifest_path = manifest_path.display().to_string();
        let loaded = load_package(db, manifest, implicit);
        let mut diags = load_package::accumulated::<Diagnostic>(db, manifest, implicit);
        let failed = diags
            .iter()
            .any(|diag| diag.inner.severity == DiagnosticSeverity::Error);
        print_diagnostics(db, source, &manifest_path, &mut diags);
        let loaded = loaded.as_ref().filter(|_| !failed);
        let Some(loaded) = loaded else {
            std::process::exit(1);
        };
        let name = loaded.name.as_ref().map_or("main", |name| name.as_str());
        if let Some(root) = loaded.executable {
            return PackageInput::Executable {
                root,
                output: dir.join(name),
            };
        }
        if target != "none" {
            eprintln!(
                "Error: package `{name}` has no `src/main.trb`; a library package builds no executable (use `--target none` to check it)"
            );
            std::process::exit(1);
        }
        let Some(library) = loaded.library else {
            std::process::exit(1);
        };
        println!("Checking library package `{name}`...");
        let mut checked = *check_library(db, library);
        let mut diags = check_library::accumulated::<Diagnostic>(db, library);
        checked &= !diags
            .iter()
            .any(|diag| diag.inner.severity == DiagnosticSeverity::Error);
        let root = library.root(db);
        let root_path = dir.join("src/lib.trb").display().to_string();
        print_diagnostics(db, *root, &root_path, &mut diags);
        if !checked {
            std::process::exit(1);
        }
        println!("✓ Checked successfully");
        PackageInput::Checked
    })
}

fn debug_file(path: std::path::PathBuf, show_env: bool) {
    let db = TributeDatabaseImpl::default();
    let source = match db.input(path.clone()) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("Error reading file: {e}");
            std::process::exit(1);
        }
    };

    db.attach(|db| {
        println!("=== Compiling: {} ===\n", path.display());

        let result = compile_with_diagnostics(db, source);

        // Show diagnostics
        if result.diagnostics.is_empty() {
            println!("✓ No errors");
        } else {
            println!("Diagnostics ({} total):", result.diagnostics.len());
            for diag in &result.diagnostics {
                println!("  [{:?}] {}", diag.phase, diag.inner.message);
            }
        }

        // Show environment if requested
        if show_env {
            println!("\n=== Package Environment ===");
            if let Some(tree) = source_tree_of(db, source) {
                let env = build_env(db, tree.module(db), Dependencies::none()).into_env();
                println!("{:#?}", env);
            } else {
                println!("(Failed to parse AST)");
            }
        }

        println!();
    });
}
