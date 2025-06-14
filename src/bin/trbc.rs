//! Tribute compiler and interpreter
//!
//! This is a command-line tool that can both interpret and compile Tribute programs.
//!
//! # Usage
//!
//! ## Interpreter mode (default)
//! ```bash
//! trbc program.trb
//! ```
//!
//! ## Compiler mode
//! ```bash
//! trbc --compile program.trb -o output_binary
//! ```

use clap::{Arg, ArgAction, Command};
use std::path::PathBuf;
use tribute::{eval_str, parse_str, TributeDatabaseImpl};
use tribute_cranelift::TributeCompiler;
use tribute_hir::queries::lower_program_to_hir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("trbc")
        .version("0.1.0")
        .about("Tribute compiler and interpreter")
        .arg(
            Arg::new("input")
                .help("Input Tribute source file")
                .required(true)
                .value_name("FILE")
                .index(1),
        )
        .arg(
            Arg::new("compile")
                .long("compile")
                .short('c')
                .help("Compile to native binary instead of interpreting")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("Output file path (required when compiling)")
                .value_name("OUTPUT")
                .requires("compile"),
        )
        .get_matches();

    let input_path = PathBuf::from(matches.get_one::<String>("input").unwrap());
    let compile_mode = matches.get_flag("compile");

    if compile_mode {
        let output_path = matches.get_one::<String>("output")
            .ok_or("Output path is required when compiling")?;
        compile_program(&input_path, output_path)?;
    } else {
        // Interpreter mode (default)
        interpret_program(&input_path)?;
    }

    Ok(())
}

/// Interprets a Tribute program
fn interpret_program(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)?;
    let db = TributeDatabaseImpl::default();

    match eval_str(&db, path, &source) {
        Ok(result) => {
            // Only print non-unit results
            match result {
                tribute::Value::Unit => {} // Don't print unit values
                _ => println!("{}", result),
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Compiles a Tribute program to native binary
fn compile_program(input_path: &PathBuf, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Compiling {} to {}...", input_path.display(), output_path);
    
    // Read source code
    let source = std::fs::read_to_string(input_path)?;
    let db = TributeDatabaseImpl::default();
    
    // Parse to AST
    let (program, diagnostics) = parse_str(&db, input_path, &source);
    
    // Check for parsing errors
    if !diagnostics.is_empty() {
        eprintln!("Compilation errors:");
        for diagnostic in diagnostics {
            eprintln!("  {:?}", diagnostic);
        }
        std::process::exit(1);
    }
    
    // Lower to HIR
    let hir_program = lower_program_to_hir(&db, program)
        .ok_or("Failed to lower program to HIR")?;
    
    // Create Cranelift compiler
    let compiler = TributeCompiler::new(None)?; // Use native target
    
    // Compile to object code
    let object_bytes = compiler.compile_program(&db, hir_program)?;
    
    // Write object file (for now, we'll just write the raw object)
    // TODO: Link with runtime library to create executable
    let object_path = format!("{}.o", output_path);
    std::fs::write(&object_path, &object_bytes)?;
    
    println!("Successfully compiled to object file: {}", object_path);
    println!("Note: Linking with runtime library is not yet implemented");
    println!("Object file size: {} bytes", object_bytes.len());
    
    Ok(())
}
