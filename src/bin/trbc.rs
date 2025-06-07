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

extern crate tribute;

use clap::{Arg, ArgAction, Command};
use std::path::PathBuf;
use tribute::{eval_with_hir, TributeDatabaseImpl};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("trbc")
        .version("0.1.0")
        .about("Tribute compiler and interpreter")
        .arg(
            Arg::new("input")
                .help("Input Tribute source file")
                .required(true)
                .value_name("FILE")
                .index(1)
        )
        .arg(
            Arg::new("compile")
                .long("compile")
                .short('c')
                .help("Compile to native binary instead of interpreting")
                .action(ArgAction::SetTrue)
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("Output file path (required when compiling)")
                .value_name("OUTPUT")
                .requires("compile")
        )
        .get_matches();

    let input_path = PathBuf::from(matches.get_one::<String>("input").unwrap());
    let compile_mode = matches.get_flag("compile");

    if compile_mode {
        eprintln!("Error: Compilation support is not yet implemented");
        std::process::exit(1);
    } else {
        // Interpreter mode (default)
        interpret_program(&input_path)?;
    }

    Ok(())
}

/// Interprets a Tribute program using HIR-based evaluation.
fn interpret_program(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)?;
    let db = TributeDatabaseImpl::default();
    
    // Use HIR-based evaluation directly
    match eval_with_hir(&db, path, &source) {
        Ok(result) => {
            // Only print non-unit results
            match result {
                tribute::Value::Unit => {}, // Don't print unit values
                _ => println!("{}", result),
            }
        },
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

