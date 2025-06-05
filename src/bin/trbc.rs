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
use tribute::{parse_with_database, TributeDatabaseImpl, eval_expr, Environment, Value};

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
        // Compilation mode
        #[cfg(feature = "compiler")]
        {
            let output_path = matches.get_one::<String>("output")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let mut path = input_path.clone();
                    path.set_extension("");
                    path
                });

            compile_program(&input_path, &output_path)?;
        }
        
        #[cfg(not(feature = "compiler"))]
        {
            eprintln!("Error: Compilation support not enabled. Rebuild with --features compiler");
            std::process::exit(1);
        }
    } else {
        // Interpreter mode (default)
        interpret_program(&input_path)?;
    }

    Ok(())
}

/// Interprets a Tribute program.
fn interpret_program(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)?;

    // Create Salsa database and parse using parse_with_database
    let db = TributeDatabaseImpl::default();
    let (program, diags) = parse_with_database(&db, path, &source);

    // Display diagnostics if any
    if !diags.is_empty() {
        eprintln!("Diagnostics:");
        for diagnostic in &diags {
            eprintln!(
                "  [{}] {} (span: {}..{})",
                diagnostic.severity, diagnostic.message, diagnostic.span.start, diagnostic.span.end
            );
        }
    }

    // Execute the program
    let mut env = Environment::toplevel();
    
    // Evaluate all expressions to register functions
    for item in program.items(&db).iter() {
        let (expr, _span) = item.expr(&db);
        if let Err(e) = eval_expr(&mut env, &expr) {
            eprintln!("Evaluation error: {}", e);
            return Ok(());
        }
    }
    
    // Try to call main function if it exists
    if let Ok(main_fn) = env.lookup(&"main".to_string()) {
        match main_fn {
            Value::Fn(_, params, body) => {
                if params.is_empty() {
                    let mut child_env = env.child(vec![]);
                    for expr in body {
                        if let Err(e) = eval_expr(&mut child_env, &expr.0) {
                            eprintln!("Runtime error in main: {}", e);
                            return Ok(());
                        }
                    }
                } else {
                    eprintln!("main function should not have parameters");
                }
            }
            _ => {
                eprintln!("main is not a function");
            }
        }
    } else {
        eprintln!("No main function found");
    }

    Ok(())
}

/// Compiles a Tribute program to a native binary.
#[cfg(feature = "compiler")]
fn compile_program(input_path: &PathBuf, output_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use tribute_compiler::TributeCompiler;
    
    println!("Compiling {} to {}...", input_path.display(), output_path.display());
    
    let mut compiler = TributeCompiler::new()?;
    compiler.compile_file(input_path, output_path)?;
    
    println!("Compilation completed successfully!");
    Ok(())
}