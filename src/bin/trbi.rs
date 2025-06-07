//! Tribute interpreter
//!
//! This is a simple command-line tool that runs Tribute programs.
extern crate tribute;

use std::{ffi::OsString, path::PathBuf};
use tribute::{eval_expr, eval_with_hir, parse_with_database, Environment, TributeDatabaseImpl, Value};

type Error = std::io::Error;

fn main() -> Result<(), Error> {
    let path = parse_args(std::env::args_os())?;
    let source = std::fs::read_to_string(&path)?;

    // Create Salsa database and parse using parse_with_database
    let db = TributeDatabaseImpl::default();
    let (program, diags) = parse_with_database(&db, &path, &source);

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

    // Try HIR-based evaluation first
    eprintln!("Attempting HIR-based evaluation...");
    match eval_with_hir(&db, &path, &source) {
        Ok(result) => {
            eprintln!("HIR evaluation successful!");
            if !matches!(result, Value::Unit) {
                println!("Result: {}", result);
            }
            return Ok(());
        }
        Err(e) => {
            eprintln!("HIR evaluation failed: {}", e);
            eprintln!("Falling back to AST-based evaluation...");
        }
    }

    // Fallback to AST-based evaluation
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

fn parse_args(args: impl Iterator<Item = OsString>) -> Result<PathBuf, Error> {
    let mut args = args.skip(1);
    let path = args.next().ok_or_else(|| {
        Error::new(
            std::io::ErrorKind::InvalidInput,
            "expected path to file to compile",
        )
    })?;
    if args.next().is_some() {
        return Err(Error::new(
            std::io::ErrorKind::InvalidInput,
            "expected only one argument",
        ));
    }
    // convert path to PathBuf
    Ok(PathBuf::from(path))
}
