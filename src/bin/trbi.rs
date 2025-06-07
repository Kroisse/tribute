//! Tribute interpreter
//!
//! This is a simple command-line tool that runs Tribute programs.
extern crate tribute;

use std::{ffi::OsString, path::PathBuf};
use tribute::{eval_str, parse_str, TributeDatabaseImpl, Value};

type Error = std::io::Error;

fn main() -> Result<(), Error> {
    let path = parse_args(std::env::args_os())?;
    let source = std::fs::read_to_string(&path)?;

    // Create Salsa database and parse using parse_with_database
    let db = TributeDatabaseImpl::default();
    let (_program, diags) = parse_str(&db, &path, &source);

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

    match eval_str(&db, &path, &source) {
        Ok(result) => {
            if !matches!(result, Value::Unit) {
                println!("Result: {}", result);
            }
        }
        Err(e) => {
            eprintln!("Evaluation error: {}", e);
            std::process::exit(1);
        }
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
