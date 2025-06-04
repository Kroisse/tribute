extern crate tribute;

use std::{ffi::OsString, path::PathBuf};
use tribute::{parse_with_database, TributeDatabaseImpl};

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

    // Display parsed AST
    println!("Parsed {} expressions:", program.expressions(&db).len());
    for (i, tracked_expr) in program.expressions(&db).iter().enumerate() {
        println!("Expression {}: {}", i + 1, tracked_expr.expr(&db));
        println!(
            "  Span: {}..{}",
            tracked_expr.span(&db).start,
            tracked_expr.span(&db).end
        );
        println!("  Debug: {:#?}", tracked_expr.expr(&db));
        println!();
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
