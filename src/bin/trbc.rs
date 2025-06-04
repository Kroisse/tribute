extern crate tribute;

use std::{ffi::OsString, path::PathBuf};

type Error = std::io::Error;

fn main() -> Result<(), Error> {
    let path = parse_args(std::env::args_os())?;
    let source = std::fs::read_to_string(path)?;
    let ast = tribute::parse(&source);
    println!("{:#?}", ast);
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
