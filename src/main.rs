//! Tribute compiler CLI entry point.

mod cli;
mod lsp;

use clap::Parser;
use cli::{Cli, Command};

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve => {
            if let Err(e) = lsp::serve() {
                eprintln!("LSP server error: {e}");
                std::process::exit(1);
            }
        }
    }
}
