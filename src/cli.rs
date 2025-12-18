//! Command-line interface for the Tribute compiler.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "tribute")]
#[command(about = "Tribute programming language compiler", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the Language Server Protocol (LSP) server
    #[command(alias = "lsp")]
    Serve,
}
