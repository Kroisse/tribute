//! Command-line interface for the Tribute compiler.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

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

    /// Compile a source file to a target
    Compile {
        /// Path to the source file to compile
        file: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Compilation target (wasm, none)
        #[arg(long, default_value = "none")]
        target: String,
    },

    /// Debug compilation of a source file
    Debug {
        /// Path to the source file to debug
        file: PathBuf,

        /// Show module environment details
        #[arg(long)]
        show_env: bool,
    },
}
