//! Error handling for the Tribute compiler.

use derive_more::{Display, Error, From};

/// Result type for compiler operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during compilation.
#[derive(Debug, Display, Error, From)]
pub enum Error {
    /// I/O errors when reading source files or writing output.
    #[display("I/O error: {_0}")]
    IoError(#[error(source)] std::io::Error),

    /// Parse errors from the Tribute parser.
    #[from(ignore)]
    #[display("Parse error: {_0}")]
    ParseError(#[error(not(source))] String),

    /// MLIR-related errors.
    #[display("MLIR error: {_0}")]
    MlirError(melior::Error),

    /// Code generation errors.
    #[from(ignore)]
    #[display("Codegen error: {_0}")]
    CodegenError(#[error(not(source))] String),

    /// Type checking or semantic analysis errors.
    #[from(ignore)]
    #[display("Semantic error: {_0}")]
    SemanticError(#[error(not(source))] String),

    /// Unsupported language features.
    #[from(ignore)]
    #[display("Unsupported feature: {_0}")]
    UnsupportedFeature(#[error(not(source))] String),
}
