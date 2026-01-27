//! Tribute compiler utilities.
pub mod diagnostic;
pub mod target;

pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use target::*;
