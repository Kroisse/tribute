//! Tribute compiler utilities.
pub mod diagnostic;
pub mod fmt;
pub mod target;

pub use diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
pub use target::*;
