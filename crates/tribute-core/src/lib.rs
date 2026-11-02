//! Tribute compiler utilities.
pub mod diagnostic;
pub mod fmt;
pub mod target;

pub use diagnostic::{CompilationPhase, DiagnosticSeverity, SpanDiagnostic};
pub use target::*;
