//! Diagnostic types for IR validation and transformation passes.

use serde::Serialize;

use crate::location::Span;

/// A diagnostic message emitted during IR validation or transformation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
    pub severity: DiagnosticSeverity,
    /// Additional labeled spans providing context (e.g., "expected `Int` here").
    #[serde(default, skip_serializing_if = "<[Label]>::is_empty")]
    pub labels: Box<[Label]>,
    /// Extra context after the main message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<Box<str>>,
}

/// A secondary label attached to a diagnostic, pointing to a related source span.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Label {
    pub span: Span,
    pub message: Box<str>,
}

/// Severity level of a diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticSeverity::Error => write!(f, "ERROR"),
            DiagnosticSeverity::Warning => write!(f, "WARNING"),
            DiagnosticSeverity::Info => write!(f, "INFO"),
        }
    }
}
