//! Diagnostic messages emitted during compilation.

use trunk_ir::Span;

/// A diagnostic message (error, warning, or info) with source location.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[salsa::accumulator]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
    pub severity: DiagnosticSeverity,
    pub phase: CompilationPhase,
}

/// Severity level of a diagnostic.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
}

/// Compilation phase where a diagnostic was emitted.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompilationPhase {
    Parsing,
    AstGeneration,
    TirGeneration,
    NameResolution,
    TypeChecking,
    Lowering,
    Optimization,
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
