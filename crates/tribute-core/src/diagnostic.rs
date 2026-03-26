//! Diagnostic messages emitted during compilation.

use serde::Serialize;

pub use trunk_ir::diagnostic::DiagnosticSeverity;

/// A diagnostic message (error, warning, or info) with source location and
/// compilation phase. Wraps [`trunk_ir::diagnostic::Diagnostic`] with
/// Tribute-specific compilation phase information.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[salsa::accumulator]
pub struct Diagnostic {
    #[serde(flatten)]
    pub inner: trunk_ir::diagnostic::Diagnostic,
    pub phase: CompilationPhase,
}

/// Compilation phase where a diagnostic was emitted.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum CompilationPhase {
    Parsing,
    AstGeneration,
    TirGeneration,
    NameResolution,
    TypeChecking,
    Lowering,
    Optimization,
}

impl Diagnostic {
    /// Create a new diagnostic with the given message, span, severity, and phase.
    pub fn new(
        message: impl Into<String>,
        span: trunk_ir::Span,
        severity: DiagnosticSeverity,
        phase: CompilationPhase,
    ) -> Self {
        Self {
            inner: trunk_ir::diagnostic::Diagnostic {
                message: message.into(),
                span,
                severity,
            },
            phase,
        }
    }

    /// Create a `Diagnostic` from a [`trunk_ir::diagnostic::Diagnostic`] and a
    /// compilation phase.
    pub fn from_ir(diag: trunk_ir::diagnostic::Diagnostic, phase: CompilationPhase) -> Self {
        Self { inner: diag, phase }
    }
}

impl std::fmt::Display for CompilationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilationPhase::Parsing => write!(f, "Parsing"),
            CompilationPhase::AstGeneration => write!(f, "AST Generation"),
            CompilationPhase::TirGeneration => write!(f, "TIR Generation"),
            CompilationPhase::NameResolution => write!(f, "Name Resolution"),
            CompilationPhase::TypeChecking => write!(f, "Type Checking"),
            CompilationPhase::Lowering => write!(f, "Lowering"),
            CompilationPhase::Optimization => write!(f, "Optimization"),
        }
    }
}
