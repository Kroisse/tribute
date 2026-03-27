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
                labels: Box::default(),
                note: None,
            },
            phase,
        }
    }

    /// Create a `Diagnostic` from a [`trunk_ir::diagnostic::Diagnostic`] and a
    /// compilation phase.
    pub fn from_ir(diag: trunk_ir::diagnostic::Diagnostic, phase: CompilationPhase) -> Self {
        Self { inner: diag, phase }
    }

    /// Create a builder for a diagnostic with secondary labels and/or notes.
    ///
    /// Use this when you need to attach additional context beyond a single
    /// message and span. For simple diagnostics, use [`Diagnostic::new`] directly.
    pub fn builder(
        message: impl Into<String>,
        span: trunk_ir::Span,
        severity: DiagnosticSeverity,
        phase: CompilationPhase,
    ) -> DiagnosticBuilder {
        DiagnosticBuilder {
            message: message.into(),
            span,
            severity,
            phase,
            labels: Vec::new(),
            note: None,
        }
    }
}

/// Builder for constructing [`Diagnostic`] values with secondary labels and notes.
///
/// Labels are collected in a `Vec` during building and converted to `Box<[Label]>`
/// once at [`build`](DiagnosticBuilder::build) time.
pub struct DiagnosticBuilder {
    message: String,
    span: trunk_ir::Span,
    severity: DiagnosticSeverity,
    phase: CompilationPhase,
    labels: Vec<trunk_ir::diagnostic::Label>,
    note: Option<Box<str>>,
}

impl DiagnosticBuilder {
    /// Add a secondary label pointing to a related source span.
    pub fn label(mut self, span: trunk_ir::Span, message: impl Into<Box<str>>) -> Self {
        self.labels.push(trunk_ir::diagnostic::Label {
            span,
            message: message.into(),
        });
        self
    }

    /// Add a note providing extra context after the main message.
    pub fn note(mut self, message: impl Into<Box<str>>) -> Self {
        self.note = Some(message.into());
        self
    }

    /// Build the [`Diagnostic`].
    pub fn build(self) -> Diagnostic {
        Diagnostic {
            inner: trunk_ir::diagnostic::Diagnostic {
                message: self.message,
                span: self.span,
                severity: self.severity,
                labels: self.labels.into_boxed_slice(),
                note: self.note,
            },
            phase: self.phase,
        }
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
