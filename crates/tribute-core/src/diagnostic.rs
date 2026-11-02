//! Diagnostic messages emitted during compilation.

use serde::Serialize;

pub use trunk_ir::diagnostic::DiagnosticSeverity;

/// A diagnostic message (error, warning, or info) with source location and
/// compilation phase. Wraps [`trunk_ir::diagnostic::Diagnostic`] with
/// Tribute-specific compilation phase information.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[salsa::accumulator]
pub struct SpanDiagnostic {
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

impl SpanDiagnostic {
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

    /// Create a `SpanDiagnostic` from a [`trunk_ir::diagnostic::Diagnostic`] and a
    /// compilation phase.
    pub fn from_ir(diag: trunk_ir::diagnostic::Diagnostic, phase: CompilationPhase) -> Self {
        Self { inner: diag, phase }
    }

    /// Create a builder for a diagnostic with secondary labels and/or notes.
    ///
    /// Use this when you need to attach additional context beyond a single
    /// message and span. For simple diagnostics, use [`SpanDiagnostic::new`] directly.
    pub fn builder(
        message: impl Into<String>,
        span: trunk_ir::Span,
        severity: DiagnosticSeverity,
        phase: CompilationPhase,
    ) -> SpanDiagnosticBuilder {
        SpanDiagnosticBuilder {
            message: message.into(),
            span,
            severity,
            phase,
            labels: Vec::new(),
            note: None,
        }
    }
}

/// Builder for constructing [`SpanDiagnostic`] values with secondary labels and notes.
///
/// Labels are collected in a `Vec` during building and converted to `Box<[Label]>`
/// once at [`build`](SpanDiagnosticBuilder::build) time.
pub struct SpanDiagnosticBuilder {
    message: String,
    span: trunk_ir::Span,
    severity: DiagnosticSeverity,
    phase: CompilationPhase,
    labels: Vec<trunk_ir::diagnostic::Label>,
    note: Option<Box<str>>,
}

impl SpanDiagnosticBuilder {
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

    /// Build the [`SpanDiagnostic`].
    pub fn build(self) -> SpanDiagnostic {
        SpanDiagnostic {
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

/// Compare diagnostics using a stable user-facing order.
///
/// Compilation phase is primary, followed by source location and message as
/// deterministic tie-breakers. This is also used by the CLI for target-specific
/// accumulator results.
pub fn compare_diagnostics(left: &SpanDiagnostic, right: &SpanDiagnostic) -> std::cmp::Ordering {
    fn phase_rank(phase: &CompilationPhase) -> u8 {
        match phase {
            CompilationPhase::Parsing => 0,
            CompilationPhase::AstGeneration => 1,
            CompilationPhase::TirGeneration => 2,
            CompilationPhase::NameResolution => 3,
            CompilationPhase::TypeChecking => 4,
            CompilationPhase::Lowering => 5,
            CompilationPhase::Optimization => 6,
        }
    }

    phase_rank(&left.phase)
        .cmp(&phase_rank(&right.phase))
        .then_with(|| left.inner.span.cmp(&right.inner.span))
        .then_with(|| left.inner.message.cmp(&right.inner.message))
}
