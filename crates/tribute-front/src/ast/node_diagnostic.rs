//! Diagnostics positioned at nodes rather than source spans.

use std::cell::RefCell;

use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};

use super::{NodeId, SpanMap};

/// A diagnostic whose positions are nodes. A query that reports one does
/// not read the span map, so its result stays valid when code moves; the
/// spans are looked up when it is reported.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeDiagnostic {
    message: String,
    node: NodeId,
    severity: DiagnosticSeverity,
    phase: CompilationPhase,
    labels: Vec<(NodeId, Box<str>)>,
}

/// The [`NodeDiagnostic`]s a phase reports while it runs, in report order.
/// A report needs only a shared reference, so a phase that reads its state
/// through `&self` reports as it goes.
#[derive(Debug, Default)]
pub(crate) struct DiagnosticLog(RefCell<Vec<NodeDiagnostic>>);

impl DiagnosticLog {
    pub(crate) fn push(&self, diagnostic: NodeDiagnostic) {
        self.0.borrow_mut().push(diagnostic);
    }

    pub(crate) fn extend(&self, diagnostics: impl IntoIterator<Item = NodeDiagnostic>) {
        self.0.borrow_mut().extend(diagnostics);
    }

    /// The diagnostics reported so far, leaving the log empty.
    pub(crate) fn take(&self) -> Vec<NodeDiagnostic> {
        self.0.take()
    }

    pub(crate) fn into_vec(self) -> Vec<NodeDiagnostic> {
        self.0.into_inner()
    }
}

impl NodeDiagnostic {
    pub(crate) fn new(
        message: impl Into<String>,
        node: NodeId,
        severity: DiagnosticSeverity,
        phase: CompilationPhase,
    ) -> Self {
        Self {
            message: message.into(),
            node,
            severity,
            phase,
            labels: Vec::new(),
        }
    }

    /// Add a secondary label at a related node.
    pub(crate) fn label(mut self, node: NodeId, message: impl Into<Box<str>>) -> Self {
        self.labels.push((node, message.into()));
        self
    }

    /// Record the diagnostic in `phase`'s log.
    pub(crate) fn report(self, phase: &impl AsRef<DiagnosticLog>) {
        phase.as_ref().push(self);
    }

    /// The diagnostic at the spans `span_map` gives its nodes.
    pub fn resolve(&self, span_map: &SpanMap<'_>) -> Diagnostic {
        let mut builder = Diagnostic::builder(
            self.message.clone(),
            span_map.get_or_default(self.node),
            self.severity,
            self.phase.clone(),
        );
        for (node, message) in &self.labels {
            builder = builder.label(span_map.get_or_default(*node), message.clone());
        }
        builder.build()
    }
}
