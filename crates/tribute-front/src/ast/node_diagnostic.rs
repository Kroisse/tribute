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

/// Something that collects the [`NodeDiagnostic`]s reported while it runs.
pub(crate) trait DiagnosticSink {
    fn node_diagnostics(&self) -> &RefCell<Vec<NodeDiagnostic>>;
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

    /// Record the diagnostic in `sink`'s report.
    pub(crate) fn report(self, sink: &impl DiagnosticSink) {
        sink.node_diagnostics().borrow_mut().push(self);
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
