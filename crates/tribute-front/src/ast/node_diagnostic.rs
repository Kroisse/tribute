//! Diagnostics positioned at nodes rather than source spans.

use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};

use super::{NodeId, SpanMaps};

/// A diagnostic whose positions are nodes. A query that reports one does
/// not read the span map, so its result stays valid when code moves. The
/// reader of a query's diagnostics looks the spans up with
/// [`locate_diagnostics`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[salsa::accumulator]
pub struct NodeDiagnostic {
    message: String,
    node: NodeId,
    severity: DiagnosticSeverity,
    phase: CompilationPhase,
    labels: Vec<(NodeId, Box<str>)>,
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

    /// The diagnostic at the spans `spans` gives its nodes.
    pub fn locate(&self, spans: &SpanMaps<'_>) -> Diagnostic {
        let mut builder = Diagnostic::builder(
            self.message.clone(),
            spans.get_or_default(self.node),
            self.severity,
            self.phase.clone(),
        );
        for (node, message) in &self.labels {
            builder = builder.label(spans.get_or_default(*node), message.clone());
        }
        builder.build()
    }
}

/// A query's `diagnostics`, followed by its `nodes` diagnostics at the spans
/// `spans` gives their nodes.
pub fn locate_diagnostics<'a>(
    diagnostics: impl IntoIterator<Item = &'a Diagnostic>,
    nodes: impl IntoIterator<Item = &'a NodeDiagnostic>,
    spans: &SpanMaps<'_>,
) -> Vec<Diagnostic> {
    diagnostics
        .into_iter()
        .cloned()
        .chain(nodes.into_iter().map(|node| node.locate(spans)))
        .collect()
}
