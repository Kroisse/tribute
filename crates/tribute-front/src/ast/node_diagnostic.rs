//! Diagnostics positioned at nodes rather than source spans.

use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::diagnostic::Label;

use super::{NodeId, SpanMaps};

/// A diagnostic whose positions are nodes. A query that reports one does
/// not read the span map, so its result stays valid when code moves. The
/// reader of a query's diagnostics looks the spans up with
/// [`located_diagnostics!`](crate::located_diagnostics).
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
        Diagnostic {
            inner: trunk_ir::diagnostic::Diagnostic {
                message: self.message.clone(),
                span: spans.get_or_default(self.node),
                severity: self.severity,
                labels: self
                    .labels
                    .iter()
                    .map(|(node, message)| Label {
                        span: spans.get_or_default(*node),
                        message: message.clone(),
                    })
                    .collect(),
                note: None,
            },
            phase: self.phase.clone(),
        }
    }
}

/// A query's `diagnostics`, followed by its `nodes` diagnostics at the spans
/// `spans` gives their nodes. [`located_diagnostics!`] reads both from a
/// query.
///
/// [`located_diagnostics!`]: crate::located_diagnostics
pub fn locate_diagnostics<'a, 's>(
    diagnostics: Vec<&'a Diagnostic>,
    nodes: Vec<&'a NodeDiagnostic>,
    spans: SpanMaps<'s>,
) -> impl Iterator<Item = Diagnostic> + use<'a, 's> {
    let located = nodes.into_iter().map(move |node| node.locate(&spans));
    diagnostics.into_iter().cloned().chain(located)
}

/// The diagnostics the tracked query `query` and the queries it ran
/// reported for `args`, with those positioned at nodes located through
/// `spans`, as an iterator.
///
/// Salsa generates `accumulated` for each query as an inherent function, so
/// no trait abstracts over queries; the macro calls it once for each
/// accumulator, evaluating `args` twice. Call it outside any query: reading
/// accumulated values inside one is an untracked read.
///
/// ```ignore
/// let diagnostics: Vec<_> =
///     located_diagnostics!(analysis::analyze(db, source, table), analysis::spans(db, source, table))
///         .collect();
/// ```
#[macro_export]
macro_rules! located_diagnostics {
    ($($query:ident)::+ ($($arg:expr),* $(,)?), $spans:expr $(,)?) => {
        $crate::ast::locate_diagnostics(
            $($query)::+::accumulated($($arg),*),
            $($query)::+::accumulated($($arg),*),
            $spans,
        )
    };
}
