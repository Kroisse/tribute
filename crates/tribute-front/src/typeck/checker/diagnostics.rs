//! Source-oriented rendering of function inference failures.

use super::TypeChecker;
use crate::ast::{NodeId, SpanMap, TypeKind, TypeOrigin};
use crate::resolve::NameDisplay;
use crate::typeck::constraint::ConstraintOriginKind;
use crate::typeck::solver::LocatedSolveError;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};

/// A type checking diagnostic whose positions are nodes. Checking a
/// function does not read the span map, so its result stays valid when
/// other items move; the spans are looked up when it is reported.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

    /// Record the diagnostic in `checker`'s report.
    pub(crate) fn report(self, checker: &TypeChecker<'_>) {
        checker.diagnostics.borrow_mut().push(self);
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

pub(super) fn solve_error_context(kind: Option<ConstraintOriginKind>) -> &'static str {
    match kind {
        Some(ConstraintOriginKind::Call) => " at call site",
        Some(ConstraintOriginKind::Lambda) => " in lambda",
        Some(ConstraintOriginKind::HandlerBoundary) => " at handler boundary",
        Some(ConstraintOriginKind::Expression) | None => "",
    }
}

pub(super) fn format_solve_error<'db>(
    db: &'db dyn salsa::Database,
    names: &NameDisplay,
    error: &super::super::solver::SolveError<'db>,
) -> String {
    // Two distinct types that this package spells alike would read as the
    // same type in a plain mismatch message.
    if let super::super::solver::SolveError::TypeMismatch { expected, actual } = error
        && let (
            TypeKind::Named {
                id: expected_id, ..
            },
            TypeKind::Named { id: actual_id, .. },
        ) = (expected.kind(db), actual.kind(db))
        && expected_id != actual_id
        && expected.display(db, names).to_string() == actual.display(db, names).to_string()
    {
        match (expected_id.origin(db), actual_id.origin(db)) {
            (TypeOrigin::Builtin(_), TypeOrigin::Source { .. }) => {
                return format!(
                    "canonical compiler-owned type `{}` is distinct from \
                     source-declared type `{}`",
                    expected.display(db, names),
                    actual.display(db, names)
                );
            }
            (TypeOrigin::Source { .. }, TypeOrigin::Builtin(_)) => {
                return format!(
                    "canonical compiler-owned type `{}` is distinct from \
                     source-declared type `{}`",
                    actual.display(db, names),
                    expected.display(db, names)
                );
            }
            _ => {}
        }
    }

    error.display(db, names).to_string()
}

impl<'db> TypeChecker<'db> {
    pub(super) fn report_solve_error(
        &self,
        func_id: crate::ast::NodeId,
        func_name: &trunk_ir::Symbol,
        effects: Option<&[crate::ast::TypeAnnotation]>,
        failure: LocatedSolveError<'db>,
    ) {
        let primary_node = failure
            .origin
            .map(|origin| origin.node_id)
            .unwrap_or(func_id);
        let context = solve_error_context(failure.origin.map(|origin| origin.kind));
        let error = format_solve_error(self.db(), self.names(), &failure.error);
        let mut diagnostic = NodeDiagnostic::new(
            format!("type error{context} in function '{}': {error}", func_name),
            primary_node,
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        );

        let is_effect_error = matches!(
            &failure.error,
            super::super::solver::SolveError::RowMismatch { .. }
                | super::super::solver::SolveError::AbilityArgArityMismatch { .. }
        );
        if is_effect_error && primary_node != func_id {
            let related_node = effects
                .and_then(|effects| effects.first())
                .map(|annotation| annotation.id)
                .unwrap_or(func_id);
            diagnostic = diagnostic.label(
                related_node,
                "enclosing function effect contract is declared here",
            );
        }

        diagnostic.report(self);
    }
}
