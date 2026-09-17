//! Source-oriented rendering of function inference failures.

use super::TypeChecker;
use crate::ast::{TypeKind, TypeOrigin};
use crate::typeck::constraint::ConstraintOriginKind;
use crate::typeck::solver::LocatedSolveError;
use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};

pub(super) fn solve_error_context(kind: Option<ConstraintOriginKind>) -> &'static str {
    match kind {
        Some(ConstraintOriginKind::Call) => " at call site",
        Some(ConstraintOriginKind::Lambda) => " in lambda",
        Some(ConstraintOriginKind::HandlerBoundary) => " at handler boundary",
        Some(ConstraintOriginKind::Expression) | None => "",
    }
}

pub(super) fn format_solve_error(
    db: &dyn salsa::Database,
    error: &super::super::solver::SolveError<'_>,
) -> String {
    if let super::super::solver::SolveError::TypeMismatch { expected, actual } = error
        && let (
            TypeKind::Named {
                id: expected_id, ..
            },
            TypeKind::Named { id: actual_id, .. },
        ) = (expected.kind(db), actual.kind(db))
        && expected_id != actual_id
        && expected.to_string() == actual.to_string()
    {
        match (expected_id.origin(db), actual_id.origin(db)) {
            (TypeOrigin::Builtin(_), TypeOrigin::Source(_)) => {
                return format!(
                    "canonical compiler-owned type `{expected}` is distinct from \
                     source-declared type `{actual}`"
                );
            }
            (TypeOrigin::Source(_), TypeOrigin::Builtin(_)) => {
                return format!(
                    "canonical compiler-owned type `{actual}` is distinct from \
                     source-declared type `{expected}`"
                );
            }
            _ => {}
        }
    }

    error.to_string()
}

impl<'db> TypeChecker<'db> {
    pub(super) fn report_solve_error(
        &self,
        func_id: crate::ast::NodeId,
        func_name: trunk_ir::Symbol,
        effects: Option<&[crate::ast::TypeAnnotation]>,
        failure: LocatedSolveError<'db>,
    ) {
        let primary_node = failure
            .origin
            .map(|origin| origin.node_id)
            .unwrap_or(func_id);
        let context = solve_error_context(failure.origin.map(|origin| origin.kind));
        let error = format_solve_error(self.db(), &failure.error);
        let mut diagnostic = Diagnostic::builder(
            format!("type error{context} in function '{}': {error}", func_name),
            self.get_span(primary_node),
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        );

        let is_effect_error = matches!(
            &failure.error,
            super::super::solver::SolveError::RowMismatch { .. }
                | super::super::solver::SolveError::EffectArgArityMismatch { .. }
        );
        if is_effect_error && primary_node != func_id {
            let related_node = effects
                .and_then(|effects| effects.first())
                .map(|annotation| annotation.id)
                .unwrap_or(func_id);
            diagnostic = diagnostic.label(
                self.get_span(related_node),
                "enclosing function effect contract is declared here",
            );
        }

        diagnostic.build().accumulate(self.db());
    }
}
