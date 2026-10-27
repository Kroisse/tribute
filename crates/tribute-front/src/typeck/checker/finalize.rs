//! Apply solved substitutions and collect binder variables in checked bodies.
//!
//! These walks preserve the existing inference and generalization policy;
//! function orchestration supplies the solved state and binder mapping.

use super::TypeChecker;
use crate::ast::NodeId;
use crate::ast::visit::{RefSite, Refs, VisitMut, walk_expr, walk_expr_mut, walk_pattern_mut};
use crate::ast::{
    Expr, ExprKind, FuncDefId, LiteralPattern, Pattern, PatternKind, ResolvedRef, Type, TypedRef,
    UniVarId,
};
use crate::typeck::solver::{RowSubst, TypeSubst};
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

impl<'db> TypeChecker<'db> {
    // =========================================================================
    // UniVar collection from body
    // =========================================================================

    /// Collect all unresolved UniVars from the body expression.
    pub(super) fn collect_univars_from_body(
        &self,
        body: &Expr<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        let db = self.db();
        walk_expr(
            &mut Refs(|_, _, value: &TypedRef<'db>| {
                type_subst.collect_univars_from_type(db, value.ty, row_subst, out);
            }),
            body,
        );
    }

    pub(super) fn collect_univars_from_deferred_resolutions(
        &self,
        deferred_resolutions: &HashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        let mut node_ids: Vec<_> = deferred_resolutions.keys().copied().collect();
        node_ids.sort();
        for node_id in node_ids {
            let (_, callee_ty) = deferred_resolutions[&node_id];
            type_subst.collect_univars_from_type(self.db(), callee_ty, row_subst, out);
        }
    }
}

/// One function's solved substitution and binder mapping.
pub(super) struct Substitution<'a, 'db> {
    pub(super) db: &'db dyn salsa::Database,
    pub(super) type_subst: &'a TypeSubst<'db>,
    pub(super) row_subst: &'a RowSubst<'db>,
    pub(super) var_to_index: &'a HashMap<UniVarId<'db>, u32>,
    pub(super) local_generalizations: &'a HashMap<UniVarId<'db>, (NodeId, u32)>,
}

impl<'db> Substitution<'_, 'db> {
    /// Resolve `ty` and turn its remaining variables into the function's
    /// binders: interface variables become `BoundVar`, variables of a local
    /// generalized scheme become `LocalBoundVar`.
    pub(super) fn apply(&self, ty: Type<'db>) -> Type<'db> {
        let substituted = self.type_subst.apply_with_rows(self.db, ty, self.row_subst);
        self.type_subst.apply_generalization_with_local_vars(
            self.db,
            substituted,
            self.row_subst,
            self.var_to_index,
            self.local_generalizations,
        )
    }
}

/// Applies one function's solved substitution to its checked body.
///
/// Deferred method calls become direct calls, deferred implicit
/// constructors their selected constructors, and case expressions whose
/// substituted arms cover their scrutinee are recorded as exhaustive.
pub(super) struct Finalize<'a, 'db> {
    pub(super) checker: &'a TypeChecker<'db>,
    pub(super) substitution: &'a Substitution<'a, 'db>,
    pub(super) deferred_resolutions: &'a HashMap<NodeId, (FuncDefId<'db>, Type<'db>)>,
    /// The constructor each implicit constructor selected after solving, and
    /// its instance.
    pub(super) constructor_resolutions: &'a HashMap<NodeId, (ResolvedRef<'db>, Type<'db>)>,
    pub(super) node_types: &'a HashMap<NodeId, Type<'db>>,
    /// The node types as inference recorded them. A diagnostic shows one
    /// solved but not generalized, so a variable nothing determined reads
    /// `_` rather than a binder of the function.
    pub(super) unbound_node_types: &'a HashMap<NodeId, Type<'db>>,
    pub(super) exhaustive_cases: &'a mut Vec<NodeId>,
    pub(super) exhaustiveness_reported: &'a mut HashSet<NodeId>,
}

impl<'db> Finalize<'_, 'db> {
    fn apply(&self, ty: Type<'db>) -> Type<'db> {
        self.substitution.apply(ty)
    }
}

impl<'db> VisitMut<TypedRef<'db>> for Finalize<'_, 'db> {
    fn visit_ref_mut(&mut self, _: RefSite, node: NodeId, value: &mut TypedRef<'db>) {
        // An implicit constructor selected during solving becomes the
        // constructor; one left unselected was reported.
        if matches!(value.resolved, ResolvedRef::ImplicitConstructor { .. })
            && let Some((resolved, instance)) = self.constructor_resolutions.get(&node)
        {
            value.resolved = resolved.clone();
            value.ty = *instance;
        }
        value.ty = self.apply(value.ty);
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr<TypedRef<'db>>) {
        // Children first: the rewrites below read substituted subtrees.
        walk_expr_mut(self, expr);
        match &mut *expr.kind {
            ExprKind::Call { callee, .. }
                if matches!(&*callee.kind, ExprKind::Functions { .. }) =>
            {
                // A function selected during solving becomes the callee; a
                // call left unselected was reported and stays as it is.
                let Some(&(func_id, callee_ty)) = self.deferred_resolutions.get(&expr.id) else {
                    return;
                };
                *callee = Expr::new(
                    callee.id,
                    ExprKind::Var(TypedRef {
                        resolved: ResolvedRef::Function { id: func_id },
                        ty: self.apply(callee_ty),
                    }),
                );
            }
            ExprKind::Functions { .. } => {
                // A name used as a value becomes the function selected for it.
                if let Some(&(func_id, callee_ty)) = self.deferred_resolutions.get(&expr.id) {
                    *expr.kind = ExprKind::Var(TypedRef {
                        resolved: ResolvedRef::Function { id: func_id },
                        ty: self.apply(callee_ty),
                    });
                }
            }
            ExprKind::Case { scrutinee, arms } => {
                let shown_ty = self.unbound_node_types.get(&scrutinee.id).map(|ty| {
                    let substitution = self.substitution;
                    substitution.type_subst.apply_with_rows(
                        substitution.db,
                        *ty,
                        substitution.row_subst,
                    )
                });
                if let Some(scrutinee_ty) = self.node_types.get(&scrutinee.id).copied()
                    && self.checker.check_exhaustiveness(
                        self.exhaustiveness_reported,
                        scrutinee_ty,
                        shown_ty.unwrap_or(scrutinee_ty),
                        arms,
                        scrutinee.id,
                    )
                    && !self.exhaustive_cases.contains(&expr.id)
                {
                    self.exhaustive_cases.push(expr.id);
                }
            }
            _ => {}
        }
    }

    fn visit_pattern_mut(&mut self, pattern: &mut Pattern<TypedRef<'db>>) {
        walk_pattern_mut(self, pattern);
        // A `String` or `Bytes` literal pattern compares through its type's
        // `==`, which a program without the prelude lacks.
        let well_known = self.checker.env.well_known_types();
        let (equality, type_name) = match &*pattern.kind {
            PatternKind::Literal(LiteralPattern::String(_)) => {
                (well_known.string_equality, "String")
            }
            PatternKind::Literal(LiteralPattern::Bytes(_)) => (well_known.bytes_equality, "Bytes"),
            _ => return,
        };
        if equality.is_none() {
            self.checker.report_type_error(
                pattern.id,
                format!("{type_name} literal patterns need the prelude's `{type_name}::==`"),
            );
        }
    }
}
