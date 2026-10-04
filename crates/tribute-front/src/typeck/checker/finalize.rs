//! Apply solved substitutions and collect binder variables in checked bodies.
//!
//! These walks preserve the existing inference and generalization policy;
//! function orchestration supplies the solved state and binder mapping.

use super::TypeChecker;
use crate::ast::NodeId;
use crate::ast::visit::{RefSite, Refs, VisitMut, walk_expr, walk_expr_mut};
use crate::ast::{Expr, ExprKind, FuncDefId, ResolvedRef, Type, TypedRef, UniVarId};
use crate::typeck::solver::{RowSubst, TypeSubst};
use rustc_hash::FxHashMap;
use std::collections::HashSet;

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
        deferred_resolutions: &FxHashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)>,
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
    pub(super) var_to_index: &'a FxHashMap<UniVarId<'db>, u32>,
    pub(super) local_generalizations: &'a FxHashMap<UniVarId<'db>, (NodeId, u32)>,
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
/// Deferred method calls become direct calls, and case expressions whose
/// substituted arms cover their scrutinee are recorded as exhaustive.
pub(super) struct Finalize<'a, 'db> {
    pub(super) checker: &'a TypeChecker<'db>,
    pub(super) substitution: &'a Substitution<'a, 'db>,
    pub(super) deferred_resolutions: &'a FxHashMap<NodeId, (FuncDefId<'db>, Type<'db>)>,
    pub(super) node_types: &'a FxHashMap<NodeId, Type<'db>>,
    pub(super) exhaustive_cases: &'a mut Vec<NodeId>,
    pub(super) exhaustiveness_reported: &'a mut HashSet<NodeId>,
}

impl<'db> Finalize<'_, 'db> {
    fn apply(&self, ty: Type<'db>) -> Type<'db> {
        self.substitution.apply(ty)
    }
}

impl<'db> VisitMut<TypedRef<'db>> for Finalize<'_, 'db> {
    fn visit_ref_mut(&mut self, _: RefSite, _: NodeId, value: &mut TypedRef<'db>) {
        value.ty = self.apply(value.ty);
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr<TypedRef<'db>>) {
        // Children first: the rewrites below read substituted subtrees.
        walk_expr_mut(self, expr);
        match &mut *expr.kind {
            ExprKind::MethodCall { .. } => {
                // A deferred method resolved during solving becomes a call;
                // an unresolved one stays a MethodCall for TDNR.
                let Some(&(func_id, callee_ty)) = self.deferred_resolutions.get(&expr.id) else {
                    return;
                };
                let callee = Expr::new(
                    expr.id,
                    ExprKind::Var(TypedRef {
                        resolved: ResolvedRef::Function { id: func_id },
                        ty: self.apply(callee_ty),
                    }),
                );
                let ExprKind::MethodCall { receiver, args, .. } =
                    std::mem::replace(&mut *expr.kind, ExprKind::Error)
                else {
                    unreachable!("matched above");
                };
                let mut all_args = Vec::with_capacity(args.len() + 1);
                all_args.push(receiver);
                all_args.extend(args);
                *expr.kind = ExprKind::Call {
                    callee,
                    args: all_args,
                };
            }
            ExprKind::Case { scrutinee, arms } => {
                if let Some(scrutinee_ty) = self.node_types.get(&scrutinee.id).copied()
                    && self.checker.check_exhaustiveness(
                        self.exhaustiveness_reported,
                        scrutinee_ty,
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
}
