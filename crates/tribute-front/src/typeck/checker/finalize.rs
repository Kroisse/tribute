//! Apply solved substitutions and collect binder variables in checked bodies.
//!
//! These walks preserve the existing inference and generalization policy;
//! function orchestration supplies the solved state and binder mapping.

use super::TypeChecker;
use crate::ast::NodeId;
use crate::ast::visit::{RefSite, Visit, VisitMut, walk_expr_mut};
use crate::ast::{Expr, ExprKind, FuncDefId, ResolvedRef, Type, TypedRef, UniVarId};
use crate::typeck::solver::{RowSubst, TypeSubst};
use std::collections::HashMap;

impl<'db> TypeChecker<'db> {
    /// Apply substitution and generalization to all types in the body expression.
    ///
    /// This ensures that all TypedRef types have UniVars replaced with:
    /// 1. Their resolved concrete type (from substitution), or
    /// 2. The corresponding BoundVar (from generalization mapping)
    ///
    /// Deferred method calls become direct calls, and case expressions whose
    /// substituted arms cover their scrutinee are recorded as exhaustive.
    pub(super) fn apply_subst_to_body(
        &mut self,
        body: &mut Expr<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        deferred_resolutions: &HashMap<NodeId, (FuncDefId<'db>, Type<'db>)>,
    ) {
        Finalize {
            checker: self,
            type_subst,
            row_subst,
            var_to_index,
            deferred_resolutions,
        }
        .visit_expr_mut(body);
    }

    /// Apply substitution and generalization to a type.
    pub(super) fn apply_subst_to_type(
        &self,
        ty: Type<'db>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Type<'db> {
        // First apply the substitution to resolve UniVars
        let substituted = type_subst.apply_with_rows(self.db(), ty, row_subst);
        // Then apply the generalization mapping to convert remaining UniVars to BoundVars
        type_subst.apply_generalization_with_local_vars(
            self.db(),
            substituted,
            row_subst,
            var_to_index,
            &self.local_generalizations,
        )
    }

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
        struct Collect<'a, 'db> {
            db: &'db dyn salsa::Database,
            type_subst: &'a TypeSubst<'db>,
            row_subst: &'a RowSubst<'db>,
            out: &'a mut Vec<UniVarId<'db>>,
        }
        impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for Collect<'_, 'db> {
            fn visit_ref(&mut self, _: RefSite, _: NodeId, value: &'ast TypedRef<'db>) {
                self.type_subst.collect_univars_from_type(
                    self.db,
                    value.ty,
                    self.row_subst,
                    self.out,
                );
            }
        }
        Collect {
            db: self.db(),
            type_subst,
            row_subst,
            out,
        }
        .visit_expr(body);
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

/// Applies one function's solved substitution to its checked body.
struct Finalize<'a, 'db> {
    checker: &'a mut TypeChecker<'db>,
    type_subst: &'a TypeSubst<'db>,
    row_subst: &'a RowSubst<'db>,
    var_to_index: &'a HashMap<UniVarId<'db>, u32>,
    deferred_resolutions: &'a HashMap<NodeId, (FuncDefId<'db>, Type<'db>)>,
}

impl<'db> Finalize<'_, 'db> {
    fn apply(&self, ty: Type<'db>) -> Type<'db> {
        self.checker
            .apply_subst_to_type(ty, self.type_subst, self.row_subst, self.var_to_index)
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
                if let Some(scrutinee_ty) = self.checker.node_types.get(&scrutinee.id).copied()
                    && self
                        .checker
                        .check_exhaustiveness(scrutinee_ty, arms, scrutinee.id)
                    && !self.checker.exhaustive_cases.contains(&expr.id)
                {
                    self.checker.exhaustive_cases.push(expr.id);
                }
            }
            _ => {}
        }
    }
}
