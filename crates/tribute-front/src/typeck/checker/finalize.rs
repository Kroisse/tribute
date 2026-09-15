//! Apply solved substitutions and collect binder variables in checked bodies.
//!
//! These walks preserve the existing inference and generalization policy;
//! function orchestration supplies the solved state and binder mapping.

use super::TypeChecker;
use crate::ast::{
    Arm, Expr, ExprKind, FieldPattern, FuncDefId, HandlerArm, HandlerKind, Pattern, PatternKind,
    ResolvedRef, Stmt, Type, TypedRef, UniVarId,
};
use crate::typeck::solver::{RowSubst, TypeSubst};
use std::collections::HashMap;

impl<'db> TypeChecker<'db> {
    /// Apply substitution and generalization to all types in the body expression.
    ///
    /// This ensures that all TypedRef types have UniVars replaced with:
    /// 1. Their resolved concrete type (from substitution), or
    /// 2. The corresponding BoundVar (from generalization mapping)
    pub(super) fn apply_subst_to_body(
        &mut self,
        body: Expr<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        deferred_resolutions: &HashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)>,
    ) -> Expr<TypedRef<'db>> {
        let kind = self.apply_subst_to_expr_kind(
            body.id,
            *body.kind,
            type_subst,
            row_subst,
            var_to_index,
            deferred_resolutions,
        );
        Expr::new(body.id, kind)
    }

    /// Apply substitution to an expression kind.
    fn apply_subst_to_expr_kind(
        &mut self,
        node_id: crate::ast::NodeId,
        kind: ExprKind<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        deferred_resolutions: &HashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)>,
    ) -> ExprKind<TypedRef<'db>> {
        match kind {
            ExprKind::NatLit(n) => ExprKind::NatLit(n),
            ExprKind::IntLit(n) => ExprKind::IntLit(n),
            ExprKind::FloatLit(f) => ExprKind::FloatLit(f),
            ExprKind::BoolLit(b) => ExprKind::BoolLit(b),
            ExprKind::StringLit(s) => ExprKind::StringLit(s),
            ExprKind::BytesLit(b) => ExprKind::BytesLit(b),
            ExprKind::Nil => ExprKind::Nil,
            ExprKind::RuneLit(r) => ExprKind::RuneLit(r),
            ExprKind::Error => ExprKind::Error,

            ExprKind::Var(typed_ref) => ExprKind::Var(self.apply_subst_to_typed_ref(
                typed_ref,
                type_subst,
                row_subst,
                var_to_index,
            )),
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: self.apply_subst_to_body(
                    callee,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
                args: args
                    .into_iter()
                    .map(|a| {
                        self.apply_subst_to_body(
                            a,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect(),
            },
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor: self.apply_subst_to_typed_ref(ctor, type_subst, row_subst, var_to_index),
                args: args
                    .into_iter()
                    .map(|a| {
                        self.apply_subst_to_body(
                            a,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect(),
            },
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => ExprKind::Record {
                type_name: self.apply_subst_to_typed_ref(
                    type_name,
                    type_subst,
                    row_subst,
                    var_to_index,
                ),
                fields: fields
                    .into_iter()
                    .map(|(name, expr)| {
                        (
                            name,
                            self.apply_subst_to_body(
                                expr,
                                type_subst,
                                row_subst,
                                var_to_index,
                                deferred_resolutions,
                            ),
                        )
                    })
                    .collect(),
                spread: spread.map(|e| {
                    self.apply_subst_to_body(
                        e,
                        type_subst,
                        row_subst,
                        var_to_index,
                        deferred_resolutions,
                    )
                }),
            },
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let converted_receiver = self.apply_subst_to_body(
                    receiver,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                );
                let converted_args: Vec<_> = args
                    .into_iter()
                    .map(|a| {
                        self.apply_subst_to_body(
                            a,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect();

                if let Some((func_id, callee_ty)) = deferred_resolutions.get(&node_id) {
                    // Deferred method was resolved — convert to Call
                    let substituted_ty =
                        self.apply_subst_to_type(*callee_ty, type_subst, row_subst, var_to_index);
                    let callee_ref = TypedRef {
                        resolved: ResolvedRef::Function { id: *func_id },
                        ty: substituted_ty,
                    };
                    let callee = Expr::new(node_id, ExprKind::Var(callee_ref));
                    let mut all_args = vec![converted_receiver];
                    all_args.extend(converted_args);
                    ExprKind::Call {
                        callee,
                        args: all_args,
                    }
                } else {
                    // Still unresolved — keep as MethodCall for TDNR
                    ExprKind::MethodCall {
                        receiver: converted_receiver,
                        method,
                        args: converted_args,
                    }
                }
            }
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: self.apply_subst_to_body(
                    lhs,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
                rhs: self.apply_subst_to_body(
                    rhs,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
            },
            ExprKind::Block { stmts, value } => ExprKind::Block {
                stmts: stmts
                    .into_iter()
                    .map(|s| {
                        self.apply_subst_to_stmt(
                            s,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect(),
                value: self.apply_subst_to_body(
                    value,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
            },
            ExprKind::Case { scrutinee, arms } => {
                let scrutinee = self.apply_subst_to_body(
                    scrutinee,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                );
                let arms = arms
                    .into_iter()
                    .map(|arm| {
                        self.apply_subst_to_arm(
                            arm,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect::<Vec<_>>();

                if let Some(scrutinee_ty) = self.node_types.get(&scrutinee.id).copied()
                    && self.check_exhaustiveness(scrutinee_ty, &arms, scrutinee.id)
                    && !self.exhaustive_cases.contains(&node_id)
                {
                    self.exhaustive_cases.push(node_id);
                }

                ExprKind::Case { scrutinee, arms }
            }
            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.apply_subst_to_body(
                    body,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
            },
            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.apply_subst_to_body(
                    body,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
                handlers: handlers
                    .into_iter()
                    .map(|h| {
                        self.apply_subst_to_handler_arm(
                            h,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect(),
            },
            ExprKind::Resume { arg, local_id } => ExprKind::Resume {
                arg: self.apply_subst_to_body(
                    arg,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
                local_id,
            },
            ExprKind::Tuple(elems) => ExprKind::Tuple(
                elems
                    .into_iter()
                    .map(|e| {
                        self.apply_subst_to_body(
                            e,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect(),
            ),
            ExprKind::List(elems) => ExprKind::List(
                elems
                    .into_iter()
                    .map(|e| {
                        self.apply_subst_to_body(
                            e,
                            type_subst,
                            row_subst,
                            var_to_index,
                            deferred_resolutions,
                        )
                    })
                    .collect(),
            ),
        }
    }

    /// Apply substitution to a TypedRef.
    fn apply_subst_to_typed_ref(
        &self,
        typed_ref: TypedRef<'db>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> TypedRef<'db> {
        let ty = self.apply_subst_to_type(typed_ref.ty, type_subst, row_subst, var_to_index);
        TypedRef {
            resolved: typed_ref.resolved,
            ty,
        }
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

    /// Apply substitution to a statement.
    fn apply_subst_to_stmt(
        &mut self,
        stmt: Stmt<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        deferred_resolutions: &HashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)>,
    ) -> Stmt<TypedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                value,
                ty,
            } => Stmt::Let {
                id,
                pattern: self.apply_subst_to_pattern(pattern, type_subst, row_subst, var_to_index),
                value: self.apply_subst_to_body(
                    value,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
                ty,
            },
            Stmt::Expr { id, expr } => Stmt::Expr {
                id,
                expr: self.apply_subst_to_body(
                    expr,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                ),
            },
        }
    }

    /// Apply substitution to a case arm.
    fn apply_subst_to_arm(
        &mut self,
        arm: Arm<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        deferred_resolutions: &HashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)>,
    ) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: self.apply_subst_to_pattern(arm.pattern, type_subst, row_subst, var_to_index),
            guard: arm.guard.map(|g| {
                self.apply_subst_to_body(
                    g,
                    type_subst,
                    row_subst,
                    var_to_index,
                    deferred_resolutions,
                )
            }),
            body: self.apply_subst_to_body(
                arm.body,
                type_subst,
                row_subst,
                var_to_index,
                deferred_resolutions,
            ),
        }
    }

    /// Apply substitution to a handler arm.
    fn apply_subst_to_handler_arm(
        &mut self,
        arm: HandlerArm<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        deferred_resolutions: &HashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)>,
    ) -> HandlerArm<TypedRef<'db>> {
        let kind = match arm.kind {
            HandlerKind::Do { binding } => HandlerKind::Do {
                binding: self.apply_subst_to_pattern(binding, type_subst, row_subst, var_to_index),
            },
            HandlerKind::Fn {
                ability,
                op,
                params,
            } => HandlerKind::Fn {
                ability: self.apply_subst_to_typed_ref(
                    ability,
                    type_subst,
                    row_subst,
                    var_to_index,
                ),
                op,
                params: params
                    .into_iter()
                    .map(|p| self.apply_subst_to_pattern(p, type_subst, row_subst, var_to_index))
                    .collect(),
            },
            HandlerKind::Op {
                ability,
                op,
                params,
                resume_local_id,
            } => HandlerKind::Op {
                ability: self.apply_subst_to_typed_ref(
                    ability,
                    type_subst,
                    row_subst,
                    var_to_index,
                ),
                op,
                params: params
                    .into_iter()
                    .map(|p| self.apply_subst_to_pattern(p, type_subst, row_subst, var_to_index))
                    .collect(),
                resume_local_id,
            },
        };
        HandlerArm {
            id: arm.id,
            kind,
            body: self.apply_subst_to_body(
                arm.body,
                type_subst,
                row_subst,
                var_to_index,
                deferred_resolutions,
            ),
        }
    }

    /// Apply substitution to a pattern.
    fn apply_subst_to_pattern(
        &self,
        pattern: Pattern<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Pattern<TypedRef<'db>> {
        let kind = match *pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind { name, local_id },
            PatternKind::Literal(lit) => PatternKind::Literal(lit),
            PatternKind::Error => PatternKind::Error,
            PatternKind::Variant { ctor, fields } => PatternKind::Variant {
                ctor: self.apply_subst_to_typed_ref(ctor, type_subst, row_subst, var_to_index),
                fields: fields
                    .into_iter()
                    .map(|p| self.apply_subst_to_pattern(p, type_subst, row_subst, var_to_index))
                    .collect(),
            },
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => PatternKind::Record {
                type_name: type_name
                    .map(|t| self.apply_subst_to_typed_ref(t, type_subst, row_subst, var_to_index)),
                fields: fields
                    .into_iter()
                    .map(|f| {
                        self.apply_subst_to_field_pattern(f, type_subst, row_subst, var_to_index)
                    })
                    .collect(),
                rest,
            },
            PatternKind::Tuple(pats) => PatternKind::Tuple(
                pats.into_iter()
                    .map(|p| self.apply_subst_to_pattern(p, type_subst, row_subst, var_to_index))
                    .collect(),
            ),
            PatternKind::List(pats) => PatternKind::List(
                pats.into_iter()
                    .map(|p| self.apply_subst_to_pattern(p, type_subst, row_subst, var_to_index))
                    .collect(),
            ),
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => PatternKind::ListRest {
                head: head
                    .into_iter()
                    .map(|p| self.apply_subst_to_pattern(p, type_subst, row_subst, var_to_index))
                    .collect(),
                rest,
                rest_local_id,
            },
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.apply_subst_to_pattern(pattern, type_subst, row_subst, var_to_index),
                name,
                local_id,
            },
        };
        Pattern::new(pattern.id, kind)
    }

    /// Apply substitution to a field pattern.
    fn apply_subst_to_field_pattern(
        &self,
        fp: FieldPattern<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            id: fp.id,
            name: fp.name,
            pattern: fp
                .pattern
                .map(|p| self.apply_subst_to_pattern(p, type_subst, row_subst, var_to_index)),
        }
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
        self.collect_univars_from_expr_kind(&body.kind, type_subst, row_subst, out);
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

    fn collect_univars_from_expr_kind(
        &self,
        kind: &ExprKind<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        match kind {
            ExprKind::NatLit(_)
            | ExprKind::IntLit(_)
            | ExprKind::FloatLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BytesLit(_)
            | ExprKind::Nil
            | ExprKind::RuneLit(_)
            | ExprKind::Error => {}

            ExprKind::Var(typed_ref) => {
                type_subst.collect_univars_from_type(self.db(), typed_ref.ty, row_subst, out);
            }
            ExprKind::Call { callee, args } => {
                self.collect_univars_from_body(callee, type_subst, row_subst, out);
                for arg in args {
                    self.collect_univars_from_body(arg, type_subst, row_subst, out);
                }
            }
            ExprKind::Cons { ctor, args } => {
                type_subst.collect_univars_from_type(self.db(), ctor.ty, row_subst, out);
                for arg in args {
                    self.collect_univars_from_body(arg, type_subst, row_subst, out);
                }
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                type_subst.collect_univars_from_type(self.db(), type_name.ty, row_subst, out);
                for (_, expr) in fields {
                    self.collect_univars_from_body(expr, type_subst, row_subst, out);
                }
                if let Some(e) = spread {
                    self.collect_univars_from_body(e, type_subst, row_subst, out);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.collect_univars_from_body(receiver, type_subst, row_subst, out);
                for arg in args {
                    self.collect_univars_from_body(arg, type_subst, row_subst, out);
                }
            }
            ExprKind::BinOp { lhs, rhs, .. } => {
                self.collect_univars_from_body(lhs, type_subst, row_subst, out);
                self.collect_univars_from_body(rhs, type_subst, row_subst, out);
            }
            ExprKind::Block { stmts, value } => {
                for stmt in stmts {
                    self.collect_univars_from_stmt(stmt, type_subst, row_subst, out);
                }
                self.collect_univars_from_body(value, type_subst, row_subst, out);
            }
            ExprKind::Case { scrutinee, arms } => {
                self.collect_univars_from_body(scrutinee, type_subst, row_subst, out);
                for arm in arms {
                    self.collect_univars_from_pattern(&arm.pattern, type_subst, row_subst, out);
                    if let Some(g) = &arm.guard {
                        self.collect_univars_from_body(g, type_subst, row_subst, out);
                    }
                    self.collect_univars_from_body(&arm.body, type_subst, row_subst, out);
                }
            }
            ExprKind::Lambda { body, .. } => {
                self.collect_univars_from_body(body, type_subst, row_subst, out);
            }
            ExprKind::Handle { body, handlers } => {
                self.collect_univars_from_body(body, type_subst, row_subst, out);
                for handler in handlers {
                    match &handler.kind {
                        HandlerKind::Do { binding } => {
                            self.collect_univars_from_pattern(binding, type_subst, row_subst, out);
                        }
                        HandlerKind::Fn {
                            ability, params, ..
                        }
                        | HandlerKind::Op {
                            ability, params, ..
                        } => {
                            type_subst.collect_univars_from_type(
                                self.db(),
                                ability.ty,
                                row_subst,
                                out,
                            );
                            for p in params {
                                self.collect_univars_from_pattern(p, type_subst, row_subst, out);
                            }
                        }
                    }
                    self.collect_univars_from_body(&handler.body, type_subst, row_subst, out);
                }
            }
            ExprKind::Resume { arg, .. } => {
                self.collect_univars_from_body(arg, type_subst, row_subst, out);
            }
            ExprKind::Tuple(elems) | ExprKind::List(elems) => {
                for elem in elems {
                    self.collect_univars_from_body(elem, type_subst, row_subst, out);
                }
            }
        }
    }

    fn collect_univars_from_stmt(
        &self,
        stmt: &Stmt<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        match stmt {
            Stmt::Let { pattern, value, .. } => {
                self.collect_univars_from_pattern(pattern, type_subst, row_subst, out);
                self.collect_univars_from_body(value, type_subst, row_subst, out);
            }
            Stmt::Expr { expr, .. } => {
                self.collect_univars_from_body(expr, type_subst, row_subst, out);
            }
        }
    }

    fn collect_univars_from_pattern(
        &self,
        pattern: &Pattern<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        match &*pattern.kind {
            PatternKind::Wildcard
            | PatternKind::Bind { .. }
            | PatternKind::Literal(_)
            | PatternKind::Error => {}
            PatternKind::Variant { ctor, fields } => {
                type_subst.collect_univars_from_type(self.db(), ctor.ty, row_subst, out);
                for f in fields {
                    self.collect_univars_from_pattern(f, type_subst, row_subst, out);
                }
            }
            PatternKind::Record {
                type_name, fields, ..
            } => {
                if let Some(t) = type_name {
                    type_subst.collect_univars_from_type(self.db(), t.ty, row_subst, out);
                }
                for f in fields {
                    if let Some(p) = &f.pattern {
                        self.collect_univars_from_pattern(p, type_subst, row_subst, out);
                    }
                }
            }
            PatternKind::Tuple(pats) | PatternKind::List(pats) => {
                for p in pats {
                    self.collect_univars_from_pattern(p, type_subst, row_subst, out);
                }
            }
            PatternKind::ListRest { head, .. } => {
                for p in head {
                    self.collect_univars_from_pattern(p, type_subst, row_subst, out);
                }
            }
            PatternKind::As { pattern, .. } => {
                self.collect_univars_from_pattern(pattern, type_subst, row_subst, out);
            }
        }
    }
}
