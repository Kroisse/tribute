//! Expression type checking.
//!
//! All expression checking methods take a `FunctionInferenceContext` as parameter,
//! enabling per-function type inference with isolated constraints.
//!
//! This module dispatches on expression kinds; the checks for each feature
//! live in its submodules.

use salsa::Accumulator;

use tribute_core::{CompilationPhase, DiagnosticSeverity};

use crate::ast::{
    AbilityInstance, AbilityRow, BinOpKind, Expr, ExprKind, FieldInit, HandlerKind, NodeId,
    ResolvedRef, Stmt, Type, TypeKind, TypedRef,
};

use super::super::constraint::ConstraintOriginKind;
use super::super::func_context::FunctionInferenceContext;
use super::NodeDiagnostic;
use super::{Mode, TypeChecker};
use crate::typeck::InstantiatedPerformOperation;

mod annotation;
mod call;
mod ctor;
mod handler;
mod pat;
#[cfg(test)]
mod tests;

pub(crate) use call::{CallTypes, MethodSelection};

/// Check if a type contains any unification variables (UniVar).
///
/// This is used to determine if it's safe to use Mode::Check with the type.
/// If the type contains UniVars, using Mode::Check can cause ICE with
/// row-polymorphic effect types.
fn type_contains_univar<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    match ty.kind(db) {
        TypeKind::UniVar { .. } => true,
        TypeKind::Named { args, .. } => args.iter().any(|a| type_contains_univar(db, *a)),
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            params.iter().any(|p| type_contains_univar(db, *p))
                || type_contains_univar(db, *result)
                || ability_row_contains_univar(db, *effect)
        }
        TypeKind::Tuple(elems) => elems.iter().any(|e| type_contains_univar(db, *e)),
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            type_contains_univar(db, *arg)
                || type_contains_univar(db, *result)
                || ability_row_contains_univar(db, *effect)
        }
        TypeKind::Int
        | TypeKind::Nat
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Bytes
        | TypeKind::Rune
        | TypeKind::Nil
        | TypeKind::Never
        | TypeKind::BoundVar { .. }
        | TypeKind::LocalBoundVar { .. }
        | TypeKind::Error => false,
    }
}

/// Check if an ability row contains any unification variables.
fn ability_row_contains_univar<'db>(db: &'db dyn salsa::Database, row: AbilityRow<'db>) -> bool {
    for effect in row.instances(db) {
        for arg in &effect.args {
            if type_contains_univar(db, *arg) {
                return true;
            }
        }
    }
    false
}

impl<'db> TypeChecker<'db> {
    // =========================================================================
    // Expression checking
    // =========================================================================

    /// Type check an expression with a FunctionInferenceContext.
    pub(crate) fn check_expr_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: &Expr<ResolvedRef<'db>>,
        mode: Mode<'db>,
    ) -> Expr<TypedRef<'db>> {
        // An error type carries no expectation: it stands in for a type that
        // was already reported missing or invalid, so the expression is
        // inferred on its own.
        let mode = match mode {
            Mode::Check(expected) if matches!(expected.kind(self.db()), TypeKind::Error) => {
                Mode::Infer
            }
            mode => mode,
        };
        let lambda_expected = match mode {
            Mode::Infer => None,
            Mode::Check(expected) => Some(expected),
        };
        let is_lambda = matches!(*expr.kind, ExprKind::Lambda { .. });
        if is_lambda && let Some(checked) = ctx.checked_lambda(expr.id, lambda_expected) {
            return checked;
        }
        // Revisited literal lambdas keep the callable signature selected by
        // their context, even when their actual body has type Never.
        let mode = if matches!(mode, Mode::Infer)
            && matches!(*expr.kind, ExprKind::Lambda { .. })
            && let Some(existing) = ctx.get_node_type(expr.id)
        {
            Mode::Check(existing)
        } else {
            mode
        };
        let ty = match &*expr.kind {
            ExprKind::NatLit(_) => ctx.nat_type(),
            ExprKind::IntLit(_) => ctx.int_type(),
            ExprKind::FloatLit(_) => ctx.float_type(),
            ExprKind::BoolLit(_) => ctx.bool_type(),
            ExprKind::StringLit(_) => ctx.string_type(),
            ExprKind::BytesLit(_) => ctx.bytes_type(),
            ExprKind::Nil => ctx.nil_type(),
            ExprKind::RuneLit(_) => ctx.rune_type(),

            ExprKind::Var(resolved @ ResolvedRef::Local { .. }) => {
                self.infer_local_reference_with_ctx(ctx, expr.id, resolved)
            }
            ExprKind::Var(resolved) => self.infer_var_with_ctx(ctx, Some(expr.id), resolved),
            ExprKind::Call { callee, .. }
                if matches!(&*callee.kind, ExprKind::Functions { .. }) =>
            {
                self.infer_method_call_with_ctx(ctx, expr)
            }
            ExprKind::Call {
                callee,
                args,
                syntax,
            } => {
                ctx.note_call_syntax(expr.id, *syntax);
                let callee_ty = self.infer_expr_type_with_ctx(ctx, callee);
                // Conversion re-visits the callee. Keep this one ability-op
                // inference instance connected to that visit through dedicated
                // semantic state, never through the concrete node-type table.
                if matches!(&*callee.kind, ExprKind::Var(ResolvedRef::AbilityOp { .. })) {
                    ctx.record_ability_op_callee_type(callee.id, callee_ty);
                }
                let arg_types = self.infer_call_args_with_ctx(ctx, callee_ty, args);
                let result = self.infer_call_with_ctx(ctx, callee_ty, &arg_types, expr.id);
                if let (
                    ExprKind::Var(ResolvedRef::AbilityOp {
                        ability,
                        op: _,
                        kind,
                    }),
                    TypeKind::Func {
                        params,
                        result: operation_result,
                        effect,
                        ..
                    },
                ) = (&*callee.kind, callee_ty.kind(self.db()))
                {
                    let matching: Vec<_> = effect
                        .instances(self.db())
                        .iter()
                        .filter(|effect| effect.ability_id == *ability)
                        .collect();
                    if let [instance] = matching.as_slice() {
                        ctx.record_perform_operation(
                            expr.id,
                            InstantiatedPerformOperation {
                                ability: *ability,
                                ability_args: instance.args.clone(),
                                kind: *kind,
                                params: params.clone(),
                                result: *operation_result,
                            },
                        );
                    }
                }
                result
            }
            ExprKind::Cons { ctor, args } => {
                self.infer_constructor_with_ctx(ctx, expr.id, ctor, args, lambda_expected)
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => self.infer_record_type_with_ctx(
                ctx,
                expr.id,
                type_name,
                fields,
                spread.as_ref(),
                lambda_expected,
            ),
            ExprKind::Functions { candidates, name } => {
                self.infer_function_value_with_ctx(ctx, expr.id, name, candidates)
            }
            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs_ty = self.infer_expr_type_with_ctx(ctx, lhs);
                let rhs_ty = self.infer_expr_type_with_ctx(ctx, rhs);
                match op {
                    BinOpKind::And | BinOpKind::Or => {
                        let bool_ty = ctx.bool_type();
                        ctx.constrain_coerce(lhs_ty, bool_ty, lhs.id);
                        ctx.constrain_coerce(rhs_ty, bool_ty, rhs.id);
                        bool_ty
                    }
                }
            }
            ExprKind::Block { stmts, value } => {
                ctx.push_scope();
                for stmt in stmts {
                    self.infer_stmt_and_bind_with_ctx(ctx, stmt);
                }
                let ty = match &mode {
                    Mode::Check(expected) => self.infer_expr_with_expected(ctx, value, *expected),
                    Mode::Infer => self.infer_expr_type_with_ctx(ctx, value),
                };
                ctx.pop_scope();
                ty
            }
            ExprKind::Case { scrutinee, arms } => {
                // Infer scrutinee type
                let scrutinee_ty = self.infer_expr_type_with_ctx(ctx, scrutinee);

                let result_ty = ctx.begin_result_join(expr.id);
                for arm in arms {
                    ctx.push_scope();
                    let pattern_ty =
                        self.infer_pattern_type_with_ctx(ctx, &arm.pattern, Some(scrutinee_ty));
                    ctx.constrain_eq(pattern_ty, scrutinee_ty);
                    self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                    let arm_ty = self.infer_expr_type_with_ctx(ctx, &arm.body);
                    ctx.add_result_source(expr.id, arm.body.id, arm_ty);
                    ctx.pop_scope();
                }
                ctx.finish_result_join(expr.id);
                result_ty
            }
            ExprKind::Lambda {
                params,
                return_ty,
                effects,
                body,
            } => {
                // Extract expected effect from mode if checking against a function type.
                // This is crucial for lambdas passed to higher-order functions with effects.
                let contextual_effect = if let Mode::Check(expected_ty) = &mode
                    && let TypeKind::Func { effect, .. } = expected_ty.kind(self.db())
                {
                    Some(*effect)
                } else {
                    None
                };
                let contextual_result = match &mode {
                    Mode::Check(expected) => match expected.kind(self.db()) {
                        TypeKind::Func { result, .. } => Some(*result),
                        _ => None,
                    },
                    Mode::Infer => None,
                };
                // A written result or row is the lambda's own signature: the
                // body is checked against it, and the context is matched with
                // it (`new-plans/type-inference.md`, lambda annotations).
                let annotated_result = return_ty
                    .as_ref()
                    .map(|ann| self.annotation_to_type_with_ctx(ctx, ann));
                let annotated_effect = effects
                    .as_deref()
                    .map(|abilities| self.ability_annotations_to_row_with_ctx(ctx, abilities));
                if let (Some(annotated), Some(contextual)) = (annotated_effect, contextual_effect) {
                    ctx.constrain_row_eq_at(
                        contextual,
                        annotated,
                        expr.id,
                        ConstraintOriginKind::Lambda,
                    );
                }
                let expected_effect = annotated_effect.or(contextual_effect);
                let expected_result = annotated_result.or(contextual_result);

                // A revisited lambda keeps the parameter types of its first
                // visit. Its nested lambdas are cached with those types, so
                // fresh variables here would be unrelated to them whenever
                // the context is not a function type that links the two.
                let recorded_params =
                    ctx.get_node_type(expr.id)
                        .and_then(|ty| match ty.kind(self.db()) {
                            TypeKind::Func {
                                params: recorded, ..
                            } if recorded.len() == params.len() => Some(recorded.clone()),
                            _ => None,
                        });
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .enumerate()
                    .map(|(index, p)| match (&p.ty, &recorded_params) {
                        (Some(ann), _) => self.annotation_to_type_with_ctx(ctx, ann),
                        (None, Some(recorded)) => recorded[index],
                        (None, None) => ctx.fresh_type_var(),
                    })
                    .collect();

                // Save the outer context's effect - lambda has its own effect context
                let outer_effect = ctx.current_effect();

                let outer_contract = ctx.effect_contract;
                ctx.effect_contract = expected_effect;
                // Accumulate performed effects from the set identity. Contextual
                // row slack is introduced after accumulation, not as a source of
                // every union in the body.
                ctx.set_current_effect(AbilityRow::pure(self.db()));

                // Use a new scope for lambda parameters so they don't leak out
                ctx.push_scope();
                ctx.enter_lambda();
                ctx.push_callable_result(expected_result);
                let (lambda_evidence, outer_evidence) = ctx.evidence.enter_callable(None);

                // Bind lambda parameters in the new scope
                for (param, ty) in params.iter().zip(param_types.iter()) {
                    if let Some(local_id) = param.local_id {
                        ctx.bind_local(local_id, *ty);
                    }
                    ctx.bind_local_by_name(param.name.clone(), *ty);
                }

                let body_ty = match expected_result {
                    Some(result) => self.infer_expr_with_expected(ctx, body, result),
                    None => self.infer_expr_type_with_ctx(ctx, body),
                };

                // Match omitted-effect named functions: retain an already-open
                // residual row, otherwise reattach only the open tail supplied
                // by the contextual callable contract. An infer-only local
                // lambda therefore stays closed when its body is pure.
                let accumulated = ctx.current_effect();
                let become_result = ctx.pop_callable_result();
                ctx.pop_scope();
                ctx.evidence.restore(outer_evidence);
                let resume_effect = ctx.exit_lambda();
                let inferred_effect = match (accumulated.rest(self.db()), resume_effect) {
                    // Resuming performs the continuation's effects, so a closed
                    // local row joins the continuation row instead of naming
                    // its tail.
                    (None, Some(resume_effect)) => {
                        let mut effects = resume_effect.instances(self.db()).to_vec();
                        for effect in accumulated.instances(self.db()) {
                            if !effects.contains(effect) {
                                effects.push(effect.clone());
                            }
                        }
                        resume_effect.with_instances(self.db(), effects)
                    }
                    (None, None) => AbilityRow::new(
                        self.db(),
                        accumulated.instances(self.db()),
                        expected_effect
                            .and_then(|effect| effect.rest(self.db()).map(|_| ctx.fresh_row_var())),
                    ),
                    (Some(_), _) => accumulated,
                };

                // Restore the outer context's effect
                ctx.set_current_effect(outer_effect);

                // A `resume` selects the ability row of the captured continuation.
                // An open local row keeps its latent tail as the lambda's
                // signature and is constrained to the continuation row so
                // solving retains both the continuation identity and local
                // effects.
                if let Some(resume_effect) = resume_effect
                    && accumulated.rest(self.db()).is_some()
                {
                    ctx.constrain_row_eq_at(
                        inferred_effect,
                        resume_effect,
                        expr.id,
                        ConstraintOriginKind::Lambda,
                    );
                }
                ctx.effect_contract = outer_contract;
                let lambda_effect = inferred_effect;
                ctx.evidence
                    .set_callable_row(lambda_evidence, lambda_effect);
                if let Some(expected) = expected_effect {
                    // This resolves generic ability arguments selected by the body.
                    ctx.constrain_row_eq_at(
                        expected,
                        lambda_effect,
                        expr.id,
                        ConstraintOriginKind::Lambda,
                    );
                }

                // A `become` in an inferred lambda fixes the lambda's result
                // type, which the body's other results then flow into.
                let inferred_result = match become_result {
                    Some(result) => {
                        ctx.constrain_coerce(body_ty, result, body.id);
                        result
                    }
                    None => body_ty,
                };
                let result_ty = match expected_result {
                    Some(result) => {
                        ctx.constrain_coerce(body_ty, result, body.id);
                        result
                    }
                    None => inferred_result,
                };
                let lambda_type = ctx.func_type(param_types, result_ty, lambda_effect);
                ctx.record_lambda_signature(
                    expr.id,
                    crate::typeck::LambdaSignature {
                        function_type: lambda_type,
                    },
                );
                lambda_type
            }
            ExprKind::Handle { body, handlers } => {
                let answer_ty = ctx.begin_result_join(expr.id);
                // Handle expression:
                // 1. Infer the body's type (body may have effects)
                // 2. Extract handled effects from handler arms
                // 3. Remove handled effects from the current ability row
                // 4. Return the body's type

                // Save the current effect state before checking body
                let effect_before_body = ctx.current_effect();

                // A handler removes effects from the actual computation.
                ctx.set_current_effect(AbilityRow::pure(self.db()));

                // Infer the body's type
                let body_evidence = ctx.evidence.enter_handle_body(expr.id, None);
                let body_ty = self.infer_expr_type_with_ctx(ctx, body);
                if let Some((_, outer_evidence)) = body_evidence {
                    ctx.evidence.restore(outer_evidence);
                }

                // Extract handled ability IDs from effect handlers
                let mut handled_ability_ids = Vec::new();
                for handler in handlers {
                    let ability_id = match &handler.kind {
                        HandlerKind::Fn { ability, .. } | HandlerKind::Op { ability, .. } => {
                            // Get the ability ID from the ResolvedRef
                            self.extract_ability_id_from_ref(ability)
                        }
                        HandlerKind::Do { .. } => None,
                    };
                    let Some(ability_id) = ability_id else {
                        continue;
                    };
                    if ability_id.is_builtin_io(self.db()) {
                        NodeDiagnostic::new(
                            "builtin ambient ability `std::io::Io` cannot be handled",
                            handler.id,
                            DiagnosticSeverity::Error,
                            CompilationPhase::TypeChecking,
                        )
                        .accumulate(self.db());
                    } else {
                        handled_ability_ids.push(ability_id);
                    }
                }
                self.report_missing_handler_arms(ctx, expr.id, handlers, &handled_ability_ids);
                self.report_duplicate_handler_arms(ctx, handlers);

                // Get the body's effect after checking (may have effects added)
                let body_effect_after = ctx.current_effect();

                // The `do` arm is the ordinary completion boundary of a
                // handle expression. Its result is the source result of the
                // whole handle, which may differ from the handled body's
                // intermediate value (for example `String -> Input` while
                // translating a `Throw` into an input result).
                let completion_ty = handlers.iter().find_map(|handler| {
                    let HandlerKind::Do { binding } = &handler.kind else {
                        return None;
                    };
                    Some(ctx.with_scope(|ctx| {
                        let pattern_ty =
                            self.infer_pattern_type_with_ctx(ctx, binding, Some(body_ty));
                        ctx.constrain_eq(pattern_ty, body_ty);
                        self.bind_pattern_vars_with_ctx(ctx, binding, body_ty);
                        self.infer_expr_type_with_ctx(ctx, &handler.body)
                    }))
                });

                let mut selected_effects = Vec::new();
                for ability_id in &handled_ability_ids {
                    if selected_effects
                        .iter()
                        .any(|effect: &AbilityInstance<'db>| effect.ability_id == *ability_id)
                    {
                        continue;
                    }
                    let count = self
                        .env
                        .lookup_ability(*ability_id)
                        .map_or(0, |ability| ability.type_params.len());
                    selected_effects.push(AbilityInstance {
                        ability_id: *ability_id,
                        args: (0..count).map(|_| ctx.fresh_type_var()).collect(),
                    });
                }
                let handled_effects = AbilityRow::new(self.db(), selected_effects, None);
                if let Some((scope, _)) = body_evidence {
                    ctx.evidence.set_handled(scope, handled_effects);
                }
                // Create a result ability row that excludes handled effects
                // This is the effect that propagates out of the handle expression
                let result_effect = if handled_ability_ids.is_empty() {
                    // No effect handlers, body's effect propagates as-is
                    body_effect_after
                } else {
                    // Remove handled effects from the body's ability row
                    self.remove_handled_effects(ctx, body_effect_after, handled_effects)
                };

                // Push handle context with the ORIGINAL body effect (including handled effects).
                // The continuation `k` represents the rest of the original computation,
                // which can still perform the handled effects. When `k(x)` is called
                // (e.g., inside `fn() { k(init) }`), the lambda needs the full effect
                // row `{e, State(s)}` so it matches the expected `comp` parameter type.
                let completion_node = handlers
                    .iter()
                    .find(|handler| matches!(handler.kind, HandlerKind::Do { .. }))
                    .map_or(body.id, |handler| handler.body.id);
                ctx.add_result_source(expr.id, completion_node, completion_ty.unwrap_or(body_ty));
                ctx.push_handle_ctx(super::super::func_context::HandleContext {
                    node_id: expr.id,
                    answer_ty,
                    body_ty,
                    body_effect: body_effect_after,
                    handled_effects,
                    evidence_body: None,
                });

                // The handler contributes its residual requirements to the
                // surrounding computation; this is union, not row equality.
                ctx.set_current_effect(effect_before_body);
                ctx.merge_handler_effect_at(result_effect, expr.id);

                answer_ty
            }
            ExprKind::Tuple(elems) => {
                let elem_tys: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.infer_expr_type_with_ctx(ctx, e))
                    .collect();
                ctx.tuple_type(elem_tys)
            }
            ExprKind::List(elems) => {
                let elem_ty = if elems.is_empty() {
                    ctx.fresh_type_var()
                } else {
                    ctx.begin_result_join(expr.id)
                };
                for elem in elems.iter() {
                    let ty = self.infer_expr_type_with_ctx(ctx, elem);
                    ctx.add_result_source(expr.id, elem.id, ty);
                }
                if !elems.is_empty() {
                    ctx.finish_result_join(expr.id);
                }
                ctx.canonical_list_type(elem_ty)
            }
            ExprKind::Resume { arg, local_id } => {
                self.infer_resume_type_with_ctx(ctx, expr.id, arg, *local_id)
            }
            ExprKind::Become { call } => self.infer_become_type_with_ctx(ctx, expr.id, call),
            ExprKind::Error => ctx.error_type(),
        };

        // Check mode: constrain inferred type to match expected type.
        // For lambdas, we split the expected function type and constrain param/return types
        // individually, skipping only the effect portion (which is already constrained via
        // constrain_row_eq during lambda body checking). Constraining the full type can cause
        // ICE with row-polymorphic effect types.
        if let Mode::Check(expected) = mode {
            if matches!(&*expr.kind, ExprKind::Lambda { .. }) {
                if let (
                    TypeKind::Func {
                        params: exp_params,
                        result: exp_result,
                        ..
                    },
                    TypeKind::Func {
                        params: inf_params,
                        result: inf_result,
                        ..
                    },
                ) = (expected.kind(self.db()), ty.kind(self.db()))
                {
                    for (inf_p, exp_p) in inf_params.iter().zip(exp_params.iter()) {
                        ctx.constrain_eq(*inf_p, *exp_p);
                    }
                    ctx.constrain_coerce(*inf_result, *exp_result, expr.id);
                }
            } else {
                ctx.constrain_coerce(ty, expected, expr.id);
            }
        }

        // If a type was already recorded for this node (e.g., by infer_expr_type_with_ctx),
        // add a constraint to ensure consistency. This handles cases like Lambda where
        // type inference may occur twice (once in infer_* and once in check_*).
        if let Some(existing_ty) = ctx.get_node_type(expr.id) {
            ctx.constrain_eq(ty, existing_ty);
        }

        let lambda_signature_type = if matches!(&*expr.kind, ExprKind::Lambda { .. }) {
            match &mode {
                Mode::Check(expected)
                    if matches!(expected.kind(self.db()), TypeKind::Func { .. }) =>
                {
                    *expected
                }
                _ => ty,
            }
        } else {
            ty
        };
        if let ExprKind::Lambda { .. } = &*expr.kind
            && let TypeKind::Func { .. } = lambda_signature_type.kind(self.db())
        {
            ctx.record_checked_lambda_signature(
                expr.id,
                crate::typeck::LambdaSignature {
                    function_type: lambda_signature_type,
                },
            );
        }

        // Record node type
        ctx.record_node_type(expr.id, ty);

        let kind = self.convert_expr_kind_with_ctx(ctx, expr.id, &expr.kind);
        let expr = Expr::new(expr.id, kind);
        if is_lambda {
            ctx.record_checked_lambda(lambda_expected, expr.clone());
        }
        expr
    }

    /// Type `become call`: the call's own type, which must equal the result
    /// type of the enclosing callable.
    fn infer_become_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        become_id: NodeId,
        call: &Expr<ResolvedRef<'db>>,
    ) -> Type<'db> {
        let call_ty = self.infer_expr_type_with_ctx(ctx, call);
        if let Some(result) = ctx.become_result_type() {
            ctx.constrain_eq_at(call_ty, result, become_id, ConstraintOriginKind::Expression);
        }
        call_ty
    }

    /// Infer the type of an expression (just returns the type, doesn't convert).
    pub(super) fn infer_expr_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: &Expr<ResolvedRef<'db>>,
    ) -> Type<'db> {
        match &*expr.kind {
            ExprKind::NatLit(_) => ctx.nat_type(),
            ExprKind::IntLit(_) => ctx.int_type(),
            ExprKind::FloatLit(_) => ctx.float_type(),
            ExprKind::BoolLit(_) => ctx.bool_type(),
            ExprKind::StringLit(_) => ctx.string_type(),
            ExprKind::BytesLit(_) => ctx.bytes_type(),
            ExprKind::Nil => ctx.nil_type(),
            ExprKind::RuneLit(_) => ctx.rune_type(),
            ExprKind::Var(resolved @ ResolvedRef::Local { .. }) => {
                self.infer_local_reference_with_ctx(ctx, expr.id, resolved)
            }
            ExprKind::Var(resolved) => self.infer_var_with_ctx(ctx, Some(expr.id), resolved),
            ExprKind::Call { callee, .. }
                if matches!(&*callee.kind, ExprKind::Functions { .. }) =>
            {
                self.infer_method_call_with_ctx(ctx, expr)
            }
            ExprKind::Call {
                callee,
                args,
                syntax,
            } => {
                ctx.note_call_syntax(expr.id, *syntax);
                let callee_ty = self.infer_expr_type_with_ctx(ctx, callee);
                if matches!(&*callee.kind, ExprKind::Var(ResolvedRef::AbilityOp { .. })) {
                    ctx.record_ability_op_callee_type(callee.id, callee_ty);
                }
                let arg_types = self.infer_call_args_with_ctx(ctx, callee_ty, args);
                let result = self.infer_call_with_ctx(ctx, callee_ty, &arg_types, expr.id);
                if let (
                    ExprKind::Var(ResolvedRef::AbilityOp {
                        ability,
                        op: _,
                        kind,
                    }),
                    TypeKind::Func {
                        params,
                        result: operation_result,
                        effect,
                        ..
                    },
                ) = (&*callee.kind, callee_ty.kind(self.db()))
                {
                    let matching: Vec<_> = effect
                        .instances(self.db())
                        .iter()
                        .filter(|effect| effect.ability_id == *ability)
                        .collect();
                    if let [instance] = matching.as_slice() {
                        ctx.record_perform_operation(
                            expr.id,
                            InstantiatedPerformOperation {
                                ability: *ability,
                                ability_args: instance.args.clone(),
                                kind: *kind,
                                params: params.clone(),
                                result: *operation_result,
                            },
                        );
                    }
                }
                result
            }
            ExprKind::Cons { ctor, args } => {
                self.infer_constructor_with_ctx(ctx, expr.id, ctor, args, None)
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => self.infer_record_type_with_ctx(
                ctx,
                expr.id,
                type_name,
                fields,
                spread.as_ref(),
                None,
            ),
            ExprKind::Block { stmts, value } => {
                ctx.push_scope();
                for stmt in stmts {
                    self.infer_stmt_and_bind_with_ctx(ctx, stmt);
                }
                let ty = self.infer_expr_type_with_ctx(ctx, value);
                ctx.pop_scope();
                ty
            }
            ExprKind::Case { scrutinee, arms } => {
                let scrutinee_ty = self.infer_expr_type_with_ctx(ctx, scrutinee);
                let result_ty = ctx.begin_result_join(expr.id);
                for arm in arms {
                    ctx.push_scope();
                    let pattern_ty =
                        self.infer_pattern_type_with_ctx(ctx, &arm.pattern, Some(scrutinee_ty));
                    ctx.constrain_eq(pattern_ty, scrutinee_ty);
                    self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                    let arm_ty = self.infer_expr_type_with_ctx(ctx, &arm.body);
                    ctx.add_result_source(expr.id, arm.body.id, arm_ty);
                    ctx.pop_scope();
                }
                ctx.finish_result_join(expr.id);
                result_ty
            }
            ExprKind::Functions { candidates, name } => {
                self.infer_function_value_with_ctx(ctx, expr.id, name, candidates)
            }
            ExprKind::Lambda { .. } | ExprKind::Handle { .. } => {
                // These constructs need their scoped bodies checked before a
                // surrounding let can solve/generalize the result relation.
                self.check_expr_with_ctx(ctx, expr, Mode::Infer);
                ctx.get_node_type(expr.id)
                    .expect("checked expression has a type")
            }
            ExprKind::Resume { arg, local_id } => {
                self.infer_resume_type_with_ctx(ctx, expr.id, arg, *local_id)
            }
            ExprKind::Become { call } => self.infer_become_type_with_ctx(ctx, expr.id, call),
            ExprKind::Tuple(elems) => {
                let elem_tys = elems
                    .iter()
                    .map(|elem| self.infer_expr_type_with_ctx(ctx, elem))
                    .collect();
                ctx.tuple_type(elem_tys)
            }
            ExprKind::List(elems) => {
                let elem_ty = if elems.is_empty() {
                    ctx.fresh_type_var()
                } else {
                    ctx.begin_result_join(expr.id)
                };
                for elem in elems {
                    let ty = self.infer_expr_type_with_ctx(ctx, elem);
                    ctx.add_result_source(expr.id, elem.id, ty);
                }
                if !elems.is_empty() {
                    ctx.finish_result_join(expr.id);
                }
                ctx.canonical_list_type(elem_ty)
            }
            _ => ctx.fresh_type_var(),
        }
    }

    /// Literal callables are checked at their definition site; existing values
    /// keep exact equality for every nested argument and result slot.
    fn infer_expr_with_expected(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: &Expr<ResolvedRef<'db>>,
        expected: Type<'db>,
    ) -> Type<'db> {
        if matches!(expected.kind(self.db()), TypeKind::Func { .. })
            && matches!(*expr.kind, ExprKind::Lambda { .. } | ExprKind::Block { .. })
        {
            self.check_expr_with_ctx(ctx, expr, Mode::Check(expected));
            ctx.get_node_type(expr.id)
                .expect("checked expression has a type")
        } else {
            // An implicit constructor takes its constructor from the type
            // expected of it.
            match &*expr.kind {
                ExprKind::Cons {
                    ctor: ctor @ ResolvedRef::ImplicitConstructor { .. },
                    args,
                } => self.infer_constructor_with_ctx(ctx, expr.id, ctor, args, Some(expected)),
                ExprKind::Record {
                    type_name: type_name @ ResolvedRef::ImplicitConstructor { .. },
                    fields,
                    spread,
                } => self.infer_record_type_with_ctx(
                    ctx,
                    expr.id,
                    type_name,
                    fields,
                    spread.as_ref(),
                    Some(expected),
                ),
                _ => self.infer_expr_type_with_ctx(ctx, expr),
            }
        }
    }

    /// Keep a literal's contextual return slot open until UFCS selects its
    /// parameter type. The actual body is checked directionally against this
    /// slot; the deferred callable itself still uses ordinary equality.
    fn infer_deferred_argument(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: &Expr<ResolvedRef<'db>>,
    ) -> Type<'db> {
        if let Some(expected) = Self::literal_callable_shape(ctx, expr) {
            self.infer_expr_with_expected(ctx, expr, expected)
        } else {
            self.infer_expr_type_with_ctx(ctx, expr)
        }
    }

    fn literal_callable_shape(
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: &Expr<ResolvedRef<'db>>,
    ) -> Option<Type<'db>> {
        match &*expr.kind {
            ExprKind::Lambda { params, body, .. } => {
                let params = params.iter().map(|_| ctx.fresh_type_var()).collect();
                let result =
                    Self::literal_callable_shape(ctx, body).unwrap_or_else(|| ctx.fresh_type_var());
                let effect = ctx.fresh_ability_row();
                Some(ctx.func_type(params, result, effect))
            }
            ExprKind::Block { value, .. } => Self::literal_callable_shape(ctx, value),
            _ => None,
        }
    }

    pub(super) fn report_type_error(&self, node: NodeId, message: String) {
        NodeDiagnostic::new(
            message,
            node,
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        )
        .accumulate(self.db());
    }

    // =========================================================================
    // Expression conversion
    // =========================================================================

    /// Convert an expression kind from ResolvedRef to TypedRef.
    fn convert_expr_kind_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr_id: crate::ast::NodeId,
        kind: &ExprKind<ResolvedRef<'db>>,
    ) -> ExprKind<TypedRef<'db>> {
        match kind {
            ExprKind::NatLit(n) => ExprKind::NatLit(*n),
            ExprKind::IntLit(n) => ExprKind::IntLit(*n),
            ExprKind::FloatLit(f) => ExprKind::FloatLit(*f),
            ExprKind::BoolLit(b) => ExprKind::BoolLit(*b),
            ExprKind::StringLit(s) => ExprKind::StringLit(s.clone()),
            ExprKind::BytesLit(b) => ExprKind::BytesLit(b.clone()),
            ExprKind::Nil => ExprKind::Nil,
            ExprKind::RuneLit(r) => ExprKind::RuneLit(*r),
            ExprKind::Var(resolved) => {
                ExprKind::Var(self.convert_ref_with_ctx(ctx, Some(expr_id), resolved))
            }
            ExprKind::Call {
                callee,
                args,
                syntax,
            } if matches!(&*callee.kind, ExprKind::Functions { .. }) => {
                let args = args
                    .iter()
                    .map(|a| self.check_expr_with_ctx(ctx, a, Mode::Infer))
                    .collect();
                // A function selected during inference is the callee; one
                // still unselected stays for the solver to decide.
                let callee = match ctx.get_resolved_method(expr_id) {
                    Some((func_id, callee_ty)) => Expr::new(
                        callee.id,
                        ExprKind::Var(TypedRef {
                            resolved: ResolvedRef::Function { id: func_id },
                            ty: callee_ty,
                        }),
                    ),
                    None => Expr::new(
                        callee.id,
                        self.convert_expr_kind_with_ctx(ctx, callee.id, &callee.kind),
                    ),
                };
                ExprKind::Call {
                    callee,
                    args,
                    syntax: *syntax,
                }
            }
            ExprKind::Call {
                callee,
                args,
                syntax,
            } => {
                let inferred_ability_op_callee = ctx.get_ability_op_callee_type(callee.id);
                // First, process callee so its type gets recorded
                let converted_callee = self.check_expr_with_ctx(ctx, callee, Mode::Infer);

                // Now get callee's type (recorded during check_expr_with_ctx)
                let callee_ty = ctx.get_node_type(converted_callee.id);

                if let (Some(inferred), Some(converted)) = (inferred_ability_op_callee, callee_ty) {
                    ctx.constrain_eq(inferred, converted);
                }

                // Extract param types if callee is a function type.
                // This enables propagating expected types to lambda arguments,
                // which is crucial for effect type inference.
                let param_types: Vec<Option<Type<'db>>> = match callee_ty {
                    Some(ty) => {
                        if let TypeKind::Func { params, .. } = ty.kind(self.db()) {
                            params.iter().map(|p| Some(*p)).collect()
                        } else {
                            vec![None; args.len()]
                        }
                    }
                    None => vec![None; args.len()],
                };

                let converted_args: Vec<_> = args
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        // Use Mode::Check only if param type is available and doesn't contain
                        // unification variables. Using Check mode with UniVar-containing types
                        // can cause ICE with row-polymorphic effect types.
                        let mode = match param_types.get(i) {
                            Some(Some(ty)) if !type_contains_univar(self.db(), *ty) => {
                                Mode::Check(*ty)
                            }
                            _ => Mode::Infer,
                        };
                        self.check_expr_with_ctx(ctx, a, mode)
                    })
                    .collect();

                ExprKind::Call {
                    callee: converted_callee,
                    args: converted_args,
                    syntax: *syntax,
                }
            }
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor: self.convert_ref_with_ctx(
                    ctx,
                    Some(expr_id),
                    &self.constructor_reference(ctx, expr_id, ctor),
                ),
                args: args
                    .iter()
                    .map(|a| self.check_expr_with_ctx(ctx, a, Mode::Infer))
                    .collect(),
            },
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => ExprKind::Record {
                type_name: TypedRef {
                    ty: self.instantiate_value_constructor_with_ctx(ctx, expr_id, type_name),
                    resolved: self.constructor_reference(ctx, expr_id, type_name),
                },
                fields: fields
                    .iter()
                    .map(|f| FieldInit {
                        id: f.id,
                        name: f.name.clone(),
                        value: self.check_expr_with_ctx(ctx, &f.value, Mode::Infer),
                    })
                    .collect(),
                spread: spread
                    .as_ref()
                    .map(|e| self.check_expr_with_ctx(ctx, e, Mode::Infer)),
            },
            ExprKind::Functions { name, candidates } => ExprKind::Functions {
                name: name.clone(),
                // Unselected, so the candidates have no type.
                candidates: candidates.map(|candidate| TypedRef {
                    resolved: candidate.clone(),
                    ty: ctx.error_type(),
                }),
            },
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op: *op,
                lhs: self.check_expr_with_ctx(ctx, lhs, Mode::Infer),
                rhs: self.check_expr_with_ctx(ctx, rhs, Mode::Infer),
            },
            ExprKind::Block { stmts, value } => {
                ctx.push_scope();
                let converted_stmts: Vec<_> = stmts
                    .iter()
                    .map(|s| self.convert_stmt_with_ctx(ctx, s))
                    .collect();
                let converted_value = self.check_expr_with_ctx(ctx, value, Mode::Infer);
                ctx.pop_scope();
                ExprKind::Block {
                    stmts: converted_stmts,
                    value: converted_value,
                }
            }
            ExprKind::Case { scrutinee, arms } => {
                let scrutinee_expr = self.check_expr_with_ctx(ctx, scrutinee, Mode::Infer);
                let scrutinee_ty = ctx
                    .get_node_type(scrutinee_expr.id)
                    .unwrap_or_else(|| ctx.fresh_type_var());

                // Note: result_ty is NOT created here - it was already created and recorded
                // in check_expr_with_ctx. The arm body types are constrained to the result
                // type during pattern processing in check_expr_with_ctx.

                let converted_arms: Vec<_> = arms
                    .iter()
                    .map(|arm| {
                        // Each arm gets its own scope for pattern bindings
                        ctx.push_scope();

                        let pattern_ty =
                            self.infer_pattern_type_with_ctx(ctx, &arm.pattern, Some(scrutinee_ty));
                        ctx.constrain_eq(pattern_ty, scrutinee_ty);
                        self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                        let converted = self.convert_arm_with_scrutinee_ctx(ctx, arm, scrutinee_ty);

                        ctx.pop_scope();
                        converted
                    })
                    .collect();

                ExprKind::Case {
                    scrutinee: scrutinee_expr,
                    arms: converted_arms,
                }
            }
            ExprKind::Lambda {
                params,
                return_ty,
                effects,
                body,
            } => {
                // Save and restore effect context for lambda body,
                // just like in the infer phase. This prevents lambda
                // body effects from leaking into the enclosing scope.
                let outer_effect = ctx.current_effect();
                let outer_contract = ctx.effect_contract;
                ctx.effect_contract =
                    ctx.get_node_type(expr_id)
                        .and_then(|ty| match ty.kind(self.db()) {
                            TypeKind::Func { effect, .. } => Some(*effect),
                            _ => None,
                        });
                ctx.set_current_effect(AbilityRow::pure(self.db()));
                let (_, outer_evidence) = ctx.evidence.enter_callable(ctx.effect_contract);

                // Bind lambda parameters so that references to them in the body
                // resolve to their concrete types (not fresh UniVars). Without this,
                // TDNR cannot determine receiver types for method calls inside lambdas.
                let param_types = ctx
                    .get_node_type(expr_id)
                    .and_then(|ty| match ty.kind(self.db()) {
                        TypeKind::Func { params, .. } => Some(params.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        params
                            .iter()
                            .map(|p| match &p.ty {
                                Some(ann) => self.annotation_to_type_with_ctx(ctx, ann),
                                None => ctx.fresh_type_var(),
                            })
                            .collect()
                    });
                ctx.push_scope();
                for (param, ty) in params.iter().zip(param_types.iter()) {
                    tracing::debug!(
                        "convert Lambda: binding param '{}' (local_id={:?}) to {:?}",
                        param.name,
                        param.local_id,
                        ty.kind(self.db())
                    );
                    if let Some(local_id) = param.local_id {
                        ctx.bind_local(local_id, *ty);
                    }
                    ctx.bind_local_by_name(param.name.clone(), *ty);
                }

                let body_mode = ctx
                    .get_node_type(expr_id)
                    .and_then(|ty| match ty.kind(self.db()) {
                        TypeKind::Func { result, .. } => Some(Mode::Check(*result)),
                        _ => None,
                    })
                    .unwrap_or(Mode::Infer);
                let converted_body = self.check_expr_with_ctx(ctx, body, body_mode);

                ctx.pop_scope();
                ctx.evidence.restore(outer_evidence);
                ctx.effect_contract = outer_contract;
                ctx.set_current_effect(outer_effect);
                ExprKind::Lambda {
                    params: params.clone(),
                    return_ty: return_ty.clone(),
                    effects: effects.clone(),
                    body: converted_body,
                }
            }
            ExprKind::Handle { body, handlers } => {
                let mut handle_ctx = ctx.pop_handle_ctx().expect(
                    "pop_handle_ctx should match a corresponding push_handle_ctx in infer phase",
                );
                // Save and restore effect context for handle body conversion,
                // just like in the infer phase. The body has its own effect
                // context; without isolation, the body's effects would leak
                // into the enclosing scope during convert-phase re-inference.
                let outer_effect = ctx.current_effect();
                ctx.set_current_effect(AbilityRow::pure(self.db()));
                let body_evidence = ctx
                    .evidence
                    .enter_handle_body(expr_id, Some(handle_ctx.handled_effects));
                let converted_body = self.check_expr_with_ctx(ctx, body, Mode::Infer);
                if let Some((scope, outer_evidence)) = body_evidence {
                    ctx.evidence.restore(outer_evidence);
                    handle_ctx.evidence_body = Some(scope);
                }
                ctx.set_current_effect(outer_effect);
                let handlers = handlers
                    .iter()
                    .map(|h| self.convert_handler_arm_with_ctx(ctx, h, &handle_ctx))
                    .collect();
                ctx.finish_result_join(expr_id);
                ExprKind::Handle {
                    body: converted_body,
                    handlers,
                }
            }
            ExprKind::Become { call } => {
                let converted = self.check_expr_with_ctx(ctx, call, Mode::Infer);
                if matches!(&*call.kind, ExprKind::Call { callee, .. } if matches!(&*callee.kind, ExprKind::Functions { .. }))
                    && let ExprKind::Call { callee, .. } = &*converted.kind
                {
                    match &*callee.kind {
                        ExprKind::Var(TypedRef {
                            resolved: ResolvedRef::Function { id },
                            ..
                        }) => self.check_become_callee(expr_id, *id),
                        // Still unselected; checked once the solver has
                        // selected its function.
                        _ => ctx.record_become_method_operand(call.id),
                    }
                }
                ExprKind::Become { call: converted }
            }
            ExprKind::Resume { arg, local_id } => ExprKind::Resume {
                arg: self.check_expr_with_ctx(ctx, arg, Mode::Infer),
                local_id: *local_id,
            },
            ExprKind::Tuple(elements) => ExprKind::Tuple(
                elements
                    .iter()
                    .map(|e| self.check_expr_with_ctx(ctx, e, Mode::Infer))
                    .collect(),
            ),
            ExprKind::List(elements) => ExprKind::List(
                elements
                    .iter()
                    .map(|e| self.check_expr_with_ctx(ctx, e, Mode::Infer))
                    .collect(),
            ),
            ExprKind::Error => ExprKind::Error,
        }
    }

    /// Convert a ResolvedRef to a TypedRef.
    fn convert_ref_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node_id: Option<crate::ast::NodeId>,
        resolved: &ResolvedRef<'db>,
    ) -> TypedRef<'db> {
        let ty = match (node_id, resolved) {
            (
                Some(_),
                ResolvedRef::Constructor { .. } | ResolvedRef::ImplicitConstructor { .. },
            ) => self.infer_var_with_ctx(ctx, node_id, resolved),
            (Some(node), ResolvedRef::Local { id, name }) => ctx
                .lookup_local_reference(node, *id, name)
                .unwrap_or_else(|| ctx.fresh_type_var()),
            (Some(node), _) => ctx
                .get_function_reference_type(node)
                .or_else(|| ctx.get_node_type(node))
                .unwrap_or_else(|| self.infer_var_with_ctx(ctx, node_id, resolved)),
            (None, _) => self.infer_var_with_ctx(ctx, node_id, resolved),
        };
        TypedRef {
            resolved: resolved.clone(),
            ty,
        }
    }

    /// Infer a statement's type and bind its pattern variables.
    /// This is used during type inference to process block statements before
    /// inferring the block value's type.
    fn infer_stmt_and_bind_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        stmt: &Stmt<ResolvedRef<'db>>,
    ) {
        match stmt {
            Stmt::Let {
                pattern, value, ty, ..
            } => {
                let outer_rows = ctx.annotation_row_names();
                let outer_effect = ctx.current_effect();
                ctx.set_current_effect(AbilityRow::pure(self.db()));
                let value_ty = if let Some(ann) = ty {
                    let expected = self.annotation_to_type_with_ctx(ctx, ann);
                    let inferred = self.infer_expr_with_expected(ctx, value, expected);
                    ctx.constrain_coerce(inferred, expected, value.id);
                    expected
                } else {
                    self.infer_expr_type_with_ctx(ctx, value)
                };
                let evaluation_effect = ctx.current_effect();
                ctx.set_current_effect(outer_effect);
                ctx.merge_effect(evaluation_effect);
                // Constrain pattern type to match value type
                let pattern_ty = self.infer_pattern_type_with_ctx(ctx, pattern, Some(value_ty));
                ctx.constrain_eq(pattern_ty, value_ty);
                self.generalize_and_bind_pattern_with_ctx(ctx, pattern, value, value_ty);
                ctx.end_let_row_scope(&outer_rows);
            }
            Stmt::Expr { expr, .. } => {
                // Just infer the type for side effects (constraints)
                self.infer_expr_type_with_ctx(ctx, expr);
            }
        }
    }

    /// Convert a statement.
    fn convert_stmt_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        stmt: &Stmt<ResolvedRef<'db>>,
    ) -> Stmt<TypedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                value,
                ty,
            } => {
                let outer_rows = ctx.annotation_row_names();
                let outer_effect = ctx.current_effect();
                ctx.set_current_effect(AbilityRow::pure(self.db()));
                let checked = if let Some(ann) = ty {
                    let expected = self.annotation_to_type_with_ctx(ctx, ann);
                    self.check_expr_with_ctx(ctx, value, Mode::Check(expected))
                } else {
                    self.check_expr_with_ctx(ctx, value, Mode::Infer)
                };
                let value_ty = if let Some(ann) = ty {
                    self.annotation_to_type_with_ctx(ctx, ann)
                } else {
                    ctx.get_node_type(checked.id)
                        .unwrap_or_else(|| ctx.fresh_type_var())
                };
                let evaluation_effect = ctx.current_effect();
                ctx.set_current_effect(outer_effect);
                ctx.merge_effect(evaluation_effect);

                // Constrain pattern type to match value type
                let pattern_ty = self.infer_pattern_type_with_ctx(ctx, pattern, Some(value_ty));
                ctx.constrain_eq(pattern_ty, value_ty);

                self.generalize_and_bind_pattern_with_ctx(ctx, pattern, value, value_ty);
                ctx.end_let_row_scope(&outer_rows);
                let pattern = self.convert_pattern_with_ctx(ctx, pattern);

                Stmt::Let {
                    id: *id,
                    pattern,
                    value: checked,
                    ty: ty.clone(),
                }
            }
            Stmt::Expr { id, expr } => {
                let expr = self.check_expr_with_ctx(ctx, expr, Mode::Infer);
                Stmt::Expr { id: *id, expr }
            }
        }
    }
}
