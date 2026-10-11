//! Expression type checking.
//!
//! All expression checking methods take a `FunctionInferenceContext` as parameter,
//! enabling per-function type inference with isolated constraints.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use itertools::Itertools;
use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::ModulePathExt as _;
use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, Arm, BinOpKind, Effect, EffectRow, Expr, ExprKind, FieldInit, FieldPattern,
    HandlerArm, HandlerKind, LiteralPattern, LocalId, ModulePath, NodeId, OpDeclKind, Pattern,
    PatternKind, ResolvedRef, Stmt, Type, TypeKind, TypeScheme, TypedRef, collect_effect_vars,
};

use super::super::constraint::ConstraintOriginKind;
use super::super::func_context::{FunctionInferenceContext, LetSchemeBinding};
use super::super::solver::{RowSubst, TypeSolver, TypeSubst};
use super::super::subst;
use super::{Mode, TypeChecker};
use crate::typeck::{InstantiatedHandlerOperation, InstantiatedPerformOperation};

/// Source and typing context needed to resolve one handler operation arm.
struct HandlerOperationRequest<'a, 'db> {
    ability: &'a ResolvedRef<'db>,
    op: Symbol,
    syntax_kind: OpDeclKind,
    params: &'a [Pattern<ResolvedRef<'db>>],
    arm_id: NodeId,
    handle_ctx: &'a super::super::func_context::HandleContext<'db>,
}

/// One local introduced by a pattern, together with its resolved binding type.
struct PatternBinding<'db> {
    name: Symbol,
    local_id: Option<LocalId>,
    scope: NodeId,
    ty: Type<'db>,
}

/// The function a method call selects for its receiver's type.
pub(crate) enum MethodSelection<'db> {
    One(crate::typeck::MethodEntry<'db>),
    /// Several functions a path may name take the receiver.
    Ambiguous(Vec<crate::ast::FuncDefId<'db>>),
    None,
}

/// The types of a call that selects among several functions: the arguments
/// inferred so far, receiver first, of the `arity` it passes, and the type
/// its result is used at.
#[derive(Clone, Copy)]
pub(crate) struct CallTypes<'a, 'db> {
    pub args: &'a [Type<'db>],
    pub arity: usize,
    pub result: Option<Type<'db>>,
}

impl<'db> CallTypes<'_, 'db> {
    /// Whether a function taking `params` takes the arguments inferred so
    /// far, whatever number of arguments the call passes.
    fn arguments_match(self, db: &'db dyn salsa::Database, params: &[Type<'db>]) -> bool {
        self.args
            .iter()
            .zip(params)
            .all(|(actual, declared)| crate::typeck::parameter_type_matches(db, *declared, *actual))
    }

    /// Whether a function from `params` to `result` is one the call may
    /// select.
    fn matches(
        self,
        db: &'db dyn salsa::Database,
        params: &[Type<'db>],
        result: Type<'db>,
    ) -> bool {
        params.len() == self.arity
            && self.arguments_match(db, params)
            && self
                .result
                .is_none_or(|actual| crate::typeck::parameter_type_matches(db, result, actual))
    }
}

/// A qualified method call's path as its node and the functions it may name.
pub(crate) fn method_path_functions<'db>(
    path: &crate::ast::MethodPath<ResolvedRef<'db>>,
) -> (NodeId, Vec<crate::ast::FuncDefId<'db>>) {
    let functions = path
        .candidates
        .iter()
        .filter_map(|candidate| match candidate {
            ResolvedRef::Function { id } => Some(*id),
            _ => None,
        })
        .collect();
    (path.id, functions)
}

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
                || effect_row_contains_univar(db, *effect)
        }
        TypeKind::Tuple(elems) => elems.iter().any(|e| type_contains_univar(db, *e)),
        TypeKind::App { ctor, args } => {
            type_contains_univar(db, *ctor) || args.iter().any(|a| type_contains_univar(db, *a))
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            type_contains_univar(db, *arg)
                || type_contains_univar(db, *result)
                || effect_row_contains_univar(db, *effect)
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

/// Check if an effect row contains any unification variables.
fn effect_row_contains_univar<'db>(db: &'db dyn salsa::Database, row: EffectRow<'db>) -> bool {
    for effect in row.effects(db) {
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
            ExprKind::Call { callee, args } => {
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
                        .effects(self.db())
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
                let ctor_ty = self.infer_var_with_ctx(ctx, Some(expr.id), ctor);
                if args.is_empty() {
                    // Unit constructor (e.g., None) - just return the constructor type
                    ctor_ty
                } else {
                    // Constructor with arguments (e.g., Some(x)) - treat as function call
                    let arg_types = self.infer_call_args_with_ctx(ctx, ctor_ty, args);
                    self.infer_call_with_ctx(ctx, ctor_ty, &arg_types, expr.id)
                }
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => self.infer_record_type_with_ctx(ctx, expr.id, type_name, fields, spread.as_ref()),
            ExprKind::MethodCall { .. } => self.infer_method_call_with_ctx(ctx, expr),
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
                    let pattern_ty = self.infer_pattern_type_with_ctx(ctx, &arm.pattern);
                    ctx.constrain_eq(pattern_ty, scrutinee_ty);
                    self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                    let arm_ty = self.infer_expr_type_with_ctx(ctx, &arm.body);
                    ctx.add_result_source(expr.id, arm.body.id, arm_ty);
                    ctx.pop_scope();
                }
                ctx.finish_result_join(expr.id);
                result_ty
            }
            ExprKind::Lambda { params, body } => {
                // Extract expected effect from mode if checking against a function type.
                // This is crucial for lambdas passed to higher-order functions with effects.
                let expected_effect = if let Mode::Check(expected_ty) = &mode
                    && let TypeKind::Func { effect, .. } = expected_ty.kind(self.db())
                {
                    Some(*effect)
                } else {
                    None
                };

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
                ctx.set_current_effect(EffectRow::pure(self.db()));

                // Use a new scope for lambda parameters so they don't leak out
                ctx.push_scope();
                ctx.enter_lambda();
                ctx.push_callable_result(match &mode {
                    Mode::Check(expected) => match expected.kind(self.db()) {
                        TypeKind::Func { result, .. } => Some(*result),
                        _ => None,
                    },
                    Mode::Infer => None,
                });
                let (lambda_evidence, outer_evidence) = ctx.evidence.enter_callable(None);

                // Bind lambda parameters in the new scope
                for (param, ty) in params.iter().zip(param_types.iter()) {
                    if let Some(local_id) = param.local_id {
                        ctx.bind_local(local_id, *ty);
                    }
                    ctx.bind_local_by_name(param.name.clone(), *ty);
                }

                let body_ty = match &mode {
                    Mode::Check(expected) => match expected.kind(self.db()) {
                        TypeKind::Func { result, .. } => {
                            self.infer_expr_with_expected(ctx, body, *result)
                        }
                        _ => self.infer_expr_type_with_ctx(ctx, body),
                    },
                    Mode::Infer => self.infer_expr_type_with_ctx(ctx, body),
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
                        let mut effects = resume_effect.effects(self.db()).to_vec();
                        for effect in accumulated.effects(self.db()) {
                            if !effects.contains(effect) {
                                effects.push(effect.clone());
                            }
                        }
                        EffectRow::new(self.db(), effects, resume_effect.rest(self.db()))
                    }
                    (None, None) => EffectRow::new(
                        self.db(),
                        accumulated.effects(self.db()),
                        expected_effect
                            .and_then(|effect| effect.rest(self.db()).map(|_| ctx.fresh_row_var())),
                    ),
                    (Some(_), _) => accumulated,
                };

                // Restore the outer context's effect
                ctx.set_current_effect(outer_effect);

                // A `resume` selects the effect row of the captured continuation.
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
                let result_ty = match mode {
                    Mode::Check(expected) => match expected.kind(self.db()) {
                        TypeKind::Func { result, .. } => {
                            ctx.constrain_coerce(body_ty, *result, body.id);
                            *result
                        }
                        _ => inferred_result,
                    },
                    Mode::Infer => inferred_result,
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
                // 3. Remove handled effects from the current effect row
                // 4. Return the body's type

                // Save the current effect state before checking body
                let effect_before_body = ctx.current_effect();

                // A handler removes effects from the actual computation.
                ctx.set_current_effect(EffectRow::pure(self.db()));

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
                        Diagnostic::new(
                            "builtin ambient ability `std::io::Io` cannot be handled",
                            self.get_span(handler.id),
                            DiagnosticSeverity::Error,
                            CompilationPhase::TypeChecking,
                        )
                        .accumulate(self.db());
                    } else {
                        handled_ability_ids.push(ability_id);
                    }
                }
                self.report_missing_handler_arms(ctx, expr.id, handlers, &handled_ability_ids);

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
                        let pattern_ty = self.infer_pattern_type_with_ctx(ctx, binding);
                        ctx.constrain_eq(pattern_ty, body_ty);
                        self.bind_pattern_vars_with_ctx(ctx, binding, body_ty);
                        self.infer_expr_type_with_ctx(ctx, &handler.body)
                    }))
                });

                let mut selected_effects = Vec::new();
                for ability_id in &handled_ability_ids {
                    if selected_effects
                        .iter()
                        .any(|effect: &Effect<'db>| effect.ability_id == *ability_id)
                    {
                        continue;
                    }
                    let count = self
                        .env
                        .lookup_ability(*ability_id)
                        .map_or(0, |ability| ability.type_params.len());
                    selected_effects.push(Effect {
                        ability_id: *ability_id,
                        args: (0..count).map(|_| ctx.fresh_type_var()).collect(),
                    });
                }
                let handled_effects = EffectRow::new(self.db(), selected_effects, None);
                if let Some((scope, _)) = body_evidence {
                    ctx.evidence.set_handled(scope, handled_effects);
                }
                // Create a result effect row that excludes handled effects
                // This is the effect that propagates out of the handle expression
                let result_effect = if handled_ability_ids.is_empty() {
                    // No effect handlers, body's effect propagates as-is
                    body_effect_after
                } else {
                    // Remove handled effects from the body's effect row
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

        // Convert expression (MethodCall needs special handling for expr.id)
        let kind = self.convert_expr_kind_with_ctx(ctx, expr.id, &expr.kind);
        let expr = Expr::new(expr.id, kind);
        if is_lambda {
            ctx.record_checked_lambda(lambda_expected, expr.clone());
        }
        expr
    }

    /// Infer the type of an expression (just returns the type, doesn't convert).
    /// Type `resume arg` against the continuation bound by the enclosing
    /// `op` arm.
    fn infer_resume_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        resume: NodeId,
        arg: &Expr<ResolvedRef<'db>>,
        local_id: Option<LocalId>,
    ) -> Type<'db> {
        let arg_ty = self.infer_expr_type_with_ctx(ctx, arg);
        let Some(local_id) = local_id else {
            return ctx.fresh_type_var();
        };
        if let Some((ability, op)) = ctx.non_resumptive_resume_op(local_id) {
            if ctx.mark_handler_error(resume, "resume in Never operation") {
                Diagnostic::new(
                    format!(
                        "cannot `resume` in the handler for `{ability}::{op}`, which returns `Never`"
                    ),
                    self.get_span(resume),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            return ctx.error_type();
        }
        if let Some(cont_ty) = ctx.lookup_local(local_id)
            && let TypeKind::Continuation {
                arg: cont_arg,
                result: cont_result,
                effect,
            } = cont_ty.kind(self.db())
        {
            ctx.constrain_coerce(arg_ty, *cont_arg, arg.id);
            ctx.record_lambda_resume_effect(*effect);
            ctx.evidence.record_resume(resume, local_id);
            *cont_result
        } else {
            ctx.fresh_type_var()
        }
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

    fn infer_expr_type_with_ctx(
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
            ExprKind::Call { callee, args } => {
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
                        .effects(self.db())
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
                let ctor_ty = self.infer_var_with_ctx(ctx, Some(expr.id), ctor);
                if args.is_empty() {
                    // Unit constructor (e.g., None) - just return the constructor type
                    ctor_ty
                } else {
                    // Constructor with arguments (e.g., Some(x)) - treat as function call
                    let arg_types = self.infer_call_args_with_ctx(ctx, ctor_ty, args);
                    self.infer_call_with_ctx(ctx, ctor_ty, &arg_types, expr.id)
                }
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => self.infer_record_type_with_ctx(ctx, expr.id, type_name, fields, spread.as_ref()),
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
                    let pattern_ty = self.infer_pattern_type_with_ctx(ctx, &arm.pattern);
                    ctx.constrain_eq(pattern_ty, scrutinee_ty);
                    self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                    let arm_ty = self.infer_expr_type_with_ctx(ctx, &arm.body);
                    ctx.add_result_source(expr.id, arm.body.id, arm_ty);
                    ctx.pop_scope();
                }
                ctx.finish_result_join(expr.id);
                result_ty
            }
            ExprKind::MethodCall { .. } => self.infer_method_call_with_ctx(ctx, expr),
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
            self.infer_expr_type_with_ctx(ctx, expr)
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
            ExprKind::Lambda { params, body } => {
                let params = params.iter().map(|_| ctx.fresh_type_var()).collect();
                let result =
                    Self::literal_callable_shape(ctx, body).unwrap_or_else(|| ctx.fresh_type_var());
                let effect = ctx.fresh_effect_row();
                Some(ctx.func_type(params, result, effect))
            }
            ExprKind::Block { value, .. } => Self::literal_callable_shape(ctx, value),
            _ => None,
        }
    }

    fn infer_call_args_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        callee: Type<'db>,
        args: &[Expr<ResolvedRef<'db>>],
    ) -> Vec<Type<'db>> {
        let params = match callee.kind(self.db()) {
            TypeKind::Func { params, .. } => params.as_slice(),
            _ => &[],
        };
        args.iter()
            .enumerate()
            .map(|(index, arg)| {
                if let Some(expected) = params.get(index) {
                    self.infer_expr_with_expected(ctx, arg, *expected)
                } else {
                    self.infer_expr_type_with_ctx(ctx, arg)
                }
            })
            .collect()
    }

    /// Keep the full constructor instance, including field types and effect rows,
    /// stable across inference and conversion of this source occurrence.
    fn instantiate_value_constructor_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node_id: NodeId,
        resolved: &ResolvedRef<'db>,
    ) -> Type<'db> {
        if let Some(ty) = ctx.get_constructor_reference_type(node_id) {
            return ty;
        }
        let ty = match resolved {
            ResolvedRef::Module { path } => {
                self.report_module_reference(ctx, node_id, *path, "a constructor")
            }
            _ => self.infer_var_with_ctx(ctx, None, resolved),
        };
        ctx.record_constructor_reference_type(node_id, ty);
        ty
    }

    /// Infer a record literal's nominal type and constrain its fields and spread.
    fn infer_record_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        record_id: NodeId,
        type_name: &ResolvedRef<'db>,
        fields: &[FieldInit<ResolvedRef<'db>>],
        spread: Option<&Expr<ResolvedRef<'db>>>,
    ) -> Type<'db> {
        let ctor_ty = self.instantiate_value_constructor_with_ctx(ctx, record_id, type_name);
        let struct_ty = if let TypeKind::Func { result, .. } = ctor_ty.kind(self.db()) {
            *result
        } else {
            ctor_ty
        };

        self.validate_record_shape_with_ctx(
            ctx,
            record_id,
            type_name,
            struct_ty,
            fields,
            spread.is_some(),
        );

        let written: Vec<Symbol> = fields.iter().map(|f| f.name.clone()).collect();
        let (_, _, variant_field_tys) =
            self.constructor_field_shape(ctx, record_id, type_name, &written);
        for (field, variant_field_ty) in fields.iter().zip(variant_field_tys) {
            let (field_name, field_expr) = (&field.name, &field.value);
            // Struct fields are read from the struct declaration; a named
            // variant's fields are its constructor instance's parameters.
            if let Some(expected_field_ty) = self
                .lookup_struct_field_type(struct_ty, field_name)
                .or(variant_field_ty)
            {
                let field_ty = self.infer_expr_with_expected(ctx, field_expr, expected_field_ty);
                ctx.constrain_coerce(field_ty, expected_field_ty, field_expr.id);
            } else {
                self.infer_expr_type_with_ctx(ctx, field_expr);
            }
        }
        if let Some(spread_expr) = spread {
            let spread_ty = self.infer_expr_type_with_ctx(ctx, spread_expr);
            ctx.constrain_coerce(spread_ty, struct_ty, spread_expr.id);
        }

        struct_ty
    }

    /// Validate declaration-owned field names once, independently of child typing.
    fn validate_record_shape_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        record_id: NodeId,
        type_name: &ResolvedRef<'db>,
        struct_ty: Type<'db>,
        fields: &[FieldInit<ResolvedRef<'db>>],
        has_spread: bool,
    ) {
        let ResolvedRef::Constructor { id, .. } = type_name else {
            return;
        };
        let (struct_id, _) = self.extract_struct_info(struct_ty);
        let Some(declared_fields) = struct_id.and_then(|id| self.env.lookup_struct_fields(id))
        else {
            self.validate_variant_record_shape_with_ctx(
                ctx, record_id, *id, struct_ty, fields, has_spread,
            );
            return;
        };
        if !ctx.mark_record_shape_checked(record_id) {
            return;
        }

        let declared: Vec<Symbol> = declared_fields
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        self.report_field_shape(
            record_id,
            format_args!("struct `{}`", id.qualified(self.db())),
            &declared,
            fields.iter().map(|f| f.name.clone()),
            has_spread,
        );
    }

    /// Validate a record literal that constructs an enum variant: the variant
    /// must have named fields, every field must be written, and a spread is
    /// not allowed because an enum value's variant is not known statically.
    fn validate_variant_record_shape_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        record_id: NodeId,
        id: crate::ast::CtorId<'db>,
        result: Type<'db>,
        fields: &[FieldInit<ResolvedRef<'db>>],
        has_spread: bool,
    ) {
        let is_variant = matches!(
            result.kind(self.db()),
            TypeKind::Named { name, .. } if self.env.lookup_enum_variants(name).is_some()
        );
        if !is_variant || !ctx.mark_record_shape_checked(record_id) {
            return;
        }
        let qualified = id.qualified(self.db());
        let Some(declared) = self.env.lookup_constructor_field_names(id) else {
            self.report_type_error(
                record_id,
                format!(
                    "`{qualified}` has positional fields; construct it with `{}(...)`",
                    id.name(self.db())
                ),
            );
            return;
        };
        if has_spread {
            self.report_type_error(
                record_id,
                format!(
                    "record spread is not allowed for variant `{qualified}`; write every field"
                ),
            );
        }
        self.report_field_shape(
            record_id,
            format_args!("variant `{qualified}`"),
            declared,
            fields.iter().map(|f| f.name.clone()),
            has_spread,
        );
    }

    /// Report written field names that do not fit the declared ones: unknown
    /// and duplicate fields in source order, then the first missing field in
    /// declaration order unless the rest are omitted explicitly.
    fn report_field_shape(
        &self,
        node: NodeId,
        owner: std::fmt::Arguments<'_>,
        declared: &[Symbol],
        written: impl IntoIterator<Item = Symbol>,
        omits_rest: bool,
    ) {
        let mut seen = HashSet::default();
        for name in written {
            if !declared.contains(&name) {
                self.report_type_error(node, format!("unknown field `{name}` for {owner}"));
            } else if !seen.insert(name.clone()) {
                self.report_type_error(node, format!("duplicate field `{name}`"));
            }
        }
        if !omits_rest && let Some(missing) = declared.iter().find(|name| !seen.contains(name)) {
            self.report_type_error(node, format!("missing field: {missing}"));
        }
    }

    pub(super) fn report_type_error(&self, node: NodeId, message: String) {
        Diagnostic::new(
            message,
            self.get_span(node),
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        )
        .accumulate(self.db());
    }

    /// Check that a positional constructor pattern names a constructor and
    /// has one sub-pattern per field. Reported once per pattern.
    fn validate_variant_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        ctor: &ResolvedRef<'db>,
        ctor_ty: Type<'db>,
        written: usize,
    ) {
        match ctor {
            ResolvedRef::Constructor { .. }
                if matches!(ctor_ty.kind(self.db()), TypeKind::Error) => {}
            ResolvedRef::Constructor { id, .. } => {
                let declared = match ctor_ty.kind(self.db()) {
                    TypeKind::Func { params, .. } => params.len(),
                    _ => 0,
                };
                if written != declared && ctx.mark_record_shape_checked(pattern_id) {
                    let plural = if declared == 1 { "" } else { "s" };
                    self.report_type_error(
                        pattern_id,
                        format!(
                            "constructor `{}` expects {declared} field{plural}, but the pattern has {written}",
                            id.qualified(self.db())
                        ),
                    );
                }
            }
            _ => self.report_non_constructor_pattern(ctx, pattern_id, ctor),
        }
    }

    /// Report a name that resolved to something other than a constructor in
    /// constructor position. Unresolved names and modules are reported where
    /// they are resolved.
    fn report_non_constructor_pattern(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        resolved: &ResolvedRef<'db>,
    ) {
        let name = match resolved {
            ResolvedRef::TypeDef { id } => id.qualified(self.db()),
            ResolvedRef::Local { id, name } if !id.is_unresolved() => name,
            _ => return,
        };
        if ctx.mark_record_shape_checked(pattern_id) {
            self.report_type_error(pattern_id, format!("`{name}` is not a constructor"));
        }
    }

    /// The constructor instance of a brace-form constructor pattern or record
    /// literal, the type it constructs, and the field type of each written
    /// field (`None` for a name the constructor does not declare).
    fn constructor_field_shape(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        type_name: &ResolvedRef<'db>,
        written: &[Symbol],
    ) -> (Type<'db>, Type<'db>, Vec<Option<Type<'db>>>) {
        let ctor_ty = self.instantiate_value_constructor_with_ctx(ctx, pattern_id, type_name);
        let (params, result) = match ctor_ty.kind(self.db()) {
            TypeKind::Func { params, result, .. } => (params.as_slice(), *result),
            _ => (&[][..], ctor_ty),
        };
        let declared = match type_name {
            ResolvedRef::Constructor { id, .. } => self.env.lookup_constructor_field_names(*id),
            _ => None,
        };
        let field_tys = written
            .iter()
            .map(|name| {
                let index = declared?.iter().position(|declared| declared == name)?;
                params.get(index).copied()
            })
            .collect();
        (ctor_ty, result, field_tys)
    }

    /// Check a brace-form constructor pattern's field names against the
    /// constructor's declaration. Reported once per pattern.
    fn validate_record_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        type_name: &ResolvedRef<'db>,
        result: Type<'db>,
        written: &[Symbol],
        omits_rest: bool,
    ) {
        let ResolvedRef::Constructor { id, .. } = type_name else {
            self.report_non_constructor_pattern(ctx, pattern_id, type_name);
            return;
        };
        if matches!(result.kind(self.db()), TypeKind::Error)
            || !ctx.mark_record_shape_checked(pattern_id)
        {
            return;
        }
        let qualified = id.qualified(self.db());
        let Some(declared) = self.env.lookup_constructor_field_names(*id) else {
            self.report_type_error(
                pattern_id,
                format!(
                    "`{qualified}` has positional fields; match it with `{}(...)`",
                    id.name(self.db())
                ),
            );
            return;
        };
        let is_variant = matches!(
            result.kind(self.db()),
            TypeKind::Named { name, .. } if self.env.lookup_enum_variants(name).is_some()
        );
        let kind = if is_variant { "variant" } else { "struct" };
        self.report_field_shape(
            pattern_id,
            format_args!("{kind} `{qualified}`"),
            declared,
            written.iter().cloned(),
            omits_rest,
        );
    }

    /// Infer the type of a variable reference.
    /// Report a module named where `expected` is required, once per node.
    fn report_module_reference(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        path: ModulePath<'db>,
        expected: &str,
    ) -> Type<'db> {
        if ctx.mark_module_value_reported(node) {
            Diagnostic::new(
                format!(
                    "expected {expected}, found module `{}`",
                    path.segments(self.db()).iter().format("::")
                ),
                self.get_span(node),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }
        ctx.error_type()
    }

    fn infer_var_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: Option<NodeId>,
        resolved: &ResolvedRef<'db>,
    ) -> Type<'db> {
        if let Some(ty) = node.and_then(|node| ctx.get_ability_op_callee_type(node)) {
            return ty;
        }
        match resolved {
            ResolvedRef::Local { id, name } => {
                // Try by LocalId first, then by name
                let by_id = if id.is_unresolved() {
                    None
                } else {
                    ctx.lookup_local(*id)
                };
                by_id
                    .or_else(|| ctx.lookup_local_by_name(name))
                    .unwrap_or_else(|| ctx.fresh_type_var())
            }
            ResolvedRef::Function { id } => node
                .and_then(|node| ctx.instantiate_function_reference(node, *id))
                .unwrap_or_else(|| ctx.fresh_type_var()),
            ResolvedRef::Constructor { id, .. } => match node {
                Some(node) => self.instantiate_value_constructor_with_ctx(ctx, node, resolved),
                None => ctx
                    .instantiate_constructor(*id)
                    .unwrap_or_else(|| ctx.fresh_type_var()),
            },
            ResolvedRef::Module { path } => match node {
                // Name resolution accepts a module path in value position (an
                // unresolved `use` leaves one behind), but no value has a
                // module's type.
                Some(node) => self.report_module_reference(ctx, node, *path, "a value"),
                None => ctx.error_type(),
            },
            ResolvedRef::TypeDef { .. } => {
                // Type definitions cannot be used as values in expression context.
                // This typically happens when an enum name like `Option` is used
                // directly without a variant like `Some` or `None`.
                ctx.error_type()
            }
            ResolvedRef::AbilityOp { ability, op, .. } => {
                // Look up the ability operation signature from the module type env
                if let Some(op_info) = self.env.lookup_ability_op(*ability, op) {
                    // Create a function type from the operation signature
                    // The effect row contains this ability (with a row variable tail for polymorphism)

                    // Generate fresh type vars for parameterized abilities
                    let ability_args: Vec<Type<'db>> =
                        if let Some(ability_info) = self.env.lookup_ability(*ability) {
                            ability_info
                                .type_params
                                .iter()
                                .map(|_| ctx.fresh_type_var())
                                .collect()
                        } else {
                            vec![]
                        };

                    if let Some(contract) = ctx.effect_contract {
                        let candidates: Vec<_> = contract
                            .effects(self.db())
                            .iter()
                            .filter(|effect| effect.ability_id == *ability)
                            .collect();
                        if let [selected] = candidates.as_slice() {
                            for (inferred, expected) in ability_args.iter().zip(&selected.args) {
                                ctx.constrain_eq(*inferred, *expected);
                            }
                        }
                    }
                    // Substitute ability type params into the operation signature.
                    // The operation's types may contain BoundVars that refer to the ability's
                    // type parameters.
                    let param_types: Vec<Type<'db>> = op_info
                        .param_types
                        .iter()
                        .map(|ty| {
                            subst::substitute_bound_vars(self.db(), *ty, &ability_args)
                                .unwrap_or_else(|index, max| {
                                    panic!(
                                        "AbilityOp param BoundVar index out of range: index={}, subst.len()={}",
                                        index, max
                                    )
                                })
                        })
                        .collect();
                    let return_type =
                        subst::substitute_bound_vars(self.db(), op_info.return_type, &ability_args)
                            .unwrap_or_else(|index, max| {
                                panic!(
                                    "AbilityOp return BoundVar index out of range: index={}, subst.len()={}",
                                    index, max
                                )
                            });

                    let ability_effect = Effect {
                        ability_id: *ability,
                        args: ability_args,
                    };
                    let effect = EffectRow::new(self.db(), vec![ability_effect], None);
                    ctx.func_type(param_types, return_type, effect)
                } else {
                    ctx.error_type()
                }
            }
            ResolvedRef::Ability { .. } => {
                // Ability definitions cannot be used as values in expression context.
                // They are only valid in handler patterns to identify which ability is being handled.
                ctx.error_type()
            }
        }
    }

    fn infer_local_reference_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        resolved: &ResolvedRef<'db>,
    ) -> Type<'db> {
        let ResolvedRef::Local { id, name } = resolved else {
            unreachable!("local-reference inference requires a local reference");
        };
        ctx.lookup_local_reference(node, *id, name)
            .unwrap_or_else(|| ctx.fresh_type_var())
    }

    /// Infer a call written `receiver.method(args)`: `method(receiver, args)`
    /// once the function it names is selected.
    ///
    /// The receiver alone is tried first, so that the other arguments are
    /// inferred against the selected function's parameters. A call it does
    /// not decide infers them on their own and tries again with all of them;
    /// one still undecided is deferred.
    fn infer_method_call_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: &Expr<ResolvedRef<'db>>,
    ) -> Type<'db> {
        let ExprKind::MethodCall {
            receiver,
            method,
            path,
            args,
        } = &*expr.kind
        else {
            unreachable!("the caller matched a method call");
        };
        let node = expr.id;
        let receiver_ty = self.infer_expr_type_with_ctx(ctx, receiver);
        let path = path.as_ref().map(method_path_functions);
        let arity = args.len() + 1;
        let selection = self.select_method(
            method,
            path.as_ref(),
            CallTypes {
                args: &[receiver_ty],
                arity,
                result: None,
            },
        );
        let field = self.method_field(receiver_ty, method, path.is_some(), args, &selection);
        if let Some(field) = &field
            && let Some(result_ty) = self.lookup_struct_field_type(receiver_ty, field)
        {
            self.record_field_instance(ctx, node, receiver_ty, field.clone(), result_ty);
            return result_ty;
        }
        if let MethodSelection::One(entry) = selection {
            let callee_ty = self.select_callee(ctx, node, entry.func_id);
            let mut arg_types = vec![receiver_ty];
            let params = match callee_ty.kind(self.db()) {
                TypeKind::Func { params, .. } => params.as_slice(),
                _ => &[],
            };
            for (index, arg) in args.iter().enumerate() {
                arg_types.push(match params.get(index + 1) {
                    Some(expected) => self.infer_expr_with_expected(ctx, arg, *expected),
                    None => self.infer_expr_type_with_ctx(ctx, arg),
                });
            }
            return self.infer_call_with_ctx(ctx, callee_ty, &arg_types, node);
        }

        let arg_types: Vec<Type<'db>> = std::iter::once(receiver_ty)
            .chain(args.iter().map(|a| self.infer_deferred_argument(ctx, a)))
            .collect();
        let selection = self.select_method(
            method,
            path.as_ref(),
            CallTypes {
                args: &arg_types,
                arity,
                result: None,
            },
        );
        if let MethodSelection::One(entry) = selection {
            let callee_ty = self.select_callee(ctx, node, entry.func_id);
            return self.infer_call_with_ctx(ctx, callee_ty, &arg_types, node);
        }

        // For known operator methods, add type constraints eagerly so that
        // type inference can propagate before the method is selected.
        let result_ty = match method.to_string().as_str() {
            "+" | "-" | "*" | "/" | "%" => {
                if let Some(rhs_ty) = arg_types.get(1) {
                    ctx.constrain_eq(receiver_ty, *rhs_ty);
                }
                receiver_ty
            }
            "==" | "!=" | "<" | "<=" | ">" | ">=" => {
                if let Some(rhs_ty) = arg_types.get(1) {
                    ctx.constrain_eq(receiver_ty, *rhs_ty);
                }
                ctx.bool_type()
            }
            _ => ctx.fresh_type_var(),
        };
        let effect = ctx
            .deferred_method_effect(node)
            .unwrap_or_else(|| ctx.fresh_effect_row());
        ctx.evidence.record_call(node, effect);
        ctx.merge_effect_at(effect, node);
        ctx.record_deferred_method(crate::typeck::func_context::DeferredMethodCall {
            node_id: node,
            receiver_ty,
            method: method.clone(),
            path,
            result_ty,
            arg_types,
            effect,
        });
        result_ty
    }

    /// The instance of `function` that the call at `node` selected.
    fn select_callee(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        function: crate::ast::FuncDefId<'db>,
    ) -> Type<'db> {
        let callee_ty = ctx
            .instantiate_function_reference(node, function)
            .unwrap_or_else(|| ctx.fresh_type_var());
        ctx.record_resolved_method(node, function, callee_ty);
        callee_ty
    }

    /// Infer the result type of a function call.
    ///
    /// This method:
    /// 1. Creates fresh type variables for parameter and result types
    /// 2. Creates a fresh effect row for the callee's effect
    /// 3. Constrains the callee to have the expected function type (or continuation type)
    /// 4. Constrains the callee's effect to be a subset of the current function's effect
    ///
    /// Effect propagation ensures that if we call `State::get()`, the `State`
    /// effect constraint is added to the current function's effect row.
    fn infer_call_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        callee_ty: Type<'db>,
        arg_types: &[Type<'db>],
        origin: NodeId,
    ) -> Type<'db> {
        // A function called with the wrong number of arguments is reported
        // as that, and the arguments it does have are checked.
        if let TypeKind::Func {
            params,
            result,
            effect,
            ..
        } = callee_ty.kind(self.db())
            && params.len() != arg_types.len()
        {
            if ctx.report_arity_once(origin) {
                self.report_call_arity(origin, params.len(), arg_types.len());
            }
            for (param_ty, arg_ty) in params.iter().zip(arg_types) {
                ctx.constrain_coerce_at(*arg_ty, *param_ty, origin, ConstraintOriginKind::Call);
            }
            ctx.evidence.record_call(origin, *effect);
            ctx.merge_effect_at(*effect, origin);
            return *result;
        }

        // Create expected type - could be either a function or continuation type.
        // We create fresh type variables and let unification determine the actual type.
        let param_types: Vec<Type<'db>> = arg_types.iter().map(|_| ctx.fresh_type_var()).collect();
        let result_ty = ctx.fresh_type_var();

        // Create both a function type and a continuation type expectation.
        // One will unify successfully depending on what the callee actually is.
        //
        // For functions: fn(params) -> result with effects
        // For continuations: Continuation { arg, result } (single-arg resume)
        if arg_types.len() == 1 {
            // Single argument: could be either a function call or continuation resume.
            // Create a continuation type for potential unification.
            let cont_arg_ty = ctx.fresh_type_var();
            let cont_result_ty = ctx.fresh_type_var();
            let cont_effect = ctx.fresh_effect_row();
            let expected_cont_ty = Type::new(
                self.db(),
                TypeKind::Continuation {
                    arg: cont_arg_ty,
                    result: cont_result_ty,
                    effect: cont_effect,
                },
            );

            // Also create function type for the normal case
            let callee_effect = ctx.fresh_effect_row();
            let expected_func_ty = ctx.func_type(param_types.clone(), result_ty, callee_effect);

            // Try to match the callee type: if it's already a concrete Continuation type,
            // use continuation semantics; otherwise use function semantics.
            // We check the callee type directly rather than relying on unification to choose.
            if matches!(callee_ty.kind(self.db()), TypeKind::Continuation { .. }) {
                // Callee is a continuation - constrain as continuation call.
                // Note: We intentionally do NOT merge cont_effect into the current effect
                // context here, unlike the function call path below. The continuation's
                // effect row is already the handler body's effect row (assigned in
                // convert_handler_arm_with_ctx), so it's already accounted for in the
                // enclosing handle expression's effect propagation.
                ctx.constrain_eq_at(
                    callee_ty,
                    expected_cont_ty,
                    origin,
                    ConstraintOriginKind::Call,
                );
                ctx.constrain_coerce_at(
                    arg_types[0],
                    cont_arg_ty,
                    origin,
                    ConstraintOriginKind::Call,
                );
                return cont_result_ty;
            } else {
                // Callee is a UniVar, concrete function type, or other - use function semantics
                // This handles cases like higher-order functions with unknown callees
                ctx.constrain_eq_at(
                    callee_ty,
                    expected_func_ty,
                    origin,
                    ConstraintOriginKind::Call,
                );
                for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                    ctx.constrain_coerce_at(*arg_ty, *param_ty, origin, ConstraintOriginKind::Call);
                }

                let effect = self.resolve_callee_effect(callee_ty, callee_effect);
                ctx.evidence.record_call(origin, effect);
                ctx.merge_effect_at(effect, origin);

                return result_ty;
            }
        }

        // Multiple arguments (or zero): must be a function call
        let callee_effect = ctx.fresh_effect_row();
        let expected_func_ty = ctx.func_type(param_types.clone(), result_ty, callee_effect);
        ctx.constrain_eq_at(
            callee_ty,
            expected_func_ty,
            origin,
            ConstraintOriginKind::Call,
        );

        // Constrain argument types
        for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
            ctx.constrain_coerce_at(*arg_ty, *param_ty, origin, ConstraintOriginKind::Call);
        }

        let effect = self.resolve_callee_effect(callee_ty, callee_effect);
        ctx.evidence.record_call(origin, effect);
        ctx.merge_effect_at(effect, origin);

        result_ty
    }

    /// Resolve the effect row to merge for a function call.
    ///
    /// If the callee type is already a concrete function type, uses that function's
    /// effect row directly. Otherwise, falls back to the fresh effect row.
    fn resolve_callee_effect(
        &self,
        callee_ty: Type<'db>,
        fallback: EffectRow<'db>,
    ) -> EffectRow<'db> {
        if let TypeKind::Func { effect, .. } = callee_ty.kind(self.db()) {
            *effect
        } else {
            fallback
        }
    }

    /// Extract struct declaration identity and type arguments from a type.
    ///
    /// Returns (Some(struct_id), type_args) if the type is a Named or App type,
    /// otherwise (None, empty vec).
    fn extract_struct_info(
        &self,
        ty: Type<'db>,
    ) -> (Option<crate::ast::TypeDefId<'db>>, Vec<Type<'db>>) {
        match ty.kind(self.db()) {
            TypeKind::Named { id, args, .. } => (Some(*id), args.clone()),
            TypeKind::App { ctor, args } => {
                if let TypeKind::Named { id, .. } = ctor.kind(self.db()) {
                    (Some(*id), args.clone())
                } else {
                    (None, vec![])
                }
            }
            _ => (None, vec![]),
        }
    }

    /// Extract the element type from a List type.
    ///
    /// If the type is `List<T>`, returns `T`. Otherwise, creates a fresh type variable.
    fn extract_list_element_type(
        &self,
        ty: Type<'db>,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
    ) -> Type<'db> {
        match ty.kind(self.db()) {
            TypeKind::Named { id, args, .. }
                if id.is_builtin_list(self.db()) && args.len() == 1 =>
            {
                args[0]
            }
            TypeKind::App { ctor, args } if args.len() == 1 => {
                if let TypeKind::Named { id, .. } = ctor.kind(self.db())
                    && id.is_builtin_list(self.db())
                {
                    return args[0];
                }
                ctx.fresh_type_var()
            }
            _ => ctx.fresh_type_var(),
        }
    }

    /// Report a call that passes `args` arguments to a function of `params`
    /// parameters.
    pub(crate) fn report_call_arity(&self, node: NodeId, params: usize, args: usize) {
        Diagnostic::new(
            format!(
                "call arity mismatch: expected {params} argument{}, found {args}",
                if params == 1 { "" } else { "s" }
            ),
            self.get_span(node),
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        )
        .accumulate(self.db());
    }

    /// The field a method call reads from its receiver, if it names one: an
    /// unqualified method is the field's name, and a path names the field's
    /// getter `T::f` in the receiver's struct `T` when the path selects that
    /// getter alone.
    fn method_field(
        &self,
        receiver_ty: Type<'db>,
        method: &Symbol,
        qualified: bool,
        args: &[Expr<ResolvedRef<'db>>],
        selection: &MethodSelection<'db>,
    ) -> Option<Symbol> {
        if !qualified {
            return args.is_empty().then(|| method.clone());
        }
        let MethodSelection::One(selected) = selection else {
            return None;
        };
        if !args.is_empty() {
            return None;
        }
        let owner = match receiver_ty.kind(self.db()) {
            TypeKind::Named { id, .. } => *id,
            TypeKind::App { ctor, .. } => match ctor.kind(self.db()) {
                TypeKind::Named { id, .. } => *id,
                _ => return None,
            },
            _ => return None,
        };
        let field = method.last_segment();
        let mut prefix = owner.qualified(self.db()).to_string();
        let getter = crate::qualified_symbol(&mut prefix, &field);
        (*selected.func_id.qualified(self.db()) == getter).then_some(field)
    }

    /// Select the function a call names among the functions its path may
    /// name, or for an unqualified method among the functions of that name
    /// its receiver's type has: the one whose signature takes the call.
    pub(crate) fn select_method(
        &self,
        method: &Symbol,
        path: Option<&(NodeId, Vec<crate::ast::FuncDefId<'db>>)>,
        call: CallTypes<'_, 'db>,
    ) -> MethodSelection<'db> {
        let candidates: Vec<crate::typeck::MethodEntry<'db>> = match path {
            Some((_, candidates)) => candidates
                .iter()
                .filter_map(|candidate| {
                    let (scheme, _) = self.env.function_scheme(*candidate)?;
                    Some(crate::typeck::MethodEntry {
                        func_id: *candidate,
                        func_ty: scheme.body(self.db()),
                    })
                })
                .collect(),
            // A call without further arguments may read a field of its
            // receiver, so the receiver's type has to be known.
            None if call.arity == 1
                && call.args.first().is_none_or(|receiver| {
                    matches!(receiver.kind(self.db()), TypeKind::UniVar { .. })
                }) =>
            {
                Vec::new()
            }
            None => self.env.methods_named(method).to_vec(),
        };
        // One function leaves nothing to select: the call is checked against
        // it, and what does not fit is an ordinary type error.
        if let [entry] = candidates[..] {
            return MethodSelection::One(entry);
        }
        let signatures: Vec<_> = candidates
            .iter()
            .filter_map(|entry| match entry.func_ty.kind(self.db()) {
                TypeKind::Func { params, result, .. } => Some((entry, &params[..], *result)),
                _ => None,
            })
            .collect();
        let select = |matches: &dyn Fn(&[Type<'db>], Type<'db>) -> bool| {
            let mut matching = signatures
                .iter()
                .filter(|(_, params, result)| matches(params, *result))
                .map(|(entry, ..)| *entry);
            match (matching.next(), matching.next()) {
                (Some(entry), None) => MethodSelection::One(*entry),
                (Some(first), Some(second)) => MethodSelection::Ambiguous(
                    [first, second]
                        .into_iter()
                        .chain(matching)
                        .map(|entry| entry.func_id)
                        .collect(),
                ),
                (None, _) => MethodSelection::None,
            }
        };
        match select(&|params, result| call.matches(self.db(), params, result)) {
            // A call that passes the wrong number of arguments still names
            // the function its arguments select, which reports the number.
            MethodSelection::None => {
                match select(&|params, _| call.arguments_match(self.db(), params)) {
                    selection @ MethodSelection::One(_) => selection,
                    _ => MethodSelection::None,
                }
            }
            selection => selection,
        }
    }

    /// Look up a struct field type from the receiver type.
    ///
    /// Given a receiver type like `Point` or `Point(Int)`, look up the field `x`
    /// and return its type with BoundVars substituted by the actual type arguments.
    fn record_field_instance(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        receiver: Type<'db>,
        field: Symbol,
        result: Type<'db>,
    ) {
        let Some(instance) = self.field_getter_instance(receiver, field, result) else {
            return;
        };
        ctx.record_resolved_method(node, instance.function, instance.callable);
        ctx.record_field_instance(node, instance);
    }

    /// The instance of the getter that reads `field` of a `receiver` whose
    /// field has type `result`.
    pub(crate) fn field_getter_instance(
        &self,
        receiver: Type<'db>,
        field: Symbol,
        result: Type<'db>,
    ) -> Option<crate::typeck::FunctionInstance<'db>> {
        let TypeKind::Named {
            id: owner, args, ..
        } = receiver.kind(self.db())
        else {
            return None;
        };
        let (parameters, field_ty) = self.env.lookup_struct_field(*owner, &field)?;
        let mut prefix = owner.qualified(self.db()).to_string();
        let function =
            crate::ast::FuncDefId::new(self.db(), crate::qualified_symbol(&mut prefix, &field));
        let receiver_template = Type::new(
            self.db(),
            TypeKind::Named {
                id: *owner,
                name: owner.qualified(self.db()).clone(),
                args: (0..parameters.len())
                    .map(|index| {
                        Type::new(
                            self.db(),
                            TypeKind::BoundVar {
                                index: index as u32,
                            },
                        )
                    })
                    .collect(),
            },
        );
        let template = self.env.func_type(
            vec![receiver_template],
            field_ty,
            EffectRow::pure(self.db()),
        );
        let scheme = TypeScheme::new(
            self.db(),
            parameters.to_vec(),
            collect_effect_vars(self.db(), template),
            template,
        );
        let callable = self
            .env
            .func_type(vec![receiver], result, EffectRow::pure(self.db()));
        Some(crate::typeck::FunctionInstance {
            origin: crate::typeck::FunctionInstanceOrigin::FieldAccessor {
                owner: *owner,
                field,
                kind: crate::typeck::FieldFunctionKind::Get,
            },
            function,
            scheme,
            callable,
            type_arguments: args.clone(),
            row_arguments: scheme
                .effect_params(self.db())
                .iter()
                .map(|var| EffectRow::open(self.db(), *var))
                .collect(),
        })
    }

    pub(crate) fn lookup_struct_field_type(
        &self,
        receiver_ty: Type<'db>,
        field_name: &Symbol,
    ) -> Option<Type<'db>> {
        // Extract struct declaration identity from receiver type.
        let struct_id = match receiver_ty.kind(self.db()) {
            TypeKind::Named { id, .. } => *id,
            TypeKind::App { ctor, .. } => {
                // Recursively extract from constructor
                match ctor.kind(self.db()) {
                    TypeKind::Named { id, .. } => *id,
                    _ => return None,
                }
            }
            TypeKind::UniVar { .. } => {
                // Type not yet known - can't resolve field
                return None;
            }
            _ => return None,
        };

        // Look up field in ModuleTypeEnv
        let (type_params, field_ty) = self.env.lookup_struct_field(struct_id, field_name)?;

        // Substitute BoundVars with actual type arguments if any
        let actual_args: Vec<Type<'db>> = match receiver_ty.kind(self.db()) {
            TypeKind::Named { args, .. } => args.clone(),
            TypeKind::App { args, .. } => args.clone(),
            _ => vec![],
        };

        // Substitute BoundVars in field_ty with actual_args
        if type_params.is_empty() {
            Some(field_ty)
        } else {
            Some(self.substitute_bound_vars(field_ty, &actual_args))
        }
    }

    /// Substitute BoundVars in a type with actual types.
    ///
    /// Panics if a BoundVar index is out of bounds.
    fn substitute_bound_vars(&self, ty: Type<'db>, args: &[Type<'db>]) -> Type<'db> {
        subst::substitute_bound_vars(self.db(), ty, args).unwrap_or_else(|index, max| {
            panic!(
                "BoundVar index out of range: index={}, subst.len()={}",
                index, max
            )
        })
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
            ExprKind::Call { callee, args } => {
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
                }
            }
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor: self.convert_ref_with_ctx(ctx, Some(expr_id), ctor),
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
                    resolved: type_name.clone(),
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
            ExprKind::MethodCall {
                receiver,
                method,
                path,
                args,
            } => {
                let converted_receiver = self.check_expr_with_ctx(ctx, receiver, Mode::Infer);

                if let Some((func_id, callee_ty)) = ctx.get_resolved_method(expr_id) {
                    // Resolved during inference — transform to Call
                    let callee_ref = TypedRef {
                        resolved: ResolvedRef::Function { id: func_id },
                        ty: callee_ty,
                    };
                    let callee = Expr::new(expr_id, ExprKind::Var(callee_ref));

                    let mut all_args = vec![converted_receiver];
                    all_args.extend(
                        args.iter()
                            .map(|a| self.check_expr_with_ctx(ctx, a, Mode::Infer)),
                    );
                    ExprKind::Call {
                        callee,
                        args: all_args,
                    }
                } else {
                    // Unresolved — keep as MethodCall
                    ExprKind::MethodCall {
                        receiver: converted_receiver,
                        method: method.clone(),
                        // Still unselected, so the candidates have no type.
                        path: path.as_ref().map(|path| crate::ast::MethodPath {
                            id: path.id,
                            candidates: path
                                .candidates
                                .iter()
                                .map(|candidate| TypedRef {
                                    resolved: candidate.clone(),
                                    ty: ctx.error_type(),
                                })
                                .collect(),
                        }),
                        args: args
                            .iter()
                            .map(|a| self.check_expr_with_ctx(ctx, a, Mode::Infer))
                            .collect(),
                    }
                }
            }
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

                        let pattern_ty = self.infer_pattern_type_with_ctx(ctx, &arm.pattern);
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
            ExprKind::Lambda { params, body } => {
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
                ctx.set_current_effect(EffectRow::pure(self.db()));
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
                ctx.set_current_effect(EffectRow::pure(self.db()));
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
                if let ExprKind::MethodCall { .. } = &*call.kind {
                    match &*converted.kind {
                        ExprKind::Call { callee, .. } => {
                            if let ExprKind::Var(TypedRef {
                                resolved: ResolvedRef::Function { id },
                                ..
                            }) = &*callee.kind
                            {
                                self.check_become_callee(expr_id, *id);
                            }
                        }
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
            (Some(_), ResolvedRef::Constructor { .. }) => {
                self.infer_var_with_ctx(ctx, node_id, resolved)
            }
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
                let outer_effect = ctx.current_effect();
                ctx.set_current_effect(EffectRow::pure(self.db()));
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
                let pattern_ty = self.infer_pattern_type_with_ctx(ctx, pattern);
                ctx.constrain_eq(pattern_ty, value_ty);
                self.generalize_and_bind_pattern_with_ctx(
                    ctx,
                    pattern,
                    value_ty,
                    evaluation_effect,
                );
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
                let outer_effect = ctx.current_effect();
                ctx.set_current_effect(EffectRow::pure(self.db()));
                let value = if let Some(ann) = ty {
                    let expected = self.annotation_to_type_with_ctx(ctx, ann);
                    self.check_expr_with_ctx(ctx, value, Mode::Check(expected))
                } else {
                    self.check_expr_with_ctx(ctx, value, Mode::Infer)
                };
                let value_ty = if let Some(ann) = ty {
                    self.annotation_to_type_with_ctx(ctx, ann)
                } else {
                    ctx.get_node_type(value.id)
                        .unwrap_or_else(|| ctx.fresh_type_var())
                };
                let evaluation_effect = ctx.current_effect();
                ctx.set_current_effect(outer_effect);
                ctx.merge_effect(evaluation_effect);

                // Constrain pattern type to match value type
                let pattern_ty = self.infer_pattern_type_with_ctx(ctx, pattern);
                ctx.constrain_eq(pattern_ty, value_ty);

                self.generalize_and_bind_pattern_with_ctx(
                    ctx,
                    pattern,
                    value_ty,
                    evaluation_effect,
                );
                let pattern = self.convert_pattern_with_ctx(ctx, pattern);

                Stmt::Let {
                    id: *id,
                    pattern,
                    value,
                    ty: ty.clone(),
                }
            }
            Stmt::Expr { id, expr } => {
                let expr = self.check_expr_with_ctx(ctx, expr, Mode::Infer);
                Stmt::Expr { id: *id, expr }
            }
        }
    }

    // =========================================================================
    // Pattern handling
    // =========================================================================

    /// Resolve the deferred method calls whose receiver `solver` has typed,
    /// as inference does for a receiver typed where the call is written.
    fn resolve_typed_deferred_methods(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        solver: &TypeSolver<'db>,
    ) -> bool {
        let mut resolved = false;
        for call in ctx.deferred_methods().to_vec() {
            let args: Vec<_> = call
                .arg_types
                .iter()
                .map(|ty| solver.type_subst().apply(self.db(), *ty))
                .collect();
            let types = CallTypes {
                args: &args,
                arity: args.len(),
                result: Some(solver.expected_type(call.result_ty)),
            };
            let MethodSelection::One(entry) =
                self.select_method(&call.method, call.path.as_ref(), types)
            else {
                continue;
            };
            let Some(callee_ty) = ctx.instantiate_function_reference(call.node_id, entry.func_id)
            else {
                continue;
            };
            ctx.resolve_deferred_method(call.node_id, entry.func_id, callee_ty);
            let slack = ctx.fresh_row_var();
            ctx.constrain_all(self.selected_call_constraints(solver, &call, callee_ty, slack));
            resolved = true;
        }
        resolved
    }

    fn generalize_and_bind_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        value_ty: Type<'db>,
        evaluation_effect: EffectRow<'db>,
    ) {
        // A revisited `let` keeps the schemes of its first visit. Solving
        // again can fail once an unrelated error has been constrained, and a
        // monomorphic rebinding would then alias the quantifiers that
        // already-checked uses instantiated.
        if let Some(bindings) = ctx.let_schemes(pattern.id) {
            for LetSchemeBinding {
                name,
                local_id,
                scope,
                scheme,
            } in bindings
            {
                if let Some(local_id) = local_id {
                    ctx.bind_local_scheme(local_id, scheme);
                    ctx.record_local_binding_owner(local_id, scope);
                }
                ctx.bind_local_scheme_by_name(name, scheme);
            }
            return;
        }
        // A call whose receiver the statements so far have typed is resolved
        // here, so the binding generalizes over what the call performs.
        let mut solver = loop {
            let mut solver = TypeSolver::new(self.db());
            solver.reserve_row_vars(ctx.next_row_var());
            for method in ctx.deferred_methods() {
                solver.defer_producer(
                    method.node_id,
                    method.result_ty,
                    method.arg_types.clone(),
                    method.effect,
                );
            }
            let solved = solver
                .solve(ctx.constraints_snapshot())
                .and_then(|()| solver.finalize_relations().map_err(|failure| failure.error));
            ctx.reserve_row_vars(solver.next_row_var());
            if solved.is_err() {
                self.bind_pattern_vars_with_ctx(ctx, pattern, value_ty);
                return;
            }
            if !self.resolve_typed_deferred_methods(ctx, &solver) {
                break solver;
            }
        };
        // The row of a call still to be resolved keeps its own name in the
        // scheme, so resolving the call reaches the scheme's uses.
        solver.make_row_representatives(
            ctx.deferred_methods()
                .iter()
                .filter_map(|method| method.effect.rest(self.db())),
        );

        let type_subst = solver.type_subst().clone();
        let row_subst = solver.row_subst().clone();
        let should_generalize = row_subst
            .apply(self.db(), evaluation_effect)
            .is_pure(self.db());

        let (mut environment_type_vars, mut environment_effect_vars) = solver.pending_variables();
        if should_generalize {
            for ty in ctx.annotation_type_parameters() {
                type_subst.collect_univars_from_type(
                    self.db(),
                    ty,
                    &row_subst,
                    &mut environment_type_vars,
                );
            }
            for scheme in ctx.visible_local_schemes() {
                let body =
                    type_subst.apply_with_rows(self.db(), scheme.body(self.db()), &row_subst);
                type_subst.collect_univars_from_type(
                    self.db(),
                    body,
                    &row_subst,
                    &mut environment_type_vars,
                );
                for var in collect_effect_vars(self.db(), body) {
                    if !scheme.effect_params(self.db()).contains(&var)
                        && !environment_effect_vars.contains(&var)
                    {
                        environment_effect_vars.push(var);
                    }
                }
            }
        }

        solver.expand_union_dependencies(&mut environment_type_vars, &mut environment_effect_vars);
        let resolved_value = type_subst.apply_with_rows(self.db(), value_ty, &row_subst);
        let mut bindings = Vec::new();
        self.collect_pattern_bindings_with_ctx(
            ctx,
            pattern,
            resolved_value,
            &type_subst,
            &row_subst,
            &mut bindings,
        );

        // Every name of this `let` shares one set of local quantifiers owned
        // by the root pattern: the right-hand side is checked once, so a
        // variable shared by several names (`let f as g = ...`) must have a
        // single owner.
        let mut let_quantifiers = HashMap::default();
        let mut let_schemes = Vec::new();
        for PatternBinding {
            name,
            local_id,
            scope,
            ty,
        } in bindings
        {
            let scheme = if should_generalize {
                let (_, mut type_params, mut mapping) = type_subst
                    .generalize_excluding_with_mapping(
                        self.db(),
                        ty,
                        &row_subst,
                        &environment_type_vars,
                    );
                let retained = solver.row_unions_for_type(ty);
                let removals = solver.row_removals_for_type(ty);
                let (mut union_types, mut union_rows) = solver.row_union_variables(&retained);
                let (removed_types, removed_rows) = solver.row_removal_variables(&removals);
                union_types.extend(removed_types);
                union_rows.extend(removed_rows);
                for var in union_types {
                    if !environment_type_vars.contains(&var) && !mapping.contains_key(&var) {
                        mapping.insert(var, type_params.len() as u32);
                        type_params.push(crate::ast::TypeParam::anonymous());
                    }
                }
                let generalized =
                    type_subst.apply_generalization(self.db(), ty, &row_subst, &mapping);
                let unions = retained
                    .iter()
                    .map(|union| solver.generalize_row_union(union, &mapping))
                    .collect();
                let mut by_index: Vec<_> =
                    mapping.iter().map(|(var, index)| (*index, *var)).collect();
                by_index.sort_unstable_by_key(|(index, _)| *index);
                for (_, var) in by_index {
                    let next = let_quantifiers.len() as u32;
                    let_quantifiers.entry(var).or_insert(next);
                }
                let mut effect_params = Vec::new();
                for var in collect_effect_vars(self.db(), generalized)
                    .into_iter()
                    .chain(union_rows)
                {
                    if !environment_effect_vars.contains(&var) && !effect_params.contains(&var) {
                        effect_params.push(var);
                    }
                }
                TypeScheme::builder(type_params, effect_params, generalized)
                    .row_unions(unions)
                    .row_removals(
                        removals
                            .iter()
                            .map(|r| solver.generalize_row_removal(r, &mapping))
                            .collect(),
                    )
                    .build(self.db())
            } else {
                TypeScheme::mono(self.db(), ty)
            };
            if let Some(local_id) = local_id {
                ctx.bind_local_scheme(local_id, scheme);
                ctx.record_local_binding_owner(local_id, scope);
            }
            ctx.bind_local_scheme_by_name(name.clone(), scheme);
            let_schemes.push(LetSchemeBinding {
                name,
                local_id,
                scope,
                scheme,
            });
        }
        ctx.record_let_schemes(pattern.id, let_schemes);
        if !let_quantifiers.is_empty() {
            ctx.record_local_generalization(pattern.id, let_quantifiers);
        }
    }

    fn collect_pattern_bindings_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        ty: Type<'db>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        bindings: &mut Vec<PatternBinding<'db>>,
    ) {
        let resolve = |ty| type_subst.apply_with_rows(self.db(), ty, row_subst);
        match &*pattern.kind {
            PatternKind::Bind { name, local_id } => {
                bindings.push(PatternBinding {
                    name: name.clone(),
                    local_id: *local_id,
                    scope: pattern.id,
                    ty: resolve(ty),
                });
            }
            PatternKind::Tuple(patterns)
            | PatternKind::Variant {
                fields: patterns, ..
            }
            | PatternKind::List(patterns) => {
                for pattern in patterns {
                    let pattern_ty = ctx
                        .get_node_type(pattern.id)
                        .map(resolve)
                        .unwrap_or_else(|| resolve(ty));
                    self.collect_pattern_bindings_with_ctx(
                        ctx, pattern, pattern_ty, type_subst, row_subst, bindings,
                    );
                }
            }
            PatternKind::Record {
                type_name, fields, ..
            } => {
                let field_tys = self.record_pattern_field_types(ctx, pattern.id, type_name, fields);
                for (field, field_ty) in fields.iter().zip(field_tys) {
                    let field_ty = field_ty
                        .map(resolve)
                        .unwrap_or_else(|| ctx.fresh_type_var());
                    if let Some(pattern) = &field.pattern {
                        self.collect_pattern_bindings_with_ctx(
                            ctx, pattern, field_ty, type_subst, row_subst, bindings,
                        );
                    } else {
                        bindings.push(PatternBinding {
                            name: field.name.clone(),
                            local_id: None,
                            scope: field.id,
                            ty: field_ty,
                        });
                    }
                }
            }
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => {
                for pattern in head {
                    let pattern_ty = ctx
                        .get_node_type(pattern.id)
                        .map(resolve)
                        .unwrap_or_else(|| resolve(ty));
                    self.collect_pattern_bindings_with_ctx(
                        ctx, pattern, pattern_ty, type_subst, row_subst, bindings,
                    );
                }
                if let Some(name) = rest {
                    bindings.push(PatternBinding {
                        name: name.clone(),
                        local_id: *rest_local_id,
                        scope: pattern.id,
                        ty: resolve(ty),
                    });
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => {
                let resolved_ty = resolve(ty);
                bindings.push(PatternBinding {
                    name: name.clone(),
                    local_id: *local_id,
                    scope: pattern.id,
                    ty: resolved_ty,
                });
                self.collect_pattern_bindings_with_ctx(
                    ctx,
                    pattern,
                    resolved_ty,
                    type_subst,
                    row_subst,
                    bindings,
                );
            }
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {}
        }
    }

    /// Infer the type that a pattern matches against.
    fn infer_pattern_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
    ) -> Type<'db> {
        let ty = match &*pattern.kind {
            PatternKind::Wildcard | PatternKind::Bind { .. } => ctx.fresh_type_var(),
            PatternKind::Literal(lit) => match lit {
                LiteralPattern::Bool(_) => ctx.bool_type(),
                LiteralPattern::Nat(_) => ctx.nat_type(),
                LiteralPattern::Int(_) => ctx.int_type(),
                LiteralPattern::Float(_) => ctx.float_type(),
                LiteralPattern::String(_) => ctx.string_type(),
                LiteralPattern::Bytes(_) => ctx.bytes_type(),
                LiteralPattern::Rune(_) => ctx.rune_type(),
                LiteralPattern::Nil => ctx.nil_type(),
            },
            PatternKind::Variant { ctor, fields } => {
                let ctor_ty = match ctor {
                    ResolvedRef::Module { path } => {
                        self.report_module_reference(ctx, pattern.id, *path, "a constructor")
                    }
                    _ => self.infer_var_with_ctx(ctx, None, ctor),
                };
                ctx.record_node_type(pattern.id, ctor_ty);
                self.validate_variant_pattern_with_ctx(
                    ctx,
                    pattern.id,
                    ctor,
                    ctor_ty,
                    fields.len(),
                );

                match ctor_ty.kind(self.db()) {
                    TypeKind::Func { params, result, .. } => {
                        for (field_pat, param_ty) in fields.iter().zip(params.iter()) {
                            let field_ty = self.infer_pattern_type_with_ctx(ctx, field_pat);
                            ctx.constrain_eq(field_ty, *param_ty);
                        }
                        *result
                    }
                    _ => ctor_ty,
                }
            }
            PatternKind::Tuple(pats) => {
                let elem_tys: Vec<_> = pats
                    .iter()
                    .map(|p| self.infer_pattern_type_with_ctx(ctx, p))
                    .collect();
                ctx.tuple_type(elem_tys)
            }
            PatternKind::List(pats) => {
                let elem_ty = ctx.fresh_type_var();
                for pat in pats {
                    let pat_ty = self.infer_pattern_type_with_ctx(ctx, pat);
                    ctx.constrain_eq(pat_ty, elem_ty);
                }
                ctx.canonical_list_type(elem_ty)
            }
            PatternKind::ListRest { head, .. } => {
                let elem_ty = ctx.fresh_type_var();
                for pat in head {
                    let pat_ty = self.infer_pattern_type_with_ctx(ctx, pat);
                    ctx.constrain_eq(pat_ty, elem_ty);
                }
                ctx.canonical_list_type(elem_ty)
            }
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let written: Vec<Symbol> = fields.iter().map(|field| field.name.clone()).collect();
                let (_, result, field_tys) =
                    self.constructor_field_shape(ctx, pattern.id, type_name, &written);
                self.validate_record_pattern_with_ctx(
                    ctx, pattern.id, type_name, result, &written, *rest,
                );
                for (field, field_ty) in fields.iter().zip(field_tys) {
                    if let Some(sub) = &field.pattern {
                        let sub_ty = self.infer_pattern_type_with_ctx(ctx, sub);
                        if let Some(field_ty) = field_ty {
                            ctx.constrain_eq(sub_ty, field_ty);
                        }
                    }
                }
                result
            }
            PatternKind::As { pattern, .. } => self.infer_pattern_type_with_ctx(ctx, pattern),
            PatternKind::Error => ctx.error_type(),
        };

        if !matches!(&*pattern.kind, PatternKind::Variant { .. }) {
            ctx.record_node_type(pattern.id, ty);
        }
        ty
    }

    /// Bind pattern variables to the given type.
    fn bind_pattern_vars_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        ty: Type<'db>,
    ) {
        match &*pattern.kind {
            PatternKind::Bind { name, local_id } => {
                if let Some(id) = local_id {
                    ctx.bind_local(*id, ty);
                }
                ctx.bind_local_by_name(name.clone(), ty);
            }
            PatternKind::Tuple(pats) => {
                if let TypeKind::Tuple(elem_tys) = ty.kind(self.db()) {
                    for (pat, elem_ty) in pats.iter().zip(elem_tys.iter()) {
                        self.bind_pattern_vars_with_ctx(ctx, pat, *elem_ty);
                    }
                } else {
                    let fresh_vars: Vec<_> = pats.iter().map(|_| ctx.fresh_type_var()).collect();
                    for (pat, fresh_ty) in pats.iter().zip(fresh_vars) {
                        self.bind_pattern_vars_with_ctx(ctx, pat, fresh_ty);
                    }
                }
            }
            PatternKind::Variant { fields, .. } => {
                // Inference recorded the constructor's callable type on this
                // pattern and constrained each field to its parameter. A
                // field's own node type is not its value type when the field
                // is itself a constructor pattern.
                let params = match ctx.get_node_type(pattern.id).map(|ty| ty.kind(self.db())) {
                    Some(TypeKind::Func { params, .. }) if params.len() == fields.len() => {
                        Some(params.clone())
                    }
                    _ => None,
                };
                for (index, field) in fields.iter().enumerate() {
                    let field_ty = params
                        .as_ref()
                        .map(|params| params[index])
                        .or_else(|| ctx.get_node_type(field.id))
                        .unwrap_or_else(|| ctx.fresh_type_var());
                    self.bind_pattern_vars_with_ctx(ctx, field, field_ty);
                }
            }
            PatternKind::Record {
                type_name, fields, ..
            } => {
                let field_tys = self.record_pattern_field_types(ctx, pattern.id, type_name, fields);
                for (field, field_ty) in fields.iter().zip(field_tys) {
                    let field_ty = field_ty.unwrap_or_else(|| ctx.fresh_type_var());

                    if let Some(pat) = &field.pattern {
                        self.bind_pattern_vars_with_ctx(ctx, pat, field_ty);
                    } else {
                        // Shorthand { name } - bind the field name directly
                        ctx.bind_local_by_name(field.name.clone(), field_ty);
                    }
                }
            }
            PatternKind::List(pats) => {
                // Extract element type from the list type, or create a fresh var
                let elem_ty = self.extract_list_element_type(ty, ctx);
                for pat in pats {
                    self.bind_pattern_vars_with_ctx(ctx, pat, elem_ty);
                }
            }
            PatternKind::ListRest {
                head,
                rest_local_id,
                ..
            } => {
                // Extract element type from the list type, or create a fresh var
                let elem_ty = self.extract_list_element_type(ty, ctx);
                for pat in head {
                    self.bind_pattern_vars_with_ctx(ctx, pat, elem_ty);
                }
                if let Some(local_id) = rest_local_id {
                    // The rest is also a list of the same element type
                    let list_ty = ctx.canonical_list_type(elem_ty);
                    ctx.bind_local(*local_id, list_ty);
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => {
                ctx.bind_local_by_name(name.clone(), ty);
                if let Some(local_id) = local_id {
                    ctx.bind_local(*local_id, ty);
                }
                self.bind_pattern_vars_with_ctx(ctx, pattern, ty);
            }
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {}
        }
    }

    /// Convert a pattern.
    fn convert_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
    ) -> Pattern<TypedRef<'db>> {
        let kind = match &*pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind {
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Literal(lit) => PatternKind::Literal(lit.clone()),
            PatternKind::Variant { ctor, fields } => PatternKind::Variant {
                ctor: TypedRef {
                    ty: ctx
                        .get_node_type(pattern.id)
                        .unwrap_or_else(|| self.infer_var_with_ctx(ctx, None, ctor)),
                    resolved: ctor.clone(),
                },
                fields: fields
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            },
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let type_name = self.record_pattern_type_name(ctx, pattern.id, type_name);
                PatternKind::Record {
                    type_name,
                    fields: fields
                        .iter()
                        .map(|f| self.convert_field_pattern_with_ctx(ctx, f))
                        .collect(),
                    rest: *rest,
                }
            }
            PatternKind::Tuple(patterns) => PatternKind::Tuple(
                patterns
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            ),
            PatternKind::List(patterns) => PatternKind::List(
                patterns
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            ),
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => PatternKind::ListRest {
                head: head
                    .iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
                rest: rest.clone(),
                rest_local_id: *rest_local_id,
            },
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.convert_pattern_with_ctx(ctx, pattern),
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Error => PatternKind::Error,
        };
        Pattern::new(pattern.id, kind)
    }

    /// The field type of each field of a record pattern, in source order,
    /// taken from its constructor instance.
    fn record_pattern_field_types<V>(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        type_name: &ResolvedRef<'db>,
        fields: &[FieldPattern<V>],
    ) -> Vec<Option<Type<'db>>> {
        let written: Vec<Symbol> = fields.iter().map(|field| field.name.clone()).collect();
        self.constructor_field_shape(ctx, pattern_id, type_name, &written)
            .2
    }

    /// The typed constructor reference of a brace-form constructor pattern,
    /// carrying the same constructor instance its fields were checked with.
    fn record_pattern_type_name(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern_id: NodeId,
        resolved: &ResolvedRef<'db>,
    ) -> TypedRef<'db> {
        let ty = self.instantiate_value_constructor_with_ctx(ctx, pattern_id, resolved);
        TypedRef {
            resolved: resolved.clone(),
            ty,
        }
    }

    fn convert_field_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        fp: &FieldPattern<ResolvedRef<'db>>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            id: fp.id,
            name_id: fp.name_id,
            name: fp.name.clone(),
            pattern: fp
                .pattern
                .as_ref()
                .map(|p| self.convert_pattern_with_ctx(ctx, p)),
        }
    }

    /// Convert a field pattern with an expected type.
    fn convert_field_pattern_with_expected_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        fp: &FieldPattern<ResolvedRef<'db>>,
        expected: Type<'db>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            id: fp.id,
            name_id: fp.name_id,
            name: fp.name.clone(),
            pattern: fp
                .pattern
                .as_ref()
                .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, expected)),
        }
    }

    /// Convert a case arm with an expected scrutinee type.
    fn convert_arm_with_scrutinee_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: &Arm<ResolvedRef<'db>>,
        scrutinee_ty: Type<'db>,
    ) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: self.convert_pattern_with_expected_ctx(ctx, &arm.pattern, scrutinee_ty),
            guard: arm.guard.as_ref().map(|g| {
                let bool_ty = ctx.bool_type();
                self.check_expr_with_ctx(ctx, g, Mode::Check(bool_ty))
            }),
            body: self.check_expr_with_ctx(ctx, &arm.body, Mode::Infer),
        }
    }

    /// Convert a pattern with an expected type.
    fn convert_pattern_with_expected_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: &Pattern<ResolvedRef<'db>>,
        expected: Type<'db>,
    ) -> Pattern<TypedRef<'db>> {
        let kind = match &*pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind {
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Literal(lit) => PatternKind::Literal(lit.clone()),
            PatternKind::Variant { ctor, fields } => {
                let ctor_ty = ctx
                    .get_node_type(pattern.id)
                    .unwrap_or_else(|| self.infer_var_with_ctx(ctx, None, ctor));

                match ctor_ty.kind(self.db()) {
                    TypeKind::Func { params, result, .. } => {
                        ctx.constrain_eq(*result, expected);
                        let fields = fields
                            .iter()
                            .zip(params.iter())
                            .map(|(p, param_ty)| {
                                self.convert_pattern_with_expected_ctx(ctx, p, *param_ty)
                            })
                            .collect();
                        PatternKind::Variant {
                            ctor: TypedRef {
                                resolved: ctor.clone(),
                                ty: ctor_ty,
                            },
                            fields,
                        }
                    }
                    _ => {
                        ctx.constrain_eq(ctor_ty, expected);
                        PatternKind::Variant {
                            ctor: TypedRef {
                                resolved: ctor.clone(),
                                ty: ctor_ty,
                            },
                            fields: vec![],
                        }
                    }
                }
            }
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let field_tys = self.record_pattern_field_types(ctx, pattern.id, type_name, fields);
                let type_name = self.record_pattern_type_name(ctx, pattern.id, type_name);
                let result = match type_name.ty.kind(self.db()) {
                    TypeKind::Func { result, .. } => *result,
                    _ => type_name.ty,
                };
                ctx.constrain_eq(result, expected);

                let converted_fields = fields
                    .iter()
                    .zip(field_tys)
                    .map(|(f, field_ty)| {
                        let field_expected = field_ty.unwrap_or_else(|| ctx.fresh_type_var());
                        self.convert_field_pattern_with_expected_ctx(ctx, f, field_expected)
                    })
                    .collect();

                PatternKind::Record {
                    type_name,
                    fields: converted_fields,
                    rest: *rest,
                }
            }
            PatternKind::Tuple(patterns) => {
                let elem_expectations: Vec<Type<'db>> =
                    if let TypeKind::Tuple(elems) = expected.kind(self.db()) {
                        elems.clone()
                    } else {
                        patterns.iter().map(|_| ctx.fresh_type_var()).collect()
                    };
                PatternKind::Tuple(
                    patterns
                        .iter()
                        .zip(elem_expectations)
                        .map(|(p, exp)| self.convert_pattern_with_expected_ctx(ctx, p, exp))
                        .collect(),
                )
            }
            PatternKind::List(patterns) => {
                let elem_ty = ctx.fresh_type_var();
                let list_ty = ctx.canonical_list_type(elem_ty);
                ctx.constrain_eq(expected, list_ty);
                PatternKind::List(
                    patterns
                        .iter()
                        .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, elem_ty))
                        .collect(),
                )
            }
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => {
                let elem_ty = ctx.fresh_type_var();
                let list_ty = ctx.canonical_list_type(elem_ty);
                ctx.constrain_eq(expected, list_ty);
                PatternKind::ListRest {
                    head: head
                        .iter()
                        .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, elem_ty))
                        .collect(),
                    rest: rest.clone(),
                    rest_local_id: *rest_local_id,
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.convert_pattern_with_expected_ctx(ctx, pattern, expected),
                name: name.clone(),
                local_id: *local_id,
            },
            PatternKind::Error => PatternKind::Error,
        };
        Pattern::new(pattern.id, kind)
    }

    /// Convert a handler arm.
    fn convert_handler_arm_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: &HandlerArm<ResolvedRef<'db>>,
        handle_ctx: &super::super::func_context::HandleContext<'db>,
    ) -> HandlerArm<TypedRef<'db>> {
        ctx.with_scope(|ctx| self.convert_handler_arm_in_scope(ctx, arm, handle_ctx))
    }

    /// Convert a handler arm within its arm-local scope.
    fn convert_handler_arm_in_scope(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: &HandlerArm<ResolvedRef<'db>>,
        handle_ctx: &super::super::func_context::HandleContext<'db>,
    ) -> HandlerArm<TypedRef<'db>> {
        let (kind, body_mode) = match &arm.kind {
            HandlerKind::Do { binding } => {
                let pattern_ty = self.infer_pattern_type_with_ctx(ctx, binding);
                ctx.constrain_eq(pattern_ty, handle_ctx.body_ty);
                self.bind_pattern_vars_with_ctx(ctx, binding, handle_ctx.body_ty);
                let binding =
                    self.convert_pattern_with_expected_ctx(ctx, binding, handle_ctx.body_ty);
                (HandlerKind::Do { binding }, Mode::Infer)
            }
            HandlerKind::Fn {
                ability,
                op,
                params,
            } => {
                let operation = self.constrain_handler_params(
                    ctx,
                    HandlerOperationRequest {
                        ability,
                        op: op.clone(),
                        syntax_kind: OpDeclKind::Fn,
                        params,
                        arm_id: arm.id,
                        handle_ctx,
                    },
                );
                ctx.record_handler_operation(arm.id, operation.clone());
                (
                    HandlerKind::Fn {
                        ability: self.convert_ref_with_ctx(ctx, None, ability),
                        op: op.clone(),
                        params: params
                            .iter()
                            .map(|p| self.convert_pattern_with_ctx(ctx, p))
                            .collect(),
                    },
                    // A tail-resumptive arm returns the operation's resume
                    // value; the continuation carries that value to the
                    // enclosing handler result.
                    Mode::Check(operation.result),
                )
            }
            HandlerKind::Op {
                ability,
                op,
                params,
                resume_local_id,
            } => {
                let operation = self.constrain_handler_params(
                    ctx,
                    HandlerOperationRequest {
                        ability,
                        op: op.clone(),
                        syntax_kind: OpDeclKind::Op,
                        params,
                        arm_id: arm.id,
                        handle_ctx,
                    },
                );
                ctx.record_handler_operation(arm.id, operation.clone());
                // Check if the operation has a `-> Never` return type (non-resumptive).
                let is_non_resumptive = matches!(operation.result.kind(self.db()), TypeKind::Never);

                // Bind the synthetic `resume` local with a Continuation type
                // so that `resume(value)` calls inside `op` arms are typed correctly.
                // A `-> Never` operation cannot be resumed: any `resume` in its
                // arm is reported and typed as an error.
                if let Some(k_local_id) = *resume_local_id {
                    if is_non_resumptive {
                        ctx.record_non_resumptive_resume(
                            k_local_id,
                            operation.ability.name(self.db()),
                            op.clone(),
                        );
                        ctx.bind_local(k_local_id, ctx.error_type());
                    } else {
                        let arg_ty = operation.result;
                        let cont_ty = Type::new(
                            self.db(),
                            TypeKind::Continuation {
                                arg: arg_ty,
                                result: handle_ctx.answer_ty,
                                effect: handle_ctx.body_effect,
                            },
                        );
                        ctx.bind_local(k_local_id, cont_ty);
                        if let Some(body) = handle_ctx.evidence_body {
                            ctx.evidence.bind_continuation(k_local_id, body);
                        }
                    }
                }

                (
                    HandlerKind::Op {
                        ability: self.convert_ref_with_ctx(ctx, None, ability),
                        op: op.clone(),
                        params: params
                            .iter()
                            .map(|p| self.convert_pattern_with_ctx(ctx, p))
                            .collect(),
                        resume_local_id: *resume_local_id,
                    },
                    // An explicit operation arm either resumes (whose result
                    // is the handler answer) or aborts with that same answer.
                    Mode::Infer,
                )
            }
        };
        let contributes_answer = matches!(kind, HandlerKind::Op { .. } | HandlerKind::Do { .. });
        // Every arm runs with the evidence of the handle's installation; only
        // an `op` arm's resumes pass the handle body's evidence.
        let outer_evidence = match (&kind, handle_ctx.evidence_body) {
            (HandlerKind::Op { .. }, Some(body)) => Some(ctx.evidence.enter_op_arm(body)),
            _ => None,
        };
        let body = self.check_expr_with_ctx(ctx, &arm.body, body_mode);
        if let Some(outer_evidence) = outer_evidence {
            ctx.evidence.restore(outer_evidence);
        }
        if contributes_answer {
            let actual = ctx
                .get_node_type(body.id)
                .expect("checked handler body has a type");
            ctx.add_result_source(handle_ctx.node_id, body.id, actual);
        }
        HandlerArm {
            id: arm.id,
            kind,
            body,
        }
    }

    /// Constrain handler arm parameters to the ability operation's parameter types.
    ///
    /// This ensures handler arm pattern bindings (e.g., `x`, `y` in
    /// `op Multi::combine(x, y)`) are constrained to the operation's declared
    /// parameter types (e.g., `Nat, Nat`), enabling TDNR to resolve operators
    /// like `x + y`.
    fn constrain_handler_params(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        request: HandlerOperationRequest<'_, 'db>,
    ) -> InstantiatedHandlerOperation<'db> {
        let HandlerOperationRequest {
            ability,
            op,
            syntax_kind,
            params,
            arm_id,
            handle_ctx,
        } = request;
        let Some(ability_id) = self.extract_ability_id_from_ref(ability) else {
            if ctx.mark_handler_error(arm_id, "unresolved ability") {
                Diagnostic::new(
                    format!("handler operation '{}' has no resolved ability", op),
                    self.get_span(arm_id),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
        };
        let Some(op_info) = self.env.lookup_ability_op(ability_id, &op) else {
            if ctx.mark_handler_error(arm_id, "unknown operation") {
                Diagnostic::new(
                    format!("unknown handler operation '{}'", op),
                    self.get_span(arm_id),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
        };
        if syntax_kind != op_info.kind && ctx.mark_handler_error(arm_id, "operation kind") {
            Diagnostic::new(
                format!(
                    "handler arm uses @{:?} for '{}', but the declared operation is @{:?}",
                    syntax_kind, op, op_info.kind
                ),
                self.get_span(arm_id),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }

        // A handler must retain the exact instantiated ability from the handled
        // computation.  Reconstructing it from operation parameter/result types
        // loses phantom parameters and accepts ambiguous repeated variables.
        let mut solver = TypeSolver::new(self.db());
        solver.reserve_row_vars(ctx.next_row_var());
        for method in ctx.deferred_methods() {
            solver.defer_producer(
                method.node_id,
                method.result_ty,
                method.arg_types.clone(),
                method.effect,
            );
        }
        let _ = solver.solve(ctx.constraints_snapshot());
        ctx.reserve_row_vars(solver.next_row_var());
        let body_row = solver.row_subst().apply(self.db(), handle_ctx.body_effect);
        let mut matching_effects = Vec::new();
        for effect in body_row
            .effects(self.db())
            .iter()
            .filter(|effect| effect.ability_id == ability_id)
        {
            let effect = Effect {
                ability_id,
                args: effect
                    .args
                    .iter()
                    .map(|ty| {
                        solver
                            .type_subst()
                            .apply_with_rows(self.db(), *ty, solver.row_subst())
                    })
                    .collect(),
            };
            if !matching_effects.contains(&effect) {
                matching_effects.push(effect);
            }
        }
        let ability_args = match matching_effects.as_slice() {
            [effect] => effect.args.clone(),
            [] if self
                .env
                .lookup_ability(ability_id)
                .is_some_and(|info| info.type_params.is_empty()) =>
            {
                Vec::new()
            }
            [] if body_row.rest(self.db()).is_some() => handle_ctx
                .handled_effects
                .effects(self.db())
                .iter()
                .find(|effect| effect.ability_id == ability_id)
                .expect("handler selection was recorded during inference")
                .args
                .clone(),
            [] => {
                if ctx.mark_handler_error(arm_id, "missing ability instance") {
                    Diagnostic::new(
                        format!(
                            "cannot determine the instantiated ability for handler operation '{}'",
                            op
                        ),
                        self.get_span(arm_id),
                        DiagnosticSeverity::Error,
                        CompilationPhase::TypeChecking,
                    )
                    .accumulate(self.db());
                }
                return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
            }
            _ => {
                if ctx.mark_handler_error(arm_id, "ambiguous ability instance") {
                    Diagnostic::new(
                        format!(
                            "handler operation '{}' ambiguously matches multiple instantiated abilities",
                            op
                        ),
                        self.get_span(arm_id),
                        DiagnosticSeverity::Error,
                        CompilationPhase::TypeChecking,
                    )
                    .accumulate(self.db());
                }
                return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
            }
        };

        if let Some(selected) = handle_ctx
            .handled_effects
            .effects(self.db())
            .iter()
            .find(|effect| effect.ability_id == ability_id)
        {
            for (variable, selected_arg) in selected.args.iter().zip(&ability_args) {
                ctx.constrain_eq(*variable, *selected_arg);
            }
        }

        // Substitute ability type params into the operation's parameter types
        let op_param_types: Vec<Type<'db>> = op_info
            .param_types
            .iter()
            .map(|ty| {
                subst::substitute_bound_vars(self.db(), *ty, &ability_args).unwrap_or_else(
                    |index, max| {
                        panic!(
                            "handler param BoundVar index out of range: index={}, subst.len()={}",
                            index, max
                        )
                    },
                )
            })
            .collect();

        // Check arity before constraining
        if params.len() != op_param_types.len() {
            let span = params
                .first()
                .map(|p| self.get_span(p.id))
                .unwrap_or_else(|| self.get_span(arm_id));
            if ctx.mark_handler_error(arm_id, "parameter arity") {
                Diagnostic::new(
                    format!(
                        "handler arm has {} parameter(s), but operation '{}' expects {}",
                        params.len(),
                        op,
                        op_param_types.len()
                    ),
                    span,
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
        }

        // Infer, constrain, and bind each pattern to the corresponding op param type
        for (pattern, op_ty) in params.iter().zip(op_param_types.iter()) {
            let pattern_ty = self.infer_pattern_type_with_ctx(ctx, pattern);
            ctx.constrain_eq(pattern_ty, *op_ty);
            self.bind_pattern_vars_with_ctx(ctx, pattern, *op_ty);
        }
        let return_type =
            subst::substitute_bound_vars(self.db(), op_info.return_type, &ability_args)
                .unwrap_or_else(|index, max| {
                    panic!(
                        "handler return BoundVar index out of range: index={}, subst.len()={}",
                        index, max
                    )
                });
        InstantiatedHandlerOperation {
            ability: ability_id,
            ability_args,
            kind: op_info.kind,
            params: op_param_types,
            result: return_type,
        }
    }

    fn invalid_handler_operation(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        _op: Symbol,
        kind: OpDeclKind,
        parameter_count: usize,
    ) -> InstantiatedHandlerOperation<'db> {
        let error = ctx.error_type();
        InstantiatedHandlerOperation {
            ability: AbilityId::source(self.db(), Symbol::new("__invalid_handler_ability")),
            ability_args: Vec::new(),
            kind,
            params: vec![error; parameter_count],
            result: error,
        }
    }

    // =========================================================================
    // Annotation conversion for function body (UniVar-based)
    // =========================================================================

    /// Convert a type annotation to a Type within a function body.
    ///
    /// Uses FunctionInferenceContext for fresh type variables.
    fn annotation_to_type_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        ann: &crate::ast::TypeAnnotation,
    ) -> Type<'db> {
        use crate::ast::TypeAnnotationKind;

        if let Some(ty) = ctx.annotation_type(ann.id) {
            return ty;
        }
        let ty = match &ann.kind {
            TypeAnnotationKind::Named(name) if ctx.annotation_type_parameter(name).is_some() => ctx
                .annotation_type_parameter(name)
                .expect("known signature parameter"),
            TypeAnnotationKind::Named(name) => {
                if *name == "Int" {
                    ctx.int_type()
                } else if *name == "Nat" {
                    ctx.nat_type()
                } else if *name == "Float" {
                    ctx.float_type()
                } else if *name == "Bool" {
                    ctx.bool_type()
                } else if *name == "Bytes" {
                    ctx.bytes_type()
                } else if *name == "Rune" {
                    ctx.rune_type()
                } else if *name == "Nil" {
                    ctx.nil_type()
                } else if *name == "Never" {
                    ctx.never_type()
                } else {
                    ctx.named_type_in_scope(name.clone(), vec![], self.current_prefix())
                }
            }
            TypeAnnotationKind::Path(parts) => ctx.path_type(parts),
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_with_ctx(ctx, ctor);
                if let TypeKind::Named { id, name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_with_ctx(ctx, a))
                        .collect();
                    ctx.named_type_with_id(*id, name.clone(), arg_types)
                } else {
                    ctx.error_type()
                }
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| self.annotation_to_type_with_ctx(ctx, p))
                    .collect();
                let result_ty = self.annotation_to_type_with_ctx(ctx, result);
                let mut rows = Vec::new();
                for ability in abilities {
                    let row = match &ability.kind {
                        TypeAnnotationKind::Named(name) if crate::ast::is_type_variable(name) => {
                            ctx.annotation_row(name.clone())
                        }
                        TypeAnnotationKind::Infer => ctx.fresh_row_var(),
                        _ => continue,
                    };
                    if !rows.contains(&row) {
                        rows.push(row);
                    }
                }
                let rest = match rows.as_slice() {
                    [] => None,
                    [row] => Some(*row),
                    _ => {
                        let result = ctx.fresh_row_var();
                        ctx.constrain_row_union(crate::ast::RowUnion {
                            sources: rows
                                .into_iter()
                                .map(|row| EffectRow::open(self.db(), row))
                                .collect(),
                            result: EffectRow::open(self.db(), result),
                        });
                        Some(result)
                    }
                };
                let converted = crate::ast::abilities_to_effect_row(
                    self.db(),
                    abilities,
                    self.current_prefix(),
                    &mut |a| self.annotation_to_type_with_ctx(ctx, a),
                    || rest.expect("row annotations allocate a tail"),
                );
                let effect = EffectRow::new(self.db(), converted.effects(self.db()), rest);
                ctx.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.annotation_to_type_with_ctx(ctx, e))
                    .collect();
                ctx.tuple_type(elem_types)
            }
            TypeAnnotationKind::Infer => ctx.fresh_type_var(),
            TypeAnnotationKind::Error => ctx.error_type(),
        };
        ctx.record_annotation_type(ann.id, ty);
        ty
    }

    // =========================================================================
    // Handle expression support
    // =========================================================================

    /// Extract the ability ID from a ResolvedRef.
    ///
    /// Used by handle expression to determine which abilities are being handled.
    /// Returns None for references that don't represent abilities.
    /// Report each handled ability whose operations lack a handler arm.
    ///
    /// An ability with at least one `fn`/`op` arm in a `handle` must provide
    /// an arm for every declared operation; a missing arm is never an
    /// implicit forward. Abilities without arms are not handled and need no
    /// check. Missing operations are listed in name order.
    fn report_missing_handler_arms(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        handle_id: NodeId,
        handlers: &[HandlerArm<ResolvedRef<'db>>],
        handled_ability_ids: &[AbilityId<'db>],
    ) {
        let mut checked = HashSet::default();
        for &ability_id in handled_ability_ids {
            if !checked.insert(ability_id) {
                continue;
            }
            let Some(info) = self.env.lookup_ability(ability_id) else {
                continue;
            };
            let arms: Vec<(NodeId, &Symbol)> = handlers
                .iter()
                .filter_map(|handler| match &handler.kind {
                    HandlerKind::Fn { ability, op, .. } | HandlerKind::Op { ability, op, .. }
                        if self.extract_ability_id_from_ref(ability) == Some(ability_id) =>
                    {
                        Some((handler.id, op))
                    }
                    _ => None,
                })
                .collect();
            let Some(&(first_arm, _)) = arms.first() else {
                continue;
            };
            let covered: HashSet<&Symbol> = arms.iter().map(|(_, op)| *op).collect();
            let mut missing: Vec<&Symbol> = info
                .operations
                .keys()
                .filter(|op| !covered.contains(op))
                .collect();
            if missing.is_empty() {
                continue;
            }
            missing.sort_by(|left, right| {
                left.with_str(|left| right.with_str(|right| left.cmp(right)))
            });
            // Keyed by the ability's first arm so each handled ability of
            // this handle reports once.
            if !ctx.mark_handler_error(first_arm, "missing handler arm") {
                continue;
            }
            let plural = if missing.len() == 1 { "an arm" } else { "arms" };
            Diagnostic::new(
                format!(
                    "handling `{}` is missing {plural} for {}",
                    ability_id.name(self.db()),
                    missing
                        .iter()
                        .format_with(", ", |op, f| f(&format_args!("`{op}`")))
                ),
                self.get_span(handle_id),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }
    }

    fn extract_ability_id_from_ref(&self, resolved: &ResolvedRef<'db>) -> Option<AbilityId<'db>> {
        match resolved {
            ResolvedRef::AbilityOp { ability, .. } => Some(*ability),
            ResolvedRef::Ability { id } => Some(*id),
            ResolvedRef::TypeDef { id } => {
                // TypeDef might be an ability reference in handler context
                // Create an AbilityId with the same qualified name
                Some(AbilityId::source(
                    self.db(),
                    id.qualified(self.db()).clone(),
                ))
            }
            // Other reference types are not abilities
            _ => None,
        }
    }

    /// Remove handled effects from an effect row.
    ///
    /// Creates a new effect row with the specified abilities removed.
    /// If the row has a row variable tail, we add a constraint to ensure
    /// the tail cannot contain the handled effects.
    ///
    /// This version uses AbilityId for proper handling of parameterized abilities:
    /// `State(Int)` and `State(Bool)` are now correctly distinguished.
    fn remove_handled_effects(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        source: EffectRow<'db>,
        removed: EffectRow<'db>,
    ) -> EffectRow<'db> {
        let result = ctx.fresh_effect_row();
        ctx.constrain_row_removal(crate::ast::RowRemoval {
            source,
            removed,
            result,
        });
        result
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

    use crate::ast::{
        AbilityId, EffectRow, Expr, ExprKind, FuncDefId, HandlerArm, HandlerKind, LocalId, NodeId,
        OpDeclKind, Pattern, PatternKind, ResolvedRef, SpanMap, Type, TypeAnnotation,
        TypeAnnotationKind, TypeDefId, TypeKind,
    };
    use crate::typeck::context::{AbilityInfo, AbilityOpInfo};
    use crate::typeck::func_context::HandleContext;
    use crate::typeck::{FunctionInferenceContext, ModuleTypeEnv};

    use super::{Mode, TypeChecker};

    /// Helper to create a TypeChecker for testing.
    fn make_test_checker(db: &dyn salsa::Database) -> TypeChecker<'_> {
        TypeChecker::new(db, SpanMap::default())
    }

    /// Helper to create a FunctionInferenceContext for testing.
    fn make_test_ctx<'a, 'db>(
        db: &'db dyn salsa::Database,
        env: &'a ModuleTypeEnv<'db>,
    ) -> FunctionInferenceContext<'a, 'db> {
        let func_id = FuncDefId::new(db, Symbol::new("test_func"));
        FunctionInferenceContext::new(db, env, func_id)
    }

    /// Helper to create a TypeAnnotation with a given kind.
    fn make_annotation(kind: TypeAnnotationKind) -> TypeAnnotation {
        static NEXT_ANNOTATION: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(1);
        TypeAnnotation {
            id: NodeId::from_raw(
                NEXT_ANNOTATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ),
            kind,
        }
    }

    fn bind_pattern<'db>(id: usize, name: Symbol, local_id: LocalId) -> Pattern<ResolvedRef<'db>> {
        Pattern::new(
            NodeId::from_raw(id),
            PatternKind::Bind {
                name,
                local_id: Some(local_id),
            },
        )
    }

    fn local_expr<'db>(id: usize, name: Symbol, local_id: LocalId) -> Expr<ResolvedRef<'db>> {
        Expr::new(
            NodeId::from_raw(id),
            ExprKind::Var(ResolvedRef::local(local_id, name)),
        )
    }

    // The current CST lowering does not preserve let annotations. Exercise
    // both checker visits directly until that separate syntax gap is closed.
    #[salsa_test]
    fn annotated_let_eliminates_never_without_retyping_its_producer(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let never = Type::new(db, TypeKind::Never);
        let nat = Type::new(db, TypeKind::Nat);
        let source = LocalId::new(0);
        let binding = LocalId::new(1);
        let source_name = Symbol::new("impossible");
        let statement = crate::ast::Stmt::Let {
            id: NodeId::from_raw(1),
            pattern: bind_pattern(2, Symbol::new("value"), binding),
            value: local_expr(3, source_name, source),
            ty: Some(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Nat",
            )))),
        };
        for convert in [false, true] {
            let mut ctx = make_test_ctx(db, &checker.env);
            ctx.bind_local(source, never);
            if convert {
                checker.convert_stmt_with_ctx(&mut ctx, &statement);
                assert_eq!(ctx.get_node_type(NodeId::from_raw(3)), Some(never));
            } else {
                checker.infer_stmt_and_bind_with_ctx(&mut ctx, &statement);
            }
            let mut solver = super::TypeSolver::new(db);
            solver.solve(ctx.take_constraints()).unwrap();
            solver.finalize_relations().unwrap();
            assert_eq!(ctx.lookup_local(binding), Some(nat));
            assert_eq!(ctx.lookup_local(source), Some(never));
        }
    }

    #[salsa_test]
    fn revisited_lambda_checks_a_different_expected_result(db: &salsa::DatabaseImpl) {
        let checker = make_test_checker(db);
        let mut ctx = make_test_ctx(db, &checker.env);
        let lambda = Expr::new(
            NodeId::from_raw(1),
            ExprKind::Lambda {
                params: vec![],
                body: Expr::new(NodeId::from_raw(2), ExprKind::NatLit(42)),
            },
        );
        let nat = ctx.nat_type();
        let boolean = ctx.bool_type();
        let expected = ctx.func_type(vec![], nat, EffectRow::pure(db));
        checker.check_expr_with_ctx(&mut ctx, &lambda, Mode::Check(expected));
        checker.check_expr_with_ctx(&mut ctx, &lambda, Mode::Infer);
        let mut solver = super::TypeSolver::new(db);
        solver.solve(ctx.constraints_snapshot()).unwrap();
        let incompatible = ctx.func_type(vec![], boolean, EffectRow::pure(db));
        checker.check_expr_with_ctx(&mut ctx, &lambda, Mode::Check(incompatible));
        assert!(solver.solve(ctx.take_constraints()).is_err());
    }

    #[salsa_test]
    fn annotated_lambda_uses_context_on_its_first_visit(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let mut ctx = make_test_ctx(db, &checker.env);
        let source = LocalId::new(0);
        let binding = LocalId::new(1);
        let name = Symbol::new("impossible");
        let never = Type::new(db, TypeKind::Never);
        ctx.bind_local(source, never);
        let statement = crate::ast::Stmt::Let {
            id: NodeId::from_raw(1),
            pattern: bind_pattern(2, Symbol::new("thunk"), binding),
            value: Expr::new(
                NodeId::from_raw(3),
                ExprKind::Lambda {
                    params: vec![],
                    body: local_expr(4, name, source),
                },
            ),
            ty: Some(make_annotation(TypeAnnotationKind::Func {
                params: vec![],
                result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                    "Nat",
                )))),
                abilities: vec![],
            })),
        };
        checker.infer_stmt_and_bind_with_ctx(&mut ctx, &statement);
        checker.convert_stmt_with_ctx(&mut ctx, &statement);
        let mut solver = super::TypeSolver::new(db);
        solver.solve(ctx.take_constraints()).unwrap();
        solver.finalize_relations().unwrap();
        let callable = ctx.lookup_local(binding).unwrap();
        let TypeKind::Func { result, .. } = callable.kind(db) else {
            panic!("expected callable");
        };
        assert!(matches!(result.kind(db), TypeKind::Nat));
        assert_eq!(ctx.get_node_type(NodeId::from_raw(4)), Some(never));
    }

    #[salsa_test]
    fn test_handler_do_binding_does_not_leak(db: &dyn salsa::Database) {
        let mut checker = make_test_checker(db);
        let name = Symbol::new("result");
        let body_ty = Type::new(db, TypeKind::Nat);
        let ability_id = AbilityId::source(db, Symbol::new("Test"));
        let get = Symbol::new("get");
        checker.env.register_ability(
            ability_id,
            AbilityInfo {
                id: ability_id,
                type_params: vec![],
                operations: [(
                    get.clone(),
                    AbilityOpInfo {
                        name: get.clone(),
                        kind: OpDeclKind::Op,
                        param_types: vec![],
                        return_type: Type::new(db, TypeKind::Nat),
                    },
                )]
                .into_iter()
                .collect::<HashMap<_, _>>(),
            },
        );
        let mut ctx = make_test_ctx(db, &checker.env);
        let handle_node_id = NodeId::from_raw(0);
        let _ = ctx.begin_result_join(handle_node_id);
        let handle_ctx = HandleContext {
            handled_effects: EffectRow::pure(db),
            node_id: handle_node_id,
            answer_ty: body_ty,
            body_ty,
            body_effect: EffectRow::pure(db),
            evidence_body: None,
        };

        let do_arm = HandlerArm {
            id: NodeId::from_raw(1),
            kind: HandlerKind::Do {
                binding: bind_pattern(2, name.clone(), LocalId::new(1)),
            },
            body: local_expr(3, name.clone(), LocalId::new(1)),
        };
        let converted_do = checker.convert_handler_arm_with_ctx(&mut ctx, &do_arm, &handle_ctx);
        assert!(matches!(
            &*converted_do.body.kind,
            ExprKind::Var(reference) if reference.ty == body_ty
        ));

        let ability = ResolvedRef::ability(ability_id);
        let op_arm = HandlerArm {
            id: NodeId::from_raw(4),
            kind: HandlerKind::Op {
                ability,
                op: get,
                params: vec![],
                resume_local_id: None,
            },
            body: local_expr(5, name.clone(), LocalId::UNRESOLVED),
        };
        let converted_op = checker.convert_handler_arm_with_ctx(&mut ctx, &op_arm, &handle_ctx);
        assert!(
            matches!(
                &*converted_op.body.kind,
                ExprKind::Var(reference)
                    if matches!(reference.ty.kind(db), TypeKind::UniVar { .. })
            ),
            "do-arm binding should not type an unresolved name in a later operation arm"
        );

        let outside = checker.check_expr_with_ctx(
            &mut ctx,
            &local_expr(6, name, LocalId::UNRESOLVED),
            Mode::Infer,
        );
        assert!(
            matches!(
                &*outside.kind,
                ExprKind::Var(reference)
                    if matches!(reference.ty.kind(db), TypeKind::UniVar { .. })
            ),
            "do-arm binding should not type an unresolved name outside the handle"
        );
    }

    #[salsa_test]
    fn test_same_name_handler_bindings_are_independent(db: &dyn salsa::Database) {
        let mut checker = make_test_checker(db);
        let ability_id = AbilityId::source(db, Symbol::new("Choice"));
        let nat_op = Symbol::new("nat");
        let bool_op = Symbol::new("bool");
        let nat_ty = Type::new(db, TypeKind::Nat);
        let bool_ty = Type::new(db, TypeKind::Bool);
        checker.env.register_ability(
            ability_id,
            AbilityInfo {
                id: ability_id,
                type_params: vec![],
                operations: [
                    (
                        nat_op.clone(),
                        AbilityOpInfo {
                            name: nat_op.clone(),
                            kind: OpDeclKind::Op,
                            param_types: vec![nat_ty],
                            return_type: Type::new(db, TypeKind::Nil),
                        },
                    ),
                    (
                        bool_op.clone(),
                        AbilityOpInfo {
                            name: bool_op.clone(),
                            kind: OpDeclKind::Op,
                            param_types: vec![bool_ty],
                            return_type: Type::new(db, TypeKind::Nil),
                        },
                    ),
                ]
                .into_iter()
                .collect::<HashMap<_, _>>(),
            },
        );

        let mut ctx = make_test_ctx(db, &checker.env);
        let handle_node_id = NodeId::from_raw(0);
        let _ = ctx.begin_result_join(handle_node_id);
        let handle_ctx = HandleContext {
            handled_effects: EffectRow::pure(db),
            node_id: handle_node_id,
            answer_ty: Type::new(db, TypeKind::Nil),
            body_ty: Type::new(db, TypeKind::Nil),
            body_effect: EffectRow::pure(db),
            evidence_body: None,
        };
        let name = Symbol::new("value");
        let ability = ResolvedRef::ability(ability_id);
        let make_arm = |id, op, local_id| HandlerArm {
            id: NodeId::from_raw(id),
            kind: HandlerKind::Op {
                ability: ability.clone(),
                op,
                params: vec![bind_pattern(id + 1, name.clone(), local_id)],
                resume_local_id: None,
            },
            body: local_expr(id + 2, name.clone(), local_id),
        };

        let nat_arm = checker.convert_handler_arm_with_ctx(
            &mut ctx,
            &make_arm(10, nat_op, LocalId::new(10)),
            &handle_ctx,
        );
        let bool_arm = checker.convert_handler_arm_with_ctx(
            &mut ctx,
            &make_arm(20, bool_op, LocalId::new(20)),
            &handle_ctx,
        );

        assert!(matches!(
            &*nat_arm.body.kind,
            ExprKind::Var(reference) if reference.ty == nat_ty
        ));
        assert!(matches!(
            &*bool_arm.body.kind,
            ExprKind::Var(reference) if reference.ty == bool_ty
        ));
        assert!(ctx.lookup_local(LocalId::new(10)).is_none());
        assert!(ctx.lookup_local(LocalId::new(20)).is_none());
        assert!(ctx.lookup_local_by_name(&name).is_none());
    }

    // =========================================================================
    // annotation_to_type_with_ctx tests
    // =========================================================================

    #[salsa_test]
    fn test_annotation_primitive_types(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        let cases = [
            ("Int", TypeKind::Int),
            ("Nat", TypeKind::Nat),
            ("Float", TypeKind::Float),
            ("Bool", TypeKind::Bool),
            ("Bytes", TypeKind::Bytes),
            ("Rune", TypeKind::Rune),
            ("Nil", TypeKind::Nil),
        ];

        for (name, expected_kind) in cases {
            let ann = make_annotation(TypeAnnotationKind::Named(Symbol::new(name)));
            let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);
            let expected = Type::new(db, expected_kind);
            assert_eq!(ty, expected, "Type annotation '{name}' mismatch");
        }
    }

    #[salsa_test]
    fn test_annotation_named_and_path_types(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // A qualified type identity preserves the complete path.
        let cases = [
            (TypeAnnotationKind::Named(Symbol::new("MyType")), "MyType"),
            (
                TypeAnnotationKind::Path(vec![Symbol::new("std"), Symbol::new("Option")]),
                "std::Option",
            ),
        ];

        for (kind, expected_name) in cases {
            let ann = make_annotation(kind);
            let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);
            let TypeKind::Named { name, args, .. } = ty.kind(db) else {
                panic!(
                    "{expected_name}: annotation should be Named, got {:?}",
                    ty.kind(db)
                );
            };
            assert_eq!(*name, Symbol::new(expected_name), "{expected_name}");
            assert!(args.is_empty(), "{expected_name}: unexpected args {args:?}");
        }
    }

    #[salsa_test]
    fn test_annotation_app(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // List(Int)
        let ann = make_annotation(TypeAnnotationKind::App {
            ctor: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "List",
            )))),
            args: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Int",
            )))],
        });
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

        // Should be Named { name: "List", args: [Int] }
        if let TypeKind::Named { name, args, .. } = ty.kind(db) {
            assert_eq!(*name, Symbol::new("List"));
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Type::new(db, TypeKind::Int));
        } else {
            panic!("App type should be Named, got {:?}", ty.kind(db));
        }
    }

    #[salsa_test]
    fn test_annotation_func_simple(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // fn(Int) -> Bool
        let ann = make_annotation(TypeAnnotationKind::Func {
            params: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Int",
            )))],
            result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Bool",
            )))),
            abilities: vec![], // pure
        });
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

        if let TypeKind::Func {
            params,
            result,
            effect,
            ..
        } = ty.kind(db)
        {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], Type::new(db, TypeKind::Int));
            assert_eq!(*result, Type::new(db, TypeKind::Bool));
            // Empty abilities means open effect row (fresh row var)
            assert!(effect.rest(db).is_some() || effect.is_pure(db));
        } else {
            panic!("Func annotation should be Func type");
        }
    }

    #[salsa_test]
    fn test_annotation_func_with_effects(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // fn(Int) ->{IO} Bool
        let ann = make_annotation(TypeAnnotationKind::Func {
            params: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Int",
            )))],
            result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Bool",
            )))),
            abilities: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "IO",
            )))],
        });
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

        if let TypeKind::Func {
            params,
            result,
            effect,
            ..
        } = ty.kind(db)
        {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], Type::new(db, TypeKind::Int));
            assert_eq!(*result, Type::new(db, TypeKind::Bool));
            // Should have IO effect
            let effects = effect.effects(db);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].ability_id.name(db), Symbol::new("IO"));
        } else {
            panic!("Func annotation should be Func type");
        }
    }

    #[salsa_test]
    fn test_annotation_tuple(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // (Int, String)
        let ann = make_annotation(TypeAnnotationKind::Tuple(vec![
            make_annotation(TypeAnnotationKind::Named(Symbol::new("Int"))),
            make_annotation(TypeAnnotationKind::Named(Symbol::new("String"))),
        ]));
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

        if let TypeKind::Tuple(elems) = ty.kind(db) {
            assert_eq!(elems.len(), 2);
            assert_eq!(elems[0], Type::new(db, TypeKind::Int));
            assert_eq!(elems[1], Type::new(db, TypeKind::string(db)));
        } else {
            panic!("Tuple annotation should be Tuple type");
        }
    }

    #[salsa_test]
    fn test_annotation_infer(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // _
        let ann = make_annotation(TypeAnnotationKind::Infer);
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

        // Should be fresh UniVar
        assert!(
            matches!(ty.kind(db), TypeKind::UniVar { .. }),
            "Infer annotation should produce UniVar"
        );
    }

    // =========================================================================
    // substitute_bound_vars tests (via TypeChecker methods)
    // =========================================================================

    // The checker delegates to `subst::substitute_bound_vars`, whose laws are
    // properties in `typeck::subst::laws`; this checks the wrapper's panic.

    #[salsa_test]
    #[should_panic(expected = "BoundVar index out of range")]
    fn test_substitute_out_of_bounds_panics(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);

        // BoundVar(5) + [Int] → should panic (out of bounds)
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 5 });
        let int_ty = Type::new(db, TypeKind::Int);
        let args = vec![int_ty];

        // This should panic
        checker.substitute_bound_vars(bound_var, &args);
    }

    // =========================================================================
    // extract_list_element_type tests
    // =========================================================================

    #[salsa_test]
    fn test_extract_list_element_type_of_list_types(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        let list_of = |args| {
            Type::new(
                db,
                TypeKind::Named {
                    id: TypeDefId::builtin_list(db),
                    name: Symbol::new("List"),
                    args,
                },
            )
        };
        let int_ty = Type::new(db, TypeKind::Int);
        let string_ty = Type::new(db, TypeKind::string(db));
        let list_int = list_of(vec![int_ty]);

        let cases = [
            ("List<Int>", list_int, int_ty),
            ("List<String>", list_of(vec![string_ty]), string_ty),
            (
                "App(List, [Int])",
                Type::new(
                    db,
                    TypeKind::App {
                        ctor: list_of(vec![]),
                        args: vec![int_ty],
                    },
                ),
                int_ty,
            ),
            ("List<List<Int>>", list_of(vec![list_int]), list_int),
        ];

        for (name, list_ty, expected) in cases {
            let elem_ty = checker.extract_list_element_type(list_ty, &mut ctx);
            assert_eq!(elem_ty, expected, "{name}");
        }
    }

    #[salsa_test]
    fn test_extract_list_element_type_of_non_list_returns_fresh_var(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        let int_ty = Type::new(db, TypeKind::Int);
        let cases = [
            ("Int", int_ty),
            (
                "Option<Int>",
                Type::new(
                    db,
                    TypeKind::Named {
                        id: TypeDefId::synthetic(db, Symbol::new("Option")),
                        name: Symbol::new("Option"),
                        args: vec![int_ty],
                    },
                ),
            ),
            (
                "List with no type args",
                Type::new(
                    db,
                    TypeKind::Named {
                        id: TypeDefId::builtin_list(db),
                        name: Symbol::new("List"),
                        args: vec![],
                    },
                ),
            ),
        ];

        for (name, ty) in cases {
            let elem_ty = checker.extract_list_element_type(ty, &mut ctx);
            assert!(
                matches!(elem_ty.kind(db), TypeKind::UniVar { .. }),
                "{name}: expected a fresh UniVar, got {:?}",
                elem_ty.kind(db)
            );
        }
    }

    #[salsa_test]
    fn local_annotation_unions_preserve_names_and_revisit_identity(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);
        let callback = |names: &[&str]| {
            make_annotation(TypeAnnotationKind::Func {
                params: vec![],
                result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                    "Nil",
                )))),
                abilities: names
                    .iter()
                    .map(|name| make_annotation(TypeAnnotationKind::Named(Symbol::new(name))))
                    .collect(),
            })
        };
        let ann = make_annotation(TypeAnnotationKind::Func {
            params: vec![callback(&["e1"]), callback(&["e2"]), callback(&["e1"])],
            result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Nil",
            )))),
            abilities: vec![
                make_annotation(TypeAnnotationKind::Named(Symbol::new("e1"))),
                make_annotation(TypeAnnotationKind::Named(Symbol::new("e2"))),
            ],
        });
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);
        let allocated = ctx.next_row_var();
        assert_eq!(checker.annotation_to_type_with_ctx(&mut ctx, &ann), ty);
        assert_eq!(ctx.next_row_var(), allocated);
        let TypeKind::Func { params, effect, .. } = ty.kind(db) else {
            panic!("function")
        };
        let row = |index: usize| match params[index].kind(db) {
            TypeKind::Func { effect, .. } => *effect,
            _ => panic!("callback"),
        };
        assert_eq!(row(0), row(2));
        assert_ne!(row(0), row(1));
        assert!(
            matches!(ctx.take_constraints().constraints(), [super::super::super::constraint::Constraint::RowUnion(union, None)]
            if union.sources == vec![row(0), row(1)] && union.result == *effect)
        );
    }
}
