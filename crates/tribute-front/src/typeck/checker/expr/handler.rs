//! `resume` and the arms of `handle` expressions.

use rustc_hash::FxHashSet as HashSet;
use salsa::Accumulator;

use itertools::Itertools;
use tribute_core::{CompilationPhase, DiagnosticSeverity};
use trunk_ir::{Symbol, SymbolPath};

use crate::ast::{
    AbilityId, AbilityInstance, AbilityRow, Expr, HandlerArm, HandlerKind, LocalId, NodeId,
    OpDeclKind, Pattern, ResolvedRef, Type, TypeKind, TypedRef,
};
use crate::typeck::InstantiatedHandlerOperation;
use crate::typeck::checker::{Mode, NodeDiagnostic, TypeChecker};
use crate::typeck::func_context::{FunctionInferenceContext, HandleContext};
use crate::typeck::solver::TypeSolver;
use crate::typeck::subst;

/// Source and typing context needed to resolve one handler operation arm.
struct HandlerOperationRequest<'a, 'db> {
    ability: &'a ResolvedRef<'db>,
    op: Symbol,
    syntax_kind: OpDeclKind,
    params: &'a [Pattern<ResolvedRef<'db>>],
    arm_id: NodeId,
    handle_ctx: &'a HandleContext<'db>,
}

impl<'db> TypeChecker<'db> {
    /// Type `resume arg` against the continuation bound by the enclosing
    /// `op` arm.
    pub(super) fn infer_resume_type_with_ctx(
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
                NodeDiagnostic::new(
                    format!(
                        "cannot `resume` in the handler for `{ability}::{op}`, which returns `Never`"
                    ),
                    resume,
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

    /// Convert a handler arm.
    pub(super) fn convert_handler_arm_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: &HandlerArm<ResolvedRef<'db>>,
        handle_ctx: &HandleContext<'db>,
    ) -> HandlerArm<TypedRef<'db>> {
        ctx.with_scope(|ctx| self.convert_handler_arm_in_scope(ctx, arm, handle_ctx))
    }

    /// Convert a handler arm within its arm-local scope.
    fn convert_handler_arm_in_scope(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: &HandlerArm<ResolvedRef<'db>>,
        handle_ctx: &HandleContext<'db>,
    ) -> HandlerArm<TypedRef<'db>> {
        let (kind, body_mode) = match &arm.kind {
            HandlerKind::Do { binding } => {
                let pattern_ty =
                    self.infer_pattern_type_with_ctx(ctx, binding, Some(handle_ctx.body_ty));
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
                NodeDiagnostic::new(
                    format!("handler operation '{}' has no resolved ability", op),
                    arm_id,
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
        };
        let Some(op_info) = self.env.lookup_ability_op(ability_id, &op) else {
            if ctx.mark_handler_error(arm_id, "unknown operation") {
                NodeDiagnostic::new(
                    format!("unknown handler operation '{}'", op),
                    arm_id,
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
        };
        if syntax_kind != op_info.kind && ctx.mark_handler_error(arm_id, "operation kind") {
            NodeDiagnostic::new(
                format!(
                    "handler arm uses @{:?} for '{}', but the declared operation is @{:?}",
                    syntax_kind, op, op_info.kind
                ),
                arm_id,
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
        for constructor in ctx.deferred_constructors() {
            constructor.defer_in(self.db(), &mut solver);
        }
        let _ = solver.solve(ctx.constraints_snapshot());
        ctx.reserve_row_vars(solver.next_row_var());
        let body_row = solver.row_subst().apply(self.db(), handle_ctx.body_effect);
        let mut matching_effects = Vec::new();
        for effect in body_row
            .instances(self.db())
            .iter()
            .filter(|effect| effect.ability_id == ability_id)
        {
            let effect = AbilityInstance {
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
                .instances(self.db())
                .iter()
                .find(|effect| effect.ability_id == ability_id)
                .expect("handler selection was recorded during inference")
                .args
                .clone(),
            [] => {
                if ctx.mark_handler_error(arm_id, "missing ability instance") {
                    NodeDiagnostic::new(
                        format!(
                            "cannot determine the instantiated ability for handler operation '{}'",
                            op
                        ),
                        arm_id,
                        DiagnosticSeverity::Error,
                        CompilationPhase::TypeChecking,
                    )
                    .accumulate(self.db());
                }
                return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
            }
            _ => {
                if ctx.mark_handler_error(arm_id, "ambiguous ability instance") {
                    NodeDiagnostic::new(
                        format!(
                            "handler operation '{}' ambiguously matches multiple instantiated abilities",
                            op
                        ),
                        arm_id,
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
            .instances(self.db())
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
                    |subst::BoundVarOutOfBounds { index, max }| {
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
            let node = params.first().map_or(arm_id, |p| p.id);
            if ctx.mark_handler_error(arm_id, "parameter arity") {
                NodeDiagnostic::new(
                    format!(
                        "handler arm has {} parameter(s), but operation '{}' expects {}",
                        params.len(),
                        op,
                        op_param_types.len()
                    ),
                    node,
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            return self.invalid_handler_operation(ctx, op, syntax_kind, params.len());
        }

        // Infer, constrain, and bind each pattern to the corresponding op param type
        for (pattern, op_ty) in params.iter().zip(op_param_types.iter()) {
            let pattern_ty = self.infer_pattern_type_with_ctx(ctx, pattern, Some(*op_ty));
            ctx.constrain_eq(pattern_ty, *op_ty);
            self.bind_pattern_vars_with_ctx(ctx, pattern, *op_ty);
        }
        let return_type =
            subst::substitute_bound_vars(self.db(), op_info.return_type, &ability_args)
                .unwrap_or_else(|subst::BoundVarOutOfBounds { index, max }| {
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
            ability: AbilityId::source(self.db(), SymbolPath::new(["__invalid_handler_ability"])),
            ability_args: Vec::new(),
            kind,
            params: vec![error; parameter_count],
            result: error,
        }
    }

    /// Report every `do` arm after the first, and every arm for an
    /// operation that an earlier arm of the same `handle` already handles.
    pub(super) fn report_duplicate_handler_arms(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        handlers: &[HandlerArm<ResolvedRef<'db>>],
    ) {
        let mut seen_do = false;
        let mut seen_ops = HashSet::default();
        for handler in handlers {
            let message = match &handler.kind {
                HandlerKind::Do { .. } => {
                    if !std::mem::replace(&mut seen_do, true) {
                        continue;
                    }
                    "`handle` may have at most one `do` arm".to_owned()
                }
                HandlerKind::Fn { ability, op, .. } | HandlerKind::Op { ability, op, .. } => {
                    let Some(ability_id) = self.extract_ability_id_from_ref(ability) else {
                        continue;
                    };
                    if seen_ops.insert((ability_id, op)) {
                        continue;
                    }
                    format!(
                        "`handle` has more than one arm for `{}::{op}`",
                        ability_id.name(self.db())
                    )
                }
            };
            if !ctx.mark_handler_error(handler.id, "duplicate arm") {
                continue;
            }
            NodeDiagnostic::new(
                message,
                handler.id,
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }
    }

    /// Report each handled ability whose operations lack a handler arm.
    ///
    /// An ability with at least one `fn`/`op` arm in a `handle` must provide
    /// an arm for every declared operation; a missing arm is never an
    /// implicit forward. Abilities without arms are not handled and need no
    /// check. Missing operations are listed in name order.
    pub(super) fn report_missing_handler_arms(
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
            NodeDiagnostic::new(
                format!(
                    "handling `{}` is missing {plural} for {}",
                    ability_id.name(self.db()),
                    missing
                        .iter()
                        .format_with(", ", |op, f| f(&format_args!("`{op}`")))
                ),
                handle_id,
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }
    }

    /// Extract the ability ID from a ResolvedRef.
    ///
    /// Used by handle expression to determine which abilities are being handled.
    /// Returns None for references that don't represent abilities.
    pub(in crate::typeck::checker) fn extract_ability_id_from_ref(
        &self,
        resolved: &ResolvedRef<'db>,
    ) -> Option<AbilityId<'db>> {
        match resolved {
            ResolvedRef::AbilityOp { ability, .. } => Some(*ability),
            ResolvedRef::Ability { id } => Some(*id),
            ResolvedRef::TypeDef { id } => {
                // TypeDef might be an ability reference in handler context
                // Create an AbilityId with the same qualified name
                Some(AbilityId::source(self.db(), id.path(self.db()).clone()))
            }
            // Other reference types are not abilities
            _ => None,
        }
    }

    /// Remove handled effects from an ability row.
    ///
    /// Creates a new ability row with the specified abilities removed.
    /// If the row has a row variable tail, we add a constraint to ensure
    /// the tail cannot contain the handled effects.
    ///
    /// This version uses AbilityId for proper handling of parameterized abilities:
    /// `State(Int)` and `State(Bool)` are now correctly distinguished.
    pub(super) fn remove_handled_effects(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        source: AbilityRow<'db>,
        removed: AbilityRow<'db>,
    ) -> AbilityRow<'db> {
        let result = ctx.fresh_ability_row();
        ctx.constrain_row_removal(crate::ast::RowRemoval {
            source,
            removed,
            result,
        });
        result
    }
}
