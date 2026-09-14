//! Function-level type checking.
//!
//! Each function is type-checked with an isolated `FunctionInferenceContext`,
//! ensuring that type variables (UniVars) are fully resolved within the function
//! before moving to the next.

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use salsa::Accumulator;
use tribute_core::fmt::joined;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};

use crate::ast::{
    ExprKind, FuncDecl, FuncDefId, ResolvedRef, Type, TypeKind, TypeScheme, TypedRef, UniVarId,
    collect_effect_vars,
};

use super::super::constraint::{ConstraintOriginKind, ConstraintSet};
use super::super::func_context::FunctionInferenceContext;
use super::super::solver::TypeSolver;
#[cfg(test)]
use super::diagnostics::{format_solve_error, solve_error_context};
use super::{Mode, TypeChecker};

impl<'db> TypeChecker<'db> {
    /// Type check a function declaration with per-function inference.
    ///
    /// This method:
    /// 1. Creates a fresh `FunctionInferenceContext` for this function
    /// 2. Binds parameters using the registered type scheme
    /// 3. Checks the function body, generating constraints
    /// 4. Solves constraints for this function only
    /// 5. Applies substitution and generalization
    /// 6. Updates the function's type scheme in ModuleTypeEnv
    pub(crate) fn check_func_decl(
        &mut self,
        func: FuncDecl<ResolvedRef<'db>>,
    ) -> FuncDecl<TypedRef<'db>> {
        // 1. Create a fresh FunctionInferenceContext for this function
        // Use function definition ID for globally unique UniVar IDs
        let func_id = self.func_def_id(func.name);
        let mut ctx = FunctionInferenceContext::new(self.db(), &self.env, func_id);

        // 2. Get the function's registered type scheme and instantiate it

        // Get the instantiated function type (with UniVars) for later generalization
        let (param_types, expected_return, instantiated_func_ty, signature_instance) =
            self.get_func_signature_with_type(&mut ctx, func_id, &func);
        if let Some((scheme, instance)) = &signature_instance
            && let Some(names) = self.signature_row_names.get(&func_id)
        {
            for (name, original) in names {
                let index = scheme
                    .effect_params(self.db())
                    .iter()
                    .position(|row| row == original)
                    .expect("named signature row must be quantified");
                let row = instance.row_args[index]
                    .rest(self.db())
                    .expect("fresh signature row must be open");
                ctx.bind_annotation_row(*name, row);
            }
        }
        let diagnostic_func_id = func.id;
        let diagnostic_func_name = func.name;
        let diagnostic_effects = func.effects.clone();

        // Bind parameters: by LocalId when present, and also by name
        for (i, param) in func.params.iter().enumerate() {
            let ty = param_types
                .get(i)
                .copied()
                .unwrap_or_else(|| ctx.fresh_type_var());
            if let Some(local_id) = param.local_id {
                ctx.bind_local(local_id, ty);
            }
            ctx.bind_local_by_name(param.name, ty);
        }

        // Set effect row from the function's declared type before checking body
        let declared_effect =
            if let TypeKind::Func { effect, .. } = instantiated_func_ty.kind(self.db()) {
                // Omission is semantically `->{e}`, but the fresh generalized
                // tail is not an effect performed by the body. Infer residual
                // effects from a closed-empty accumulator and reattach the tail
                // to the resulting function type after solving.
                if func.effects.is_none() {
                    ctx.set_current_effect(crate::ast::EffectRow::pure(self.db()));
                } else {
                    ctx.set_current_effect(*effect);
                }
                Some(*effect)
            } else {
                None
            };

        ctx.effect_contract = declared_effect;
        // 3. Check body against expected return type
        let body = self.check_expr_with_ctx(&mut ctx, func.body, Mode::Check(expected_return));

        if func.effects.is_none()
            && ctx.current_effect().rest(self.db()).is_some()
            && let Some(declared) = declared_effect
        {
            ctx.constrain_row_eq(declared, ctx.current_effect());
        }
        if let Some(declared) = declared_effect.filter(|row| row.rest(self.db()).is_none()) {
            // A sole handler expression owns the whole body's residual row.
            // Retain that boundary's location after delayed union solving.
            // With preceding statements the row belongs to the whole function.
            let mut effect_body = &body;
            while let ExprKind::Block { stmts, value } = &*effect_body.kind {
                if !stmts.is_empty() {
                    break;
                }
                effect_body = value;
            }
            let (node_id, kind) = if matches!(&*effect_body.kind, ExprKind::Handle { .. }) {
                (effect_body.id, ConstraintOriginKind::HandlerBoundary)
            } else {
                (func.id, ConstraintOriginKind::Expression)
            };
            ctx.constrain_row_eq_at(declared, ctx.current_effect(), node_id, kind);
        }
        // 4. Solve constraints for this function only
        let constraints = ctx.take_constraints();
        // Take node_types now while ctx is still alive, before we need mutable self access
        let func_node_types = ctx.take_node_types();
        let mut func_instances = ctx.take_function_instances();
        let func_handler_operations = ctx.take_handler_operations();
        let func_perform_operations = ctx.take_perform_operations();
        let func_lambda_signatures = ctx.take_lambda_signatures();
        let local_generalizations = ctx.take_local_generalizations();
        // Save the accumulated effect row from the body before dropping ctx
        let body_effect_row = ctx.current_effect();
        // Take deferred methods for post-solve resolution
        let deferred_methods = ctx.take_deferred_methods();
        let next_row_var = ctx.next_row_var();
        // Drop ctx now to release the borrow of self.env
        drop(ctx);
        self.local_generalizations = local_generalizations;

        let mut solver = TypeSolver::new(self.db());
        solver.reserve_row_vars(next_row_var);
        for method in &deferred_methods {
            solver.defer_producer(method.node_id, method.result_ty, method.arg_types.clone());
        }

        let mut solve_failed = false;
        if let Err(error) = solver.solve_with_origin(constraints) {
            solve_failed = true;
            self.report_solve_error(
                diagnostic_func_id,
                diagnostic_func_name,
                diagnostic_effects.as_deref(),
                error,
            );
        }
        if let Err(error) = solver.finalize_relations() {
            self.report_solve_error(
                diagnostic_func_id,
                diagnostic_func_name,
                diagnostic_effects.as_deref(),
                error,
            );
        }

        // 4b. Post-solve: resolve deferred method calls
        // After solving, UniVar receiver types may now be concrete.
        // Look up methods and add type constraints for return types.
        let deferred_resolutions = self.resolve_deferred_methods(
            &mut solver,
            deferred_methods,
            func.id,
            &mut func_instances,
        );
        if let Err(error) = solver.finalize_relations() {
            self.report_solve_error(
                diagnostic_func_id,
                diagnostic_func_name,
                diagnostic_effects.as_deref(),
                error,
            );
        }

        // 5. Apply substitution and generalization
        let type_subst = solver.type_subst();
        let row_subst = solver.row_subst();

        // Only the exact root `main` is an entrypoint. Its omitted effect
        // annotation is closed over the effects actually performed by its body:
        // pure roots remain Direct and ambient-Io roots EvidenceDirect.  The
        // later control/backend pipeline owns any root completion adaptation.
        let is_root_main = crate::is_root_main(func.name, self.current_prefix().is_empty());

        // An omitted annotation denotes an open effect row. Preserve the
        // concrete residual effects discovered while checking the body, then
        // reattach the generalized tail supplied by the collected signature.
        let inferred_func_ty = if func.effects.is_none() {
            match instantiated_func_ty.kind(self.db()) {
                TypeKind::Func {
                    params,
                    result,
                    effect: declared_effect,
                    minimum_convention,
                    ..
                } => {
                    let inferred_effect = if is_root_main {
                        crate::ast::EffectRow::new(
                            self.db(),
                            body_effect_row.effects(self.db()).clone(),
                            None,
                        )
                    } else if body_effect_row.rest(self.db()).is_some() {
                        body_effect_row
                    } else {
                        crate::ast::EffectRow::new(
                            self.db(),
                            body_effect_row.effects(self.db()).clone(),
                            declared_effect.rest(self.db()),
                        )
                    };
                    Type::new(
                        self.db(),
                        TypeKind::Func {
                            params: params.clone(),
                            result: *result,
                            effect: inferred_effect,
                            minimum_convention: *minimum_convention,
                        },
                    )
                }
                _ => instantiated_func_ty,
            }
        } else {
            instantiated_func_ty
        };

        // Solver aliases can point at a representative created after a local
        // scheme was generalized. Preserve both spellings before collecting
        // body and deferred metadata variables for finalization.
        for (var, binding) in self.local_generalizations.clone() {
            let resolved = type_subst.apply_with_rows(
                self.db(),
                Type::new(self.db(), TypeKind::UniVar { id: var }),
                row_subst,
            );
            if let TypeKind::UniVar { id } = resolved.kind(self.db()) {
                self.local_generalizations.entry(*id).or_insert(binding);
            }
        }

        // Only interface and retained-relation variables are function binders.
        // Body-local existentials (for example an unused constructor argument)
        // stay owned by this body and must not become call-site arguments.
        let mut all_univars = Vec::new();
        type_subst.collect_univars_from_type(
            self.db(),
            inferred_func_ty,
            row_subst,
            &mut all_univars,
        );
        let mut body_univars = Vec::new();
        self.collect_univars_from_body(&body, type_subst, row_subst, &mut body_univars);
        self.collect_univars_from_deferred_resolutions(
            &deferred_resolutions,
            type_subst,
            row_subst,
            &mut body_univars,
        );
        for rows in solver
            .row_unions_for_type(inferred_func_ty)
            .into_iter()
            .map(|u| {
                u.sources
                    .into_iter()
                    .chain(std::iter::once(u.result))
                    .collect::<Vec<_>>()
            })
            .chain(
                solver
                    .row_removals_for_type(inferred_func_ty)
                    .into_iter()
                    .map(|r| r.rows().to_vec()),
            )
        {
            for row in rows {
                for effect in row.effects(self.db()) {
                    for arg in &effect.args {
                        type_subst.collect_univars_from_type(
                            self.db(),
                            *arg,
                            row_subst,
                            &mut all_univars,
                        );
                    }
                }
            }
        }
        for (index, var) in body_univars
            .into_iter()
            .filter(|var| !all_univars.contains(var))
            .enumerate()
        {
            self.local_generalizations
                .entry(var)
                .or_insert((func.id, index as u32));
        }
        all_univars.retain(|id| !self.local_generalizations.contains_key(id));
        let var_to_index: HashMap<UniVarId<'db>, u32> = all_univars
            .into_iter()
            .enumerate()
            .map(|(index, id)| (id, index as u32))
            .collect();

        // Apply substitution and generalization to the function type.
        let substituted_ty = type_subst.apply_with_rows(self.db(), inferred_func_ty, row_subst);

        // Validate that root `main` returns Nil.
        if is_root_main
            && let TypeKind::Func { result, .. } = substituted_ty.kind(self.db())
            && !matches!(result.kind(self.db()), TypeKind::Nil)
        {
            Diagnostic::new(
                format!("function 'main' must return Nil, but returns `{}`", result),
                self.get_span(func.id),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }

        // Validate that root `main` has no unhandled effects.
        // We check body_effect_row (the accumulated effect from type-checking the body)
        // rather than the function signature's effect row, because effect inference
        // tracks effects in the context's current_effect rather than constraining
        // the function type's row variable.
        if is_root_main {
            let resolved_effect = row_subst.apply(self.db(), body_effect_row);
            let unhandled = resolved_effect
                .effects(self.db())
                .iter()
                .filter(|effect| !effect.ability_id.is_builtin_io(self.db()))
                .collect_vec();
            if !unhandled.is_empty() {
                Diagnostic::new(
                    format!(
                        "function 'main' has unhandled effects: {}",
                        joined(", ", &unhandled)
                    ),
                    self.get_span(func.id),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
        }

        // Validate that functions with explicit closed effect annotations
        // do not use undeclared effects in their body.
        if func.effects.is_some()
            && let Some(declared) = declared_effect
        {
            let resolved_declared = row_subst.apply(self.db(), declared);
            if let Some(duplicate) = self
                .effect_annotation_origins
                .get(&func_id)
                .and_then(|origins| origins.find_duplicate(self.db(), resolved_declared))
            {
                Diagnostic::builder(
                    format!(
                        "function '{}' declares duplicate effect: {}",
                        func.name,
                        joined(", ", &duplicate.effects),
                    ),
                    self.get_span(duplicate.duplicate_annotation_id),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .label(
                    self.get_span(duplicate.first_annotation_id),
                    "first matching effect annotation is here",
                )
                .build()
                .accumulate(self.db());
            }

            // Only check if the declared row is closed (no rest variable)
            if resolved_declared.rest(self.db()).is_none() {
                let resolved_body = row_subst.apply(self.db(), body_effect_row);
                let declared_ids: HashSet<_> = resolved_declared
                    .effects(self.db())
                    .iter()
                    .map(|e| e.ability_id)
                    .collect();
                let body_effects = resolved_body.effects(self.db());
                let mut undeclared = body_effects
                    .iter()
                    .filter(|e| !declared_ids.contains(&e.ability_id))
                    .peekable();
                if !solve_failed && undeclared.peek().is_some() {
                    Diagnostic::new(
                        format!(
                            "function '{}' uses undeclared effects: {}",
                            func.name,
                            joined(", ", undeclared),
                        ),
                        self.get_span(func.id),
                        DiagnosticSeverity::Error,
                        CompilationPhase::TypeChecking,
                    )
                    .accumulate(self.db());
                }
            }
        }

        let generalized =
            type_subst.apply_generalization(self.db(), substituted_ty, row_subst, &var_to_index);

        // Create type params for all generalized UniVars
        let type_params: Vec<crate::ast::TypeParam> = (0..var_to_index.len())
            .map(|_| crate::ast::TypeParam::anonymous())
            .collect();

        let mut effect_params = collect_effect_vars(self.db(), generalized);
        for var in solver
            .row_union_variables(&solver.row_unions_for_type(inferred_func_ty))
            .1
            .into_iter()
            .chain(
                solver
                    .row_removal_variables(&solver.row_removals_for_type(inferred_func_ty))
                    .1,
            )
        {
            if !effect_params.contains(&var) {
                effect_params.push(var);
            }
        }
        let unions = solver
            .row_unions_for_type(inferred_func_ty)
            .iter()
            .map(|union| solver.generalize_row_union(union, &var_to_index))
            .collect();
        let new_scheme = TypeScheme::builder(type_params, effect_params, generalized)
            .row_unions(unions)
            .row_removals(
                solver
                    .row_removals_for_type(inferred_func_ty)
                    .iter()
                    .map(|r| solver.generalize_row_removal(r, &var_to_index))
                    .collect(),
            )
            .build(self.db());
        if let Some((source_scheme, instance)) = signature_instance {
            let mut types = vec![None; new_scheme.type_params(self.db()).len()];
            for (source_index, ty) in instance.type_args.iter().enumerate() {
                let ty = type_subst.apply_with_rows(self.db(), *ty, row_subst);
                if let TypeKind::UniVar { id } = ty.kind(self.db())
                    && let Some(index) = var_to_index.get(id)
                {
                    types[*index as usize] = Some(source_index);
                }
            }
            let rows = new_scheme
                .effect_params(self.db())
                .iter()
                .map(|target| {
                    instance.row_args.iter().position(|row| {
                        row_subst.apply(self.db(), *row).rest(self.db()) == Some(*target)
                    })
                })
                .collect();
            self.function_rebindings.insert(
                (func_id, source_scheme),
                super::FunctionRebinding {
                    scheme: new_scheme,
                    types,
                    rows,
                },
            );
        }
        // Update the function's type scheme with the generalized version
        self.env.register_function(func_id, new_scheme);

        // 6. Materialize the solved node types before rebuilding the body.
        // Post-solve case coverage consults these entries during body substitution.
        for (node_id, ty) in func_node_types {
            let substituted = self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index);
            self.node_types.insert(node_id, substituted);
        }

        // 7. Apply substitution and generalization to all TypedRef types in the body.
        let body = self.apply_subst_to_body(
            body,
            type_subst,
            row_subst,
            &var_to_index,
            &deferred_resolutions,
        );

        for (node, mut instance) in func_instances {
            instance.callable =
                self.apply_subst_to_type(instance.callable, type_subst, row_subst, &var_to_index);
            instance.type_arguments = instance
                .type_arguments
                .into_iter()
                .map(|ty| self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index))
                .collect();
            instance.row_arguments = instance
                .row_arguments
                .into_iter()
                .map(|row| {
                    let row = row_subst.apply(self.db(), row);
                    crate::typeck::solver::map_effect_row_type_args(self.db(), row, |ty| {
                        self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index)
                    })
                })
                .collect();
            self.function_instances.insert(node, instance);
        }
        for (arm_id, operation) in func_handler_operations {
            self.handler_operations.insert(
                arm_id,
                crate::typeck::InstantiatedHandlerOperation {
                    ability: operation.ability,
                    ability_args: operation
                        .ability_args
                        .into_iter()
                        .map(|ty| {
                            self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index)
                        })
                        .collect(),
                    kind: operation.kind,
                    params: operation
                        .params
                        .into_iter()
                        .map(|ty| {
                            self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index)
                        })
                        .collect(),
                    result: self.apply_subst_to_type(
                        operation.result,
                        type_subst,
                        row_subst,
                        &var_to_index,
                    ),
                },
            );
        }
        for (call_id, operation) in func_perform_operations {
            self.perform_operations.insert(
                call_id,
                crate::typeck::InstantiatedPerformOperation {
                    ability: operation.ability,
                    ability_args: operation
                        .ability_args
                        .into_iter()
                        .map(|ty| {
                            self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index)
                        })
                        .collect(),
                    kind: operation.kind,
                    params: operation
                        .params
                        .into_iter()
                        .map(|ty| {
                            self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index)
                        })
                        .collect(),
                    result: self.apply_subst_to_type(
                        operation.result,
                        type_subst,
                        row_subst,
                        &var_to_index,
                    ),
                },
            );
        }
        let ability_conventions = self
            .env
            .export_ability_conventions()
            .into_iter()
            .collect::<HashMap<_, _>>();
        for (lambda_id, signature) in func_lambda_signatures {
            let function_type = self.apply_subst_to_type(
                signature.function_type,
                type_subst,
                row_subst,
                &var_to_index,
            );
            let convention = crate::ast::calling_convention_for_function_type(
                self.db(),
                function_type,
                &ability_conventions,
            )
            .expect("lambda semantic signature must remain a function type");
            self.lambda_signatures.insert(
                lambda_id,
                crate::typeck::LambdaSignature {
                    function_type,
                    convention,
                },
            );
        }
        FuncDecl {
            id: func.id,
            is_pub: func.is_pub,
            name: func.name,
            type_params: func.type_params,
            params: func.params,
            return_ty: func.return_ty,
            effects: func.effects,
            body,
        }
    }

    /// Resolve deferred method calls after constraint solving.
    ///
    /// Iteratively resolves methods whose receiver types are now concrete after solving.
    /// Each resolved method adds new type constraints (return type, param types),
    /// which are re-solved. Repeats until no more progress.
    fn resolve_deferred_methods(
        &self,
        solver: &mut TypeSolver<'db>,
        mut deferred: Vec<crate::typeck::func_context::DeferredMethodCall<'db>>,
        func_node_id: crate::ast::NodeId,
        instances: &mut HashMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
    ) -> HashMap<crate::ast::NodeId, (FuncDefId<'db>, Type<'db>)> {
        let mut resolved = HashMap::new();
        loop {
            let mut new_constraints = ConstraintSet::new();
            let mut remaining = Vec::new();

            for mc in std::mem::take(&mut deferred) {
                let resolved_receiver = solver.type_subst().apply(self.db(), mc.receiver_ty);
                if let Some(entry) = self.env.lookup_method(mc.method, resolved_receiver) {
                    // Method found — instantiate the TypeScheme to get fresh types
                    let func_ty = if let Some(scheme) = self.env.lookup_function(entry.func_id) {
                        let instance = crate::typeck::subst::instantiate_scheme_details_for_solver(
                            self.db(),
                            scheme,
                            solver,
                        );
                        let callable = instance.ty;
                        instances.insert(
                            mc.node_id,
                            crate::typeck::FunctionInstance {
                                origin: crate::typeck::FunctionInstanceOrigin::Declaration,
                                function: entry.func_id,
                                scheme,
                                callable,
                                type_arguments: instance.type_args,
                                row_arguments: instance.row_args,
                            },
                        );
                        callable
                    } else {
                        entry.func_ty
                    };

                    // Record the resolution for MethodCall → Call conversion
                    resolved.insert(mc.node_id, (entry.func_id, func_ty));

                    if let TypeKind::Func {
                        params,
                        result,
                        effect,
                        ..
                    } = func_ty.kind(self.db())
                    {
                        // Arity check
                        if mc.arg_types.len() != params.len() {
                            Diagnostic::new(
                                format!(
                                    "UFCS arity mismatch for '{}': expected {} args, got {}",
                                    mc.method,
                                    params.len(),
                                    mc.arg_types.len(),
                                ),
                                self.get_span(mc.node_id),
                                DiagnosticSeverity::Error,
                                CompilationPhase::TypeChecking,
                            )
                            .accumulate(self.db());
                        }

                        // Constrain result type
                        new_constraints.add_type_eq(mc.result_ty, *result);
                        solver.resolve_producer(mc.node_id);

                        // Constrain arg types against function params (min of both lengths)
                        for (arg, param) in mc.arg_types.iter().zip(params.iter()) {
                            new_constraints.add_type_coerce(
                                *arg,
                                *param,
                                super::super::constraint::ConstraintOrigin {
                                    node_id: mc.node_id,
                                    kind: super::super::constraint::ConstraintOriginKind::Call,
                                },
                            );
                        }

                        // Propagate effect row
                        let pure = crate::ast::EffectRow::pure(self.db());
                        new_constraints.add_row_eq(*effect, pure);
                    }
                } else {
                    remaining.push(mc);
                }
            }

            if new_constraints.is_empty() {
                // These calls may become resolvable after the surrounding
                // function's types have propagated through TDNR. Keep them in
                // the AST for that pass; any calls still unresolved afterward
                // are diagnosed at the frontend boundary.
                break;
            }
            if let Err(error) = solver.solve(new_constraints) {
                Diagnostic::new(
                    format!("type error during UFCS method resolution: {}", error),
                    self.get_span(func_node_id),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            if let Err(error) = solver.finalize_relations() {
                Diagnostic::new(
                    format!("type error during UFCS method resolution: {}", error.error),
                    self.get_span(error.origin.map_or(func_node_id, |origin| origin.node_id)),
                    DiagnosticSeverity::Error,
                    CompilationPhase::TypeChecking,
                )
                .accumulate(self.db());
            }
            deferred = remaining;
        }
        resolved
    }

    /// Get function signature from the registered scheme.
    ///
    /// Instantiates the scheme with fresh type variables for this function's inference.
    /// Returns (param_types, return_type, instantiated_func_type).
    fn get_func_signature_with_type(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        func_id: FuncDefId<'db>,
        func: &FuncDecl<ResolvedRef<'db>>,
    ) -> (
        Vec<Type<'db>>,
        Type<'db>,
        Type<'db>,
        Option<(TypeScheme<'db>, crate::typeck::subst::SchemeInstance<'db>)>,
    ) {
        if let Some(scheme) = self.env.lookup_function(func_id) {
            let instance = ctx.instantiate_scheme_details(scheme);
            let func_ty = instance.ty;
            if let TypeKind::Func { params, result, .. } = func_ty.kind(self.db()) {
                return (params.clone(), *result, func_ty, Some((scheme, instance)));
            }
        }

        // Fallback: create fresh type variables
        let param_types: Vec<Type<'db>> =
            func.params.iter().map(|_| ctx.fresh_type_var()).collect();
        let return_ty = ctx.fresh_type_var();
        let effect = ctx.fresh_effect_row();
        let func_ty = ctx.func_type(param_types.clone(), return_ty, effect);
        (param_types, return_ty, func_ty, None)
    }

    // =========================================================================
    // Body type transformation
    // =========================================================================
}

#[cfg(test)]
mod tests;
