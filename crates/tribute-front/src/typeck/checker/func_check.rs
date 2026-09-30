//! Function-level type checking.
//!
//! Each function is type-checked with an isolated `FunctionInferenceContext`,
//! ensuring that type variables (UniVars) are fully resolved within the function
//! before moving to the next.

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;

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
        func: &FuncDecl<ResolvedRef<'db>>,
    ) -> FuncDecl<TypedRef<'db>> {
        // 1. Create a fresh FunctionInferenceContext for this function
        // Use function definition ID for globally unique UniVar IDs
        let func_id = self.func_def_id(func.name);
        let mut ctx = FunctionInferenceContext::new(self.db(), &self.env, func_id);
        // Only the exact root `main` is an entrypoint.
        let is_root_main = crate::is_root_main(func.name, self.current_prefix().is_empty());

        // 2. Get the function's registered type scheme and instantiate it

        // Get the instantiated function type (with UniVars) for later generalization
        let (param_types, expected_return, instantiated_func_ty, signature_instance) =
            self.get_func_signature_with_type(&mut ctx, func_id, func);
        if let Some((_, instance)) = &signature_instance
            && let Some(names) = self.signature_type_names.get(&func_id)
        {
            for (name, index) in names {
                ctx.bind_annotation_type_parameter(*name, instance.type_args[*index as usize]);
            }
        }
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

        // Set effect row from the function's declared type before checking
        // body. An omitted annotation is exactly a fresh `->{e}`; the body's
        // effects never widen the declared row.
        let declared_effect =
            if let TypeKind::Func { effect, .. } = instantiated_func_ty.kind(self.db()) {
                ctx.set_current_effect(*effect);
                Some(*effect)
            } else {
                None
            };

        ctx.effect_contract = declared_effect;
        // 3. Check body against expected return type
        let body = self.check_expr_with_ctx(&mut ctx, &func.body, Mode::Check(expected_return));
        let mut reported_undeclared = false;

        if let Some(declared) = declared_effect {
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
            // An open declared row would absorb any concrete effect through
            // its tail, so name the undeclared ones instead of a mismatch.
            if declared.rest(self.db()).is_some()
                && let Some(undeclared) = self.undeclared_effects(declared, ctx.current_effect())
            {
                self.report_undeclared_effects(func, is_root_main, &undeclared);
                reported_undeclared = true;
            } else {
                ctx.constrain_row_eq_at(declared, ctx.current_effect(), node_id, kind);
            }
        }
        // 4. Solve constraints for this function only
        let constraints = ctx.take_constraints();
        // Take node_types now while ctx is still alive, before we need mutable self access
        let func_node_types = ctx.take_node_types();
        let mut func_instances = ctx.take_function_instances();
        let local_instances = ctx.take_local_instances();
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

        // The declared signature is the function's type; the body is checked
        // against it and never refines it.
        let inferred_func_ty = instantiated_func_ty;

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
                let owner = *self.local_generalizations.entry(*id).or_insert(binding);
                debug_assert_eq!(
                    owner, binding,
                    "solver representative {id:?} aliases local quantifiers of two owners"
                );
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
        let retained_unions = solver.row_unions_for_type(inferred_func_ty);
        let retained_removals = solver.row_removals_for_type(inferred_func_ty);
        let (union_univars, union_effect_vars) = solver.row_union_variables(&retained_unions);
        let (removal_univars, removal_effect_vars) =
            solver.row_removal_variables(&retained_removals);
        for var in union_univars.into_iter().chain(removal_univars) {
            if !all_univars.contains(&var) {
                all_univars.push(var);
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
            && !matches!(result.kind(self.db()), TypeKind::Nil | TypeKind::Error)
        {
            Diagnostic::new(
                format!("function 'main' must return Nil, but returns `{}`", result),
                self.get_span(func.id),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }

        if let Some(declared) = declared_effect {
            if func.effects.is_some() {
                self.report_duplicate_effect(func, func_id, row_subst.apply(self.db(), declared));
            }
            if !solve_failed
                && !reported_undeclared
                && let Some(undeclared) =
                    self.undeclared_effects(declared, row_subst.apply(self.db(), body_effect_row))
            {
                // The body may perform only the concrete effects the signature
                // declares; the declared tail stands for the caller's effects.
                self.report_undeclared_effects(func, is_root_main, &undeclared);
            }

            // Root `main` may leave only the ambient `Io` unhandled.
            if is_root_main {
                let unhandled = declared
                    .effects(self.db())
                    .iter()
                    .filter(|effect| !effect.ability_id.is_builtin_io(self.db()))
                    .collect_vec();
                if !unhandled.is_empty() {
                    Diagnostic::new(
                        format!(
                            "function 'main' has unhandled effects: {}",
                            unhandled.iter().format(", ")
                        ),
                        self.get_span(func.id),
                        DiagnosticSeverity::Error,
                        CompilationPhase::TypeChecking,
                    )
                    .accumulate(self.db());
                }
            }
        }

        if !solve_failed && let Some((scheme, instance)) = &signature_instance {
            self.report_signature_rigidity(func, func_id, *scheme, instance, type_subst, row_subst);
            self.report_undeclared_row_unions(
                func,
                func_id,
                *scheme,
                instance,
                &retained_unions,
                row_subst,
            );
        }

        let generalized =
            type_subst.apply_generalization(self.db(), substituted_ty, row_subst, &var_to_index);

        // Create type params for all generalized UniVars
        let type_params: Vec<crate::ast::TypeParam> = (0..var_to_index.len())
            .map(|_| crate::ast::TypeParam::anonymous())
            .collect();

        let mut effect_params = collect_effect_vars(self.db(), generalized);
        for var in union_effect_vars.into_iter().chain(removal_effect_vars) {
            if !effect_params.contains(&var) {
                effect_params.push(var);
            }
        }
        let unions = retained_unions
            .iter()
            .map(|union| solver.generalize_row_union(union, &var_to_index))
            .collect();
        let new_scheme = TypeScheme::builder(type_params, effect_params, generalized)
            .row_unions(unions)
            .row_removals(
                retained_removals
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
        let mut body = body;
        self.apply_subst_to_body(
            &mut body,
            type_subst,
            row_subst,
            &var_to_index,
            &deferred_resolutions,
        );

        for (node, mut instance) in local_instances {
            instance.callable =
                self.apply_subst_to_type(instance.callable, type_subst, row_subst, &var_to_index);
            let map_type = |ty| self.apply_subst_to_type(ty, type_subst, row_subst, &var_to_index);
            instance.scheme = instance
                .scheme
                .to_builder(self.db())
                .map_types(self.db(), map_type)
                .build(self.db());
            instance.row_arguments = instance
                .row_arguments
                .into_iter()
                .map(|row| {
                    crate::typeck::solver::map_effect_row_type_args(
                        self.db(),
                        row_subst.apply(self.db(), row),
                        map_type,
                    )
                })
                .collect();
            self.local_instances.insert(node, instance);
        }
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
            type_params: func.type_params.clone(),
            params: func.params.clone(),
            return_ty: func.return_ty.clone(),
            effects: func.effects.clone(),
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

    /// Concrete effects of `body` that `declared` does not name.
    fn undeclared_effects(
        &self,
        declared: crate::ast::EffectRow<'db>,
        body: crate::ast::EffectRow<'db>,
    ) -> Option<Vec<crate::ast::Effect<'db>>> {
        let declared_ids: HashSet<_> = declared
            .effects(self.db())
            .iter()
            .map(|effect| effect.ability_id)
            .collect();
        let undeclared: Vec<_> = body
            .effects(self.db())
            .iter()
            .filter(|effect| !declared_ids.contains(&effect.ability_id))
            .cloned()
            .collect();
        (!undeclared.is_empty()).then_some(undeclared)
    }

    /// Report effects the body performs outside its declared row. Nothing
    /// handles an effect that escapes root `main`, so there any effect but
    /// the ambient `Io` is unhandled rather than merely undeclared.
    fn report_undeclared_effects(
        &self,
        func: &FuncDecl<ResolvedRef<'db>>,
        is_root_main: bool,
        undeclared: &[crate::ast::Effect<'db>],
    ) {
        let (unhandled, undeclared): (Vec<_>, Vec<_>) = undeclared
            .iter()
            .partition(|effect| is_root_main && !effect.ability_id.is_builtin_io(self.db()));
        for (message, effects) in [
            ("has unhandled effects", unhandled),
            ("uses undeclared effects", undeclared),
        ] {
            if effects.is_empty() {
                continue;
            }
            Diagnostic::new(
                format!(
                    "function '{}' {message}: {}",
                    func.name,
                    effects.iter().format(", "),
                ),
                self.get_span(func.id),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }
    }

    /// Report signature rows whose effects the body propagates into the
    /// function's own effect row without the signature saying so.
    ///
    /// Calling a callback whose row is not the function's own joins that row
    /// into the function's effects through a retained union. The signature
    /// is final, so a row may flow into the function's effects only when it
    /// is that row's tail or a declared union reaches it.
    fn report_undeclared_row_unions(
        &self,
        func: &FuncDecl<ResolvedRef<'db>>,
        func_id: FuncDefId<'db>,
        scheme: TypeScheme<'db>,
        instance: &crate::typeck::subst::SchemeInstance<'db>,
        retained: &[crate::ast::RowUnion<'db>],
        row_subst: &crate::typeck::solver::RowSubst<'db>,
    ) {
        let db = self.db();
        let tail = |row: &crate::ast::EffectRow<'db>| row_subst.apply(db, *row).rest(db);
        let Some(own) = (match instance.ty.kind(db) {
            TypeKind::Func { effect, .. } => tail(effect),
            _ => None,
        }) else {
            return;
        };

        // Rows the signature itself puts into the function's effects.
        let mut declared = HashSet::from([own]);
        loop {
            let before = declared.len();
            for union in &instance.row_unions {
                if tail(&union.result).is_some_and(|result| declared.contains(&result)) {
                    declared.extend(union.sources.iter().filter_map(tail));
                }
            }
            if declared.len() == before {
                break;
            }
        }

        // Rows each row flows into through the body's retained unions.
        let mut flows: HashMap<crate::ast::EffectVar, Vec<crate::ast::EffectVar>> = HashMap::new();
        for union in retained {
            if let Some(result) = tail(&union.result) {
                for source in union.sources.iter().filter_map(tail) {
                    flows.entry(source).or_default().push(result);
                }
            }
        }
        let reaches_own = |start: crate::ast::EffectVar| {
            let mut seen = HashSet::from([start]);
            let mut pending = vec![start];
            while let Some(row) = pending.pop() {
                if row == own {
                    return true;
                }
                for next in flows.get(&row).into_iter().flatten() {
                    if seen.insert(*next) {
                        pending.push(*next);
                    }
                }
            }
            false
        };

        let mut reported = HashSet::new();
        for (index, row) in instance.row_args.iter().enumerate() {
            let Some(row) = tail(row) else { continue };
            if declared.contains(&row) || !reaches_own(row) || !reported.insert(row) {
                continue;
            }
            Diagnostic::new(
                format!(
                    "function '{}' performs the effects of {} without declaring them; \
                     use the same effect variable in its own effect row",
                    func.name,
                    self.signature_row_name(func_id, scheme, index),
                ),
                self.get_span(func.id),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(db);
        }
    }

    /// How diagnostics name the signature's `index`-th effect parameter.
    fn signature_row_name(
        &self,
        func_id: FuncDefId<'db>,
        scheme: TypeScheme<'db>,
        index: usize,
    ) -> String {
        let var = scheme.effect_params(self.db()).get(index).copied();
        self.signature_row_names
            .get(&func_id)
            .into_iter()
            .flatten()
            .find(|(_, candidate)| Some(**candidate) == var)
            .map_or_else(
                || "an omitted effect row".to_owned(),
                |(name, _)| format!("effect variable `{name}`"),
            )
    }

    /// Report a duplicate effect in the function's annotation.
    fn report_duplicate_effect(
        &self,
        func: &FuncDecl<ResolvedRef<'db>>,
        func_id: FuncDefId<'db>,
        resolved_declared: crate::ast::EffectRow<'db>,
    ) {
        let Some(duplicate) = self
            .effect_annotation_origins
            .get(&func_id)
            .and_then(|origins| origins.find_duplicate(self.db(), resolved_declared))
        else {
            return;
        };
        Diagnostic::builder(
            format!(
                "function '{}' declares duplicate effect: {}",
                func.name,
                duplicate.effects.iter().format(", "),
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

    /// Report signature variables that the body made more specific.
    ///
    /// The declared signature is the function's final type, so its type and
    /// row variables are rigid in the body: each must stay an unsolved
    /// variable, distinct from the others. The function's own effect tail is
    /// covered by the undeclared-effect check.
    fn report_signature_rigidity(
        &self,
        func: &FuncDecl<ResolvedRef<'db>>,
        func_id: FuncDefId<'db>,
        scheme: TypeScheme<'db>,
        instance: &crate::typeck::subst::SchemeInstance<'db>,
        type_subst: &crate::typeck::solver::TypeSubst<'db>,
        row_subst: &crate::typeck::solver::RowSubst<'db>,
    ) {
        let db = self.db();
        let report = |message: String| {
            Diagnostic::new(
                message,
                self.get_span(func.id),
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(db);
        };

        let type_names: HashMap<u32, Symbol> = self
            .signature_type_names
            .get(&func_id)
            .into_iter()
            .flatten()
            .map(|(name, index)| (*index, *name))
            .collect();
        let type_name = |index: usize| {
            type_names
                .get(&(index as u32))
                .map_or_else(|| format!("#{index}"), |name| name.to_string())
        };
        let mut seen_types: HashMap<UniVarId<'db>, usize> = HashMap::new();
        for (index, ty) in instance.type_args.iter().enumerate() {
            let resolved = type_subst.apply_with_rows(db, *ty, row_subst);
            match resolved.kind(db) {
                TypeKind::UniVar { id } => {
                    if let Some(first) = seen_types.insert(*id, index) {
                        report(format!(
                            "type variables `{}` and `{}` in the signature of `{}` are the same type in its body",
                            type_name(first),
                            type_name(index),
                            func.name,
                        ));
                    }
                }
                TypeKind::Error => {}
                _ => report(format!(
                    "type variable `{}` in the signature of `{}` is `{}` in its body",
                    type_name(index),
                    func.name,
                    resolved,
                )),
            }
        }

        let row_names: HashMap<crate::ast::EffectVar, Symbol> = self
            .signature_row_names
            .get(&func_id)
            .into_iter()
            .flatten()
            .map(|(name, var)| (*var, *name))
            .collect();
        let effect_params = scheme.effect_params(db);
        let row_name = |index: usize| {
            effect_params
                .get(index)
                .and_then(|var| row_names.get(var))
                .map_or_else(
                    || "an omitted effect row".to_owned(),
                    |name| format!("effect variable `{name}`"),
                )
        };
        let own_tail = match instance.ty.kind(db) {
            TypeKind::Func { effect, .. } => effect.rest(db),
            _ => None,
        };
        let mut seen_rows: HashMap<crate::ast::EffectVar, usize> = HashMap::new();
        for (index, row) in instance.row_args.iter().enumerate() {
            let resolved = row_subst.apply(db, *row);
            let Some(tail) = resolved.rest(db) else {
                report(format!(
                    "{} in the signature of `{}` is closed in its body",
                    row_name(index),
                    func.name,
                ));
                continue;
            };
            if let Some(first) = seen_rows.insert(tail, index) {
                report(format!(
                    "{} and {} in the signature of `{}` are the same row in its body",
                    row_name(first),
                    row_name(index),
                    func.name,
                ));
            }
            if row.rest(db) != own_tail && !resolved.effects(db).is_empty() {
                report(format!(
                    "{} in the signature of `{}` has {} in its body",
                    row_name(index),
                    func.name,
                    resolved.effects(db).iter().format(", "),
                ));
            }
        }
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
