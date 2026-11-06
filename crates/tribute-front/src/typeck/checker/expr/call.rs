//! Variable references, calls, and method selection.

use salsa::Accumulator;

use tribute_core::{CompilationPhase, DiagnosticSeverity};
use trunk_ir::SymbolPath;

use crate::ast::{
    AbilityInstance, AbilityRow, Candidates, Expr, ExprKind, ModulePath, NodeId, ResolvedRef, Type,
    TypeKind,
};
use crate::typeck::checker::{NodeDiagnostic, TypeChecker};
use crate::typeck::constraint::ConstraintOriginKind;
use crate::typeck::func_context::FunctionInferenceContext;
use crate::typeck::solver::TypeSolver;
use crate::typeck::subst;

/// The function a method call selects for its receiver's type.
pub(crate) enum MethodSelection<'db> {
    One(crate::typeck::MethodEntry<'db>),
    /// Several functions a path may name take the call, so it waits for
    /// more of its types.
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
    /// Whether the types are as determined as they will be. Until then, a
    /// function that fits is selected only when no function of a lower rank
    /// fits, since a type determined later may rule it out.
    pub settled: bool,
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

/// The functions name resolution found for a call's callee.
pub(crate) fn candidate_functions<'db>(
    candidates: &Candidates<ResolvedRef<'db>>,
) -> Candidates<crate::ast::FuncDefId<'db>> {
    candidates.filter_map(|candidate| match candidate {
        ResolvedRef::Function { id } => Some(*id),
        _ => None,
    })
}

impl<'db> TypeChecker<'db> {
    pub(super) fn infer_call_args_with_ctx(
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

    /// Infer the type of a variable reference.
    /// Report a module named where `expected` is required, once per node.
    pub(super) fn report_module_reference(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        path: ModulePath<'db>,
        expected: &str,
    ) -> Type<'db> {
        if ctx.mark_module_value_reported(node) {
            NodeDiagnostic::new(
                format!(
                    "expected {expected}, found module `{}`",
                    self.names().path(path.segments(self.db()))
                ),
                node,
                DiagnosticSeverity::Error,
                CompilationPhase::TypeChecking,
            )
            .accumulate(self.db());
        }
        ctx.error_type()
    }

    pub(super) fn infer_var_with_ctx(
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
            ResolvedRef::ImplicitConstructor { .. } => {
                let Some(node) = node else {
                    return ctx.fresh_type_var();
                };
                if let Some(deferred) = self.deferred_constructor(ctx, node) {
                    return deferred.ctor_ty;
                }
                match self.constructor_reference(ctx, node, resolved) {
                    ResolvedRef::ImplicitConstructor { .. } => ctx.fresh_type_var(),
                    decided => self.infer_var_with_ctx(ctx, Some(node), &decided),
                }
            }
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
                    // The ability row contains this ability (with a row variable tail for polymorphism)

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
                            .instances(self.db())
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
                                .unwrap_or_else(|subst::BoundVarOutOfBounds { index, max }| {
                                    panic!(
                                        "AbilityOp param BoundVar index out of range: index={}, subst.len()={}",
                                        index, max
                                    )
                                })
                        })
                        .collect();
                    let return_type =
                        subst::substitute_bound_vars(self.db(), op_info.return_type, &ability_args)
                            .unwrap_or_else(|subst::BoundVarOutOfBounds { index, max }| {
                                panic!(
                                    "AbilityOp return BoundVar index out of range: index={}, subst.len()={}",
                                    index, max
                                )
                            });

                    let ability_effect = AbilityInstance {
                        ability_id: *ability,
                        args: ability_args,
                    };
                    let effect = AbilityRow::new(self.db(), vec![ability_effect], None);
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

    pub(super) fn infer_local_reference_with_ctx(
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

    /// Infer a call whose callee names several functions, written
    /// `receiver.method(args)` or `method(receiver, args)`.
    ///
    /// The first argument alone is tried first, so that the other arguments are
    /// inferred against the selected function's parameters. A call it does
    /// not decide infers them on their own and tries again with all of them;
    /// one still undecided is deferred.
    pub(super) fn infer_method_call_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: &Expr<ResolvedRef<'db>>,
    ) -> Type<'db> {
        let ExprKind::Call {
            callee,
            args,
            syntax,
        } = &*expr.kind
        else {
            unreachable!("the caller matched a call");
        };
        ctx.note_call_syntax(expr.id, *syntax);
        let ExprKind::Functions {
            name: method,
            candidates,
        } = &*callee.kind
        else {
            unreachable!("the caller matched a callee naming functions");
        };
        let node = expr.id;
        let candidates = candidate_functions(candidates);
        let arity = args.len();
        let first_ty = args
            .first()
            .map(|first| self.infer_expr_type_with_ctx(ctx, first));
        let selection = self.select_method(
            &candidates,
            CallTypes {
                args: first_ty.as_slice(),
                arity,
                result: None,
                settled: false,
            },
        );
        if let MethodSelection::One(entry) = selection {
            let callee_ty = self.select_callee(ctx, node, callee.id, entry.func_id);
            let mut arg_types: Vec<Type<'db>> = first_ty.into_iter().collect();
            let params = match callee_ty.kind(self.db()) {
                TypeKind::Func { params, .. } => params.as_slice(),
                _ => &[],
            };
            for (index, arg) in args.iter().enumerate().skip(1) {
                arg_types.push(match params.get(index) {
                    Some(expected) => self.infer_expr_with_expected(ctx, arg, *expected),
                    None => self.infer_expr_type_with_ctx(ctx, arg),
                });
            }
            return self.infer_call_with_ctx(ctx, callee_ty, &arg_types, node);
        }

        let arg_types: Vec<Type<'db>> = first_ty
            .into_iter()
            .chain(
                args.iter()
                    .skip(1)
                    .map(|a| self.infer_deferred_argument(ctx, a)),
            )
            .collect();
        let selection = self.select_method(
            &candidates,
            CallTypes {
                args: &arg_types,
                arity,
                result: None,
                settled: false,
            },
        );
        if let MethodSelection::One(entry) = selection {
            let callee_ty = self.select_callee(ctx, node, callee.id, entry.func_id);
            return self.infer_call_with_ctx(ctx, callee_ty, &arg_types, node);
        }

        // For known operator methods, add type constraints eagerly so that
        // type inference can propagate before the method is selected. The
        // result is a variable of its own, not an operand's type: an operand
        // that is a `case` result would otherwise be this call's result, and
        // a common result waits while a deferred call produces it.
        let result_ty = match (method.to_string().as_str(), &arg_types[..]) {
            ("+" | "-" | "*" | "/" | "%", [lhs, rhs]) => {
                ctx.constrain_eq(*lhs, *rhs);
                ctx.fresh_type_var()
            }
            ("==" | "!=" | "<" | "<=" | ">" | ">=", [lhs, rhs]) => {
                ctx.constrain_eq(*lhs, *rhs);
                ctx.bool_type()
            }
            _ => ctx.fresh_type_var(),
        };
        let effect = ctx
            .deferred_method_effect(node)
            .unwrap_or_else(|| ctx.fresh_ability_row());
        ctx.evidence.record_call(node, effect);
        ctx.merge_effect_at(effect, node);
        ctx.record_deferred_method(crate::typeck::func_context::DeferredMethodCall {
            node_id: node,
            value: false,
            syntax: *syntax,
            method: method.clone(),
            callee: callee.id,
            candidates,
            result_ty,
            arg_types,
            effect,
        });
        result_ty
    }

    /// Infer a name that stands for several functions and is used as a
    /// value. The type it is used at selects the function, so its type is
    /// left open and the selection deferred.
    pub(super) fn infer_function_value_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        name: &SymbolPath,
        candidates: &Candidates<ResolvedRef<'db>>,
    ) -> Type<'db> {
        let result_ty = ctx.fresh_type_var();
        let effect = ctx
            .deferred_method_effect(node)
            .unwrap_or_else(|| ctx.fresh_ability_row());
        ctx.record_deferred_method(crate::typeck::func_context::DeferredMethodCall {
            node_id: node,
            value: true,
            syntax: crate::ast::CallSyntax::Call,
            method: name.clone(),
            callee: node,
            candidates: candidate_functions(candidates),
            result_ty,
            arg_types: Vec::new(),
            effect,
        });
        result_ty
    }

    /// The instance of `function` that the call at `node` selected for its
    /// callee at `callee`.
    fn select_callee(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        callee: NodeId,
        function: crate::ast::FuncDefId<'db>,
    ) -> Type<'db> {
        let callee_ty = ctx
            .instantiate_function_reference(callee, function)
            .unwrap_or_else(|| ctx.fresh_type_var());
        ctx.record_resolved_method(node, function, callee_ty);
        callee_ty
    }

    /// Infer the result type of a function call.
    ///
    /// This method:
    /// 1. Creates fresh type variables for parameter and result types
    /// 2. Creates a fresh ability row for the callee's effect
    /// 3. Constrains the callee to have the expected function type (or continuation type)
    /// 4. Constrains the callee's effect to be a subset of the current function's effect
    ///
    /// Effect propagation ensures that if we call `State::get()`, the `State`
    /// effect constraint is added to the current function's ability row.
    pub(super) fn infer_call_with_ctx(
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
                self.report_call_arity(
                    origin,
                    ctx.call_syntax(origin),
                    params.len(),
                    arg_types.len(),
                );
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
            let cont_effect = ctx.fresh_ability_row();
            let expected_cont_ty = Type::new(
                self.db(),
                TypeKind::Continuation {
                    arg: cont_arg_ty,
                    result: cont_result_ty,
                    effect: cont_effect,
                },
            );

            // Also create function type for the normal case
            let callee_effect = ctx.fresh_ability_row();
            let expected_func_ty = ctx.func_type(param_types.clone(), result_ty, callee_effect);

            // Try to match the callee type: if it's already a concrete Continuation type,
            // use continuation semantics; otherwise use function semantics.
            // We check the callee type directly rather than relying on unification to choose.
            if matches!(callee_ty.kind(self.db()), TypeKind::Continuation { .. }) {
                // Callee is a continuation - constrain as continuation call.
                // Note: We intentionally do NOT merge cont_effect into the current effect
                // context here, unlike the function call path below. The continuation's
                // ability row is already the handler body's ability row (assigned in
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
        let callee_effect = ctx.fresh_ability_row();
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

    /// Resolve the ability row to merge for a function call.
    ///
    /// If the callee type is already a concrete function type, uses that function's
    /// ability row directly. Otherwise, falls back to the fresh ability row.
    fn resolve_callee_effect(
        &self,
        callee_ty: Type<'db>,
        fallback: AbilityRow<'db>,
    ) -> AbilityRow<'db> {
        if let TypeKind::Func { effect, .. } = callee_ty.kind(self.db()) {
            *effect
        } else {
            fallback
        }
    }

    /// Report a call that passes `args` arguments to a function of `params`
    /// parameters. A method call counts the arguments after its receiver,
    /// as it is written.
    pub(crate) fn report_call_arity(
        &self,
        node: NodeId,
        syntax: crate::ast::CallSyntax,
        params: usize,
        args: usize,
    ) {
        let (params, args, receiver) = match syntax {
            crate::ast::CallSyntax::Method if params > 0 && args > 0 => {
                (params - 1, args - 1, " after the receiver")
            }
            _ => (params, args, ""),
        };
        NodeDiagnostic::new(
            format!(
                "call arity mismatch: expected {params} argument{}{receiver}, found {args}",
                if params == 1 { "" } else { "s" }
            ),
            node,
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        )
        .accumulate(self.db());
    }

    /// Select the function a call names among the functions name resolution
    /// found for it in the call's scope, by rank: the one whose signature
    /// takes the call.
    pub(crate) fn select_method(
        &self,
        candidates: &Candidates<crate::ast::FuncDefId<'db>>,
        call: CallTypes<'_, 'db>,
    ) -> MethodSelection<'db> {
        match self.select_function(candidates, call) {
            // A call that passes the wrong number of arguments still names
            // the function its arguments select, which reports the number.
            (MethodSelection::None, Some(by_arguments)) => MethodSelection::One(by_arguments),
            (selection, _) => selection,
        }
    }

    /// The function whose whole signature takes `call` in the highest rank
    /// of `candidates` that has one, and, where no rank does, the one function of
    /// any rank whose parameters take the arguments the call has, whatever
    /// their number and its result. Before `call` is settled, a function
    /// that lower ranks compete with is not selected yet.
    pub(crate) fn select_function(
        &self,
        candidates: &Candidates<crate::ast::FuncDefId<'db>>,
        call: CallTypes<'_, 'db>,
    ) -> (
        MethodSelection<'db>,
        Option<crate::typeck::MethodEntry<'db>>,
    ) {
        let fits = |params: &[Type<'db>], result| call.matches(self.db(), params, result);
        let ranks = candidates.ranks();
        for (index, rank) in ranks.iter().enumerate() {
            match self.select_matching(rank, fits) {
                MethodSelection::None => {}
                MethodSelection::One(entry) if !call.settled => {
                    let lower: Vec<_> = ranks[index + 1..]
                        .iter()
                        .copied()
                        .flatten()
                        .copied()
                        .filter(|function| {
                            self.method_entry(*function).is_some_and(|entry| {
                                matches!(
                                    entry.func_ty.kind(self.db()),
                                    TypeKind::Func { params, result, .. } if fits(params, *result)
                                )
                            })
                        })
                        .collect();
                    if lower.is_empty() {
                        return (MethodSelection::One(entry), None);
                    }
                    return (
                        MethodSelection::Ambiguous(
                            std::iter::once(entry.func_id).chain(lower).collect(),
                        ),
                        None,
                    );
                }
                selection => return (selection, None),
            }
        }
        let all: Vec<_> = candidates.iter().copied().collect();
        let by_arguments =
            match self.select_matching(&all, |params, _| call.arguments_match(self.db(), params)) {
                MethodSelection::One(entry) => Some(entry),
                _ => None,
            };
        (MethodSelection::None, by_arguments)
    }

    /// The functions among `candidates` whose signatures `matches`.
    fn select_matching(
        &self,
        candidates: &[crate::ast::FuncDefId<'db>],
        matches: impl Fn(&[Type<'db>], Type<'db>) -> bool,
    ) -> MethodSelection<'db> {
        let mut matching = candidates.iter().filter_map(|candidate| {
            let entry = self.method_entry(*candidate)?;
            match entry.func_ty.kind(self.db()) {
                TypeKind::Func { params, result, .. } if matches(params, *result) => Some(entry),
                _ => None,
            }
        });
        match (matching.next(), matching.next()) {
            (Some(entry), None) => MethodSelection::One(entry),
            (Some(first), Some(second)) => MethodSelection::Ambiguous(
                [first, second]
                    .into_iter()
                    .chain(matching)
                    .map(|entry| entry.func_id)
                    .collect(),
            ),
            (None, _) => MethodSelection::None,
        }
    }

    /// `function` with the body of its type scheme.
    pub(crate) fn method_entry(
        &self,
        function: crate::ast::FuncDefId<'db>,
    ) -> Option<crate::typeck::MethodEntry<'db>> {
        let (scheme, _) = self.env.function_scheme(function)?;
        Some(crate::typeck::MethodEntry {
            func_id: function,
            func_ty: *scheme.body(self.db()),
        })
    }

    /// Resolve the deferred method calls whose receiver `solver` has typed,
    /// as inference does for a receiver typed where the call is written.
    pub(super) fn resolve_typed_deferred_methods(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        solver: &TypeSolver<'db>,
    ) -> bool {
        let mut resolved = false;
        for call in ctx.deferred_methods().to_vec() {
            let MethodSelection::One(entry) = self.select_deferred_method(solver, &call, false)
            else {
                continue;
            };
            let Some(callee_ty) = ctx.instantiate_function_reference(call.callee, entry.func_id)
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
}
