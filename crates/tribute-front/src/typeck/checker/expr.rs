//! Expression type checking.
//!
//! All expression checking methods take a `FunctionInferenceContext` as parameter,
//! enabling per-function type inference with isolated constraints.

use std::collections::HashSet;

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, Arm, BinOpKind, BuiltinRef, Effect, EffectRow, Expr, ExprKind, FieldPattern,
    HandlerArm, HandlerKind, LiteralPattern, Pattern, PatternKind, ResolvedRef, Stmt, Type,
    TypeKind, TypedRef,
};

use super::super::func_context::FunctionInferenceContext;
use super::super::subst;
use super::{Mode, TypeChecker};

impl<'db> TypeChecker<'db> {
    // =========================================================================
    // Expression checking
    // =========================================================================

    /// Type check an expression with a FunctionInferenceContext.
    pub(crate) fn check_expr_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        expr: Expr<ResolvedRef<'db>>,
        mode: Mode<'db>,
    ) -> Expr<TypedRef<'db>> {
        let ty = match &*expr.kind {
            ExprKind::NatLit(_) => ctx.nat_type(),
            ExprKind::IntLit(_) => ctx.int_type(),
            ExprKind::FloatLit(_) => ctx.float_type(),
            ExprKind::BoolLit(_) => ctx.bool_type(),
            ExprKind::StringLit(_) => ctx.string_type(),
            ExprKind::BytesLit(_) => ctx.bytes_type(),
            ExprKind::Nil => ctx.nil_type(),
            ExprKind::RuneLit(_) => ctx.rune_type(),

            ExprKind::Var(resolved) => self.infer_var_with_ctx(ctx, resolved),
            ExprKind::Call { callee, args } => {
                let callee_ty = self.infer_expr_type_with_ctx(ctx, callee);
                let arg_types: Vec<Type<'db>> = args
                    .iter()
                    .map(|a| self.infer_expr_type_with_ctx(ctx, a))
                    .collect();
                self.infer_call_with_ctx(ctx, callee_ty, &arg_types)
            }
            ExprKind::Cons { ctor, args } => {
                let ctor_ty = self.infer_var_with_ctx(ctx, ctor);
                if args.is_empty() {
                    // Unit constructor (e.g., None) - just return the constructor type
                    ctor_ty
                } else {
                    // Constructor with arguments (e.g., Some(x)) - treat as function call
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.infer_expr_type_with_ctx(ctx, a))
                        .collect();
                    self.infer_call_with_ctx(ctx, ctor_ty, &arg_types)
                }
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                // Get the struct constructor type and extract return type
                let ctor_ty = self.infer_var_with_ctx(ctx, type_name);
                let struct_ty = if let TypeKind::Func { result, .. } = ctor_ty.kind(self.db()) {
                    // Constructor has function type: fn(fields...) -> StructType
                    *result
                } else {
                    // Fallback: use constructor type directly (shouldn't happen)
                    ctor_ty
                };

                // Validate each field expression against the declared field type
                for (field_name, field_expr) in fields {
                    let expr_ty = self.infer_expr_type_with_ctx(ctx, field_expr);
                    if let Some(expected_field_ty) =
                        self.lookup_struct_field_type(ctx, struct_ty, *field_name)
                    {
                        ctx.constrain_eq(expr_ty, expected_field_ty);
                    }
                    // If field not found, we'll let later phases handle the error
                }

                // Validate spread expression if present
                if let Some(spread_expr) = spread {
                    let spread_ty = self.infer_expr_type_with_ctx(ctx, spread_expr);
                    ctx.constrain_eq(spread_ty, struct_ty);
                }

                struct_ty
            }
            ExprKind::MethodCall {
                receiver, method, ..
            } => {
                // Infer receiver type first
                let receiver_ty = self.infer_expr_type_with_ctx(ctx, receiver);

                // Try to look up the method as a struct field accessor
                if let Some(result_ty) = self.lookup_struct_field_type(ctx, receiver_ty, *method) {
                    result_ty
                } else {
                    // Method not found as field - leave as fresh type var for TDNR
                    ctx.fresh_type_var()
                }
            }
            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs_ty = self.infer_expr_type_with_ctx(ctx, lhs);
                let rhs_ty = self.infer_expr_type_with_ctx(ctx, rhs);
                match op {
                    // Arithmetic operators: operands must be same type, result is that type
                    BinOpKind::Add
                    | BinOpKind::Sub
                    | BinOpKind::Mul
                    | BinOpKind::Div
                    | BinOpKind::Mod => {
                        ctx.constrain_eq(lhs_ty, rhs_ty);
                        lhs_ty
                    }
                    // Comparison operators: operands must be same type, result is Bool
                    BinOpKind::Eq
                    | BinOpKind::Ne
                    | BinOpKind::Lt
                    | BinOpKind::Le
                    | BinOpKind::Gt
                    | BinOpKind::Ge => {
                        ctx.constrain_eq(lhs_ty, rhs_ty);
                        ctx.bool_type()
                    }
                    // Boolean operators: operands must be Bool, result is Bool
                    BinOpKind::And | BinOpKind::Or => {
                        let bool_ty = ctx.bool_type();
                        ctx.constrain_eq(lhs_ty, bool_ty);
                        ctx.constrain_eq(rhs_ty, bool_ty);
                        bool_ty
                    }
                    // String concatenation: operands must be Text, result is Text
                    BinOpKind::Concat => {
                        ctx.constrain_eq(lhs_ty, rhs_ty);
                        lhs_ty
                    }
                }
            }
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
                // Infer scrutinee type
                let scrutinee_ty = self.infer_expr_type_with_ctx(ctx, scrutinee);

                // Create result type and constrain all arms to it
                let result_ty = ctx.fresh_type_var();
                for arm in arms {
                    ctx.push_scope();
                    let pattern_ty = self.infer_pattern_type_with_ctx(ctx, &arm.pattern);
                    ctx.constrain_eq(pattern_ty, scrutinee_ty);
                    self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                    let arm_ty = self.infer_expr_type_with_ctx(ctx, &arm.body);
                    ctx.constrain_eq(arm_ty, result_ty);
                    ctx.pop_scope();
                }

                result_ty
            }
            ExprKind::Lambda { params, body } => {
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| match &p.ty {
                        Some(ann) => self.annotation_to_type_with_ctx(ctx, ann),
                        None => ctx.fresh_type_var(),
                    })
                    .collect();

                // Save the outer context's effect - lambda has its own effect context
                let outer_effect = ctx.current_effect();

                // Lambda body starts with pure effect (no effects yet)
                ctx.set_current_effect(EffectRow::pure(self.db()));

                // Use a new scope for lambda parameters so they don't leak out
                ctx.push_scope();

                // Bind lambda parameters in the new scope
                for (param, ty) in params.iter().zip(param_types.iter()) {
                    if let Some(local_id) = param.local_id {
                        ctx.bind_local(local_id, *ty);
                    }
                    ctx.bind_local_by_name(param.name, *ty);
                }

                let body_ty = self.infer_expr_type_with_ctx(ctx, body);

                // The lambda's effect is what accumulated during body inference
                let lambda_effect = ctx.current_effect();

                ctx.pop_scope();

                // Restore the outer context's effect
                ctx.set_current_effect(outer_effect);

                let result_ty = ctx.fresh_type_var();
                ctx.constrain_eq(result_ty, body_ty);
                ctx.func_type(param_types, result_ty, lambda_effect)
            }
            ExprKind::Handle { body, handlers } => {
                // Handle expression:
                // 1. Infer the body's type (body may have effects)
                // 2. Extract handled effects from handler arms
                // 3. Remove handled effects from the current effect row
                // 4. Return the body's type

                // Save the current effect state before checking body
                let effect_before_body = ctx.current_effect();

                // Create a fresh effect row for the body
                let body_effect = ctx.fresh_effect_row();
                ctx.set_current_effect(body_effect);

                // Infer the body's type
                let body_ty = self.infer_expr_type_with_ctx(ctx, body);

                // Extract handled ability IDs from effect handlers
                let handled_ability_ids: Vec<AbilityId<'db>> = handlers
                    .iter()
                    .filter_map(|h| match &h.kind {
                        HandlerKind::Effect { ability, .. } => {
                            // Get the ability ID from the ResolvedRef
                            self.extract_ability_id_from_ref(ability)
                        }
                        HandlerKind::Result { .. } => None,
                    })
                    .collect();

                // Get the body's effect after checking (may have effects added)
                let body_effect_after = ctx.current_effect();

                // Create a result effect row that excludes handled effects
                // This is the effect that propagates out of the handle expression
                let result_effect = if handled_ability_ids.is_empty() {
                    // No effect handlers, body's effect propagates as-is
                    body_effect_after
                } else {
                    // Remove handled effects from the body's effect row
                    self.remove_handled_effects(ctx, body_effect_after, &handled_ability_ids)
                };

                // Merge the result effect with the effect before body
                // (The outer function's effect now includes the unhandled effects)
                ctx.set_current_effect(effect_before_body);
                ctx.merge_effect(result_effect);

                body_ty
            }
            ExprKind::Tuple(elems) => {
                let elem_tys: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.infer_expr_type_with_ctx(ctx, e))
                    .collect();
                ctx.tuple_type(elem_tys)
            }
            ExprKind::List(elems) => {
                let elem_ty = ctx.fresh_type_var();
                for elem in elems.iter() {
                    let ty = self.infer_expr_type_with_ctx(ctx, elem);
                    ctx.constrain_eq(ty, elem_ty);
                }
                ctx.named_type(Symbol::new("List"), vec![elem_ty])
            }
            ExprKind::Error => ctx.error_type(),
        };

        // Check mode
        if let Mode::Check(expected) = mode {
            ctx.constrain_eq(ty, expected);
        }

        // Record node type
        ctx.record_node_type(expr.id, ty);

        // Convert expression
        let kind = self.convert_expr_kind_with_ctx(ctx, *expr.kind);
        Expr::new(expr.id, kind)
    }

    /// Infer the type of an expression (just returns the type, doesn't convert).
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
            ExprKind::Var(resolved) => self.infer_var_with_ctx(ctx, resolved),
            ExprKind::Call { callee, args } => {
                let callee_ty = self.infer_expr_type_with_ctx(ctx, callee);
                let arg_types: Vec<Type<'db>> = args
                    .iter()
                    .map(|a| self.infer_expr_type_with_ctx(ctx, a))
                    .collect();
                self.infer_call_with_ctx(ctx, callee_ty, &arg_types)
            }
            ExprKind::Cons { ctor, args } => {
                let ctor_ty = self.infer_var_with_ctx(ctx, ctor);
                if args.is_empty() {
                    // Unit constructor (e.g., None) - just return the constructor type
                    ctor_ty
                } else {
                    // Constructor with arguments (e.g., Some(x)) - treat as function call
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.infer_expr_type_with_ctx(ctx, a))
                        .collect();
                    self.infer_call_with_ctx(ctx, ctor_ty, &arg_types)
                }
            }
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
                let result_ty = ctx.fresh_type_var();
                for arm in arms {
                    ctx.push_scope();
                    self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                    let arm_ty = self.infer_expr_type_with_ctx(ctx, &arm.body);
                    ctx.constrain_eq(arm_ty, result_ty);
                    ctx.pop_scope();
                }
                result_ty
            }
            _ => ctx.fresh_type_var(),
        }
    }

    /// Infer the type of a variable reference.
    fn infer_var_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        resolved: &ResolvedRef<'db>,
    ) -> Type<'db> {
        match resolved {
            ResolvedRef::Local { id, name } => {
                // Try by LocalId first, then by name
                let by_id = if id.is_unresolved() {
                    None
                } else {
                    ctx.lookup_local(*id)
                };
                let by_name = ctx.lookup_local_by_name(*name);
                by_id.or(by_name).unwrap_or_else(|| ctx.fresh_type_var())
            }
            ResolvedRef::Function { id } => ctx
                .instantiate_function(*id)
                .unwrap_or_else(|| ctx.fresh_type_var()),
            ResolvedRef::Constructor { id, .. } => ctx
                .instantiate_constructor(*id)
                .unwrap_or_else(|| ctx.fresh_type_var()),
            ResolvedRef::Module { .. } => ctx.error_type(),
            ResolvedRef::TypeDef { .. } => {
                // Type definitions cannot be used as values in expression context.
                // This typically happens when an enum name like `Option` is used
                // directly without a variant like `Some` or `None`.
                ctx.error_type()
            }
            ResolvedRef::Builtin(builtin) => self.infer_builtin_with_ctx(ctx, builtin),
            ResolvedRef::AbilityOp { ability, op } => {
                // Look up the ability operation signature from the module type env
                if let Some(op_info) = self.env.lookup_ability_op(*ability, *op) {
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
                    let row_var = ctx.fresh_row_var();
                    let effect = EffectRow::new(self.db(), vec![ability_effect], Some(row_var));
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

    /// Infer the type of a builtin reference.
    fn infer_builtin_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        builtin: &BuiltinRef,
    ) -> Type<'db> {
        let effect = EffectRow::pure(self.db());
        match builtin {
            // Arithmetic operations: (a, a) -> a
            BuiltinRef::Add
            | BuiltinRef::Sub
            | BuiltinRef::Mul
            | BuiltinRef::Div
            | BuiltinRef::Mod => {
                let a = ctx.fresh_type_var();
                ctx.func_type(vec![a, a], a, effect)
            }
            // Unary negation: a -> a
            BuiltinRef::Neg => {
                let a = ctx.fresh_type_var();
                ctx.func_type(vec![a], a, effect)
            }
            // Comparison operations: (a, a) -> Bool
            BuiltinRef::Eq
            | BuiltinRef::Ne
            | BuiltinRef::Lt
            | BuiltinRef::Le
            | BuiltinRef::Gt
            | BuiltinRef::Ge => {
                let a = ctx.fresh_type_var();
                ctx.func_type(vec![a, a], ctx.bool_type(), effect)
            }
            // Boolean binary ops: (Bool, Bool) -> Bool
            BuiltinRef::And | BuiltinRef::Or => ctx.func_type(
                vec![ctx.bool_type(), ctx.bool_type()],
                ctx.bool_type(),
                effect,
            ),
            // Boolean unary op: Bool -> Bool
            BuiltinRef::Not => ctx.func_type(vec![ctx.bool_type()], ctx.bool_type(), effect),
            // String concatenation: (String, String) -> String
            BuiltinRef::Concat => ctx.func_type(
                vec![ctx.string_type(), ctx.string_type()],
                ctx.string_type(),
                effect,
            ),
            // List cons: (a, List a) -> List a
            BuiltinRef::Cons => {
                let a = ctx.fresh_type_var();
                let list_a = ctx.named_type(Symbol::new("List"), vec![a]);
                ctx.func_type(vec![a, list_a], list_a, effect)
            }
            // List concatenation: (List a, List a) -> List a
            BuiltinRef::ListConcat => {
                let a = ctx.fresh_type_var();
                let list_a = ctx.named_type(Symbol::new("List"), vec![a]);
                ctx.func_type(vec![list_a, list_a], list_a, effect)
            }
            // IO operations
            BuiltinRef::Print => {
                let a = ctx.fresh_type_var();
                let io_effect = ctx.fresh_effect_row();
                ctx.func_type(vec![a], ctx.nil_type(), io_effect)
            }
            BuiltinRef::ReadLine => {
                let io_effect = ctx.fresh_effect_row();
                ctx.func_type(vec![], ctx.string_type(), io_effect)
            }
        }
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
    ) -> Type<'db> {
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
            let expected_cont_ty = Type::new(
                self.db(),
                TypeKind::Continuation {
                    arg: cont_arg_ty,
                    result: cont_result_ty,
                },
            );

            // Also create function type for the normal case
            let callee_effect = ctx.fresh_effect_row();
            let expected_func_ty = ctx.func_type(param_types.clone(), result_ty, callee_effect);

            // Try to match the callee type: if it's already a concrete Continuation type,
            // use continuation semantics; otherwise use function semantics.
            // We check the callee type directly rather than relying on unification to choose.
            if matches!(callee_ty.kind(self.db()), TypeKind::Continuation { .. }) {
                // Callee is a continuation - constrain as continuation call
                ctx.constrain_eq(callee_ty, expected_cont_ty);
                ctx.constrain_eq(cont_arg_ty, arg_types[0]);
                return cont_result_ty;
            } else {
                // Callee is a UniVar, concrete function type, or other - use function semantics
                // This handles cases like higher-order functions with unknown callees
                ctx.constrain_eq(callee_ty, expected_func_ty);
                for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                    ctx.constrain_eq(*param_ty, *arg_ty);
                }
                return result_ty;
            }
        }

        // Multiple arguments (or zero): must be a function call
        let callee_effect = ctx.fresh_effect_row();
        let expected_func_ty = ctx.func_type(param_types.clone(), result_ty, callee_effect);
        ctx.constrain_eq(callee_ty, expected_func_ty);

        // Constrain argument types
        for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
            ctx.constrain_eq(*param_ty, *arg_ty);
        }

        // Effect handling: We rely on type unification to handle effect compatibility.
        //
        // When the callee's type is unified with the expected function type,
        // the effect row is also unified. This naturally handles:
        // - Pure callees: their effect unifies with any effect (subsumption)
        // - Effectful callees: their effects must be present in the context
        //
        // We DON'T add merge_effect or constrain_row_eq here because:
        // 1. The callee's type is constrained via constrain_eq(callee_ty, expected_func_ty)
        // 2. If callee is pure (like apply_pure), its effect {} unifies with callee_effect
        // 3. If callee is effectful, its effects propagate through type unification
        //
        // The actual effect checking happens when:
        // - The function's return type is checked against the declared signature
        // - Ability operations add their effect to the caller's effect row via
        //   the row variable in their type signature
        let _ = callee_effect; // Effect is handled through type unification

        result_ty
    }

    /// Extract struct name and type arguments from a type.
    ///
    /// Returns (Some(struct_name), type_args) if the type is a Named or App type,
    /// otherwise (None, empty vec).
    fn extract_struct_info(&self, ty: Type<'db>) -> (Option<Symbol>, Vec<Type<'db>>) {
        match ty.kind(self.db()) {
            TypeKind::Named { name, args } => (Some(*name), args.clone()),
            TypeKind::App { ctor, args } => {
                if let TypeKind::Named { name, .. } = ctor.kind(self.db()) {
                    (Some(*name), args.clone())
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
        let list_sym = Symbol::new("List");
        match ty.kind(self.db()) {
            TypeKind::Named { name, args } if *name == list_sym && args.len() == 1 => args[0],
            TypeKind::App { ctor, args } if args.len() == 1 => {
                if let TypeKind::Named { name, .. } = ctor.kind(self.db())
                    && *name == list_sym
                {
                    return args[0];
                }
                ctx.fresh_type_var()
            }
            _ => ctx.fresh_type_var(),
        }
    }

    /// Look up a struct field type by struct name and field name.
    ///
    /// Returns the field type with BoundVars substituted by the given type arguments.
    fn lookup_field_type_from_struct(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        struct_name: Symbol,
        field_name: Symbol,
        type_args: &[Type<'db>],
    ) -> Option<Type<'db>> {
        let (type_params, field_ty) = self.env.lookup_struct_field(struct_name, field_name)?;
        if type_params.is_empty() || type_args.is_empty() {
            Some(field_ty)
        } else {
            Some(self.substitute_bound_vars(ctx, field_ty, type_args))
        }
    }

    /// Look up a struct field type from the receiver type.
    ///
    /// Given a receiver type like `Point` or `Point(Int)`, look up the field `x`
    /// and return its type with BoundVars substituted by the actual type arguments.
    fn lookup_struct_field_type(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        receiver_ty: Type<'db>,
        field_name: Symbol,
    ) -> Option<Type<'db>> {
        // Extract struct name from receiver type
        let struct_name = match receiver_ty.kind(self.db()) {
            TypeKind::Named { name, .. } => *name,
            TypeKind::App { ctor, .. } => {
                // Recursively extract from constructor
                match ctor.kind(self.db()) {
                    TypeKind::Named { name, .. } => *name,
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
        let (type_params, field_ty) = self.env.lookup_struct_field(struct_name, field_name)?;

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
            Some(self.substitute_bound_vars(ctx, field_ty, &actual_args))
        }
    }

    /// Substitute BoundVars in a type with actual types.
    ///
    /// Panics if a BoundVar index is out of bounds.
    fn substitute_bound_vars(
        &self,
        _ctx: &mut FunctionInferenceContext<'_, 'db>,
        ty: Type<'db>,
        args: &[Type<'db>],
    ) -> Type<'db> {
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
        kind: ExprKind<ResolvedRef<'db>>,
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
            ExprKind::Var(resolved) => ExprKind::Var(self.convert_ref_with_ctx(ctx, resolved)),
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: self.check_expr_with_ctx(ctx, callee, Mode::Infer),
                args: args
                    .into_iter()
                    .map(|a| self.check_expr_with_ctx(ctx, a, Mode::Infer))
                    .collect(),
            },
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor: self.convert_ref_with_ctx(ctx, ctor),
                args: args
                    .into_iter()
                    .map(|a| self.check_expr_with_ctx(ctx, a, Mode::Infer))
                    .collect(),
            },
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => ExprKind::Record {
                type_name: self.convert_ref_with_ctx(ctx, type_name),
                fields: fields
                    .into_iter()
                    .map(|(name, expr)| (name, self.check_expr_with_ctx(ctx, expr, Mode::Infer)))
                    .collect(),
                spread: spread.map(|e| self.check_expr_with_ctx(ctx, e, Mode::Infer)),
            },
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => ExprKind::MethodCall {
                receiver: self.check_expr_with_ctx(ctx, receiver, Mode::Infer),
                method,
                args: args
                    .into_iter()
                    .map(|a| self.check_expr_with_ctx(ctx, a, Mode::Infer))
                    .collect(),
            },
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: self.check_expr_with_ctx(ctx, lhs, Mode::Infer),
                rhs: self.check_expr_with_ctx(ctx, rhs, Mode::Infer),
            },
            ExprKind::Block { stmts, value } => {
                ctx.push_scope();
                let converted_stmts: Vec<_> = stmts
                    .into_iter()
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
                    .into_iter()
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

                // Check exhaustiveness
                self.check_exhaustiveness(scrutinee_ty, &converted_arms, scrutinee_expr.id);

                ExprKind::Case {
                    scrutinee: scrutinee_expr,
                    arms: converted_arms,
                }
            }
            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.check_expr_with_ctx(ctx, body, Mode::Infer),
            },
            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.check_expr_with_ctx(ctx, body, Mode::Infer),
                handlers: handlers
                    .into_iter()
                    .map(|h| self.convert_handler_arm_with_ctx(ctx, h))
                    .collect(),
            },
            ExprKind::Tuple(elements) => ExprKind::Tuple(
                elements
                    .into_iter()
                    .map(|e| self.check_expr_with_ctx(ctx, e, Mode::Infer))
                    .collect(),
            ),
            ExprKind::List(elements) => ExprKind::List(
                elements
                    .into_iter()
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
        resolved: ResolvedRef<'db>,
    ) -> TypedRef<'db> {
        let ty = self.infer_var_with_ctx(ctx, &resolved);
        TypedRef { resolved, ty }
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
                let value_ty = if let Some(ann) = ty {
                    let expected = self.annotation_to_type_with_ctx(ctx, ann);
                    let inferred = self.infer_expr_type_with_ctx(ctx, value);
                    ctx.constrain_eq(inferred, expected);
                    expected
                } else {
                    self.infer_expr_type_with_ctx(ctx, value)
                };
                // Constrain pattern type to match value type
                let pattern_ty = self.infer_pattern_type_with_ctx(ctx, pattern);
                ctx.constrain_eq(pattern_ty, value_ty);
                self.bind_pattern_vars_with_ctx(ctx, pattern, value_ty);
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
        stmt: Stmt<ResolvedRef<'db>>,
    ) -> Stmt<TypedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                value,
                ty,
            } => {
                let value = if let Some(ann) = &ty {
                    let expected = self.annotation_to_type_with_ctx(ctx, ann);
                    self.check_expr_with_ctx(ctx, value, Mode::Check(expected))
                } else {
                    self.check_expr_with_ctx(ctx, value, Mode::Infer)
                };
                let value_ty = ctx
                    .get_node_type(value.id)
                    .unwrap_or_else(|| ctx.fresh_type_var());

                // Constrain pattern type to match value type
                let pattern_ty = self.infer_pattern_type_with_ctx(ctx, &pattern);
                ctx.constrain_eq(pattern_ty, value_ty);

                self.bind_pattern_vars_with_ctx(ctx, &pattern, value_ty);
                let pattern = self.convert_pattern_with_ctx(ctx, pattern);

                Stmt::Let {
                    id,
                    pattern,
                    value,
                    ty,
                }
            }
            Stmt::Expr { id, expr } => {
                let expr = self.check_expr_with_ctx(ctx, expr, Mode::Infer);
                Stmt::Expr { id, expr }
            }
        }
    }

    // =========================================================================
    // Pattern handling
    // =========================================================================

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
                LiteralPattern::Unit => ctx.nil_type(),
            },
            PatternKind::Variant { ctor, fields } => {
                let ctor_ty = self.infer_var_with_ctx(ctx, ctor);
                ctx.record_node_type(pattern.id, ctor_ty);

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
                ctx.named_type(Symbol::new("List"), vec![elem_ty])
            }
            PatternKind::ListRest { head, .. } => {
                let elem_ty = ctx.fresh_type_var();
                for pat in head {
                    let pat_ty = self.infer_pattern_type_with_ctx(ctx, pat);
                    ctx.constrain_eq(pat_ty, elem_ty);
                }
                ctx.named_type(Symbol::new("List"), vec![elem_ty])
            }
            PatternKind::Record { type_name, .. } => {
                if let Some(type_ref) = type_name {
                    self.infer_var_with_ctx(ctx, type_ref)
                } else {
                    ctx.fresh_type_var()
                }
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
                ctx.bind_local_by_name(*name, ty);
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
                for field in fields {
                    let field_ty = ctx
                        .get_node_type(field.id)
                        .unwrap_or_else(|| ctx.fresh_type_var());
                    self.bind_pattern_vars_with_ctx(ctx, field, field_ty);
                }
            }
            PatternKind::Record { fields, .. } => {
                let (struct_name, type_args) = self.extract_struct_info(ty);

                for field in fields {
                    let field_ty = struct_name
                        .and_then(|name| {
                            self.lookup_field_type_from_struct(ctx, name, field.name, &type_args)
                        })
                        .unwrap_or_else(|| ctx.fresh_type_var());

                    if let Some(pat) = &field.pattern {
                        self.bind_pattern_vars_with_ctx(ctx, pat, field_ty);
                    } else {
                        // Shorthand { name } - bind the field name directly
                        ctx.bind_local_by_name(field.name, field_ty);
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
                    let list_ty = ctx.named_type(Symbol::new("List"), vec![elem_ty]);
                    ctx.bind_local(*local_id, list_ty);
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => {
                ctx.bind_local_by_name(*name, ty);
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
        pattern: Pattern<ResolvedRef<'db>>,
    ) -> Pattern<TypedRef<'db>> {
        let kind = match *pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind { name, local_id },
            PatternKind::Literal(lit) => PatternKind::Literal(lit),
            PatternKind::Variant { ctor, fields } => PatternKind::Variant {
                ctor: self.convert_ref_with_ctx(ctx, ctor),
                fields: fields
                    .into_iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            },
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => PatternKind::Record {
                type_name: type_name.map(|t| self.convert_ref_with_ctx(ctx, t)),
                fields: fields
                    .into_iter()
                    .map(|f| self.convert_field_pattern_with_ctx(ctx, f))
                    .collect(),
                rest,
            },
            PatternKind::Tuple(patterns) => PatternKind::Tuple(
                patterns
                    .into_iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            ),
            PatternKind::List(patterns) => PatternKind::List(
                patterns
                    .into_iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
            ),
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => PatternKind::ListRest {
                head: head
                    .into_iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
                rest,
                rest_local_id,
            },
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.convert_pattern_with_ctx(ctx, pattern),
                name,
                local_id,
            },
            PatternKind::Error => PatternKind::Error,
        };
        Pattern::new(pattern.id, kind)
    }

    /// Convert a field pattern.
    fn convert_field_pattern_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        fp: FieldPattern<ResolvedRef<'db>>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            id: fp.id,
            name: fp.name,
            pattern: fp.pattern.map(|p| self.convert_pattern_with_ctx(ctx, p)),
        }
    }

    /// Convert a field pattern with an expected type.
    fn convert_field_pattern_with_expected_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        fp: FieldPattern<ResolvedRef<'db>>,
        expected: Type<'db>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            id: fp.id,
            name: fp.name,
            pattern: fp
                .pattern
                .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, expected)),
        }
    }

    /// Convert a case arm with an expected scrutinee type.
    fn convert_arm_with_scrutinee_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: Arm<ResolvedRef<'db>>,
        scrutinee_ty: Type<'db>,
    ) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: self.convert_pattern_with_expected_ctx(ctx, arm.pattern, scrutinee_ty),
            guard: arm.guard.map(|g| {
                let bool_ty = ctx.bool_type();
                self.check_expr_with_ctx(ctx, g, Mode::Check(bool_ty))
            }),
            body: self.check_expr_with_ctx(ctx, arm.body, Mode::Infer),
        }
    }

    /// Convert a pattern with an expected type.
    fn convert_pattern_with_expected_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        pattern: Pattern<ResolvedRef<'db>>,
        expected: Type<'db>,
    ) -> Pattern<TypedRef<'db>> {
        let kind = match *pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind { name, local_id },
            PatternKind::Literal(lit) => PatternKind::Literal(lit),
            PatternKind::Variant { ctor, fields } => {
                let ctor_ty = ctx
                    .get_node_type(pattern.id)
                    .unwrap_or_else(|| self.infer_var_with_ctx(ctx, &ctor));

                match ctor_ty.kind(self.db()) {
                    TypeKind::Func { params, result, .. } => {
                        ctx.constrain_eq(*result, expected);
                        let fields = fields
                            .into_iter()
                            .zip(params.iter())
                            .map(|(p, param_ty)| {
                                self.convert_pattern_with_expected_ctx(ctx, p, *param_ty)
                            })
                            .collect();
                        PatternKind::Variant {
                            ctor: TypedRef {
                                resolved: ctor,
                                ty: ctor_ty,
                            },
                            fields,
                        }
                    }
                    _ => {
                        ctx.constrain_eq(ctor_ty, expected);
                        PatternKind::Variant {
                            ctor: TypedRef {
                                resolved: ctor,
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
                let (struct_name, type_args) = self.extract_struct_info(expected);

                let converted_fields = fields
                    .into_iter()
                    .map(|f| {
                        let field_expected = struct_name
                            .and_then(|name| {
                                self.lookup_field_type_from_struct(ctx, name, f.name, &type_args)
                            })
                            .unwrap_or_else(|| ctx.fresh_type_var());

                        self.convert_field_pattern_with_expected_ctx(ctx, f, field_expected)
                    })
                    .collect();

                PatternKind::Record {
                    type_name: type_name.map(|t| self.convert_ref_with_ctx(ctx, t)),
                    fields: converted_fields,
                    rest,
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
                        .into_iter()
                        .zip(elem_expectations)
                        .map(|(p, exp)| self.convert_pattern_with_expected_ctx(ctx, p, exp))
                        .collect(),
                )
            }
            PatternKind::List(patterns) => {
                let elem_ty = ctx.fresh_type_var();
                let list_ty = ctx.named_type(Symbol::new("List"), vec![elem_ty]);
                ctx.constrain_eq(expected, list_ty);
                PatternKind::List(
                    patterns
                        .into_iter()
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
                let list_ty = ctx.named_type(Symbol::new("List"), vec![elem_ty]);
                ctx.constrain_eq(expected, list_ty);
                PatternKind::ListRest {
                    head: head
                        .into_iter()
                        .map(|p| self.convert_pattern_with_expected_ctx(ctx, p, elem_ty))
                        .collect(),
                    rest,
                    rest_local_id,
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.convert_pattern_with_expected_ctx(ctx, pattern, expected),
                name,
                local_id,
            },
            PatternKind::Error => PatternKind::Error,
        };
        Pattern::new(pattern.id, kind)
    }

    /// Convert a handler arm.
    fn convert_handler_arm_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        arm: HandlerArm<ResolvedRef<'db>>,
    ) -> HandlerArm<TypedRef<'db>> {
        let kind = match arm.kind {
            HandlerKind::Result { binding } => HandlerKind::Result {
                binding: self.convert_pattern_with_ctx(ctx, binding),
            },
            HandlerKind::Effect {
                ability,
                op,
                params,
                continuation,
                continuation_local_id,
            } => {
                // Bind continuation variable with Continuation type if named
                if let (Some(k_name), Some(k_local_id)) = (continuation, continuation_local_id) {
                    // Create a Continuation type for the continuation variable.
                    // The arg type is the type passed when resuming (effect operation's return type),
                    // and the result type is the handler's result type.
                    // For now, use fresh type variables since we don't have precise type info.
                    let arg_ty = ctx.fresh_type_var();
                    let result_ty = ctx.fresh_type_var();
                    let cont_ty = Type::new(
                        self.db(),
                        TypeKind::Continuation {
                            arg: arg_ty,
                            result: result_ty,
                        },
                    );
                    ctx.bind_local(k_local_id, cont_ty);
                    ctx.bind_local_by_name(k_name, cont_ty);
                }

                HandlerKind::Effect {
                    ability: self.convert_ref_with_ctx(ctx, ability),
                    op,
                    params: params
                        .into_iter()
                        .map(|p| self.convert_pattern_with_ctx(ctx, p))
                        .collect(),
                    continuation,
                    continuation_local_id,
                }
            }
        };
        HandlerArm {
            id: arm.id,
            kind,
            body: self.check_expr_with_ctx(ctx, arm.body, Mode::Infer),
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

        match &ann.kind {
            TypeAnnotationKind::Named(name) => {
                if *name == "Int" {
                    ctx.int_type()
                } else if *name == "Nat" {
                    ctx.nat_type()
                } else if *name == "Float" {
                    ctx.float_type()
                } else if *name == "Bool" {
                    ctx.bool_type()
                } else if *name == "String" {
                    ctx.string_type()
                } else if *name == "Bytes" {
                    ctx.bytes_type()
                } else if *name == "Rune" {
                    ctx.rune_type()
                } else if *name == "Nil" {
                    ctx.nil_type()
                } else {
                    ctx.named_type(*name, vec![])
                }
            }
            TypeAnnotationKind::Path(parts) if !parts.is_empty() => {
                if let Some(&name) = parts.last() {
                    ctx.named_type(name, vec![])
                } else {
                    ctx.error_type()
                }
            }
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_with_ctx(ctx, ctor);
                if let TypeKind::Named { name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_with_ctx(ctx, a))
                        .collect();
                    ctx.named_type(*name, arg_types)
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
                let row_var = ctx.fresh_row_var();
                let module_path = self.current_module_path();
                let effect = crate::ast::abilities_to_effect_row(
                    self.db(),
                    abilities,
                    &module_path,
                    &mut |a| self.annotation_to_type_with_ctx(ctx, a),
                    || row_var,
                );
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
            TypeAnnotationKind::Path(_) | TypeAnnotationKind::Error => ctx.error_type(),
        }
    }

    // =========================================================================
    // Exhaustiveness checking
    // =========================================================================

    /// Check that a case expression is exhaustive.
    ///
    /// This performs a simplified exhaustiveness check:
    /// - If the last arm has a wildcard or bind pattern, it's exhaustive
    /// - If matching an enum, all variants must be covered
    /// - Otherwise, emit a warning for patterns we can't fully analyze
    fn check_exhaustiveness(
        &self,
        scrutinee_ty: Type<'db>,
        arms: &[Arm<TypedRef<'db>>],
        span_node_id: crate::ast::NodeId,
    ) {
        // Empty arms is definitely non-exhaustive
        if arms.is_empty() {
            let span = self.get_span(span_node_id);
            Diagnostic {
                message: "non-exhaustive case expression: no patterns provided".to_string(),
                span,
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(self.db());
            return;
        }

        // Check if the last arm is a catch-all (wildcard or bind)
        let last_arm = &arms[arms.len() - 1];
        if self.is_catch_all_pattern(&last_arm.pattern) {
            return; // Exhaustive via catch-all
        }

        // Extract the enum name from the scrutinee type
        let enum_name = match scrutinee_ty.kind(self.db()) {
            TypeKind::Named { name, .. } => *name,
            TypeKind::App { ctor, .. } => match ctor.kind(self.db()) {
                TypeKind::Named { name, .. } => *name,
                _ => {
                    // Can't determine type - emit warning
                    self.emit_exhaustiveness_warning(span_node_id);
                    return;
                }
            },
            TypeKind::UniVar { .. } => {
                // Type not yet resolved - can't check
                return;
            }
            TypeKind::Bool => {
                // Bool can be exhaustive if both True and False are covered
                let mut has_true = false;
                let mut has_false = false;
                for arm in arms {
                    self.collect_bool_coverage(&arm.pattern, &mut has_true, &mut has_false);
                }
                if has_true && has_false {
                    return; // Exhaustive
                }
                let span = self.get_span(span_node_id);
                let missing = match (has_true, has_false) {
                    (true, false) => "False",
                    (false, true) => "True",
                    _ => "True, False",
                };
                Diagnostic {
                    message: format!("non-exhaustive case expression: missing cases: {}", missing),
                    span,
                    severity: DiagnosticSeverity::Error,
                    phase: CompilationPhase::TypeChecking,
                }
                .accumulate(self.db());
                return;
            }
            TypeKind::Int | TypeKind::String | TypeKind::Float | TypeKind::Nat => {
                // Primitive types without catch-all are non-exhaustive
                let span = self.get_span(span_node_id);
                Diagnostic {
                    message: "non-exhaustive case expression: not all cases are covered"
                        .to_string(),
                    span,
                    severity: DiagnosticSeverity::Error,
                    phase: CompilationPhase::TypeChecking,
                }
                .accumulate(self.db());
                return;
            }
            _ => {
                self.emit_exhaustiveness_warning(span_node_id);
                return;
            }
        };

        // Look up the enum's variants
        let Some(all_variants) = self.env.lookup_enum_variants(enum_name) else {
            // Not a known enum - emit warning
            self.emit_exhaustiveness_warning(span_node_id);
            return;
        };

        // Collect covered variants from arms
        let mut covered_variants: HashSet<Symbol> = HashSet::new();
        for arm in arms {
            self.collect_covered_variants(&arm.pattern, &mut covered_variants);
        }

        // Check if all variants are covered
        let missing: Vec<_> = all_variants
            .iter()
            .filter(|v| !covered_variants.contains(v))
            .collect();

        if !missing.is_empty() {
            let missing_names: Vec<_> = missing.iter().map(|s| s.to_string()).collect();
            let span = self.get_span(span_node_id);
            Diagnostic {
                message: format!(
                    "non-exhaustive case expression: missing variants: {}",
                    missing_names.join(", ")
                ),
                span,
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(self.db());
        }
    }

    /// Check if a pattern is a catch-all (covers all values).
    fn is_catch_all_pattern(&self, pattern: &Pattern<TypedRef<'db>>) -> bool {
        match &*pattern.kind {
            PatternKind::Wildcard => true,
            PatternKind::Bind { .. } => true,
            PatternKind::As { pattern, .. } => self.is_catch_all_pattern(pattern),
            _ => false,
        }
    }

    /// Collect variant names covered by a pattern.
    fn collect_covered_variants(
        &self,
        pattern: &Pattern<TypedRef<'db>>,
        covered: &mut HashSet<Symbol>,
    ) {
        match &*pattern.kind {
            PatternKind::Variant { ctor, .. } => {
                // Extract variant name from constructor
                if let crate::ast::ResolvedRef::Constructor { variant, .. } = &ctor.resolved {
                    covered.insert(*variant);
                }
            }
            PatternKind::As { pattern, .. } => {
                self.collect_covered_variants(pattern, covered);
            }
            _ => {}
        }
    }

    /// Collect Bool literal coverage from a pattern.
    fn collect_bool_coverage(
        &self,
        pattern: &Pattern<TypedRef<'db>>,
        has_true: &mut bool,
        has_false: &mut bool,
    ) {
        match &*pattern.kind {
            PatternKind::Literal(LiteralPattern::Bool(true)) => *has_true = true,
            PatternKind::Literal(LiteralPattern::Bool(false)) => *has_false = true,
            PatternKind::As { pattern, .. } => {
                self.collect_bool_coverage(pattern, has_true, has_false);
            }
            _ => {}
        }
    }

    /// Emit a warning for patterns we can't fully analyze.
    fn emit_exhaustiveness_warning(&self, span_node_id: crate::ast::NodeId) {
        let span = self.get_span(span_node_id);
        Diagnostic {
            message: "exhaustiveness check: unable to verify all cases are covered".to_string(),
            span,
            severity: DiagnosticSeverity::Warning,
            phase: CompilationPhase::TypeChecking,
        }
        .accumulate(self.db());
    }

    // =========================================================================
    // Handle expression support
    // =========================================================================

    /// Extract the ability ID from a ResolvedRef.
    ///
    /// Used by handle expression to determine which abilities are being handled.
    /// Returns None for references that don't represent abilities.
    fn extract_ability_id_from_ref(&self, resolved: &ResolvedRef<'db>) -> Option<AbilityId<'db>> {
        match resolved {
            ResolvedRef::AbilityOp { ability, .. } => Some(*ability),
            ResolvedRef::TypeDef { id } => {
                // TypeDef might be an ability reference in handler context
                // Create an AbilityId with the same module path and name
                Some(AbilityId::new(
                    self.db(),
                    id.module_path(self.db()).clone(),
                    id.name(self.db()),
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
        row: EffectRow<'db>,
        handled_ability_ids: &[AbilityId<'db>],
    ) -> EffectRow<'db> {
        let db = ctx.db();
        let effects = row.effects(db);

        // Filter out effects whose ability_id is in the handled list
        let remaining_effects: Vec<_> = effects
            .iter()
            .filter(|effect| !handled_ability_ids.contains(&effect.ability_id))
            .cloned()
            .collect();

        // If there's a row variable tail, we need to constrain it
        if let Some(tail_var) = row.rest(db) {
            // Create a fresh row variable for the result (the "unhandled" portion of the tail)
            let result_var = ctx.fresh_row_var();

            // Collect Effect entries from the original effects list, preserving type args.
            // Use full Effect equality (including args) for proper deduplication of
            // parameterized abilities like State(Int) vs State(Bool).
            let mut seen_effects: HashSet<Effect<'db>> = HashSet::new();
            let handled_effects: Vec<crate::ast::Effect<'db>> = effects
                .iter()
                .filter(|effect| handled_ability_ids.contains(&effect.ability_id))
                .filter(|effect| seen_effects.insert((*effect).clone()))
                .cloned()
                .collect();

            // Add constraint: tail_var = {handled_effects | result_var}
            // This decomposes the tail into the handled effects plus the result,
            // ensuring result_var cannot contain any of the handled abilities.
            let tail_row = EffectRow::open(db, tail_var);
            let decomposed_row = EffectRow::new(db, handled_effects, Some(result_var));
            ctx.constrain_row_eq(tail_row, decomposed_row);

            // Return the remaining effects with the fresh result variable
            EffectRow::new(db, remaining_effects, Some(result_var))
        } else {
            // Closed row: just return the remaining effects
            EffectRow::new(db, remaining_effects, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Symbol, SymbolVec};

    use crate::ast::{
        BuiltinRef, EffectRow, FuncDefId, NodeId, SpanMap, Type, TypeAnnotation,
        TypeAnnotationKind, TypeKind,
    };
    use crate::typeck::{FunctionInferenceContext, ModuleTypeEnv};

    use super::TypeChecker;

    /// Helper to create a TypeChecker for testing.
    fn make_test_checker(db: &dyn salsa::Database) -> TypeChecker<'_> {
        TypeChecker::new(db, SpanMap::default())
    }

    /// Helper to create a FunctionInferenceContext for testing.
    fn make_test_ctx<'a, 'db>(
        db: &'db dyn salsa::Database,
        env: &'a ModuleTypeEnv<'db>,
    ) -> FunctionInferenceContext<'a, 'db> {
        let func_id = FuncDefId::new(db, SymbolVec::new(), Symbol::new("test_func"));
        FunctionInferenceContext::new(db, env, func_id)
    }

    /// Helper to create a TypeAnnotation with a given kind.
    fn make_annotation(kind: TypeAnnotationKind) -> TypeAnnotation {
        TypeAnnotation {
            id: NodeId::from_raw(0),
            kind,
        }
    }

    // =========================================================================
    // infer_builtin_with_ctx tests
    // =========================================================================

    #[salsa_test]
    fn test_builtin_arithmetic_ops(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        for builtin in [
            BuiltinRef::Add,
            BuiltinRef::Sub,
            BuiltinRef::Mul,
            BuiltinRef::Div,
            BuiltinRef::Mod,
        ] {
            let ty = checker.infer_builtin_with_ctx(&mut ctx, &builtin);

            // Should be fn(?a, ?a) -> ?a
            if let TypeKind::Func {
                params,
                result,
                effect,
            } = ty.kind(db)
            {
                assert_eq!(params.len(), 2, "{:?} should have 2 params", builtin);
                assert_eq!(
                    params[0], params[1],
                    "{:?} params should be same type",
                    builtin
                );
                assert_eq!(
                    params[0], *result,
                    "{:?} param and result should be same",
                    builtin
                );
                // Effect should be pure
                assert!(effect.is_pure(db), "{:?} should be pure", builtin);
                // Params should be UniVar (fresh type var)
                assert!(
                    matches!(params[0].kind(db), TypeKind::UniVar { .. }),
                    "{:?} param should be UniVar",
                    builtin
                );
            } else {
                panic!(
                    "{:?} should return Func type, got {:?}",
                    builtin,
                    ty.kind(db)
                );
            }
        }
    }

    #[salsa_test]
    fn test_builtin_neg(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        let ty = checker.infer_builtin_with_ctx(&mut ctx, &BuiltinRef::Neg);

        // Should be fn(?a) -> ?a
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = ty.kind(db)
        {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], *result);
            assert!(effect.is_pure(db));
            assert!(matches!(params[0].kind(db), TypeKind::UniVar { .. }));
        } else {
            panic!("Neg should return Func type");
        }
    }

    #[salsa_test]
    fn test_builtin_comparison_ops(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);
        let bool_ty = Type::new(db, TypeKind::Bool);

        for builtin in [
            BuiltinRef::Eq,
            BuiltinRef::Ne,
            BuiltinRef::Lt,
            BuiltinRef::Le,
            BuiltinRef::Gt,
            BuiltinRef::Ge,
        ] {
            let ty = checker.infer_builtin_with_ctx(&mut ctx, &builtin);

            // Should be fn(?a, ?a) -> Bool
            if let TypeKind::Func {
                params,
                result,
                effect,
            } = ty.kind(db)
            {
                assert_eq!(params.len(), 2, "{:?} should have 2 params", builtin);
                assert_eq!(
                    params[0], params[1],
                    "{:?} params should be same type",
                    builtin
                );
                assert_eq!(*result, bool_ty, "{:?} result should be Bool", builtin);
                assert!(effect.is_pure(db), "{:?} should be pure", builtin);
            } else {
                panic!("{:?} should return Func type", builtin);
            }
        }
    }

    #[salsa_test]
    fn test_builtin_boolean_binary_ops(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);
        let bool_ty = Type::new(db, TypeKind::Bool);

        for builtin in [BuiltinRef::And, BuiltinRef::Or] {
            let ty = checker.infer_builtin_with_ctx(&mut ctx, &builtin);

            // Should be fn(Bool, Bool) -> Bool
            if let TypeKind::Func {
                params,
                result,
                effect,
            } = ty.kind(db)
            {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0], bool_ty);
                assert_eq!(params[1], bool_ty);
                assert_eq!(*result, bool_ty);
                assert!(effect.is_pure(db));
            } else {
                panic!("{:?} should return Func type", builtin);
            }
        }
    }

    #[salsa_test]
    fn test_builtin_boolean_not(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);
        let bool_ty = Type::new(db, TypeKind::Bool);

        let ty = checker.infer_builtin_with_ctx(&mut ctx, &BuiltinRef::Not);

        // Should be fn(Bool) -> Bool
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = ty.kind(db)
        {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], bool_ty);
            assert_eq!(*result, bool_ty);
            assert!(effect.is_pure(db));
        } else {
            panic!("Not should return Func type");
        }
    }

    #[salsa_test]
    fn test_builtin_concat(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);
        let string_ty = Type::new(db, TypeKind::String);

        let ty = checker.infer_builtin_with_ctx(&mut ctx, &BuiltinRef::Concat);

        // Should be fn(String, String) -> String
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = ty.kind(db)
        {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0], string_ty);
            assert_eq!(params[1], string_ty);
            assert_eq!(*result, string_ty);
            assert!(effect.is_pure(db));
        } else {
            panic!("Concat should return Func type");
        }
    }

    #[salsa_test]
    fn test_builtin_list_ops(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // Cons: (a, List a) -> List a
        let cons_ty = checker.infer_builtin_with_ctx(&mut ctx, &BuiltinRef::Cons);
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = cons_ty.kind(db)
        {
            assert_eq!(params.len(), 2);
            // First param is element type
            let elem_ty = params[0];
            // Second param is List(elem_ty)
            if let TypeKind::Named { name, args } = params[1].kind(db) {
                assert_eq!(*name, Symbol::new("List"));
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], elem_ty);
            } else {
                panic!("Cons second param should be Named(List)");
            }
            // Result is List(elem_ty)
            if let TypeKind::Named { name, args } = result.kind(db) {
                assert_eq!(*name, Symbol::new("List"));
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], elem_ty);
            } else {
                panic!("Cons result should be Named(List)");
            }
            assert!(effect.is_pure(db));
        } else {
            panic!("Cons should return Func type");
        }

        // ListConcat: (List a, List a) -> List a
        let list_concat_ty = checker.infer_builtin_with_ctx(&mut ctx, &BuiltinRef::ListConcat);
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = list_concat_ty.kind(db)
        {
            assert_eq!(params.len(), 2);
            // Both params should be List types
            assert_eq!(params[0], params[1]);
            assert_eq!(params[0], *result);
            if let TypeKind::Named { name, .. } = params[0].kind(db) {
                assert_eq!(*name, Symbol::new("List"));
            } else {
                panic!("ListConcat params should be Named(List)");
            }
            assert!(effect.is_pure(db));
        } else {
            panic!("ListConcat should return Func type");
        }
    }

    #[salsa_test]
    fn test_builtin_io_ops(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);
        let nil_ty = Type::new(db, TypeKind::Nil);
        let string_ty = Type::new(db, TypeKind::String);

        // Print: (a) ->{?e} Nil
        let print_ty = checker.infer_builtin_with_ctx(&mut ctx, &BuiltinRef::Print);
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = print_ty.kind(db)
        {
            assert_eq!(params.len(), 1);
            // Param should be a fresh type var
            assert!(matches!(params[0].kind(db), TypeKind::UniVar { .. }));
            assert_eq!(*result, nil_ty);
            // Effect should be open (fresh row var)
            assert!(
                effect.rest(db).is_some(),
                "Print should have open effect row"
            );
        } else {
            panic!("Print should return Func type");
        }

        // ReadLine: () ->{?e} String
        let readline_ty = checker.infer_builtin_with_ctx(&mut ctx, &BuiltinRef::ReadLine);
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = readline_ty.kind(db)
        {
            assert!(params.is_empty());
            assert_eq!(*result, string_ty);
            // Effect should be open (fresh row var)
            assert!(
                effect.rest(db).is_some(),
                "ReadLine should have open effect row"
            );
        } else {
            panic!("ReadLine should return Func type");
        }
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
            ("String", TypeKind::String),
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
    fn test_annotation_user_defined_type(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        let ann = make_annotation(TypeAnnotationKind::Named(Symbol::new("MyType")));
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

        // Should be Named { name: "MyType", args: [] }
        if let TypeKind::Named { name, args } = ty.kind(db) {
            assert_eq!(*name, Symbol::new("MyType"));
            assert!(args.is_empty());
        } else {
            panic!("User-defined type should be Named");
        }
    }

    #[salsa_test]
    fn test_annotation_path(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        let ann = make_annotation(TypeAnnotationKind::Path(vec![
            Symbol::new("std"),
            Symbol::new("Option"),
        ]));
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

        // Should use last segment as name
        if let TypeKind::Named { name, args } = ty.kind(db) {
            assert_eq!(*name, Symbol::new("Option"));
            assert!(args.is_empty());
        } else {
            panic!("Path type should be Named");
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
        if let TypeKind::Named { name, args } = ty.kind(db) {
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
            assert_eq!(elems[1], Type::new(db, TypeKind::String));
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

    #[salsa_test]
    fn test_substitute_basic(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // BoundVar(0) + [Int] → Int
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let int_ty = Type::new(db, TypeKind::Int);
        let args = vec![int_ty];

        let result = checker.substitute_bound_vars(&mut ctx, bound_var, &args);
        assert_eq!(result, int_ty);
    }

    #[salsa_test]
    fn test_substitute_multiple(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // (BoundVar(0), BoundVar(1)) + [Int, Bool] → (Int, Bool)
        let bound0 = Type::new(db, TypeKind::BoundVar { index: 0 });
        let bound1 = Type::new(db, TypeKind::BoundVar { index: 1 });
        let tuple_ty = Type::new(db, TypeKind::Tuple(vec![bound0, bound1]));

        let int_ty = Type::new(db, TypeKind::Int);
        let bool_ty = Type::new(db, TypeKind::Bool);
        let args = vec![int_ty, bool_ty];

        let result = checker.substitute_bound_vars(&mut ctx, tuple_ty, &args);
        let expected = Type::new(db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        assert_eq!(result, expected);
    }

    #[salsa_test]
    fn test_substitute_in_func(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // fn(BoundVar(0)) -> BoundVar(0) + [Int] → fn(Int) -> Int
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 0 });
        let effect = EffectRow::pure(db);
        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![bound_var],
                result: bound_var,
                effect,
            },
        );

        let int_ty = Type::new(db, TypeKind::Int);
        let args = vec![int_ty];

        let result = checker.substitute_bound_vars(&mut ctx, func_ty, &args);
        let expected = Type::new(
            db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect,
            },
        );
        assert_eq!(result, expected);
    }

    #[salsa_test]
    #[should_panic(expected = "BoundVar index out of range")]
    fn test_substitute_out_of_bounds_panics(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // BoundVar(5) + [Int] → should panic (out of bounds)
        let bound_var = Type::new(db, TypeKind::BoundVar { index: 5 });
        let int_ty = Type::new(db, TypeKind::Int);
        let args = vec![int_ty];

        // This should panic
        checker.substitute_bound_vars(&mut ctx, bound_var, &args);
    }

    #[salsa_test]
    fn test_substitute_primitive_unchanged(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // Int + [Bool] → Int (primitives are unchanged)
        let int_ty = Type::new(db, TypeKind::Int);
        let bool_ty = Type::new(db, TypeKind::Bool);
        let args = vec![bool_ty];

        let result = checker.substitute_bound_vars(&mut ctx, int_ty, &args);
        assert_eq!(result, int_ty);
    }

    // =========================================================================
    // extract_list_element_type tests
    // =========================================================================

    #[salsa_test]
    fn test_extract_list_element_type_from_list(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // List<Int> → Int
        let int_ty = Type::new(db, TypeKind::Int);
        let list_ty = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![int_ty],
            },
        );

        let elem_ty = checker.extract_list_element_type(list_ty, &mut ctx);
        assert_eq!(elem_ty, int_ty);
    }

    #[salsa_test]
    fn test_extract_list_element_type_from_list_string(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // List<String> → String
        let string_ty = Type::new(db, TypeKind::String);
        let list_ty = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![string_ty],
            },
        );

        let elem_ty = checker.extract_list_element_type(list_ty, &mut ctx);
        assert_eq!(elem_ty, string_ty);
    }

    #[salsa_test]
    fn test_extract_list_element_type_non_list_returns_fresh_var(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // Int → fresh type var (not a List type)
        let int_ty = Type::new(db, TypeKind::Int);

        let elem_ty = checker.extract_list_element_type(int_ty, &mut ctx);

        // Should be a fresh UniVar, not Int
        assert!(
            matches!(elem_ty.kind(db), TypeKind::UniVar { .. }),
            "Expected UniVar for non-list type, got {:?}",
            elem_ty.kind(db)
        );
    }

    #[salsa_test]
    fn test_extract_list_element_type_other_named_returns_fresh_var(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // Option<Int> → fresh type var (not a List type)
        let int_ty = Type::new(db, TypeKind::Int);
        let option_ty = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![int_ty],
            },
        );

        let elem_ty = checker.extract_list_element_type(option_ty, &mut ctx);

        // Should be a fresh UniVar, not Int
        assert!(
            matches!(elem_ty.kind(db), TypeKind::UniVar { .. }),
            "Expected UniVar for Option type, got {:?}",
            elem_ty.kind(db)
        );
    }

    #[salsa_test]
    fn test_extract_list_element_type_empty_args_returns_fresh_var(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // List with no type args → fresh type var
        let list_ty = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![],
            },
        );

        let elem_ty = checker.extract_list_element_type(list_ty, &mut ctx);

        // Should be a fresh UniVar
        assert!(
            matches!(elem_ty.kind(db), TypeKind::UniVar { .. }),
            "Expected UniVar for List with no args, got {:?}",
            elem_ty.kind(db)
        );
    }

    #[salsa_test]
    fn test_extract_list_element_type_nested_list(db: &dyn salsa::Database) {
        let checker = make_test_checker(db);
        let env = ModuleTypeEnv::new(db);
        let mut ctx = make_test_ctx(db, &env);

        // List<List<Int>> → List<Int>
        let int_ty = Type::new(db, TypeKind::Int);
        let inner_list = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        let outer_list = Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![inner_list],
            },
        );

        let elem_ty = checker.extract_list_element_type(outer_list, &mut ctx);
        assert_eq!(elem_ty, inner_list);
    }
}
