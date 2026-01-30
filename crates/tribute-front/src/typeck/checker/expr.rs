//! Expression type checking.
//!
//! All expression checking methods take a `FunctionInferenceContext` as parameter,
//! enabling per-function type inference with isolated constraints.

use std::collections::HashSet;

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::{Span, Symbol};

use crate::ast::{
    Arm, BinOpKind, BuiltinRef, Effect, EffectRow, Expr, ExprKind, FieldPattern, HandlerArm,
    HandlerKind, LiteralPattern, Pattern, PatternKind, ResolvedRef, Stmt, Type, TypeKind, TypedRef,
};

use super::super::func_context::FunctionInferenceContext;
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
            ExprKind::Record { type_name, .. } => {
                // Get the struct constructor type and extract return type
                let ctor_ty = self.infer_var_with_ctx(ctx, type_name);
                if let TypeKind::Func { result, .. } = ctor_ty.kind(self.db()) {
                    // Constructor has function type: fn(fields...) -> StructType
                    *result
                } else {
                    // Fallback: use constructor type directly (shouldn't happen)
                    ctor_ty
                }
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
            ExprKind::Block { stmts: _, value } => self.infer_expr_type_with_ctx(ctx, value),
            ExprKind::Case { .. } => ctx.fresh_type_var(),
            ExprKind::Lambda { params, body } => {
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| match &p.ty {
                        Some(ann) => self.annotation_to_type_with_ctx(ctx, ann),
                        None => ctx.fresh_type_var(),
                    })
                    .collect();

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

                ctx.pop_scope();

                let result_ty = ctx.fresh_type_var();
                ctx.constrain_eq(result_ty, body_ty);
                let effect = ctx.fresh_effect_row();
                ctx.func_type(param_types, result_ty, effect)
            }
            ExprKind::Handle { body, .. } => {
                // Handle returns the body's type, with handled effects removed
                self.infer_expr_type_with_ctx(ctx, body)
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
            ExprKind::Block { value, .. } => self.infer_expr_type_with_ctx(ctx, value),
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
            ResolvedRef::Builtin(builtin) => self.infer_builtin_with_ctx(ctx, builtin),
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
    fn infer_call_with_ctx(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        callee_ty: Type<'db>,
        arg_types: &[Type<'db>],
    ) -> Type<'db> {
        // Create expected function type
        let param_types: Vec<Type<'db>> = arg_types.iter().map(|_| ctx.fresh_type_var()).collect();
        let result_ty = ctx.fresh_type_var();
        let effect = ctx.fresh_effect_row();

        let expected_func_ty = ctx.func_type(param_types.clone(), result_ty, effect);
        ctx.constrain_eq(callee_ty, expected_func_ty);

        // Constrain argument types
        for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
            ctx.constrain_eq(*param_ty, *arg_ty);
        }

        result_ty
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
    fn substitute_bound_vars(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        ty: Type<'db>,
        args: &[Type<'db>],
    ) -> Type<'db> {
        match ty.kind(self.db()) {
            TypeKind::BoundVar { index } => {
                // Substitute with actual type argument
                args.get(*index as usize).copied().unwrap_or_else(|| {
                    // BoundVar index out of range - use fresh type var
                    ctx.fresh_type_var()
                })
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let subst_params: Vec<_> = params
                    .iter()
                    .map(|p| self.substitute_bound_vars(ctx, *p, args))
                    .collect();
                let subst_result = self.substitute_bound_vars(ctx, *result, args);
                let subst_effect = self.substitute_bound_vars_in_effect(*effect, args);
                ctx.func_type(subst_params, subst_result, subst_effect)
            }
            TypeKind::Named { name, args: targs } => {
                let subst_args: Vec<_> = targs
                    .iter()
                    .map(|a| self.substitute_bound_vars(ctx, *a, args))
                    .collect();
                ctx.named_type(*name, subst_args)
            }
            TypeKind::App {
                ctor,
                args: app_args,
            } => {
                let subst_ctor = self.substitute_bound_vars(ctx, *ctor, args);
                let subst_args: Vec<_> = app_args
                    .iter()
                    .map(|a| self.substitute_bound_vars(ctx, *a, args))
                    .collect();
                Type::new(
                    self.db(),
                    TypeKind::App {
                        ctor: subst_ctor,
                        args: subst_args,
                    },
                )
            }
            TypeKind::Tuple(elems) => {
                let subst_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.substitute_bound_vars(ctx, *e, args))
                    .collect();
                Type::new(self.db(), TypeKind::Tuple(subst_elems))
            }
            // Other types don't contain BoundVars
            _ => ty,
        }
    }

    /// Substitute BoundVars in an effect row with actual types.
    fn substitute_bound_vars_in_effect(
        &self,
        effect: EffectRow<'db>,
        args: &[Type<'db>],
    ) -> EffectRow<'db> {
        let effects = effect.effects(self.db());
        if effects.is_empty() {
            return effect;
        }

        let subst_effects: Vec<Effect<'db>> = effects
            .iter()
            .map(|e| {
                let subst_args: Vec<Type<'db>> = e
                    .args
                    .iter()
                    .map(|a| self.substitute_bound_vars_in_type(*a, args))
                    .collect();
                Effect {
                    name: e.name,
                    args: subst_args,
                }
            })
            .collect();

        EffectRow::new(self.db(), subst_effects, effect.rest(self.db()))
    }

    /// Substitute BoundVars in a type without needing a FunctionInferenceContext.
    ///
    /// This is used for effect row substitution where we don't have access to ctx.
    fn substitute_bound_vars_in_type(&self, ty: Type<'db>, args: &[Type<'db>]) -> Type<'db> {
        match ty.kind(self.db()) {
            TypeKind::BoundVar { index } => args.get(*index as usize).copied().unwrap_or(ty),
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let subst_params: Vec<_> = params
                    .iter()
                    .map(|p| self.substitute_bound_vars_in_type(*p, args))
                    .collect();
                let subst_result = self.substitute_bound_vars_in_type(*result, args);
                let subst_effect = self.substitute_bound_vars_in_effect(*effect, args);
                Type::new(
                    self.db(),
                    TypeKind::Func {
                        params: subst_params,
                        result: subst_result,
                        effect: subst_effect,
                    },
                )
            }
            TypeKind::Named { name, args: targs } => {
                let subst_args: Vec<_> = targs
                    .iter()
                    .map(|a| self.substitute_bound_vars_in_type(*a, args))
                    .collect();
                Type::new(
                    self.db(),
                    TypeKind::Named {
                        name: *name,
                        args: subst_args,
                    },
                )
            }
            TypeKind::App {
                ctor,
                args: app_args,
            } => {
                let subst_ctor = self.substitute_bound_vars_in_type(*ctor, args);
                let subst_args: Vec<_> = app_args
                    .iter()
                    .map(|a| self.substitute_bound_vars_in_type(*a, args))
                    .collect();
                Type::new(
                    self.db(),
                    TypeKind::App {
                        ctor: subst_ctor,
                        args: subst_args,
                    },
                )
            }
            TypeKind::Tuple(elems) => {
                let subst_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.substitute_bound_vars_in_type(*e, args))
                    .collect();
                Type::new(self.db(), TypeKind::Tuple(subst_elems))
            }
            _ => ty,
        }
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
            ExprKind::Block { stmts, value } => ExprKind::Block {
                stmts: stmts
                    .into_iter()
                    .map(|s| self.convert_stmt_with_ctx(ctx, s))
                    .collect(),
                value: self.check_expr_with_ctx(ctx, value, Mode::Infer),
            },
            ExprKind::Case { scrutinee, arms } => {
                let scrutinee_expr = self.check_expr_with_ctx(ctx, scrutinee, Mode::Infer);
                let scrutinee_ty = ctx
                    .get_node_type(scrutinee_expr.id)
                    .unwrap_or_else(|| ctx.fresh_type_var());

                let result_ty = ctx.fresh_type_var();

                let converted_arms: Vec<_> = arms
                    .into_iter()
                    .map(|arm| {
                        // Each arm gets its own scope for pattern bindings
                        ctx.push_scope();

                        let pattern_ty = self.infer_pattern_type_with_ctx(ctx, &arm.pattern);
                        ctx.constrain_eq(pattern_ty, scrutinee_ty);
                        self.bind_pattern_vars_with_ctx(ctx, &arm.pattern, scrutinee_ty);
                        let converted = self.convert_arm_with_scrutinee_ctx(ctx, arm, scrutinee_ty);
                        if let Some(body_ty) = ctx.get_node_type(converted.body.id) {
                            ctx.constrain_eq(body_ty, result_ty);
                        }

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
                for field in fields {
                    if let Some(pat) = &field.pattern {
                        let fresh_ty = ctx.fresh_type_var();
                        self.bind_pattern_vars_with_ctx(ctx, pat, fresh_ty);
                    }
                }
            }
            PatternKind::List(pats) => {
                let fresh_vars: Vec<_> = pats.iter().map(|_| ctx.fresh_type_var()).collect();
                for (pat, fresh_ty) in pats.iter().zip(fresh_vars) {
                    self.bind_pattern_vars_with_ctx(ctx, pat, fresh_ty);
                }
            }
            PatternKind::ListRest {
                head,
                rest_local_id,
                ..
            } => {
                let fresh_vars: Vec<_> = head.iter().map(|_| ctx.fresh_type_var()).collect();
                for (pat, fresh_ty) in head.iter().zip(fresh_vars) {
                    self.bind_pattern_vars_with_ctx(ctx, pat, fresh_ty);
                }
                if let Some(local_id) = rest_local_id {
                    let rest_ty = ctx.fresh_type_var();
                    ctx.bind_local(*local_id, rest_ty);
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
            } => PatternKind::Record {
                type_name: type_name.map(|t| self.convert_ref_with_ctx(ctx, t)),
                fields: fields
                    .into_iter()
                    .map(|f| self.convert_field_pattern_with_ctx(ctx, f))
                    .collect(),
                rest,
            },
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
            } => HandlerKind::Effect {
                ability: self.convert_ref_with_ctx(ctx, ability),
                op,
                params: params
                    .into_iter()
                    .map(|p| self.convert_pattern_with_ctx(ctx, p))
                    .collect(),
                continuation,
            },
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
                } else if *name == "()" {
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
                let effect = crate::ast::abilities_to_effect_row(
                    self.db(),
                    abilities,
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
            let span = Span::new(0, 0); // TODO: Get proper span from scrutinee
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
                let span = Span::new(0, 0);
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
                let span = Span::new(0, 0);
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
            let span = Span::new(0, 0);
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
    fn emit_exhaustiveness_warning(&self, _span_node_id: crate::ast::NodeId) {
        let span = Span::new(0, 0);
        Diagnostic {
            message: "exhaustiveness check: unable to verify all cases are covered".to_string(),
            span,
            severity: DiagnosticSeverity::Warning,
            phase: CompilationPhase::TypeChecking,
        }
        .accumulate(self.db());
    }
}
