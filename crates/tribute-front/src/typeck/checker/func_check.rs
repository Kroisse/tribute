//! Function-level type checking.
//!
//! Each function is type-checked with an isolated `FunctionInferenceContext`,
//! ensuring that type variables (UniVars) are fully resolved within the function
//! before moving to the next.

use std::collections::HashMap;

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};

use crate::ast::{
    Arm, Expr, ExprKind, FieldPattern, FuncDecl, FuncDefId, HandlerArm, HandlerKind, Pattern,
    PatternKind, ResolvedRef, Stmt, Type, TypeKind, TypeScheme, TypedRef, UniVarId,
};

use super::super::func_context::FunctionInferenceContext;
use super::super::solver::{RowSubst, TypeSolver, TypeSubst};
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
        let (param_types, expected_return, instantiated_func_ty) =
            self.get_func_signature_with_type(&mut ctx, func_id, &func);

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
        if let TypeKind::Func { effect, .. } = instantiated_func_ty.kind(self.db()) {
            ctx.set_current_effect(*effect);
        }

        // 3. Check body against expected return type
        let body = self.check_expr_with_ctx(&mut ctx, func.body, Mode::Check(expected_return));

        // 4. Solve constraints for this function only
        let constraints = ctx.take_constraints();
        // Take node_types now while ctx is still alive, before we need mutable self access
        let func_node_types = ctx.take_node_types();
        // Drop ctx now to release the borrow of self.env
        drop(ctx);

        let mut solver = TypeSolver::new(self.db());

        if let Err(error) = solver.solve(constraints) {
            let span = self.get_span(func.id);
            Diagnostic {
                message: format!("Type error in function '{}': {:?}", func.name, error),
                span,
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(self.db());
        }

        // 5. Apply substitution and generalization
        let type_subst = solver.type_subst();
        let row_subst = solver.row_subst();

        // First, collect ALL unresolved UniVars from both the function type AND the body.
        // This ensures that UniVars created during body type checking (e.g., from builtin calls)
        // are also included in the generalization mapping.
        let mut all_univars = Vec::new();
        type_subst.collect_univars_from_type(
            self.db(),
            instantiated_func_ty,
            row_subst,
            &mut all_univars,
        );
        self.collect_univars_from_body(&body, type_subst, row_subst, &mut all_univars);

        // Create a comprehensive mapping from all UniVars to BoundVars
        let var_to_index: HashMap<UniVarId<'db>, u32> = all_univars
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, i as u32))
            .collect();

        // Apply substitution and generalization to the function type
        let substituted_ty = type_subst.apply_with_rows(self.db(), instantiated_func_ty, row_subst);
        let generalized =
            type_subst.apply_generalization(self.db(), substituted_ty, row_subst, &var_to_index);

        // Create type params for all generalized UniVars
        let type_params: Vec<crate::ast::TypeParam> = (0..var_to_index.len())
            .map(|_| crate::ast::TypeParam::anonymous())
            .collect();

        let new_scheme = TypeScheme::new(self.db(), type_params, generalized);
        // Update the function's type scheme with the generalized version
        self.env.register_function(func_id, new_scheme);

        // 6. Apply substitution and generalization to all TypedRef types in the body
        let body = self.apply_subst_to_body(body, type_subst, row_subst, &var_to_index);

        // 7. Collect node types from this function's context and apply substitution.
        // Note: We apply substitution but NOT generalization, because these types
        // are used for IR lowering which needs concrete types, not polymorphic ones.
        for (node_id, ty) in func_node_types {
            let substituted = type_subst.apply_with_rows(self.db(), ty, row_subst);
            self.node_types.insert(node_id, substituted);
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

    /// Get function signature from the registered scheme.
    ///
    /// Instantiates the scheme with fresh type variables for this function's inference.
    /// Returns (param_types, return_type, instantiated_func_type).
    fn get_func_signature_with_type(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        func_id: FuncDefId<'db>,
        func: &FuncDecl<ResolvedRef<'db>>,
    ) -> (Vec<Type<'db>>, Type<'db>, Type<'db>) {
        if let Some(scheme) = self.env.lookup_function(func_id) {
            let func_ty = ctx.instantiate_scheme(scheme);
            if let TypeKind::Func { params, result, .. } = func_ty.kind(self.db()) {
                return (params.clone(), *result, func_ty);
            }
        }

        // Fallback: create fresh type variables
        let param_types: Vec<Type<'db>> =
            func.params.iter().map(|_| ctx.fresh_type_var()).collect();
        let return_ty = ctx.fresh_type_var();
        let effect = ctx.fresh_effect_row();
        let func_ty = ctx.func_type(param_types.clone(), return_ty, effect);
        (param_types, return_ty, func_ty)
    }

    // =========================================================================
    // Body type transformation
    // =========================================================================

    /// Apply substitution and generalization to all types in the body expression.
    ///
    /// This ensures that all TypedRef types have UniVars replaced with:
    /// 1. Their resolved concrete type (from substitution), or
    /// 2. The corresponding BoundVar (from generalization mapping)
    fn apply_subst_to_body(
        &self,
        body: Expr<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Expr<TypedRef<'db>> {
        let kind = self.apply_subst_to_expr_kind(*body.kind, type_subst, row_subst, var_to_index);
        Expr::new(body.id, kind)
    }

    /// Apply substitution to an expression kind.
    fn apply_subst_to_expr_kind(
        &self,
        kind: ExprKind<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
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
                callee: self.apply_subst_to_body(callee, type_subst, row_subst, var_to_index),
                args: args
                    .into_iter()
                    .map(|a| self.apply_subst_to_body(a, type_subst, row_subst, var_to_index))
                    .collect(),
            },
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor: self.apply_subst_to_typed_ref(ctor, type_subst, row_subst, var_to_index),
                args: args
                    .into_iter()
                    .map(|a| self.apply_subst_to_body(a, type_subst, row_subst, var_to_index))
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
                            self.apply_subst_to_body(expr, type_subst, row_subst, var_to_index),
                        )
                    })
                    .collect(),
                spread: spread
                    .map(|e| self.apply_subst_to_body(e, type_subst, row_subst, var_to_index)),
            },
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => ExprKind::MethodCall {
                receiver: self.apply_subst_to_body(receiver, type_subst, row_subst, var_to_index),
                method,
                args: args
                    .into_iter()
                    .map(|a| self.apply_subst_to_body(a, type_subst, row_subst, var_to_index))
                    .collect(),
            },
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: self.apply_subst_to_body(lhs, type_subst, row_subst, var_to_index),
                rhs: self.apply_subst_to_body(rhs, type_subst, row_subst, var_to_index),
            },
            ExprKind::Block { stmts, value } => ExprKind::Block {
                stmts: stmts
                    .into_iter()
                    .map(|s| self.apply_subst_to_stmt(s, type_subst, row_subst, var_to_index))
                    .collect(),
                value: self.apply_subst_to_body(value, type_subst, row_subst, var_to_index),
            },
            ExprKind::Case { scrutinee, arms } => ExprKind::Case {
                scrutinee: self.apply_subst_to_body(scrutinee, type_subst, row_subst, var_to_index),
                arms: arms
                    .into_iter()
                    .map(|arm| self.apply_subst_to_arm(arm, type_subst, row_subst, var_to_index))
                    .collect(),
            },
            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.apply_subst_to_body(body, type_subst, row_subst, var_to_index),
            },
            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.apply_subst_to_body(body, type_subst, row_subst, var_to_index),
                handlers: handlers
                    .into_iter()
                    .map(|h| {
                        self.apply_subst_to_handler_arm(h, type_subst, row_subst, var_to_index)
                    })
                    .collect(),
            },
            ExprKind::Tuple(elems) => ExprKind::Tuple(
                elems
                    .into_iter()
                    .map(|e| self.apply_subst_to_body(e, type_subst, row_subst, var_to_index))
                    .collect(),
            ),
            ExprKind::List(elems) => ExprKind::List(
                elems
                    .into_iter()
                    .map(|e| self.apply_subst_to_body(e, type_subst, row_subst, var_to_index))
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
    fn apply_subst_to_type(
        &self,
        ty: Type<'db>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Type<'db> {
        // First apply the substitution to resolve UniVars
        let substituted = type_subst.apply_with_rows(self.db(), ty, row_subst);
        // Then apply the generalization mapping to convert remaining UniVars to BoundVars
        type_subst.apply_generalization(self.db(), substituted, row_subst, var_to_index)
    }

    /// Apply substitution to a statement.
    fn apply_subst_to_stmt(
        &self,
        stmt: Stmt<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
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
                value: self.apply_subst_to_body(value, type_subst, row_subst, var_to_index),
                ty,
            },
            Stmt::Expr { id, expr } => Stmt::Expr {
                id,
                expr: self.apply_subst_to_body(expr, type_subst, row_subst, var_to_index),
            },
        }
    }

    /// Apply substitution to a case arm.
    fn apply_subst_to_arm(
        &self,
        arm: Arm<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: self.apply_subst_to_pattern(arm.pattern, type_subst, row_subst, var_to_index),
            guard: arm
                .guard
                .map(|g| self.apply_subst_to_body(g, type_subst, row_subst, var_to_index)),
            body: self.apply_subst_to_body(arm.body, type_subst, row_subst, var_to_index),
        }
    }

    /// Apply substitution to a handler arm.
    fn apply_subst_to_handler_arm(
        &self,
        arm: HandlerArm<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> HandlerArm<TypedRef<'db>> {
        let kind = match arm.kind {
            HandlerKind::Result { binding } => HandlerKind::Result {
                binding: self.apply_subst_to_pattern(binding, type_subst, row_subst, var_to_index),
            },
            HandlerKind::Effect {
                ability,
                op,
                params,
                continuation,
                continuation_local_id,
            } => HandlerKind::Effect {
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
                continuation,
                continuation_local_id,
            },
        };
        HandlerArm {
            id: arm.id,
            kind,
            body: self.apply_subst_to_body(arm.body, type_subst, row_subst, var_to_index),
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
    fn collect_univars_from_body(
        &self,
        body: &Expr<TypedRef<'db>>,
        type_subst: &TypeSubst<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        self.collect_univars_from_expr_kind(&body.kind, type_subst, row_subst, out);
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
                        HandlerKind::Result { binding } => {
                            self.collect_univars_from_pattern(binding, type_subst, row_subst, out);
                        }
                        HandlerKind::Effect {
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
