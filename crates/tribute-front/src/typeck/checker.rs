//! Type checker implementation.
//!
//! Performs bidirectional type checking on the AST, transforming
//! `Module<ResolvedRef<'db>>` into `Module<TypedRef<'db>>`.

use crate::ast::{
    Arm, BuiltinRef, ConstDecl, Decl, EffectRow, EnumDecl, Expr, ExprKind, FieldPattern, FuncDecl,
    HandlerArm, HandlerKind, Module, Pattern, PatternKind, ResolvedRef, Stmt, StructDecl, Type,
    TypeKind, TypeParam, TypeScheme, TypedRef,
};

use super::context::TypeContext;
use super::solver::TypeSolver;

/// Type checking mode.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Mode<'db> {
    /// Infer the type of an expression.
    Infer,
    /// Check that an expression has a specific type.
    Check(Type<'db>),
}

/// Type checker for AST expressions.
pub struct TypeChecker<'db> {
    ctx: TypeContext<'db>,
}

impl<'db> TypeChecker<'db> {
    /// Create a new type checker.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            ctx: TypeContext::new(db),
        }
    }

    /// Get the database.
    fn db(&self) -> &'db dyn salsa::Database {
        self.ctx.db()
    }

    /// Type check a module.
    pub fn check_module(mut self, module: Module<ResolvedRef<'db>>) -> Module<TypedRef<'db>> {
        // Phase 1: Collect type definitions and function signatures
        self.collect_declarations(&module);

        // Phase 2: Type check all declarations and generate constraints
        let decls = module
            .decls
            .into_iter()
            .map(|decl| self.check_decl(decl))
            .collect();

        // Phase 3: Solve constraints
        let constraints = self.ctx.take_constraints();
        let mut solver = TypeSolver::new(self.db());

        // Solve constraints and log any errors
        // TODO: Collect errors into diagnostics instead of just logging
        if let Err(error) = solver.solve(constraints) {
            tracing::warn!("Type constraint solving failed: {:?}", error);
        }

        // Phase 4: Apply substitution to produce final types
        // TODO: Apply substitution to node types

        Module {
            id: module.id,
            name: module.name,
            decls,
        }
    }

    // =========================================================================
    // Declaration collection (Phase 1)
    // =========================================================================

    /// Collect type definitions and function signatures from declarations.
    fn collect_declarations(&mut self, module: &Module<ResolvedRef<'db>>) {
        for decl in &module.decls {
            match decl {
                Decl::Function(func) => {
                    self.collect_function_signature(func);
                }
                Decl::Struct(s) => {
                    self.collect_struct_def(s);
                }
                Decl::Enum(e) => {
                    self.collect_enum_def(e);
                }
                Decl::Const(c) => {
                    self.collect_const_def(c);
                }
                Decl::Ability(_) | Decl::Use(_) => {
                    // Abilities and imports don't define types directly
                }
            }
        }
    }

    /// Collect a function's type signature.
    fn collect_function_signature(&mut self, func: &FuncDecl<ResolvedRef<'db>>) {
        // Build type parameters
        let type_params: Vec<TypeParam> = func
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name))
            .collect();

        // Build parameter types
        // TODO: Convert TypeAnnotation to Type<'db> instead of using fresh vars.
        // This requires implementing annotation_to_type() that walks TypeAnnotationKind
        // and produces the corresponding Type. For now, type inference will infer types.
        let param_types: Vec<Type<'db>> = func
            .params
            .iter()
            .map(|_p| self.ctx.fresh_type_var())
            .collect();

        // Build return type
        // TODO: Convert return type annotation when present
        let return_ty = self.ctx.fresh_type_var();

        // Build effect row
        // TODO: Convert effect annotations when present
        let effect = func
            .effects
            .as_ref()
            .map(|_| self.ctx.fresh_effect_row())
            .unwrap_or_else(|| EffectRow::pure(self.db()));

        // Create function type
        let func_ty = self.ctx.func_type(param_types, return_ty, effect);

        // Create type scheme
        let scheme = TypeScheme::new(self.db(), type_params, func_ty);

        // Register the function (we need the FuncDefId from ResolvedRef)
        // For now, skip registration since we'd need to look up the ID
        let _ = scheme;
    }

    /// Collect a struct definition.
    fn collect_struct_def(&mut self, s: &StructDecl) {
        let name = s.name;
        let type_params: Vec<TypeParam> = s
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name))
            .collect();

        // The struct type itself
        let args: Vec<Type<'db>> = (0..type_params.len() as u32)
            .map(|i| Type::new(self.db(), TypeKind::BoundVar { index: i }))
            .collect();
        let struct_ty = self.ctx.named_type(name, args);

        let scheme = TypeScheme::new(self.db(), type_params, struct_ty);
        self.ctx.register_type_def(name, scheme);
    }

    /// Collect an enum definition.
    fn collect_enum_def(&mut self, e: &EnumDecl) {
        let name = e.name;
        let type_params: Vec<TypeParam> = e
            .type_params
            .iter()
            .map(|tp| TypeParam::named(tp.name))
            .collect();

        // The enum type itself
        let args: Vec<Type<'db>> = (0..type_params.len() as u32)
            .map(|i| Type::new(self.db(), TypeKind::BoundVar { index: i }))
            .collect();
        let enum_ty = self.ctx.named_type(name, args);

        let scheme = TypeScheme::new(self.db(), type_params, enum_ty);
        self.ctx.register_type_def(name, scheme);

        // TODO: Register constructors for each variant
    }

    /// Collect a constant definition.
    fn collect_const_def(&mut self, c: &ConstDecl<ResolvedRef<'db>>) {
        // Constants will be type-checked with the expression
        let _ = c;
    }

    // =========================================================================
    // Declaration checking (Phase 2)
    // =========================================================================

    /// Type check a declaration.
    fn check_decl(&mut self, decl: Decl<ResolvedRef<'db>>) -> Decl<TypedRef<'db>> {
        match decl {
            Decl::Function(func) => Decl::Function(self.check_func_decl(func)),
            Decl::Struct(s) => Decl::Struct(self.check_struct_decl(s)),
            Decl::Enum(e) => Decl::Enum(self.check_enum_decl(e)),
            Decl::Ability(a) => Decl::Ability(self.check_ability_decl(a)),
            Decl::Const(c) => Decl::Const(self.check_const_decl(c)),
            Decl::Use(u) => Decl::Use(self.check_use_decl(u)),
        }
    }

    /// Type check a function declaration.
    fn check_func_decl(&mut self, func: FuncDecl<ResolvedRef<'db>>) -> FuncDecl<TypedRef<'db>> {
        // Bind parameter types to the context
        // TODO: ParamDecl currently lacks LocalId. To properly bind parameters:
        // 1. Add LocalId field to ParamDecl during resolve phase
        // 2. Call self.ctx.bind_local(param.local_id, ty) here
        // For now, parameter lookups rely on name-based matching.
        for param in &func.params {
            let ty = self.ctx.fresh_type_var();
            // Bind by name as a workaround until LocalId is available
            self.ctx.bind_local_by_name(param.name, ty);
        }

        // Check body
        let body = self.check_expr(func.body, Mode::Infer);

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

    /// Type check a struct declaration (no body to check).
    fn check_struct_decl(&mut self, s: StructDecl) -> StructDecl {
        s // Struct declarations don't contain expressions
    }

    /// Type check an enum declaration (no body to check).
    fn check_enum_decl(&mut self, e: EnumDecl) -> EnumDecl {
        e // Enum declarations don't contain expressions
    }

    /// Type check an ability declaration (no body to check).
    fn check_ability_decl(&mut self, a: crate::ast::AbilityDecl) -> crate::ast::AbilityDecl {
        a // Ability declarations don't contain expressions
    }

    /// Type check a constant declaration.
    fn check_const_decl(&mut self, c: ConstDecl<ResolvedRef<'db>>) -> ConstDecl<TypedRef<'db>> {
        let value = self.check_expr(c.value, Mode::Infer);

        ConstDecl {
            id: c.id,
            is_pub: c.is_pub,
            name: c.name,
            ty: c.ty,
            value,
        }
    }

    /// Type check a use declaration (nothing to check).
    fn check_use_decl(&mut self, u: crate::ast::UseDecl) -> crate::ast::UseDecl {
        u
    }

    // =========================================================================
    // Expression checking
    // =========================================================================

    /// Type check an expression.
    fn check_expr(&mut self, expr: Expr<ResolvedRef<'db>>, mode: Mode<'db>) -> Expr<TypedRef<'db>> {
        let ty = match &*expr.kind {
            ExprKind::NatLit(_) => self.ctx.nat_type(),
            ExprKind::IntLit(_) => self.ctx.int_type(),
            ExprKind::FloatLit(_) => self.ctx.float_type(),
            ExprKind::BoolLit(_) => self.ctx.bool_type(),
            ExprKind::StringLit(_) => self.ctx.string_type(),
            ExprKind::BytesLit(_) => self.ctx.bytes_type(),
            ExprKind::Nil => self.ctx.nil_type(),

            ExprKind::Var(resolved) => self.infer_var(resolved),
            ExprKind::Call { callee, args } => {
                // Infer callee type
                let callee_ty = self.infer_expr_type(callee);
                self.infer_call(callee_ty, args.len())
            }
            ExprKind::Cons { ctor, args } => {
                // Constructor application
                let ctor_ty = self.infer_var(ctor);
                self.infer_call(ctor_ty, args.len())
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                // Record construction
                let _ = (type_name, fields, spread);
                self.ctx.fresh_type_var() // TODO: Proper record typing
            }
            ExprKind::FieldAccess { expr, field } => {
                // Field access
                let _ = (expr, field);
                self.ctx.fresh_type_var() // TODO: Proper field access typing
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Method call (UFCS)
                let _ = (receiver, method, args);
                self.ctx.fresh_type_var() // TODO: Proper method call typing
            }
            ExprKind::BinOp { op, lhs, rhs } => {
                // Binary operation
                let _ = (op, lhs, rhs);
                self.ctx.fresh_type_var() // TODO: Proper binop typing
            }
            ExprKind::Block { stmts, value } => {
                // Block returns type of value expression
                let _ = stmts;
                self.infer_expr_type(value)
            }
            ExprKind::Case { scrutinee, arms } => {
                // Case expression
                let _ = (scrutinee, arms);
                self.ctx.fresh_type_var() // TODO: Proper case typing
            }
            ExprKind::Lambda { params, body } => {
                // Lambda expression
                let _ = (params, body);
                self.ctx.fresh_type_var() // TODO: Proper lambda typing
            }
            ExprKind::Handle { body, handlers } => {
                // Handle expression
                let _ = (body, handlers);
                self.ctx.fresh_type_var() // TODO: Proper handle typing
            }
            ExprKind::Tuple(elements) => {
                // Tuple type
                let _ = elements;
                self.ctx.fresh_type_var() // TODO: Proper tuple typing
            }
            ExprKind::List(elements) => {
                // List type
                let _ = elements;
                self.ctx.fresh_type_var() // TODO: Proper list typing
            }
            ExprKind::Error => self.ctx.error_type(),
        };

        // If in check mode, add constraint
        if let Mode::Check(expected) = mode {
            self.ctx.constrain_eq(ty, expected);
        }

        // Record the type for this node
        self.ctx.record_node_type(expr.id, ty);

        // Convert to TypedRef
        let kind = self.convert_expr_kind(*expr.kind);
        Expr::new(expr.id, kind)
    }

    /// Infer the type of an expression.
    fn infer_expr_type(&mut self, expr: &Expr<ResolvedRef<'db>>) -> Type<'db> {
        match &*expr.kind {
            ExprKind::Var(resolved) => self.infer_var(resolved),
            ExprKind::NatLit(_) => self.ctx.nat_type(),
            ExprKind::IntLit(_) => self.ctx.int_type(),
            ExprKind::FloatLit(_) => self.ctx.float_type(),
            ExprKind::BoolLit(_) => self.ctx.bool_type(),
            ExprKind::StringLit(_) => self.ctx.string_type(),
            ExprKind::BytesLit(_) => self.ctx.bytes_type(),
            ExprKind::Nil => self.ctx.nil_type(),
            _ => self.ctx.fresh_type_var(),
        }
    }

    /// Infer the type of a variable reference.
    fn infer_var(&mut self, resolved: &ResolvedRef<'db>) -> Type<'db> {
        match resolved {
            ResolvedRef::Local { id, .. } => self
                .ctx
                .lookup_local(*id)
                .unwrap_or_else(|| self.ctx.fresh_type_var()),
            ResolvedRef::Function { id } => self
                .ctx
                .instantiate_function(*id)
                .unwrap_or_else(|| self.ctx.fresh_type_var()),
            ResolvedRef::Constructor { id, .. } => self
                .ctx
                .instantiate_constructor(*id)
                .unwrap_or_else(|| self.ctx.fresh_type_var()),
            ResolvedRef::Builtin(builtin) => self.infer_builtin(builtin),
            ResolvedRef::Module { .. } => self.ctx.error_type(),
        }
    }

    /// Infer the type of a builtin operation.
    fn infer_builtin(&mut self, builtin: &BuiltinRef) -> Type<'db> {
        match builtin {
            BuiltinRef::Print => {
                // print: (a) -> Nil
                let a = self.ctx.fresh_type_var();
                let effect = self.ctx.fresh_effect_row();
                self.ctx.func_type(vec![a], self.ctx.nil_type(), effect)
            }
            BuiltinRef::ReadLine => {
                // read_line: () -> String
                let effect = self.ctx.fresh_effect_row();
                self.ctx.func_type(vec![], self.ctx.string_type(), effect)
            }
            // Arithmetic operations: (a, a) -> a
            BuiltinRef::Add
            | BuiltinRef::Sub
            | BuiltinRef::Mul
            | BuiltinRef::Div
            | BuiltinRef::Mod => {
                let a = self.ctx.fresh_type_var();
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(vec![a, a], a, effect)
            }
            // Unary negation: (a) -> a
            BuiltinRef::Neg => {
                let a = self.ctx.fresh_type_var();
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(vec![a], a, effect)
            }
            // Comparison operations: (a, a) -> Bool
            BuiltinRef::Eq
            | BuiltinRef::Ne
            | BuiltinRef::Lt
            | BuiltinRef::Le
            | BuiltinRef::Gt
            | BuiltinRef::Ge => {
                let a = self.ctx.fresh_type_var();
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(vec![a, a], self.ctx.bool_type(), effect)
            }
            // Boolean operations: (Bool, Bool) -> Bool or (Bool) -> Bool
            BuiltinRef::And | BuiltinRef::Or => {
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(
                    vec![self.ctx.bool_type(), self.ctx.bool_type()],
                    self.ctx.bool_type(),
                    effect,
                )
            }
            BuiltinRef::Not => {
                let effect = EffectRow::pure(self.db());
                self.ctx
                    .func_type(vec![self.ctx.bool_type()], self.ctx.bool_type(), effect)
            }
            // String concatenation: (String, String) -> String
            BuiltinRef::Concat => {
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(
                    vec![self.ctx.string_type(), self.ctx.string_type()],
                    self.ctx.string_type(),
                    effect,
                )
            }
            // List operations
            BuiltinRef::Cons => {
                // cons: (a, List(a)) -> List(a)
                let a = self.ctx.fresh_type_var();
                let list_a = self.ctx.named_type(trunk_ir::Symbol::new("List"), vec![a]);
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(vec![a, list_a], list_a, effect)
            }
            BuiltinRef::ListConcat => {
                // list_concat: (List(a), List(a)) -> List(a)
                let a = self.ctx.fresh_type_var();
                let list_a = self.ctx.named_type(trunk_ir::Symbol::new("List"), vec![a]);
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(vec![list_a, list_a], list_a, effect)
            }
        }
    }

    /// Infer the result type of a function call.
    fn infer_call(&mut self, callee_ty: Type<'db>, arg_count: usize) -> Type<'db> {
        // Create fresh type variables for parameters and result
        let param_types: Vec<Type<'db>> =
            (0..arg_count).map(|_| self.ctx.fresh_type_var()).collect();
        let result_ty = self.ctx.fresh_type_var();
        let effect = self.ctx.fresh_effect_row();

        // Constrain callee to be a function type
        let expected_func_ty = self.ctx.func_type(param_types, result_ty, effect);
        self.ctx.constrain_eq(callee_ty, expected_func_ty);

        result_ty
    }

    /// Convert expression kind from ResolvedRef to TypedRef.
    fn convert_expr_kind(&mut self, kind: ExprKind<ResolvedRef<'db>>) -> ExprKind<TypedRef<'db>> {
        match kind {
            ExprKind::NatLit(n) => ExprKind::NatLit(n),
            ExprKind::IntLit(n) => ExprKind::IntLit(n),
            ExprKind::FloatLit(f) => ExprKind::FloatLit(f),
            ExprKind::BoolLit(b) => ExprKind::BoolLit(b),
            ExprKind::StringLit(s) => ExprKind::StringLit(s),
            ExprKind::BytesLit(b) => ExprKind::BytesLit(b),
            ExprKind::Nil => ExprKind::Nil,
            ExprKind::Var(resolved) => ExprKind::Var(self.convert_ref(resolved)),
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: self.check_expr(callee, Mode::Infer),
                args: args
                    .into_iter()
                    .map(|a| self.check_expr(a, Mode::Infer))
                    .collect(),
            },
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor: self.convert_ref(ctor),
                args: args
                    .into_iter()
                    .map(|a| self.check_expr(a, Mode::Infer))
                    .collect(),
            },
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => ExprKind::Record {
                type_name: self.convert_ref(type_name),
                fields: fields
                    .into_iter()
                    .map(|(name, expr)| (name, self.check_expr(expr, Mode::Infer)))
                    .collect(),
                spread: spread.map(|e| self.check_expr(e, Mode::Infer)),
            },
            ExprKind::FieldAccess { expr, field } => ExprKind::FieldAccess {
                expr: self.check_expr(expr, Mode::Infer),
                field,
            },
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => ExprKind::MethodCall {
                receiver: self.check_expr(receiver, Mode::Infer),
                method,
                args: args
                    .into_iter()
                    .map(|a| self.check_expr(a, Mode::Infer))
                    .collect(),
            },
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: self.check_expr(lhs, Mode::Infer),
                rhs: self.check_expr(rhs, Mode::Infer),
            },
            ExprKind::Block { stmts, value } => ExprKind::Block {
                stmts: stmts.into_iter().map(|s| self.convert_stmt(s)).collect(),
                value: self.check_expr(value, Mode::Infer),
            },
            ExprKind::Case { scrutinee, arms } => ExprKind::Case {
                scrutinee: self.check_expr(scrutinee, Mode::Infer),
                arms: arms.into_iter().map(|a| self.convert_arm(a)).collect(),
            },
            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.check_expr(body, Mode::Infer),
            },
            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.check_expr(body, Mode::Infer),
                handlers: handlers
                    .into_iter()
                    .map(|h| self.convert_handler_arm(h))
                    .collect(),
            },
            ExprKind::Tuple(elements) => ExprKind::Tuple(
                elements
                    .into_iter()
                    .map(|e| self.check_expr(e, Mode::Infer))
                    .collect(),
            ),
            ExprKind::List(elements) => ExprKind::List(
                elements
                    .into_iter()
                    .map(|e| self.check_expr(e, Mode::Infer))
                    .collect(),
            ),
            ExprKind::Error => ExprKind::Error,
        }
    }

    /// Convert a ResolvedRef to a TypedRef.
    fn convert_ref(&mut self, resolved: ResolvedRef<'db>) -> TypedRef<'db> {
        let ty = self.infer_var(&resolved);
        TypedRef { resolved, ty }
    }

    /// Convert a statement.
    fn convert_stmt(&mut self, stmt: Stmt<ResolvedRef<'db>>) -> Stmt<TypedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                value,
                ty,
            } => {
                let value = self.check_expr(value, Mode::Infer);
                let pattern = self.convert_pattern(pattern);
                Stmt::Let {
                    id,
                    pattern,
                    value,
                    ty,
                }
            }
            Stmt::Expr { id, expr } => {
                let expr = self.check_expr(expr, Mode::Infer);
                Stmt::Expr { id, expr }
            }
        }
    }

    /// Convert a pattern.
    fn convert_pattern(&mut self, pattern: Pattern<ResolvedRef<'db>>) -> Pattern<TypedRef<'db>> {
        let kind = match *pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name } => PatternKind::Bind { name },
            PatternKind::Literal(lit) => PatternKind::Literal(lit),
            PatternKind::Variant { ctor, fields } => PatternKind::Variant {
                ctor: self.convert_ref(ctor),
                fields: fields
                    .into_iter()
                    .map(|p| self.convert_pattern(p))
                    .collect(),
            },
            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => PatternKind::Record {
                type_name: type_name.map(|t| self.convert_ref(t)),
                fields: fields
                    .into_iter()
                    .map(|f| self.convert_field_pattern(f))
                    .collect(),
                rest,
            },
            PatternKind::Tuple(patterns) => PatternKind::Tuple(
                patterns
                    .into_iter()
                    .map(|p| self.convert_pattern(p))
                    .collect(),
            ),
            PatternKind::List(patterns) => PatternKind::List(
                patterns
                    .into_iter()
                    .map(|p| self.convert_pattern(p))
                    .collect(),
            ),
            PatternKind::ListRest { head, rest } => PatternKind::ListRest {
                head: head.into_iter().map(|p| self.convert_pattern(p)).collect(),
                rest,
            },
            PatternKind::Or(patterns) => PatternKind::Or(
                patterns
                    .into_iter()
                    .map(|p| self.convert_pattern(p))
                    .collect(),
            ),
            PatternKind::As { pattern, name } => PatternKind::As {
                pattern: self.convert_pattern(pattern),
                name,
            },
            PatternKind::Error => PatternKind::Error,
        };
        Pattern::new(pattern.id, kind)
    }

    /// Convert a field pattern.
    fn convert_field_pattern(
        &mut self,
        fp: FieldPattern<ResolvedRef<'db>>,
    ) -> FieldPattern<TypedRef<'db>> {
        FieldPattern {
            name: fp.name,
            pattern: fp.pattern.map(|p| self.convert_pattern(p)),
        }
    }

    /// Convert a case arm.
    fn convert_arm(&mut self, arm: Arm<ResolvedRef<'db>>) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: self.convert_pattern(arm.pattern),
            guard: arm.guard.map(|g| self.check_expr(g, Mode::Infer)),
            body: self.check_expr(arm.body, Mode::Infer),
        }
    }

    /// Convert a handler arm.
    fn convert_handler_arm(
        &mut self,
        arm: HandlerArm<ResolvedRef<'db>>,
    ) -> HandlerArm<TypedRef<'db>> {
        let kind = match arm.kind {
            HandlerKind::Result { binding } => HandlerKind::Result {
                binding: self.convert_pattern(binding),
            },
            HandlerKind::Effect {
                ability,
                op,
                params,
                continuation,
            } => HandlerKind::Effect {
                ability: self.convert_ref(ability),
                op,
                params: params
                    .into_iter()
                    .map(|p| self.convert_pattern(p))
                    .collect(),
                continuation,
            },
        };
        HandlerArm {
            id: arm.id,
            kind,
            body: self.check_expr(arm.body, Mode::Infer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOpKind, LocalId, NodeId};
    use trunk_ir::Symbol;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    fn fresh_node_id() -> NodeId {
        NodeId::from_raw(1)
    }

    #[test]
    fn test_type_checker_creation() {
        let db = test_db();
        let _checker = TypeChecker::new(&db);
    }

    #[test]
    fn test_infer_nat_literal() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create a NatLit expression
        let expr = Expr::new(fresh_node_id(), ExprKind::NatLit(42));

        // Check the expression
        let typed_expr = checker.check_expr(expr, Mode::Infer);

        // Should produce a typed expression
        assert!(matches!(*typed_expr.kind, ExprKind::NatLit(42)));
    }

    #[test]
    fn test_infer_int_literal() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create an IntLit expression
        let expr = Expr::new(fresh_node_id(), ExprKind::IntLit(-10));

        let typed_expr = checker.check_expr(expr, Mode::Infer);
        assert!(matches!(*typed_expr.kind, ExprKind::IntLit(-10)));
    }

    #[test]
    fn test_infer_float_literal() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create a FloatLit expression
        let f = crate::ast::FloatBits::new(1.5);
        let expr = Expr::new(fresh_node_id(), ExprKind::FloatLit(f));

        let typed_expr = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::FloatLit(result_f) = &*typed_expr.kind {
            assert!((result_f.value() - 1.5).abs() < 0.001);
        } else {
            panic!("Expected FloatLit");
        }
    }

    #[test]
    fn test_infer_bool_literal() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let expr_true = Expr::new(fresh_node_id(), ExprKind::BoolLit(true));
        let typed = checker.check_expr(expr_true, Mode::Infer);
        assert!(matches!(*typed.kind, ExprKind::BoolLit(true)));
    }

    #[test]
    fn test_infer_string_literal() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let expr = Expr::new(fresh_node_id(), ExprKind::StringLit("hello".to_string()));
        let typed = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::StringLit(s) = &*typed.kind {
            assert_eq!(s, "hello");
        } else {
            panic!("Expected StringLit");
        }
    }

    #[test]
    fn test_infer_nil() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let expr = Expr::new(fresh_node_id(), ExprKind::Nil);
        let typed = checker.check_expr(expr, Mode::Infer);
        assert!(matches!(*typed.kind, ExprKind::Nil));
    }

    #[test]
    fn test_infer_binop_add() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: 1 + 2
        let lhs = Expr::new(fresh_node_id(), ExprKind::NatLit(1));
        let rhs = Expr::new(fresh_node_id(), ExprKind::NatLit(2));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::BinOp {
                op: BinOpKind::Add,
                lhs,
                rhs,
            },
        );

        let typed = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::BinOp { op, .. } = &*typed.kind {
            assert_eq!(*op, BinOpKind::Add);
        } else {
            panic!("Expected BinOp");
        }
    }

    #[test]
    fn test_infer_binop_comparison() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: 1 < 2
        let lhs = Expr::new(fresh_node_id(), ExprKind::NatLit(1));
        let rhs = Expr::new(fresh_node_id(), ExprKind::NatLit(2));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::BinOp {
                op: BinOpKind::Lt,
                lhs,
                rhs,
            },
        );

        let typed = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::BinOp { op, .. } = &*typed.kind {
            assert_eq!(*op, BinOpKind::Lt);
        } else {
            panic!("Expected BinOp");
        }
    }

    #[test]
    fn test_infer_tuple() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: (1, true, "hi")
        let elements = vec![
            Expr::new(fresh_node_id(), ExprKind::NatLit(1)),
            Expr::new(fresh_node_id(), ExprKind::BoolLit(true)),
            Expr::new(fresh_node_id(), ExprKind::StringLit("hi".to_string())),
        ];
        let expr = Expr::new(fresh_node_id(), ExprKind::Tuple(elements));

        let typed = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::Tuple(elems) = &*typed.kind {
            assert_eq!(elems.len(), 3);
        } else {
            panic!("Expected Tuple");
        }
    }

    #[test]
    fn test_infer_list() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: [1, 2, 3]
        let elements = vec![
            Expr::new(fresh_node_id(), ExprKind::NatLit(1)),
            Expr::new(fresh_node_id(), ExprKind::NatLit(2)),
            Expr::new(fresh_node_id(), ExprKind::NatLit(3)),
        ];
        let expr = Expr::new(fresh_node_id(), ExprKind::List(elements));

        let typed = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::List(elems) = &*typed.kind {
            assert_eq!(elems.len(), 3);
        } else {
            panic!("Expected List");
        }
    }

    #[test]
    fn test_infer_local_var() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Bind a local variable with a type
        let local_id = LocalId::new(0);
        let int_ty = Type::new(&db, TypeKind::Int);
        checker.ctx.bind_local(local_id, int_ty);

        // Create a Var expression referencing the local
        let resolved_ref = ResolvedRef::Local {
            id: local_id,
            name: Symbol::new("x"),
        };
        let expr = Expr::new(fresh_node_id(), ExprKind::Var(resolved_ref));

        let typed = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::Var(typed_ref) = &*typed.kind {
            // The type should be Int
            assert_eq!(*typed_ref.ty.kind(&db), TypeKind::Int);
        } else {
            panic!("Expected Var");
        }
    }

    #[test]
    fn test_infer_block() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: { 42 }
        let value = Expr::new(fresh_node_id(), ExprKind::NatLit(42));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::Block {
                stmts: vec![],
                value,
            },
        );

        let typed = checker.check_expr(expr, Mode::Infer);
        if let ExprKind::Block { value, .. } = &*typed.kind {
            assert!(matches!(*value.kind, ExprKind::NatLit(42)));
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_infer_lambda() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: |x| x (identity function)
        let param = crate::ast::Param {
            id: fresh_node_id(),
            name: Symbol::new("x"),
            ty: None,
        };
        let body = Expr::new(
            fresh_node_id(),
            ExprKind::Var(ResolvedRef::Local {
                id: LocalId::new(0),
                name: Symbol::new("x"),
            }),
        );
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::Lambda {
                params: vec![param],
                body,
            },
        );

        let typed = checker.check_expr(expr, Mode::Infer);
        assert!(matches!(*typed.kind, ExprKind::Lambda { .. }));
    }
}
