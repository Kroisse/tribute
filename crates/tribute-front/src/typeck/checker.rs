//! Type checker implementation.
//!
//! Performs bidirectional type checking on the AST, transforming
//! `Module<ResolvedRef<'db>>` into `Module<TypedRef<'db>>`.

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Span;

use crate::ast::{
    Arm, BuiltinRef, Decl, EffectRow, EnumDecl, Expr, ExprKind, FieldPattern, FuncDecl, FuncDefId,
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

        // Solve constraints and emit diagnostics for errors
        if let Err(error) = solver.solve(constraints) {
            // TODO: Extract proper span from the types involved in the error
            let span = Span::new(0, 0);
            Diagnostic {
                message: format!("Type error: {:?}", error),
                span,
                severity: DiagnosticSeverity::Error,
                phase: CompilationPhase::TypeChecking,
            }
            .accumulate(self.db());
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
                Decl::Ability(_) | Decl::Use(_) => {
                    // Abilities and imports don't define types directly
                }
                Decl::Module(m) => {
                    // For inline modules, recursively collect from nested declarations
                    if let Some(body) = &m.body {
                        // Create a temporary module to reuse collect_declarations
                        let inner_module = Module {
                            id: m.id,
                            name: Some(m.name),
                            decls: body.clone(),
                        };
                        self.collect_declarations(&inner_module);
                    }
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

        // Build parameter types from annotations when available
        let param_types: Vec<Type<'db>> = func
            .params
            .iter()
            .map(|p| match &p.ty {
                Some(ann) => self.annotation_to_type(ann),
                None => self.ctx.fresh_type_var(),
            })
            .collect();

        // Build return type from annotation when present
        let return_ty = func
            .return_ty
            .as_ref()
            .map(|ann| self.annotation_to_type(ann))
            .unwrap_or_else(|| self.ctx.fresh_type_var());

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

        // Register the function with its FuncDefId
        let func_id = FuncDefId::new(self.db(), func.name);
        self.ctx.register_function(func_id, scheme);
    }

    /// Convert a type annotation to a Type.
    fn annotation_to_type(&mut self, ann: &crate::ast::TypeAnnotation) -> Type<'db> {
        use crate::ast::TypeAnnotationKind;

        match &ann.kind {
            TypeAnnotationKind::Named(name) => {
                // Map well-known type names
                if *name == "Int" {
                    self.ctx.int_type()
                } else if *name == "Nat" {
                    self.ctx.nat_type()
                } else if *name == "Float" {
                    self.ctx.float_type()
                } else if *name == "Bool" {
                    self.ctx.bool_type()
                } else if *name == "String" {
                    self.ctx.string_type()
                } else if *name == "Bytes" {
                    self.ctx.bytes_type()
                } else if *name == "Rune" {
                    self.ctx.rune_type()
                } else if *name == "()" {
                    self.ctx.nil_type()
                } else {
                    // Named type with no args
                    self.ctx.named_type(*name, vec![])
                }
            }
            TypeAnnotationKind::Path(parts) if !parts.is_empty() => {
                // Use the last part as the type name
                if let Some(&name) = parts.last() {
                    self.ctx.named_type(name, vec![])
                } else {
                    self.ctx.error_type()
                }
            }
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type(ctor);
                // For now, extract the name from the constructor type
                if let TypeKind::Named { name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> =
                        args.iter().map(|a| self.annotation_to_type(a)).collect();
                    self.ctx.named_type(*name, arg_types)
                } else {
                    self.ctx.error_type()
                }
            }
            TypeAnnotationKind::Func { params, result } => {
                let param_types: Vec<Type<'db>> =
                    params.iter().map(|p| self.annotation_to_type(p)).collect();
                let result_ty = self.annotation_to_type(result);
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> =
                    elems.iter().map(|e| self.annotation_to_type(e)).collect();
                self.ctx.tuple_type(elem_types)
            }
            TypeAnnotationKind::WithEffects { inner, effects: _ } => {
                // TODO: Proper effect handling
                self.annotation_to_type(inner)
            }
            TypeAnnotationKind::Infer => self.ctx.fresh_type_var(),
            TypeAnnotationKind::Path(_) | TypeAnnotationKind::Error => self.ctx.error_type(),
        }
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
            Decl::Use(u) => Decl::Use(self.check_use_decl(u)),
            Decl::Module(m) => Decl::Module(self.check_module_decl(m)),
        }
    }

    /// Type check a module declaration.
    fn check_module_decl(
        &mut self,
        module: crate::ast::ModuleDecl<ResolvedRef<'db>>,
    ) -> crate::ast::ModuleDecl<TypedRef<'db>> {
        let body = module
            .body
            .map(|decls| decls.into_iter().map(|d| self.check_decl(d)).collect());

        crate::ast::ModuleDecl {
            id: module.id,
            name: module.name,
            is_pub: module.is_pub,
            body,
        }
    }

    /// Type check a function declaration.
    fn check_func_decl(&mut self, func: FuncDecl<ResolvedRef<'db>>) -> FuncDecl<TypedRef<'db>> {
        // Bind parameter types to the context using the registered type scheme
        // This ensures annotated parameter types are used instead of fresh variables
        let func_id = FuncDefId::new(self.db(), func.name);

        // Get function type info from scheme
        let func_type_info: Option<(Vec<Type<'db>>, Type<'db>)> =
            self.ctx.lookup_function(func_id).and_then(|scheme| {
                let func_ty = self.ctx.instantiate_scheme(scheme);
                if let TypeKind::Func { params, result, .. } = func_ty.kind(self.db()) {
                    Some((params.clone(), *result))
                } else {
                    None
                }
            });

        let param_types_from_scheme = func_type_info.as_ref().map(|(params, _)| params.clone());

        // Bind parameters: use scheme types if available, otherwise fresh vars.
        // Bind by LocalId when present for precision under shadowing,
        // and also bind by name to preserve name-based lookup.
        for (i, param) in func.params.iter().enumerate() {
            let ty = param_types_from_scheme
                .as_ref()
                .and_then(|types| types.get(i).copied())
                .unwrap_or_else(|| self.ctx.fresh_type_var());
            if let Some(local_id) = param.local_id {
                self.ctx.bind_local(local_id, ty);
            }
            self.ctx.bind_local_by_name(param.name, ty);
        }

        // Determine expected return type:
        // 1. From type scheme (if available)
        // 2. From return type annotation (if available)
        // 3. Fresh type variable (otherwise)
        let expected_return = func_type_info
            .map(|(_, result)| result)
            .or_else(|| {
                func.return_ty
                    .as_ref()
                    .map(|ann| self.annotation_to_type(ann))
            })
            .unwrap_or_else(|| self.ctx.fresh_type_var());

        // Check body against expected return type
        let body = self.check_expr(func.body, Mode::Check(expected_return));

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
            ExprKind::RuneLit(_) => self.ctx.rune_type(),

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
                // Lambda expression: infer param types and body type
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| {
                        let ty = match &p.ty {
                            Some(ann) => self.annotation_to_type(ann),
                            None => self.ctx.fresh_type_var(),
                        };
                        // Bind by LocalId when present for precision under shadowing
                        if let Some(local_id) = p.local_id {
                            self.ctx.bind_local(local_id, ty);
                        }
                        self.ctx.bind_local_by_name(p.name, ty);
                        ty
                    })
                    .collect();

                let body_ty = self.infer_expr_type(body);
                // Create fresh result type variable and unify with body type
                // This ensures proper type propagation through the solver
                let result_ty = self.ctx.fresh_type_var();
                self.ctx.constrain_eq(result_ty, body_ty);
                let effect = self.ctx.fresh_effect_row();
                self.ctx.func_type(param_types, result_ty, effect)
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

        // Convert to TypedRef, using precomputed type for Var/Cons to avoid
        // re-instantiation of type schemes (which would create new type variables)
        let kind = match *expr.kind {
            ExprKind::Var(resolved) => ExprKind::Var(self.convert_ref_with_type(resolved, ty)),
            ExprKind::Cons { ctor, args } => {
                // For Cons, we need to get the constructor type, not the result type
                let ctor_ty = self.infer_var(&ctor);
                ExprKind::Cons {
                    ctor: self.convert_ref_with_type(ctor, ctor_ty),
                    args: args
                        .into_iter()
                        .map(|a| self.check_expr(a, Mode::Infer))
                        .collect(),
                }
            }
            other => self.convert_expr_kind(other),
        };
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
            ExprKind::RuneLit(_) => self.ctx.rune_type(),
            _ => self.ctx.fresh_type_var(),
        }
    }

    /// Infer the type of a variable reference.
    fn infer_var(&mut self, resolved: &ResolvedRef<'db>) -> Type<'db> {
        match resolved {
            ResolvedRef::Local { id, name } => {
                // Try LocalId first, then fall back to name-based lookup
                // (needed for function parameters which may not have LocalId in TypeContext)
                self.ctx
                    .lookup_local(*id)
                    .or_else(|| self.ctx.lookup_local_by_name(*name))
                    .unwrap_or_else(|| self.ctx.fresh_type_var())
            }
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
            ExprKind::RuneLit(c) => ExprKind::RuneLit(c),
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

    /// Convert a ResolvedRef to a TypedRef with a precomputed type.
    ///
    /// Use this when the type has already been computed to avoid re-instantiation
    /// of type schemes which would create fresh type variables.
    fn convert_ref_with_type(&self, resolved: ResolvedRef<'db>, ty: Type<'db>) -> TypedRef<'db> {
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
                let value = if let Some(ann) = &ty {
                    let expected = self.annotation_to_type(ann);
                    self.check_expr(value, Mode::Check(expected))
                } else {
                    self.check_expr(value, Mode::Infer)
                };
                // Get the type of the value expression
                let value_ty = self
                    .ctx
                    .get_node_type(value.id)
                    .unwrap_or_else(|| self.ctx.fresh_type_var());

                // Bind pattern variables with the value type
                self.bind_pattern_vars(&pattern, value_ty);

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

    /// Bind pattern variables to the given type.
    fn bind_pattern_vars(&mut self, pattern: &Pattern<ResolvedRef<'db>>, ty: Type<'db>) {
        match &*pattern.kind {
            PatternKind::Bind { name, local_id } => {
                if let Some(id) = local_id {
                    self.ctx.bind_local(*id, ty);
                }
                self.ctx.bind_local_by_name(*name, ty);
            }
            PatternKind::Tuple(pats) => {
                // Destructure tuple type
                if let TypeKind::Tuple(elem_tys) = ty.kind(self.db()) {
                    for (pat, elem_ty) in pats.iter().zip(elem_tys.iter()) {
                        self.bind_pattern_vars(pat, *elem_ty);
                    }
                } else {
                    // Type mismatch - bind with fresh vars
                    let fresh_vars: Vec<_> =
                        pats.iter().map(|_| self.ctx.fresh_type_var()).collect();
                    for (pat, fresh_ty) in pats.iter().zip(fresh_vars) {
                        self.bind_pattern_vars(pat, fresh_ty);
                    }
                }
            }
            PatternKind::Variant { fields, .. } => {
                // For variants, bind each field with a fresh type var
                let fresh_vars: Vec<_> = fields.iter().map(|_| self.ctx.fresh_type_var()).collect();
                for (field, fresh_ty) in fields.iter().zip(fresh_vars) {
                    self.bind_pattern_vars(field, fresh_ty);
                }
            }
            PatternKind::Record { fields, .. } => {
                // For records, bind each field with a fresh type var
                for field in fields {
                    if let Some(pat) = &field.pattern {
                        let fresh_ty = self.ctx.fresh_type_var();
                        self.bind_pattern_vars(pat, fresh_ty);
                    }
                }
            }
            PatternKind::List(pats) => {
                // For lists, bind each element with a fresh type var
                let fresh_vars: Vec<_> = pats.iter().map(|_| self.ctx.fresh_type_var()).collect();
                for (pat, fresh_ty) in pats.iter().zip(fresh_vars) {
                    self.bind_pattern_vars(pat, fresh_ty);
                }
            }
            PatternKind::ListRest {
                head,
                rest_local_id,
                ..
            } => {
                // Bind head elements
                let fresh_vars: Vec<_> = head.iter().map(|_| self.ctx.fresh_type_var()).collect();
                for (pat, fresh_ty) in head.iter().zip(fresh_vars) {
                    self.bind_pattern_vars(pat, fresh_ty);
                }
                // Bind rest element if it has a LocalId
                if let Some(local_id) = rest_local_id {
                    // Rest is a list of the same element type
                    let rest_ty = self.ctx.fresh_type_var();
                    self.ctx.bind_local(*local_id, rest_ty);
                }
            }
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => {
                // Bind by name for backwards compatibility
                self.ctx.bind_local_by_name(*name, ty);
                // Also bind by LocalId for precise lookups
                if let Some(local_id) = local_id {
                    self.ctx.bind_local(*local_id, ty);
                }
                self.bind_pattern_vars(pattern, ty);
            }
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {}
        }
    }

    /// Convert a pattern.
    fn convert_pattern(&mut self, pattern: Pattern<ResolvedRef<'db>>) -> Pattern<TypedRef<'db>> {
        let kind = match *pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,
            PatternKind::Bind { name, local_id } => PatternKind::Bind { name, local_id },
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
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => PatternKind::ListRest {
                head: head.into_iter().map(|p| self.convert_pattern(p)).collect(),
                rest,
                rest_local_id,
            },
            PatternKind::As {
                pattern,
                name,
                local_id,
            } => PatternKind::As {
                pattern: self.convert_pattern(pattern),
                name,
                local_id,
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
            id: fp.id,
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
    use crate::ast::{BinOpKind, LocalId, NodeId, TypeAnnotation, TypeAnnotationKind};
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
            local_id: Some(LocalId::new(0)),
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

    #[test]
    fn test_annotation_to_type_int() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("Int")),
        };

        let ty = checker.annotation_to_type(&ann);
        assert!(matches!(*ty.kind(&db), TypeKind::Int));
    }

    #[test]
    fn test_annotation_to_type_bool() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("Bool")),
        };

        let ty = checker.annotation_to_type(&ann);
        assert!(matches!(*ty.kind(&db), TypeKind::Bool));
    }

    #[test]
    fn test_annotation_to_type_string() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("String")),
        };

        let ty = checker.annotation_to_type(&ann);
        assert!(matches!(*ty.kind(&db), TypeKind::String));
    }

    #[test]
    fn test_annotation_to_type_named() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("Option")),
        };

        let ty = checker.annotation_to_type(&ann);
        if let TypeKind::Named { name, args } = ty.kind(&db) {
            assert_eq!(*name, Symbol::new("Option"));
            assert!(args.is_empty());
        } else {
            panic!("Expected Named type");
        }
    }

    #[test]
    fn test_annotation_to_type_tuple() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // (Int, Bool)
        let ann = crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: crate::ast::TypeAnnotationKind::Tuple(vec![
                crate::ast::TypeAnnotation {
                    id: fresh_node_id(),
                    kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("Int")),
                },
                crate::ast::TypeAnnotation {
                    id: fresh_node_id(),
                    kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("Bool")),
                },
            ]),
        };

        let ty = checker.annotation_to_type(&ann);
        if let TypeKind::Tuple(elems) = ty.kind(&db) {
            assert_eq!(elems.len(), 2);
        } else {
            panic!("Expected Tuple type");
        }
    }

    #[test]
    fn test_annotation_to_type_infer() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: crate::ast::TypeAnnotationKind::Infer,
        };

        let ty = checker.annotation_to_type(&ann);
        // Should be a fresh UniVar
        assert!(matches!(*ty.kind(&db), TypeKind::UniVar { .. }));
    }

    #[test]
    fn test_bind_pattern_vars_simple() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let local_id = LocalId::new(0);
        let int_ty = Type::new(&db, TypeKind::Int);

        let pattern = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name: Symbol::new("x"),
                local_id: Some(local_id),
            },
        );

        checker.bind_pattern_vars(&pattern, int_ty);

        // The local should be bound
        assert!(checker.ctx.lookup_local(local_id).is_some());
        assert_eq!(
            *checker.ctx.lookup_local(local_id).unwrap().kind(&db),
            TypeKind::Int
        );
    }

    #[test]
    fn test_bind_pattern_vars_wildcard() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        let pattern = Pattern::new(fresh_node_id(), PatternKind::Wildcard);

        // Should not panic
        checker.bind_pattern_vars(&pattern, int_ty);
    }

    #[test]
    fn test_lambda_binds_params() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: |x: Int| x
        let param = crate::ast::Param {
            id: fresh_node_id(),
            name: Symbol::new("x"),
            ty: Some(crate::ast::TypeAnnotation {
                id: fresh_node_id(),
                kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("Int")),
            }),
            local_id: Some(LocalId::new(0)),
        };
        let body = Expr::new(fresh_node_id(), ExprKind::NatLit(42));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::Lambda {
                params: vec![param],
                body,
            },
        );

        let typed = checker.check_expr(expr, Mode::Infer);

        // Should produce a function type
        if let ExprKind::Lambda { .. } = &*typed.kind {
            // After lambda type check, x should be bound by name
            let x_ty = checker.ctx.lookup_local_by_name(Symbol::new("x"));
            assert!(x_ty.is_some());
        } else {
            panic!("Expected Lambda");
        }
    }

    // =========================================================================
    // Rune Type Tests
    // =========================================================================

    #[test]
    fn test_infer_rune_literal() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create a RuneLit expression
        let expr = Expr::new(fresh_node_id(), ExprKind::RuneLit('a'));

        let typed_expr = checker.check_expr(expr, Mode::Infer);
        assert!(matches!(*typed_expr.kind, ExprKind::RuneLit('a')));
    }

    #[test]
    fn test_annotation_to_type_rune() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: crate::ast::TypeAnnotationKind::Named(Symbol::new("Rune")),
        };

        let ty = checker.annotation_to_type(&ann);
        assert!(
            matches!(*ty.kind(&db), TypeKind::Rune),
            "Rune annotation should produce Rune type, got {:?}",
            ty.kind(&db)
        );
    }

    #[test]
    fn test_rune_literal_unifies_with_rune_annotation() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create a RuneLit expression and check against Rune type annotation
        let expr = Expr::new(fresh_node_id(), ExprKind::RuneLit('x'));
        let expected_ty = checker.ctx.rune_type();

        let typed_expr = checker.check_expr(expr, Mode::Check(expected_ty));

        // Should succeed without type error (no panic during unification)
        assert!(matches!(*typed_expr.kind, ExprKind::RuneLit('x')));
    }

    // =========================================================================
    // Let Statement Type Annotation Tests
    // =========================================================================

    #[test]
    fn test_let_with_annotation_checks_value() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: let x: Nat = 42
        let pattern = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name: Symbol::new("x"),
                local_id: Some(LocalId::new(0)),
            },
        );
        let value = Expr::new(fresh_node_id(), ExprKind::NatLit(42));
        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("Nat")),
        };
        let stmt = Stmt::Let {
            id: fresh_node_id(),
            pattern,
            ty: Some(ann),
            value,
        };

        // Should succeed: NatLit matches Nat annotation
        let typed_stmt = checker.convert_stmt(stmt);
        if let Stmt::Let { value, .. } = &typed_stmt {
            assert!(matches!(*value.kind, ExprKind::NatLit(42)));
        } else {
            panic!("Expected Let statement");
        }
    }

    #[test]
    fn test_let_with_mismatched_annotation_adds_constraint() {
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Create: let x: Bool = 42
        // The annotation says Bool but the value is a Nat literal.
        let pattern = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name: Symbol::new("x"),
                local_id: Some(LocalId::new(0)),
            },
        );
        let value = Expr::new(fresh_node_id(), ExprKind::NatLit(42));
        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("Bool")),
        };
        let stmt = Stmt::Let {
            id: fresh_node_id(),
            pattern,
            ty: Some(ann),
            value,
        };

        // This should add an equality constraint (Bool = Nat) that will fail when solved.
        let _typed_stmt = checker.convert_stmt(stmt);

        // Verify a constraint was emitted by solving — it should produce an error
        let constraints = checker.ctx.take_constraints();
        assert!(
            !constraints.is_empty(),
            "Expected at least one constraint from type mismatch"
        );

        let mut solver = TypeSolver::new(&db);
        let result = solver.solve(constraints);
        assert!(
            result.is_err(),
            "Expected type error from Bool vs Nat mismatch"
        );
    }
}
