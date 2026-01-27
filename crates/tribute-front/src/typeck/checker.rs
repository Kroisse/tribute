//! Type checker implementation.
//!
//! Performs bidirectional type checking on the AST, transforming
//! `Module<ResolvedRef<'db>>` into `Module<TypedRef<'db>>`.

use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::{Span, Symbol};

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
    pub fn check_module(self, module: Module<ResolvedRef<'db>>) -> Module<TypedRef<'db>> {
        self.check_module_inner(module).0
    }

    /// Type check a module and also return function type schemes.
    pub fn check_module_with_types(
        self,
        module: Module<ResolvedRef<'db>>,
    ) -> (Module<TypedRef<'db>>, Vec<(Symbol, TypeScheme<'db>)>) {
        self.check_module_inner(module)
    }

    /// Internal implementation for module type checking.
    fn check_module_inner(
        mut self,
        module: Module<ResolvedRef<'db>>,
    ) -> (Module<TypedRef<'db>>, Vec<(Symbol, TypeScheme<'db>)>) {
        // Phase 1: Collect type definitions and function signatures
        self.collect_declarations(&module);

        // Phase 2: Type check all declarations and generate constraints
        let decls: Vec<Decl<TypedRef<'db>>> = module
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

        // Export function type schemes before applying substitution
        let function_types = self.ctx.export_function_types();

        // Phase 4: Apply substitution to produce final types
        let type_subst = solver.type_subst();
        let row_subst = solver.row_subst();

        let decls = decls
            .into_iter()
            .map(|d| apply_subst_to_decl(self.db(), d, type_subst, row_subst))
            .collect();

        let typed_module = Module {
            id: module.id,
            name: module.name,
            decls,
        };

        (typed_module, function_types)
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

        // Build effect row from annotations
        let effect = match &func.effects {
            Some(anns) => {
                let row_var = self.ctx.fresh_row_var();
                crate::ast::abilities_to_effect_row(
                    self.db(),
                    anns,
                    &mut |ann| self.annotation_to_type(ann),
                    || row_var,
                )
            }
            None => EffectRow::pure(self.db()),
        };

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
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                let param_types: Vec<Type<'db>> =
                    params.iter().map(|p| self.annotation_to_type(p)).collect();
                let result_ty = self.annotation_to_type(result);
                let row_var = self.ctx.fresh_row_var();
                let effect = crate::ast::abilities_to_effect_row(
                    self.db(),
                    abilities,
                    &mut |ann| self.annotation_to_type(ann),
                    || row_var,
                );
                self.ctx.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> =
                    elems.iter().map(|e| self.annotation_to_type(e)).collect();
                self.ctx.tuple_type(elem_types)
            }
            TypeAnnotationKind::Infer => self.ctx.fresh_type_var(),
            TypeAnnotationKind::Path(_) | TypeAnnotationKind::Error => self.ctx.error_type(),
        }
    }

    /// Convert a type annotation to a Type, resolving type parameter names
    /// to `BoundVar` indices using the provided lookup table.
    ///
    /// This is used when building constructor types for enum variants, where
    /// type parameters in field annotations must map to `BoundVar` rather than
    /// fresh unification variables.
    fn annotation_to_type_with_bound_vars(
        &mut self,
        ann: &crate::ast::TypeAnnotation,
        type_param_indices: &[(trunk_ir::Symbol, u32)],
    ) -> Type<'db> {
        use crate::ast::TypeAnnotationKind;

        match &ann.kind {
            TypeAnnotationKind::Named(name) => {
                // Check if this name is a type parameter
                if let Some(&(_, index)) = type_param_indices.iter().find(|(n, _)| n == name) {
                    return Type::new(self.db(), TypeKind::BoundVar { index });
                }
                // Otherwise delegate to normal annotation_to_type
                self.annotation_to_type(ann)
            }
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type_with_bound_vars(ctor, type_param_indices);
                if let TypeKind::Named { name, .. } = ctor_ty.kind(self.db()) {
                    let arg_types: Vec<Type<'db>> = args
                        .iter()
                        .map(|a| self.annotation_to_type_with_bound_vars(a, type_param_indices))
                        .collect();
                    self.ctx.named_type(*name, arg_types)
                } else {
                    self.ctx.error_type()
                }
            }
            TypeAnnotationKind::Func {
                params,
                result,
                abilities,
            } => {
                let param_types: Vec<Type<'db>> = params
                    .iter()
                    .map(|p| self.annotation_to_type_with_bound_vars(p, type_param_indices))
                    .collect();
                let result_ty = self.annotation_to_type_with_bound_vars(result, type_param_indices);
                let row_var = self.ctx.fresh_row_var();
                let effect = crate::ast::abilities_to_effect_row(
                    self.db(),
                    abilities,
                    &mut |a| self.annotation_to_type_with_bound_vars(a, type_param_indices),
                    || row_var,
                );
                self.ctx.func_type(param_types, result_ty, effect)
            }
            TypeAnnotationKind::Tuple(elems) => {
                let elem_types: Vec<Type<'db>> = elems
                    .iter()
                    .map(|e| self.annotation_to_type_with_bound_vars(e, type_param_indices))
                    .collect();
                self.ctx.tuple_type(elem_types)
            }
            // For other kinds, delegate to normal annotation_to_type
            _ => self.annotation_to_type(ann),
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

        let scheme = TypeScheme::new(self.db(), type_params.clone(), enum_ty);
        self.ctx.register_type_def(name, scheme);

        // Register constructors for each variant
        // Build name → BoundVar index lookup for type parameter resolution
        let type_param_indices: Vec<(Symbol, u32)> = e
            .type_params
            .iter()
            .enumerate()
            .map(|(i, tp)| (tp.name, i as u32))
            .collect();

        for variant in &e.variants {
            let ctor_ty = if variant.fields.is_empty() {
                // Unit variant: constructor type is just the enum type
                enum_ty
            } else {
                // Field variant: constructor type is fn(field_types...) -> enum_ty
                let field_types: Vec<Type<'db>> = variant
                    .fields
                    .iter()
                    .map(|f| self.annotation_to_type_with_bound_vars(&f.ty, &type_param_indices))
                    .collect();
                let effect = EffectRow::pure(self.db());
                self.ctx.func_type(field_types, enum_ty, effect)
            };

            let ctor_scheme = TypeScheme::new(self.db(), type_params.clone(), ctor_ty);
            let ctor_id = crate::ast::CtorId::new(self.db(), variant.name);
            self.ctx.register_constructor(ctor_id, ctor_scheme);
        }
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
                // Infer callee and argument types
                let callee_ty = self.infer_expr_type(callee);
                let arg_types: Vec<Type<'db>> =
                    args.iter().map(|a| self.infer_expr_type(a)).collect();
                self.infer_call(callee_ty, &arg_types)
            }
            ExprKind::Cons { ctor, args } => {
                // Constructor application
                let ctor_ty = self.infer_var(ctor);
                let arg_types: Vec<Type<'db>> =
                    args.iter().map(|a| self.infer_expr_type(a)).collect();
                self.infer_call(ctor_ty, &arg_types)
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
    ///
    /// Creates fresh type variables for parameters and result, constrains the
    /// callee to be a function type, and constrains each argument type to match
    /// the corresponding parameter type variable.
    fn infer_call(&mut self, callee_ty: Type<'db>, arg_types: &[Type<'db>]) -> Type<'db> {
        // Create fresh type variables for parameters and result
        let param_types: Vec<Type<'db>> = (0..arg_types.len())
            .map(|_| self.ctx.fresh_type_var())
            .collect();
        let result_ty = self.ctx.fresh_type_var();
        let effect = self.ctx.fresh_effect_row();

        // Constrain callee to be a function type
        let expected_func_ty = self.ctx.func_type(param_types.clone(), result_ty, effect);
        self.ctx.constrain_eq(callee_ty, expected_func_ty);

        // Constrain each argument type to match the corresponding parameter
        for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
            self.ctx.constrain_eq(*param_ty, *arg_ty);
        }

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

// =========================================================================
// Substitution post-pass
// =========================================================================

use super::solver::{RowSubst, TypeSubst};

/// Apply a type reference substitution to a `TypedRef`.
fn apply_subst_to_ref<'db>(
    db: &'db dyn salsa::Database,
    r: TypedRef<'db>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> TypedRef<'db> {
    TypedRef {
        resolved: r.resolved,
        ty: type_subst.apply_with_rows(db, r.ty, row_subst),
    }
}

/// Apply substitution to a declaration.
fn apply_subst_to_decl<'db>(
    db: &'db dyn salsa::Database,
    decl: Decl<TypedRef<'db>>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> Decl<TypedRef<'db>> {
    match decl {
        Decl::Function(f) => Decl::Function(apply_subst_to_func(db, f, type_subst, row_subst)),
        Decl::Module(m) => {
            let body = m.body.map(|ds| {
                ds.into_iter()
                    .map(|d| apply_subst_to_decl(db, d, type_subst, row_subst))
                    .collect()
            });
            Decl::Module(crate::ast::ModuleDecl {
                id: m.id,
                name: m.name,
                is_pub: m.is_pub,
                body,
            })
        }
        // Struct, Enum, Ability, Use don't contain TypedRefs
        other => other,
    }
}

/// Apply substitution to a function declaration.
fn apply_subst_to_func<'db>(
    db: &'db dyn salsa::Database,
    f: FuncDecl<TypedRef<'db>>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> FuncDecl<TypedRef<'db>> {
    FuncDecl {
        id: f.id,
        is_pub: f.is_pub,
        name: f.name,
        type_params: f.type_params,
        params: f.params,
        return_ty: f.return_ty,
        effects: f.effects,
        body: apply_subst_to_expr(db, f.body, type_subst, row_subst),
    }
}

/// Apply substitution to an expression.
fn apply_subst_to_expr<'db>(
    db: &'db dyn salsa::Database,
    expr: Expr<TypedRef<'db>>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> Expr<TypedRef<'db>> {
    let kind = match *expr.kind {
        ExprKind::Var(r) => ExprKind::Var(apply_subst_to_ref(db, r, type_subst, row_subst)),
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: apply_subst_to_expr(db, callee, type_subst, row_subst),
            args: args
                .into_iter()
                .map(|a| apply_subst_to_expr(db, a, type_subst, row_subst))
                .collect(),
        },
        ExprKind::Cons { ctor, args } => ExprKind::Cons {
            ctor: apply_subst_to_ref(db, ctor, type_subst, row_subst),
            args: args
                .into_iter()
                .map(|a| apply_subst_to_expr(db, a, type_subst, row_subst))
                .collect(),
        },
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => ExprKind::Record {
            type_name: apply_subst_to_ref(db, type_name, type_subst, row_subst),
            fields: fields
                .into_iter()
                .map(|(name, e)| (name, apply_subst_to_expr(db, e, type_subst, row_subst)))
                .collect(),
            spread: spread.map(|e| apply_subst_to_expr(db, e, type_subst, row_subst)),
        },
        ExprKind::FieldAccess { expr: inner, field } => ExprKind::FieldAccess {
            expr: apply_subst_to_expr(db, inner, type_subst, row_subst),
            field,
        },
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => ExprKind::MethodCall {
            receiver: apply_subst_to_expr(db, receiver, type_subst, row_subst),
            method,
            args: args
                .into_iter()
                .map(|a| apply_subst_to_expr(db, a, type_subst, row_subst))
                .collect(),
        },
        ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
            op,
            lhs: apply_subst_to_expr(db, lhs, type_subst, row_subst),
            rhs: apply_subst_to_expr(db, rhs, type_subst, row_subst),
        },
        ExprKind::Block { stmts, value } => ExprKind::Block {
            stmts: stmts
                .into_iter()
                .map(|s| apply_subst_to_stmt(db, s, type_subst, row_subst))
                .collect(),
            value: apply_subst_to_expr(db, value, type_subst, row_subst),
        },
        ExprKind::Case { scrutinee, arms } => ExprKind::Case {
            scrutinee: apply_subst_to_expr(db, scrutinee, type_subst, row_subst),
            arms: arms
                .into_iter()
                .map(|a| apply_subst_to_arm(db, a, type_subst, row_subst))
                .collect(),
        },
        ExprKind::Lambda { params, body } => ExprKind::Lambda {
            params,
            body: apply_subst_to_expr(db, body, type_subst, row_subst),
        },
        ExprKind::Handle { body, handlers } => ExprKind::Handle {
            body: apply_subst_to_expr(db, body, type_subst, row_subst),
            handlers: handlers
                .into_iter()
                .map(|h| apply_subst_to_handler_arm(db, h, type_subst, row_subst))
                .collect(),
        },
        ExprKind::Tuple(elems) => ExprKind::Tuple(
            elems
                .into_iter()
                .map(|e| apply_subst_to_expr(db, e, type_subst, row_subst))
                .collect(),
        ),
        ExprKind::List(elems) => ExprKind::List(
            elems
                .into_iter()
                .map(|e| apply_subst_to_expr(db, e, type_subst, row_subst))
                .collect(),
        ),
        // Literals and Error don't contain TypedRefs
        kind @ (ExprKind::NatLit(_)
        | ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::Nil
        | ExprKind::RuneLit(_)
        | ExprKind::Error) => kind,
    };
    Expr::new(expr.id, kind)
}

/// Apply substitution to a statement.
fn apply_subst_to_stmt<'db>(
    db: &'db dyn salsa::Database,
    stmt: Stmt<TypedRef<'db>>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> Stmt<TypedRef<'db>> {
    match stmt {
        Stmt::Let {
            id,
            pattern,
            value,
            ty,
        } => Stmt::Let {
            id,
            pattern: apply_subst_to_pattern(db, pattern, type_subst, row_subst),
            value: apply_subst_to_expr(db, value, type_subst, row_subst),
            ty,
        },
        Stmt::Expr { id, expr } => Stmt::Expr {
            id,
            expr: apply_subst_to_expr(db, expr, type_subst, row_subst),
        },
    }
}

/// Apply substitution to a pattern.
fn apply_subst_to_pattern<'db>(
    db: &'db dyn salsa::Database,
    pattern: Pattern<TypedRef<'db>>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> Pattern<TypedRef<'db>> {
    let kind = match *pattern.kind {
        PatternKind::Variant { ctor, fields } => PatternKind::Variant {
            ctor: apply_subst_to_ref(db, ctor, type_subst, row_subst),
            fields: fields
                .into_iter()
                .map(|p| apply_subst_to_pattern(db, p, type_subst, row_subst))
                .collect(),
        },
        PatternKind::Record {
            type_name,
            fields,
            rest,
        } => PatternKind::Record {
            type_name: type_name.map(|t| apply_subst_to_ref(db, t, type_subst, row_subst)),
            fields: fields
                .into_iter()
                .map(|f| FieldPattern {
                    id: f.id,
                    name: f.name,
                    pattern: f
                        .pattern
                        .map(|p| apply_subst_to_pattern(db, p, type_subst, row_subst)),
                })
                .collect(),
            rest,
        },
        PatternKind::Tuple(pats) => PatternKind::Tuple(
            pats.into_iter()
                .map(|p| apply_subst_to_pattern(db, p, type_subst, row_subst))
                .collect(),
        ),
        PatternKind::List(pats) => PatternKind::List(
            pats.into_iter()
                .map(|p| apply_subst_to_pattern(db, p, type_subst, row_subst))
                .collect(),
        ),
        PatternKind::ListRest {
            head,
            rest,
            rest_local_id,
        } => PatternKind::ListRest {
            head: head
                .into_iter()
                .map(|p| apply_subst_to_pattern(db, p, type_subst, row_subst))
                .collect(),
            rest,
            rest_local_id,
        },
        PatternKind::As {
            pattern: inner,
            name,
            local_id,
        } => PatternKind::As {
            pattern: apply_subst_to_pattern(db, inner, type_subst, row_subst),
            name,
            local_id,
        },
        // Wildcard, Bind, Literal, Error don't contain TypedRefs
        kind @ (PatternKind::Wildcard
        | PatternKind::Bind { .. }
        | PatternKind::Literal(_)
        | PatternKind::Error) => kind,
    };
    Pattern::new(pattern.id, kind)
}

/// Apply substitution to a case arm.
fn apply_subst_to_arm<'db>(
    db: &'db dyn salsa::Database,
    arm: Arm<TypedRef<'db>>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> Arm<TypedRef<'db>> {
    Arm {
        id: arm.id,
        pattern: apply_subst_to_pattern(db, arm.pattern, type_subst, row_subst),
        guard: arm
            .guard
            .map(|g| apply_subst_to_expr(db, g, type_subst, row_subst)),
        body: apply_subst_to_expr(db, arm.body, type_subst, row_subst),
    }
}

/// Apply substitution to a handler arm.
fn apply_subst_to_handler_arm<'db>(
    db: &'db dyn salsa::Database,
    arm: HandlerArm<TypedRef<'db>>,
    type_subst: &TypeSubst<'db>,
    row_subst: &RowSubst<'db>,
) -> HandlerArm<TypedRef<'db>> {
    let kind = match arm.kind {
        HandlerKind::Result { binding } => HandlerKind::Result {
            binding: apply_subst_to_pattern(db, binding, type_subst, row_subst),
        },
        HandlerKind::Effect {
            ability,
            op,
            params,
            continuation,
        } => HandlerKind::Effect {
            ability: apply_subst_to_ref(db, ability, type_subst, row_subst),
            op,
            params: params
                .into_iter()
                .map(|p| apply_subst_to_pattern(db, p, type_subst, row_subst))
                .collect(),
            continuation,
        },
    };
    HandlerArm {
        id: arm.id,
        kind,
        body: apply_subst_to_expr(db, arm.body, type_subst, row_subst),
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

    // =========================================================================
    // Call argument type constraint tests
    // =========================================================================

    #[test]
    fn test_infer_call_constrains_arg_types() {
        // infer_call should constrain each argument type to the corresponding
        // parameter type variable, linking args to the function signature.
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // Simulate: callee has type fn(Int) -> Bool
        let int_ty = checker.ctx.int_type();
        let bool_ty = checker.ctx.bool_type();
        let callee_ty = checker
            .ctx
            .func_type(vec![int_ty], bool_ty, EffectRow::pure(checker.db()));

        // Call with arg type = Int (matching)
        let result_ty = checker.infer_call(callee_ty, &[int_ty]);

        // Solve constraints: should succeed
        let constraints = checker.ctx.take_constraints();
        let mut solver = TypeSolver::new(&db);
        assert!(
            solver.solve(constraints).is_ok(),
            "Expected no type error when arg matches parameter"
        );

        // The result should resolve to Bool
        let resolved = solver.type_subst().apply(&db, result_ty);
        assert!(
            matches!(*resolved.kind(&db), TypeKind::Bool),
            "Expected Bool result type, got {:?}",
            resolved.kind(&db)
        );
    }

    #[test]
    fn test_infer_call_detects_arg_type_mismatch() {
        // When the argument type doesn't match the parameter,
        // the solver should report a type error.
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        // callee: fn(Int) -> Bool
        let int_ty = checker.ctx.int_type();
        let bool_ty = checker.ctx.bool_type();
        let callee_ty = checker
            .ctx
            .func_type(vec![int_ty], bool_ty, EffectRow::pure(checker.db()));

        // Call with arg type = Bool (mismatched — expected Int)
        let _result_ty = checker.infer_call(callee_ty, &[bool_ty]);

        // Solve constraints: should fail
        let constraints = checker.ctx.take_constraints();
        let mut solver = TypeSolver::new(&db);
        assert!(
            solver.solve(constraints).is_err(),
            "Expected type error when arg Bool doesn't match parameter Int"
        );
    }

    // =========================================================================
    // Ability annotation → EffectRow tests
    // =========================================================================

    #[test]
    fn test_func_annotation_pure() {
        // fn(Int) ->{} Bool  →  EffectRow { effects: [], rest: None }
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Func {
                params: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                }],
                result: Box::new(TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Bool")),
                }),
                abilities: vec![], // empty = pure
            },
        };

        let ty = checker.annotation_to_type(&ann);
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = ty.kind(&db)
        {
            assert_eq!(params.len(), 1);
            assert!(matches!(*params[0].kind(&db), TypeKind::Int));
            assert!(matches!(*result.kind(&db), TypeKind::Bool));
            // Pure: no effects, closed row
            assert!(effect.effects(&db).is_empty());
            assert!(effect.rest(&db).is_none());
        } else {
            panic!("Expected Func type, got {:?}", ty.kind(&db));
        }
    }

    #[test]
    fn test_func_annotation_with_ability() {
        // fn(Int) ->{State} Bool  →  EffectRow { effects: [State], rest: None }
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Func {
                params: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                }],
                result: Box::new(TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Bool")),
                }),
                abilities: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("State")),
                }],
            },
        };

        let ty = checker.annotation_to_type(&ann);
        if let TypeKind::Func { effect, .. } = ty.kind(&db) {
            let effects = effect.effects(&db);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].name, Symbol::new("State"));
            assert!(effects[0].args.is_empty());
            // Closed row (no lowercase / Infer)
            assert!(effect.rest(&db).is_none());
        } else {
            panic!("Expected Func type, got {:?}", ty.kind(&db));
        }
    }

    #[test]
    fn test_func_annotation_effect_polymorphic() {
        // fn(Int) ->{State, e} Bool  →  EffectRow { effects: [State], rest: Some(_) }
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Func {
                params: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                }],
                result: Box::new(TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Bool")),
                }),
                abilities: vec![
                    TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("State")),
                    },
                    TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("e")), // lowercase = row variable
                    },
                ],
            },
        };

        let ty = checker.annotation_to_type(&ann);
        if let TypeKind::Func { effect, .. } = ty.kind(&db) {
            let effects = effect.effects(&db);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].name, Symbol::new("State"));
            // Open row: lowercase "e" triggers a row variable
            assert!(
                effect.rest(&db).is_some(),
                "Expected open effect row for lowercase type variable"
            );
        } else {
            panic!("Expected Func type, got {:?}", ty.kind(&db));
        }
    }

    #[test]
    fn test_func_annotation_infer_effect() {
        // fn(Int) ->{_} Bool  →  EffectRow { effects: [], rest: Some(_) }
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Func {
                params: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                }],
                result: Box::new(TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Bool")),
                }),
                abilities: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Infer,
                }],
            },
        };

        let ty = checker.annotation_to_type(&ann);
        if let TypeKind::Func { effect, .. } = ty.kind(&db) {
            assert!(effect.effects(&db).is_empty());
            assert!(
                effect.rest(&db).is_some(),
                "Expected open effect row for Infer annotation"
            );
        } else {
            panic!("Expected Func type, got {:?}", ty.kind(&db));
        }
    }

    #[test]
    fn test_func_annotation_parameterized_ability() {
        // fn(Int) ->{State(Int)} Bool  →  EffectRow { effects: [State(Int)], rest: None }
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Func {
                params: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                }],
                result: Box::new(TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Bool")),
                }),
                abilities: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::App {
                        ctor: Box::new(TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("State")),
                        }),
                        args: vec![TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                        }],
                    },
                }],
            },
        };

        let ty = checker.annotation_to_type(&ann);
        if let TypeKind::Func { effect, .. } = ty.kind(&db) {
            let effects = effect.effects(&db);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].name, Symbol::new("State"));
            assert_eq!(effects[0].args.len(), 1);
            assert!(matches!(*effects[0].args[0].kind(&db), TypeKind::Int));
            assert!(effect.rest(&db).is_none());
        } else {
            panic!("Expected Func type, got {:?}", ty.kind(&db));
        }
    }

    // =========================================================================
    // Enum variant constructor registration tests
    // =========================================================================

    #[salsa::tracked]
    fn test_collect_enum_def_registers_constructors_inner(db: &dyn salsa::Database) -> bool {
        let mut checker = TypeChecker::new(db);

        let enum_decl = crate::ast::EnumDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("Color"),
            type_params: vec![],
            variants: vec![
                crate::ast::VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Red"),
                    fields: vec![],
                },
                crate::ast::VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Green"),
                    fields: vec![],
                },
            ],
        };

        assert_eq!(
            checker.ctx.constructor_count(),
            0,
            "No constructors before collect_enum_def"
        );

        checker.collect_enum_def(&enum_decl);

        // The type_defs should have Color registered
        let color_scheme = checker.ctx.lookup_type_def(Symbol::new("Color"));
        assert!(color_scheme.is_some(), "Color type should be registered");
        let color_scheme = color_scheme.unwrap();
        assert_eq!(color_scheme.arity(db), 0, "Color has no type params");
        if let TypeKind::Named { name, args } = color_scheme.body(db).kind(db) {
            assert_eq!(*name, Symbol::new("Color"));
            assert!(args.is_empty());
        } else {
            panic!(
                "Expected Named type for Color, got {:?}",
                color_scheme.body(db).kind(db)
            );
        }

        // Verify that 2 variant constructors were registered (Red and Green)
        assert_eq!(
            checker.ctx.constructor_count(),
            2,
            "Expected 2 constructors (Red and Green)"
        );

        true
    }

    #[test]
    fn test_collect_enum_def_registers_constructors() {
        let db = test_db();
        assert!(test_collect_enum_def_registers_constructors_inner(&db));
    }

    #[test]
    fn test_annotation_to_type_with_bound_vars() {
        // Test that annotation_to_type_with_bound_vars correctly maps type parameter
        // names to BoundVar indices.
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let type_param_indices = vec![(Symbol::new("a"), 0u32), (Symbol::new("b"), 1u32)];

        // "a" should become BoundVar { index: 0 }
        let ann_a = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("a")),
        };
        let ty_a = checker.annotation_to_type_with_bound_vars(&ann_a, &type_param_indices);
        assert!(
            matches!(*ty_a.kind(&db), TypeKind::BoundVar { index: 0 }),
            "Expected BoundVar(0) for 'a', got {:?}",
            ty_a.kind(&db)
        );

        // "b" should become BoundVar { index: 1 }
        let ann_b = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("b")),
        };
        let ty_b = checker.annotation_to_type_with_bound_vars(&ann_b, &type_param_indices);
        assert!(
            matches!(*ty_b.kind(&db), TypeKind::BoundVar { index: 1 }),
            "Expected BoundVar(1) for 'b', got {:?}",
            ty_b.kind(&db)
        );

        // "Int" should remain as Named { name: "Int", ... } (not a type param)
        let ann_int = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("Int")),
        };
        let ty_int = checker.annotation_to_type_with_bound_vars(&ann_int, &type_param_indices);
        assert!(
            matches!(*ty_int.kind(&db), TypeKind::Int),
            "Expected Int for 'Int', got {:?}",
            ty_int.kind(&db)
        );
    }

    #[salsa::tracked]
    fn test_collect_polymorphic_enum_constructors_inner(db: &dyn salsa::Database) -> bool {
        // enum Option(a) { None, Some(a) }
        // None  → forall a. Option(BoundVar(0))
        // Some  → forall a. fn(BoundVar(0)) -> Option(BoundVar(0))
        let mut checker = TypeChecker::new(db);

        let enum_decl = crate::ast::EnumDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("Option"),
            type_params: vec![crate::ast::TypeParamDecl {
                id: fresh_node_id(),
                name: Symbol::new("a"),
                bounds: vec![],
            }],
            variants: vec![
                crate::ast::VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("None"),
                    fields: vec![],
                },
                crate::ast::VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("Some"),
                    fields: vec![crate::ast::FieldDecl {
                        id: fresh_node_id(),
                        is_pub: false,
                        name: None,
                        ty: TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("a")),
                        },
                    }],
                },
            ],
        };

        checker.collect_enum_def(&enum_decl);

        // Should have registered 2 constructors (None and Some)
        assert_eq!(
            checker.ctx.constructor_count(),
            2,
            "Expected 2 constructors for Option"
        );

        // Verify the type scheme for the enum type itself
        let opt_scheme = checker.ctx.lookup_type_def(Symbol::new("Option")).unwrap();
        assert_eq!(opt_scheme.arity(db), 1, "Option has 1 type param");
        if let TypeKind::Named { name, args } = opt_scheme.body(db).kind(db) {
            assert_eq!(*name, Symbol::new("Option"));
            assert_eq!(args.len(), 1);
            assert!(
                matches!(*args[0].kind(db), TypeKind::BoundVar { index: 0 }),
                "Expected BoundVar(0) in Option type, got {:?}",
                args[0].kind(db)
            );
        } else {
            panic!("Expected Named type for Option");
        }

        true
    }

    #[test]
    fn test_collect_polymorphic_enum_constructors() {
        let db = test_db();
        assert!(test_collect_polymorphic_enum_constructors_inner(&db));
    }

    #[salsa::tracked]
    fn test_collect_multi_field_variant_inner(db: &dyn salsa::Database) -> bool {
        // enum Pair(a, b) { MkPair(a, b) }
        // MkPair → forall a b. fn(BoundVar(0), BoundVar(1)) -> Pair(BoundVar(0), BoundVar(1))
        let mut checker = TypeChecker::new(db);

        let enum_decl = crate::ast::EnumDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("Pair"),
            type_params: vec![
                crate::ast::TypeParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("a"),
                    bounds: vec![],
                },
                crate::ast::TypeParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("b"),
                    bounds: vec![],
                },
            ],
            variants: vec![crate::ast::VariantDecl {
                id: fresh_node_id(),
                name: Symbol::new("MkPair"),
                fields: vec![
                    crate::ast::FieldDecl {
                        id: fresh_node_id(),
                        is_pub: false,
                        name: None,
                        ty: TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("a")),
                        },
                    },
                    crate::ast::FieldDecl {
                        id: fresh_node_id(),
                        is_pub: false,
                        name: None,
                        ty: TypeAnnotation {
                            id: fresh_node_id(),
                            kind: TypeAnnotationKind::Named(Symbol::new("b")),
                        },
                    },
                ],
            }],
        };

        checker.collect_enum_def(&enum_decl);

        assert_eq!(
            checker.ctx.constructor_count(),
            1,
            "Expected 1 constructor (MkPair)"
        );

        // Verify Pair type scheme has 2 type params with BoundVar(0) and BoundVar(1)
        let pair_scheme = checker.ctx.lookup_type_def(Symbol::new("Pair")).unwrap();
        assert_eq!(pair_scheme.arity(db), 2, "Pair has 2 type params");
        if let TypeKind::Named { name, args } = pair_scheme.body(db).kind(db) {
            assert_eq!(*name, Symbol::new("Pair"));
            assert_eq!(args.len(), 2);
            assert!(matches!(*args[0].kind(db), TypeKind::BoundVar { index: 0 }));
            assert!(matches!(*args[1].kind(db), TypeKind::BoundVar { index: 1 }));
        } else {
            panic!("Expected Named type for Pair");
        }

        true
    }

    #[test]
    fn test_collect_multi_field_variant() {
        let db = test_db();
        assert!(test_collect_multi_field_variant_inner(&db));
    }

    #[test]
    fn test_annotation_to_type_with_bound_vars_app() {
        // List(a) with type_param_indices [(a, 0)] should become Named("List", [BoundVar(0)])
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let type_param_indices = vec![(Symbol::new("a"), 0u32)];

        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::App {
                ctor: Box::new(TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("List")),
                }),
                args: vec![TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("a")),
                }],
            },
        };

        let ty = checker.annotation_to_type_with_bound_vars(&ann, &type_param_indices);
        if let TypeKind::Named { name, args } = ty.kind(&db) {
            assert_eq!(*name, Symbol::new("List"));
            assert_eq!(args.len(), 1);
            assert!(
                matches!(*args[0].kind(&db), TypeKind::BoundVar { index: 0 }),
                "Expected BoundVar(0) for type arg 'a', got {:?}",
                args[0].kind(&db)
            );
        } else {
            panic!("Expected Named type for List(a), got {:?}", ty.kind(&db));
        }
    }

    #[test]
    fn test_annotation_to_type_with_bound_vars_tuple() {
        // (a, Int) with type_param_indices [(a, 0)] → Tuple([BoundVar(0), Int])
        let db = test_db();
        let mut checker = TypeChecker::new(&db);

        let type_param_indices = vec![(Symbol::new("a"), 0u32)];

        let ann = TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Tuple(vec![
                TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("a")),
                },
                TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                },
            ]),
        };

        let ty = checker.annotation_to_type_with_bound_vars(&ann, &type_param_indices);
        if let TypeKind::Tuple(elems) = ty.kind(&db) {
            assert_eq!(elems.len(), 2);
            assert!(
                matches!(*elems[0].kind(&db), TypeKind::BoundVar { index: 0 }),
                "Expected BoundVar(0) for 'a', got {:?}",
                elems[0].kind(&db)
            );
            assert!(
                matches!(*elems[1].kind(&db), TypeKind::Int),
                "Expected Int, got {:?}",
                elems[1].kind(&db)
            );
        } else {
            panic!("Expected Tuple type, got {:?}", ty.kind(&db));
        }
    }

    // =========================================================================
    // Substitution post-pass tests
    // =========================================================================

    /// Helper: build `fn main() { <stmts>; <value> }` module for testing check_module.
    fn make_func_module<'db>(
        stmts: Vec<Stmt<ResolvedRef<'db>>>,
        value: Expr<ResolvedRef<'db>>,
    ) -> Module<ResolvedRef<'db>> {
        let func = FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![],
            return_ty: None,
            effects: None,
            body: Expr::new(fresh_node_id(), ExprKind::Block { stmts, value }),
        };
        Module {
            id: fresh_node_id(),
            name: None,
            decls: vec![Decl::Function(func)],
        }
    }

    #[salsa::tracked]
    fn test_subst_applied_inner(db: &dyn salsa::Database) -> bool {
        let local_id = LocalId::new(0);
        let pattern = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name: Symbol::new("x"),
                local_id: Some(local_id),
            },
        );
        let value = Expr::new(fresh_node_id(), ExprKind::NatLit(42));

        let module = make_func_module(
            vec![Stmt::Let {
                id: fresh_node_id(),
                pattern,
                ty: None,
                value,
            }],
            Expr::new(fresh_node_id(), ExprKind::NatLit(0)),
        );

        let checker = TypeChecker::new(db);
        let typed_module = checker.check_module(module);

        if let Decl::Function(f) = &typed_module.decls[0]
            && let ExprKind::Block { stmts, .. } = &*f.body.kind
            && let Stmt::Let { value, .. } = &stmts[0]
        {
            assert!(matches!(*value.kind, ExprKind::NatLit(42)));
            return true;
        }
        panic!("Expected Function/Block/Let structure");
    }

    #[test]
    fn test_subst_applied_to_let_value_type() {
        let db = test_db();
        assert!(test_subst_applied_inner(&db));
    }

    #[salsa::tracked]
    fn test_subst_resolves_univar_inner(db: &dyn salsa::Database) -> bool {
        let local_x = LocalId::new(0);

        let let_pattern = Pattern::new(
            fresh_node_id(),
            PatternKind::Bind {
                name: Symbol::new("x"),
                local_id: Some(local_x),
            },
        );
        let let_value = Expr::new(fresh_node_id(), ExprKind::NatLit(42));
        let var_ref = ResolvedRef::Local {
            id: local_x,
            name: Symbol::new("x"),
        };
        let var_expr = Expr::new(fresh_node_id(), ExprKind::Var(var_ref));

        let module = make_func_module(
            vec![Stmt::Let {
                id: fresh_node_id(),
                pattern: let_pattern,
                ty: None,
                value: let_value,
            }],
            var_expr,
        );

        let checker = TypeChecker::new(db);
        let typed_module = checker.check_module(module);

        if let Decl::Function(f) = &typed_module.decls[0]
            && let ExprKind::Block { value, .. } = &*f.body.kind
            && let ExprKind::Var(typed_ref) = &*value.kind
        {
            assert!(
                matches!(*typed_ref.ty.kind(db), TypeKind::Nat),
                "Expected Nat type after substitution, got {:?}",
                typed_ref.ty.kind(db)
            );
            return true;
        }
        panic!("Expected Function/Block/Var structure");
    }

    #[test]
    fn test_subst_resolves_univar_to_concrete() {
        let db = test_db();
        assert!(test_subst_resolves_univar_inner(&db));
    }

    #[salsa::tracked]
    fn test_subst_binop_inner(db: &dyn salsa::Database) -> bool {
        // fn main() { let x = 1 + 2; x }
        // After substitution, x should have concrete type (not UniVar).
        let local_x = LocalId::new(0);

        let binop = Expr::new(
            fresh_node_id(),
            ExprKind::BinOp {
                op: BinOpKind::Add,
                lhs: Expr::new(fresh_node_id(), ExprKind::NatLit(1)),
                rhs: Expr::new(fresh_node_id(), ExprKind::NatLit(2)),
            },
        );

        let let_x = Stmt::Let {
            id: fresh_node_id(),
            pattern: Pattern::new(
                fresh_node_id(),
                PatternKind::Bind {
                    name: Symbol::new("x"),
                    local_id: Some(local_x),
                },
            ),
            ty: None,
            value: binop,
        };

        let var_x = Expr::new(
            fresh_node_id(),
            ExprKind::Var(ResolvedRef::Local {
                id: local_x,
                name: Symbol::new("x"),
            }),
        );

        let module = make_func_module(vec![let_x], var_x);

        let checker = TypeChecker::new(db);
        let typed_module = checker.check_module(module);

        if let Decl::Function(f) = &typed_module.decls[0]
            && let ExprKind::Block { value, .. } = &*f.body.kind
            && let ExprKind::Var(typed_ref) = &*value.kind
        {
            assert!(
                matches!(*typed_ref.ty.kind(db), TypeKind::Nat),
                "Expected Nat after substitution for 1+2, got {:?}",
                typed_ref.ty.kind(db)
            );
            return true;
        }
        panic!("Expected Function/Block/Var structure");
    }

    #[test]
    fn test_subst_binop() {
        let db = test_db();
        assert!(test_subst_binop_inner(&db));
    }

    #[salsa::tracked]
    fn test_subst_bool_literal_inner(db: &dyn salsa::Database) -> bool {
        // fn main() { let b = true; b }
        // After substitution, b should be Bool.
        let local_b = LocalId::new(0);

        let let_b = Stmt::Let {
            id: fresh_node_id(),
            pattern: Pattern::new(
                fresh_node_id(),
                PatternKind::Bind {
                    name: Symbol::new("b"),
                    local_id: Some(local_b),
                },
            ),
            ty: None,
            value: Expr::new(fresh_node_id(), ExprKind::BoolLit(true)),
        };

        let var_b = Expr::new(
            fresh_node_id(),
            ExprKind::Var(ResolvedRef::Local {
                id: local_b,
                name: Symbol::new("b"),
            }),
        );

        let module = make_func_module(vec![let_b], var_b);

        let checker = TypeChecker::new(db);
        let typed_module = checker.check_module(module);

        if let Decl::Function(f) = &typed_module.decls[0]
            && let ExprKind::Block { value, .. } = &*f.body.kind
            && let ExprKind::Var(typed_ref) = &*value.kind
        {
            assert!(
                matches!(*typed_ref.ty.kind(db), TypeKind::Bool),
                "Expected Bool after substitution, got {:?}",
                typed_ref.ty.kind(db)
            );
            return true;
        }
        panic!("Expected Function/Block/Var structure");
    }

    #[test]
    fn test_subst_bool_literal() {
        let db = test_db();
        assert!(test_subst_bool_literal_inner(&db));
    }
}
