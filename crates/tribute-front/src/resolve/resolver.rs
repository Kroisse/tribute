//! Name resolution logic.
//!
//! This module transforms `Expr<UnresolvedName>` into `Expr<ResolvedRef<'db>>`
//! by looking up names in the module environment and local scopes.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    AbilityDecl, Arm, BuiltinRef, ConstDecl, Decl, EnumDecl, Expr, ExprKind, FieldPattern,
    FuncDecl, HandlerArm, HandlerKind, LocalId, LocalIdGen, Module, ModulePath, Param, Pattern,
    PatternKind, ResolvedRef, Stmt, StructDecl, UnresolvedName, UseDecl,
};

use super::env::{Binding, ModuleEnv};

/// Resolver for transforming unresolved names to resolved references.
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    /// Module-level environment with function and type definitions.
    env: ModuleEnv<'db>,
    /// Stack of local scopes (function parameters, let bindings, etc.).
    local_scopes: Vec<HashMap<Symbol, LocalId>>,
    /// Generator for unique LocalIds.
    local_id_gen: LocalIdGen,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>) -> Self {
        Self {
            db,
            env,
            local_scopes: vec![HashMap::new()],
            local_id_gen: LocalIdGen::new(),
        }
    }

    /// Enter a new local scope.
    fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    /// Exit the current local scope.
    fn pop_scope(&mut self) {
        self.local_scopes.pop();
    }

    /// Bind a local variable in the current scope.
    fn bind_local(&mut self, name: Symbol) -> LocalId {
        let id = self.local_id_gen.fresh();
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, id);
        }
        id
    }

    /// Look up a local variable in all scopes.
    fn lookup_local(&self, name: Symbol) -> Option<LocalId> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(&id) = scope.get(&name) {
                return Some(id);
            }
        }
        None
    }

    /// Resolve an unresolved name to a ResolvedRef.
    fn resolve_name(&self, name: &UnresolvedName) -> ResolvedRef<'db> {
        let sym = name.name;

        // First check local variables
        if let Some(local_id) = self.lookup_local(sym) {
            return ResolvedRef::local(local_id, sym);
        }

        // Check for builtin operators
        if let Some(builtin) = self.resolve_builtin(sym) {
            return ResolvedRef::builtin(builtin);
        }

        // Check module environment
        if let Some(binding) = self.env.lookup(sym) {
            return self.binding_to_ref(binding, sym);
        }

        // Not found - return as unresolved local (will be caught by later passes)
        // TODO: Emit diagnostic for unresolved name
        let id = LocalId::new(u32::MAX); // Sentinel value for unresolved
        ResolvedRef::local(id, sym)
    }

    /// Convert a binding to a resolved reference.
    fn binding_to_ref(&self, binding: &Binding<'db>, name: Symbol) -> ResolvedRef<'db> {
        match binding {
            Binding::Function { id } => ResolvedRef::function(*id),
            Binding::Constructor { id, tag, .. } => {
                ResolvedRef::constructor(*id, tag.unwrap_or(name))
            }
            Binding::TypeDef {
                ctor_id: Some(id), ..
            } => ResolvedRef::constructor(*id, name),
            Binding::TypeDef { .. } => {
                // Type used as value - error
                let id = LocalId::new(u32::MAX);
                ResolvedRef::local(id, name)
            }
            Binding::Module { path } => {
                let path_ref = ModulePath::new(self.db, path.clone());
                ResolvedRef::Module { path: path_ref }
            }
        }
    }

    /// Check if a name refers to a builtin operation.
    fn resolve_builtin(&self, name: Symbol) -> Option<BuiltinRef> {
        // Use Symbol's PartialEq<&str> implementation
        if name == "print" {
            Some(BuiltinRef::Print)
        } else if name == "readLine" || name == "read_line" {
            Some(BuiltinRef::ReadLine)
        } else {
            None
        }
    }

    /// Resolve a module, transforming all declarations.
    pub fn resolve_module(&mut self, module: Module<UnresolvedName>) -> Module<ResolvedRef<'db>> {
        let decls = module
            .decls
            .into_iter()
            .map(|decl| self.resolve_decl(decl))
            .collect();

        Module {
            id: module.id,
            name: module.name,
            decls,
        }
    }

    /// Resolve a declaration.
    fn resolve_decl(&mut self, decl: Decl<UnresolvedName>) -> Decl<ResolvedRef<'db>> {
        match decl {
            Decl::Function(f) => Decl::Function(self.resolve_func_decl(f)),
            Decl::Struct(s) => Decl::Struct(self.resolve_struct_decl(s)),
            Decl::Enum(e) => Decl::Enum(self.resolve_enum_decl(e)),
            Decl::Ability(a) => Decl::Ability(self.resolve_ability_decl(a)),
            Decl::Const(c) => Decl::Const(self.resolve_const_decl(c)),
            Decl::Use(u) => Decl::Use(self.resolve_use_decl(u)),
        }
    }

    /// Resolve a function declaration.
    fn resolve_func_decl(&mut self, func: FuncDecl<UnresolvedName>) -> FuncDecl<ResolvedRef<'db>> {
        // Enter a new scope for function body
        self.push_scope();

        // Bind parameters
        let params = func
            .params
            .into_iter()
            .inspect(|p| {
                self.bind_local(p.name);
            })
            .collect();

        // Resolve body
        let body = self.resolve_expr(func.body);

        self.pop_scope();

        FuncDecl {
            id: func.id,
            is_pub: func.is_pub,
            name: func.name,
            type_params: func.type_params,
            params,
            return_ty: func.return_ty,
            effects: func.effects,
            body,
        }
    }

    /// Resolve a struct declaration.
    fn resolve_struct_decl(&self, s: StructDecl) -> StructDecl {
        // Struct declarations don't contain expressions to resolve
        s
    }

    /// Resolve an enum declaration.
    fn resolve_enum_decl(&self, e: EnumDecl) -> EnumDecl {
        // Enum declarations don't contain expressions to resolve
        e
    }

    /// Resolve an ability declaration.
    fn resolve_ability_decl(&self, a: AbilityDecl) -> AbilityDecl {
        // Ability declarations don't contain expressions to resolve
        a
    }

    /// Resolve a constant declaration.
    fn resolve_const_decl(&mut self, c: ConstDecl<UnresolvedName>) -> ConstDecl<ResolvedRef<'db>> {
        let value = self.resolve_expr(c.value);
        ConstDecl {
            id: c.id,
            is_pub: c.is_pub,
            name: c.name,
            ty: c.ty,
            value,
        }
    }

    /// Resolve a use declaration.
    fn resolve_use_decl(&self, u: UseDecl) -> UseDecl {
        // Use declarations don't contain expressions to resolve
        u
    }

    /// Resolve an expression.
    pub fn resolve_expr(&mut self, expr: Expr<UnresolvedName>) -> Expr<ResolvedRef<'db>> {
        let kind = match *expr.kind {
            ExprKind::Var(name) => ExprKind::Var(self.resolve_name(&name)),

            ExprKind::NatLit(n) => ExprKind::NatLit(n),
            ExprKind::IntLit(n) => ExprKind::IntLit(n),
            ExprKind::FloatLit(f) => ExprKind::FloatLit(f),
            ExprKind::StringLit(s) => ExprKind::StringLit(s),
            ExprKind::BytesLit(b) => ExprKind::BytesLit(b),
            ExprKind::BoolLit(b) => ExprKind::BoolLit(b),
            ExprKind::Nil => ExprKind::Nil,

            ExprKind::Call { callee, args } => {
                let callee = self.resolve_expr(callee);
                let args = args.into_iter().map(|a| self.resolve_expr(a)).collect();
                ExprKind::Call { callee, args }
            }

            ExprKind::Cons { ctor, args } => {
                let resolved_ctor = self.resolve_name(&ctor);
                let args = args.into_iter().map(|a| self.resolve_expr(a)).collect();
                ExprKind::Cons {
                    ctor: resolved_ctor,
                    args,
                }
            }

            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                let resolved_type = self.resolve_name(&type_name);
                let fields = fields
                    .into_iter()
                    .map(|(name, expr)| (name, self.resolve_expr(expr)))
                    .collect();
                let spread = spread.map(|e| self.resolve_expr(e));
                ExprKind::Record {
                    type_name: resolved_type,
                    fields,
                    spread,
                }
            }

            ExprKind::FieldAccess { expr, field } => {
                let expr = self.resolve_expr(expr);
                ExprKind::FieldAccess { expr, field }
            }

            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let receiver = self.resolve_expr(receiver);
                let args = args.into_iter().map(|a| self.resolve_expr(a)).collect();
                ExprKind::MethodCall {
                    receiver,
                    method,
                    args,
                }
            }

            ExprKind::Block { stmts, value } => {
                self.push_scope();
                let stmts = stmts.into_iter().map(|s| self.resolve_stmt(s)).collect();
                let value = self.resolve_expr(value);
                self.pop_scope();
                ExprKind::Block { stmts, value }
            }

            ExprKind::Case { scrutinee, arms } => {
                let scrutinee = self.resolve_expr(scrutinee);
                let arms = arms.into_iter().map(|a| self.resolve_arm(a)).collect();
                ExprKind::Case { scrutinee, arms }
            }

            ExprKind::Lambda { params, body } => {
                self.push_scope();
                let params: Vec<Param> = params
                    .into_iter()
                    .inspect(|p| {
                        self.bind_local(p.name);
                    })
                    .collect();
                let body = self.resolve_expr(body);
                self.pop_scope();
                ExprKind::Lambda { params, body }
            }

            ExprKind::Handle { body, handlers } => {
                let body = self.resolve_expr(body);
                let handlers = handlers
                    .into_iter()
                    .map(|h| self.resolve_handler_arm(h))
                    .collect();
                ExprKind::Handle { body, handlers }
            }

            ExprKind::Tuple(exprs) => {
                let exprs = exprs.into_iter().map(|e| self.resolve_expr(e)).collect();
                ExprKind::Tuple(exprs)
            }

            ExprKind::List(exprs) => {
                let exprs = exprs.into_iter().map(|e| self.resolve_expr(e)).collect();
                ExprKind::List(exprs)
            }

            ExprKind::BinOp { op, lhs, rhs } => {
                let lhs = self.resolve_expr(lhs);
                let rhs = self.resolve_expr(rhs);
                ExprKind::BinOp { op, lhs, rhs }
            }

            ExprKind::Error => ExprKind::Error,
        };

        Expr::new(expr.id, kind)
    }

    /// Resolve a statement.
    fn resolve_stmt(&mut self, stmt: Stmt<UnresolvedName>) -> Stmt<ResolvedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                ty,
                value,
            } => {
                let value = self.resolve_expr(value);
                let pattern = self.resolve_pattern_with_bindings(pattern);
                Stmt::Let {
                    id,
                    pattern,
                    ty,
                    value,
                }
            }
            Stmt::Expr { id, expr } => {
                let expr = self.resolve_expr(expr);
                Stmt::Expr { id, expr }
            }
        }
    }

    /// Resolve a case arm.
    fn resolve_arm(&mut self, arm: Arm<UnresolvedName>) -> Arm<ResolvedRef<'db>> {
        self.push_scope();
        let pattern = self.resolve_pattern_with_bindings(arm.pattern);
        let guard = arm.guard.map(|e| self.resolve_expr(e));
        let body = self.resolve_expr(arm.body);
        self.pop_scope();

        Arm {
            id: arm.id,
            pattern,
            guard,
            body,
        }
    }

    /// Resolve a handler arm.
    fn resolve_handler_arm(
        &mut self,
        handler: HandlerArm<UnresolvedName>,
    ) -> HandlerArm<ResolvedRef<'db>> {
        self.push_scope();

        let kind = match handler.kind {
            HandlerKind::Result { binding } => {
                let binding = self.resolve_pattern_with_bindings(binding);
                HandlerKind::Result { binding }
            }
            HandlerKind::Effect {
                ability,
                op,
                params,
                continuation,
            } => {
                // Resolve ability reference
                let resolved_ability = self.resolve_name(&ability);

                // Bind pattern params
                let resolved_params = params
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();

                // Bind continuation if present
                if let Some(cont_name) = continuation {
                    self.bind_local(cont_name);
                }

                HandlerKind::Effect {
                    ability: resolved_ability,
                    op,
                    params: resolved_params,
                    continuation,
                }
            }
        };

        let body = self.resolve_expr(handler.body);
        self.pop_scope();

        HandlerArm {
            id: handler.id,
            kind,
            body,
        }
    }

    /// Resolve a pattern, binding any names it introduces.
    fn resolve_pattern_with_bindings(
        &mut self,
        pattern: Pattern<UnresolvedName>,
    ) -> Pattern<ResolvedRef<'db>> {
        let kind = match *pattern.kind {
            PatternKind::Wildcard => PatternKind::Wildcard,

            PatternKind::Bind { name, .. } => {
                let local_id = self.bind_local(name);
                PatternKind::Bind {
                    name,
                    local_id: Some(local_id),
                }
            }

            PatternKind::Literal(lit) => PatternKind::Literal(lit),

            PatternKind::Variant { ctor, fields } => {
                let resolved_ctor = self.resolve_name(&ctor);
                let fields = fields
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                PatternKind::Variant {
                    ctor: resolved_ctor,
                    fields,
                }
            }

            PatternKind::Record {
                type_name,
                fields,
                rest,
            } => {
                let resolved_type = type_name.map(|t| self.resolve_name(&t));
                let fields = fields
                    .into_iter()
                    .map(|f| FieldPattern {
                        name: f.name,
                        pattern: f.pattern.map(|p| self.resolve_pattern_with_bindings(p)),
                    })
                    .collect();
                PatternKind::Record {
                    type_name: resolved_type,
                    fields,
                    rest,
                }
            }

            PatternKind::Tuple(patterns) => {
                let patterns = patterns
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                PatternKind::Tuple(patterns)
            }

            PatternKind::List(patterns) => {
                let patterns = patterns
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                PatternKind::List(patterns)
            }

            PatternKind::ListRest { head, rest } => {
                let head = head
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                // Bind the rest variable if it's not "_"
                if let Some(rest_name) = rest
                    && rest_name != "_"
                {
                    self.bind_local(rest_name);
                }
                PatternKind::ListRest { head, rest }
            }

            PatternKind::As { pattern, name } => {
                let pattern = self.resolve_pattern_with_bindings(pattern);
                self.bind_local(name);
                PatternKind::As { pattern, name }
            }

            PatternKind::Error => PatternKind::Error,
        };

        Pattern::new(pattern.id, kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::NodeId;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    #[test]
    fn test_resolve_local_variable() {
        let db = test_db();
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(&db, env);

        let param_name = Symbol::new("x");

        // Bind the parameter
        resolver.push_scope();
        resolver.bind_local(param_name);

        // Create unresolved reference to the parameter
        let body_var = UnresolvedName {
            name: param_name,
            id: NodeId::from_raw(2),
        };

        // Resolve the variable reference
        let resolved = resolver.resolve_name(&body_var);

        // Should resolve to a local variable
        match resolved {
            ResolvedRef::Local { name, .. } => {
                assert_eq!(name, param_name);
            }
            _ => panic!("Expected local variable, got {:?}", resolved),
        }
    }

    #[test]
    fn test_resolve_builtin_print() {
        let db = test_db();
        let env = ModuleEnv::new();
        let resolver = Resolver::new(&db, env);

        // Create unresolved reference to print
        let unresolved = UnresolvedName {
            name: Symbol::new("print"),
            id: NodeId::from_raw(1),
        };

        // Resolve the builtin
        let resolved = resolver.resolve_name(&unresolved);

        // Should resolve to a builtin
        match resolved {
            ResolvedRef::Builtin(BuiltinRef::Print) => {}
            _ => panic!("Expected print builtin, got {:?}", resolved),
        }
    }

    #[test]
    fn test_local_shadows_builtin() {
        let db = test_db();
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(&db, env);

        // Bind a local variable named 'print' (same as builtin)
        let name = Symbol::new("print");
        resolver.push_scope();
        resolver.bind_local(name);

        // Create unresolved reference
        let unresolved = UnresolvedName {
            name,
            id: NodeId::from_raw(1),
        };

        // Resolve - should get local, not builtin
        let resolved = resolver.resolve_name(&unresolved);

        match resolved {
            ResolvedRef::Local { name: n, .. } => {
                assert_eq!(n, name);
            }
            _ => panic!("Expected local to shadow builtin, got {:?}", resolved),
        }
    }

    #[test]
    fn test_scope_isolation() {
        let db = test_db();
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(&db, env);

        let name = Symbol::new("x");

        // Bind in inner scope
        resolver.push_scope();
        resolver.bind_local(name);
        resolver.pop_scope();

        // Create unresolved reference
        let unresolved = UnresolvedName {
            name,
            id: NodeId::from_raw(1),
        };

        // Resolve - should NOT find the local (it was in popped scope)
        let resolved = resolver.resolve_name(&unresolved);

        // Should be unresolved (sentinel LocalId)
        match resolved {
            ResolvedRef::Local { id, name: n } => {
                // Unresolved names use u32::MAX as sentinel
                assert_eq!(id.raw(), u32::MAX);
                assert_eq!(n, name);
            }
            _ => panic!("Expected unresolved local, got {:?}", resolved),
        }
    }

    #[test]
    fn test_nested_scopes() {
        let db = test_db();
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(&db, env);

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Outer scope: bind x
        resolver.push_scope();
        let x_id = resolver.bind_local(x);

        // Inner scope: bind y
        resolver.push_scope();
        let y_id = resolver.bind_local(y);

        // Both x and y should be visible
        let x_ref = UnresolvedName {
            name: x,
            id: NodeId::from_raw(1),
        };
        let y_ref = UnresolvedName {
            name: y,
            id: NodeId::from_raw(2),
        };

        let resolved_x = resolver.resolve_name(&x_ref);
        let resolved_y = resolver.resolve_name(&y_ref);

        match resolved_x {
            ResolvedRef::Local { id, .. } => assert_eq!(id, x_id),
            _ => panic!("Expected local x"),
        }
        match resolved_y {
            ResolvedRef::Local { id, .. } => assert_eq!(id, y_id),
            _ => panic!("Expected local y"),
        }

        // Exit inner scope
        resolver.pop_scope();

        // x should still be visible, but y should not
        let resolved_x = resolver.resolve_name(&x_ref);
        let resolved_y = resolver.resolve_name(&y_ref);

        match resolved_x {
            ResolvedRef::Local { id, .. } => assert_eq!(id, x_id),
            _ => panic!("Expected local x after pop"),
        }
        match resolved_y {
            ResolvedRef::Local { id, .. } => {
                // y should be unresolved (sentinel)
                assert_eq!(id.raw(), u32::MAX);
            }
            _ => panic!("Expected unresolved y after pop"),
        }
    }
}
