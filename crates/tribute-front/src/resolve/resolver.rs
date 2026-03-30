//! Name resolution logic.
//!
//! This module transforms `Expr<UnresolvedName>` into `Expr<ResolvedRef<'db>>`
//! by looking up names in the module environment and local scopes.

use std::collections::HashMap;

use salsa::Accumulator as _;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;

use crate::ast::{
    AbilityDecl, Arm, Decl, EnumDecl, Expr, ExprKind, FieldPattern, FuncDecl, HandlerArm,
    HandlerKind, LocalId, LocalIdGen, Module, ModulePath, Param, Pattern, PatternKind, ResolvedRef,
    SpanMap, Stmt, StructDecl, TypeAnnotation, TypeAnnotationKind, UnresolvedName, UseDecl,
};

use super::env::{Binding, ModuleEnv};

/// Find the best matches from `candidates` by a caller-provided score function.
///
/// Returns up to `max` items sorted by descending score, keeping only scores >= `threshold`.
/// Uses a min-heap to maintain only the top entries without collecting all candidates.
fn best_matches_by<T>(
    candidates: impl IntoIterator<Item = T>,
    score: impl Fn(&T) -> f64,
    threshold: f64,
    max: usize,
) -> Vec<T> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    struct Entry<T> {
        item: T,
        score: f64,
    }

    impl<T> PartialEq for Entry<T> {
        fn eq(&self, other: &Self) -> bool {
            self.score == other.score
        }
    }

    impl<T> Eq for Entry<T> {}

    impl<T> Ord for Entry<T> {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse: lower score = greater → pops first from max-heap
            other
                .score
                .partial_cmp(&self.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl<T> PartialOrd for Entry<T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: BinaryHeap<Entry<T>> = BinaryHeap::with_capacity(max + 1);

    for item in candidates {
        let s = score(&item);
        if s < threshold {
            continue;
        }
        if heap.len() == max && heap.peek().is_some_and(|top| s <= top.score) {
            continue;
        }
        heap.push(Entry { item, score: s });
        if heap.len() > max {
            heap.pop();
        }
    }

    heap.into_sorted_vec().into_iter().map(|e| e.item).collect()
}

/// Resolver for transforming unresolved names to resolved references.
pub struct Resolver<'db> {
    db: &'db dyn salsa::Database,
    /// Module-level environment with function and type definitions.
    env: ModuleEnv<'db>,
    /// Stack of local scopes (function parameters, let bindings, etc.).
    local_scopes: Vec<HashMap<Symbol, LocalId>>,
    /// Generator for unique LocalIds.
    local_id_gen: LocalIdGen,
    /// Span map for emitting diagnostics with source locations.
    span_map: SpanMap,
    /// Stack of LocalIds for `resume` in `op` handler arms.
    resume_local_id_stack: Vec<LocalId>,
    /// Ability operations injected from effect annotations (effect-directed resolution).
    /// Maps unqualified operation name → Binding. Cleared on each function scope.
    effect_ops: HashMap<Symbol, Binding<'db>>,
}

impl<'db> Resolver<'db> {
    /// Create a new resolver with the given environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>, span_map: SpanMap) -> Self {
        Self {
            db,
            env,
            local_scopes: vec![HashMap::new()],
            local_id_gen: LocalIdGen::new(),
            resume_local_id_stack: Vec::new(),
            span_map,
            effect_ops: HashMap::new(),
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
        let sym = name.name();

        // For simple names: check locals, builtins, then module environment
        if name.is_simple() {
            // First check local variables
            if let Some(local_id) = self.lookup_local(sym) {
                return ResolvedRef::local(local_id, sym);
            }

            // Check module environment (unqualified lookup)
            if let Some(binding) = self.env.lookup(sym) {
                return self.binding_to_ref(binding, sym);
            }

            // Check effect-injected ability operations (effect-directed resolution)
            if let Some(binding) = self.effect_ops.get(&sym) {
                return self.binding_to_ref(binding, sym);
            }
        } else {
            // Qualified path: e.g., State::get, Option::Some, abilities::Throw::throw
            if let Some(namespace) = name.namespace()
                && let Some(binding) = self.env.lookup_qualified(namespace, sym)
            {
                return self.binding_to_ref(binding, sym);
            }
        }

        // Not found - emit diagnostic and return unresolved sentinel
        self.report_unresolved_name(name);
        ResolvedRef::local(LocalId::UNRESOLVED, sym)
    }

    /// Report an unresolved name diagnostic, with "did you mean?" suggestions.
    fn report_unresolved_name(&self, name: &UnresolvedName) {
        let span = self.span_map.get_or_default(name.id);
        let similar = self.find_similar_names(name.name());

        let message = if similar.is_empty() {
            format!("unresolved name `{}`", name)
        } else {
            let suggestions = tribute_core::fmt::joined_by(", ", &similar, |s, f| {
                s.with_str(|name| write!(f, "`{}`", name))
            });
            format!("unresolved name `{}`; did you mean {}?", name, suggestions)
        };

        Diagnostic::new(
            message,
            span,
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(self.db);
    }

    /// Find names in scope that are similar to the given name.
    fn find_similar_names(&self, name: Symbol) -> Vec<Symbol> {
        use std::collections::HashSet;

        let candidates: HashSet<Symbol> = self
            .local_scopes
            .iter()
            .flat_map(|scope| scope.keys().copied())
            .chain(self.env.iter_all_names())
            .collect();

        let target = name.to_string();
        best_matches_by(
            candidates,
            |sym: &Symbol| sym.with_str(|s| strsim::jaro_winkler(&target, s)),
            0.8,
            3,
        )
    }

    /// Convert a binding to a resolved reference.
    fn binding_to_ref(&self, binding: &Binding<'db>, name: Symbol) -> ResolvedRef<'db> {
        match binding {
            Binding::Function { id } => ResolvedRef::function(*id),
            Binding::Constructor { id, tag, .. } => {
                ResolvedRef::constructor(*id, tag.unwrap_or(name))
            }
            Binding::TypeDef { id } => ResolvedRef::type_def(*id),
            Binding::Module { path } => {
                let path_ref = ModulePath::new(self.db, path.clone());
                ResolvedRef::Module { path: path_ref }
            }
            Binding::AbilityOp { ability, op, kind } => {
                ResolvedRef::ability_op(*ability, *op, *kind)
            }
            Binding::Ability { id } => ResolvedRef::ability(*id),
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
            Decl::ExternFunction(e) => Decl::ExternFunction(e),
            Decl::Struct(s) => Decl::Struct(self.resolve_struct_decl(s)),
            Decl::Enum(e) => Decl::Enum(self.resolve_enum_decl(e)),
            Decl::Ability(a) => Decl::Ability(self.resolve_ability_decl(a)),
            Decl::Use(u) => Decl::Use(self.resolve_use_decl(u)),
            Decl::Module(m) => Decl::Module(self.resolve_module_decl(m)),
        }
    }

    /// Resolve a module declaration.
    fn resolve_module_decl(
        &mut self,
        module: crate::ast::ModuleDecl<UnresolvedName>,
    ) -> crate::ast::ModuleDecl<ResolvedRef<'db>> {
        // For inline modules, recursively resolve nested declarations
        let body = module
            .body
            .map(|decls| decls.into_iter().map(|d| self.resolve_decl(d)).collect());

        crate::ast::ModuleDecl {
            id: module.id,
            name: module.name,
            is_pub: module.is_pub,
            body,
        }
    }

    /// Resolve a function declaration.
    fn resolve_func_decl(&mut self, func: FuncDecl<UnresolvedName>) -> FuncDecl<ResolvedRef<'db>> {
        // Enter a new scope for function body
        self.push_scope();

        // Bind parameters and assign local IDs
        let params = func
            .params
            .into_iter()
            .map(|mut p| {
                let local_id = self.bind_local(p.name);
                p.local_id = Some(local_id);
                p
            })
            .collect();

        // Inject ability operations from effect annotations into scope.
        // This enables effect-directed name resolution: when a function declares
        // an effect like `->{abilities::Abort}`, its operations (e.g., `abort()`)
        // become directly callable without qualification.
        if let Some(effects) = &func.effects {
            self.inject_ability_operations(effects);
        }

        // Resolve body
        let body = self.resolve_expr(func.body);

        self.effect_ops.clear();
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

    /// Inject ability operations from effect annotations into the current scope.
    ///
    /// For each ability in the effect row, look up its operations in the module
    /// environment and make them available as unqualified names. This enables
    /// calling `abort()` instead of `abilities::Abort::abort()` when the function
    /// declares `->{abilities::Abort}`.
    ///
    /// Parameters and local variables take precedence (already bound before this).
    fn inject_ability_operations(&mut self, effects: &[TypeAnnotation]) {
        self.effect_ops.clear();
        for ann in effects {
            let Some(ability_name) = Self::extract_ability_name(ann) else {
                continue;
            };

            for (op_name, binding) in self.env.iter_namespace(ability_name) {
                if matches!(binding, Binding::AbilityOp { .. }) {
                    // Don't override existing entries (first ability wins;
                    // TODO: detect ambiguity when multiple abilities export same op name)
                    self.effect_ops.entry(op_name).or_insert(binding.clone());
                }
            }
        }
    }

    /// Extract the ability name (as a Symbol) from a type annotation.
    ///
    /// - `Named(sym)` → sym (e.g., `Abort`)
    /// - `Path(segs)` → qualified symbol (e.g., `abilities::Throw`)
    /// - `App { ctor, .. }` → recurse into ctor (e.g., `Throw` from `Throw(Nat)`)
    fn extract_ability_name(ann: &TypeAnnotation) -> Option<Symbol> {
        match &ann.kind {
            TypeAnnotationKind::Named(sym) => {
                // Row tail variables (lowercase, e.g., `e`) are not concrete abilities
                sym.with_str(|s| {
                    if s.starts_with(|c: char| c.is_ascii_uppercase()) {
                        Some(*sym)
                    } else {
                        None
                    }
                })
            }
            TypeAnnotationKind::Path(segs) => {
                if segs.is_empty() {
                    None
                } else {
                    Some(Symbol::from_dynamic(
                        &segs
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join("::"),
                    ))
                }
            }
            TypeAnnotationKind::App { ctor, .. } => Self::extract_ability_name(ctor),
            _ => None,
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
            ExprKind::RuneLit(c) => ExprKind::RuneLit(c),

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
                    .map(|mut p| {
                        let local_id = self.bind_local(p.name);
                        p.local_id = Some(local_id);
                        p
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

            ExprKind::Resume { arg, .. } => {
                let arg = self.resolve_expr(arg);
                let local_id = self.resume_local_id_stack.last().copied();
                ExprKind::Resume { arg, local_id }
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
            HandlerKind::Do { binding } => {
                let binding = self.resolve_pattern_with_bindings(binding);
                HandlerKind::Do { binding }
            }
            HandlerKind::Fn {
                ability,
                op,
                params,
            } => {
                let resolved_ability = self.resolve_handler_ability(&ability);
                let resolved_params = params
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                HandlerKind::Fn {
                    ability: resolved_ability,
                    op,
                    params: resolved_params,
                }
            }
            HandlerKind::Op {
                ability,
                op,
                params,
                ..
            } => {
                let resolved_ability = self.resolve_handler_ability(&ability);
                let resolved_params = params
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                // Allocate a synthetic LocalId for `resume` so that lambda
                // capture analysis can track the continuation value.
                let resume_id = self.local_id_gen.fresh();
                self.resume_local_id_stack.push(resume_id);
                HandlerKind::Op {
                    ability: resolved_ability,
                    op,
                    params: resolved_params,
                    resume_local_id: Some(resume_id),
                }
            }
        };

        let is_op = matches!(kind, HandlerKind::Op { .. });
        let body = self.resolve_expr(handler.body);
        if is_op {
            self.resume_local_id_stack.pop();
        }
        self.pop_scope();

        HandlerArm {
            id: handler.id,
            kind,
            body,
        }
    }

    /// Resolve ability reference in a handler arm.
    /// Skips "_" placeholder (unqualified ops) without emitting diagnostics.
    fn resolve_handler_ability(&mut self, ability: &UnresolvedName) -> ResolvedRef<'db> {
        if ability.qualified == Symbol::new("_") {
            ResolvedRef::local(LocalId::UNRESOLVED, ability.qualified)
        } else {
            self.resolve_name(ability)
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
                        id: f.id,
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

            PatternKind::ListRest { head, rest, .. } => {
                let head = head
                    .into_iter()
                    .map(|p| self.resolve_pattern_with_bindings(p))
                    .collect();
                // Bind the rest variable if it's not "_"
                let rest_local_id = if let Some(rest_name) = rest
                    && rest_name != "_"
                {
                    Some(self.bind_local(rest_name))
                } else {
                    None
                };
                PatternKind::ListRest {
                    head,
                    rest,
                    rest_local_id,
                }
            }

            PatternKind::As { pattern, name, .. } => {
                let pattern = self.resolve_pattern_with_bindings(pattern);
                let local_id = Some(self.bind_local(name));
                PatternKind::As {
                    pattern,
                    name,
                    local_id,
                }
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
    use salsa_test_macros::salsa_test;

    #[salsa_test]
    fn test_resolve_local_variable(db: &salsa::DatabaseImpl) {
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());

        let param_name = Symbol::new("x");

        // Bind the parameter
        resolver.push_scope();
        resolver.bind_local(param_name);

        // Create unresolved reference to the parameter
        let body_var = UnresolvedName::new(param_name, NodeId::from_raw(2));

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

    #[salsa_test]
    fn test_scope_isolation(db: &salsa::DatabaseImpl) {
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Bind x in outer scope, y in inner scope
        resolver.push_scope();
        let x_id = resolver.bind_local(x);

        resolver.push_scope();
        let y_id = resolver.bind_local(y);
        resolver.pop_scope();

        // After popping inner scope, x should still be visible
        let x_ref = UnresolvedName::new(x, NodeId::from_raw(1));
        let resolved = resolver.resolve_name(&x_ref);

        match resolved {
            ResolvedRef::Local { id, .. } => assert_eq!(id, x_id),
            _ => panic!("Expected local x to still be visible after inner scope pop"),
        }

        // Note: Testing that y is NOT visible would trigger accumulate(),
        // which requires a tracked function context. Such tests belong in
        // integration tests via resolved_module query.
        let _ = y_id; // suppress unused warning
    }

    #[salsa_test]
    fn test_nested_scopes(db: &salsa::DatabaseImpl) {
        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Outer scope: bind x
        resolver.push_scope();
        let x_id = resolver.bind_local(x);

        // Inner scope: bind y
        resolver.push_scope();
        let y_id = resolver.bind_local(y);

        // Both x and y should be visible from inner scope
        let x_ref = UnresolvedName::new(x, NodeId::from_raw(1));
        let y_ref = UnresolvedName::new(y, NodeId::from_raw(2));

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

        // x should still be visible
        let resolved_x = resolver.resolve_name(&x_ref);
        match resolved_x {
            ResolvedRef::Local { id, .. } => assert_eq!(id, x_id),
            _ => panic!("Expected local x after pop"),
        }

        // Note: Testing that y is NOT visible after pop would trigger accumulate(),
        // which requires a tracked function context. Such tests belong in
        // integration tests via resolved_module query.
    }

    #[salsa_test]
    fn test_list_rest_pattern_local_id(db: &salsa::DatabaseImpl) {
        use crate::ast::{Pattern, PatternKind};

        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());
        resolver.push_scope();

        // Create a ListRest pattern: [head, ..rest]
        let head_pattern = Pattern::new(
            NodeId::from_raw(1),
            PatternKind::Bind {
                name: Symbol::new("head"),
                local_id: None,
            },
        );
        let rest_name = Symbol::new("rest");
        let list_rest_pattern = Pattern::new(
            NodeId::from_raw(2),
            PatternKind::ListRest {
                head: vec![head_pattern],
                rest: Some(rest_name),
                rest_local_id: None,
            },
        );

        // Resolve the pattern
        let resolved = resolver.resolve_pattern_with_bindings(list_rest_pattern);

        // Check that rest has a LocalId
        let PatternKind::ListRest {
            rest_local_id,
            rest,
            ..
        } = resolved.kind.as_ref()
        else {
            panic!("Expected ListRest pattern");
        };

        assert!(rest.is_some());
        assert!(rest_local_id.is_some(), "rest should have a LocalId");
    }

    #[salsa_test]
    fn test_as_pattern_local_id(db: &salsa::DatabaseImpl) {
        use crate::ast::{Pattern, PatternKind};

        let env = ModuleEnv::new();
        let mut resolver = Resolver::new(db, env, SpanMap::default());
        resolver.push_scope();

        // Create an As pattern: _ as all
        let inner_pattern = Pattern::new(NodeId::from_raw(1), PatternKind::Wildcard);
        let as_name = Symbol::new("all");
        let as_pattern = Pattern::new(
            NodeId::from_raw(2),
            PatternKind::As {
                pattern: inner_pattern,
                name: as_name,
                local_id: None,
            },
        );

        // Resolve the pattern
        let resolved = resolver.resolve_pattern_with_bindings(as_pattern);

        // Check that the as-binding has a LocalId
        let PatternKind::As { local_id, name, .. } = resolved.kind.as_ref() else {
            panic!("Expected As pattern");
        };

        assert_eq!(*name, as_name);
        assert!(local_id.is_some(), "as-binding should have a LocalId");
    }

    mod best_matches {
        use super::super::best_matches_by;

        #[test]
        fn finds_close_typo() {
            let candidates = ["compute", "compare", "display"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("compue", s),
                0.8,
                3,
            );
            assert!(result.contains(&&"compute"));
            assert!(!result.contains(&&"display"));
        }

        #[test]
        fn no_match_for_unrelated_name() {
            let candidates = ["foo", "bar", "baz"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("xyzzy", s),
                0.8,
                3,
            );
            assert!(result.is_empty());
        }

        #[test]
        fn respects_max_limit() {
            let candidates = ["print", "printf", "println", "printa", "printi"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("printt", s),
                0.5,
                2,
            );
            assert!(result.len() <= 2);
        }

        #[test]
        fn sorted_by_descending_score() {
            // "prnt" vs candidates: "print" is closer than "point"
            let candidates = ["point", "print", "paint"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("prnt", s),
                0.5,
                3,
            );
            assert!(result.len() >= 2);
            // Verify descending score order
            for pair in result.windows(2) {
                let s0 = strsim::jaro_winkler("prnt", pair[0]);
                let s1 = strsim::jaro_winkler("prnt", pair[1]);
                assert!(
                    s0 >= s1,
                    "{} ({}) should score >= {} ({})",
                    pair[0],
                    s0,
                    pair[1],
                    s1
                );
            }
        }

        #[test]
        fn threshold_filters_low_scores() {
            let candidates = ["abc", "xyz", "abcd"];
            let result = best_matches_by(
                candidates.iter(),
                |s| strsim::jaro_winkler("abcde", s),
                0.95,
                3,
            );
            // With threshold 0.95, only very close matches survive
            assert!(result.len() <= 1);
        }
    }
}
