//! Shared traversals over the phase-parameterized AST.
//!
//! [`Visit`] walks a tree by reference and [`VisitMut`] walks it in place.
//! Each `visit_*` method defaults to the matching `walk_*` function, which
//! visits the node's own identity and phase values and then its children. An
//! implementor overrides only the hooks it needs; an override that still wants
//! the children calls the `walk_*` function itself, before or after its own
//! work.
//!
//! The phase value `V` appears at the positions named by [`RefSite`]. Node
//! identities are visited for functions, expressions, statements, case and
//! handler arms, patterns, and record field patterns.
//!
//! Some parts of the tree are deliberately not visited. A pass that needs them
//! reads them from the enclosing node's hook:
//!
//! - Type annotations (`let` and parameter types, return types, and effect
//!   lists) hold no phase value.
//! - Function and lambda parameters, including their identities.
//! - Local binding identities (`LocalId`) of patterns, `resume`, and `op`
//!   handler arms, which are not node identities.
//!
//! `walk_stmt` visits a `let` pattern before its value, the order the passes
//! built on it use. The value is evaluated outside the pattern's bindings, so
//! a pass that tracks scopes overrides `visit_stmt`.
//!
//! The walks match every variant without a wildcard, so a new expression or
//! pattern variant fails to compile here until its traversal is written.

use super::{
    Arm, Decl, Expr, ExprKind, FieldPattern, FuncDecl, HandlerArm, HandlerKind, Module, ModuleDecl,
    NodeId, Pattern, PatternKind, Stmt,
};

/// Where a phase value sits in the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefSite {
    /// `ExprKind::Var`.
    Var,
    /// The constructor of `ExprKind::Cons`.
    ConsCtor,
    /// The type name of `ExprKind::Record`.
    RecordType,
    /// The ability of a `fn` or `op` handler arm.
    HandlerAbility,
    /// The constructor of `PatternKind::Variant`.
    PatternCtor,
    /// The type name of `PatternKind::Record`.
    PatternRecordType,
}

/// A traversal that reads the tree.
pub trait Visit<'ast, V: 'ast> {
    fn visit_decl(&mut self, decl: &'ast Decl<V>) {
        walk_decl(self, decl);
    }

    fn visit_module_decl(&mut self, module: &'ast ModuleDecl<V>) {
        walk_module_decl(self, module);
    }

    fn visit_func_decl(&mut self, func: &'ast FuncDecl<V>) {
        walk_func_decl(self, func);
    }

    fn visit_expr(&mut self, expr: &'ast Expr<V>) {
        walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt<V>) {
        walk_stmt(self, stmt);
    }

    fn visit_arm(&mut self, arm: &'ast Arm<V>) {
        walk_arm(self, arm);
    }

    fn visit_handler_arm(&mut self, arm: &'ast HandlerArm<V>) {
        walk_handler_arm(self, arm);
    }

    fn visit_pattern(&mut self, pattern: &'ast Pattern<V>) {
        walk_pattern(self, pattern);
    }

    fn visit_field_pattern(&mut self, field: &'ast FieldPattern<V>) {
        walk_field_pattern(self, field);
    }

    /// A phase value at `site`, owned by the node `node`.
    fn visit_ref(&mut self, site: RefSite, node: NodeId, value: &'ast V) {
        let _ = (site, node, value);
    }

    /// The identity of a visited node.
    fn visit_node_id(&mut self, id: NodeId) {
        let _ = id;
    }
}

/// Visit every declaration of a source module. A source module has no hook of
/// its own; this is the entry point for walking one.
pub fn walk_module<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(
    visitor: &mut T,
    module: &'ast Module<V>,
) {
    for decl in &module.decls {
        visitor.visit_decl(decl);
    }
}

pub fn walk_decl<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(visitor: &mut T, decl: &'ast Decl<V>) {
    match decl {
        Decl::Function(func) => visitor.visit_func_decl(func),
        Decl::Module(module) => visitor.visit_module_decl(module),
        Decl::ExternFunction(_)
        | Decl::Struct(_)
        | Decl::Enum(_)
        | Decl::Ability(_)
        | Decl::Use(_) => {}
    }
}

pub fn walk_module_decl<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(
    visitor: &mut T,
    module: &'ast ModuleDecl<V>,
) {
    for decl in module.body.iter().flatten() {
        visitor.visit_decl(decl);
    }
}

pub fn walk_func_decl<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(
    visitor: &mut T,
    func: &'ast FuncDecl<V>,
) {
    visitor.visit_node_id(func.id);
    visitor.visit_expr(&func.body);
}

pub fn walk_expr<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(visitor: &mut T, expr: &'ast Expr<V>) {
    visitor.visit_node_id(expr.id);
    match &*expr.kind {
        ExprKind::Var(value) => visitor.visit_ref(RefSite::Var, expr.id, value),
        ExprKind::Call { callee, args } => {
            visitor.visit_expr(callee);
            for arg in args {
                visitor.visit_expr(arg);
            }
        }
        ExprKind::Cons { ctor, args } => {
            visitor.visit_ref(RefSite::ConsCtor, expr.id, ctor);
            for arg in args {
                visitor.visit_expr(arg);
            }
        }
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => {
            visitor.visit_ref(RefSite::RecordType, expr.id, type_name);
            for field in fields {
                visitor.visit_node_id(field.id);
                visitor.visit_expr(&field.value);
            }
            if let Some(spread) = spread {
                visitor.visit_expr(spread);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            visitor.visit_expr(receiver);
            for arg in args {
                visitor.visit_expr(arg);
            }
        }
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                visitor.visit_stmt(stmt);
            }
            visitor.visit_expr(value);
        }
        ExprKind::Case { scrutinee, arms } => {
            visitor.visit_expr(scrutinee);
            for arm in arms {
                visitor.visit_arm(arm);
            }
        }
        ExprKind::Lambda { body, .. } => visitor.visit_expr(body),
        ExprKind::Handle { body, handlers } => {
            visitor.visit_expr(body);
            for arm in handlers {
                visitor.visit_handler_arm(arm);
            }
        }
        ExprKind::Resume { arg, .. } => visitor.visit_expr(arg),
        ExprKind::Tuple(elements) | ExprKind::List(elements) => {
            for element in elements {
                visitor.visit_expr(element);
            }
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            visitor.visit_expr(lhs);
            visitor.visit_expr(rhs);
        }
        ExprKind::NatLit(_)
        | ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::Nil
        | ExprKind::RuneLit(_)
        | ExprKind::Error => {}
    }
}

pub fn walk_stmt<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(visitor: &mut T, stmt: &'ast Stmt<V>) {
    match stmt {
        Stmt::Let {
            id, pattern, value, ..
        } => {
            visitor.visit_node_id(*id);
            visitor.visit_pattern(pattern);
            visitor.visit_expr(value);
        }
        Stmt::Expr { id, expr } => {
            visitor.visit_node_id(*id);
            visitor.visit_expr(expr);
        }
    }
}

pub fn walk_arm<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(visitor: &mut T, arm: &'ast Arm<V>) {
    visitor.visit_node_id(arm.id);
    visitor.visit_pattern(&arm.pattern);
    if let Some(guard) = &arm.guard {
        visitor.visit_expr(guard);
    }
    visitor.visit_expr(&arm.body);
}

pub fn walk_handler_arm<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(
    visitor: &mut T,
    arm: &'ast HandlerArm<V>,
) {
    visitor.visit_node_id(arm.id);
    match &arm.kind {
        HandlerKind::Do { binding } => visitor.visit_pattern(binding),
        HandlerKind::Fn {
            ability, params, ..
        }
        | HandlerKind::Op {
            ability, params, ..
        } => {
            visitor.visit_ref(RefSite::HandlerAbility, arm.id, ability);
            for param in params {
                visitor.visit_pattern(param);
            }
        }
    }
    visitor.visit_expr(&arm.body);
}

pub fn walk_pattern<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(
    visitor: &mut T,
    pattern: &'ast Pattern<V>,
) {
    visitor.visit_node_id(pattern.id);
    match &*pattern.kind {
        PatternKind::Variant { ctor, fields } => {
            visitor.visit_ref(RefSite::PatternCtor, pattern.id, ctor);
            for field in fields {
                visitor.visit_pattern(field);
            }
        }
        PatternKind::Record {
            type_name, fields, ..
        } => {
            visitor.visit_ref(RefSite::PatternRecordType, pattern.id, type_name);
            for field in fields {
                visitor.visit_field_pattern(field);
            }
        }
        PatternKind::Tuple(elements) | PatternKind::List(elements) => {
            for element in elements {
                visitor.visit_pattern(element);
            }
        }
        PatternKind::ListRest { head, .. } => {
            for element in head {
                visitor.visit_pattern(element);
            }
        }
        PatternKind::As { pattern, .. } => visitor.visit_pattern(pattern),
        PatternKind::Wildcard
        | PatternKind::Bind { .. }
        | PatternKind::Literal(_)
        | PatternKind::Error => {}
    }
}

/// A shorthand field (`{ name }`) has no pattern of its own.
pub fn walk_field_pattern<'ast, V: 'ast, T: Visit<'ast, V> + ?Sized>(
    visitor: &mut T,
    field: &'ast FieldPattern<V>,
) {
    visitor.visit_node_id(field.id);
    visitor.visit_node_id(field.name_id);
    if let Some(pattern) = &field.pattern {
        visitor.visit_pattern(pattern);
    }
}

impl<V> Expr<V> {
    /// Call `f` on this expression and every expression nested in it, in
    /// pre-order.
    pub fn for_each<'ast>(&'ast self, f: impl FnMut(&'ast Expr<V>)) {
        struct Exprs<F>(F);
        impl<'ast, V: 'ast, F: FnMut(&'ast Expr<V>)> Visit<'ast, V> for Exprs<F> {
            fn visit_expr(&mut self, expr: &'ast Expr<V>) {
                (self.0)(expr);
                walk_expr(self, expr);
            }
        }
        Exprs(f).visit_expr(self);
    }

    /// Call `f` on this expression and then on every expression nested in
    /// it, in pre-order. The children walked are those `f` leaves in place.
    pub fn for_each_mut(&mut self, f: impl FnMut(&mut Expr<V>)) {
        struct Exprs<F>(F);
        impl<V, F: FnMut(&mut Expr<V>)> VisitMut<V> for Exprs<F> {
            fn visit_expr_mut(&mut self, expr: &mut Expr<V>) {
                (self.0)(expr);
                walk_expr_mut(self, expr);
            }
        }
        Exprs(f).visit_expr_mut(self);
    }
}

/// A visitor that calls a closure on each phase value and uses the default
/// walk everywhere else: `walk_module(&mut Refs(|site, node, value| ..), m)`.
pub struct Refs<F>(pub F);

impl<'ast, V: 'ast, F: FnMut(RefSite, NodeId, &'ast V)> Visit<'ast, V> for Refs<F> {
    fn visit_ref(&mut self, site: RefSite, node: NodeId, value: &'ast V) {
        (self.0)(site, node, value);
    }
}

impl<V, F: FnMut(RefSite, NodeId, &mut V)> VisitMut<V> for Refs<F> {
    fn visit_ref_mut(&mut self, site: RefSite, node: NodeId, value: &mut V) {
        (self.0)(site, node, value);
    }
}

/// A traversal that rewrites the tree in place.
///
/// A node's identity is visited before its phase values, so a phase-value
/// hook receives the identity as `visit_node_id_mut` left it.
pub trait VisitMut<V> {
    fn visit_decl_mut(&mut self, decl: &mut Decl<V>) {
        walk_decl_mut(self, decl);
    }

    fn visit_module_decl_mut(&mut self, module: &mut ModuleDecl<V>) {
        walk_module_decl_mut(self, module);
    }

    fn visit_func_decl_mut(&mut self, func: &mut FuncDecl<V>) {
        walk_func_decl_mut(self, func);
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr<V>) {
        walk_expr_mut(self, expr);
    }

    fn visit_stmt_mut(&mut self, stmt: &mut Stmt<V>) {
        walk_stmt_mut(self, stmt);
    }

    fn visit_arm_mut(&mut self, arm: &mut Arm<V>) {
        walk_arm_mut(self, arm);
    }

    fn visit_handler_arm_mut(&mut self, arm: &mut HandlerArm<V>) {
        walk_handler_arm_mut(self, arm);
    }

    fn visit_pattern_mut(&mut self, pattern: &mut Pattern<V>) {
        walk_pattern_mut(self, pattern);
    }

    fn visit_field_pattern_mut(&mut self, field: &mut FieldPattern<V>) {
        walk_field_pattern_mut(self, field);
    }

    /// A phase value at `site`, owned by the node `node`.
    fn visit_ref_mut(&mut self, site: RefSite, node: NodeId, value: &mut V) {
        let _ = (site, node, value);
    }

    /// The identity of a visited node.
    fn visit_node_id_mut(&mut self, id: &mut NodeId) {
        let _ = id;
    }
}

/// Rewrite every declaration of a source module in place. A source module has
/// no hook of its own; this is the entry point for walking one.
pub fn walk_module_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, module: &mut Module<V>) {
    for decl in &mut module.decls {
        visitor.visit_decl_mut(decl);
    }
}

pub fn walk_decl_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, decl: &mut Decl<V>) {
    match decl {
        Decl::Function(func) => visitor.visit_func_decl_mut(func),
        Decl::Module(module) => visitor.visit_module_decl_mut(module),
        Decl::ExternFunction(_)
        | Decl::Struct(_)
        | Decl::Enum(_)
        | Decl::Ability(_)
        | Decl::Use(_) => {}
    }
}

pub fn walk_module_decl_mut<V, T: VisitMut<V> + ?Sized>(
    visitor: &mut T,
    module: &mut ModuleDecl<V>,
) {
    for decl in module.body.iter_mut().flatten() {
        visitor.visit_decl_mut(decl);
    }
}

pub fn walk_func_decl_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, func: &mut FuncDecl<V>) {
    visitor.visit_node_id_mut(&mut func.id);
    visitor.visit_expr_mut(&mut func.body);
}

pub fn walk_expr_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, expr: &mut Expr<V>) {
    visitor.visit_node_id_mut(&mut expr.id);
    let id = expr.id;
    match &mut *expr.kind {
        ExprKind::Var(value) => visitor.visit_ref_mut(RefSite::Var, id, value),
        ExprKind::Call { callee, args } => {
            visitor.visit_expr_mut(callee);
            for arg in args {
                visitor.visit_expr_mut(arg);
            }
        }
        ExprKind::Cons { ctor, args } => {
            visitor.visit_ref_mut(RefSite::ConsCtor, id, ctor);
            for arg in args {
                visitor.visit_expr_mut(arg);
            }
        }
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => {
            visitor.visit_ref_mut(RefSite::RecordType, id, type_name);
            for field in fields {
                visitor.visit_node_id_mut(&mut field.id);
                visitor.visit_expr_mut(&mut field.value);
            }
            if let Some(spread) = spread {
                visitor.visit_expr_mut(spread);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            visitor.visit_expr_mut(receiver);
            for arg in args {
                visitor.visit_expr_mut(arg);
            }
        }
        ExprKind::Block { stmts, value } => {
            for stmt in stmts {
                visitor.visit_stmt_mut(stmt);
            }
            visitor.visit_expr_mut(value);
        }
        ExprKind::Case { scrutinee, arms } => {
            visitor.visit_expr_mut(scrutinee);
            for arm in arms {
                visitor.visit_arm_mut(arm);
            }
        }
        ExprKind::Lambda { body, .. } => visitor.visit_expr_mut(body),
        ExprKind::Handle { body, handlers } => {
            visitor.visit_expr_mut(body);
            for arm in handlers {
                visitor.visit_handler_arm_mut(arm);
            }
        }
        ExprKind::Resume { arg, .. } => visitor.visit_expr_mut(arg),
        ExprKind::Tuple(elements) | ExprKind::List(elements) => {
            for element in elements {
                visitor.visit_expr_mut(element);
            }
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            visitor.visit_expr_mut(lhs);
            visitor.visit_expr_mut(rhs);
        }
        ExprKind::NatLit(_)
        | ExprKind::IntLit(_)
        | ExprKind::FloatLit(_)
        | ExprKind::StringLit(_)
        | ExprKind::BytesLit(_)
        | ExprKind::BoolLit(_)
        | ExprKind::Nil
        | ExprKind::RuneLit(_)
        | ExprKind::Error => {}
    }
}

pub fn walk_stmt_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, stmt: &mut Stmt<V>) {
    match stmt {
        Stmt::Let {
            id, pattern, value, ..
        } => {
            visitor.visit_node_id_mut(id);
            visitor.visit_pattern_mut(pattern);
            visitor.visit_expr_mut(value);
        }
        Stmt::Expr { id, expr } => {
            visitor.visit_node_id_mut(id);
            visitor.visit_expr_mut(expr);
        }
    }
}

pub fn walk_arm_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, arm: &mut Arm<V>) {
    visitor.visit_node_id_mut(&mut arm.id);
    visitor.visit_pattern_mut(&mut arm.pattern);
    if let Some(guard) = &mut arm.guard {
        visitor.visit_expr_mut(guard);
    }
    visitor.visit_expr_mut(&mut arm.body);
}

pub fn walk_handler_arm_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, arm: &mut HandlerArm<V>) {
    visitor.visit_node_id_mut(&mut arm.id);
    match &mut arm.kind {
        HandlerKind::Do { binding } => visitor.visit_pattern_mut(binding),
        HandlerKind::Fn {
            ability, params, ..
        }
        | HandlerKind::Op {
            ability, params, ..
        } => {
            visitor.visit_ref_mut(RefSite::HandlerAbility, arm.id, ability);
            for param in params {
                visitor.visit_pattern_mut(param);
            }
        }
    }
    visitor.visit_expr_mut(&mut arm.body);
}

pub fn walk_pattern_mut<V, T: VisitMut<V> + ?Sized>(visitor: &mut T, pattern: &mut Pattern<V>) {
    visitor.visit_node_id_mut(&mut pattern.id);
    let id = pattern.id;
    match &mut *pattern.kind {
        PatternKind::Variant { ctor, fields } => {
            visitor.visit_ref_mut(RefSite::PatternCtor, id, ctor);
            for field in fields {
                visitor.visit_pattern_mut(field);
            }
        }
        PatternKind::Record {
            type_name, fields, ..
        } => {
            visitor.visit_ref_mut(RefSite::PatternRecordType, id, type_name);
            for field in fields {
                visitor.visit_field_pattern_mut(field);
            }
        }
        PatternKind::Tuple(elements) | PatternKind::List(elements) => {
            for element in elements {
                visitor.visit_pattern_mut(element);
            }
        }
        PatternKind::ListRest { head, .. } => {
            for element in head {
                visitor.visit_pattern_mut(element);
            }
        }
        PatternKind::As { pattern, .. } => visitor.visit_pattern_mut(pattern),
        PatternKind::Wildcard
        | PatternKind::Bind { .. }
        | PatternKind::Literal(_)
        | PatternKind::Error => {}
    }
}

/// A shorthand field (`{ name }`) has no pattern of its own.
pub fn walk_field_pattern_mut<V, T: VisitMut<V> + ?Sized>(
    visitor: &mut T,
    field: &mut FieldPattern<V>,
) {
    visitor.visit_node_id_mut(&mut field.id);
    visitor.visit_node_id_mut(&mut field.name_id);
    if let Some(pattern) = &mut field.pattern {
        visitor.visit_pattern_mut(pattern);
    }
}

#[cfg(test)]
mod tests;
