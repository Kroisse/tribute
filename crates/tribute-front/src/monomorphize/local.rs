//! One copy of a let-bound lambda per convention class its uses select.

use std::hash::{Hash, Hasher};
use std::num::NonZero;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::MonomorphizeMetadata;
use super::instance::InstanceKeys;
use crate::ast::visit::{
    RefSite, Visit, VisitMut, walk_expr, walk_expr_mut, walk_handler_arm, walk_pattern,
    walk_pattern_mut, walk_stmt, walk_stmt_mut,
};
use crate::ast::{
    CallingConvention, Decl, EffectRow, EffectVar, Expr, ExprKind, HandlerArm, HandlerKind,
    LocalId, NodeId, Pattern, PatternKind, ResolvedRef, Stmt, Type, TypedRef, collect_effect_vars,
};
use crate::typeck::{RowSubst, TypeSubst};

/// Give every let-bound lambda of `decls` one class per binding, copying a
/// binding whose uses select several.
///
/// A use inside one lambda can pass on the class of another, so the classes
/// are settled again after each round of copies.
pub(super) fn settle_local_lambdas<'db>(
    db: &'db dyn salsa::Database,
    decls: &mut [Decl<TypedRef<'db>>],
    metadata: &mut MonomorphizeMetadata<'db>,
    keys: &mut InstanceKeys<'db>,
) {
    loop {
        while keys.settle_local_classes(decls, metadata) {}
        if !split_local_lambdas(db, decls, metadata, keys) {
            return;
        }
    }
}

/// A copy of the `let` that binds `binding`, for the uses that select one
/// class.
struct LambdaCopy {
    binding: NodeId,
    local: LocalId,
    variant: NonZero<u64>,
    uses: HashSet<NodeId>,
}

/// Bind a let-bound lambda of `decls` once per class its uses select. The
/// first class in source order keeps the original binding. Returns whether
/// a binding was copied.
fn split_local_lambdas<'db>(
    db: &'db dyn salsa::Database,
    decls: &mut [Decl<TypedRef<'db>>],
    metadata: &mut MonomorphizeMetadata<'db>,
    keys: &InstanceKeys<'db>,
) -> bool {
    let mut split = false;
    for decl in decls {
        let Decl::Function(function) = decl else {
            continue;
        };
        // The lambdas a `let` binds, by the local they bind.
        let mut bindings = HashMap::default();
        function.body.for_each(|expr| {
            let ExprKind::Block { stmts, .. } = &*expr.kind else {
                return;
            };
            for stmt in stmts {
                if let Stmt::Let { pattern, value, .. } = stmt
                    && let PatternKind::Bind {
                        local_id: Some(local),
                        ..
                    } = &*pattern.kind
                    && matches!(&*value.kind, ExprKind::Lambda { .. })
                {
                    bindings.insert(*local, (pattern.id, value.id));
                }
            }
        });
        if bindings.is_empty() {
            continue;
        }
        // The classes each use selects, in source order. A use the checker
        // recorded no instance for reads the binding as it is.
        let mut uses: HashMap<LocalId, Vec<(NodeId, Vec<CallingConvention>)>> = HashMap::default();
        let mut shared = HashSet::default();
        // The row variables each binding quantifies, as its lambda spells
        // them.
        let mut rows: HashMap<NodeId, Vec<EffectVar>> = HashMap::default();
        function.body.for_each(|expr| {
            let ExprKind::Var(TypedRef {
                resolved: ResolvedRef::Local { id, .. },
                ..
            }) = &*expr.kind
            else {
                return;
            };
            let Some((binding, lambda)) = bindings.get(id) else {
                return;
            };
            let instance = metadata
                .local_instances
                .get(&expr.id)
                .filter(|instance| instance.binding == *binding);
            let spelled = instance.and_then(|instance| {
                let signature = metadata.lambda_signatures.get(lambda)?;
                let quantified = collect_effect_vars(db, instance.scheme.body(db));
                let spelled = collect_effect_vars(db, signature.function_type);
                (quantified.len() == spelled.len()).then(|| {
                    instance
                        .scheme
                        .effect_params(db)
                        .iter()
                        .filter_map(|row| quantified.iter().position(|var| var == row))
                        .map(|at| spelled[at])
                        .collect()
                })
            });
            match (instance, spelled) {
                (Some(instance), Some(spelled)) => {
                    let classes = keys.classes_of_rows(&instance.row_arguments, &function.name);
                    uses.entry(*id).or_default().push((expr.id, classes));
                    rows.insert(*binding, spelled);
                }
                _ => {
                    shared.insert(*id);
                }
            }
        });
        // A copy's locals must differ from every local of the function.
        let mut locals = Locals::default();
        for param in &function.params {
            locals.reserve(param.local_id);
        }
        locals.visit_expr(&function.body);
        let mut next_local = locals.next;
        let mut copies = Vec::new();
        let mut uses: Vec<_> = uses.into_iter().collect();
        uses.sort_by_key(|(local, _)| local.raw());
        for (local, uses) in uses {
            if shared.contains(&local) {
                continue;
            }
            let (binding, _) = bindings[&local];
            let original = &uses[0].1;
            let mut classes: Vec<&Vec<CallingConvention>> = Vec::new();
            for (_, selected) in &uses {
                if selected != original && !classes.contains(&selected) {
                    classes.push(selected);
                }
            }
            for selected in classes {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                (binding, selected).hash(&mut hasher);
                copies.push(LambdaCopy {
                    binding,
                    local: LocalId::new(next_local),
                    variant: NonZero::new(hasher.finish()).unwrap_or(NonZero::<u64>::MIN),
                    uses: uses
                        .iter()
                        .filter(|(_, classes)| classes == selected)
                        .map(|(node, _)| *node)
                        .collect(),
                });
                next_local += 1;
            }
        }
        if copies.is_empty() {
            continue;
        }
        Split {
            db,
            metadata,
            copies: &copies,
            rows: &rows,
            next_local,
        }
        .visit_expr_mut(&mut function.body);
        split = true;
    }
    split
}

struct Split<'a, 'db> {
    db: &'db dyn salsa::Database,
    metadata: &'a mut MonomorphizeMetadata<'db>,
    copies: &'a [LambdaCopy],
    rows: &'a HashMap<NodeId, Vec<EffectVar>>,
    next_local: u32,
}

impl<'db> Split<'_, 'db> {
    /// Copy `stmt`, the `let` of `copy`, with its own node identities,
    /// locals, and row variables, and point the uses of `copy` at it.
    fn copy(&mut self, stmt: &Stmt<TypedRef<'db>>, copy: &LambdaCopy) -> Stmt<TypedRef<'db>> {
        let mut origins = Origins::default();
        walk_stmt(&mut origins, stmt);
        // Every local the statement binds gets its own identity in the copy,
        // and every row variable a lambda in it quantifies its own class.
        let mut locals = HashMap::default();
        let mut renaming = RowSubst::new();
        let mut renamed = 0;
        for (pattern, local) in &origins.binds {
            let fresh = if *pattern == copy.binding {
                copy.local
            } else {
                self.next_local += 1;
                LocalId::new(self.next_local - 1)
            };
            locals.insert(*local, fresh);
            for row in self.rows.get(pattern).into_iter().flatten() {
                // Checked row variables are numbered from zero per function.
                let id = (copy.variant.get() | 1 << 63).wrapping_add(renamed);
                renaming.insert(row.id, EffectRow::open(self.db, EffectVar { id }));
                renamed += 1;
            }
        }
        let rename = |ty| TypeSubst::new().apply_with_rows(self.db, ty, &renaming);
        let mut stmt = stmt.clone();
        walk_stmt_mut(
            &mut Rebind {
                variant: copy.variant,
                locals: &locals,
                rename: &rename,
            },
            &mut stmt,
        );
        super::clone_metadata(self.db, self.metadata, copy.variant, &[], &origins.nodes);
        for origin in &origins.nodes {
            let node = origin.with_variant(copy.variant);
            // A use in the copy can read a binding outside it.
            let outer = self
                .metadata
                .local_instances
                .get(origin)
                .map(|instance| instance.binding)
                .filter(|binding| !origins.nodes.contains(binding));
            if let Some(instance) = self.metadata.local_instances.get_mut(&node) {
                if let Some(binding) = outer {
                    instance.binding = binding;
                }
                if let Some(local) = locals.get(&instance.local) {
                    instance.local = *local;
                }
            }
            super::map_metadata_types(self.db, self.metadata, node, &rename, &renaming);
        }
        for node in &copy.uses {
            if let Some(instance) = self.metadata.local_instances.get_mut(node) {
                instance.binding = copy.binding.with_variant(copy.variant);
                instance.local = copy.local;
            }
        }
        stmt
    }
}

impl<'db> VisitMut<TypedRef<'db>> for Split<'_, 'db> {
    fn visit_expr_mut(&mut self, expr: &mut Expr<TypedRef<'db>>) {
        walk_expr_mut(self, expr);
        let ExprKind::Block { stmts, .. } = &mut *expr.kind else {
            return;
        };
        let mut index = 0;
        while index < stmts.len() {
            let Stmt::Let { pattern, .. } = &stmts[index] else {
                index += 1;
                continue;
            };
            let binding = pattern.id;
            let copies = self.copies;
            for copy in copies.iter().filter(|copy| copy.binding == binding) {
                let stmt = self.copy(&stmts[index], copy);
                index += 1;
                stmts.insert(index, stmt);
            }
            index += 1;
        }
    }

    fn visit_ref_mut(&mut self, site: RefSite, node: NodeId, value: &mut TypedRef<'db>) {
        if site == RefSite::Var
            && let ResolvedRef::Local { id, .. } = &mut value.resolved
            && let Some(copy) = self.copies.iter().find(|copy| copy.uses.contains(&node))
        {
            *id = copy.local;
        }
    }
}

/// The first local identity a function does not use.
#[derive(Default)]
struct Locals {
    next: u32,
}

impl Locals {
    fn reserve(&mut self, local: Option<LocalId>) {
        if let Some(local) = local.filter(|local| !local.is_unresolved()) {
            self.next = self.next.max(local.raw() + 1);
        }
    }
}

impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for Locals {
    fn visit_expr(&mut self, expr: &'ast Expr<TypedRef<'db>>) {
        match &*expr.kind {
            ExprKind::Lambda { params, .. } => {
                for param in params {
                    self.reserve(param.local_id);
                }
            }
            ExprKind::Resume { local_id, .. } => self.reserve(*local_id),
            _ => {}
        }
        walk_expr(self, expr);
    }

    fn visit_handler_arm(&mut self, arm: &'ast HandlerArm<TypedRef<'db>>) {
        if let HandlerKind::Op {
            resume_local_id, ..
        } = &arm.kind
        {
            self.reserve(*resume_local_id);
        }
        walk_handler_arm(self, arm);
    }

    fn visit_pattern(&mut self, pattern: &'ast Pattern<TypedRef<'db>>) {
        match &*pattern.kind {
            PatternKind::Bind { local_id, .. } | PatternKind::As { local_id, .. } => {
                self.reserve(*local_id);
            }
            PatternKind::ListRest { rest_local_id, .. } => self.reserve(*rest_local_id),
            _ => {}
        }
        walk_pattern(self, pattern);
    }

    fn visit_ref(&mut self, _: RefSite, _: NodeId, value: &'ast TypedRef<'db>) {
        if let ResolvedRef::Local { id, .. } = &value.resolved {
            self.reserve(Some(*id));
        }
    }
}

/// The nodes of one statement and the locals it binds, by their pattern.
#[derive(Default)]
struct Origins {
    nodes: HashSet<NodeId>,
    binds: Vec<(NodeId, LocalId)>,
}

impl<'ast, V: 'ast> Visit<'ast, V> for Origins {
    fn visit_node_id(&mut self, id: NodeId) {
        self.nodes.insert(id);
    }

    fn visit_pattern(&mut self, pattern: &'ast Pattern<V>) {
        if let PatternKind::Bind {
            local_id: Some(local),
            ..
        } = &*pattern.kind
        {
            self.binds.push((pattern.id, *local));
        }
        walk_pattern(self, pattern);
    }
}

/// Gives a copied statement its own node identities, locals, and row
/// variables.
struct Rebind<'a, 'db> {
    variant: NonZero<u64>,
    locals: &'a HashMap<LocalId, LocalId>,
    rename: &'a dyn Fn(Type<'db>) -> Type<'db>,
}

impl<'db> VisitMut<TypedRef<'db>> for Rebind<'_, 'db> {
    fn visit_ref_mut(&mut self, _: RefSite, _: NodeId, value: &mut TypedRef<'db>) {
        value.ty = (self.rename)(value.ty);
        if let ResolvedRef::Local { id, .. } = &mut value.resolved
            && let Some(local) = self.locals.get(id)
        {
            *id = *local;
        }
    }

    fn visit_node_id_mut(&mut self, id: &mut NodeId) {
        *id = id.with_variant(self.variant);
    }

    fn visit_pattern_mut(&mut self, pattern: &mut Pattern<TypedRef<'db>>) {
        if let PatternKind::Bind {
            local_id: Some(local),
            ..
        } = &mut *pattern.kind
            && let Some(fresh) = self.locals.get(local)
        {
            *local = *fresh;
        }
        walk_pattern_mut(self, pattern);
    }
}
