//! One copy of a let-bound lambda per convention class its uses select.

use std::hash::{Hash, Hasher};
use std::num::NonZero;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::MonomorphizeMetadata;
use super::instance::InstanceKeys;
use crate::ast::visit::{RefSite, Visit, VisitMut, walk_expr_mut, walk_stmt, walk_stmt_mut};
use crate::ast::{
    CallingConvention, Decl, EffectRow, EffectVar, Expr, ExprKind, LocalId, NodeId, PatternKind,
    ResolvedRef, Stmt, Type, TypedRef, collect_effect_vars,
};
use crate::typeck::{RowSubst, TypeSubst};

/// A copy of a let-bound lambda for the uses that select `classes`.
struct Copy {
    binding: NodeId,
    local: LocalId,
    variant: NonZero<u64>,
    uses: HashSet<NodeId>,
    /// The row variables the binding quantifies, as the lambda spells them.
    /// The copy gets its own, so that each copy has its own class.
    rows: Vec<EffectVar>,
}

impl Copy {
    fn renaming<'db>(&self, db: &'db dyn salsa::Database) -> RowSubst<'db> {
        let mut renaming = RowSubst::new();
        for (index, row) in self.rows.iter().enumerate() {
            // Checked row variables are numbered from zero per function.
            let id = (self.variant.get() | 1 << 63).wrapping_add(index as u64);
            renaming.insert(row.id, EffectRow::open(db, EffectVar { id }));
        }
        renaming
    }
}

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
        let mut next_local = 0;
        function.body.for_each(|expr| {
            if let ExprKind::Var(TypedRef {
                resolved: ResolvedRef::Local { id, .. },
                ..
            }) = &*expr.kind
            {
                next_local = next_local.max(id.raw().saturating_add(1));
            }
            let ExprKind::Block { stmts, .. } = &*expr.kind else {
                return;
            };
            for stmt in stmts {
                if let Stmt::Let { pattern, value, .. } = stmt
                    && let PatternKind::Bind {
                        local_id: Some(local),
                        ..
                    } = &*pattern.kind
                {
                    next_local = next_local.max(local.raw().saturating_add(1));
                    if matches!(&*value.kind, ExprKind::Lambda { .. }) {
                        bindings.insert(*local, (pattern.id, value.id));
                    }
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
        let mut schemes = HashMap::default();
        function.body.for_each(|expr| {
            let ExprKind::Var(TypedRef {
                resolved: ResolvedRef::Local { id, .. },
                ..
            }) = &*expr.kind
            else {
                return;
            };
            let Some((binding, _)) = bindings.get(id) else {
                return;
            };
            match metadata.local_instances.get(&expr.id) {
                Some(instance) if instance.binding == *binding => {
                    let classes = keys.classes_of_rows(&instance.row_arguments, &function.name);
                    uses.entry(*id).or_default().push((expr.id, classes));
                    schemes.insert(*id, instance.scheme);
                }
                _ => {
                    shared.insert(*id);
                }
            }
        });
        let mut copies = Vec::new();
        let mut uses: Vec<_> = uses.into_iter().collect();
        uses.sort_by_key(|(local, _)| local.raw());
        for (local, uses) in uses {
            if shared.contains(&local) {
                continue;
            }
            let (binding, lambda) = bindings[&local];
            let Some(signature) = metadata.lambda_signatures.get(&lambda) else {
                continue;
            };
            let scheme = schemes[&local];
            let quantified = collect_effect_vars(db, scheme.body(db));
            let spelled = collect_effect_vars(db, signature.function_type);
            if quantified.len() != spelled.len() {
                continue;
            }
            let rows: Vec<_> = scheme
                .effect_params(db)
                .iter()
                .filter_map(|row| quantified.iter().position(|var| var == row))
                .map(|at| spelled[at])
                .collect();
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
                copies.push(Copy {
                    binding,
                    local: LocalId::new(next_local),
                    variant: NonZero::new(hasher.finish()).unwrap_or(NonZero::<u64>::MIN),
                    uses: uses
                        .iter()
                        .filter(|(_, classes)| classes == selected)
                        .map(|(node, _)| *node)
                        .collect(),
                    rows: rows.clone(),
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
        }
        .visit_expr_mut(&mut function.body);
        split = true;
    }
    split
}

struct Split<'a, 'db> {
    db: &'db dyn salsa::Database,
    metadata: &'a mut MonomorphizeMetadata<'db>,
    copies: &'a [Copy],
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
            let mut origins = Origins::default();
            walk_stmt(&mut origins, &stmts[index]);
            let copies: Vec<_> = self
                .copies
                .iter()
                .filter(|copy| copy.binding == binding)
                .collect();
            for copy in &copies {
                let renaming = copy.renaming(self.db);
                let rename = |ty| TypeSubst::new().apply_with_rows(self.db, ty, &renaming);
                let mut stmt = stmts[index].clone();
                walk_stmt_mut(
                    &mut Rebind {
                        copy,
                        rename: &rename,
                    },
                    &mut stmt,
                );
                super::clone_metadata(self.db, self.metadata, copy.variant, &[], &[], &origins.0);
                for origin in &origins.0 {
                    let node = origin.with_variant(copy.variant);
                    // A use in the copy can read a binding outside it.
                    if let Some(binding) = self
                        .metadata
                        .local_instances
                        .get(origin)
                        .map(|instance| instance.binding)
                        .filter(|binding| !origins.0.contains(binding))
                        && let Some(instance) = self.metadata.local_instances.get_mut(&node)
                    {
                        instance.binding = binding;
                    }
                    super::map_metadata_types(self.db, self.metadata, node, &rename, &renaming);
                }
                for node in &copy.uses {
                    if let Some(instance) = self.metadata.local_instances.get_mut(node) {
                        instance.binding = binding.with_variant(copy.variant);
                        instance.local = copy.local;
                    }
                }
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

/// The nodes of one statement.
#[derive(Default)]
struct Origins(HashSet<NodeId>);

impl<'ast, V: 'ast> Visit<'ast, V> for Origins {
    fn visit_node_id(&mut self, id: NodeId) {
        self.0.insert(id);
    }
}

/// Gives a copied `let` its own node identities, the local it binds, and
/// its own row variables.
struct Rebind<'a, 'db> {
    copy: &'a Copy,
    rename: &'a dyn Fn(Type<'db>) -> Type<'db>,
}

impl<'db> VisitMut<TypedRef<'db>> for Rebind<'_, 'db> {
    fn visit_ref_mut(&mut self, _: RefSite, _: NodeId, value: &mut TypedRef<'db>) {
        value.ty = (self.rename)(value.ty);
    }

    fn visit_node_id_mut(&mut self, id: &mut NodeId) {
        *id = id.with_variant(self.copy.variant);
    }

    fn visit_pattern_mut(&mut self, pattern: &mut crate::ast::Pattern<TypedRef<'db>>) {
        if pattern.id == self.copy.binding
            && let PatternKind::Bind { local_id, .. } = &mut *pattern.kind
        {
            *local_id = Some(self.copy.local);
        }
        crate::ast::visit::walk_pattern_mut(self, pattern);
    }
}
