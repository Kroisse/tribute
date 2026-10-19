//! Identity of a specialized function instance: its type arguments and the
//! convention class of each class variable.

use std::hash::{Hash, Hasher};
use std::num::NonZero;
use trunk_ir::SymbolPath;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use trunk_ir::Symbol;

use super::MonomorphizeMetadata;
use crate::ast::visit::{Visit, walk_expr, walk_pattern};
use crate::ast::{
    AbilityId, AbilityRow, CallingConvention, Decl, DeclEntry, DeclTable, Expr, ExprKind, FuncDecl,
    FuncDefId, NodeId, Pattern, PatternKind, RowClasses, RowVar, Stmt, Type, TypeKind, TypeScheme,
    TypedRef, calling_convention_for_ability_row_in, collect_row_vars,
};
use crate::typeck::{
    FunctionInstance, FunctionInstanceOrigin, LambdaSignature, LocalCallableInstance,
};

/// The arguments that select one instance of a function definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InstanceKey<'db> {
    pub(crate) type_args: Vec<Type<'db>>,
    /// One class per class variable of the definition, in binder order.
    pub(crate) class_args: Vec<CallingConvention>,
}

impl<'db> InstanceKey<'db> {
    #[cfg(test)]
    pub(crate) fn of_types(type_args: Vec<Type<'db>>) -> Self {
        Self {
            type_args,
            class_args: Vec::new(),
        }
    }

    /// Whether some class is weaker than `Cps`, the class of a row variable
    /// that is not specialized.
    pub(crate) fn has_weaker_class(&self) -> bool {
        self.class_args
            .iter()
            .any(|class| *class != CallingConvention::Cps)
    }

    /// The NodeId variant of this instance's clone. Classes that are all
    /// `Cps` do not take part, as in the instance's name.
    pub(crate) fn variant(&self) -> NonZero<u64> {
        if !self.has_weaker_class() {
            return super::specialize::type_args_variant(&self.type_args);
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        NonZero::new(hasher.finish()).unwrap_or(NonZero::<u64>::MIN)
    }
}

/// The name of the root entry function, which is its own instance.
pub(crate) const ROOT_MAIN: &str = "main";

/// The convention class of each class variable of every function instance,
/// by the instance's name.
pub type InstanceRowClasses = HashMap<SymbolPath, Vec<(RowVar, CallingConvention)>>;

/// What a reference needs to select an instance: the class variables of each
/// definition, the convention each ability requires, and the classes of the
/// instances made so far.
pub(crate) struct InstanceKeys<'db> {
    db: &'db dyn salsa::Database,
    /// Class variables of each definition with their binder positions.
    class_variables: HashMap<FuncDefId<'db>, Vec<(usize, RowVar)>>,
    abilities: HashMap<AbilityId<'db>, CallingConvention>,
    pub(crate) row_classes: InstanceRowClasses,
    /// The definition each instance made so far specializes.
    origins: HashMap<SymbolPath, FuncDefId<'db>>,
}

impl<'db> InstanceKeys<'db> {
    pub(crate) fn new(
        db: &'db dyn salsa::Database,
        module: &DeclTable<TypedRef<'db>>,
        function_types: &[(SymbolPath, TypeScheme<'db>)],
        metadata: &MonomorphizeMetadata<'db>,
    ) -> Self {
        let schemes: HashMap<&SymbolPath, TypeScheme<'db>> = function_types
            .iter()
            .map(|(name, scheme)| (name, *scheme))
            .collect();
        let mut definitions = Vec::new();
        let mut accessors = HashMap::default();
        for entry in &module.entries {
            let Decl::Function(function) = &entry.decl else {
                continue;
            };
            let name = entry.module.child(function.name.clone());
            if let Some(scheme) = schemes.get(&name) {
                let id = FuncDefId::new(db, name);
                let definition = Definition::new(db, *scheme, function, metadata, &mut accessors);
                definitions.push((id, definition));
            }
        }
        // A field function has no body, so its signature alone decides.
        definitions.extend(accessors.into_iter().map(|(id, scheme)| {
            let definition = Definition::of_signature(db, scheme, HashSet::default(), Vec::new());
            (id, definition)
        }));
        let class_variables = settle_class_variables(definitions);
        // A definition is its own instance with every class `Cps`, except
        // root `main`: nothing instantiates its row, so its tail is empty.
        let root = FuncDefId::new(db, SymbolPath::new([ROOT_MAIN]));
        let row_classes = class_variables
            .iter()
            .map(|(id, variables)| {
                let class = if *id == root {
                    CallingConvention::Direct
                } else {
                    CallingConvention::Cps
                };
                let classes = variables.iter().map(|(_, variable)| (*variable, class));
                (id.path(db).clone(), classes.collect())
            })
            .collect();
        Self {
            db,
            class_variables,
            abilities: metadata.ability_conventions.clone(),
            row_classes,
            origins: HashMap::default(),
        }
    }

    /// The key of the reference `instance` inside the definition `enclosing`,
    /// or `None` when it selects the definition itself.
    pub(crate) fn key(
        &self,
        instance: &FunctionInstance<'db>,
        enclosing: Option<&[Symbol]>,
    ) -> Option<InstanceKey<'db>> {
        let enclosing: &RowClasses = enclosing
            .and_then(|name| self.row_classes.get(name))
            .map_or(&[], Vec::as_slice);
        let class_args = self
            .class_variables
            .get(&instance.function)
            .into_iter()
            .flatten()
            .map(|(position, _)| {
                instance
                    .row_arguments
                    .get(*position)
                    .map_or(CallingConvention::Cps, |row| {
                        calling_convention_for_ability_row_in(
                            self.db,
                            *row,
                            &self.abilities,
                            enclosing,
                        )
                    })
            })
            .collect();
        let key = InstanceKey {
            type_args: instance.type_arguments.clone(),
            class_args,
        };
        (!key.type_args.is_empty() || key.has_weaker_class()).then_some(key)
    }

    /// The class of each of `rows` inside the instance `enclosing`.
    pub(crate) fn classes_of_rows(
        &self,
        rows: &[crate::ast::AbilityRow<'db>],
        enclosing: &[Symbol],
    ) -> Vec<CallingConvention> {
        let classes: &RowClasses = self.row_classes.get(enclosing).map_or(&[], Vec::as_slice);
        rows.iter()
            .map(|row| {
                calling_convention_for_ability_row_in(self.db, *row, &self.abilities, classes)
            })
            .collect()
    }

    /// Fix the class of each row variable a let-bound lambda of `decls`
    /// quantifies, inside the instance that binds it: the join of the classes
    /// its uses select. Returns whether a class changed.
    pub(crate) fn settle_local_classes(
        &mut self,
        decls: &[DeclEntry<TypedRef<'db>>],
        metadata: &MonomorphizeMetadata<'db>,
    ) -> bool {
        let mut changed = false;
        for_each_declaration(decls, &mut |name, decl| {
            let (Some(name), Decl::Function(function)) = (name, decl) else {
                return;
            };
            let enclosing = self.row_classes.get(name);
            let mut classes = enclosing.cloned().unwrap_or_default();
            let mut locals: Vec<(RowVar, CallingConvention)> = Vec::new();
            let instantiations = local_instantiations(
                self.db,
                function,
                |node| metadata.local_instances.get(node),
                |node| metadata.lambda_signatures.get(node),
            );
            for (variable, row) in instantiations.into_iter().flat_map(|use_| use_.rows) {
                let class =
                    calling_convention_for_ability_row_in(self.db, row, &self.abilities, &classes);
                match locals.iter_mut().find(|(local, _)| *local == variable) {
                    Some((_, joined)) => *joined = joined.join(class),
                    None => locals.push((variable, class)),
                }
            }
            if locals.is_empty() {
                return;
            }
            classes.retain(|(variable, _)| !locals.iter().any(|(local, _)| local == variable));
            classes.extend(locals);
            if enclosing != Some(&classes) {
                self.row_classes
                    .insert(SymbolPath::new(name.iter().cloned()), classes);
                changed = true;
            }
        });
        changed
    }

    /// Whether the reference to `callee` is a reference to `function`: the
    /// definition itself, or an instance made of it, which a clone of an
    /// already rewritten body names and which its own classes may change.
    pub(crate) fn refers_to(&self, callee: FuncDefId<'db>, function: FuncDefId<'db>) -> bool {
        callee == function || self.origins.get(callee.path(self.db)) == Some(&function)
    }

    /// Record the classes of the instance `name` of `function` that `key`
    /// selects.
    pub(crate) fn record(
        &mut self,
        name: SymbolPath,
        function: FuncDefId<'db>,
        key: &InstanceKey<'db>,
    ) {
        self.origins.insert(name.clone(), function);
        let Some(variables) = self.class_variables.get(&function) else {
            return;
        };
        let classes = variables
            .iter()
            .map(|(_, variable)| *variable)
            .zip(key.class_args.iter().copied())
            .collect();
        self.row_classes.insert(name, classes);
    }
}

/// One use of a let-bound lambda: the row it passes for each row variable
/// the lambda's type spells, in the order its scheme quantifies them.
pub(crate) struct LocalInstantiation<'db> {
    pub(crate) site: NodeId,
    pub(crate) binding: NodeId,
    pub(crate) rows: Vec<(RowVar, AbilityRow<'db>)>,
}

/// Each use of a let-bound lambda in `function`, in use order. A use whose
/// lambda does not spell every row its scheme quantifies passes none.
pub(crate) fn local_instantiations<'a, 'db: 'a>(
    db: &'db dyn salsa::Database,
    function: &FuncDecl<TypedRef<'db>>,
    local_instance: impl Fn(&NodeId) -> Option<&'a LocalCallableInstance<'db>>,
    lambda_signature: impl Fn(&NodeId) -> Option<&'a LambdaSignature<'db>>,
) -> Vec<LocalInstantiation<'db>> {
    // The lambda each `let` binds, by its pattern.
    let mut bound = HashMap::default();
    function.body.for_each(|expr| {
        let ExprKind::Block { stmts, .. } = &*expr.kind else {
            return;
        };
        for stmt in stmts {
            if let Stmt::Let { pattern, value, .. } = stmt
                && matches!(&*value.kind, ExprKind::Lambda { .. })
            {
                bound.insert(pattern.id, value.id);
            }
        }
    });
    let mut instantiations = Vec::new();
    function.body.for_each(|expr| {
        let Some(instance) = local_instance(&expr.id) else {
            return;
        };
        let Some(signature) = bound.get(&instance.binding).and_then(&lambda_signature) else {
            return;
        };
        // The lambda's own type spells the rows its scheme quantifies with
        // the variables its body uses.
        let quantified = collect_row_vars(db, instance.scheme.body(db));
        let spelled = collect_row_vars(db, signature.function_type);
        if quantified.len() != spelled.len() {
            return;
        }
        let variables = instance.scheme.effect_params(db);
        let rows = variables
            .iter()
            .zip(&instance.row_arguments)
            .filter_map(|(variable, row)| {
                let at = quantified.iter().position(|var| var == variable)?;
                Some((spelled[at], *row))
            })
            .collect();
        instantiations.push(LocalInstantiation {
            site: expr.id,
            binding: instance.binding,
            rows,
        });
    });
    instantiations
}

/// Calls `f` on each declaration of `entries`, with the path of a function.
/// That path is the instance its references are inside.
pub(super) fn for_each_declaration<'a, 'db>(
    entries: &'a [DeclEntry<TypedRef<'db>>],
    f: &mut impl FnMut(Option<&[Symbol]>, &'a Decl<TypedRef<'db>>),
) {
    for entry in entries {
        let path = function_path(entry);
        f(path.as_deref(), &entry.decl);
    }
}

/// [`for_each_declaration`] over declarations to change.
pub(super) fn for_each_declaration_mut<'db>(
    entries: &mut [DeclEntry<TypedRef<'db>>],
    f: &mut impl FnMut(Option<&[Symbol]>, &mut Decl<TypedRef<'db>>),
) {
    for entry in entries {
        let path = function_path(entry);
        f(path.as_deref(), &mut entry.decl);
    }
}

/// The path of the function `entry` declares.
fn function_path(entry: &DeclEntry<TypedRef<'_>>) -> Option<SymbolPath> {
    match &entry.decl {
        Decl::Function(function) => Some(entry.module.child(function.name.clone())),
        _ => None,
    }
}

/// What decides the class variables of one definition.
struct Definition<'db> {
    /// The row variables the scheme quantifies, in binder order, without
    /// those that occur inside an ability argument of the signature or the
    /// body.
    candidates: Vec<(usize, RowVar)>,
    /// The candidates known to be class variables.
    selected: HashSet<RowVar>,
    /// The tail of each row argument of every function reference in the
    /// body, by the referenced definition.
    references: Vec<(FuncDefId<'db>, Vec<Option<RowVar>>)>,
}

impl<'db> Definition<'db> {
    fn new(
        db: &'db dyn salsa::Database,
        scheme: TypeScheme<'db>,
        function: &FuncDecl<TypedRef<'db>>,
        metadata: &MonomorphizeMetadata<'db>,
        accessors: &mut HashMap<FuncDefId<'db>, TypeScheme<'db>>,
    ) -> Self {
        struct Body<'a, 'db> {
            db: &'db dyn salsa::Database,
            metadata: &'a MonomorphizeMetadata<'db>,
            accessors: &'a mut HashMap<FuncDefId<'db>, TypeScheme<'db>>,
            excluded: HashSet<RowVar>,
            callable_tails: Vec<RowVar>,
            references: Vec<(FuncDefId<'db>, Vec<Option<RowVar>>)>,
        }
        impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for Body<'_, 'db> {
            fn visit_expr(&mut self, expr: &'ast Expr<TypedRef<'db>>) {
                if matches!(&*expr.kind, ExprKind::Lambda { .. }) {
                    self.callable(expr.id);
                }
                walk_expr(self, expr);
            }

            fn visit_pattern(&mut self, pattern: &'ast Pattern<TypedRef<'db>>) {
                if matches!(&*pattern.kind, PatternKind::Bind { .. }) {
                    self.callable(pattern.id);
                }
                walk_pattern(self, pattern);
            }

            fn visit_node_id(&mut self, id: NodeId) {
                let handler = self.metadata.handler_operations.get(&id);
                let perform = self.metadata.perform_operations.get(&id);
                let arguments = handler
                    .map(|operation| &operation.ability_args)
                    .into_iter()
                    .chain(perform.map(|operation| &operation.ability_args))
                    .flatten();
                for argument in arguments {
                    self.excluded.extend(collect_row_vars(self.db, *argument));
                }
                if let Some(ty) = self.metadata.node_types.get(&id) {
                    collect_ability_argument_tails(self.db, *ty, &mut self.excluded);
                }
                // A row passed for the row variable of a let-bound callable
                // decides that callable's class.
                if let Some(local) = self.metadata.local_instances.get(&id) {
                    let tails = local
                        .row_arguments
                        .iter()
                        .filter_map(|row| row.rest(self.db));
                    self.callable_tails.extend(tails);
                }
                if let Some(instance) = self.metadata.function_instances.get(&id) {
                    if matches!(
                        instance.origin,
                        FunctionInstanceOrigin::FieldAccessor { .. }
                    ) {
                        self.accessors.insert(instance.function, instance.scheme);
                    }
                    let tails = instance
                        .row_arguments
                        .iter()
                        .map(|row| row.rest(self.db))
                        .collect();
                    self.references.push((instance.function, tails));
                }
            }
        }
        impl Body<'_, '_> {
            /// Record the row tails of the callable type of a lambda or a
            /// local binding.
            fn callable(&mut self, node: NodeId) {
                if let Some(ty) = self.metadata.node_types.get(&node) {
                    collect_callable_tails(self.db, *ty, &mut self.callable_tails);
                }
            }
        }
        let mut body = Body {
            db,
            metadata,
            accessors,
            excluded: HashSet::default(),
            callable_tails: Vec::new(),
            references: Vec::new(),
        };
        body.visit_func_decl(function);

        let mut definition = Self::of_signature(db, scheme, body.excluded, body.callable_tails);
        definition.references = body.references;
        definition
    }

    /// The definition of `scheme` whose body has the given row variables in
    /// ability arguments and callable tails, and no references.
    fn of_signature(
        db: &'db dyn salsa::Database,
        scheme: TypeScheme<'db>,
        mut excluded: HashSet<RowVar>,
        mut callable_tails: Vec<RowVar>,
    ) -> Self {
        collect_ability_argument_tails(db, scheme.body(db), &mut excluded);
        if let TypeKind::Func { params, result, .. } = scheme.body(db).kind(db) {
            for ty in params.iter().chain([result]) {
                collect_callable_tails(db, *ty, &mut callable_tails);
            }
        }
        let candidates: Vec<_> = scheme
            .effect_params(db)
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, variable)| !excluded.contains(variable))
            .collect();
        let selected = candidates
            .iter()
            .map(|(_, variable)| *variable)
            .filter(|variable| callable_tails.contains(variable))
            .collect();
        Self {
            candidates,
            selected,
            references: Vec::new(),
        }
    }
}

/// Select the class variables of every definition: a candidate that is the
/// row tail of a function type inside a parameter type or the result type,
/// of a lambda or a local binding in the body, or of a row the body passes
/// for a class variable of a definition it references.
fn settle_class_variables<'db>(
    mut definitions: Vec<(FuncDefId<'db>, Definition<'db>)>,
) -> HashMap<FuncDefId<'db>, Vec<(usize, RowVar)>> {
    let positions = |definition: &Definition<'db>| -> Vec<(usize, RowVar)> {
        definition
            .candidates
            .iter()
            .copied()
            .filter(|(_, variable)| definition.selected.contains(variable))
            .collect()
    };
    loop {
        let class_positions: HashMap<FuncDefId<'db>, Vec<usize>> = definitions
            .iter()
            .map(|(id, definition)| {
                let positions = positions(definition).into_iter().map(|(at, _)| at);
                (*id, positions.collect())
            })
            .collect();
        let mut changed = false;
        for (_, definition) in &mut definitions {
            for (callee, tails) in &definition.references {
                let Some(positions) = class_positions.get(callee) else {
                    continue;
                };
                for tail in positions
                    .iter()
                    .filter_map(|at| tails.get(*at).copied().flatten())
                {
                    let candidate = definition.candidates.iter().any(|(_, var)| *var == tail);
                    changed |= candidate && definition.selected.insert(tail);
                }
            }
        }
        if !changed {
            break;
        }
    }
    definitions
        .iter()
        .map(|(id, definition)| (*id, positions(definition)))
        .filter(|(_, variables)| !variables.is_empty())
        .collect()
}

fn collect_callable_tails<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    found: &mut Vec<RowVar>,
) {
    match ty.kind(db) {
        TypeKind::Named { args, .. } | TypeKind::Tuple(args) => {
            for arg in args {
                collect_callable_tails(db, *arg, found);
            }
        }
        TypeKind::App { ctor, args } => {
            collect_callable_tails(db, *ctor, found);
            for arg in args {
                collect_callable_tails(db, *arg, found);
            }
        }
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            for param in params {
                collect_callable_tails(db, *param, found);
            }
            collect_callable_tails(db, *result, found);
            found.extend(effect.rest(db));
        }
        _ => {}
    }
}

/// Collect the row variables that occur inside an ability argument of a row
/// anywhere in `ty`.
pub(crate) fn collect_ability_argument_tails<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    excluded: &mut HashSet<RowVar>,
) {
    let row = |effect: crate::ast::AbilityRow<'db>, excluded: &mut HashSet<RowVar>| {
        for ability in effect.instances(db) {
            for arg in &ability.args {
                excluded.extend(collect_row_vars(db, *arg));
            }
        }
    };
    match ty.kind(db) {
        TypeKind::Named { args, .. } | TypeKind::Tuple(args) => {
            for arg in args {
                collect_ability_argument_tails(db, *arg, excluded);
            }
        }
        TypeKind::App { ctor, args } => {
            collect_ability_argument_tails(db, *ctor, excluded);
            for arg in args {
                collect_ability_argument_tails(db, *arg, excluded);
            }
        }
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            for param in params {
                collect_ability_argument_tails(db, *param, excluded);
            }
            collect_ability_argument_tails(db, *result, excluded);
            row(*effect, excluded);
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            collect_ability_argument_tails(db, *arg, excluded);
            collect_ability_argument_tails(db, *result, excluded);
            row(*effect, excluded);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityInstance, AbilityRow};

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    fn func<'db>(
        db: &'db TestDb,
        params: Vec<Type<'db>>,
        result: Type<'db>,
        effect: AbilityRow<'db>,
    ) -> Type<'db> {
        Type::new(
            db,
            TypeKind::Func {
                params,
                result,
                effect,
            },
        )
    }

    fn open(db: &TestDb, id: u64) -> AbilityRow<'_> {
        AbilityRow::open(db, RowVar { id })
    }

    fn definition<'db>(db: &'db TestDb, row_vars: &[u64], body: Type<'db>) -> Definition<'db> {
        let effect_params = row_vars.iter().map(|id| RowVar { id: *id }).collect();
        let scheme = TypeScheme::new(db, Vec::new(), effect_params, body);
        Definition::of_signature(db, scheme, HashSet::default(), Vec::new())
    }

    fn id<'db>(db: &'db TestDb, name: &str) -> FuncDefId<'db> {
        FuncDefId::new(db, trunk_ir::SymbolPath::new([name]))
    }

    #[test]
    fn a_tail_of_a_callable_parameter_is_a_class_variable() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let first = func(&db, vec![], int, open(&db, 2));
        let nested = func(&db, vec![], int, open(&db, 1));
        let second = func(&db, vec![nested], int, AbilityRow::pure(&db));
        let both = func(&db, vec![first, second], int, AbilityRow::pure(&db));
        let settled =
            settle_class_variables(vec![(id(&db, "both"), definition(&db, &[1, 2], both))]);

        // Binder order, not the order of occurrence.
        assert_eq!(
            settled[&id(&db, "both")],
            [(0, RowVar { id: 1 }), (1, RowVar { id: 2 })]
        );
    }

    #[test]
    fn a_tail_that_reaches_no_callable_is_not_a_class_variable() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let own_row = func(&db, vec![int], int, open(&db, 1));
        let settled =
            settle_class_variables(vec![(id(&db, "own_row"), definition(&db, &[1], own_row))]);

        assert!(settled.is_empty());
    }

    #[test]
    fn the_tail_of_a_returned_function_type_is_a_class_variable() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let returned = func(&db, vec![], int, open(&db, 1));
        let result = func(&db, vec![int], returned, AbilityRow::pure(&db));
        let settled =
            settle_class_variables(vec![(id(&db, "result"), definition(&db, &[1], result))]);

        assert_eq!(
            settled.get(&id(&db, "result")),
            Some(&vec![(0, RowVar { id: 1 })])
        );
    }

    #[test]
    fn a_tail_passed_for_a_class_variable_is_a_class_variable() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let callback = func(&db, vec![int], int, open(&db, 1));
        let app = func(&db, vec![int, callback], int, open(&db, 1));
        let own_row = func(&db, vec![int], int, open(&db, 7));
        // `outer` calls `inner` with its own tail, and `inner` calls `app`.
        let mut inner = definition(&db, &[7], own_row);
        inner.references = vec![(id(&db, "app"), vec![Some(RowVar { id: 7 })])];
        let mut outer = definition(&db, &[7], own_row);
        outer.references = vec![(id(&db, "inner"), vec![Some(RowVar { id: 7 })])];
        let mut closed = definition(&db, &[7], own_row);
        closed.references = vec![(id(&db, "app"), vec![None])];
        let settled = settle_class_variables(vec![
            (id(&db, "outer"), outer),
            (id(&db, "inner"), inner),
            (id(&db, "closed"), closed),
            (id(&db, "app"), definition(&db, &[1], app)),
        ]);

        assert_eq!(settled[&id(&db, "inner")], [(0, RowVar { id: 7 })]);
        assert_eq!(settled[&id(&db, "outer")], [(0, RowVar { id: 7 })]);
        assert!(!settled.contains_key(&id(&db, "closed")));
    }

    #[test]
    fn a_tail_in_an_ability_argument_is_not_a_class_variable() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let stored = func(&db, vec![], int, open(&db, 1));
        let ability = AbilityInstance {
            ability_id: AbilityId::source(&db, trunk_ir::SymbolPath::new(["State"])),
            args: vec![stored],
        };
        let callback = func(&db, vec![int], int, open(&db, 1));
        let body = func(
            &db,
            vec![callback],
            int,
            AbilityRow::new(&db, vec![ability], None),
        );
        let settled = settle_class_variables(vec![(id(&db, "run"), definition(&db, &[1], body))]);

        assert!(settled.is_empty());
    }

    #[test]
    fn classes_that_are_all_cps_select_the_same_clone_as_no_classes() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let plain = InstanceKey::of_types(vec![int]);
        let cps = InstanceKey {
            type_args: vec![int],
            class_args: vec![CallingConvention::Cps],
        };
        let direct = InstanceKey {
            type_args: vec![int],
            class_args: vec![CallingConvention::Direct],
        };

        assert_eq!(plain.variant(), cps.variant());
        assert_ne!(plain.variant(), direct.variant());
    }
}
