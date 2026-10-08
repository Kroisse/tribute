//! Identity of a specialized function instance: its type arguments and the
//! convention class of each class variable.

use std::hash::{Hash, Hasher};
use std::num::NonZero;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use trunk_ir::Symbol;

use super::MonomorphizeMetadata;
use crate::ast::visit::Visit;
use crate::ast::{
    AbilityId, CallingConvention, Decl, EffectVar, FuncDecl, FuncDefId, Module, NodeId, RowClasses,
    Type, TypeKind, TypeScheme, TypedRef, calling_convention_for_effect_row_in,
    collect_effect_vars,
};
use crate::typeck::FunctionInstance;

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

/// The convention class of each class variable of every function instance,
/// by the instance's name.
pub type InstanceRowClasses = HashMap<Symbol, Vec<(EffectVar, CallingConvention)>>;

/// What a reference needs to select an instance: the class variables of each
/// definition, the convention each ability requires, and the classes of the
/// instances made so far.
pub(crate) struct InstanceKeys<'db> {
    db: &'db dyn salsa::Database,
    /// Class variables of each definition with their binder positions.
    class_variables: HashMap<FuncDefId<'db>, Vec<(usize, EffectVar)>>,
    abilities: HashMap<AbilityId<'db>, CallingConvention>,
    pub(crate) row_classes: InstanceRowClasses,
}

impl<'db> InstanceKeys<'db> {
    pub(crate) fn new(
        db: &'db dyn salsa::Database,
        module: &Module<TypedRef<'db>>,
        function_types: &[(Symbol, TypeScheme<'db>)],
        metadata: &MonomorphizeMetadata<'db>,
    ) -> Self {
        let schemes: HashMap<&Symbol, TypeScheme<'db>> = function_types
            .iter()
            .map(|(name, scheme)| (name, *scheme))
            .collect();
        let mut definitions = Vec::new();
        let mut prefix = String::new();
        for_each_function(&module.decls, &mut prefix, &mut |name, function| {
            if let Some(scheme) = schemes.get(&name) {
                let id = FuncDefId::new(db, name);
                definitions.push((id, Definition::new(db, *scheme, function, metadata)));
            }
        });
        let class_variables = settle_class_variables(definitions);
        Self {
            db,
            class_variables,
            abilities: metadata.ability_conventions.clone(),
            row_classes: HashMap::default(),
        }
    }

    /// The key of the reference `instance` inside the definition `enclosing`,
    /// or `None` when it selects the definition itself.
    pub(crate) fn key(
        &self,
        instance: &FunctionInstance<'db>,
        enclosing: Option<&Symbol>,
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
                        calling_convention_for_effect_row_in(
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

    /// Record the classes of the instance `name` of `function` that `key`
    /// selects.
    pub(crate) fn record(
        &mut self,
        name: Symbol,
        function: FuncDefId<'db>,
        key: &InstanceKey<'db>,
    ) {
        if !key.has_weaker_class() {
            return;
        }
        let variables = &self.class_variables[&function];
        let classes = variables
            .iter()
            .map(|(_, variable)| *variable)
            .zip(key.class_args.iter().copied())
            .collect();
        self.row_classes.insert(name, classes);
    }
}

fn for_each_function<'a, 'db>(
    decls: &'a [Decl<TypedRef<'db>>],
    prefix: &mut String,
    f: &mut impl FnMut(Symbol, &'a FuncDecl<TypedRef<'db>>),
) {
    for decl in decls {
        match decl {
            Decl::Function(function) => {
                f(crate::qualified_symbol(prefix, &function.name), function)
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    let len = crate::push_prefix(prefix, &module.name);
                    for_each_function(body, prefix, f);
                    prefix.truncate(len);
                }
            }
            _ => {}
        }
    }
}

/// What decides the class variables of one definition.
struct Definition<'db> {
    /// The row variables the scheme quantifies, in binder order, without
    /// those that occur inside an ability argument of the signature or the
    /// body.
    candidates: Vec<(usize, EffectVar)>,
    /// The candidates known to be class variables.
    selected: HashSet<EffectVar>,
    /// The tail of each row argument of every function reference in the
    /// body, by the referenced definition.
    references: Vec<(FuncDefId<'db>, Vec<Option<EffectVar>>)>,
}

impl<'db> Definition<'db> {
    fn new(
        db: &'db dyn salsa::Database,
        scheme: TypeScheme<'db>,
        function: &FuncDecl<TypedRef<'db>>,
        metadata: &MonomorphizeMetadata<'db>,
    ) -> Self {
        struct Body<'a, 'db> {
            db: &'db dyn salsa::Database,
            metadata: &'a MonomorphizeMetadata<'db>,
            excluded: HashSet<EffectVar>,
            references: Vec<(FuncDefId<'db>, Vec<Option<EffectVar>>)>,
        }
        impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for Body<'_, 'db> {
            fn visit_node_id(&mut self, id: NodeId) {
                let handler = self.metadata.handler_operations.get(&id);
                let perform = self.metadata.perform_operations.get(&id);
                let arguments = handler
                    .map(|operation| &operation.ability_args)
                    .into_iter()
                    .chain(perform.map(|operation| &operation.ability_args))
                    .flatten();
                for argument in arguments {
                    self.excluded
                        .extend(collect_effect_vars(self.db, *argument));
                }
                if let Some(ty) = self.metadata.node_types.get(&id) {
                    collect_ability_argument_tails(self.db, *ty, &mut self.excluded);
                }
                if let Some(instance) = self.metadata.function_instances.get(&id) {
                    let tails = instance
                        .row_arguments
                        .iter()
                        .map(|row| row.rest(self.db))
                        .collect();
                    self.references.push((instance.function, tails));
                }
            }
        }
        let mut body = Body {
            db,
            metadata,
            excluded: HashSet::default(),
            references: Vec::new(),
        };
        collect_ability_argument_tails(db, scheme.body(db), &mut body.excluded);
        body.visit_func_decl(function);

        let mut in_parameters = Vec::new();
        if let TypeKind::Func { params, .. } = scheme.body(db).kind(db) {
            for param in params {
                collect_callable_tails(db, *param, &mut in_parameters);
            }
        }
        let candidates: Vec<_> = scheme
            .effect_params(db)
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, variable)| !body.excluded.contains(variable))
            .collect();
        let selected = candidates
            .iter()
            .map(|(_, variable)| *variable)
            .filter(|variable| in_parameters.contains(variable))
            .collect();
        Self {
            candidates,
            selected,
            references: body.references,
        }
    }
}

/// Select the class variables of every definition: a candidate that occurs
/// as the row of a function type inside a parameter type, or that is the tail
/// of a row the body passes for a class variable of a definition it
/// references.
fn settle_class_variables<'db>(
    mut definitions: Vec<(FuncDefId<'db>, Definition<'db>)>,
) -> HashMap<FuncDefId<'db>, Vec<(usize, EffectVar)>> {
    let positions = |definition: &Definition<'db>| -> Vec<(usize, EffectVar)> {
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
    found: &mut Vec<EffectVar>,
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
fn collect_ability_argument_tails<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    excluded: &mut HashSet<EffectVar>,
) {
    let row = |effect: crate::ast::EffectRow<'db>, excluded: &mut HashSet<EffectVar>| {
        for ability in effect.effects(db) {
            for arg in &ability.args {
                excluded.extend(collect_effect_vars(db, *arg));
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

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

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
