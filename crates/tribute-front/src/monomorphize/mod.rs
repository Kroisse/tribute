pub mod collect;
mod instance;

pub(crate) use instance::{
    LocalInstantiation, collect_ability_argument_tails, local_instantiations,
};
pub mod mangle;
mod nominal;
mod nominal_index;
mod package;
pub use package::{checked_dependencies, specialized_dependencies, specialized_package};
mod recursion;
mod result_check;
pub use result_check::SpecializationInvariant;
mod rewrite;
mod templates;
mod validate;
pub use validate::{InstanceError, InstanceErrorKind};
pub mod specialize;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use trunk_ir::{Symbol, SymbolPath};

use self::instance::InstanceKey;
use crate::SortedMap;
use crate::resolve::NameDisplay;
use std::ops::ControlFlow;

use crate::ast::fold::{ForEachType, TypeFoldable, TypeVisitable};
use crate::ast::{
    CtorId, Decl, DeclEntry, DeclTable, FuncDefId, NodeId, Type, TypeDefId, TypeScheme, TypedRef,
};
use crate::typeck::subst::{BoundVarOutOfBounds, BoundVarSubst};
use crate::typeck::{
    ConstructorTypeMetadata, DeclarationTypeMetadata, ExpressionTypeMetadata, TypeCheckOutput,
};

const MAX_TRANSITIVE_SPECIALIZATION_ROUNDS: usize = 64;

/// What specialization reads: one checked module, with the declarations of
/// the packages it depends on for lookup and, when merged, those packages'
/// bodies and per-expression metadata ahead of its own
/// (`new-plans/implementation.md`, "프론트엔드 단계 경계와 정보 소유권").
pub struct SpecializationInput<'db> {
    module: DeclTable<TypedRef<'db>>,
    declarations: DeclarationTypeMetadata<'db>,
    expressions: ExpressionTypeMetadata<'db>,
    compiler_intrinsics: SortedMap<NodeId, Symbol>,
    templates: Option<Templates<'db>>,
    names: NameDisplay,
}

/// The dependencies whose generic bodies a package's specialization clones
/// without specializing their bodies.
struct Templates<'db> {
    /// The module's own declarations, which its result extends.
    declarations: DeclarationTypeMetadata<'db>,
    /// Each dependency's checked result, each after the packages it depends
    /// on.
    dependencies: Vec<TypeCheckOutput<'db>>,
}

impl<'db> SpecializationInput<'db> {
    /// The checked module, with the declarations of `dependencies` under its
    /// own. `dependencies` are the packages it depends on, transitively,
    /// each after the packages it depends on. Diagnostics spell declarations
    /// as `names` does.
    pub fn new(
        db: &'db dyn salsa::Database,
        checked: TypeCheckOutput<'db>,
        dependencies: impl IntoIterator<Item = TypeCheckOutput<'db>>,
        names: NameDisplay,
    ) -> Self {
        let below = dependencies
            .into_iter()
            .fold(DeclarationTypeMetadata::default(), |below, dependency| {
                dependency.declarations(db).after_dependency(&below)
            });
        let declarations = checked.declarations(db).after_dependency(&below);
        Self {
            module: checked.module(db).clone(),
            declarations,
            expressions: checked.expression_types(db).clone(),
            compiler_intrinsics: SortedMap::default(),
            templates: None,
            names,
        }
    }

    /// The checked module of a package, reading the generic bodies of
    /// `dependencies` as templates. `dependencies` are the packages it
    /// depends on, transitively, each after the packages it depends on.
    /// Every instance the module reaches is specialized, a dependency's
    /// generic at the dependency's types included; the result has the
    /// module's own sources and instances, which
    /// [`merge_specializations`] merges with the dependencies' results.
    pub fn package(
        db: &'db dyn salsa::Database,
        checked: TypeCheckOutput<'db>,
        dependencies: impl IntoIterator<Item = TypeCheckOutput<'db>>,
        names: NameDisplay,
    ) -> Self {
        let dependencies: Vec<_> = dependencies.into_iter().collect();
        let mut input = Self::new(db, checked, dependencies.iter().copied(), names);
        if dependencies.is_empty() {
            return input;
        }
        // A dependency that registers an unsupported intrinsic reports it
        // in its own specialization.
        for dependency in &dependencies {
            if let Ok(registered) =
                crate::ast_to_ir::registered_compiler_intrinsics(dependency.module(db))
            {
                input.compiler_intrinsics.extend(registered);
            }
        }
        input.templates = Some(Templates {
            declarations: checked.declarations(db).clone(),
            dependencies,
        });
        input
    }

    /// Specialize the bodies of `dependencies` too, given in the order of
    /// [`SpecializationInput::new`]: their declarations come first in the
    /// module in that order, and their per-expression metadata under this
    /// module's.
    pub fn merge_bodies(
        mut self,
        db: &'db dyn salsa::Database,
        dependencies: impl IntoIterator<Item = TypeCheckOutput<'db>>,
    ) -> Self {
        let mut entries = Vec::new();
        let below = dependencies.into_iter().fold(
            ExpressionTypeMetadata::default(),
            |below, dependency| {
                entries.extend(dependency.module(db).entries.iter().cloned());
                dependency.expression_types(db).after_dependency(&below)
            },
        );
        entries.append(&mut self.module.entries);
        self.module.entries = entries;
        self.expressions = self.expressions.after_dependency(&below);
        self
    }

    /// Carry the compiler intrinsic identities of the module's declarations.
    pub fn with_compiler_intrinsics(
        mut self,
        intrinsics: impl IntoIterator<Item = (NodeId, Symbol)>,
    ) -> Self {
        self.compiler_intrinsics.extend(intrinsics);
        self
    }

    /// The declaration table specialization starts from.
    pub fn module(&self) -> &DeclTable<TypedRef<'db>> {
        &self.module
    }

    pub fn declarations(&self) -> &DeclarationTypeMetadata<'db> {
        &self.declarations
    }
}

/// The specialization result, the only input the lowering input is built
/// from: the declaration table with the concrete instances added, the
/// metadata substituted with it, and the compiler intrinsic identities.
#[salsa::tracked]
pub struct SpecializationOutput<'db> {
    pub module: DeclTable<TypedRef<'db>>,
    pub declarations: DeclarationTypeMetadata<'db>,
    pub expression_types: ExpressionTypeMetadata<'db>,
    /// Exact base identities transported to generated generic extern
    /// declarations. This is specialization metadata, not source-provenance
    /// or ownership data.
    pub compiler_intrinsics: SortedMap<NodeId, Symbol>,
    /// How many of the module's entries, from the start, are source
    /// declarations; the instances follow them.
    pub source_entries: usize,
    /// The path of every generated function instance, sorted.
    pub function_instances: Vec<SymbolPath>,
    /// Every nominal instance, sorted by path.
    pub nominal_instances: Vec<NominalInstance<'db>>,
}

/// A generic nominal declaration at concrete type arguments, and the path of
/// its specialization.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct NominalInstance<'db> {
    pub owner: TypeDefId<'db>,
    pub arguments: Vec<Type<'db>>,
    pub path: SymbolPath,
}

/// The metadata specialization reads and extends.
struct MonomorphizeMetadata<'db> {
    constructor_types: ConstructorTypeMetadata<'db>,
    expressions: ExpressionTypeMetadata<'db>,
    compiler_intrinsics: SortedMap<NodeId, Symbol>,
}

/// The dependencies' checked bodies and metadata, which a package's
/// specialization clones its instances of their generics from.
struct Dependency<'db> {
    modules: Vec<&'db DeclTable<TypedRef<'db>>>,
    expressions: Vec<&'db ExpressionTypeMetadata<'db>>,
    exhaustive_cases: Vec<NodeId>,
}

impl<'db> Dependency<'db> {
    fn new(db: &'db dyn salsa::Database, templates: &Templates<'db>) -> Self {
        let mut dependency = Self {
            modules: Vec::with_capacity(templates.dependencies.len()),
            expressions: Vec::with_capacity(templates.dependencies.len()),
            exhaustive_cases: Vec::new(),
        };
        for checked in &templates.dependencies {
            dependency.modules.push(checked.module(db));
            let expressions = checked.expression_types(db);
            dependency
                .exhaustive_cases
                .extend_from_slice(&expressions.exhaustive_cases);
            dependency.expressions.push(expressions);
        }
        dependency.exhaustive_cases.sort();
        dependency.exhaustive_cases.dedup();
        dependency
    }
}

/// The per-expression metadata a specialization reads: the module's own,
/// over that of the dependencies whose bodies it clones.
#[derive(Clone, Copy)]
struct Expressions<'a, 'db> {
    own: &'a ExpressionTypeMetadata<'db>,
    dependency: Option<&'a Dependency<'db>>,
}

impl<'a, 'db> Expressions<'a, 'db> {
    fn get<V>(
        self,
        table: fn(&'a ExpressionTypeMetadata<'db>) -> &'a SortedMap<NodeId, V>,
        id: &NodeId,
    ) -> Option<&'a V> {
        table(self.own).get(id).or_else(|| {
            self.dependency?
                .expressions
                .iter()
                .find_map(|&expressions| table(expressions).get(id))
        })
    }

    fn is_exhaustive(self, id: &NodeId) -> bool {
        self.own.exhaustive_cases.binary_search(id).is_ok()
            || self
                .dependency
                .is_some_and(|dependency| dependency.exhaustive_cases.binary_search(id).is_ok())
    }
}

/// Specialize `input`: collect every generic function and type
/// instantiation it reaches, generate concrete copies, rewrite call sites
/// and type references to them, and append them to the module.
pub fn monomorphize_functions<'db>(
    db: &'db dyn salsa::Database,
    input: SpecializationInput<'db>,
) -> Result<SpecializationOutput<'db>, Vec<InstanceError>> {
    let SpecializationInput {
        module,
        declarations,
        expressions,
        compiler_intrinsics,
        templates,
        names,
    } = input;
    let source_entries = module.entries.len();
    let schemes: HashMap<_, _> = declarations.function_types.iter().cloned().collect();
    let constructor_schemes: HashMap<_, _> = declarations
        .constructor_types
        .schemes
        .iter()
        .copied()
        .collect();
    let dependency = templates
        .as_ref()
        .map(|templates| Dependency::new(db, templates));
    // A package's result keeps its own declarations; the dependencies' are
    // only looked up, by the result check too.
    let (own_declarations, lookup) = match templates {
        Some(templates) => (templates.declarations, Some(declarations)),
        None => (declarations, None),
    };
    let DeclarationTypeMetadata {
        function_types,
        constructor_types,
        ability_definitions,
        well_known_types,
    } = own_declarations;
    let Monomorphized {
        module,
        function_types,
        metadata,
        mut function_instances,
        nominal_instances,
        removed_templates,
    } = monomorphize(
        db,
        module,
        &names,
        Declared {
            functions: &schemes,
            constructors: &constructor_schemes,
        },
        function_types,
        MonomorphizeMetadata {
            constructor_types,
            expressions,
            compiler_intrinsics,
        },
        dependency.as_ref(),
    )?;
    let declarations = DeclarationTypeMetadata {
        function_types,
        constructor_types: metadata.constructor_types,
        ability_definitions,
        well_known_types,
    };
    let expressions = metadata.expressions;
    let source_entries = source_entries - removed_templates;
    function_instances.sort();
    let errors = match &lookup {
        Some(lookup) => {
            let declared = declarations.after_dependency(lookup);
            result_check::check(
                db,
                &module,
                dependency
                    .as_ref()
                    .map_or(&[], |dependency| &dependency.modules),
                &declared.function_types,
                &declared.constructor_types,
                &expressions,
            )
        }
        None => result_check::check(
            db,
            &module,
            &[],
            &declarations.function_types,
            &declarations.constructor_types,
            &expressions,
        ),
    };
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(SpecializationOutput::new(
        db,
        module,
        declarations,
        expressions,
        metadata.compiler_intrinsics,
        source_entries,
        function_instances,
        nominal_instances,
    ))
}

/// Merge the specializations of the packages of one build into one result
/// (`new-plans/implementation.md`, "프론트엔드 단계 경계와 정보 소유권").
/// `results` are given each after the packages it depends on. The merged
/// table has every result's sources, then every result's instances, each
/// in that order; an instance more than one result has is kept once, the
/// first. Metadata of a later result wins, and agrees for a shared key.
pub fn merge_specializations<'db>(
    db: &'db dyn salsa::Database,
    results: impl IntoIterator<Item = SpecializationOutput<'db>>,
) -> Result<SpecializationOutput<'db>, Vec<InstanceError>> {
    let results: Vec<_> = results.into_iter().collect();
    let Some((&last, dependencies)) = results.split_last() else {
        panic!("a build merges at least one specialization");
    };
    if dependencies.is_empty() {
        return Ok(last);
    }
    let mut module = last.module(db).clone();
    let mut entries = Vec::new();
    for result in &results {
        entries.extend_from_slice(&result.module(db).entries[..*result.source_entries(db)]);
    }
    let source_entries = entries.len();
    let mut seen = HashSet::default();
    for result in &results {
        let instances = &result.module(db).entries[*result.source_entries(db)..];
        entries.extend(
            instances
                .iter()
                .filter(|entry| {
                    instance_identity(entry).is_none_or(|identity| seen.insert(identity))
                })
                .cloned(),
        );
    }
    module.entries = entries;

    let first = results[0];
    let mut declarations = first.declarations(db).clone();
    let mut expressions = first.expression_types(db).clone();
    let mut compiler_intrinsics = first.compiler_intrinsics(db).clone();
    let mut function_instances = first.function_instances(db).clone();
    let mut nominal_instances = first.nominal_instances(db).clone();
    for result in &results[1..] {
        declarations = result.declarations(db).after_dependency(&declarations);
        expressions = result.expression_types(db).after_dependency(&expressions);
        compiler_intrinsics.extend(result.compiler_intrinsics(db).iter().cloned());
        function_instances.extend(result.function_instances(db).iter().cloned());
        nominal_instances.extend(result.nominal_instances(db).iter().cloned());
    }
    function_instances.sort();
    function_instances.dedup();
    let mut seen = HashSet::default();
    nominal_instances.retain(|instance| seen.insert(instance.clone()));
    nominal_instances.sort_by(|left, right| left.path.cmp(&right.path));
    let errors = result_check::check(
        db,
        &module,
        &[],
        &declarations.function_types,
        &declarations.constructor_types,
        &expressions,
    );
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(SpecializationOutput::new(
        db,
        module,
        declarations,
        expressions,
        compiler_intrinsics,
        source_entries,
        function_instances,
        nominal_instances,
    ))
}

/// What identifies an instance entry across the results of several
/// specializations. A function instance is identified by its path, which
/// determines its body and metadata. A nominal instance is identified by its
/// node, which its type arguments determine: arguments spelled differently
/// can mangle to one path, and each spelling has its own instance.
#[derive(PartialEq, Eq, Hash)]
enum InstanceIdentity {
    Function(SymbolPath),
    Nominal(NodeId),
}

fn instance_identity<V>(entry: &DeclEntry<V>) -> Option<InstanceIdentity> {
    match &entry.decl {
        Decl::Function(decl) => Some(InstanceIdentity::Function(
            entry.module.child(decl.name.clone()),
        )),
        Decl::ExternFunction(decl) => Some(InstanceIdentity::Function(
            entry.module.child(decl.name.clone()),
        )),
        Decl::Struct(decl) => Some(InstanceIdentity::Nominal(decl.id)),
        Decl::Enum(decl) => Some(InstanceIdentity::Nominal(decl.id)),
        Decl::Ability(_) | Decl::Use(_) => None,
    }
}

/// The specialized module, with every function's scheme and the metadata
/// substituted with it.
struct Monomorphized<'db> {
    module: DeclTable<TypedRef<'db>>,
    function_types: Vec<(SymbolPath, TypeScheme<'db>)>,
    metadata: MonomorphizeMetadata<'db>,
    /// Every generated function instance's path.
    function_instances: Vec<SymbolPath>,
    /// The nominal instances this specialization generated.
    nominal_instances: Vec<NominalInstance<'db>>,
    /// How many source declarations were templates, and left the module.
    removed_templates: usize,
}

/// The schemes that look every declaration up, the dependency's included.
struct Declared<'a, 'db> {
    functions: &'a HashMap<SymbolPath, TypeScheme<'db>>,
    constructors: &'a HashMap<CtorId<'db>, TypeScheme<'db>>,
}

/// Specialize `module`. `declared` looks every declaration up;
/// `function_types` and `metadata` are the module's own, which the result
/// extends.
fn monomorphize<'db>(
    db: &'db dyn salsa::Database,
    module: DeclTable<TypedRef<'db>>,
    names: &NameDisplay,
    declared: Declared<'_, 'db>,
    function_types: Vec<(SymbolPath, TypeScheme<'db>)>,
    mut metadata: MonomorphizeMetadata<'db>,
    dependency: Option<&Dependency<'db>>,
) -> Result<Monomorphized<'db>, Vec<InstanceError>> {
    // Cloning finds exhaustive cases by binary search.
    metadata.expressions.exhaustive_cases.sort();
    metadata.expressions.exhaustive_cases.dedup();
    let mut errors = recursion::check(
        db,
        names,
        &module,
        declared.functions,
        &metadata.expressions.function_instances,
    );
    if !errors.is_empty() {
        errors.sort_by_key(|error| error.node);
        return Err(errors);
    }
    let errors = validate::validate(
        db,
        &module,
        dependency,
        declared.functions,
        Expressions {
            own: &metadata.expressions,
            dependency,
        },
    );
    if !errors.is_empty() {
        return Err(errors);
    }

    // === Function monomorphization ===

    let source_function_types: Vec<(SymbolPath, TypeScheme<'db>)> = declared
        .functions
        .iter()
        .map(|(path, scheme)| (path.clone(), *scheme))
        .collect();
    let mut module = module;
    let mut all_function_types = function_types;
    let mut instantiations: HashMap<_, HashSet<_>> = HashMap::default();
    let mut generated = HashSet::default();
    let mut reached_fixpoint = false;

    // A concrete clone can reveal direct calls that were abstract in its
    // source body. Clone the metadata first, then collect only unseen keys.
    for _ in 0..MAX_TRANSITIVE_SPECIALIZATION_ROUNDS {
        let discovered = collect::collect_instantiations(
            db,
            &module,
            &source_function_types,
            &metadata.expressions.function_instances,
        );
        let mut new_instantiations: HashMap<_, HashSet<_>> = HashMap::default();
        for (func_id, discovered) in discovered {
            let known = instantiations.entry(func_id).or_default();
            for key in discovered {
                if known.insert(key.clone()) {
                    new_instantiations.entry(func_id).or_default().insert(key);
                }
            }
        }
        if new_instantiations.is_empty() {
            reached_fixpoint = true;
            break;
        }

        let specializations = specialize::generate_specializations(
            db,
            &module,
            dependency.map_or(&[], |dependency| &dependency.modules),
            &new_instantiations,
            &source_function_types,
            &metadata.compiler_intrinsics,
            &mut generated,
        );
        let specialize::GeneratedSpecializations {
            specialized_declarations,
            specialized_extern_declarations,
            specialized_function_types,
            metadata_origins,
            compiler_intrinsic_specializations,
        } = specializations;
        let expressions = Expressions {
            own: &metadata.expressions,
            dependency,
        };
        let clones: Vec<_> = metadata_origins
            .into_iter()
            .map(|(key, origins)| clone_instance(db, expressions, &key, &origins))
            .collect();
        metadata.expressions.extend(clones);
        metadata
            .compiler_intrinsics
            .extend(compiler_intrinsic_specializations);

        let rewrite_map = build_rewrite_map(db, &instantiations, &source_function_types);
        rewrite::rewrite_module(
            db,
            &mut module,
            &rewrite_map,
            &metadata.expressions.function_instances,
        );
        let mut specialized_decls: Vec<_> = specialized_declarations
            .into_iter()
            .map(|(placement, declaration)| placement.place(Decl::Function(declaration)))
            .collect();
        rewrite::rewrite_decls(
            db,
            &mut specialized_decls,
            &rewrite_map,
            &metadata.expressions.function_instances,
        );

        module.entries.extend(specialized_decls);
        module.entries.extend(
            specialized_extern_declarations
                .into_iter()
                .map(|(placement, declaration)| placement.place(Decl::ExternFunction(declaration))),
        );
        all_function_types.extend(specialized_function_types);
    }
    if !reached_fixpoint {
        return Err(vec![InstanceError {
            node: module.id,
            kind: InstanceErrorKind::ExpansionLimit,
        }]);
    }
    let mut fn_types_vec = all_function_types;
    let function_instances = generated.into_iter().collect();
    let removed_templates = templates::remove_function_templates(
        db,
        &mut module,
        &mut fn_types_vec,
        &mut metadata.expressions,
    );

    // === Type monomorphization (struct/enum) ===

    // Every rewritten metadata type is a root, even if absent from the AST.
    let mut extra_types = Vec::new();
    let mut seeds = ForEachType {
        db,
        visit: |ty| extra_types.push(ty),
    };
    for (_, scheme) in &fn_types_vec {
        let ControlFlow::Continue(()) = scheme.visit_with(&mut seeds);
    }
    let ControlFlow::Continue(()) = metadata.constructor_types.visit_types(&mut seeds);
    let ControlFlow::Continue(()) = metadata.expressions.visit_types(&mut seeds);
    let nominal_index = nominal_index::NominalIndex::new(
        db,
        &module,
        dependency.map_or(&[], |dependency| &dependency.modules),
    );
    let seeds =
        collect::collect_type_instantiations_with_index(db, &module, extra_types, &nominal_index);
    let nominal = nominal::collect(db, &nominal_index, declared.constructors, seeds)?;
    let type_instantiations = nominal.instances;
    let mut nominal_instances: Vec<_> = type_instantiations
        .iter()
        .flat_map(|(owner, argument_sets)| {
            argument_sets.iter().map(|arguments| NominalInstance {
                owner: *owner,
                arguments: arguments.clone(),
                path: mangle::mangle_type_path(db, *owner, arguments),
            })
        })
        .collect();
    nominal_instances.sort_by(|left, right| left.path.cmp(&right.path));
    let mut type_rewrite_map = rewrite::TypeRewriteMap::default();
    for instance in &nominal_instances {
        type_rewrite_map
            .entry(instance.owner)
            .or_default()
            .push((instance.arguments.clone(), instance.path.clone()));
    }
    let enum_variants = &mut metadata.constructor_types.specialized_enum_variants;
    assert!(
        nominal
            .enum_variants
            .keys()
            .all(|node| !enum_variants.contains_key(node)),
        "specialized enum variant schema already exists"
    );
    enum_variants.extend(nominal.enum_variants);

    let module = if !type_rewrite_map.is_empty() {
        // Generate specialized struct/enum declarations
        let specialized_structs = specialize::generate_struct_specializations_with_index(
            db,
            &nominal_index,
            &type_instantiations,
        );
        // Dependency closure already validated and substituted each scheme. Only
        // assign its emitted key now, after expansion has successfully terminated.
        let mut struct_constructors: Vec<_> = nominal
            .struct_constructors
            .into_iter()
            .map(|(owner, arguments, scheme)| {
                (
                    CtorId::new(db, mangle::mangle_type_path(db, owner, &arguments)),
                    scheme,
                )
            })
            .collect();
        struct_constructors.sort_by_key(|(id, _)| id.path(db));
        struct_constructors.dedup_by_key(|(id, _)| *id);
        metadata.constructor_types.schemes.extend(
            struct_constructors
                .into_iter()
                .filter(|(ctor, _)| !declared.constructors.contains_key(ctor)),
        );
        let specialized_enums = specialize::generate_enum_specializations_with_index(
            db,
            &nominal_index,
            &type_instantiations,
        );

        // Rewrite Named types throughout the module
        rewrite::rewrite_field_function_refs(
            db,
            &mut module,
            &type_rewrite_map,
            &mut metadata.expressions.function_instances,
        );
        let mut rewrite = rewrite::NominalRewrite {
            db,
            map: &type_rewrite_map,
        };
        for (_, scheme) in &mut fn_types_vec {
            *scheme = scheme.fold_with(&mut rewrite);
        }
        let Ok(()) = metadata.constructor_types.try_fold_types(&mut rewrite);
        let Ok(()) = metadata.expressions.try_fold_types(&mut rewrite);
        rewrite::rewrite_types_in_module(db, &mut module, &type_rewrite_map);

        // Append specialized types to module
        module.entries.extend(
            specialized_structs
                .into_iter()
                .map(|(placement, declaration)| placement.place(Decl::Struct(declaration))),
        );
        module.entries.extend(
            specialized_enums
                .into_iter()
                .map(|(placement, declaration)| placement.place(Decl::Enum(declaration))),
        );
        module
    } else {
        rewrite::rewrite_field_function_refs(
            db,
            &mut module,
            &rewrite::TypeRewriteMap::default(),
            &mut metadata.expressions.function_instances,
        );
        module
    };

    Ok(Monomorphized {
        module,
        function_types: fn_types_vec,
        metadata,
        function_instances,
        nominal_instances,
        removed_templates,
    })
}

/// The metadata of `origins`, the nodes of the generic declaration `key`
/// instantiates, at the instance's nodes, with `key`'s type arguments
/// substituted. A clone reads only the nodes of the declaration it copies,
/// which no clone writes, so a round's clones are merged at once.
fn clone_instance<'db>(
    db: &'db dyn salsa::Database,
    expressions: Expressions<'_, 'db>,
    key: &InstanceKey<'db>,
    origins: &HashSet<NodeId>,
) -> ExpressionTypeMetadata<'db> {
    let variant = key.variant();
    let mut clone = EntryClone {
        expressions,
        origins,
        variant,
        substitution: BoundVarSubst::new(db, &key.type_args),
    };
    let mut local_instances = clone.entries(|table| &table.local_instances);
    for instance in local_instances.values_mut() {
        instance.binding = instance.binding.with_variant(variant);
    }
    ExpressionTypeMetadata {
        node_types: clone.entries(|table| &table.node_types),
        function_instances: clone.entries(|table| &table.function_instances),
        local_instances,
        evidence_plans: clone.entries(|table| &table.evidence_plans),
        handler_operations: clone.entries(|table| &table.handler_operations),
        perform_operations: clone.entries(|table| &table.perform_operations),
        lambda_signatures: clone.entries(|table| &table.lambda_signatures),
        exhaustive_cases: origins
            .iter()
            .filter(|id| expressions.is_exhaustive(id))
            .map(|id| id.with_variant(variant))
            .collect(),
    }
}

/// Copies the entries of a declaration's nodes to one instance's nodes,
/// substituting the instance's type arguments as it copies each value once.
struct EntryClone<'a, 'db> {
    expressions: Expressions<'a, 'db>,
    origins: &'a HashSet<NodeId>,
    variant: std::num::NonZero<u64>,
    substitution: BoundVarSubst<'a, 'db>,
}

impl<'a, 'db> EntryClone<'a, 'db> {
    fn entries<V: TypeFoldable<'db> + Clone>(
        &mut self,
        table: fn(&'a ExpressionTypeMetadata<'db>) -> &'a SortedMap<NodeId, V>,
    ) -> SortedMap<NodeId, V> {
        let (expressions, origins, variant) = (self.expressions, self.origins, self.variant);
        origins
            .iter()
            .filter_map(|id| {
                expressions
                    .get(table, id)
                    .map(|value| (id.with_variant(variant), value))
            })
            .map(|(id, value)| {
                let value = value
                    .clone()
                    .try_fold_with(&mut self.substitution)
                    .unwrap_or_else(|BoundVarOutOfBounds { index, max }| {
                        panic!("BoundVar index out of range in specialization metadata: index={index}, subst.len()={max}")
                    });
                (id, value)
            })
            .collect()
    }
}

/// Build a map from (original FuncDefId, concrete callee type) → mangled Symbol
/// for use during call site rewriting.
fn build_rewrite_map<'db>(
    db: &'db dyn salsa::Database,
    instantiations: &HashMap<FuncDefId<'db>, HashSet<InstanceKey<'db>>>,
    function_types: &[(SymbolPath, TypeScheme<'db>)],
) -> rewrite::RewriteMap<'db> {
    let scheme_map: HashMap<SymbolPath, TypeScheme<'db>> = function_types.iter().cloned().collect();
    let mut rewrite_map = rewrite::RewriteMap::default();

    for (func_id, keys) in instantiations {
        let path = func_id.path(db);
        if !scheme_map.contains_key(path) {
            continue;
        }

        let mut entries: Vec<(InstanceKey<'db>, SymbolPath)> = keys
            .iter()
            .map(|key| (key.clone(), mangle::mangle_instance_path(db, path, key)))
            .collect();
        entries.sort_by(|left, right| left.1.cmp(&right.1));
        rewrite_map.insert(*func_id, entries);
    }

    rewrite_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityId, AbilityRow, OpDeclKind, Type, TypeKind};
    use crate::typeck::{
        InstantiatedHandlerOperation, InstantiatedPerformOperation, LambdaSignature,
    };

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    #[test]
    fn specialization_metadata_rekeys_and_substitutes_sparse_tables() {
        let db = TestDb::default();
        let origin = NodeId::from_raw(1);
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(&db, TypeKind::Int);
        let function = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bound],
                result: bound,
                effect: AbilityRow::pure(&db),
            },
        );
        let specialized_function = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect: AbilityRow::pure(&db),
            },
        );
        let ability = AbilityId::source(&db, trunk_ir::SymbolPath::new(["Audit"]));
        let operation = |kind| InstantiatedHandlerOperation {
            ability,
            ability_args: vec![bound],
            kind,
            params: vec![bound],
            result: bound,
        };
        let mut metadata = MonomorphizeMetadata {
            constructor_types: ConstructorTypeMetadata::default(),
            expressions: ExpressionTypeMetadata {
                node_types: [(origin, bound)].into_iter().collect(),
                function_instances: SortedMap::default(),
                evidence_plans: [(
                    origin,
                    vec![crate::typeck::EvidenceStep::Mask(
                        crate::ast::AbilityInstance {
                            ability_id: ability,
                            args: vec![bound],
                        },
                    )],
                )]
                .into_iter()
                .collect(),
                local_instances: [(
                    origin,
                    crate::typeck::LocalCallableInstance {
                        binding: NodeId::from_raw(2),
                        local: crate::ast::LocalId::new(7),
                        scheme: TypeScheme::mono(&db, function),
                        callable: function,
                        row_arguments: vec![],
                    },
                )]
                .into_iter()
                .collect(),
                handler_operations: [(origin, operation(OpDeclKind::Op))].into_iter().collect(),
                perform_operations: [(
                    origin,
                    InstantiatedPerformOperation {
                        ability,
                        ability_args: vec![bound],
                        kind: OpDeclKind::Op,
                        params: vec![bound],
                        result: bound,
                    },
                )]
                .into_iter()
                .collect(),
                lambda_signatures: [(
                    origin,
                    LambdaSignature {
                        function_type: function,
                    },
                )]
                .into_iter()
                .collect(),
                exhaustive_cases: vec![origin],
            },
            compiler_intrinsics: SortedMap::default(),
        };
        let key = InstanceKey::of_types([int]);
        let origins = [origin].into_iter().collect::<HashSet<_>>();

        let clone = clone_instance(
            &db,
            Expressions {
                own: &metadata.expressions,
                dependency: None,
            },
            &key,
            &origins,
        );
        metadata.expressions.extend([clone]);

        let clone = origin.with_variant(key.variant());
        assert_eq!(metadata.expressions.node_types.get(&clone), Some(&int));
        let local = metadata.expressions.local_instances.get(&clone).unwrap();
        assert_eq!(
            local.binding,
            NodeId::from_raw(2).with_variant(key.variant())
        );
        assert_eq!(local.local, crate::ast::LocalId::new(7));
        assert_eq!(local.callable, specialized_function);
        assert_eq!(local.scheme.body(&db), specialized_function);
        assert_eq!(
            metadata
                .expressions
                .local_instances
                .get(&origin)
                .unwrap()
                .callable,
            function
        );
        assert_eq!(
            metadata
                .expressions
                .handler_operations
                .get(&clone)
                .unwrap()
                .ability_args,
            [int]
        );
        assert_eq!(
            metadata
                .expressions
                .handler_operations
                .get(&clone)
                .unwrap()
                .params,
            [int]
        );
        assert_eq!(
            metadata
                .expressions
                .handler_operations
                .get(&clone)
                .unwrap()
                .result,
            int
        );
        assert_eq!(
            metadata
                .expressions
                .perform_operations
                .get(&clone)
                .unwrap()
                .ability_args,
            [int]
        );
        assert_eq!(
            metadata
                .expressions
                .perform_operations
                .get(&clone)
                .unwrap()
                .params,
            [int]
        );
        assert_eq!(
            metadata
                .expressions
                .perform_operations
                .get(&clone)
                .unwrap()
                .result,
            int
        );
        assert_eq!(
            metadata.expressions.evidence_plans.get(&clone).unwrap(),
            &[crate::typeck::EvidenceStep::Mask(
                crate::ast::AbilityInstance {
                    ability_id: ability,
                    args: vec![int],
                }
            )]
        );
        assert_eq!(
            metadata
                .expressions
                .lambda_signatures
                .get(&clone)
                .unwrap()
                .function_type,
            specialized_function
        );
        assert!(metadata.expressions.exhaustive_cases.contains(&clone));
        assert!(
            metadata
                .expressions
                .handler_operations
                .contains_key(&origin)
        );
    }
}
