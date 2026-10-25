pub mod collect;
mod instance;

pub(crate) use instance::{
    LocalInstantiation, collect_ability_argument_tails, local_instantiations,
};
pub mod mangle;
mod nominal;
mod nominal_index;
mod rewrite;
mod validate;
pub(crate) use collect::is_concrete_type;
pub use validate::{InstanceError, InstanceErrorKind};
pub mod specialize;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use trunk_ir::{Symbol, SymbolPath};

use self::instance::InstanceKey;
use crate::SortedMap;
use crate::ast::{
    AbilityRow, CtorId, Decl, DeclTable, FuncDefId, NodeId, Type, TypeScheme, TypedRef,
};
use crate::typeck::subst::{BoundVarOutOfBounds, substitute_ability_row, substitute_bound_vars};
use crate::typeck::{
    ConstructorTypeMetadata, DeclarationTypeMetadata, ExpressionTypeMetadata,
    InstantiatedHandlerOperation, InstantiatedPerformOperation, LambdaSignature, TypeCheckOutput,
};

const MAX_TRANSITIVE_SPECIALIZATION_ROUNDS: usize = 64;

/// What specialization reads: one checked module, with the declarations of
/// the package it depends on for lookup and, when merged, that package's
/// bodies and per-expression metadata ahead of its own
/// (`new-plans/implementation.md`, "프론트엔드 단계 경계와 정보 소유권").
pub struct SpecializationInput<'db> {
    module: DeclTable<TypedRef<'db>>,
    declarations: DeclarationTypeMetadata<'db>,
    expressions: ExpressionTypeMetadata<'db>,
    compiler_intrinsics: SortedMap<NodeId, Symbol>,
}

impl<'db> SpecializationInput<'db> {
    /// The checked module, with the declarations of `dependency` under its
    /// own.
    pub fn new(
        db: &'db dyn salsa::Database,
        checked: TypeCheckOutput<'db>,
        dependency: Option<TypeCheckOutput<'db>>,
    ) -> Self {
        let declarations = match dependency {
            Some(dependency) => checked
                .declarations(db)
                .after_dependency(dependency.declarations(db)),
            None => checked.declarations(db).clone(),
        };
        Self {
            module: checked.module(db).clone(),
            declarations,
            expressions: checked.expression_types(db).clone(),
            compiler_intrinsics: SortedMap::default(),
        }
    }

    /// Specialize `dependency`'s bodies too: its declarations come first in
    /// the module, and its per-expression metadata under this module's.
    pub fn merge_bodies(
        mut self,
        db: &'db dyn salsa::Database,
        dependency: TypeCheckOutput<'db>,
    ) -> Self {
        let mut entries = dependency.module(db).entries.clone();
        entries.append(&mut self.module.entries);
        self.module.entries = entries;
        self.expressions = self
            .expressions
            .after_dependency(dependency.expression_types(db));
        self
    }

    /// Carry the compiler intrinsic identities of the module's declarations.
    pub fn with_compiler_intrinsics(
        mut self,
        intrinsics: impl IntoIterator<Item = (NodeId, Symbol)>,
    ) -> Self {
        self.compiler_intrinsics = intrinsics.into_iter().collect();
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
}

/// The metadata specialization reads and extends.
struct MonomorphizeMetadata<'db> {
    constructor_types: ConstructorTypeMetadata<'db>,
    expressions: ExpressionTypeMetadata<'db>,
    compiler_intrinsics: SortedMap<NodeId, Symbol>,
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
    } = input;
    let DeclarationTypeMetadata {
        function_types,
        constructor_types,
        ability_definitions,
        well_known_types,
    } = declarations;
    let Monomorphized {
        module,
        function_types,
        metadata,
    } = monomorphize(
        db,
        module,
        function_types.into_iter().collect(),
        MonomorphizeMetadata {
            constructor_types,
            expressions,
            compiler_intrinsics,
        },
    )?;
    Ok(SpecializationOutput::new(
        db,
        module,
        DeclarationTypeMetadata {
            function_types,
            constructor_types: metadata.constructor_types,
            ability_definitions,
            well_known_types,
        },
        metadata.expressions,
        metadata.compiler_intrinsics,
    ))
}

/// The specialized module, with every function's scheme and the metadata
/// substituted with it.
struct Monomorphized<'db> {
    module: DeclTable<TypedRef<'db>>,
    function_types: Vec<(SymbolPath, TypeScheme<'db>)>,
    metadata: MonomorphizeMetadata<'db>,
}

fn monomorphize<'db>(
    db: &'db dyn salsa::Database,
    module: DeclTable<TypedRef<'db>>,
    function_types: HashMap<SymbolPath, TypeScheme<'db>>,
    mut metadata: MonomorphizeMetadata<'db>,
) -> Result<Monomorphized<'db>, Vec<InstanceError>> {
    // Cloning finds exhaustive cases by binary search.
    metadata.expressions.exhaustive_cases.sort();
    metadata.expressions.exhaustive_cases.dedup();
    let errors = validate::validate(db, &module, &function_types, &metadata);
    if !errors.is_empty() {
        return Err(errors);
    }
    let fn_types_vec: Vec<(SymbolPath, TypeScheme<'db>)> = function_types
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    // === Function monomorphization ===

    let source_function_types = fn_types_vec.clone();
    let mut module = module;
    let mut all_function_types = fn_types_vec;
    let mut instantiations: HashMap<_, HashSet<_>> = HashMap::default();
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
            &new_instantiations,
            &source_function_types,
            &metadata.compiler_intrinsics,
        );
        let specialize::GeneratedSpecializations {
            specialized_declarations,
            specialized_extern_declarations,
            specialized_function_types,
            metadata_origins,
            compiler_intrinsic_specializations,
        } = specializations;
        let mut clones = MetadataClones::default();
        for (key, origins) in metadata_origins {
            clones.clone_instance(db, &metadata.expressions, &key, &origins);
        }
        clones.add_to(&mut metadata.expressions);
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

    // === Type monomorphization (struct/enum) ===

    // Every rewritten metadata type is a root, even if absent from the AST.
    let mut extra_types = Vec::new();
    for scheme in fn_types_vec
        .iter()
        .map(|(_, scheme)| *scheme)
        .chain(
            metadata
                .constructor_types
                .schemes
                .iter()
                .map(|(_, scheme)| *scheme),
        )
        .chain(
            metadata
                .constructor_types
                .specialized_enum_variants
                .values()
                .copied(),
        )
    {
        scheme.for_each_type(db, |ty| extra_types.push(ty));
    }
    for instance in metadata.expressions.function_instances.values() {
        extra_types.push(instance.callable);
        extra_types.extend(instance.type_arguments.iter().copied());
        for row in &instance.row_arguments {
            for effect in row.instances(db) {
                extra_types.extend(effect.args.iter().copied());
            }
        }
    }
    for instance in metadata.expressions.local_instances.values() {
        instance.scheme.for_each_type(db, |ty| extra_types.push(ty));
        extra_types.push(instance.callable);
        for row in &instance.row_arguments {
            extra_types.extend(
                row.instances(db)
                    .iter()
                    .flat_map(|effect| effect.args.iter().copied()),
            );
        }
    }
    extra_types.extend(metadata.expressions.node_types.values().copied());
    extra_types.extend(
        metadata
            .expressions
            .lambda_signatures
            .values()
            .map(|sig| sig.function_type),
    );
    for op in metadata.expressions.handler_operations.values() {
        extra_types.extend(op.ability_args.iter().copied());
        extra_types.extend(op.params.iter().copied());
        extra_types.push(op.result);
    }
    for op in metadata.expressions.perform_operations.values() {
        extra_types.extend(op.ability_args.iter().copied());
        extra_types.extend(op.params.iter().copied());
        extra_types.push(op.result);
    }
    for step in metadata.expressions.evidence_plans.values().flatten() {
        step.for_each_instance(|instance| extra_types.extend(instance.args.iter().copied()));
    }
    let nominal_index = nominal_index::NominalIndex::new(db, &module);
    let seeds =
        collect::collect_type_instantiations_with_index(db, &module, extra_types, &nominal_index);
    let constructor_schemes: HashMap<_, _> =
        metadata.constructor_types.schemes.iter().copied().collect();
    let nominal = nominal::collect(db, &nominal_index, &constructor_schemes, seeds)?;
    let type_instantiations = nominal.instances;
    let enum_variants = &mut metadata.constructor_types.specialized_enum_variants;
    assert!(
        nominal
            .enum_variants
            .keys()
            .all(|node| !enum_variants.contains_key(node)),
        "specialized enum variant schema already exists"
    );
    enum_variants.extend(nominal.enum_variants);

    let module = if !type_instantiations.is_empty() {
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
                .filter(|(ctor, _)| !constructor_schemes.contains_key(ctor)),
        );
        let specialized_enums = specialize::generate_enum_specializations_with_index(
            db,
            &nominal_index,
            &type_instantiations,
        );

        // Build type rewrite map and rewrite Named types throughout the module
        let type_rewrite_map = rewrite::build_type_rewrite_map(db, &type_instantiations);
        rewrite::rewrite_field_function_refs(
            db,
            &mut module,
            &type_rewrite_map,
            &mut metadata.expressions.function_instances,
        );
        let rewrite_ty = |ty| rewrite::rewrite_type(db, ty, &type_rewrite_map);
        let rewrite_scheme =
            |scheme: TypeScheme<'db>| scheme.to_builder(db).map_types(db, rewrite_ty).build(db);
        for (_, scheme) in &mut fn_types_vec {
            *scheme = rewrite_scheme(*scheme);
        }
        for (_, scheme) in &mut metadata.constructor_types.schemes {
            *scheme = rewrite_scheme(*scheme);
        }
        for scheme in metadata
            .constructor_types
            .specialized_enum_variants
            .values_mut()
        {
            *scheme = rewrite_scheme(*scheme);
        }
        for ty in metadata.expressions.node_types.values_mut() {
            *ty = rewrite_ty(*ty);
        }
        for instance in metadata.expressions.function_instances.values_mut() {
            instance.callable = rewrite_ty(instance.callable);
            for ty in &mut instance.type_arguments {
                *ty = rewrite_ty(*ty);
            }
            for row in &mut instance.row_arguments {
                *row = rewrite::rewrite_row(db, *row, &type_rewrite_map);
            }
        }
        for instance in metadata.expressions.local_instances.values_mut() {
            instance.scheme = rewrite_scheme(instance.scheme);
            instance.callable = rewrite_ty(instance.callable);
            for row in &mut instance.row_arguments {
                *row = rewrite::rewrite_row(db, *row, &type_rewrite_map);
            }
        }
        for signature in metadata.expressions.lambda_signatures.values_mut() {
            signature.function_type = rewrite_ty(signature.function_type);
        }
        for op in metadata.expressions.handler_operations.values_mut() {
            for ty in op.ability_args.iter_mut().chain(op.params.iter_mut()) {
                *ty = rewrite_ty(*ty);
            }
            op.result = rewrite_ty(op.result);
        }
        for op in metadata.expressions.perform_operations.values_mut() {
            for ty in op.ability_args.iter_mut().chain(op.params.iter_mut()) {
                *ty = rewrite_ty(*ty);
            }
            op.result = rewrite_ty(op.result);
        }
        for step in metadata.expressions.evidence_plans.values_mut().flatten() {
            *step = step.map_types(rewrite_ty);
        }
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
    })
}

/// The metadata cloned for one round of instances. A clone reads only the
/// nodes of the generic declaration it copies, which no clone writes, so the
/// round's clones are added to each table at once.
#[derive(Default)]
struct MetadataClones<'db> {
    node_types: Vec<(NodeId, Type<'db>)>,
    function_instances: Vec<(NodeId, crate::typeck::FunctionInstance<'db>)>,
    local_instances: Vec<(NodeId, crate::typeck::LocalCallableInstance<'db>)>,
    handler_operations: Vec<(NodeId, InstantiatedHandlerOperation<'db>)>,
    perform_operations: Vec<(NodeId, InstantiatedPerformOperation<'db>)>,
    evidence_plans: Vec<(NodeId, Vec<crate::typeck::EvidenceStep<'db>>)>,
    lambda_signatures: Vec<(NodeId, LambdaSignature<'db>)>,
    exhaustive_cases: Vec<NodeId>,
}

impl<'db> MetadataClones<'db> {
    /// Clone the metadata of `origins`, the nodes of the generic declaration
    /// `key` instantiates, to the instance's nodes, with `key`'s type
    /// arguments substituted.
    fn clone_instance(
        &mut self,
        db: &'db dyn salsa::Database,
        expressions: &ExpressionTypeMetadata<'db>,
        key: &InstanceKey<'db>,
        origins: &HashSet<NodeId>,
    ) {
        let (variant, type_args) = (key.variant(), key.type_args.as_slice());
        let substitute = |ty| substitute_type(db, ty, type_args);
        clone_entries(
            &expressions.node_types,
            origins,
            variant,
            substitute,
            &mut self.node_types,
        );
        clone_entries(
            &expressions.function_instances,
            origins,
            variant,
            |mut instance| {
                instance.callable = substitute(instance.callable);
                instance.type_arguments = instance
                    .type_arguments
                    .into_iter()
                    .map(substitute)
                    .collect();
                instance.row_arguments = instance
                    .row_arguments
                    .into_iter()
                    .map(|row| substitute_row(db, row, type_args))
                    .collect();
                instance
            },
            &mut self.function_instances,
        );
        clone_entries(
            &expressions.local_instances,
            origins,
            variant,
            |mut instance| {
                instance.binding = instance.binding.with_variant(variant);
                instance.callable = substitute(instance.callable);
                instance.scheme = instance
                    .scheme
                    .to_builder(db)
                    .map_types(db, substitute)
                    .build(db);
                for row in &mut instance.row_arguments {
                    *row = substitute_row(db, *row, type_args);
                }
                instance
            },
            &mut self.local_instances,
        );
        clone_entries(
            &expressions.handler_operations,
            origins,
            variant,
            |operation| InstantiatedHandlerOperation {
                ability: operation.ability,
                ability_args: operation.ability_args.into_iter().map(substitute).collect(),
                kind: operation.kind,
                params: operation.params.into_iter().map(substitute).collect(),
                result: substitute(operation.result),
            },
            &mut self.handler_operations,
        );
        clone_entries(
            &expressions.perform_operations,
            origins,
            variant,
            |operation| InstantiatedPerformOperation {
                ability: operation.ability,
                ability_args: operation.ability_args.into_iter().map(substitute).collect(),
                kind: operation.kind,
                params: operation.params.into_iter().map(substitute).collect(),
                result: substitute(operation.result),
            },
            &mut self.perform_operations,
        );
        clone_entries(
            &expressions.evidence_plans,
            origins,
            variant,
            |plan| {
                plan.into_iter()
                    .map(|step| step.map_types(substitute))
                    .collect()
            },
            &mut self.evidence_plans,
        );
        clone_entries(
            &expressions.lambda_signatures,
            origins,
            variant,
            |signature| LambdaSignature {
                function_type: substitute(signature.function_type),
            },
            &mut self.lambda_signatures,
        );
        self.exhaustive_cases.extend(
            origins
                .iter()
                .filter(|id| expressions.exhaustive_cases.binary_search(id).is_ok())
                .map(|id| id.with_variant(variant)),
        );
    }

    /// Add the clones to their tables, one merge per table.
    fn add_to(self, expressions: &mut ExpressionTypeMetadata<'db>) {
        expressions.node_types.extend(self.node_types);
        expressions
            .function_instances
            .extend(self.function_instances);
        expressions.local_instances.extend(self.local_instances);
        expressions
            .handler_operations
            .extend(self.handler_operations);
        expressions
            .perform_operations
            .extend(self.perform_operations);
        expressions.evidence_plans.extend(self.evidence_plans);
        expressions.lambda_signatures.extend(self.lambda_signatures);
        if !self.exhaustive_cases.is_empty() {
            expressions.exhaustive_cases.extend(self.exhaustive_cases);
            expressions.exhaustive_cases.sort();
            expressions.exhaustive_cases.dedup();
        }
    }
}

/// Add to `clones` the entry of each origin `table` holds, at the node with
/// `variant`, as `clone` makes it from the origin's value.
fn clone_entries<V: Clone>(
    table: &SortedMap<NodeId, V>,
    origins: &HashSet<NodeId>,
    variant: std::num::NonZero<u64>,
    clone: impl Fn(V) -> V,
    clones: &mut Vec<(NodeId, V)>,
) {
    clones.extend(origins.iter().filter_map(|id| {
        table
            .get(id)
            .map(|value| (id.with_variant(variant), clone(value.clone())))
    }));
}

fn substitute_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_args: &[Type<'db>],
) -> Type<'db> {
    substitute_bound_vars(db, ty, type_args).unwrap_or_else(|index, max| {
        panic!("BoundVar index out of range in specialization metadata: index={index}, subst.len()={max}")
    })
}

fn substitute_row<'db>(
    db: &'db dyn salsa::Database,
    row: AbilityRow<'db>,
    type_args: &[Type<'db>],
) -> AbilityRow<'db> {
    substitute_ability_row(db, row, type_args).unwrap_or_else(|BoundVarOutOfBounds { index, max }| {
        panic!("BoundVar index out of range in specialization metadata: index={index}, subst.len()={max}")
    })
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
    use crate::ast::{AbilityId, OpDeclKind, TypeKind};

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

        let mut clones = MetadataClones::default();
        clones.clone_instance(&db, &metadata.expressions, &key, &origins);
        clones.add_to(&mut metadata.expressions);

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
