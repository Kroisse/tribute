pub mod collect;
pub mod mangle;
mod rewrite;
pub mod specialize;

use std::collections::{HashMap, HashSet};

use trunk_ir::Symbol;

use crate::ast::{CtorId, Decl, FuncDefId, Module, NodeId, Type, TypeDefId, TypeScheme, TypedRef};
use crate::typeck::subst::substitute_bound_vars;
use crate::typeck::{InstantiatedHandlerOperation, InstantiatedPerformOperation, LambdaSignature};

const MAX_TRANSITIVE_SPECIALIZATION_ROUNDS: usize = 64;

/// Exact typechecking metadata keyed by source NodeId.
pub struct MonomorphizeMetadata<'db> {
    pub constructor_types: HashMap<CtorId<'db>, TypeScheme<'db>>,
    pub node_types: HashMap<NodeId, Type<'db>>,
    pub call_callee_types: HashMap<NodeId, Type<'db>>,
    pub handler_operations: HashMap<NodeId, InstantiatedHandlerOperation<'db>>,
    pub perform_operations: HashMap<NodeId, InstantiatedPerformOperation<'db>>,
    pub lambda_signatures: HashMap<NodeId, LambdaSignature<'db>>,
    pub exhaustive_cases: HashSet<NodeId>,
}

/// Result of monomorphization: updated module + function types.
pub struct MonomorphizeResult<'db> {
    pub module: Module<TypedRef<'db>>,
    pub function_types: Vec<(Symbol, TypeScheme<'db>)>,
    pub metadata: MonomorphizeMetadata<'db>,
}

/// Run monomorphization on a typed module.
///
/// This is the main entry point that:
/// 1. Collects all generic function/type instantiations
/// 2. Generates specialized copies with concrete types
/// 3. Rewrites call sites and type references to use specialized versions
/// 4. Appends specialized functions/types to the module
pub fn monomorphize_functions<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
    mut metadata: MonomorphizeMetadata<'db>,
) -> MonomorphizeResult<'db> {
    let fn_types_vec: Vec<(Symbol, TypeScheme<'db>)> =
        function_types.iter().map(|(k, v)| (*k, *v)).collect();

    // === Function monomorphization ===

    let source_function_types = fn_types_vec.clone();
    let mut module = module;
    let mut all_function_types = fn_types_vec;
    let mut instantiations = HashMap::new();
    let mut reached_fixpoint = false;

    // A concrete clone can reveal direct calls that were abstract in its
    // source body. Clone the metadata first, then collect only unseen keys.
    for _ in 0..MAX_TRANSITIVE_SPECIALIZATION_ROUNDS {
        let discovered = collect::collect_instantiations(
            db,
            &module,
            &source_function_types,
            &metadata.call_callee_types,
        );
        let mut new_instantiations = HashMap::new();
        for (func_id, type_arg_sets) in discovered {
            let known = instantiations.entry(func_id).or_insert_with(HashSet::new);
            for type_args in type_arg_sets {
                if known.insert(type_args.clone()) {
                    new_instantiations
                        .entry(func_id)
                        .or_insert_with(HashSet::new)
                        .insert(type_args);
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
        );
        for (type_args, origins) in specializations.metadata_origins {
            specialize_metadata(db, &mut metadata, &type_args, &origins);
        }

        let rewrite_map = build_rewrite_map(db, &instantiations, &source_function_types);
        let rewritten_module =
            rewrite::rewrite_module(db, module, &source_function_types, &rewrite_map);
        let specialized_decls: Vec<Decl<TypedRef<'db>>> = specializations
            .specialized_declarations
            .into_iter()
            .map(Decl::Function)
            .collect();
        let rewritten_specialized =
            rewrite::rewrite_decls(db, specialized_decls, &source_function_types, &rewrite_map);

        let mut decls = rewritten_module.decls;
        decls.extend(rewritten_specialized);
        module = Module::new(rewritten_module.id, rewritten_module.name, decls);
        all_function_types.extend(specializations.specialized_function_types);
    }
    assert!(
        reached_fixpoint,
        "generic specialization exceeded the deterministic expansion limit"
    );
    let fn_types_vec = all_function_types;

    // === Type monomorphization (struct/enum) ===

    let type_instantiations = collect::collect_type_instantiations(db, &module);

    let module = if !type_instantiations.is_empty() {
        // Generate specialized struct/enum declarations
        let specialized_structs =
            specialize::generate_struct_specializations(db, &module, &type_instantiations);
        specialize_struct_constructor_metadata(
            db,
            &module,
            &type_instantiations,
            &mut metadata.constructor_types,
        );
        let specialized_enums =
            specialize::generate_enum_specializations(db, &module, &type_instantiations);

        // Build type rewrite map and rewrite Named types throughout the module
        let type_rewrite_map = rewrite::build_type_rewrite_map(db, &type_instantiations);
        let rewritten_module = rewrite::rewrite_types_in_module(db, module, &type_rewrite_map);

        // Append specialized types to module
        let mut decls = rewritten_module.decls;
        decls.extend(specialized_structs.into_iter().map(Decl::Struct));
        decls.extend(specialized_enums.into_iter().map(Decl::Enum));

        Module::new(rewritten_module.id, rewritten_module.name, decls)
    } else {
        module
    };

    MonomorphizeResult {
        module,
        function_types: fn_types_vec,
        metadata,
    }
}

/// Generate exact constructor schemes for specialized struct declarations.
///
/// A struct constructor shares its canonical qualified identity with its type,
/// so source-logical lowering looks up the generated mangled name directly.
fn specialize_struct_constructor_metadata<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    instantiations: &HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
    constructor_types: &mut HashMap<CtorId<'db>, TypeScheme<'db>>,
) {
    let struct_decls = specialize::collect_struct_decls(db, module);
    let mut entries = Vec::new();

    for (id, type_arg_sets) in instantiations {
        let Some(structure) = struct_decls.get(id) else {
            continue;
        };
        if structure.type_params.is_empty() {
            continue;
        }
        let Some(source_scheme) = constructor_types
            .get(&CtorId::new(db, id.qualified(db)))
            .copied()
        else {
            continue;
        };
        for type_args in type_arg_sets {
            entries.push(specialize_struct_constructor_scheme(
                db,
                *id,
                type_args,
                source_scheme,
            ));
        }
    }

    entries.sort_by_key(|(id, _)| id.qualified(db));
    for (ctor, scheme) in entries {
        constructor_types.entry(ctor).or_insert(scheme);
    }
}

fn specialize_struct_constructor_scheme<'db>(
    db: &'db dyn salsa::Database,
    type_id: TypeDefId<'db>,
    type_args: &[Type<'db>],
    source_scheme: TypeScheme<'db>,
) -> (CtorId<'db>, TypeScheme<'db>) {
    let name = mangle::mangle_type_name(db, type_id, type_id.qualified(db), type_args);
    let body = substitute_type(db, source_scheme.body(db), type_args);
    let scheme = TypeScheme::new(
        db,
        Vec::new(),
        source_scheme.effect_params(db).clone(),
        body,
    );
    (CtorId::new(db, name), scheme)
}

fn specialize_metadata<'db>(
    db: &'db dyn salsa::Database,
    metadata: &mut MonomorphizeMetadata<'db>,
    type_args: &[Type<'db>],
    origins: &HashSet<NodeId>,
) {
    let variant = specialize::type_args_variant(type_args);
    clone_type_table(db, &mut metadata.node_types, variant, type_args, origins);
    clone_type_table(
        db,
        &mut metadata.call_callee_types,
        variant,
        type_args,
        origins,
    );
    let handlers: Vec<_> = origins
        .iter()
        .filter_map(|id| {
            metadata
                .handler_operations
                .get(id)
                .map(|operation| (*id, operation.clone()))
        })
        .collect();
    for (id, operation) in handlers {
        metadata.handler_operations.insert(
            id.with_variant(variant),
            InstantiatedHandlerOperation {
                ability: operation.ability,
                ability_args: operation
                    .ability_args
                    .into_iter()
                    .map(|ty| substitute_type(db, ty, type_args))
                    .collect(),
                kind: operation.kind,
                params: operation
                    .params
                    .into_iter()
                    .map(|ty| substitute_type(db, ty, type_args))
                    .collect(),
                result: substitute_type(db, operation.result, type_args),
            },
        );
    }
    let performs: Vec<_> = origins
        .iter()
        .filter_map(|id| {
            metadata
                .perform_operations
                .get(id)
                .map(|operation| (*id, operation.clone()))
        })
        .collect();
    for (id, operation) in performs {
        metadata.perform_operations.insert(
            id.with_variant(variant),
            InstantiatedPerformOperation {
                ability: operation.ability,
                ability_args: operation
                    .ability_args
                    .into_iter()
                    .map(|ty| substitute_type(db, ty, type_args))
                    .collect(),
                kind: operation.kind,
                params: operation
                    .params
                    .into_iter()
                    .map(|ty| substitute_type(db, ty, type_args))
                    .collect(),
                result: substitute_type(db, operation.result, type_args),
            },
        );
    }
    let lambdas: Vec<_> = origins
        .iter()
        .filter_map(|id| {
            metadata
                .lambda_signatures
                .get(id)
                .map(|signature| (*id, signature.clone()))
        })
        .collect();
    for (id, signature) in lambdas {
        metadata.lambda_signatures.insert(
            id.with_variant(variant),
            LambdaSignature {
                function_type: substitute_type(db, signature.function_type, type_args),
                convention: signature.convention,
            },
        );
    }
    let exhaustive: Vec<_> = origins
        .iter()
        .copied()
        .filter(|id| metadata.exhaustive_cases.contains(id))
        .collect();
    metadata
        .exhaustive_cases
        .extend(exhaustive.into_iter().map(|id| id.with_variant(variant)));
}

fn clone_type_table<'db>(
    db: &'db dyn salsa::Database,
    table: &mut HashMap<NodeId, Type<'db>>,
    variant: std::num::NonZero<u64>,
    type_args: &[Type<'db>],
    origins: &HashSet<NodeId>,
) {
    let source: Vec<_> = origins
        .iter()
        .filter_map(|id| table.get(id).map(|ty| (*id, *ty)))
        .collect();
    for (id, ty) in source {
        table.insert(id.with_variant(variant), substitute_type(db, ty, type_args));
    }
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

/// Build a map from (original FuncDefId, concrete callee type) → mangled Symbol
/// for use during call site rewriting.
fn build_rewrite_map<'db>(
    db: &'db dyn salsa::Database,
    instantiations: &HashMap<FuncDefId<'db>, std::collections::HashSet<Vec<Type<'db>>>>,
    function_types: &[(Symbol, TypeScheme<'db>)],
) -> HashMap<FuncDefId<'db>, Vec<(Vec<Type<'db>>, Symbol)>> {
    let scheme_map: HashMap<Symbol, TypeScheme<'db>> = function_types.iter().cloned().collect();
    let mut rewrite_map: HashMap<FuncDefId<'db>, Vec<(Vec<Type<'db>>, Symbol)>> = HashMap::new();

    for (func_id, type_arg_sets) in instantiations {
        let qualified = func_id.qualified(db);
        let Some(_scheme) = scheme_map.get(&qualified) else {
            continue;
        };

        let mut entries: Vec<(Vec<Type<'db>>, Symbol)> = type_arg_sets
            .iter()
            .map(|type_args| {
                let mangled = mangle::mangle_name(db, qualified, type_args);
                (type_args.clone(), mangled)
            })
            .collect();
        entries.sort_by_key(|e| e.1);
        rewrite_map.insert(*func_id, entries);
    }

    rewrite_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        AbilityId, CallingConvention, EffectRow, OpDeclKind, StructDecl, TypeKind, TypeParam,
        TypeParamDecl,
    };

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    #[test]
    fn specialized_struct_constructor_scheme_uses_mangled_key_and_concrete_fields() {
        let db = TestDb::default();
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(&db, TypeKind::Int);
        let structure = StructDecl {
            id: NodeId::from_raw(1),
            is_pub: false,
            name: Symbol::new("nested::Holder"),
            type_params: vec![TypeParamDecl {
                id: NodeId::from_raw(2),
                name: Symbol::new("a"),
                bounds: vec![],
            }],
            fields: vec![],
        };
        let type_id = TypeDefId::source(&db, structure.name, structure.id);
        let holder = Type::new(
            &db,
            TypeKind::Named {
                id: type_id,
                name: structure.name,
                args: vec![bound],
            },
        );
        let source_scheme = TypeScheme::new(
            &db,
            vec![TypeParam::anonymous()],
            Vec::new(),
            Type::new(
                &db,
                TypeKind::Func {
                    params: vec![bound],
                    result: holder,
                    effect: EffectRow::pure(&db),
                    minimum_convention: CallingConvention::Direct,
                },
            ),
        );
        let module = Module::new(NodeId::from_raw(0), None, vec![Decl::Struct(structure)]);
        let instantiations = HashMap::from([(type_id, HashSet::from([vec![int]]))]);
        let source_ctor = CtorId::new(&db, Symbol::new("nested::Holder"));
        let mut constructor_types = HashMap::from([(source_ctor, source_scheme)]);
        specialize_struct_constructor_metadata(
            &db,
            &module,
            &instantiations,
            &mut constructor_types,
        );

        let ctor = CtorId::new(&db, Symbol::new("nested::Holder$Int"));
        assert_eq!(constructor_types.get(&source_ctor), Some(&source_scheme));
        let specialized = constructor_types
            .get(&ctor)
            .expect("one generic struct instantiation must produce one constructor scheme");
        assert!(specialized.is_mono(&db));
        let TypeKind::Func { params, result, .. } = specialized.body(&db).kind(&db) else {
            panic!("specialized constructor must retain a callable schema")
        };
        assert!(matches!(params.as_slice(), [param] if *param == int));
        assert!(matches!(
            result.kind(&db),
            TypeKind::Named { args, .. } if args.as_slice() == [int]
        ));
    }

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
                effect: EffectRow::pure(&db),
                minimum_convention: CallingConvention::Direct,
            },
        );
        let specialized_function = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect: EffectRow::pure(&db),
                minimum_convention: CallingConvention::Direct,
            },
        );
        let ability = AbilityId::source(&db, Symbol::new("Audit"));
        let operation = |kind| InstantiatedHandlerOperation {
            ability,
            ability_args: vec![bound],
            kind,
            params: vec![bound],
            result: bound,
        };
        let mut metadata = MonomorphizeMetadata {
            constructor_types: HashMap::new(),
            node_types: HashMap::from([(origin, bound)]),
            call_callee_types: HashMap::from([(origin, bound)]),
            handler_operations: HashMap::from([(origin, operation(OpDeclKind::Op))]),
            perform_operations: HashMap::from([(
                origin,
                InstantiatedPerformOperation {
                    ability,
                    ability_args: vec![bound],
                    kind: OpDeclKind::Op,
                    params: vec![bound],
                    result: bound,
                },
            )]),
            lambda_signatures: HashMap::from([(
                origin,
                LambdaSignature {
                    function_type: function,
                    convention: CallingConvention::Direct,
                },
            )]),
            exhaustive_cases: HashSet::from([origin]),
        };
        let type_args = vec![int];
        let origins = HashSet::from([origin]);

        specialize_metadata(&db, &mut metadata, &type_args, &origins);

        let clone = origin.with_variant(specialize::type_args_variant(&type_args));
        assert_eq!(metadata.node_types.get(&clone), Some(&int));
        assert_eq!(metadata.call_callee_types.get(&clone), Some(&int));
        assert_eq!(
            metadata
                .handler_operations
                .get(&clone)
                .unwrap()
                .ability_args,
            [int]
        );
        assert_eq!(
            metadata.handler_operations.get(&clone).unwrap().params,
            [int]
        );
        assert_eq!(metadata.handler_operations.get(&clone).unwrap().result, int);
        assert_eq!(
            metadata
                .perform_operations
                .get(&clone)
                .unwrap()
                .ability_args,
            [int]
        );
        assert_eq!(
            metadata.perform_operations.get(&clone).unwrap().params,
            [int]
        );
        assert_eq!(metadata.perform_operations.get(&clone).unwrap().result, int);
        assert_eq!(
            metadata
                .lambda_signatures
                .get(&clone)
                .unwrap()
                .function_type,
            specialized_function
        );
        assert!(metadata.exhaustive_cases.contains(&clone));
        assert!(metadata.handler_operations.contains_key(&origin));
    }
}
