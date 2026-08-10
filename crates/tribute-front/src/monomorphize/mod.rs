pub mod collect;
pub mod mangle;
mod rewrite;
pub mod specialize;

use std::collections::{HashMap, HashSet};

use trunk_ir::Symbol;

use crate::ast::{CtorId, Decl, FuncDefId, Module, NodeId, Type, TypeScheme, TypedRef};
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

        let (specialized_decls, specialized_fn_types, metadata_origins) =
            specialize::generate_specializations(
                db,
                &module,
                &new_instantiations,
                &source_function_types,
            );
        for (type_args, origins) in metadata_origins {
            specialize_metadata(db, &mut metadata, &type_args, &origins);
        }

        let rewrite_map = build_rewrite_map(db, &instantiations, &source_function_types);
        let rewritten_module =
            rewrite::rewrite_module(db, module, &source_function_types, &rewrite_map);
        let specialized_decls: Vec<Decl<TypedRef<'db>>> =
            specialized_decls.into_iter().map(Decl::Function).collect();
        let rewritten_specialized =
            rewrite::rewrite_decls(db, specialized_decls, &source_function_types, &rewrite_map);

        let mut decls = rewritten_module.decls;
        decls.extend(rewritten_specialized);
        module = Module::new(rewritten_module.id, rewritten_module.name, decls);
        all_function_types.extend(specialized_fn_types);
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
