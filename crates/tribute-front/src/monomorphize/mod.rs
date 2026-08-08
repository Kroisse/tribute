pub mod collect;
pub mod mangle;
mod rewrite;
pub mod specialize;

use std::collections::{HashMap, HashSet};

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Span;
use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, CallingConvention, CtorId, Decl, FuncDefId, Module, NodeId, Type, TypeScheme,
    TypedRef,
};
use crate::typeck::subst::substitute_bound_vars;
use crate::typeck::{
    AbilityInfo, InstantiatedHandlerOperation, InstantiatedPerformOperation, LambdaSignature,
};

/// Exact semantic metadata consumed by typed AST lowering.
///
/// Monomorphization owns this bundle because both function cloning and later
/// nominal type rewriting change the identities used as lookup keys.
pub struct MonomorphizeMetadata<'db> {
    pub constructor_types: HashMap<CtorId<'db>, TypeScheme<'db>>,
    pub type_definitions: HashMap<Symbol, TypeScheme<'db>>,
    pub node_types: HashMap<NodeId, Type<'db>>,
    pub call_callee_types: HashMap<NodeId, Type<'db>>,
    /// Call-callee metadata entries rewritten to a concrete specialization.
    pub specialized_call_callee_nodes: HashSet<NodeId>,
    pub ability_conventions: HashMap<AbilityId<'db>, CallingConvention>,
    pub ability_definitions: HashMap<AbilityId<'db>, AbilityInfo<'db>>,
    pub handler_operations: HashMap<NodeId, InstantiatedHandlerOperation<'db>>,
    pub perform_operations: HashMap<NodeId, InstantiatedPerformOperation<'db>>,
    pub lambda_signatures: HashMap<NodeId, LambdaSignature<'db>>,
    pub exhaustive_cases: HashSet<NodeId>,
}

/// Result of monomorphization: updated module and its exact semantic tables.
pub struct MonomorphizeResult<'db> {
    pub module: Module<TypedRef<'db>>,
    pub function_types: Vec<(Symbol, TypeScheme<'db>)>,
    pub metadata: MonomorphizeMetadata<'db>,
}

/// Internal guardrails for deterministic specialization expansion.
///
/// The production values keep recursive generic discovery bounded while the
/// private test entry point can exercise the diagnostic without constructing a
/// production-sized specialization graph.
#[derive(Clone, Copy)]
struct SpecializationLimits {
    rounds: usize,
    instantiations: usize,
}

const PRODUCTION_SPECIALIZATION_LIMITS: SpecializationLimits = SpecializationLimits {
    rounds: 64,
    instantiations: 2048,
};

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
    metadata: MonomorphizeMetadata<'db>,
) -> MonomorphizeResult<'db> {
    monomorphize_functions_with_limits(
        db,
        module,
        function_types,
        metadata,
        PRODUCTION_SPECIALIZATION_LIMITS,
    )
}

fn monomorphize_functions_with_limits<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
    mut metadata: MonomorphizeMetadata<'db>,
    limits: SpecializationLimits,
) -> MonomorphizeResult<'db> {
    let source_function_types: Vec<(Symbol, TypeScheme<'db>)> =
        function_types.iter().map(|(k, v)| (*k, *v)).collect();

    // === Function monomorphization ===
    // A specialization can expose concrete generic calls that were abstract in
    // its source declaration. Iterate over newly created bodies until no such
    // call remains. The original generic declarations are deliberately kept.
    let mut module = module;
    let mut function_types = source_function_types.clone();
    let mut instantiations = HashMap::new();
    let mut exceeded_limit = false;

    for _ in 0..limits.rounds {
        let discovered = collect::collect_instantiations(
            db,
            &module,
            &source_function_types,
            &metadata.call_callee_types,
        );
        let new_instantiations = retain_new_instantiations(&mut instantiations, discovered);
        if new_instantiations.is_empty() {
            break;
        }
        if instantiation_count(&instantiations) > limits.instantiations {
            exceeded_limit = true;
            break;
        }

        let (specialized_decls, specialized_fn_types, specializations) =
            specialize::generate_specializations(
                db,
                &module,
                &new_instantiations,
                &source_function_types,
            );
        specialize_node_metadata(db, &mut metadata, &specializations);
        let specialized_extern_types = specialize::generate_extern_specialization_types(
            db,
            &module,
            &new_instantiations,
            &source_function_types,
        );
        let rewrite_map = build_rewrite_map(db, &instantiations, &source_function_types);
        module = rewrite::rewrite_module(
            db,
            module,
            &function_types,
            &rewrite_map,
            &mut metadata.call_callee_types,
            &mut metadata.specialized_call_callee_nodes,
        );
        let rewritten_specialized = rewrite::rewrite_decls(
            db,
            specialized_decls.into_iter().map(Decl::Function).collect(),
            &function_types,
            &rewrite_map,
            &mut metadata.call_callee_types,
            &mut metadata.specialized_call_callee_nodes,
        );
        let mut decls = module.decls;
        decls.extend(rewritten_specialized);
        module = Module::new(module.id, module.name, decls);
        function_types.extend(specialized_fn_types);
        function_types.extend(specialized_extern_types);
    }

    if exceeded_limit
        || !retain_new_instantiations(
            &mut instantiations,
            collect::collect_instantiations(
                db,
                &module,
                &source_function_types,
                &metadata.call_callee_types,
            ),
        )
        .is_empty()
    {
        Diagnostic::new(
            "generic specialization exceeded the deterministic expansion limit",
            Span::default(),
            DiagnosticSeverity::Error,
            CompilationPhase::Lowering,
        )
        .accumulate(db);
    }

    let fn_types_vec = function_types;

    // === Type monomorphization (struct/enum) ===

    // Function cloning creates variant NodeIds. Materialize their exact, substituted
    // semantic metadata before collecting nominal instantiations so types that only
    // occur in a specialized body or parameter are not lost.
    let type_instantiations = collect::collect_type_instantiations(
        db,
        &module,
        &metadata.type_definitions,
        &metadata.node_types,
        &fn_types_vec,
    );
    let constructor_specializations = specialize::generate_constructor_specializations(
        db,
        &module,
        &type_instantiations,
        &metadata.type_definitions,
    );

    let (module, type_rewrite_map) = if !type_instantiations.is_empty() {
        // Generate specialized struct/enum declarations
        let specialized_structs = specialize::generate_struct_specializations(
            db,
            &module,
            &type_instantiations,
            &metadata.type_definitions,
        );
        let specialized_enums = specialize::generate_enum_specializations(
            db,
            &module,
            &type_instantiations,
            &metadata.type_definitions,
        );

        // Build type rewrite map and rewrite Named types throughout the module
        let type_rewrite_map = rewrite::build_type_rewrite_map(db, &type_instantiations);
        let rewritten_module = rewrite::rewrite_types_in_module(db, module, &type_rewrite_map);

        // Append specialized types to module
        let mut decls = rewritten_module.decls;
        decls.extend(specialized_structs.into_iter().map(Decl::Struct));
        decls.extend(specialized_enums.into_iter().map(Decl::Enum));

        (
            Module::new(rewritten_module.id, rewritten_module.name, decls),
            type_rewrite_map,
        )
    } else {
        (module, rewrite::TypeRewriteMap::new())
    };

    rewrite_semantic_types(
        db,
        &mut metadata,
        &constructor_specializations,
        &type_rewrite_map,
    );
    let function_types = fn_types_vec
        .into_iter()
        .map(|(name, scheme)| (name, rewrite_scheme(db, scheme, &type_rewrite_map)))
        .collect();

    MonomorphizeResult {
        module,
        function_types,
        metadata,
    }
}

fn retain_new_instantiations<'db>(
    known: &mut HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>>,
    discovered: HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>>,
) -> HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let mut new = HashMap::new();
    for (function, args) in discovered {
        let known_args = known.entry(function).or_default();
        for args in args {
            if known_args.insert(args.clone()) {
                new.entry(function)
                    .or_insert_with(HashSet::new)
                    .insert(args);
            }
        }
    }
    new
}

fn instantiation_count(instantiations: &HashMap<FuncDefId<'_>, HashSet<Vec<Type<'_>>>>) -> usize {
    instantiations.values().map(HashSet::len).sum()
}

fn specialize_node_metadata<'db>(
    db: &'db dyn salsa::Database,
    metadata: &mut MonomorphizeMetadata<'db>,
    specializations: &[specialize::FunctionSpecialization<'db>],
) {
    let source_node_types = metadata.node_types.clone();
    let source_call_callee_types = metadata.call_callee_types.clone();
    let source_handlers = metadata.handler_operations.clone();
    let source_performs = metadata.perform_operations.clone();
    let source_lambdas = metadata.lambda_signatures.clone();
    let source_exhaustive = metadata.exhaustive_cases.clone();

    for specialization in specializations {
        for &origin in &specialization.node_origins {
            let specialized = origin.with_variant(specialization.variant);
            if let Some(&ty) = source_node_types.get(&origin) {
                metadata.node_types.insert(
                    specialized,
                    substitute_type(db, ty, &specialization.type_args),
                );
            }
            if let Some(&ty) = source_call_callee_types.get(&origin) {
                metadata.call_callee_types.insert(
                    specialized,
                    substitute_type(db, ty, &specialization.type_args),
                );
            }
            if let Some(operation) = source_handlers.get(&origin) {
                metadata.handler_operations.insert(
                    specialized,
                    substitute_handler_operation(db, operation, &specialization.type_args),
                );
            }
            if let Some(operation) = source_performs.get(&origin) {
                metadata.perform_operations.insert(
                    specialized,
                    substitute_perform_operation(db, operation, &specialization.type_args),
                );
            }
            if let Some(signature) = source_lambdas.get(&origin) {
                metadata.lambda_signatures.insert(
                    specialized,
                    LambdaSignature {
                        function_type: substitute_type(
                            db,
                            signature.function_type,
                            &specialization.type_args,
                        ),
                        body_is_effect_free: signature.body_is_effect_free,
                        contains_control_transfer: signature.contains_control_transfer,
                        lexical_convention: signature.lexical_convention,
                        convention: signature.convention,
                    },
                );
            }
            if source_exhaustive.contains(&origin) {
                metadata.exhaustive_cases.insert(specialized);
            }
        }
    }
}

fn rewrite_semantic_types<'db>(
    db: &'db dyn salsa::Database,
    metadata: &mut MonomorphizeMetadata<'db>,
    constructor_specializations: &[specialize::ConstructorSpecialization<'db>],
    type_rewrite_map: &rewrite::TypeRewriteMap<'db>,
) {
    let source_constructors = metadata.constructor_types.clone();
    for (&constructor, &scheme) in &source_constructors {
        metadata
            .constructor_types
            .insert(constructor, rewrite_scheme(db, scheme, type_rewrite_map));
    }
    for specialization in constructor_specializations {
        let scheme = source_constructors
            .get(&specialization.source)
            .unwrap_or_else(|| {
                panic!(
                    "missing source constructor scheme for exact specialization `{}`",
                    specialization.source.qualified(db)
                )
            });
        metadata.constructor_types.insert(
            specialization.specialized,
            specialize_scheme(db, *scheme, &specialization.type_args, type_rewrite_map),
        );
    }
    for scheme in metadata.type_definitions.values_mut() {
        *scheme = rewrite_scheme(db, *scheme, type_rewrite_map);
    }

    for ty in metadata.node_types.values_mut() {
        *ty = rewrite::rewrite_type(db, *ty, type_rewrite_map);
    }
    for ty in metadata.call_callee_types.values_mut() {
        *ty = rewrite::rewrite_type(db, *ty, type_rewrite_map);
    }
    for operation in metadata.handler_operations.values_mut() {
        rewrite_handler_operation(db, operation, type_rewrite_map);
    }
    for operation in metadata.perform_operations.values_mut() {
        rewrite_perform_operation(db, operation, type_rewrite_map);
    }
    for signature in metadata.lambda_signatures.values_mut() {
        signature.function_type =
            rewrite::rewrite_type(db, signature.function_type, type_rewrite_map);
    }
    for ability in metadata.ability_definitions.values_mut() {
        for operation in ability.operations.values_mut() {
            operation.param_types = operation
                .param_types
                .iter()
                .map(|ty| rewrite::rewrite_type(db, *ty, type_rewrite_map))
                .collect();
            operation.return_type =
                rewrite::rewrite_type(db, operation.return_type, type_rewrite_map);
        }
    }
}

fn substitute_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    arguments: &[Type<'db>],
) -> Type<'db> {
    substitute_bound_vars(db, ty, arguments).unwrap_or_else(|index, max| {
        panic!(
            "BoundVar index out of range in semantic metadata specialization: \
             index={index}, subst.len()={max}"
        )
    })
}

fn substitute_handler_operation<'db>(
    db: &'db dyn salsa::Database,
    operation: &InstantiatedHandlerOperation<'db>,
    arguments: &[Type<'db>],
) -> InstantiatedHandlerOperation<'db> {
    InstantiatedHandlerOperation {
        ability: operation.ability,
        ability_args: operation
            .ability_args
            .iter()
            .map(|ty| substitute_type(db, *ty, arguments))
            .collect(),
        kind: operation.kind,
        params: operation
            .params
            .iter()
            .map(|ty| substitute_type(db, *ty, arguments))
            .collect(),
        result: substitute_type(db, operation.result, arguments),
    }
}

fn substitute_perform_operation<'db>(
    db: &'db dyn salsa::Database,
    operation: &InstantiatedPerformOperation<'db>,
    arguments: &[Type<'db>],
) -> InstantiatedPerformOperation<'db> {
    InstantiatedPerformOperation {
        ability: operation.ability,
        ability_args: operation
            .ability_args
            .iter()
            .map(|ty| substitute_type(db, *ty, arguments))
            .collect(),
        kind: operation.kind,
        params: operation
            .params
            .iter()
            .map(|ty| substitute_type(db, *ty, arguments))
            .collect(),
        result: substitute_type(db, operation.result, arguments),
    }
}

fn rewrite_handler_operation<'db>(
    db: &'db dyn salsa::Database,
    operation: &mut InstantiatedHandlerOperation<'db>,
    type_rewrite_map: &rewrite::TypeRewriteMap<'db>,
) {
    operation.ability_args = operation
        .ability_args
        .iter()
        .map(|ty| rewrite::rewrite_type(db, *ty, type_rewrite_map))
        .collect();
    operation.params = operation
        .params
        .iter()
        .map(|ty| rewrite::rewrite_type(db, *ty, type_rewrite_map))
        .collect();
    operation.result = rewrite::rewrite_type(db, operation.result, type_rewrite_map);
}

fn rewrite_perform_operation<'db>(
    db: &'db dyn salsa::Database,
    operation: &mut InstantiatedPerformOperation<'db>,
    type_rewrite_map: &rewrite::TypeRewriteMap<'db>,
) {
    operation.ability_args = operation
        .ability_args
        .iter()
        .map(|ty| rewrite::rewrite_type(db, *ty, type_rewrite_map))
        .collect();
    operation.params = operation
        .params
        .iter()
        .map(|ty| rewrite::rewrite_type(db, *ty, type_rewrite_map))
        .collect();
    operation.result = rewrite::rewrite_type(db, operation.result, type_rewrite_map);
}

fn rewrite_scheme<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    type_rewrite_map: &rewrite::TypeRewriteMap<'db>,
) -> TypeScheme<'db> {
    TypeScheme::new(
        db,
        scheme.type_params(db).clone(),
        scheme.effect_params(db).clone(),
        rewrite::rewrite_type(db, scheme.body(db), type_rewrite_map),
    )
}

fn specialize_scheme<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    arguments: &[Type<'db>],
    type_rewrite_map: &rewrite::TypeRewriteMap<'db>,
) -> TypeScheme<'db> {
    TypeScheme::new(
        db,
        vec![],
        scheme.effect_params(db).clone(),
        rewrite::rewrite_type(
            db,
            substitute_type(db, scheme.body(db), arguments),
            type_rewrite_map,
        ),
    )
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
    use std::collections::HashMap;

    use salsa_test_macros::salsa_test;
    use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};

    use super::*;
    use crate::SourceCst;

    /// Runs the real typed frontend inputs through monomorphization with a
    /// deliberately tiny private budget. Keeping this query tracked makes the
    /// lowering diagnostic observable through Salsa's accumulator.
    #[salsa::tracked]
    fn monomorphize_with_tiny_limit(db: &dyn salsa::Database, source: SourceCst) {
        let parsed = crate::query::parsed_ast(db, source).expect("fixture must parse");
        let ast = parsed.module(db).clone();
        let span_map = parsed.span_map(db).clone();
        let environment = crate::resolve::build_env(db, &ast);
        let resolved = crate::resolve::resolve_with_env(db, ast, environment, span_map);
        let output = crate::typeck::typecheck_module(db, resolved, parsed.span_map(db).clone());
        let module = crate::tdnr::resolve_tdnr(db, output.module(db).clone(), std::iter::empty());
        let nominal_types = output.nominal_types(db);
        let expression_types = output.expression_types(db);
        let metadata = MonomorphizeMetadata {
            constructor_types: nominal_types.constructor_types.iter().copied().collect(),
            type_definitions: nominal_types.type_definitions.iter().copied().collect(),
            node_types: expression_types.node_types.iter().copied().collect(),
            call_callee_types: expression_types.call_callee_types.iter().copied().collect(),
            specialized_call_callee_nodes: Default::default(),
            ability_conventions: output.ability_conventions(db).iter().copied().collect(),
            ability_definitions: crate::typeck::ability_definitions_from_schemas(
                output.ability_definitions(db),
            ),
            handler_operations: output.handler_operations(db).iter().cloned().collect(),
            perform_operations: output.perform_operations(db).iter().cloned().collect(),
            lambda_signatures: output.lambda_signatures(db).iter().cloned().collect(),
            exhaustive_cases: output.exhaustive_cases(db).iter().copied().collect(),
        };
        let function_types: HashMap<_, _> = output.function_types(db).iter().copied().collect();

        let _ = monomorphize_functions_with_limits(
            db,
            module,
            function_types,
            metadata,
            SpecializationLimits {
                rounds: 4,
                instantiations: 1,
            },
        );
    }

    #[salsa_test]
    fn specialization_limit_is_a_tracked_lowering_error(db: &salsa::DatabaseImpl) {
        let source = SourceCst::from_source_str(
            db,
            "specialization_limit.trb",
            r#"
fn identity(value: a) -> a { value }

fn use_nat() -> Nat { identity(1) }

fn main() -> Bool { identity(true) }
"#,
        );

        monomorphize_with_tiny_limit(db, source);
        let diagnostics: Vec<_> =
            monomorphize_with_tiny_limit::accumulated::<Diagnostic>(db, source)
                .into_iter()
                .filter(|diagnostic| {
                    diagnostic.phase == CompilationPhase::Lowering
                        && diagnostic.inner.severity == DiagnosticSeverity::Error
                        && diagnostic.inner.message
                            == "generic specialization exceeded the deterministic expansion limit"
                })
                .collect();

        assert_eq!(diagnostics.len(), 1, "the limit must reject compilation");
    }
}
