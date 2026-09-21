//! Close nominal instances over their checked constructor schemas before cloning.
use std::collections::{HashMap, HashSet};

use crate::ast::{CtorId, Decl, Module, NodeId, Type, TypeDefId, TypeKind, TypeScheme, TypedRef};
use crate::typeck::subst::{SubstResult, substitute_bound_vars};

use super::{InstanceError, InstanceErrorKind, collect, specialize};

const MAX_NOMINAL_ROUNDS: usize = 64;
const MAX_NOMINAL_INSTANCES: usize = 4096;

type Instances<'db> = HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>;

pub(super) struct NominalInstances<'db> {
    pub instances: Instances<'db>,
    pub enum_variants: HashMap<NodeId, TypeScheme<'db>>,
}

struct Constructor<'db> {
    node: NodeId,
    id: CtorId<'db>,
    fields: usize,
}

struct Declaration<'db> {
    node: NodeId,
    params: usize,
    is_enum: bool,
    constructors: Vec<Constructor<'db>>,
}

fn declarations<'db>(
    db: &'db dyn salsa::Database,
    decls: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
    result: &mut HashMap<TypeDefId<'db>, Declaration<'db>>,
) {
    for decl in decls {
        let (name, declaration) = match decl {
            Decl::Struct(s) if !s.type_params.is_empty() => (
                s.name,
                Declaration {
                    node: s.id,
                    params: s.type_params.len(),
                    is_enum: false,
                    constructors: vec![Constructor {
                        node: s.id,
                        id: CtorId::new(db, crate::qualified_symbol(prefix, s.name)),
                        fields: s.fields.len(),
                    }],
                },
            ),
            Decl::Enum(e) if !e.type_params.is_empty() => (
                e.name,
                Declaration {
                    node: e.id,
                    params: e.type_params.len(),
                    is_enum: true,
                    constructors: e
                        .variants
                        .iter()
                        .map(|v| Constructor {
                            node: v.id,
                            id: CtorId::new(db, crate::qualified_symbol(prefix, v.name)),
                            fields: v.fields.len(),
                        })
                        .collect(),
                },
            ),
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let saved = crate::push_prefix(prefix, m.name);
                    declarations(db, body, prefix, result);
                    prefix.truncate(saved);
                }
                continue;
            }
            _ => continue,
        };
        result.insert(
            TypeDefId::source(db, crate::qualified_symbol(prefix, name), declaration.node),
            declaration,
        );
    }
}

fn instantiate<'db>(
    db: &'db dyn salsa::Database,
    owner: TypeDefId<'db>,
    declaration: &Declaration<'db>,
    constructor: &Constructor<'db>,
    schemes: &HashMap<CtorId<'db>, TypeScheme<'db>>,
    arguments: &[Type<'db>],
) -> Result<TypeScheme<'db>, InstanceError> {
    let fail = |kind| InstanceError {
        node: constructor.node,
        kind,
    };
    let scheme = schemes
        .get(&constructor.id)
        .copied()
        .ok_or_else(|| fail(InstanceErrorKind::MissingInstance))?;
    if scheme.type_params(db).len() != declaration.params {
        return Err(fail(InstanceErrorKind::TypeArgumentArity {
            expected: declaration.params,
            found: scheme.type_params(db).len(),
        }));
    }
    if arguments.len() != declaration.params {
        return Err(fail(InstanceErrorKind::TypeArgumentArity {
            expected: declaration.params,
            found: arguments.len(),
        }));
    }
    let (fields, result) = match scheme.body(db).kind(db) {
        TypeKind::Func { params, result, .. } => (params.len(), *result),
        _ => (0, scheme.body(db)),
    };
    if fields != constructor.fields {
        return Err(fail(InstanceErrorKind::InconsistentCallable));
    }
    if !matches!(result.kind(db), TypeKind::Named { id, .. } if *id == owner) {
        return Err(fail(InstanceErrorKind::WrongDeclaration));
    }
    let mut invalid = false;
    let specialized = scheme
        .to_builder(db)
        .map_types(db, |ty| match substitute_bound_vars(db, ty, arguments) {
            SubstResult::Ok(ty) => ty,
            _ => {
                invalid = true;
                ty
            }
        })
        .type_params(Vec::new())
        .build(db);
    if invalid {
        return Err(fail(InstanceErrorKind::IncompleteTypeArgument));
    }
    Ok(specialized)
}

pub(super) fn collect<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    schemes: &HashMap<CtorId<'db>, TypeScheme<'db>>,
    seeds: Instances<'db>,
) -> Result<NominalInstances<'db>, Vec<InstanceError>> {
    close_dependencies(
        db,
        module,
        schemes,
        seeds,
        MAX_NOMINAL_ROUNDS,
        MAX_NOMINAL_INSTANCES,
    )
    .map_err(|error| vec![error])
}

fn close_dependencies<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    schemes: &HashMap<CtorId<'db>, TypeScheme<'db>>,
    mut pending: Instances<'db>,
    max_rounds: usize,
    max_instances: usize,
) -> Result<NominalInstances<'db>, InstanceError> {
    let mut definitions = HashMap::new();
    declarations(db, &module.decls, &mut String::new(), &mut definitions);
    let generic_ids = definitions.keys().copied().collect();
    let mut instances = Instances::new();
    let mut enum_variants = HashMap::new();
    let mut count = 0;
    let limit = || InstanceError {
        node: module.id,
        kind: InstanceErrorKind::ExpansionLimit,
    };
    for _ in 0..max_rounds {
        if pending.is_empty() {
            return Ok(NominalInstances {
                instances,
                enum_variants,
            });
        }
        let mut discovered = Instances::new();
        for (owner, argument_sets) in pending {
            let declaration = &definitions[&owner];
            for arguments in argument_sets {
                if !instances
                    .entry(owner)
                    .or_default()
                    .insert(arguments.clone())
                {
                    continue;
                }
                count += 1;
                if count > max_instances {
                    return Err(limit());
                }
                for constructor in &declaration.constructors {
                    let scheme =
                        instantiate(db, owner, declaration, constructor, schemes, &arguments)?;
                    scheme.for_each_type(db, |ty| {
                        collect::collect_from_type(db, ty, &generic_ids, &mut discovered)
                    });
                    if declaration.is_enum {
                        let node = constructor
                            .node
                            .with_variant(specialize::type_args_variant(&arguments));
                        if enum_variants.insert(node, scheme).is_some() {
                            return Err(InstanceError {
                                node: constructor.node,
                                kind: InstanceErrorKind::InconsistentCallable,
                            });
                        }
                    }
                }
            }
        }
        for (owner, arguments) in &mut discovered {
            if let Some(known) = instances.get(owner) {
                arguments.retain(|args| !known.contains(args));
            }
        }
        discovered.retain(|_, arguments| !arguments.is_empty());
        pending = discovered;
    }
    if pending.is_empty() {
        Ok(NominalInstances {
            instances,
            enum_variants,
        })
    } else {
        Err(limit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::Symbol;

    #[salsa::tracked]
    fn checked<'db>(
        db: &'db dyn salsa::Database,
        source: crate::SourceCst,
    ) -> crate::typeck::TypeCheckOutput<'db> {
        let parsed = crate::query::parsed_ast(db, source).unwrap();
        let ast = parsed.module(db).clone();
        let resolved = crate::resolve::resolve_with_env(
            db,
            ast.clone(),
            crate::resolve::build_env(db, &ast),
            parsed.span_map(db).clone(),
        );
        crate::typeck::typecheck_module(db, resolved, parsed.span_map(db).clone())
    }

    #[salsa_test]
    fn nominal_dependencies_include_substituted_row_constraints(db: &salsa::DatabaseImpl) {
        use crate::ast::{AbilityId, Effect, EffectRow, RowRemoval, RowUnion};
        let input = checked(
            db,
            crate::SourceCst::from_source_str(
                db,
                "constraint_dependency.trb",
                "struct Hidden(a) { value: a }\nenum Root(a) { Value(a), Empty }\nextern \"C\" fn hold(value: Root(Int)) -> Nil",
            ),
        );
        let hidden_decl = input
            .module(db)
            .decls
            .iter()
            .find_map(|d| match d {
                Decl::Struct(s) => Some(s),
                _ => None,
            })
            .unwrap();
        let hidden_id = TypeDefId::source(db, hidden_decl.name, hidden_decl.id);
        let hidden = Type::new(
            db,
            TypeKind::Named {
                id: hidden_id,
                name: hidden_decl.name,
                args: vec![Type::new(db, TypeKind::BoundVar { index: 0 })],
            },
        );
        let row = EffectRow::single(
            db,
            Effect {
                ability_id: AbilityId::source(db, Symbol::new("Marker")),
                args: vec![hidden],
            },
        );
        let ctor = CtorId::new(db, Symbol::new("Value"));
        for removal in [false, true] {
            let mut schemas: HashMap<_, _> = input
                .constructor_types(db)
                .schemes
                .iter()
                .copied()
                .collect();
            let builder = schemas[&ctor].to_builder(db);
            let constrained = if removal {
                builder
                    .row_removals(vec![RowRemoval {
                        source: row,
                        removed: row,
                        result: row,
                    }])
                    .build(db)
            } else {
                builder
                    .row_unions(vec![RowUnion {
                        sources: vec![row],
                        result: row,
                    }])
                    .build(db)
            };
            schemas.insert(ctor, constrained);
            let seeds = collect::collect_type_instantiations(
                db,
                input.module(db),
                input.function_types(db).iter().map(|(_, s)| s.body(db)),
            );
            assert!(!seeds.contains_key(&hidden_id));
            let result = collect(db, input.module(db), &schemas, seeds).unwrap();
            assert_eq!(
                result.instances[&hidden_id],
                HashSet::from([vec![Type::new(db, TypeKind::Int)]])
            );
            let variant = result
                .enum_variants
                .values()
                .find(|s| matches!(s.body(db).kind(db), TypeKind::Func { .. }))
                .unwrap();
            let row = if removal {
                variant.row_removals(db)[0].result
            } else {
                variant.row_unions(db)[0].result
            };
            assert!(
                matches!(row.effects(db)[0].args[0].kind(db), TypeKind::Named { id, args, .. } if *id == hidden_id && args == &[Type::new(db, TypeKind::Int)])
            );
            assert_eq!(schemas[&ctor], constrained);
        }
    }

    #[salsa_test]
    fn nominal_dependencies_deduplicate_cycles_and_bound_expansion(db: &salsa::DatabaseImpl) {
        for (source, rounds, count, succeeds) in [
            (
                "struct Ring(a) { next: Ring(a) }\nextern \"C\" fn hold(value: Ring(Int)) -> Nil",
                1,
                1,
                true,
            ),
            (
                "struct Left(a) { next: Right(a) }\nstruct Right(a) { next: Left(a) }\nextern \"C\" fn hold(value: Left(Int)) -> Nil",
                2,
                2,
                true,
            ),
            (
                "struct Left(a) { next: Right(a) }\nstruct Right(a) { next: Left(a) }\nextern \"C\" fn hold(value: Left(Int)) -> Nil",
                1,
                2,
                false,
            ),
            (
                "struct Left(a) { next: Right(a) }\nstruct Right(a) { next: Left(a) }\nextern \"C\" fn hold(value: Left(Int)) -> Nil",
                2,
                1,
                false,
            ),
            (
                "enum Grow(a) { More(Grow(#(a, a))), End }\nextern \"C\" fn hold(value: Grow(Int)) -> Nil",
                3,
                100,
                false,
            ),
        ] {
            let input = checked(
                db,
                crate::SourceCst::from_source_str(db, "nominal_limits.trb", source),
            );
            let roots = input.function_types(db).iter().map(|(_, s)| s.body(db));
            let seeds = collect::collect_type_instantiations(db, input.module(db), roots);
            assert!(!seeds.is_empty());
            let schemas = input
                .constructor_types(db)
                .schemes
                .iter()
                .copied()
                .collect();
            let result = close_dependencies(db, input.module(db), &schemas, seeds, rounds, count);
            if succeeds {
                let result = result.unwrap();
                assert_eq!(
                    result.instances.values().map(HashSet::len).sum::<usize>(),
                    count
                );
            } else {
                assert!(matches!(
                    result,
                    Err(InstanceError {
                        kind: InstanceErrorKind::ExpansionLimit,
                        ..
                    })
                ));
            }
        }
    }

    #[salsa_test]
    fn nominal_dependency_production_limit_handles_shared_type_growth(db: &salsa::DatabaseImpl) {
        let input = checked(
            db,
            crate::SourceCst::from_source_str(
                db,
                "growing.trb",
                "enum Grow(a) { More(Grow(#(a, a))), End }\nextern \"C\" fn hold(value: Grow(Int)) -> Nil",
            ),
        );
        let seeds = collect::collect_type_instantiations(
            db,
            input.module(db),
            input.function_types(db).iter().map(|(_, s)| s.body(db)),
        );
        let schemes = input
            .constructor_types(db)
            .schemes
            .iter()
            .copied()
            .collect();
        let result = collect(db, input.module(db), &schemes, seeds);
        assert!(
            matches!(result, Err(errors) if errors[0].kind == InstanceErrorKind::ExpansionLimit)
        );
    }

    #[salsa_test]
    fn nominal_dependencies_reject_missing_wrong_owner_and_arity_schemas(db: &salsa::DatabaseImpl) {
        let source = crate::SourceCst::from_source_str(
            db,
            "invalid_nominal_schema.trb",
            "enum Cell(a) { Value(a), Empty }\nextern \"C\" fn hold(value: Cell(Int)) -> Nil",
        );
        let input = checked(db, source);
        let original: HashMap<_, _> = input
            .constructor_types(db)
            .schemes
            .iter()
            .copied()
            .collect();
        let ctor = CtorId::new(db, Symbol::new("Value"));
        for kind in [
            InstanceErrorKind::MissingInstance,
            InstanceErrorKind::WrongDeclaration,
            InstanceErrorKind::InconsistentCallable,
            InstanceErrorKind::TypeArgumentArity {
                expected: 1,
                found: 0,
            },
            InstanceErrorKind::IncompleteTypeArgument,
        ] {
            let mut schemas = original.clone();
            let scheme = original[&ctor];
            if kind == InstanceErrorKind::MissingInstance {
                schemas.remove(&ctor);
            } else if matches!(kind, InstanceErrorKind::TypeArgumentArity { .. }) {
                schemas.insert(ctor, scheme.to_builder(db).type_params(vec![]).build(db));
            } else {
                let TypeKind::Func {
                    params,
                    result,
                    effect,
                    minimum_convention,
                } = scheme.body(db).kind(db)
                else {
                    panic!("constructor signature")
                };
                let body = Type::new(
                    db,
                    TypeKind::Func {
                        params: if kind == InstanceErrorKind::InconsistentCallable {
                            vec![]
                        } else if kind == InstanceErrorKind::IncompleteTypeArgument {
                            vec![Type::new(db, TypeKind::BoundVar { index: 1 })]
                        } else {
                            params.clone()
                        },
                        result: if kind == InstanceErrorKind::WrongDeclaration {
                            Type::new(db, TypeKind::Int)
                        } else {
                            *result
                        },
                        effect: *effect,
                        minimum_convention: *minimum_convention,
                    },
                );
                schemas.insert(
                    ctor,
                    scheme
                        .to_builder(db)
                        .map_types(db, |ty| if ty == scheme.body(db) { body } else { ty })
                        .build(db),
                );
            }
            let seeds = collect::collect_type_instantiations(
                db,
                input.module(db),
                input.function_types(db).iter().map(|(_, s)| s.body(db)),
            );
            let result = collect(db, input.module(db), &schemas, seeds);
            let Err(errors) = result else {
                panic!("invalid schema accepted")
            };
            assert_eq!(errors[0].kind, kind);
            assert_eq!(
                input
                    .constructor_types(db)
                    .schemes
                    .iter()
                    .copied()
                    .collect::<HashMap<_, _>>(),
                original
            );
        }
    }
}
