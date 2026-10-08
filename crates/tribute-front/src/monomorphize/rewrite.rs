//! Call site and type rewriting for monomorphization.
//!
//! Rewrites references to generic functions with their specialized versions
//! by matching the callee's concrete type against collected instantiations.
//! Also rewrites Named types with type arguments to their mangled monomorphic versions.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use trunk_ir::Symbol;

use crate::ast::visit::{RefSite, Refs, walk_decl_mut, walk_module_mut};
use crate::ast::{
    CtorId, Decl, FuncDefId, Module, NodeId, ResolvedRef, Type, TypeDefId, TypeKind, TypedRef,
};

use super::instance::{InstanceKey, InstanceKeys};
use super::mangle::mangle_type_name;

/// Rewrite map: original FuncDefId → list of (type_args, mangled_name) pairs.
pub type RewriteMap<'db> = HashMap<FuncDefId<'db>, Vec<(InstanceKey<'db>, Symbol)>>;

/// Type rewrite map: declaration identity → specialized argument/name pairs.
pub type TypeRewriteMap<'db> = HashMap<TypeDefId<'db>, Vec<(Vec<Type<'db>>, Symbol)>>;

/// Rewrite all generic function call sites in a module to use specialized versions.
pub fn rewrite_module<'db>(
    db: &'db dyn salsa::Database,
    module: &mut Module<TypedRef<'db>>,
    rewrite_map: &RewriteMap<'db>,
    instances: &HashMap<NodeId, crate::typeck::FunctionInstance<'db>>,
    keys: &InstanceKeys<'db>,
) {
    rewrite_decls(db, &mut module.decls, rewrite_map, instances, keys);
}

/// Rewrite call sites in a list of declarations (e.g., specialized function bodies).
pub fn rewrite_decls<'db>(
    db: &'db dyn salsa::Database,
    decls: &mut [Decl<TypedRef<'db>>],
    rewrite_map: &RewriteMap<'db>,
    instances: &HashMap<NodeId, crate::typeck::FunctionInstance<'db>>,
    keys: &InstanceKeys<'db>,
) {
    for decl in decls {
        let enclosing = super::collect::enclosing_instance(decl).cloned();
        let mut rewrite = Refs(|site, node, value: &mut TypedRef<'db>| {
            // Only a function reference in expression position is a call site.
            if site == RefSite::Var
                && let Some(instance) = instances.get(&node)
                && let Some(callee) =
                    specialized_callee(db, rewrite_map, keys, instance, enclosing.as_ref(), value)
            {
                *value = callee;
            }
        });
        walk_decl_mut(&mut rewrite, decl);
    }
}

/// The specialized function the reference `instance` inside `enclosing`
/// refers to, if the callee was specialized for the reference's arguments.
fn specialized_callee<'db>(
    db: &'db dyn salsa::Database,
    rewrite_map: &RewriteMap<'db>,
    keys: &InstanceKeys<'db>,
    instance: &crate::typeck::FunctionInstance<'db>,
    enclosing: Option<&Symbol>,
    typed_ref: &TypedRef<'db>,
) -> Option<TypedRef<'db>> {
    let ResolvedRef::Function { id } = &typed_ref.resolved else {
        return None;
    };
    let entries = rewrite_map.get(id)?;
    if instance.function != *id {
        return None;
    }
    let key = keys.key(instance, enclosing)?;
    let (_, mangled) = entries.iter().find(|(entry, _)| *entry == key)?;
    Some(TypedRef::new(
        ResolvedRef::Function {
            id: FuncDefId::new(db, mangled.clone()),
        },
        typed_ref.ty,
    ))
}

// ============================================================================
// Type rewriting: Named { name, args } → Named { mangled, args: [] }
// ============================================================================

/// Build a type rewrite map from collected type instantiations.
pub fn build_type_rewrite_map<'db>(
    db: &'db dyn salsa::Database,
    instantiations: &HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) -> TypeRewriteMap<'db> {
    let mut map = TypeRewriteMap::default();
    for (id, type_arg_sets) in instantiations {
        let mut entries: Vec<(Vec<Type<'db>>, Symbol)> = type_arg_sets
            .iter()
            .map(|type_args| {
                let mangled = mangle_type_name(db, *id, id.qualified(db).clone(), type_args);
                (type_args.clone(), mangled)
            })
            .collect();
        entries.sort_by_key(|e| e.1.clone());
        map.insert(*id, entries);
    }
    map
}

/// Redirect references to a generic struct's field functions (`T::f`,
/// `T::f::set`, `T::f::modify`) to the same function of the struct's
/// specialization for the reference's type arguments.
///
/// Field functions have no source declaration to clone; each specialized
/// struct declaration has its own.  Run this before the instances' types are
/// rewritten, since the map is keyed by the original type arguments.
pub fn rewrite_field_function_refs<'db>(
    db: &'db dyn salsa::Database,
    module: &mut Module<TypedRef<'db>>,
    type_rewrite_map: &TypeRewriteMap<'db>,
    instances: &mut HashMap<NodeId, crate::typeck::FunctionInstance<'db>>,
) {
    walk_module_mut(
        &mut Refs(|site, node, value: &mut TypedRef<'db>| {
            if site != RefSite::Var {
                return;
            }
            let ResolvedRef::Function { id } = &value.resolved else {
                return;
            };
            let Some(instance) = instances.get_mut(&node) else {
                return;
            };
            let crate::typeck::FunctionInstanceOrigin::FieldAccessor { owner, field, kind } =
                &mut instance.origin
            else {
                return;
            };
            if instance.function != *id || instance.type_arguments.is_empty() {
                return;
            }
            let Some((_, mangled)) = type_rewrite_map.get(owner).and_then(|entries| {
                entries
                    .iter()
                    .find(|(args, _)| *args == instance.type_arguments)
            }) else {
                return;
            };
            let function = FuncDefId::new(db, kind.qualified(mangled, field));
            *owner = owner.with_qualified(db, mangled.clone());
            instance.function = function;
            value.resolved = ResolvedRef::Function { id: function };
        }),
        module,
    );
}

/// Rewrite all Named types with type arguments to their mangled monomorphic versions
/// throughout a module's expressions and patterns.
pub fn rewrite_types_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: &mut Module<TypedRef<'db>>,
    type_rewrite_map: &TypeRewriteMap<'db>,
) {
    walk_module_mut(
        &mut Refs(|_, _, value: &mut TypedRef<'db>| {
            *value = rewrite_typed_ref_type(db, value.clone(), type_rewrite_map);
        }),
        module,
    );
}

/// Rewrite a Type, replacing Named types with non-empty args with their mangled versions.
pub fn rewrite_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    map: &TypeRewriteMap<'db>,
) -> Type<'db> {
    match ty.kind(db) {
        TypeKind::Named { id, name, args } if !args.is_empty() => {
            // The rewrite map is keyed on the original (un-rewritten) args as
            // collected from the module, so look up using `args` directly —
            // not the recursively rewritten ones.
            if let Some(entries) = map.get(id)
                && let Some((_, mangled)) = entries.iter().find(|(ta, _)| ta == args)
            {
                return Type::new(
                    db,
                    TypeKind::Named {
                        id: id.with_qualified(db, mangled.clone()),
                        name: mangled.clone(),
                        args: vec![],
                    },
                );
            }
            // Not in rewrite map — recurse into args (for nested generics like
            // List(Option(Int)) where the outer ctor is non-generic but inner
            // args still need rewriting).
            let rewritten_args: Vec<Type<'db>> =
                args.iter().map(|a| rewrite_type(db, *a, map)).collect();
            Type::new(
                db,
                TypeKind::Named {
                    id: *id,
                    name: name.clone(),
                    args: rewritten_args,
                },
            )
        }
        TypeKind::Func {
            params,
            result,
            effect,
            minimum_convention,
        } => {
            let new_params: Vec<_> = params.iter().map(|p| rewrite_type(db, *p, map)).collect();
            let new_result = rewrite_type(db, *result, map);
            let new_effect = rewrite_row(db, *effect, map);
            if new_params == *params && new_result == *result && new_effect == *effect {
                return ty;
            }
            Type::new(
                db,
                TypeKind::Func {
                    params: new_params,
                    result: new_result,
                    effect: new_effect,
                    minimum_convention: *minimum_convention,
                },
            )
        }
        TypeKind::Tuple(elems) => {
            let new_elems: Vec<_> = elems.iter().map(|e| rewrite_type(db, *e, map)).collect();
            if new_elems == *elems {
                return ty;
            }
            Type::new(db, TypeKind::Tuple(new_elems))
        }
        TypeKind::App { ctor, args } => {
            let new_ctor = rewrite_type(db, *ctor, map);
            let new_args: Vec<_> = args.iter().map(|a| rewrite_type(db, *a, map)).collect();
            if new_ctor == *ctor && new_args == *args {
                return ty;
            }
            Type::new(
                db,
                TypeKind::App {
                    ctor: new_ctor,
                    args: new_args,
                },
            )
        }
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            let new_arg = rewrite_type(db, *arg, map);
            let new_result = rewrite_type(db, *result, map);
            let new_effect = rewrite_row(db, *effect, map);
            if new_arg == *arg && new_result == *result && new_effect == *effect {
                return ty;
            }
            Type::new(
                db,
                TypeKind::Continuation {
                    arg: new_arg,
                    result: new_result,
                    effect: new_effect,
                },
            )
        }
        TypeKind::Named { .. }
        | TypeKind::Int
        | TypeKind::Nat
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Bytes
        | TypeKind::Rune
        | TypeKind::Nil
        | TypeKind::Never
        | TypeKind::BoundVar { .. }
        | TypeKind::LocalBoundVar { .. }
        | TypeKind::UniVar { .. }
        | TypeKind::Error => ty,
    }
}

pub(super) fn rewrite_row<'db>(
    db: &'db dyn salsa::Database,
    row: crate::ast::EffectRow<'db>,
    map: &TypeRewriteMap<'db>,
) -> crate::ast::EffectRow<'db> {
    crate::ast::EffectRow::new(
        db,
        row.effects(db)
            .iter()
            .map(|effect| crate::ast::Effect {
                ability_id: effect.ability_id,
                args: effect
                    .args
                    .iter()
                    .map(|ty| rewrite_type(db, *ty, map))
                    .collect(),
            })
            .collect::<Vec<_>>(),
        row.rest(db),
    )
}

fn rewrite_typed_ref_type<'db>(
    db: &'db dyn salsa::Database,
    tr: TypedRef<'db>,
    map: &TypeRewriteMap<'db>,
) -> TypedRef<'db> {
    let new_ty = rewrite_type(db, tr.ty, map);
    // Also rewrite CtorId/TypeDefId if the type was rewritten
    let resolved = match &tr.resolved {
        ResolvedRef::Constructor { id, variant } => {
            if let Some(mangled) = find_mangled_for_ctor(db, *id, tr.ty, map) {
                ResolvedRef::Constructor {
                    id: CtorId::new(db, mangled),
                    variant: variant.clone(),
                }
            } else {
                tr.resolved.clone()
            }
        }
        ResolvedRef::TypeDef { id } => {
            if let Some(mangled) = find_mangled_for_typedef(db, *id, tr.ty, map) {
                ResolvedRef::TypeDef {
                    id: id.with_qualified(db, mangled),
                }
            } else {
                tr.resolved.clone()
            }
        }
        _ => tr.resolved.clone(),
    };
    TypedRef::new(resolved, new_ty)
}

fn find_mangled_for_ctor<'db>(
    db: &'db dyn salsa::Database,
    _ctor_id: CtorId<'db>,
    ty: Type<'db>,
    map: &TypeRewriteMap<'db>,
) -> Option<Symbol> {
    // Extract the result type from the constructor's function type
    let result_ty = match ty.kind(db) {
        TypeKind::Func { result, .. } => *result,
        _ => ty,
    };
    // Check if the result type is a Named type with args
    match result_ty.kind(db) {
        TypeKind::Named { id, args, .. } if !args.is_empty() => {
            let entries = map.get(id)?;
            // Match against the original args stored in the map.
            let (_, mangled) = entries.iter().find(|(ta, _)| ta == args)?;
            Some(mangled.clone())
        }
        _ => None,
    }
}

fn find_mangled_for_typedef<'db>(
    db: &'db dyn salsa::Database,
    _type_def_id: TypeDefId<'db>,
    ty: Type<'db>,
    map: &TypeRewriteMap<'db>,
) -> Option<Symbol> {
    match ty.kind(db) {
        TypeKind::Named { id, args, .. } if !args.is_empty() => {
            let entries = map.get(id)?;
            // Match against the original args stored in the map.
            let (_, mangled) = entries.iter().find(|(ta, _)| ta == args)?;
            Some(mangled.clone())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::EffectRow;

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    fn pure_effect(db: &dyn salsa::Database) -> EffectRow<'_> {
        EffectRow::new(db, vec![], None)
    }

    fn make_type_rewrite_map<'db>(
        db: &'db dyn salsa::Database,
        entries: Vec<(Symbol, Vec<Type<'db>>, Symbol)>,
    ) -> TypeRewriteMap<'db> {
        let mut map = TypeRewriteMap::default();
        for (name, args, mangled) in entries {
            let id = TypeDefId::synthetic(db, name);
            map.entry(id).or_default().push((args, mangled));
        }
        map
    }

    #[test]
    fn test_rewrite_type_named_with_args() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Option")),
                name: Symbol::new("Option"),
                args: vec![int],
            },
        );
        let map = make_type_rewrite_map(
            &db,
            vec![(Symbol::new("Option"), vec![int], Symbol::new("Option$Int"))],
        );

        let result = rewrite_type(&db, option_int, &map);
        match result.kind(&db) {
            TypeKind::Named { name, args, .. } => {
                assert_eq!(name.to_string(), "Option$Int");
                assert!(args.is_empty());
            }
            other => panic!("expected Named, got {:?}", other),
        }
    }

    #[test]
    fn test_type_rewrite_map_keeps_same_spelled_types_distinct() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let source = |name: &str, node| {
            TypeDefId::source(&db, Symbol::new(name), crate::ast::NodeId::from_raw(node))
        };
        let cases = [
            (
                "builtin and source List",
                [
                    (TypeDefId::builtin_list(&db), "5$List$Int"),
                    (source("List", 1), "List$Int"),
                ],
            ),
            (
                "same-spelled source types",
                [
                    (source("A::Thing", 1), "A::Thing$Int"),
                    (source("B::Thing", 2), "B::Thing$Int"),
                ],
            ),
        ];
        for (name, ids) in cases {
            let instantiations = ids
                .iter()
                .map(|(id, _)| (*id, [vec![int]].into_iter().collect::<HashSet<_>>()))
                .collect::<HashMap<_, _>>();

            let map = build_type_rewrite_map(&db, &instantiations);

            assert_eq!(map.len(), 2, "{name}");
            for (id, mangled) in ids {
                assert_eq!(map[&id][0].1.to_string(), mangled, "{name}");
            }
        }
    }

    #[test]
    fn test_rewrite_type_without_map_entry_is_unchanged() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let named = |name: &str, args| {
            Type::new(
                &db,
                TypeKind::Named {
                    id: crate::ast::TypeDefId::synthetic(&db, Symbol::new(name)),
                    name: Symbol::new(name),
                    args,
                },
            )
        };
        let map = TypeRewriteMap::default();
        for (name, ty) in [
            ("primitive Int", int),
            ("Text without args", named("Text", vec![])),
            ("Unknown(Int) not in map", named("Unknown", vec![int])),
        ] {
            assert_eq!(rewrite_type(&db, ty, &map), ty, "{name}");
        }
    }

    #[test]
    fn test_rewrite_type_nested_in_func() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Option")),
                name: Symbol::new("Option"),
                args: vec![int],
            },
        );
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![option_int],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let map = make_type_rewrite_map(
            &db,
            vec![(Symbol::new("Option"), vec![int], Symbol::new("Option$Int"))],
        );

        let result = rewrite_type(&db, func_ty, &map);
        match result.kind(&db) {
            TypeKind::Func { params, .. } => match params[0].kind(&db) {
                TypeKind::Named { name, args, .. } => {
                    assert_eq!(name.to_string(), "Option$Int");
                    assert!(args.is_empty());
                }
                other => panic!("expected Named, got {:?}", other),
            },
            other => panic!("expected Func, got {:?}", other),
        }
    }

    /// Regression: outer generic with nested generic args must still be
    /// mangled. The map is keyed on original args, so recursing before lookup
    /// would cause a miss.
    #[test]
    fn test_rewrite_type_outer_generic_with_nested_generic_args() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Option")),
                name: Symbol::new("Option"),
                args: vec![int],
            },
        );
        let pair_option_int_bool = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, Symbol::new("Pair")),
                name: Symbol::new("Pair"),
                args: vec![option_int, bool_ty],
            },
        );
        let map = make_type_rewrite_map(
            &db,
            vec![
                (Symbol::new("Option"), vec![int], Symbol::new("Option$Int")),
                (
                    Symbol::new("Pair"),
                    vec![option_int, bool_ty],
                    Symbol::new("Pair$Option$0$Int$1$Bool"),
                ),
            ],
        );

        let result = rewrite_type(&db, pair_option_int_bool, &map);
        match result.kind(&db) {
            TypeKind::Named { name, args, .. } => {
                assert_eq!(name.to_string(), "Pair$Option$0$Int$1$Bool");
                assert!(args.is_empty());
            }
            other => panic!("expected mangled Named, got {:?}", other),
        }
    }
}
