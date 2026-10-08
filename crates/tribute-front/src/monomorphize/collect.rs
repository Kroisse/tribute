use super::instance::InstanceKey;
use super::nominal_index::NominalIndex;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use crate::ast::visit::{RefSite, Refs, walk_module};
use crate::ast::{FuncDefId, Module, ResolvedRef, Type, TypeDefId, TypeKind, TypeScheme, TypedRef};

/// Collect all generic function instantiations from a typed module.
///
/// Traverses the AST and records which concrete type argument combinations
/// each generic function is called with. The result maps each polymorphic
/// `FuncDefId` to the set of concrete type argument lists used.
pub(super) fn collect_instantiations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    function_types: &[(trunk_ir::Symbol, TypeScheme<'db>)],
    function_instances: &HashMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
) -> HashMap<FuncDefId<'db>, HashSet<InstanceKey<'db>>> {
    let mut collector = InstantiationCollector::new(db, function_types, function_instances);
    walk_module(
        &mut Refs(|site, node, value: &TypedRef<'db>| {
            // Only a function reference in expression position is a call site.
            if site == RefSite::Var {
                collector.try_record(node, value);
            }
        }),
        module,
    );
    collector.instantiations
}

/// Extract concrete type arguments by walking the scheme body and concrete type
/// in parallel. When the scheme has `BoundVar(i)`, the concrete type at that
/// position becomes `type_args[i]`.
///
/// Returns `None` if the scheme is monomorphic, or if extraction fails
/// (e.g., structural mismatch or inconsistent BoundVar mappings).
#[cfg(test)]
fn extract_type_args<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    concrete: Type<'db>,
) -> Option<Vec<Type<'db>>> {
    let num_params = scheme.type_params(db).len();
    if num_params == 0 {
        return None;
    }
    let mut type_args: Vec<Option<Type<'db>>> = vec![None; num_params];
    if !extract_recursive(db, scheme.body(db), concrete, &mut type_args) {
        return None;
    }
    type_args.into_iter().collect()
}

#[cfg(test)]
fn extract_recursive<'db>(
    db: &'db dyn salsa::Database,
    scheme_ty: Type<'db>,
    concrete_ty: Type<'db>,
    type_args: &mut [Option<Type<'db>>],
) -> bool {
    match (scheme_ty.kind(db), concrete_ty.kind(db)) {
        (TypeKind::BoundVar { index }, _) => {
            let i = *index as usize;
            if i >= type_args.len() {
                return false;
            }
            match type_args[i] {
                Some(existing) => existing == concrete_ty,
                None => {
                    type_args[i] = Some(concrete_ty);
                    true
                }
            }
        }
        (
            TypeKind::Func {
                params: sp,
                result: sr,
                ..
            },
            TypeKind::Func {
                params: cp,
                result: cr,
                ..
            },
        ) => {
            if sp.len() != cp.len() {
                return false;
            }
            for (s, c) in sp.iter().zip(cp.iter()) {
                if !extract_recursive(db, *s, *c, type_args) {
                    return false;
                }
            }
            extract_recursive(db, *sr, *cr, type_args)
        }
        (
            TypeKind::Named {
                id: si, args: sa, ..
            },
            TypeKind::Named {
                id: ci, args: ca, ..
            },
        ) => {
            if si != ci || sa.len() != ca.len() {
                return false;
            }
            for (s, c) in sa.iter().zip(ca.iter()) {
                if !extract_recursive(db, *s, *c, type_args) {
                    return false;
                }
            }
            true
        }
        (TypeKind::Tuple(se), TypeKind::Tuple(ce)) => {
            if se.len() != ce.len() {
                return false;
            }
            for (s, c) in se.iter().zip(ce.iter()) {
                if !extract_recursive(db, *s, *c, type_args) {
                    return false;
                }
            }
            true
        }
        // Primitives and other identical types: interned equality
        _ => scheme_ty == concrete_ty,
    }
}

struct InstantiationCollector<'a, 'db> {
    db: &'db dyn salsa::Database,
    schemes: HashMap<FuncDefId<'db>, TypeScheme<'db>>,
    function_instances: &'a HashMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
    instantiations: HashMap<FuncDefId<'db>, HashSet<InstanceKey<'db>>>,
}

impl<'a, 'db> InstantiationCollector<'a, 'db> {
    fn new(
        db: &'db dyn salsa::Database,
        function_types: &[(trunk_ir::Symbol, TypeScheme<'db>)],
        function_instances: &'a HashMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
    ) -> Self {
        let schemes = function_types
            .iter()
            .filter(|(_, scheme)| !scheme.is_mono(db))
            .map(|(sym, scheme)| (FuncDefId::new(db, sym.clone()), *scheme))
            .collect();
        Self {
            db,
            schemes,
            function_instances,
            instantiations: HashMap::default(),
        }
    }

    fn try_record(&mut self, node_id: crate::ast::NodeId, typed_ref: &TypedRef<'db>) {
        let ResolvedRef::Function { id } = &typed_ref.resolved else {
            return;
        };
        let Some(scheme) = self.schemes.get(id) else {
            return;
        };
        let Some(instance) = self.function_instances.get(&node_id) else {
            return;
        };
        if instance.function != *id {
            return;
        }
        let Some(key) = InstanceKey::of(self.db, *scheme, instance) else {
            return;
        };
        if !key
            .type_args
            .iter()
            .all(|ty| is_concrete_type(self.db, *ty))
        {
            return;
        }
        self.instantiations.entry(*id).or_default().insert(key);
    }
}

pub(crate) fn is_concrete_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    is_concrete_type_cached(db, ty, &mut HashMap::default())
}

fn is_concrete_type_cached<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    cache: &mut HashMap<Type<'db>, bool>,
) -> bool {
    if let Some(concrete) = cache.get(&ty) {
        return *concrete;
    }
    let concrete = match ty.kind(db) {
        TypeKind::Named { args, .. } => args
            .iter()
            .all(|arg| is_concrete_type_cached(db, *arg, cache)),
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            params
                .iter()
                .all(|param| is_concrete_type_cached(db, *param, cache))
                && is_concrete_type_cached(db, *result, cache)
                && is_concrete_effect_row_cached(db, *effect, cache)
        }
        TypeKind::Tuple(elements) => elements
            .iter()
            .all(|element| is_concrete_type_cached(db, *element, cache)),
        TypeKind::Int
        | TypeKind::Nat
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Bytes
        | TypeKind::Rune
        | TypeKind::Nil
        | TypeKind::Never => true,
        TypeKind::BoundVar { .. }
        | TypeKind::LocalBoundVar { .. }
        | TypeKind::UniVar { .. }
        | TypeKind::App { .. }
        | TypeKind::Error => false,
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            is_concrete_type_cached(db, *arg, cache)
                && is_concrete_type_cached(db, *result, cache)
                && is_concrete_effect_row_cached(db, *effect, cache)
        }
    };
    cache.insert(ty, concrete);
    concrete
}

fn is_concrete_effect_row_cached<'db>(
    db: &'db dyn salsa::Database,
    row: crate::ast::EffectRow<'db>,
    cache: &mut HashMap<Type<'db>, bool>,
) -> bool {
    row.rest(db).is_none()
        && row.effects(db).iter().all(|effect| {
            effect
                .args
                .iter()
                .all(|arg| is_concrete_type_cached(db, *arg, cache))
        })
}

// ============================================================================
// Type instantiation collection (for generic struct/enum monomorphization)
// ============================================================================

/// Collect all generic type instantiations from a typed module.
///
/// Walks all types in the AST (recursively through Func, Tuple, Named, etc.)
/// and records which concrete type argument combinations each generic type
/// is used with. Only collects instances of indexed generic struct/enum
/// declarations (those with non-empty `type_params`), matched by TypeDefId.
pub fn collect_type_instantiations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    extra_types: impl IntoIterator<Item = Type<'db>>,
) -> HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let index = NominalIndex::new(db, module);
    collect_type_instantiations_with_index(db, module, extra_types, &index)
}

pub(super) fn collect_type_instantiations_with_index<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    extra_types: impl IntoIterator<Item = Type<'db>>,
    index: &NominalIndex<'_, 'db>,
) -> HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let mut result: HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> = HashMap::default();

    walk_module(
        &mut Refs(|site, _, value: &TypedRef<'db>| {
            // Patterns and handler abilities are not collected.
            if matches!(site, RefSite::Var | RefSite::ConsCtor | RefSite::RecordType) {
                collect_from_type(db, value.ty, index, &mut result);
            }
        }),
        module,
    );
    for ty in extra_types {
        collect_from_type(db, ty, index, &mut result);
    }
    result
}

/// Recursively walk a Type and collect all declaration-backed generic types.
pub(super) fn collect_from_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    index: &NominalIndex<'_, 'db>,
    result: &mut HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) {
    collect_from_type_inner(db, ty, index, result, &mut HashSet::default());
}

// Interned types form a DAG. Visit shared arguments once, including during
// expanding nominal dependency discovery, so its round limit remains effective.
fn collect_from_type_inner<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    index: &NominalIndex<'_, 'db>,
    result: &mut HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
    seen: &mut HashSet<Type<'db>>,
) {
    if !seen.insert(ty) {
        return;
    }
    match ty.kind(db) {
        TypeKind::Named { id, args, .. } => {
            if !args.is_empty()
                && index.is_generic(*id)
                && args.iter().all(|arg| is_concrete_type(db, *arg))
            {
                result.entry(*id).or_default().insert(args.clone());
            }
            // Recurse into type arguments (e.g., List(Option(Int)) → collect Option(Int))
            for arg in args {
                collect_from_type_inner(db, *arg, index, result, seen);
            }
        }
        TypeKind::Func {
            params,
            result: ret,
            effect,
            ..
        } => {
            for p in params {
                collect_from_type_inner(db, *p, index, result, seen);
            }
            collect_from_type_inner(db, *ret, index, result, seen);
            for effect in effect.effects(db) {
                for arg in &effect.args {
                    collect_from_type_inner(db, *arg, index, result, seen);
                }
            }
        }
        TypeKind::Tuple(elems) => {
            for e in elems {
                collect_from_type_inner(db, *e, index, result, seen);
            }
        }
        TypeKind::App { ctor, args } => {
            collect_from_type_inner(db, *ctor, index, result, seen);
            for a in args {
                collect_from_type_inner(db, *a, index, result, seen);
            }
        }
        TypeKind::Continuation {
            arg,
            result: ret,
            effect,
        } => {
            collect_from_type_inner(db, *arg, index, result, seen);
            collect_from_type_inner(db, *ret, index, result, seen);
            for effect in effect.effects(db) {
                for arg in &effect.args {
                    collect_from_type_inner(db, *arg, index, result, seen);
                }
            }
        }
        // Primitives and variables: no nested Named types
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{AbilityId, Decl, Effect, EffectRow, EffectVar, NodeId, TypeParam};
    use trunk_ir::Symbol;

    use super::*;

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    fn make_scheme<'db>(
        db: &'db dyn salsa::Database,
        num_params: usize,
        body: Type<'db>,
    ) -> TypeScheme<'db> {
        let type_params: Vec<_> = (0..num_params).map(|_| TypeParam::anonymous()).collect();
        TypeScheme::new(db, type_params, Vec::new(), body)
    }

    fn pure_effect(db: &dyn salsa::Database) -> EffectRow<'_> {
        EffectRow::new(db, vec![], None)
    }

    fn direct_function_type<'db>(
        db: &'db dyn salsa::Database,
        effect: EffectRow<'db>,
    ) -> Type<'db> {
        let int = Type::new(db, TypeKind::Int);
        Type::new(
            db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        )
    }

    #[test]
    fn concrete_type_accepts_closed_effect_rows_with_concrete_ability_arguments() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let row = EffectRow::new(
            &db,
            vec![Effect {
                ability_id: AbilityId::source(&db, Symbol::new("State")),
                args: vec![int],
            }],
            None,
        );

        assert!(is_concrete_type(&db, direct_function_type(&db, row)));
    }

    #[test]
    fn concrete_type_rejects_open_effect_rows() {
        let db = TestDb::default();
        let open = EffectRow::open(&db, EffectVar { id: 0 });
        let int = Type::new(&db, TypeKind::Int);
        let continuation = Type::new(
            &db,
            TypeKind::Continuation {
                arg: int,
                result: int,
                effect: open,
            },
        );

        assert!(!is_concrete_type(&db, direct_function_type(&db, open)));
        assert!(!is_concrete_type(&db, continuation));
    }

    #[test]
    fn concrete_type_rejects_error_types_in_effect_arguments() {
        let db = TestDb::default();
        let error = Type::new(&db, TypeKind::Error);
        let row = EffectRow::new(
            &db,
            vec![Effect {
                ability_id: AbilityId::source(&db, Symbol::new("State")),
                args: vec![error],
            }],
            None,
        );

        assert!(!is_concrete_type(&db, error));
        assert!(!is_concrete_type(&db, direct_function_type(&db, row)));
    }

    // ========================================================================
    // extract_type_args tests
    // ========================================================================

    #[test]
    fn test_extract_type_args_from_instantiation() {
        let db = TestDb::default();
        let func = |params, result| {
            Type::new(
                &db,
                TypeKind::Func {
                    params,
                    result,
                    effect: pure_effect(&db),
                    minimum_convention: crate::ast::CallingConvention::Direct,
                },
            )
        };
        let named = |name: &str, args| {
            Type::new(
                &db,
                TypeKind::Named {
                    id: crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new(name)),
                    name: trunk_ir::Symbol::new(name),
                    args,
                },
            )
        };
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let bv1 = Type::new(&db, TypeKind::BoundVar { index: 1 });
        let int = Type::new(&db, TypeKind::Int);
        let float = Type::new(&db, TypeKind::Float);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let text = named("Text", vec![]);

        // (name, scheme parameter count, scheme body, concrete type, expected)
        let cases = [
            (
                "∀a. a → a",
                1,
                func(vec![bv0], bv0),
                func(vec![int], int),
                Some(vec![int]),
            ),
            (
                "∀a,b. (a, b) → a",
                2,
                func(vec![bv0, bv1], bv0),
                func(vec![int, float], int),
                Some(vec![int, float]),
            ),
            (
                "∀a. (a, a) → a",
                1,
                func(vec![bv0, bv0], bv0),
                func(vec![int, int], int),
                Some(vec![int]),
            ),
            (
                "∀a. (a, a) → a with inconsistent (Int, Text) → Int",
                1,
                func(vec![bv0, bv0], bv0),
                func(vec![int, text], int),
                None,
            ),
            (
                "∀a. Option(a) → a",
                1,
                func(vec![named("Option", vec![bv0])], bv0),
                func(vec![named("Option", vec![int])], int),
                Some(vec![int]),
            ),
            (
                "∀a,b. (fn(a) → b, a) → b",
                2,
                func(vec![func(vec![bv0], bv1), bv0], bv1),
                func(vec![func(vec![int], bool_ty), int], bool_ty),
                Some(vec![int, bool_ty]),
            ),
            (
                "monomorphic Int → Int",
                0,
                func(vec![int], int),
                func(vec![int], int),
                None,
            ),
        ];

        for (name, num_params, body, concrete, expected) in cases {
            let scheme = make_scheme(&db, num_params, body);
            assert_eq!(extract_type_args(&db, scheme, concrete), expected, "{name}");
        }
    }

    // ========================================================================
    // collect_type_instantiations tests
    // ========================================================================

    fn nominal_module<'db>() -> Module<TypedRef<'db>> {
        Module {
            id: NodeId::from_raw(0),
            name: None,
            decls: [
                ("Option", 1),
                ("Result", 2),
                ("List", 1),
                ("Pair", 2),
                ("Plain", 0),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, (name, arity))| {
                Decl::Struct(crate::ast::StructDecl {
                    id: NodeId::from_raw(i + 1),
                    is_pub: false,
                    name: Symbol::new(name),
                    type_params: (0..arity)
                        .map(|p| crate::ast::TypeParamDecl {
                            id: NodeId::from_raw(100 + i * 2 + p),
                            name: Symbol::new(&format!("t{p}")),
                            bounds: vec![],
                        })
                        .collect(),
                    fields: vec![],
                })
            })
            .collect(),
        }
    }

    fn nominal_id<'db>(
        index: &NominalIndex<'_, 'db>,
        db: &'db dyn salsa::Database,
        name: &str,
    ) -> TypeDefId<'db> {
        *index
            .declarations
            .keys()
            .find(|id| id.qualified(db).with_str(|s| s == name))
            .unwrap()
    }

    #[test]
    fn test_collect_type_instantiations() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let option_id = nominal_id(&index, &db, "Option");
        let result_id = nominal_id(&index, &db, "Result");
        let list_id = nominal_id(&index, &db, "List");
        let pair_id = nominal_id(&index, &db, "Pair");
        let named = |id, args| {
            Type::new(
                &db,
                TypeKind::Named {
                    id,
                    name: id.qualified(&db).clone(),
                    args,
                },
            )
        };
        let int = Type::new(&db, TypeKind::Int);
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let local_bound = Type::new(
            &db,
            TypeKind::LocalBoundVar {
                scope: NodeId::from_raw(0),
                index: 0,
            },
        );
        let option_int = named(option_id, vec![int]);
        let pair_int_int = named(pair_id, vec![int, int]);

        // (name, type, expected instantiations by declaration)
        let cases = [
            (
                "concrete Option(Int)",
                option_int,
                vec![(option_id, vec![vec![int]])],
            ),
            ("Option(BoundVar)", named(option_id, vec![bound]), vec![]),
            (
                "Option(LocalBoundVar)",
                named(option_id, vec![local_bound]),
                vec![],
            ),
            (
                "concrete child below skipped Result(BoundVar, Option(Int))",
                named(result_id, vec![bound, option_int]),
                vec![(option_id, vec![vec![int]])],
            ),
            (
                "nested List(Option(Int))",
                named(list_id, vec![option_int]),
                vec![
                    (option_id, vec![vec![int]]),
                    (list_id, vec![vec![option_int]]),
                ],
            ),
            (
                "fn(Pair(Int, Int)) -> Int",
                Type::new(
                    &db,
                    TypeKind::Func {
                        params: vec![pair_int_int],
                        result: int,
                        effect: pure_effect(&db),
                        minimum_convention: crate::ast::CallingConvention::Direct,
                    },
                ),
                vec![(pair_id, vec![vec![int, int]])],
            ),
        ];

        for (name, ty, expected) in cases {
            let mut result = HashMap::default();
            collect_from_type(&db, ty, &index, &mut result);
            let expected = expected
                .into_iter()
                .map(|(id, args)| (id, args.into_iter().collect::<HashSet<_>>()))
                .collect::<HashMap<_, _>>();
            assert_eq!(result, expected, "{name}");
        }
    }

    #[test]
    fn test_collect_type_ignores_non_generic() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let int = Type::new(&db, TypeKind::Int);
        let mut result = HashMap::default();
        // Neither an unknown type nor an indexed non-generic declaration qualifies,
        // even if the type carries arguments.
        for id in [
            TypeDefId::synthetic(&db, Symbol::new("Unknown")),
            nominal_id(&index, &db, "Plain"),
        ] {
            assert!(!index.is_generic(id));
            let ty = Type::new(
                &db,
                TypeKind::Named {
                    id,
                    name: id.qualified(&db).clone(),
                    args: vec![int],
                },
            );
            collect_from_type(&db, ty, &index, &mut result);
        }

        assert!(result.is_empty());
    }

    // ========================================================================
    // extract_type_args tests (continued)
    // ========================================================================
}
