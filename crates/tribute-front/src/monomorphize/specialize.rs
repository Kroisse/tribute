use super::nominal_index::{Declaration, NominalDeclaration, NominalIndex};
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::hash::{Hash, Hasher};
use std::num::NonZero;

use trunk_ir::Symbol;

use crate::ast::visit::{RefSite, Visit, VisitMut};
use crate::ast::{
    Decl, EnumDecl, ExternFuncDecl, FieldDecl, FuncDecl, FuncDefId, Module, NodeId, StructDecl,
    Type, TypeAnnotation, TypeAnnotationKind, TypeDefId, TypeKind, TypeScheme, TypedRef,
    VariantDecl,
};
use crate::typeck::subst::substitute_bound_vars;

use super::instance::InstanceKey;
use super::mangle::{mangle_instance_name, mangle_type_name};

pub(super) struct GeneratedSpecializations<'db> {
    pub(super) specialized_declarations: Vec<FuncDecl<TypedRef<'db>>>,
    pub(super) specialized_extern_declarations: Vec<ExternFuncDecl>,
    pub(super) specialized_function_types: Vec<(Symbol, TypeScheme<'db>)>,
    /// The name, key, and source nodes of each generated function.
    pub(super) metadata_origins: Vec<(Symbol, InstanceKey<'db>, HashSet<NodeId>)>,
    pub(super) compiler_intrinsic_specializations: Vec<(NodeId, Symbol)>,
}

struct SpecializationEntry<'db> {
    name: Symbol,
    declaration: FuncDecl<TypedRef<'db>>,
    scheme: TypeScheme<'db>,
    key: InstanceKey<'db>,
    origins: HashSet<NodeId>,
}

/// Generate specialized copies of generic functions for each instantiation.
///
/// Returns a list of new specialized `FuncDecl`s and their corresponding
/// `(Symbol, TypeScheme)` entries for `function_types`.
pub(super) fn generate_specializations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    instantiations: &HashMap<FuncDefId<'db>, HashSet<InstanceKey<'db>>>,
    function_types: &[(Symbol, TypeScheme<'db>)],
    compiler_intrinsics: &HashMap<NodeId, Symbol>,
) -> GeneratedSpecializations<'db> {
    let func_decls = collect_func_decls(module);
    let extern_functions = collect_extern_function_decls(module);
    let scheme_map: HashMap<Symbol, TypeScheme<'db>> = function_types.iter().cloned().collect();

    let mut entries: Vec<SpecializationEntry<'db>> = Vec::new();
    let mut extern_function_types = Vec::new();
    let mut specialized_extern_declarations = Vec::new();
    let mut compiler_intrinsic_specializations = Vec::new();

    for (func_id, keys) in instantiations {
        let qualified = func_id.qualified(db);
        let func = func_decls.get(qualified).copied();
        let extern_function = extern_functions.get(qualified).copied();
        if func.is_none() && extern_function.is_none() {
            continue;
        }
        let Some(scheme) = scheme_map.get(qualified) else {
            continue;
        };
        let origins = func.map(semantic_node_ids);

        for key in keys {
            let type_args = key.type_args.as_slice();
            let mangled = mangle_instance_name(db, qualified, key);
            let specialized_scheme = scheme
                .to_builder(db)
                .map_types(db, |ty| {
                    substitute_bound_vars(db, ty, type_args).unwrap_or_else(|index, max| {
                        panic!(
                            "BoundVar index out of range in specialization of {}: index={}, subst.len()={}",
                            qualified, index, max
                        )
                    })
                })
                .type_params(Vec::new())
                .build(db);
            if let Some(func) = func {
                let specialized = specialize_func_decl(db, func, key, mangled.clone());
                entries.push(SpecializationEntry {
                    name: mangled,
                    declaration: specialized,
                    scheme: specialized_scheme,
                    key: key.clone(),
                    origins: origins.clone().expect("function specialization origins"),
                });
            } else {
                // Extern functions have no AST body to clone, but rewritten
                // call sites still need their concrete scheme during logical
                // lowering under the mangled identity.
                extern_function_types.push((mangled.clone(), specialized_scheme));
                let extern_function = extern_function.expect("extern specialization declaration");
                if let Some(identity) = compiler_intrinsics.get(&extern_function.id).cloned() {
                    let declaration = specialize_extern_decl(extern_function, key, mangled);
                    compiler_intrinsic_specializations.push((declaration.id, identity));
                    specialized_extern_declarations.push(declaration);
                }
            }
        }
    }

    // Sort by mangled name for deterministic output (HashMap/HashSet iteration is unordered)
    entries.sort_by_key(|entry| entry.name.clone());

    let mut new_decls = Vec::with_capacity(entries.len());
    let mut new_function_types = Vec::with_capacity(entries.len() + extern_function_types.len());
    let mut metadata_origins = Vec::with_capacity(entries.len());
    for entry in entries {
        new_decls.push(entry.declaration);
        new_function_types.push((entry.name.clone(), entry.scheme));
        metadata_origins.push((entry.name, entry.key, entry.origins));
    }
    new_function_types.extend(extern_function_types);
    new_function_types.sort_by_key(|(name, _)| name.clone());
    specialized_extern_declarations.sort_by_key(|declaration| declaration.name.clone());
    compiler_intrinsic_specializations.sort_by_key(|(id, identity)| (identity.clone(), *id));

    GeneratedSpecializations {
        specialized_declarations: new_decls,
        specialized_extern_declarations,
        specialized_function_types: new_function_types,
        metadata_origins,
        compiler_intrinsic_specializations,
    }
}

fn semantic_node_ids<'db>(func: &FuncDecl<TypedRef<'db>>) -> HashSet<NodeId> {
    struct Ids(HashSet<NodeId>);
    impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for Ids {
        fn visit_node_id(&mut self, id: NodeId) {
            self.0.insert(id);
        }
    }
    let mut ids = Ids(HashSet::default());
    ids.visit_func_decl(func);
    ids.0
}

// ============================================================================
// Generic struct/enum specialization
// ============================================================================

/// Generate specialized struct declarations for each type instantiation.
pub fn generate_struct_specializations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    instantiations: &HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) -> Vec<StructDecl> {
    let index = NominalIndex::new(db, module);
    generate_struct_specializations_with_index(db, &index, instantiations)
}

pub(super) fn generate_struct_specializations_with_index<'db>(
    db: &'db dyn salsa::Database,
    index: &NominalIndex<'_, 'db>,
    instantiations: &HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) -> Vec<StructDecl> {
    let mut entries: Vec<(Symbol, StructDecl)> = Vec::new();

    for (id, type_arg_sets) in instantiations {
        let Some(Declaration {
            source: NominalDeclaration::Struct(decl),
            ..
        }) = index.declarations.get(id)
        else {
            continue;
        };
        if decl.type_params.is_empty() {
            continue;
        }

        for type_args in type_arg_sets {
            let mangled = mangle_type_name(db, *id, id.qualified(db).clone(), type_args);
            let specialized = specialize_struct_decl(db, decl, type_args, mangled.clone());
            entries.push((mangled, specialized));
        }
    }

    entries.sort_by_key(|e| e.0.clone());
    entries.into_iter().map(|(_, decl)| decl).collect()
}

/// Generate specialized enum declarations for each type instantiation.
pub fn generate_enum_specializations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    instantiations: &HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) -> Vec<EnumDecl> {
    let index = NominalIndex::new(db, module);
    generate_enum_specializations_with_index(db, &index, instantiations)
}

pub(super) fn generate_enum_specializations_with_index<'db>(
    db: &'db dyn salsa::Database,
    index: &NominalIndex<'_, 'db>,
    instantiations: &HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) -> Vec<EnumDecl> {
    let mut entries: Vec<(Symbol, EnumDecl)> = Vec::new();

    for (id, type_arg_sets) in instantiations {
        let Some(Declaration {
            source: NominalDeclaration::Enum(decl),
            ..
        }) = index.declarations.get(id)
        else {
            continue;
        };
        if decl.type_params.is_empty() {
            continue;
        }

        for type_args in type_arg_sets {
            let mangled = mangle_type_name(db, *id, id.qualified(db).clone(), type_args);
            let specialized = specialize_enum_decl(db, decl, type_args, mangled.clone());
            entries.push((mangled, specialized));
        }
    }

    entries.sort_by_key(|e| e.0.clone());
    entries.into_iter().map(|(_, decl)| decl).collect()
}

fn specialize_struct_decl<'db>(
    db: &'db dyn salsa::Database,
    decl: &StructDecl,
    type_args: &[Type<'db>],
    mangled_name: Symbol,
) -> StructDecl {
    let variant = type_args_variant(type_args);
    let param_names: Vec<Symbol> = decl.type_params.iter().map(|p| p.name.clone()).collect();

    StructDecl {
        id: decl.id.with_variant(variant),
        is_pub: false,
        name: mangled_name,
        type_params: vec![],
        fields: decl
            .fields
            .iter()
            .map(|f| FieldDecl {
                id: f.id.with_variant(variant),
                name_id: f.name_id.with_variant(variant),
                is_pub: f.is_pub,
                name: f.name.clone(),
                ty: substitute_annotation(db, &f.ty, &param_names, type_args),
            })
            .collect(),
    }
}

fn specialize_enum_decl<'db>(
    db: &'db dyn salsa::Database,
    decl: &EnumDecl,
    type_args: &[Type<'db>],
    mangled_name: Symbol,
) -> EnumDecl {
    let variant = type_args_variant(type_args);
    let param_names: Vec<Symbol> = decl.type_params.iter().map(|p| p.name.clone()).collect();

    EnumDecl {
        id: decl.id.with_variant(variant),
        is_pub: false,
        name: mangled_name,
        type_params: vec![],
        variants: decl
            .variants
            .iter()
            .map(|v| VariantDecl {
                id: v.id.with_variant(variant),
                name: v.name.clone(),
                fields: v
                    .fields
                    .iter()
                    .map(|f| FieldDecl {
                        id: f.id.with_variant(variant),
                        name_id: f.name_id.with_variant(variant),
                        is_pub: f.is_pub,
                        name: f.name.clone(),
                        ty: substitute_annotation(db, &f.ty, &param_names, type_args),
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Substitute type parameter names in a TypeAnnotation with concrete types.
///
/// If a `TypeAnnotationKind::Named(name)` matches a type parameter name,
/// it's replaced with a TypeAnnotation for the concrete type.
fn substitute_annotation<'db>(
    db: &'db dyn salsa::Database,
    ann: &TypeAnnotation,
    param_names: &[Symbol],
    type_args: &[Type<'db>],
) -> TypeAnnotation {
    let kind = match &ann.kind {
        TypeAnnotationKind::Named(name) => {
            // Check if this name matches a type parameter
            if let Some(idx) = param_names.iter().position(|p| p == name)
                && let Some(ty) = type_args.get(idx)
            {
                return type_to_annotation(db, *ty, ann.id);
            }
            ann.kind.clone()
        }
        TypeAnnotationKind::App { ctor, args } => TypeAnnotationKind::App {
            ctor: Box::new(substitute_annotation(db, ctor, param_names, type_args)),
            args: args
                .iter()
                .map(|a| substitute_annotation(db, a, param_names, type_args))
                .collect(),
        },
        TypeAnnotationKind::Func {
            params,
            result,
            abilities,
        } => TypeAnnotationKind::Func {
            params: params
                .iter()
                .map(|p| substitute_annotation(db, p, param_names, type_args))
                .collect(),
            result: Box::new(substitute_annotation(db, result, param_names, type_args)),
            abilities: abilities.clone(),
        },
        TypeAnnotationKind::Tuple(elems) => TypeAnnotationKind::Tuple(
            elems
                .iter()
                .map(|e| substitute_annotation(db, e, param_names, type_args))
                .collect(),
        ),
        TypeAnnotationKind::Path(_) | TypeAnnotationKind::Infer | TypeAnnotationKind::Error => {
            ann.kind.clone()
        }
    };
    TypeAnnotation { id: ann.id, kind }
}

/// Convert a semantic Type to a TypeAnnotation.
///
/// Used when generating specialized field types — the concrete Type
/// from type checking is mapped back to a source-level annotation.
fn type_to_annotation(db: &dyn salsa::Database, ty: Type<'_>, id: NodeId) -> TypeAnnotation {
    let kind = match ty.kind(db) {
        TypeKind::Int => TypeAnnotationKind::Named(Symbol::new("Int")),
        TypeKind::Nat => TypeAnnotationKind::Named(Symbol::new("Nat")),
        TypeKind::Float => TypeAnnotationKind::Named(Symbol::new("Float")),
        TypeKind::Bool => TypeAnnotationKind::Named(Symbol::new("Bool")),
        TypeKind::Bytes => TypeAnnotationKind::Named(Symbol::new("Bytes")),
        TypeKind::Rune => TypeAnnotationKind::Named(Symbol::new("Rune")),
        TypeKind::Nil => TypeAnnotationKind::Named(Symbol::new("Nil")),
        TypeKind::Never => TypeAnnotationKind::Named(Symbol::new("Never")),
        TypeKind::Named {
            id: type_id,
            name,
            args,
        } => {
            if args.is_empty() {
                TypeAnnotationKind::Named(name.clone())
            } else {
                // Use mangled name for generic types with args
                let mangled = mangle_type_name(db, *type_id, name.clone(), args);
                TypeAnnotationKind::Named(mangled)
            }
        }
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            // Preserve the effect row as ability annotations. Each Effect
            // becomes a Named (or App) annotation; a row variable (`rest`) is
            // represented as `Infer`, matching the "effect polymorphic" encoding
            // documented on TypeAnnotationKind::Func.
            let mut abilities: Vec<TypeAnnotation> = effect
                .effects(db)
                .iter()
                .map(|eff| {
                    let name = eff.ability_id.name(db);
                    if eff.args.is_empty() {
                        TypeAnnotation {
                            id,
                            kind: TypeAnnotationKind::Named(name),
                        }
                    } else {
                        TypeAnnotation {
                            id,
                            kind: TypeAnnotationKind::App {
                                ctor: Box::new(TypeAnnotation {
                                    id,
                                    kind: TypeAnnotationKind::Named(name),
                                }),
                                args: eff
                                    .args
                                    .iter()
                                    .map(|a| type_to_annotation(db, *a, id))
                                    .collect(),
                            },
                        }
                    }
                })
                .collect();
            if effect.rest(db).is_some() {
                abilities.push(TypeAnnotation {
                    id,
                    kind: TypeAnnotationKind::Infer,
                });
            }
            TypeAnnotationKind::Func {
                params: params
                    .iter()
                    .map(|p| type_to_annotation(db, *p, id))
                    .collect(),
                result: Box::new(type_to_annotation(db, *result, id)),
                abilities,
            }
        }
        TypeKind::Tuple(elems) => TypeAnnotationKind::Tuple(
            elems
                .iter()
                .map(|e| type_to_annotation(db, *e, id))
                .collect(),
        ),
        _ => TypeAnnotationKind::Infer,
    };
    TypeAnnotation { id, kind }
}

// ============================================================================
// Generic function specialization helpers
// ============================================================================

fn collect_func_decls<'a, 'db>(
    module: &'a Module<TypedRef<'db>>,
) -> HashMap<Symbol, &'a FuncDecl<TypedRef<'db>>> {
    let mut map = HashMap::default();
    let mut prefix = String::new();
    collect_func_decls_inner(&module.decls, &mut prefix, &mut map);
    map
}

fn collect_func_decls_inner<'a, 'db>(
    decls: &'a [Decl<TypedRef<'db>>],
    prefix: &mut String,
    map: &mut HashMap<Symbol, &'a FuncDecl<TypedRef<'db>>>,
) {
    for decl in decls {
        match decl {
            Decl::Function(func) => {
                let qualified = crate::qualified_symbol(prefix, &func.name);
                map.insert(qualified, func);
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let len = crate::push_prefix(prefix, &m.name);
                    collect_func_decls_inner(body, prefix, map);
                    prefix.truncate(len);
                }
            }
            _ => {}
        }
    }
}

fn collect_extern_function_decls<'a, 'db>(
    module: &'a Module<TypedRef<'db>>,
) -> HashMap<Symbol, &'a ExternFuncDecl> {
    let mut declarations = HashMap::default();
    let mut prefix = String::new();
    collect_extern_function_decls_inner(&module.decls, &mut prefix, &mut declarations);
    declarations
}

fn collect_extern_function_decls_inner<'a, 'db>(
    decls: &'a [Decl<TypedRef<'db>>],
    prefix: &mut String,
    declarations: &mut HashMap<Symbol, &'a ExternFuncDecl>,
) {
    for decl in decls {
        match decl {
            Decl::ExternFunction(func) => {
                declarations.insert(crate::qualified_symbol(prefix, &func.name), func);
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    let len = crate::push_prefix(prefix, &module.name);
                    collect_extern_function_decls_inner(body, prefix, declarations);
                    prefix.truncate(len);
                }
            }
            _ => {}
        }
    }
}

fn specialize_extern_decl(
    declaration: &ExternFuncDecl,
    key: &InstanceKey<'_>,
    mangled_name: Symbol,
) -> ExternFuncDecl {
    ExternFuncDecl {
        id: declaration.id.with_variant(key.variant()),
        is_pub: false,
        name: mangled_name,
        abi: declaration.abi.clone(),
        params: declaration.params.clone(),
        return_ty: declaration.return_ty.clone(),
    }
}

/// Compute a NonZero<u64> variant hash from concrete type arguments.
///
/// This is used to give specialized AST nodes unique NodeIds that
/// don't collide with the original or other specializations.
pub(crate) fn type_args_variant(type_args: &[Type<'_>]) -> NonZero<u64> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    type_args.hash(&mut hasher);
    let hash = hasher.finish();
    // Ensure non-zero: if hash happens to be 0, use 1
    NonZero::new(hash).unwrap_or(NonZero::new(1).unwrap())
}

fn specialize_func_decl<'db>(
    db: &'db dyn salsa::Database,
    func: &FuncDecl<TypedRef<'db>>,
    key: &InstanceKey<'db>,
    mangled_name: Symbol,
) -> FuncDecl<TypedRef<'db>> {
    let mut specialized = FuncDecl {
        is_pub: false,
        name: mangled_name,
        type_params: vec![],
        ..func.clone()
    };
    Substitute::new(db, key).visit_func_decl_mut(&mut specialized);
    specialized
}

// ============================================================================
// Expr-level type substitution
// ============================================================================

fn subst_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    type_args: &[Type<'db>],
) -> Type<'db> {
    substitute_bound_vars(db, ty, type_args).unwrap_or_else(|index, max| {
        panic!(
            "BoundVar index out of range during monomorphization: index={index}, subst.len()={max}"
        )
    })
}

/// Substitutes the type arguments into every reference type of a copied
/// declaration, and gives every semantic node identity the specialization's
/// variant.
struct Substitute<'a, 'db> {
    db: &'db dyn salsa::Database,
    type_args: &'a [Type<'db>],
    variant: NonZero<u64>,
}

impl<'a, 'db> Substitute<'a, 'db> {
    fn new(db: &'db dyn salsa::Database, key: &'a InstanceKey<'db>) -> Self {
        Self {
            db,
            type_args: &key.type_args,
            variant: key.variant(),
        }
    }
}

impl<'db> VisitMut<TypedRef<'db>> for Substitute<'_, 'db> {
    fn visit_ref_mut(&mut self, _: RefSite, _: NodeId, value: &mut TypedRef<'db>) {
        value.ty = subst_type(self.db, value.ty, self.type_args);
    }

    fn visit_node_id_mut(&mut self, id: &mut NodeId) {
        *id = id.with_variant(self.variant);
    }
}

#[cfg(test)]
mod tests {
    use super::super::mangle::mangle_name;
    use crate::ast::{
        Arm, EffectRow, Expr, ExprKind, FieldPattern, HandlerArm, HandlerKind, Pattern,
        PatternKind, ResolvedRef, Stmt, TypeParam,
    };

    use super::*;

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

    fn node_id(n: usize) -> NodeId {
        NodeId::from_raw(n)
    }

    #[test]
    fn semantic_node_ids_include_nested_pattern_nodes() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let typed_ref = |index| {
            TypedRef::new(
                ResolvedRef::Local {
                    id: crate::ast::LocalId::new(index),
                    name: Symbol::new("value"),
                },
                int,
            )
        };
        let bind = |id, name| {
            Pattern::new(
                node_id(id),
                PatternKind::Bind {
                    name: Symbol::new(name),
                    local_id: None,
                },
            )
        };
        let nil = |id| Expr::new(node_id(id), ExprKind::Nil);

        let let_pattern = Pattern::new(
            node_id(10),
            PatternKind::Record {
                type_name: typed_ref(2),
                fields: vec![FieldPattern {
                    id: node_id(11),
                    name_id: node_id(11),
                    name: Symbol::new("field"),
                    pattern: Some(Pattern::new(
                        node_id(12),
                        PatternKind::As {
                            pattern: Pattern::new(
                                node_id(13),
                                PatternKind::Tuple(vec![bind(14, "tuple")]),
                            ),
                            name: Symbol::new("record"),
                            local_id: None,
                        },
                    )),
                }],
                rest: false,
            },
        );
        let case = Expr::new(
            node_id(20),
            ExprKind::Case {
                scrutinee: nil(21),
                arms: vec![Arm {
                    id: node_id(22),
                    pattern: Pattern::new(
                        node_id(23),
                        PatternKind::Variant {
                            ctor: typed_ref(0),
                            fields: vec![Pattern::new(
                                node_id(24),
                                PatternKind::ListRest {
                                    head: vec![bind(25, "head")],
                                    rest: Some(Symbol::new("tail")),
                                    rest_local_id: None,
                                },
                            )],
                        },
                    ),
                    guard: None,
                    body: nil(26),
                }],
            },
        );
        let handled = Expr::new(
            node_id(30),
            ExprKind::Handle {
                body: case,
                handlers: vec![
                    HandlerArm {
                        id: node_id(31),
                        kind: HandlerKind::Do {
                            binding: Pattern::new(
                                node_id(32),
                                PatternKind::List(vec![bind(33, "result")]),
                            ),
                        },
                        body: nil(34),
                    },
                    HandlerArm {
                        id: node_id(35),
                        kind: HandlerKind::Fn {
                            ability: typed_ref(1),
                            op: Symbol::new("fn_op"),
                            params: vec![Pattern::new(
                                node_id(36),
                                PatternKind::Tuple(vec![bind(37, "fn_param")]),
                            )],
                        },
                        body: nil(38),
                    },
                ],
            },
        );
        let function = FuncDecl {
            id: node_id(1),
            is_pub: false,
            name: Symbol::new("patterns"),
            type_params: vec![],
            params: vec![],
            return_ty: None,
            effects: None,
            body: Expr::new(
                node_id(2),
                ExprKind::Block {
                    stmts: vec![Stmt::Let {
                        id: node_id(3),
                        pattern: let_pattern,
                        ty: None,
                        value: nil(4),
                    }],
                    value: handled,
                },
            ),
        };

        let ids = semantic_node_ids(&function);
        for id in [
            1, 3, 10, 11, 12, 13, 14, 22, 23, 24, 25, 31, 32, 33, 35, 36, 37,
        ] {
            assert!(
                ids.contains(&node_id(id)),
                "semantic metadata must follow nested pattern node {id}"
            );
        }
    }

    #[test]
    fn test_substitute_expr_var() {
        let db = TestDb::default();
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(&db, TypeKind::Int);

        let tr = TypedRef::new(
            ResolvedRef::Function {
                id: FuncDefId::new(&db, Symbol::new("f")),
            },
            bv0,
        );
        let mut result = Expr::new(node_id(1), ExprKind::Var(tr));
        Substitute::new(&db, &InstanceKey::of_types(vec![int])).visit_expr_mut(&mut result);

        // NodeId should have the variant applied
        assert!(result.id.variant().is_some());
        assert_eq!(result.id.origin(), node_id(1));

        match result.kind.as_ref() {
            ExprKind::Var(tr) => assert_eq!(tr.ty, int),
            _ => panic!("expected Var"),
        }
    }

    #[test]
    fn test_specialize_func_decl_basic() {
        let db = TestDb::default();
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(&db, TypeKind::Int);

        // Build a simple generic function: fn identity(a)(x: a) -> a { x }
        let body_ref = TypedRef::new(
            ResolvedRef::Local {
                id: crate::ast::LocalId::new(0),
                name: Symbol::new("x"),
            },
            bv0,
        );
        let body = Expr::new(node_id(10), ExprKind::Var(body_ref));

        let func = FuncDecl {
            id: node_id(1),
            is_pub: true,
            name: Symbol::new("identity"),
            type_params: vec![crate::ast::TypeParamDecl {
                id: node_id(2),
                name: Symbol::new("a"),
                bounds: vec![],
            }],
            params: vec![],
            return_ty: None,
            effects: None,
            body,
        };

        let mangled = mangle_name(&db, &Symbol::new("identity"), &[int]);
        let specialized =
            specialize_func_decl(&db, &func, &InstanceKey::of_types(vec![int]), mangled);

        assert_eq!(specialized.name.to_string(), "identity$Int");
        assert!(specialized.type_params.is_empty());
        assert!(!specialized.is_pub);

        // Body should have Int instead of BoundVar(0)
        match specialized.body.kind.as_ref() {
            ExprKind::Var(tr) => assert_eq!(tr.ty, int),
            _ => panic!("expected Var"),
        }
    }

    #[test]
    fn test_generate_specializations_produces_correct_count() {
        let db = TestDb::default();
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(&db, TypeKind::Int);
        let float = Type::new(&db, TypeKind::Float);

        let func_name = Symbol::new("identity");
        let func_id = FuncDefId::new(&db, func_name.clone());

        // Build function
        let body_ref = TypedRef::new(
            ResolvedRef::Local {
                id: crate::ast::LocalId::new(0),
                name: Symbol::new("x"),
            },
            bv0,
        );
        let body = Expr::new(node_id(10), ExprKind::Var(body_ref));
        let func = FuncDecl {
            id: node_id(1),
            is_pub: true,
            name: func_name.clone(),
            type_params: vec![crate::ast::TypeParamDecl {
                id: node_id(2),
                name: Symbol::new("a"),
                bounds: vec![],
            }],
            params: vec![],
            return_ty: None,
            effects: None,
            body,
        };

        let module = Module::new(node_id(0), None, vec![Decl::Function(func)]);

        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bv0],
                result: bv0,
                effect: pure_effect(&db),
            },
        );
        let writer = crate::ast::AbilityId::source(&db, Symbol::new("Writer"));
        let row = |ty| {
            crate::ast::EffectRow::single(
                &db,
                crate::ast::Effect {
                    ability_id: writer,
                    args: vec![ty],
                },
            )
        };
        let scheme = TypeScheme::builder(vec![TypeParam::anonymous()], vec![], scheme_body)
            .row_unions(vec![crate::ast::RowUnion {
                sources: vec![row(bv0)],
                result: row(bv0),
            }])
            .row_removals(vec![crate::ast::RowRemoval {
                source: row(bv0),
                removed: row(bv0),
                result: pure_effect(&db),
            }])
            .build(&db);
        let function_types = vec![(func_name, scheme)];

        let mut type_arg_sets = HashSet::default();
        type_arg_sets.insert(InstanceKey::of_types(vec![int]));
        type_arg_sets.insert(InstanceKey::of_types(vec![float]));
        let mut instantiations = HashMap::default();
        instantiations.insert(func_id, type_arg_sets);

        let specializations = generate_specializations(
            &db,
            &module,
            &instantiations,
            &function_types,
            &HashMap::default(),
        );

        assert_eq!(specializations.specialized_declarations.len(), 2);
        assert_eq!(specializations.specialized_function_types.len(), 2);

        // Verify names are mangled
        let names: HashSet<String> = specializations
            .specialized_declarations
            .iter()
            .map(|d| d.name.to_string())
            .collect();
        assert!(names.contains("identity$Int"));
        assert!(names.contains("identity$Float"));

        // All specialized TypeSchemes must be monomorphic
        for (name, scheme) in &specializations.specialized_function_types {
            assert!(
                scheme.is_mono(&db),
                "specialized scheme should have no type params"
            );
            let expected = if *name == "identity$Int" { int } else { float };
            assert_eq!(
                scheme.row_unions(&db),
                &vec![crate::ast::RowUnion {
                    sources: vec![row(expected)],
                    result: row(expected),
                }]
            );
            assert_eq!(
                scheme.row_removals(&db),
                &vec![crate::ast::RowRemoval {
                    source: row(expected),
                    removed: row(expected),
                    result: pure_effect(&db),
                }]
            );
        }
    }

    // ========================================================================
    // type_to_annotation tests
    // ========================================================================

    #[test]
    fn test_type_to_annotation_named_types() {
        let db = TestDb::default();
        let id = node_id(1);
        let named = |name: &str, args| TypeKind::Named {
            id: crate::ast::TypeDefId::synthetic(&db, Symbol::new(name)),
            name: Symbol::new(name),
            args,
        };
        let int = Type::new(&db, TypeKind::Int);
        // A named type with arguments uses its mangled name.
        let cases = [
            (TypeKind::Int, "Int"),
            (TypeKind::Nat, "Nat"),
            (TypeKind::Float, "Float"),
            (TypeKind::Bool, "Bool"),
            (TypeKind::Nil, "Nil"),
            (named("Text", vec![]), "Text"),
            (named("Option", vec![int]), "Option$Int"),
        ];
        for (kind, expected_name) in cases {
            let ty = Type::new(&db, kind);
            let ann = type_to_annotation(&db, ty, id);
            match &ann.kind {
                TypeAnnotationKind::Named(name) => {
                    assert_eq!(name.to_string(), expected_name);
                }
                other => panic!("expected Named annotation for {expected_name:?}, got {other:?}"),
            }
        }
    }

    /// Regression: effect row must be preserved when converting Func types
    /// back to annotations (previously the `abilities` field was hardcoded
    /// to `vec![]`, silently turning effectful functions into pure ones).
    #[test]
    fn test_type_to_annotation_func_preserves_abilities() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let console_id = crate::ast::AbilityId::source(&db, Symbol::new("std::console::Console"));
        let state_id = crate::ast::AbilityId::source(&db, Symbol::new("std::state::State"));
        let effect = EffectRow::new(
            &db,
            vec![
                crate::ast::Effect {
                    ability_id: console_id,
                    args: vec![],
                },
                crate::ast::Effect {
                    ability_id: state_id,
                    args: vec![int],
                },
            ],
            None,
        );
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
            },
        );
        let ann = type_to_annotation(&db, func_ty, node_id(1));
        match &ann.kind {
            TypeAnnotationKind::Func { abilities, .. } => {
                assert_eq!(abilities.len(), 2);
                match &abilities[0].kind {
                    TypeAnnotationKind::Named(name) => {
                        assert_eq!(name.to_string(), "Console");
                    }
                    other => panic!("expected Named(Console), got {:?}", other),
                }
                match &abilities[1].kind {
                    TypeAnnotationKind::App { ctor, args } => {
                        match &ctor.kind {
                            TypeAnnotationKind::Named(name) => {
                                assert_eq!(name.to_string(), "State");
                            }
                            other => panic!("expected Named(State) ctor, got {:?}", other),
                        }
                        assert_eq!(args.len(), 1);
                        match &args[0].kind {
                            TypeAnnotationKind::Named(name) => {
                                assert_eq!(name.to_string(), "Int");
                            }
                            other => panic!("expected Named(Int) arg, got {:?}", other),
                        }
                    }
                    other => panic!("expected App(State, [Int]), got {:?}", other),
                }
            }
            other => panic!("expected Func annotation, got {:?}", other),
        }
    }

    /// Regression: a row variable on the effect row should map to `Infer`
    /// (effect-polymorphic), not be silently dropped.
    #[test]
    fn test_type_to_annotation_func_preserves_rest_var() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let effect = EffectRow::new(&db, vec![], Some(crate::ast::EffectVar { id: 0 }));
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
            },
        );
        let ann = type_to_annotation(&db, func_ty, node_id(1));
        match &ann.kind {
            TypeAnnotationKind::Func { abilities, .. } => {
                assert_eq!(abilities.len(), 1);
                assert!(matches!(abilities[0].kind, TypeAnnotationKind::Infer));
            }
            other => panic!("expected Func annotation, got {:?}", other),
        }
    }

    #[test]
    fn test_type_to_annotation_func_preserves_closed_empty_row() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect: EffectRow::pure(&db),
            },
        );

        let ann = type_to_annotation(&db, func_ty, node_id(1));
        let TypeAnnotationKind::Func { abilities, .. } = ann.kind else {
            panic!("expected Func annotation");
        };
        assert!(abilities.is_empty());
    }

    // ========================================================================
    // specialize_struct_decl tests
    // ========================================================================

    #[test]
    fn test_specialize_struct_decl_basic() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        // struct Pair(a, b) { first: a, second: b }
        let decl = StructDecl {
            id: node_id(1),
            is_pub: true,
            name: Symbol::new("Pair"),
            type_params: vec![
                crate::ast::TypeParamDecl {
                    id: node_id(2),
                    name: Symbol::new("a"),
                    bounds: vec![],
                },
                crate::ast::TypeParamDecl {
                    id: node_id(3),
                    name: Symbol::new("b"),
                    bounds: vec![],
                },
            ],
            fields: vec![
                FieldDecl {
                    id: node_id(4),
                    name_id: node_id(4),
                    is_pub: false,
                    name: Some(Symbol::new("first")),
                    ty: TypeAnnotation {
                        id: node_id(5),
                        kind: TypeAnnotationKind::Named(Symbol::new("a")),
                    },
                },
                FieldDecl {
                    id: node_id(6),
                    name_id: node_id(6),
                    is_pub: false,
                    name: Some(Symbol::new("second")),
                    ty: TypeAnnotation {
                        id: node_id(7),
                        kind: TypeAnnotationKind::Named(Symbol::new("b")),
                    },
                },
            ],
        };

        let mangled = mangle_name(&db, &Symbol::new("Pair"), &[int, bool_ty]);
        let specialized = specialize_struct_decl(&db, &decl, &[int, bool_ty], mangled);

        assert_eq!(specialized.name.to_string(), "Pair$Int$Bool");
        assert!(specialized.type_params.is_empty());
        assert_eq!(specialized.fields.len(), 2);

        // first field should be Int
        match &specialized.fields[0].ty.kind {
            TypeAnnotationKind::Named(name) => assert_eq!(name.to_string(), "Int"),
            other => panic!("expected Named(Int), got {:?}", other),
        }
        // second field should be Bool
        match &specialized.fields[1].ty.kind {
            TypeAnnotationKind::Named(name) => assert_eq!(name.to_string(), "Bool"),
            other => panic!("expected Named(Bool), got {:?}", other),
        }
    }

    // ========================================================================
    // specialize_enum_decl tests
    // ========================================================================

    #[test]
    fn test_specialize_enum_decl_basic() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);

        // enum Option(a) { Some(a), None }
        let decl = EnumDecl {
            id: node_id(1),
            is_pub: true,
            name: Symbol::new("Option"),
            type_params: vec![crate::ast::TypeParamDecl {
                id: node_id(2),
                name: Symbol::new("a"),
                bounds: vec![],
            }],
            variants: vec![
                VariantDecl {
                    id: node_id(3),
                    name: Symbol::new("Some"),
                    fields: vec![FieldDecl {
                        id: node_id(4),
                        name_id: node_id(4),
                        is_pub: false,
                        name: None,
                        ty: TypeAnnotation {
                            id: node_id(5),
                            kind: TypeAnnotationKind::Named(Symbol::new("a")),
                        },
                    }],
                },
                VariantDecl {
                    id: node_id(6),
                    name: Symbol::new("None"),
                    fields: vec![],
                },
            ],
        };

        let mangled = mangle_name(&db, &Symbol::new("Option"), &[int]);
        let specialized = specialize_enum_decl(&db, &decl, &[int], mangled);

        assert_eq!(specialized.name.to_string(), "Option$Int");
        assert!(specialized.type_params.is_empty());
        assert_eq!(specialized.variants.len(), 2);

        // Some variant should have Int field
        assert_eq!(specialized.variants[0].name.to_string(), "Some");
        assert_eq!(specialized.variants[0].fields.len(), 1);
        match &specialized.variants[0].fields[0].ty.kind {
            TypeAnnotationKind::Named(name) => assert_eq!(name.to_string(), "Int"),
            other => panic!("expected Named(Int), got {:?}", other),
        }

        // None variant should have no fields
        assert_eq!(specialized.variants[1].name.to_string(), "None");
        assert!(specialized.variants[1].fields.is_empty());
    }

    // ========================================================================
    // substitute_annotation tests
    // ========================================================================

    #[test]
    fn test_substitute_annotation_replaces_only_params() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        for (input, expected) in [("a", "Int"), ("Text", "Text")] {
            let ann = TypeAnnotation {
                id: node_id(1),
                kind: TypeAnnotationKind::Named(Symbol::new(input)),
            };
            let result = substitute_annotation(&db, &ann, &[Symbol::new("a")], &[int]);
            match &result.kind {
                TypeAnnotationKind::Named(name) => {
                    assert_eq!(name.to_string(), expected, "{input}")
                }
                other => panic!("{input}: expected Named({expected}), got {other:?}"),
            }
        }
    }
}
