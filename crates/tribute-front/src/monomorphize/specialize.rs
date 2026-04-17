use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZero;

use trunk_ir::Symbol;

use crate::ast::{
    Arm, Decl, EnumDecl, Expr, ExprKind, FieldDecl, FieldPattern, FuncDecl, FuncDefId, HandlerArm,
    HandlerKind, Module, NodeId, Pattern, PatternKind, Stmt, StructDecl, Type, TypeAnnotation,
    TypeAnnotationKind, TypeKind, TypeScheme, TypedRef, VariantDecl,
};
use crate::typeck::subst::substitute_bound_vars;

use super::mangle::mangle_name;

/// Generate specialized copies of generic functions for each instantiation.
///
/// Returns a list of new specialized `FuncDecl`s and their corresponding
/// `(Symbol, TypeScheme)` entries for `function_types`.
pub fn generate_specializations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    instantiations: &HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>>,
    function_types: &[(Symbol, TypeScheme<'db>)],
) -> (Vec<FuncDecl<TypedRef<'db>>>, Vec<(Symbol, TypeScheme<'db>)>) {
    let func_decls = collect_func_decls(module);
    let scheme_map: HashMap<Symbol, TypeScheme<'db>> = function_types.iter().cloned().collect();

    let mut entries: Vec<(Symbol, FuncDecl<TypedRef<'db>>, TypeScheme<'db>)> = Vec::new();

    for (func_id, type_arg_sets) in instantiations {
        let qualified = func_id.qualified(db);
        let Some(func) = func_decls.get(&qualified) else {
            continue;
        };
        let Some(scheme) = scheme_map.get(&qualified) else {
            continue;
        };

        for type_args in type_arg_sets {
            let mangled = mangle_name(db, qualified, type_args);
            let specialized = specialize_func_decl(db, func, type_args, mangled);
            let specialized_scheme = TypeScheme::new(
                db,
                vec![],
                substitute_bound_vars(db, scheme.body(db), type_args).unwrap_or_else(
                    |index, max| {
                        panic!(
                            "BoundVar index out of range in specialization of {}: index={}, subst.len()={}",
                            qualified, index, max
                        )
                    },
                ),
            );
            entries.push((mangled, specialized, specialized_scheme));
        }
    }

    // Sort by mangled name for deterministic output (HashMap/HashSet iteration is unordered)
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut new_decls = Vec::with_capacity(entries.len());
    let mut new_function_types = Vec::with_capacity(entries.len());
    for (name, decl, scheme) in entries {
        new_decls.push(decl);
        new_function_types.push((name, scheme));
    }

    (new_decls, new_function_types)
}

// ============================================================================
// Generic struct/enum specialization
// ============================================================================

/// Generate specialized struct declarations for each type instantiation.
pub fn generate_struct_specializations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    instantiations: &HashMap<Symbol, HashSet<Vec<Type<'db>>>>,
) -> Vec<StructDecl> {
    let struct_decls = collect_struct_decls(module);
    let mut entries: Vec<(Symbol, StructDecl)> = Vec::new();

    for (name, type_arg_sets) in instantiations {
        let Some(decl) = struct_decls.get(name) else {
            continue;
        };
        if decl.type_params.is_empty() {
            continue;
        }

        for type_args in type_arg_sets {
            let mangled = mangle_name(db, *name, type_args);
            let specialized = specialize_struct_decl(db, decl, type_args, mangled);
            entries.push((mangled, specialized));
        }
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.into_iter().map(|(_, decl)| decl).collect()
}

/// Generate specialized enum declarations for each type instantiation.
pub fn generate_enum_specializations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    instantiations: &HashMap<Symbol, HashSet<Vec<Type<'db>>>>,
) -> Vec<EnumDecl> {
    let enum_decls = collect_enum_decls(module);
    let mut entries: Vec<(Symbol, EnumDecl)> = Vec::new();

    for (name, type_arg_sets) in instantiations {
        let Some(decl) = enum_decls.get(name) else {
            continue;
        };
        if decl.type_params.is_empty() {
            continue;
        }

        for type_args in type_arg_sets {
            let mangled = mangle_name(db, *name, type_args);
            let specialized = specialize_enum_decl(db, decl, type_args, mangled);
            entries.push((mangled, specialized));
        }
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.into_iter().map(|(_, decl)| decl).collect()
}

fn specialize_struct_decl<'db>(
    db: &'db dyn salsa::Database,
    decl: &StructDecl,
    type_args: &[Type<'db>],
    mangled_name: Symbol,
) -> StructDecl {
    let variant = type_args_variant(type_args);
    let param_names: Vec<Symbol> = decl.type_params.iter().map(|p| p.name).collect();

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
                is_pub: f.is_pub,
                name: f.name,
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
    let param_names: Vec<Symbol> = decl.type_params.iter().map(|p| p.name).collect();

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
                name: v.name,
                fields: v
                    .fields
                    .iter()
                    .map(|f| FieldDecl {
                        id: f.id.with_variant(variant),
                        is_pub: f.is_pub,
                        name: f.name,
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
        TypeKind::Named { name, args } => {
            if args.is_empty() {
                TypeAnnotationKind::Named(*name)
            } else {
                // Use mangled name for generic types with args
                let mangled = mangle_name(db, *name, args);
                TypeAnnotationKind::Named(mangled)
            }
        }
        TypeKind::Func {
            params,
            result,
            effect,
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

fn collect_struct_decls<'a>(module: &'a Module<TypedRef<'_>>) -> HashMap<Symbol, &'a StructDecl> {
    let mut map = HashMap::new();
    let mut prefix = String::new();
    collect_struct_decls_inner(&module.decls, &mut prefix, &mut map);
    map
}

fn collect_struct_decls_inner<'a>(
    decls: &'a [Decl<TypedRef<'_>>],
    prefix: &mut String,
    map: &mut HashMap<Symbol, &'a StructDecl>,
) {
    for decl in decls {
        match decl {
            Decl::Struct(s) => {
                let qualified = crate::qualified_symbol(prefix, s.name);
                map.insert(qualified, s);
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let len = crate::push_prefix(prefix, m.name);
                    collect_struct_decls_inner(body, prefix, map);
                    prefix.truncate(len);
                }
            }
            _ => {}
        }
    }
}

fn collect_enum_decls<'a>(module: &'a Module<TypedRef<'_>>) -> HashMap<Symbol, &'a EnumDecl> {
    let mut map = HashMap::new();
    let mut prefix = String::new();
    collect_enum_decls_inner(&module.decls, &mut prefix, &mut map);
    map
}

fn collect_enum_decls_inner<'a>(
    decls: &'a [Decl<TypedRef<'_>>],
    prefix: &mut String,
    map: &mut HashMap<Symbol, &'a EnumDecl>,
) {
    for decl in decls {
        match decl {
            Decl::Enum(e) => {
                let qualified = crate::qualified_symbol(prefix, e.name);
                map.insert(qualified, e);
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let len = crate::push_prefix(prefix, m.name);
                    collect_enum_decls_inner(body, prefix, map);
                    prefix.truncate(len);
                }
            }
            _ => {}
        }
    }
}

// ============================================================================
// Generic function specialization helpers
// ============================================================================

fn collect_func_decls<'a, 'db>(
    module: &'a Module<TypedRef<'db>>,
) -> HashMap<Symbol, &'a FuncDecl<TypedRef<'db>>> {
    let mut map = HashMap::new();
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
                let qualified = crate::qualified_symbol(prefix, func.name);
                map.insert(qualified, func);
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let len = crate::push_prefix(prefix, m.name);
                    collect_func_decls_inner(body, prefix, map);
                    prefix.truncate(len);
                }
            }
            _ => {}
        }
    }
}

/// Compute a NonZero<u64> variant hash from concrete type arguments.
///
/// This is used to give specialized AST nodes unique NodeIds that
/// don't collide with the original or other specializations.
fn type_args_variant(type_args: &[Type<'_>]) -> NonZero<u64> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    type_args.hash(&mut hasher);
    let hash = hasher.finish();
    // Ensure non-zero: if hash happens to be 0, use 1
    NonZero::new(hash).unwrap_or(NonZero::new(1).unwrap())
}

fn specialize_func_decl<'db>(
    db: &'db dyn salsa::Database,
    func: &FuncDecl<TypedRef<'db>>,
    type_args: &[Type<'db>],
    mangled_name: Symbol,
) -> FuncDecl<TypedRef<'db>> {
    let variant = type_args_variant(type_args);
    FuncDecl {
        id: func.id.with_variant(variant),
        is_pub: false,
        name: mangled_name,
        type_params: vec![],
        params: func.params.clone(),
        return_ty: func.return_ty.clone(),
        effects: func.effects.clone(),
        body: substitute_expr(db, func.body.clone(), type_args, variant),
    }
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

fn subst_typed_ref<'db>(
    db: &'db dyn salsa::Database,
    tr: TypedRef<'db>,
    type_args: &[Type<'db>],
) -> TypedRef<'db> {
    TypedRef {
        resolved: tr.resolved,
        ty: subst_type(db, tr.ty, type_args),
    }
}

fn substitute_expr<'db>(
    db: &'db dyn salsa::Database,
    expr: Expr<TypedRef<'db>>,
    type_args: &[Type<'db>],
    variant: NonZero<u64>,
) -> Expr<TypedRef<'db>> {
    let kind = match *expr.kind {
        ExprKind::Var(tr) => ExprKind::Var(subst_typed_ref(db, tr, type_args)),
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: substitute_expr(db, callee, type_args, variant),
            args: args
                .into_iter()
                .map(|a| substitute_expr(db, a, type_args, variant))
                .collect(),
        },
        ExprKind::Block { stmts, value } => ExprKind::Block {
            stmts: stmts
                .into_iter()
                .map(|s| substitute_stmt(db, s, type_args, variant))
                .collect(),
            value: substitute_expr(db, value, type_args, variant),
        },
        ExprKind::Case { scrutinee, arms } => ExprKind::Case {
            scrutinee: substitute_expr(db, scrutinee, type_args, variant),
            arms: arms
                .into_iter()
                .map(|a| substitute_arm(db, a, type_args, variant))
                .collect(),
        },
        ExprKind::Lambda { params, body } => ExprKind::Lambda {
            params,
            body: substitute_expr(db, body, type_args, variant),
        },
        ExprKind::Handle { body, handlers } => ExprKind::Handle {
            body: substitute_expr(db, body, type_args, variant),
            handlers: handlers
                .into_iter()
                .map(|h| substitute_handler_arm(db, h, type_args, variant))
                .collect(),
        },
        ExprKind::Resume { arg, local_id } => ExprKind::Resume {
            arg: substitute_expr(db, arg, type_args, variant),
            local_id,
        },
        ExprKind::Cons { ctor, args } => ExprKind::Cons {
            ctor: subst_typed_ref(db, ctor, type_args),
            args: args
                .into_iter()
                .map(|a| substitute_expr(db, a, type_args, variant))
                .collect(),
        },
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => ExprKind::Record {
            type_name: subst_typed_ref(db, type_name, type_args),
            fields: fields
                .into_iter()
                .map(|(name, e)| (name, substitute_expr(db, e, type_args, variant)))
                .collect(),
            spread: spread.map(|s| substitute_expr(db, s, type_args, variant)),
        },
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => ExprKind::MethodCall {
            receiver: substitute_expr(db, receiver, type_args, variant),
            method,
            args: args
                .into_iter()
                .map(|a| substitute_expr(db, a, type_args, variant))
                .collect(),
        },
        ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
            op,
            lhs: substitute_expr(db, lhs, type_args, variant),
            rhs: substitute_expr(db, rhs, type_args, variant),
        },
        ExprKind::Tuple(es) => ExprKind::Tuple(
            es.into_iter()
                .map(|e| substitute_expr(db, e, type_args, variant))
                .collect(),
        ),
        ExprKind::List(es) => ExprKind::List(
            es.into_iter()
                .map(|e| substitute_expr(db, e, type_args, variant))
                .collect(),
        ),
        // Leaf nodes — no types to substitute
        ExprKind::NatLit(v) => ExprKind::NatLit(v),
        ExprKind::IntLit(v) => ExprKind::IntLit(v),
        ExprKind::FloatLit(v) => ExprKind::FloatLit(v),
        ExprKind::StringLit(v) => ExprKind::StringLit(v),
        ExprKind::BytesLit(v) => ExprKind::BytesLit(v),
        ExprKind::BoolLit(v) => ExprKind::BoolLit(v),
        ExprKind::RuneLit(v) => ExprKind::RuneLit(v),
        ExprKind::Nil => ExprKind::Nil,
        ExprKind::Error => ExprKind::Error,
    };
    Expr::new(expr.id.with_variant(variant), kind)
}

fn substitute_stmt<'db>(
    db: &'db dyn salsa::Database,
    stmt: Stmt<TypedRef<'db>>,
    type_args: &[Type<'db>],
    variant: NonZero<u64>,
) -> Stmt<TypedRef<'db>> {
    match stmt {
        Stmt::Let {
            id,
            pattern,
            ty,
            value,
        } => Stmt::Let {
            id: id.with_variant(variant),
            pattern: substitute_pattern(db, pattern, type_args, variant),
            ty,
            value: substitute_expr(db, value, type_args, variant),
        },
        Stmt::Expr { id, expr } => Stmt::Expr {
            id: id.with_variant(variant),
            expr: substitute_expr(db, expr, type_args, variant),
        },
    }
}

fn substitute_arm<'db>(
    db: &'db dyn salsa::Database,
    arm: Arm<TypedRef<'db>>,
    type_args: &[Type<'db>],
    variant: NonZero<u64>,
) -> Arm<TypedRef<'db>> {
    Arm {
        id: arm.id.with_variant(variant),
        pattern: substitute_pattern(db, arm.pattern, type_args, variant),
        guard: arm
            .guard
            .map(|g| substitute_expr(db, g, type_args, variant)),
        body: substitute_expr(db, arm.body, type_args, variant),
    }
}

fn substitute_handler_arm<'db>(
    db: &'db dyn salsa::Database,
    arm: HandlerArm<TypedRef<'db>>,
    type_args: &[Type<'db>],
    variant: NonZero<u64>,
) -> HandlerArm<TypedRef<'db>> {
    let kind = match arm.kind {
        HandlerKind::Do { binding } => HandlerKind::Do {
            binding: substitute_pattern(db, binding, type_args, variant),
        },
        HandlerKind::Fn {
            ability,
            op,
            params,
        } => HandlerKind::Fn {
            ability: subst_typed_ref(db, ability, type_args),
            op,
            params: params
                .into_iter()
                .map(|p| substitute_pattern(db, p, type_args, variant))
                .collect(),
        },
        HandlerKind::Op {
            ability,
            op,
            params,
            resume_local_id,
        } => HandlerKind::Op {
            ability: subst_typed_ref(db, ability, type_args),
            op,
            params: params
                .into_iter()
                .map(|p| substitute_pattern(db, p, type_args, variant))
                .collect(),
            resume_local_id,
        },
    };
    HandlerArm {
        id: arm.id.with_variant(variant),
        kind,
        body: substitute_expr(db, arm.body, type_args, variant),
    }
}

fn substitute_pattern<'db>(
    db: &'db dyn salsa::Database,
    pattern: Pattern<TypedRef<'db>>,
    type_args: &[Type<'db>],
    variant: NonZero<u64>,
) -> Pattern<TypedRef<'db>> {
    let kind = match *pattern.kind {
        PatternKind::Variant { ctor, fields } => PatternKind::Variant {
            ctor: subst_typed_ref(db, ctor, type_args),
            fields: fields
                .into_iter()
                .map(|f| substitute_pattern(db, f, type_args, variant))
                .collect(),
        },
        PatternKind::Record {
            type_name,
            fields,
            rest,
        } => PatternKind::Record {
            type_name: type_name.map(|tn| subst_typed_ref(db, tn, type_args)),
            fields: fields
                .into_iter()
                .map(|f| FieldPattern {
                    id: f.id.with_variant(variant),
                    name: f.name,
                    pattern: f
                        .pattern
                        .map(|p| substitute_pattern(db, p, type_args, variant)),
                })
                .collect(),
            rest,
        },
        PatternKind::Tuple(ps) => PatternKind::Tuple(
            ps.into_iter()
                .map(|p| substitute_pattern(db, p, type_args, variant))
                .collect(),
        ),
        PatternKind::List(ps) => PatternKind::List(
            ps.into_iter()
                .map(|p| substitute_pattern(db, p, type_args, variant))
                .collect(),
        ),
        PatternKind::ListRest {
            head,
            rest,
            rest_local_id,
        } => PatternKind::ListRest {
            head: head
                .into_iter()
                .map(|p| substitute_pattern(db, p, type_args, variant))
                .collect(),
            rest,
            rest_local_id,
        },
        PatternKind::As {
            pattern: inner,
            name,
            local_id,
        } => PatternKind::As {
            pattern: substitute_pattern(db, inner, type_args, variant),
            name,
            local_id,
        },
        // Leaf patterns — no types to substitute
        PatternKind::Wildcard => PatternKind::Wildcard,
        PatternKind::Bind { name, local_id } => PatternKind::Bind { name, local_id },
        PatternKind::Literal(lit) => PatternKind::Literal(lit),
        PatternKind::Error => PatternKind::Error,
    };
    Pattern::new(pattern.id.with_variant(variant), kind)
}

#[cfg(test)]
mod tests {
    use crate::ast::{EffectRow, NodeId, ResolvedRef, TypeKind, TypeParam};

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
    fn test_subst_type_replaces_bound_var() {
        let db = TestDb::default();
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(&db, TypeKind::Int);
        assert_eq!(subst_type(&db, bv0, &[int]), int);
    }

    #[test]
    fn test_subst_type_leaves_concrete_unchanged() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        assert_eq!(subst_type(&db, int, &[]), int);
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
        let expr = Expr::new(node_id(1), ExprKind::Var(tr));
        let variant = type_args_variant(&[int]);
        let result = substitute_expr(&db, expr, &[int], variant);

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

        let mangled = mangle_name(&db, Symbol::new("identity"), &[int]);
        let specialized = specialize_func_decl(&db, &func, &[int], mangled);

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
        let func_id = FuncDefId::new(&db, func_name);

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
            name: func_name,
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
        let scheme = TypeScheme::new(&db, vec![TypeParam::anonymous()], scheme_body);
        let function_types = vec![(func_name, scheme)];

        let mut type_arg_sets = HashSet::new();
        type_arg_sets.insert(vec![int]);
        type_arg_sets.insert(vec![float]);
        let mut instantiations = HashMap::new();
        instantiations.insert(func_id, type_arg_sets);

        let (new_decls, new_fn_types) =
            generate_specializations(&db, &module, &instantiations, &function_types);

        assert_eq!(new_decls.len(), 2);
        assert_eq!(new_fn_types.len(), 2);

        // Verify names are mangled
        let names: HashSet<String> = new_decls.iter().map(|d| d.name.to_string()).collect();
        assert!(names.contains("identity$Int"));
        assert!(names.contains("identity$Float"));

        // All specialized TypeSchemes must be monomorphic
        for (_, scheme) in &new_fn_types {
            assert!(
                scheme.is_mono(&db),
                "specialized scheme should have no type params"
            );
        }
    }

    // ========================================================================
    // type_to_annotation tests
    // ========================================================================

    #[test]
    fn test_type_to_annotation_primitives() {
        let db = TestDb::default();
        let id = node_id(1);
        let cases = [
            (TypeKind::Int, "Int"),
            (TypeKind::Nat, "Nat"),
            (TypeKind::Float, "Float"),
            (TypeKind::Bool, "Bool"),
            (TypeKind::Nil, "Nil"),
        ];
        for (kind, expected_name) in cases {
            let ty = Type::new(&db, kind);
            let ann = type_to_annotation(&db, ty, id);
            match &ann.kind {
                TypeAnnotationKind::Named(name) => {
                    assert_eq!(name.to_string(), expected_name);
                }
                _ => panic!("expected Named annotation for {:?}", expected_name),
            }
        }
    }

    #[test]
    fn test_type_to_annotation_named_no_args() {
        let db = TestDb::default();
        let ty = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Text"),
                args: vec![],
            },
        );
        let ann = type_to_annotation(&db, ty, node_id(1));
        match &ann.kind {
            TypeAnnotationKind::Named(name) => assert_eq!(name.to_string(), "Text"),
            _ => panic!("expected Named"),
        }
    }

    #[test]
    fn test_type_to_annotation_named_with_args_uses_mangled() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let ty = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![int],
            },
        );
        let ann = type_to_annotation(&db, ty, node_id(1));
        match &ann.kind {
            TypeAnnotationKind::Named(name) => {
                assert_eq!(name.to_string(), "Option$Int");
            }
            _ => panic!("expected Named with mangled name"),
        }
    }

    /// Regression: effect row must be preserved when converting Func types
    /// back to annotations (previously the `abilities` field was hardcoded
    /// to `vec![]`, silently turning effectful functions into pure ones).
    #[test]
    fn test_type_to_annotation_func_preserves_abilities() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let console_id = crate::ast::AbilityId::new(&db, Symbol::new("std::console::Console"));
        let state_id = crate::ast::AbilityId::new(&db, Symbol::new("std::state::State"));
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
                    is_pub: false,
                    name: Some(Symbol::new("first")),
                    ty: TypeAnnotation {
                        id: node_id(5),
                        kind: TypeAnnotationKind::Named(Symbol::new("a")),
                    },
                },
                FieldDecl {
                    id: node_id(6),
                    is_pub: false,
                    name: Some(Symbol::new("second")),
                    ty: TypeAnnotation {
                        id: node_id(7),
                        kind: TypeAnnotationKind::Named(Symbol::new("b")),
                    },
                },
            ],
        };

        let mangled = mangle_name(&db, Symbol::new("Pair"), &[int, bool_ty]);
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

        let mangled = mangle_name(&db, Symbol::new("Option"), &[int]);
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
    fn test_substitute_annotation_replaces_param() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let ann = TypeAnnotation {
            id: node_id(1),
            kind: TypeAnnotationKind::Named(Symbol::new("a")),
        };
        let result = substitute_annotation(&db, &ann, &[Symbol::new("a")], &[int]);
        match &result.kind {
            TypeAnnotationKind::Named(name) => assert_eq!(name.to_string(), "Int"),
            other => panic!("expected Named(Int), got {:?}", other),
        }
    }

    #[test]
    fn test_substitute_annotation_leaves_non_param() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let ann = TypeAnnotation {
            id: node_id(1),
            kind: TypeAnnotationKind::Named(Symbol::new("Text")),
        };
        let result = substitute_annotation(&db, &ann, &[Symbol::new("a")], &[int]);
        match &result.kind {
            TypeAnnotationKind::Named(name) => assert_eq!(name.to_string(), "Text"),
            other => panic!("expected Named(Text), got {:?}", other),
        }
    }
}
