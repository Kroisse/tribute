//! Call site and type rewriting for monomorphization.
//!
//! Rewrites references to generic functions with their specialized versions
//! by matching the callee's concrete type against collected instantiations.
//! Also rewrites Named types with type arguments to their mangled monomorphic versions.

use std::collections::{HashMap, HashSet};

use trunk_ir::Symbol;

use crate::ast::{
    Arm, CtorId, Decl, Expr, ExprKind, FieldPattern, FuncDefId, HandlerArm, HandlerKind, Module,
    ModuleDecl, Pattern, PatternKind, ResolvedRef, Stmt, Type, TypeDefId, TypeKind, TypeScheme,
    TypedRef,
};

use super::collect::extract_type_args;
use super::mangle::mangle_name;

/// Rewrite map: original FuncDefId → list of (type_args, mangled_name) pairs.
pub type RewriteMap<'db> = HashMap<FuncDefId<'db>, Vec<(Vec<Type<'db>>, Symbol)>>;

/// Type rewrite map: type name → set of (type_args, mangled_name) pairs.
pub type TypeRewriteMap<'db> = HashMap<Symbol, Vec<(Vec<Type<'db>>, Symbol)>>;

/// Rewrite all generic function call sites in a module to use specialized versions.
pub fn rewrite_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    function_types: &[(Symbol, TypeScheme<'db>)],
    rewrite_map: &RewriteMap<'db>,
) -> Module<TypedRef<'db>> {
    let mut rewriter = make_rewriter(db, function_types, rewrite_map);
    let decls = module
        .decls
        .into_iter()
        .map(|d| rewriter.rewrite_decl(d))
        .collect();
    Module::new(module.id, module.name, decls)
}

/// Rewrite call sites in a list of declarations (e.g., specialized function bodies).
pub fn rewrite_decls<'db>(
    db: &'db dyn salsa::Database,
    decls: Vec<Decl<TypedRef<'db>>>,
    function_types: &[(Symbol, TypeScheme<'db>)],
    rewrite_map: &RewriteMap<'db>,
) -> Vec<Decl<TypedRef<'db>>> {
    let mut rewriter = make_rewriter(db, function_types, rewrite_map);
    decls
        .into_iter()
        .map(|d| rewriter.rewrite_decl(d))
        .collect()
}

fn make_rewriter<'a, 'db>(
    db: &'db dyn salsa::Database,
    function_types: &[(Symbol, TypeScheme<'db>)],
    rewrite_map: &'a RewriteMap<'db>,
) -> CallSiteRewriter<'a, 'db> {
    let scheme_map: HashMap<Symbol, TypeScheme<'db>> = function_types.iter().cloned().collect();
    CallSiteRewriter {
        db,
        scheme_map,
        rewrite_map,
    }
}

struct CallSiteRewriter<'a, 'db> {
    db: &'db dyn salsa::Database,
    scheme_map: HashMap<Symbol, TypeScheme<'db>>,
    rewrite_map: &'a RewriteMap<'db>,
}

impl<'a, 'db> CallSiteRewriter<'a, 'db> {
    fn try_rewrite_ref(&self, typed_ref: &TypedRef<'db>) -> Option<TypedRef<'db>> {
        let ResolvedRef::Function { id } = &typed_ref.resolved else {
            return None;
        };
        let entries = self.rewrite_map.get(id)?;
        let qualified = id.qualified(self.db);
        let scheme = self.scheme_map.get(&qualified)?;
        let type_args = extract_type_args(self.db, *scheme, typed_ref.ty)?;

        // Find the matching mangled name
        let mangled = entries.iter().find_map(|(args, name)| {
            if args == &type_args {
                Some(*name)
            } else {
                None
            }
        })?;

        let specialized_id = FuncDefId::new(self.db, mangled);
        Some(TypedRef::new(
            ResolvedRef::Function { id: specialized_id },
            typed_ref.ty,
        ))
    }

    fn rewrite_typed_ref(&self, tr: TypedRef<'db>) -> TypedRef<'db> {
        self.try_rewrite_ref(&tr).unwrap_or(tr)
    }

    fn rewrite_decl(&mut self, decl: Decl<TypedRef<'db>>) -> Decl<TypedRef<'db>> {
        match decl {
            Decl::Function(mut func) => {
                func.body = self.rewrite_expr(func.body);
                Decl::Function(func)
            }
            Decl::Module(m) => {
                let body = m
                    .body
                    .map(|decls| decls.into_iter().map(|d| self.rewrite_decl(d)).collect());
                Decl::Module(ModuleDecl {
                    id: m.id,
                    name: m.name,
                    is_pub: m.is_pub,
                    body,
                })
            }
            other => other,
        }
    }

    fn rewrite_expr(&mut self, expr: Expr<TypedRef<'db>>) -> Expr<TypedRef<'db>> {
        let kind = match *expr.kind {
            ExprKind::Var(tr) => ExprKind::Var(self.rewrite_typed_ref(tr)),
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: self.rewrite_expr(callee),
                args: args.into_iter().map(|a| self.rewrite_expr(a)).collect(),
            },
            ExprKind::Block { stmts, value } => ExprKind::Block {
                stmts: stmts.into_iter().map(|s| self.rewrite_stmt(s)).collect(),
                value: self.rewrite_expr(value),
            },
            ExprKind::Case { scrutinee, arms } => ExprKind::Case {
                scrutinee: self.rewrite_expr(scrutinee),
                arms: arms.into_iter().map(|a| self.rewrite_arm(a)).collect(),
            },
            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.rewrite_expr(body),
            },
            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.rewrite_expr(body),
                handlers: handlers
                    .into_iter()
                    .map(|h| self.rewrite_handler_arm(h))
                    .collect(),
            },
            ExprKind::Resume { arg, local_id } => ExprKind::Resume {
                arg: self.rewrite_expr(arg),
                local_id,
            },
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor,
                args: args.into_iter().map(|a| self.rewrite_expr(a)).collect(),
            },
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => ExprKind::Record {
                type_name,
                fields: fields
                    .into_iter()
                    .map(|(name, e)| (name, self.rewrite_expr(e)))
                    .collect(),
                spread: spread.map(|s| self.rewrite_expr(s)),
            },
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => ExprKind::MethodCall {
                receiver: self.rewrite_expr(receiver),
                method,
                args: args.into_iter().map(|a| self.rewrite_expr(a)).collect(),
            },
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: self.rewrite_expr(lhs),
                rhs: self.rewrite_expr(rhs),
            },
            ExprKind::Tuple(es) => {
                ExprKind::Tuple(es.into_iter().map(|e| self.rewrite_expr(e)).collect())
            }
            ExprKind::List(es) => {
                ExprKind::List(es.into_iter().map(|e| self.rewrite_expr(e)).collect())
            }
            // Leaf nodes
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
        Expr::new(expr.id, kind)
    }

    fn rewrite_stmt(&mut self, stmt: Stmt<TypedRef<'db>>) -> Stmt<TypedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                ty,
                value,
            } => Stmt::Let {
                id,
                pattern,
                ty,
                value: self.rewrite_expr(value),
            },
            Stmt::Expr { id, expr } => Stmt::Expr {
                id,
                expr: self.rewrite_expr(expr),
            },
        }
    }

    fn rewrite_arm(&mut self, arm: Arm<TypedRef<'db>>) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: arm.pattern,
            guard: arm.guard.map(|g| self.rewrite_expr(g)),
            body: self.rewrite_expr(arm.body),
        }
    }

    fn rewrite_handler_arm(&mut self, arm: HandlerArm<TypedRef<'db>>) -> HandlerArm<TypedRef<'db>> {
        HandlerArm {
            id: arm.id,
            kind: arm.kind,
            body: self.rewrite_expr(arm.body),
        }
    }
}

// ============================================================================
// Type rewriting: Named { name, args } → Named { mangled, args: [] }
// ============================================================================

/// Build a type rewrite map from collected type instantiations.
pub fn build_type_rewrite_map<'db>(
    db: &'db dyn salsa::Database,
    instantiations: &HashMap<Symbol, HashSet<Vec<Type<'db>>>>,
) -> TypeRewriteMap<'db> {
    let mut map = TypeRewriteMap::new();
    for (name, type_arg_sets) in instantiations {
        let mut entries: Vec<(Vec<Type<'db>>, Symbol)> = type_arg_sets
            .iter()
            .map(|type_args| {
                let mangled = mangle_name(db, *name, type_args);
                (type_args.clone(), mangled)
            })
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));
        map.insert(*name, entries);
    }
    map
}

/// Rewrite all Named types with type arguments to their mangled monomorphic versions
/// throughout a module's expressions.
pub fn rewrite_types_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    type_rewrite_map: &TypeRewriteMap<'db>,
) -> Module<TypedRef<'db>> {
    let decls = module
        .decls
        .into_iter()
        .map(|d| rewrite_types_in_decl(db, d, type_rewrite_map))
        .collect();
    Module::new(module.id, module.name, decls)
}

fn rewrite_types_in_decl<'db>(
    db: &'db dyn salsa::Database,
    decl: Decl<TypedRef<'db>>,
    map: &TypeRewriteMap<'db>,
) -> Decl<TypedRef<'db>> {
    match decl {
        Decl::Function(mut func) => {
            func.body = rewrite_types_in_expr(db, func.body, map);
            Decl::Function(func)
        }
        Decl::Module(m) => {
            let body = m.body.map(|decls| {
                decls
                    .into_iter()
                    .map(|d| rewrite_types_in_decl(db, d, map))
                    .collect()
            });
            Decl::Module(ModuleDecl {
                id: m.id,
                name: m.name,
                is_pub: m.is_pub,
                body,
            })
        }
        other => other,
    }
}

/// Rewrite a Type, replacing Named types with non-empty args with their mangled versions.
pub fn rewrite_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    map: &TypeRewriteMap<'db>,
) -> Type<'db> {
    match ty.kind(db) {
        TypeKind::Named { name, args } if !args.is_empty() => {
            // The rewrite map is keyed on the original (un-rewritten) args as
            // collected from the module, so look up using `args` directly —
            // not the recursively rewritten ones.
            if let Some(entries) = map.get(name)
                && let Some((_, mangled)) = entries.iter().find(|(ta, _)| ta == args)
            {
                return Type::new(
                    db,
                    TypeKind::Named {
                        name: *mangled,
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
                    name: *name,
                    args: rewritten_args,
                },
            )
        }
        TypeKind::Func {
            params,
            result,
            effect,
        } => {
            let new_params: Vec<_> = params.iter().map(|p| rewrite_type(db, *p, map)).collect();
            let new_result = rewrite_type(db, *result, map);
            if new_params == *params && new_result == *result {
                return ty;
            }
            Type::new(
                db,
                TypeKind::Func {
                    params: new_params,
                    result: new_result,
                    effect: *effect,
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
            if new_arg == *arg && new_result == *result {
                return ty;
            }
            Type::new(
                db,
                TypeKind::Continuation {
                    arg: new_arg,
                    result: new_result,
                    effect: *effect,
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
        | TypeKind::UniVar { .. }
        | TypeKind::Error => ty,
    }
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
                    variant: *variant,
                }
            } else {
                tr.resolved.clone()
            }
        }
        ResolvedRef::TypeDef { id } => {
            if let Some(mangled) = find_mangled_for_typedef(db, *id, tr.ty, map) {
                ResolvedRef::TypeDef {
                    id: TypeDefId::new(db, mangled),
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
        TypeKind::Named { name, args } if !args.is_empty() => {
            let entries = map.get(name)?;
            // Match against the original args stored in the map.
            let (_, mangled) = entries.iter().find(|(ta, _)| ta == args)?;
            Some(*mangled)
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
        TypeKind::Named { name, args } if !args.is_empty() => {
            let entries = map.get(name)?;
            // Match against the original args stored in the map.
            let (_, mangled) = entries.iter().find(|(ta, _)| ta == args)?;
            Some(*mangled)
        }
        _ => None,
    }
}

fn rewrite_types_in_expr<'db>(
    db: &'db dyn salsa::Database,
    expr: Expr<TypedRef<'db>>,
    map: &TypeRewriteMap<'db>,
) -> Expr<TypedRef<'db>> {
    let kind = match *expr.kind {
        ExprKind::Var(tr) => ExprKind::Var(rewrite_typed_ref_type(db, tr, map)),
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: rewrite_types_in_expr(db, callee, map),
            args: args
                .into_iter()
                .map(|a| rewrite_types_in_expr(db, a, map))
                .collect(),
        },
        ExprKind::Block { stmts, value } => ExprKind::Block {
            stmts: stmts
                .into_iter()
                .map(|s| rewrite_types_in_stmt(db, s, map))
                .collect(),
            value: rewrite_types_in_expr(db, value, map),
        },
        ExprKind::Case { scrutinee, arms } => ExprKind::Case {
            scrutinee: rewrite_types_in_expr(db, scrutinee, map),
            arms: arms
                .into_iter()
                .map(|a| rewrite_types_in_arm(db, a, map))
                .collect(),
        },
        ExprKind::Lambda { params, body } => ExprKind::Lambda {
            params,
            body: rewrite_types_in_expr(db, body, map),
        },
        ExprKind::Handle { body, handlers } => ExprKind::Handle {
            body: rewrite_types_in_expr(db, body, map),
            handlers: handlers
                .into_iter()
                .map(|h| {
                    let kind = match h.kind {
                        HandlerKind::Do { binding } => HandlerKind::Do {
                            binding: rewrite_types_in_pattern(db, binding, map),
                        },
                        HandlerKind::Fn {
                            ability,
                            op,
                            params,
                        } => HandlerKind::Fn {
                            ability: rewrite_typed_ref_type(db, ability, map),
                            op,
                            params: params
                                .into_iter()
                                .map(|p| rewrite_types_in_pattern(db, p, map))
                                .collect(),
                        },
                        HandlerKind::Op {
                            ability,
                            op,
                            params,
                            resume_local_id,
                        } => HandlerKind::Op {
                            ability: rewrite_typed_ref_type(db, ability, map),
                            op,
                            params: params
                                .into_iter()
                                .map(|p| rewrite_types_in_pattern(db, p, map))
                                .collect(),
                            resume_local_id,
                        },
                    };
                    HandlerArm {
                        id: h.id,
                        kind,
                        body: rewrite_types_in_expr(db, h.body, map),
                    }
                })
                .collect(),
        },
        ExprKind::Resume { arg, local_id } => ExprKind::Resume {
            arg: rewrite_types_in_expr(db, arg, map),
            local_id,
        },
        ExprKind::Cons { ctor, args } => ExprKind::Cons {
            ctor: rewrite_typed_ref_type(db, ctor, map),
            args: args
                .into_iter()
                .map(|a| rewrite_types_in_expr(db, a, map))
                .collect(),
        },
        ExprKind::Record {
            type_name,
            fields,
            spread,
        } => ExprKind::Record {
            type_name: rewrite_typed_ref_type(db, type_name, map),
            fields: fields
                .into_iter()
                .map(|(name, e)| (name, rewrite_types_in_expr(db, e, map)))
                .collect(),
            spread: spread.map(|s| rewrite_types_in_expr(db, s, map)),
        },
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => ExprKind::MethodCall {
            receiver: rewrite_types_in_expr(db, receiver, map),
            method,
            args: args
                .into_iter()
                .map(|a| rewrite_types_in_expr(db, a, map))
                .collect(),
        },
        ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
            op,
            lhs: rewrite_types_in_expr(db, lhs, map),
            rhs: rewrite_types_in_expr(db, rhs, map),
        },
        ExprKind::Tuple(es) => ExprKind::Tuple(
            es.into_iter()
                .map(|e| rewrite_types_in_expr(db, e, map))
                .collect(),
        ),
        ExprKind::List(es) => ExprKind::List(
            es.into_iter()
                .map(|e| rewrite_types_in_expr(db, e, map))
                .collect(),
        ),
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
    Expr::new(expr.id, kind)
}

fn rewrite_types_in_stmt<'db>(
    db: &'db dyn salsa::Database,
    stmt: Stmt<TypedRef<'db>>,
    map: &TypeRewriteMap<'db>,
) -> Stmt<TypedRef<'db>> {
    match stmt {
        Stmt::Let {
            id,
            pattern,
            ty,
            value,
        } => Stmt::Let {
            id,
            pattern: rewrite_types_in_pattern(db, pattern, map),
            ty,
            value: rewrite_types_in_expr(db, value, map),
        },
        Stmt::Expr { id, expr } => Stmt::Expr {
            id,
            expr: rewrite_types_in_expr(db, expr, map),
        },
    }
}

fn rewrite_types_in_arm<'db>(
    db: &'db dyn salsa::Database,
    arm: Arm<TypedRef<'db>>,
    map: &TypeRewriteMap<'db>,
) -> Arm<TypedRef<'db>> {
    Arm {
        id: arm.id,
        pattern: rewrite_types_in_pattern(db, arm.pattern, map),
        guard: arm.guard.map(|g| rewrite_types_in_expr(db, g, map)),
        body: rewrite_types_in_expr(db, arm.body, map),
    }
}

fn rewrite_types_in_pattern<'db>(
    db: &'db dyn salsa::Database,
    pattern: Pattern<TypedRef<'db>>,
    map: &TypeRewriteMap<'db>,
) -> Pattern<TypedRef<'db>> {
    let kind = match *pattern.kind {
        PatternKind::Variant { ctor, fields } => PatternKind::Variant {
            ctor: rewrite_typed_ref_type(db, ctor, map),
            fields: fields
                .into_iter()
                .map(|f| rewrite_types_in_pattern(db, f, map))
                .collect(),
        },
        PatternKind::Record {
            type_name,
            fields,
            rest,
        } => PatternKind::Record {
            type_name: type_name.map(|tn| rewrite_typed_ref_type(db, tn, map)),
            fields: fields
                .into_iter()
                .map(|f| FieldPattern {
                    id: f.id,
                    name: f.name,
                    pattern: f.pattern.map(|p| rewrite_types_in_pattern(db, p, map)),
                })
                .collect(),
            rest,
        },
        PatternKind::Tuple(ps) => PatternKind::Tuple(
            ps.into_iter()
                .map(|p| rewrite_types_in_pattern(db, p, map))
                .collect(),
        ),
        PatternKind::List(ps) => PatternKind::List(
            ps.into_iter()
                .map(|p| rewrite_types_in_pattern(db, p, map))
                .collect(),
        ),
        PatternKind::ListRest {
            head,
            rest,
            rest_local_id,
        } => PatternKind::ListRest {
            head: head
                .into_iter()
                .map(|p| rewrite_types_in_pattern(db, p, map))
                .collect(),
            rest,
            rest_local_id,
        },
        PatternKind::As {
            pattern: inner,
            name,
            local_id,
        } => PatternKind::As {
            pattern: rewrite_types_in_pattern(db, inner, map),
            name,
            local_id,
        },
        PatternKind::Wildcard => PatternKind::Wildcard,
        PatternKind::Bind { name, local_id } => PatternKind::Bind { name, local_id },
        PatternKind::Literal(lit) => PatternKind::Literal(lit),
        PatternKind::Error => PatternKind::Error,
    };
    Pattern::new(pattern.id, kind)
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
        _db: &'db dyn salsa::Database,
        entries: Vec<(Symbol, Vec<Type<'db>>, Symbol)>,
    ) -> TypeRewriteMap<'db> {
        let mut map = TypeRewriteMap::new();
        for (name, args, mangled) in entries {
            map.entry(name).or_default().push((args, mangled));
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
            TypeKind::Named { name, args } => {
                assert_eq!(name.to_string(), "Option$Int");
                assert!(args.is_empty());
            }
            other => panic!("expected Named, got {:?}", other),
        }
    }

    #[test]
    fn test_rewrite_type_leaves_primitives() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let map = TypeRewriteMap::new();
        assert_eq!(rewrite_type(&db, int, &map), int);
    }

    #[test]
    fn test_rewrite_type_named_no_args_unchanged() {
        let db = TestDb::default();
        let text = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Text"),
                args: vec![],
            },
        );
        let map = TypeRewriteMap::new();
        assert_eq!(rewrite_type(&db, text, &map), text);
    }

    #[test]
    fn test_rewrite_type_nested_in_func() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
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
            },
        );
        let map = make_type_rewrite_map(
            &db,
            vec![(Symbol::new("Option"), vec![int], Symbol::new("Option$Int"))],
        );

        let result = rewrite_type(&db, func_ty, &map);
        match result.kind(&db) {
            TypeKind::Func { params, .. } => match params[0].kind(&db) {
                TypeKind::Named { name, args } => {
                    assert_eq!(name.to_string(), "Option$Int");
                    assert!(args.is_empty());
                }
                other => panic!("expected Named, got {:?}", other),
            },
            other => panic!("expected Func, got {:?}", other),
        }
    }

    #[test]
    fn test_rewrite_type_not_in_map_preserves_args() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let unknown = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Unknown"),
                args: vec![int],
            },
        );
        let map = TypeRewriteMap::new(); // empty

        let result = rewrite_type(&db, unknown, &map);
        // Should be unchanged
        assert_eq!(result, unknown);
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
                name: Symbol::new("Option"),
                args: vec![int],
            },
        );
        let pair_option_int_bool = Type::new(
            &db,
            TypeKind::Named {
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
            TypeKind::Named { name, args } => {
                assert_eq!(name.to_string(), "Pair$Option$0$Int$1$Bool");
                assert!(args.is_empty());
            }
            other => panic!("expected mangled Named, got {:?}", other),
        }
    }
}
