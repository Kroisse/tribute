use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZero;

use trunk_ir::Symbol;

use crate::ast::{
    Arm, Decl, Expr, ExprKind, FieldPattern, FuncDecl, FuncDefId, HandlerArm, HandlerKind, Module,
    Pattern, PatternKind, Stmt, Type, TypeScheme, TypedRef,
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
                substitute_bound_vars(db, scheme.body(db), type_args).unwrap_or(scheme.body(db)),
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

fn collect_func_decls<'a, 'db>(
    module: &'a Module<TypedRef<'db>>,
) -> HashMap<Symbol, &'a FuncDecl<TypedRef<'db>>> {
    let mut map = HashMap::new();
    collect_func_decls_inner(&module.decls, &mut map);
    map
}

fn collect_func_decls_inner<'a, 'db>(
    decls: &'a [Decl<TypedRef<'db>>],
    map: &mut HashMap<Symbol, &'a FuncDecl<TypedRef<'db>>>,
) {
    for decl in decls {
        match decl {
            Decl::Function(func) => {
                map.insert(func.name, func);
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    collect_func_decls_inner(body, map);
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
    substitute_bound_vars(db, ty, type_args).unwrap_or(ty)
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
}
