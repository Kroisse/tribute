use rustc_hash::FxHashMap as HashMap;

use salsa_test_macros::salsa_test;
use trunk_ir::Symbol;

use crate::ast::{
    AbilityId, AbilityRow, Expr, ExprKind, FuncDefId, HandlerArm, HandlerKind, LocalId, NodeId,
    OpDeclKind, Pattern, PatternKind, ResolvedRef, Type, TypeAnnotation, TypeAnnotationKind,
    TypeDefId, TypeKind,
};
use crate::typeck::context::{AbilityInfo, AbilityOpInfo};
use crate::typeck::func_context::HandleContext;
use crate::typeck::solver::TypeSolver;
use crate::typeck::{FunctionInferenceContext, ModuleTypeEnv};

use super::{Mode, TypeChecker};

/// Helper to create a TypeChecker for testing.
fn make_test_checker(db: &dyn salsa::Database) -> TypeChecker<'_> {
    TypeChecker::new(
        db,
        crate::resolve::NameDisplay::new(&crate::resolve::PackageName::executable_root(), []),
    )
}

/// Helper to create a FunctionInferenceContext for testing.
fn make_test_ctx<'a, 'db>(
    db: &'db dyn salsa::Database,
    env: &'a ModuleTypeEnv<'db>,
) -> FunctionInferenceContext<'a, 'db> {
    let func_id = FuncDefId::new(db, trunk_ir::SymbolPath::new(["test_func"]));
    FunctionInferenceContext::new(db, env, func_id)
}

/// Helper to create a TypeAnnotation with a given kind.
fn make_annotation(kind: TypeAnnotationKind) -> TypeAnnotation {
    static NEXT_ANNOTATION: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
    TypeAnnotation {
        id: NodeId::from_raw(NEXT_ANNOTATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
        kind,
    }
}

fn bind_pattern<'db>(id: usize, name: Symbol, local_id: LocalId) -> Pattern<ResolvedRef<'db>> {
    Pattern::new(
        NodeId::from_raw(id),
        PatternKind::Bind {
            name,
            local_id: Some(local_id),
        },
    )
}

fn local_expr<'db>(id: usize, name: Symbol, local_id: LocalId) -> Expr<ResolvedRef<'db>> {
    Expr::new(
        NodeId::from_raw(id),
        ExprKind::Var(ResolvedRef::local(local_id, name)),
    )
}

// The current CST lowering does not preserve let annotations. Exercise
// both checker visits directly until that separate syntax gap is closed.
#[salsa_test]
fn annotated_let_eliminates_never_without_retyping_its_producer(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let never = Type::new(db, TypeKind::Never);
    let nat = Type::new(db, TypeKind::Nat);
    let source = LocalId::new(0);
    let binding = LocalId::new(1);
    let source_name = Symbol::new("impossible");
    let statement = crate::ast::Stmt::Let {
        id: NodeId::from_raw(1),
        pattern: bind_pattern(2, Symbol::new("value"), binding),
        value: local_expr(3, source_name, source),
        ty: Some(make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "Nat",
        )))),
    };
    for convert in [false, true] {
        let mut ctx = make_test_ctx(db, &checker.env);
        ctx.bind_local(source, never);
        if convert {
            checker.convert_stmt_with_ctx(&mut ctx, &statement);
            assert_eq!(ctx.get_node_type(NodeId::from_raw(3)), Some(never));
        } else {
            checker.infer_stmt_and_bind_with_ctx(&mut ctx, &statement);
        }
        let mut solver = TypeSolver::new(db);
        solver.solve(ctx.take_constraints()).unwrap();
        solver.finalize_relations().unwrap();
        assert_eq!(ctx.lookup_local(binding), Some(nat));
        assert_eq!(ctx.lookup_local(source), Some(never));
    }
}

#[salsa_test]
fn revisited_lambda_checks_a_different_expected_result(db: &salsa::DatabaseImpl) {
    let checker = make_test_checker(db);
    let mut ctx = make_test_ctx(db, &checker.env);
    let lambda = Expr::new(
        NodeId::from_raw(1),
        ExprKind::Lambda {
            params: vec![],
            return_ty: None,
            effects: None,
            body: Expr::new(NodeId::from_raw(2), ExprKind::NatLit(42)),
        },
    );
    let nat = ctx.nat_type();
    let boolean = ctx.bool_type();
    let expected = ctx.func_type(vec![], nat, AbilityRow::pure(db));
    checker.check_expr_with_ctx(&mut ctx, &lambda, Mode::Check(expected));
    checker.check_expr_with_ctx(&mut ctx, &lambda, Mode::Infer);
    let mut solver = TypeSolver::new(db);
    solver.solve(ctx.constraints_snapshot()).unwrap();
    let incompatible = ctx.func_type(vec![], boolean, AbilityRow::pure(db));
    checker.check_expr_with_ctx(&mut ctx, &lambda, Mode::Check(incompatible));
    assert!(solver.solve(ctx.take_constraints()).is_err());
}

#[salsa_test]
fn annotated_lambda_uses_context_on_its_first_visit(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let mut ctx = make_test_ctx(db, &checker.env);
    let source = LocalId::new(0);
    let binding = LocalId::new(1);
    let name = Symbol::new("impossible");
    let never = Type::new(db, TypeKind::Never);
    ctx.bind_local(source, never);
    let statement = crate::ast::Stmt::Let {
        id: NodeId::from_raw(1),
        pattern: bind_pattern(2, Symbol::new("thunk"), binding),
        value: Expr::new(
            NodeId::from_raw(3),
            ExprKind::Lambda {
                params: vec![],
                return_ty: None,
                effects: None,
                body: local_expr(4, name, source),
            },
        ),
        ty: Some(make_annotation(TypeAnnotationKind::Func {
            params: vec![],
            result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Nat",
            )))),
            abilities: vec![],
        })),
    };
    checker.infer_stmt_and_bind_with_ctx(&mut ctx, &statement);
    checker.convert_stmt_with_ctx(&mut ctx, &statement);
    let mut solver = TypeSolver::new(db);
    solver.solve(ctx.take_constraints()).unwrap();
    solver.finalize_relations().unwrap();
    let callable = ctx.lookup_local(binding).unwrap();
    let TypeKind::Func { result, .. } = callable.kind(db) else {
        panic!("expected callable");
    };
    assert!(matches!(result.kind(db), TypeKind::Nat));
    assert_eq!(ctx.get_node_type(NodeId::from_raw(4)), Some(never));
}

#[salsa_test]
fn test_handler_do_binding_does_not_leak(db: &dyn salsa::Database) {
    let mut checker = make_test_checker(db);
    let name = Symbol::new("result");
    let body_ty = Type::new(db, TypeKind::Nat);
    let ability_id = AbilityId::source(db, trunk_ir::SymbolPath::new(["Test"]));
    let get = Symbol::new("get");
    checker.env.register_ability(
        ability_id,
        AbilityInfo {
            id: ability_id,
            type_params: vec![],
            operations: [(
                get.clone(),
                AbilityOpInfo {
                    name: get.clone(),
                    kind: OpDeclKind::Op,
                    param_types: vec![],
                    return_type: Type::new(db, TypeKind::Nat),
                },
            )]
            .into_iter()
            .collect::<HashMap<_, _>>(),
        },
    );
    let mut ctx = make_test_ctx(db, &checker.env);
    let handle_node_id = NodeId::from_raw(0);
    let _ = ctx.begin_result_join(handle_node_id);
    let handle_ctx = HandleContext {
        handled_effects: AbilityRow::pure(db),
        node_id: handle_node_id,
        body_ty,
        body_effect: AbilityRow::pure(db),
        evidence_body: None,
    };

    let do_arm = HandlerArm {
        id: NodeId::from_raw(1),
        kind: HandlerKind::Do {
            binding: bind_pattern(2, name.clone(), LocalId::new(1)),
        },
        body: local_expr(3, name.clone(), LocalId::new(1)),
    };
    let converted_do = checker.convert_handler_arm_with_ctx(&mut ctx, &do_arm, &handle_ctx);
    assert!(matches!(
        &*converted_do.body.kind,
        ExprKind::Var(reference) if reference.ty == body_ty
    ));

    let ability = ResolvedRef::ability(ability_id);
    let op_arm = HandlerArm {
        id: NodeId::from_raw(4),
        kind: HandlerKind::Op {
            ability,
            op: get,
            params: vec![],
            resume_local_id: None,
        },
        body: local_expr(5, name.clone(), LocalId::UNRESOLVED),
    };
    let converted_op = checker.convert_handler_arm_with_ctx(&mut ctx, &op_arm, &handle_ctx);
    assert!(
        matches!(
            &*converted_op.body.kind,
            ExprKind::Var(reference)
                if matches!(reference.ty.kind(db), TypeKind::UniVar { .. })
        ),
        "do-arm binding should not type an unresolved name in a later operation arm"
    );

    let outside = checker.check_expr_with_ctx(
        &mut ctx,
        &local_expr(6, name, LocalId::UNRESOLVED),
        Mode::Infer,
    );
    assert!(
        matches!(
            &*outside.kind,
            ExprKind::Var(reference)
                if matches!(reference.ty.kind(db), TypeKind::UniVar { .. })
        ),
        "do-arm binding should not type an unresolved name outside the handle"
    );
}

#[salsa_test]
fn test_same_name_handler_bindings_are_independent(db: &dyn salsa::Database) {
    let mut checker = make_test_checker(db);
    let ability_id = AbilityId::source(db, trunk_ir::SymbolPath::new(["Choice"]));
    let nat_op = Symbol::new("nat");
    let bool_op = Symbol::new("bool");
    let nat_ty = Type::new(db, TypeKind::Nat);
    let bool_ty = Type::new(db, TypeKind::Bool);
    checker.env.register_ability(
        ability_id,
        AbilityInfo {
            id: ability_id,
            type_params: vec![],
            operations: [
                (
                    nat_op.clone(),
                    AbilityOpInfo {
                        name: nat_op.clone(),
                        kind: OpDeclKind::Op,
                        param_types: vec![nat_ty],
                        return_type: Type::new(db, TypeKind::Nil),
                    },
                ),
                (
                    bool_op.clone(),
                    AbilityOpInfo {
                        name: bool_op.clone(),
                        kind: OpDeclKind::Op,
                        param_types: vec![bool_ty],
                        return_type: Type::new(db, TypeKind::Nil),
                    },
                ),
            ]
            .into_iter()
            .collect::<HashMap<_, _>>(),
        },
    );

    let mut ctx = make_test_ctx(db, &checker.env);
    let handle_node_id = NodeId::from_raw(0);
    let _ = ctx.begin_result_join(handle_node_id);
    let handle_ctx = HandleContext {
        handled_effects: AbilityRow::pure(db),
        node_id: handle_node_id,
        body_ty: Type::new(db, TypeKind::Nil),
        body_effect: AbilityRow::pure(db),
        evidence_body: None,
    };
    let name = Symbol::new("value");
    let ability = ResolvedRef::ability(ability_id);
    let make_arm = |id, op, local_id| HandlerArm {
        id: NodeId::from_raw(id),
        kind: HandlerKind::Op {
            ability: ability.clone(),
            op,
            params: vec![bind_pattern(id + 1, name.clone(), local_id)],
            resume_local_id: None,
        },
        body: local_expr(id + 2, name.clone(), local_id),
    };

    let nat_arm = checker.convert_handler_arm_with_ctx(
        &mut ctx,
        &make_arm(10, nat_op, LocalId::new(10)),
        &handle_ctx,
    );
    let bool_arm = checker.convert_handler_arm_with_ctx(
        &mut ctx,
        &make_arm(20, bool_op, LocalId::new(20)),
        &handle_ctx,
    );

    assert!(matches!(
        &*nat_arm.body.kind,
        ExprKind::Var(reference) if reference.ty == nat_ty
    ));
    assert!(matches!(
        &*bool_arm.body.kind,
        ExprKind::Var(reference) if reference.ty == bool_ty
    ));
    assert!(ctx.lookup_local(LocalId::new(10)).is_none());
    assert!(ctx.lookup_local(LocalId::new(20)).is_none());
    assert!(ctx.lookup_local_by_name(&name).is_none());
}

// =========================================================================
// annotation_to_type_with_ctx tests
// =========================================================================

#[salsa_test]
fn test_annotation_primitive_types(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    let cases = [
        ("Int", TypeKind::Int),
        ("Nat", TypeKind::Nat),
        ("Float", TypeKind::Float),
        ("Bool", TypeKind::Bool),
        ("Bytes", TypeKind::Bytes),
        ("Rune", TypeKind::Rune),
        ("Nil", TypeKind::Nil),
    ];

    for (name, expected_kind) in cases {
        let ann = make_annotation(TypeAnnotationKind::Named(Symbol::new(name)));
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);
        let expected = Type::new(db, expected_kind);
        assert_eq!(ty, expected, "Type annotation '{name}' mismatch");
    }
}

#[salsa_test]
fn test_annotation_named_and_path_types(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    // A qualified type identity preserves the complete path.
    let cases = [
        (TypeAnnotationKind::Named(Symbol::new("MyType")), "MyType"),
        (
            TypeAnnotationKind::Path(vec![Symbol::new("std"), Symbol::new("Option")]),
            "std::Option",
        ),
    ];

    for (kind, expected_name) in cases {
        let ann = make_annotation(kind);
        let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);
        let TypeKind::Named { name, args, .. } = ty.kind(db) else {
            panic!(
                "{expected_name}: annotation should be Named, got {:?}",
                ty.kind(db)
            );
        };
        assert_eq!(*name, Symbol::new(expected_name), "{expected_name}");
        assert!(args.is_empty(), "{expected_name}: unexpected args {args:?}");
    }
}

#[salsa_test]
fn test_annotation_app(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    // List(Int)
    let ann = make_annotation(TypeAnnotationKind::App {
        ctor: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "List",
        )))),
        row_args: vec![],
        args: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "Int",
        )))],
    });
    let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

    // Should be Named { name: "List", args: [Int] }
    if let TypeKind::Named { name, args, .. } = ty.kind(db) {
        assert_eq!(*name, Symbol::new("List"));
        assert_eq!(args.len(), 1);
        assert_eq!(args[0], Type::new(db, TypeKind::Int));
    } else {
        panic!("App type should be Named, got {:?}", ty.kind(db));
    }
}

#[salsa_test]
fn test_annotation_func_simple(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    // fn(Int) -> Bool
    let ann = make_annotation(TypeAnnotationKind::Func {
        params: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "Int",
        )))],
        result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "Bool",
        )))),
        abilities: vec![], // pure
    });
    let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

    if let TypeKind::Func {
        params,
        result,
        effect,
        ..
    } = ty.kind(db)
    {
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], Type::new(db, TypeKind::Int));
        assert_eq!(*result, Type::new(db, TypeKind::Bool));
        // Empty abilities means open ability row (fresh row var)
        assert!(effect.rest(db).is_some() || effect.is_pure(db));
    } else {
        panic!("Func annotation should be Func type");
    }
}

#[salsa_test]
fn test_annotation_func_with_effects(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    // fn(Int) ->{IO} Bool
    let ann = make_annotation(TypeAnnotationKind::Func {
        params: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "Int",
        )))],
        result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "Bool",
        )))),
        abilities: vec![make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "IO",
        )))],
    });
    let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

    if let TypeKind::Func {
        params,
        result,
        effect,
        ..
    } = ty.kind(db)
    {
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], Type::new(db, TypeKind::Int));
        assert_eq!(*result, Type::new(db, TypeKind::Bool));
        // Should have IO effect
        let effects = effect.instances(db);
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].ability_id.name(db), Symbol::new("IO"));
    } else {
        panic!("Func annotation should be Func type");
    }
}

#[salsa_test]
fn test_annotation_tuple(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    // (Int, String)
    let ann = make_annotation(TypeAnnotationKind::Tuple(vec![
        make_annotation(TypeAnnotationKind::Named(Symbol::new("Int"))),
        make_annotation(TypeAnnotationKind::Named(Symbol::new("String"))),
    ]));
    let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

    if let TypeKind::Tuple(elems) = ty.kind(db) {
        assert_eq!(elems.len(), 2);
        assert_eq!(elems[0], Type::new(db, TypeKind::Int));
        assert_eq!(elems[1], Type::new(db, TypeKind::string(db)));
    } else {
        panic!("Tuple annotation should be Tuple type");
    }
}

#[salsa_test]
fn test_annotation_infer(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    // _
    let ann = make_annotation(TypeAnnotationKind::Infer);
    let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);

    // Should be fresh UniVar
    assert!(
        matches!(ty.kind(db), TypeKind::UniVar { .. }),
        "Infer annotation should produce UniVar"
    );
}

// =========================================================================
// substitute_bound_vars tests (via TypeChecker methods)
// =========================================================================

// The checker delegates to `subst::substitute_bound_vars`, whose laws are
// properties in `typeck::subst::laws`; this checks the wrapper's panic.

#[salsa_test]
#[should_panic(expected = "BoundVar index out of range")]
fn test_substitute_out_of_bounds_panics(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);

    // BoundVar(5) + [Int] → should panic (out of bounds)
    let bound_var = Type::new(db, TypeKind::BoundVar { index: 5 });
    let int_ty = Type::new(db, TypeKind::Int);
    let args = vec![int_ty];

    // This should panic
    checker.substitute_bound_vars(bound_var, &args);
}

// =========================================================================
// extract_list_element_type tests
// =========================================================================

#[salsa_test]
fn test_extract_list_element_type_of_list_types(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    let list_of = |args| {
        Type::new(
            db,
            TypeKind::Named {
                id: TypeDefId::builtin_list(db),
                name: Symbol::new("List"),
                row_args: vec![],
                args,
            },
        )
    };
    let int_ty = Type::new(db, TypeKind::Int);
    let string_ty = Type::new(db, TypeKind::string(db));
    let list_int = list_of(vec![int_ty]);

    let cases = [
        ("List<Int>", list_int, int_ty),
        ("List<String>", list_of(vec![string_ty]), string_ty),
        ("List<List<Int>>", list_of(vec![list_int]), list_int),
    ];

    for (name, list_ty, expected) in cases {
        let elem_ty = checker.extract_list_element_type(list_ty, &mut ctx);
        assert_eq!(elem_ty, expected, "{name}");
    }
}

#[salsa_test]
fn test_extract_list_element_type_of_non_list_returns_fresh_var(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);

    let int_ty = Type::new(db, TypeKind::Int);
    let cases = [
        ("Int", int_ty),
        (
            "Option<Int>",
            Type::new(
                db,
                TypeKind::Named {
                    id: TypeDefId::synthetic(db, trunk_ir::SymbolPath::new(["Option"])),
                    name: Symbol::new("Option"),
                    row_args: vec![],
                    args: vec![int_ty],
                },
            ),
        ),
        (
            "List with no type args",
            Type::new(
                db,
                TypeKind::Named {
                    id: TypeDefId::builtin_list(db),
                    name: Symbol::new("List"),
                    row_args: vec![],
                    args: vec![],
                },
            ),
        ),
    ];

    for (name, ty) in cases {
        let elem_ty = checker.extract_list_element_type(ty, &mut ctx);
        assert!(
            matches!(elem_ty.kind(db), TypeKind::UniVar { .. }),
            "{name}: expected a fresh UniVar, got {:?}",
            elem_ty.kind(db)
        );
    }
}

#[salsa_test]
fn local_annotation_unions_preserve_names_and_revisit_identity(db: &dyn salsa::Database) {
    let checker = make_test_checker(db);
    let env = ModuleTypeEnv::new(db);
    let mut ctx = make_test_ctx(db, &env);
    let callback = |names: &[&str]| {
        make_annotation(TypeAnnotationKind::Func {
            params: vec![],
            result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
                "Nil",
            )))),
            abilities: names
                .iter()
                .map(|name| make_annotation(TypeAnnotationKind::Named(Symbol::new(name))))
                .collect(),
        })
    };
    let ann = make_annotation(TypeAnnotationKind::Func {
        params: vec![callback(&["e1"]), callback(&["e2"]), callback(&["e1"])],
        result: Box::new(make_annotation(TypeAnnotationKind::Named(Symbol::new(
            "Nil",
        )))),
        abilities: vec![
            make_annotation(TypeAnnotationKind::Named(Symbol::new("e1"))),
            make_annotation(TypeAnnotationKind::Named(Symbol::new("e2"))),
        ],
    });
    let ty = checker.annotation_to_type_with_ctx(&mut ctx, &ann);
    let allocated = ctx.next_row_var();
    assert_eq!(checker.annotation_to_type_with_ctx(&mut ctx, &ann), ty);
    assert_eq!(ctx.next_row_var(), allocated);
    let TypeKind::Func { params, effect, .. } = ty.kind(db) else {
        panic!("function")
    };
    let row = |index: usize| match params[index].kind(db) {
        TypeKind::Func { effect, .. } => *effect,
        _ => panic!("callback"),
    };
    assert_eq!(row(0), row(2));
    assert_ne!(row(0), row(1));
    assert!(
        matches!(ctx.take_constraints().constraints(), [super::super::super::constraint::Constraint::RowUnion(union, None)]
        if union.sources == vec![row(0), row(1)] && union.result == *effect)
    );
}
