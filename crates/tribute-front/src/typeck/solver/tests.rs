use super::*;
use crate::ast::{AbilityId, Effect, EffectRow, TypeDefId, UniVarSource};
use trunk_ir::Symbol;

fn test_db() -> salsa::DatabaseImpl {
    salsa::DatabaseImpl::new()
}

/// Create an AbilityId for testing (with empty module path).
fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
    AbilityId::source(db, Symbol::from_dynamic(name))
}

/// Create a fresh type variable for testing.
fn fresh_var(db: &dyn salsa::Database, n: u64) -> Type<'_> {
    let source = UniVarSource::Anonymous(n);
    let id = UniVarId::new(db, source, 0);
    Type::new(db, TypeKind::UniVar { id })
}

#[test]
fn fresh_row_var_avoids_rows_already_present_in_constraints() {
    let db = test_db();
    let existing = EffectVar { id: 1000 };
    let existing_row = EffectRow::open(&db, existing);
    let mut constraints = ConstraintSet::new();
    constraints.add_row_eq(existing_row, existing_row);
    let mut solver = TypeSolver::new(&db);

    solver.solve(constraints).expect("constraint should solve");

    assert_eq!(solver.fresh_row_var(), EffectVar { id: 1001 });
}

#[test]
fn test_unify_same_type() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);
    solver.unify_types(int_ty, int_ty).unwrap();
}

#[test]
fn test_unify_type_var() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    let int_ty = Type::new(&db, TypeKind::Int);

    solver.unify_types(var_ty, int_ty).unwrap();

    // Check that the substitution was recorded
    let result = solver.type_subst.apply(&db, var_ty);
    assert_eq!(result, int_ty);
}

#[test]
fn test_unify_type_mismatch() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);

    let result = solver.unify_types(int_ty, bool_ty);
    assert!(result.is_err());
}

#[test]
fn test_occurs_check() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    // Try to unify x with List(x) - should fail occurs check
    let list_ty = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: trunk_ir::Symbol::new("List"),
            args: vec![var_ty],
        },
    );

    let result = solver.unify_types(var_ty, list_ty);
    assert!(matches!(result, Err(SolveError::OccursCheck { .. })));
}

#[test]
fn test_occurs_check_in_effect_row() {
    // Unifying ?a with fn() ->{State(?a)} Int should fail the occurs check,
    // because ?a appears inside the effect row's type arguments.
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    let int_ty = Type::new(&db, TypeKind::Int);

    // Effect row: {State(?a)}
    let effect = EffectRow::new(
        &db,
        vec![Effect {
            ability_id: test_ability_id(&db, "State"),
            args: vec![var_ty],
        }],
        None,
    );

    // fn() ->{State(?a)} Int
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let result = solver.unify_types(var_ty, func_ty);
    assert!(
        matches!(result, Err(SolveError::OccursCheck { .. })),
        "Expected occurs check failure for ?a = fn() ->{{State(?a)}} Int"
    );
}

#[test]
fn test_occurs_check_applies_effect_row_substitution() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    let TypeKind::UniVar { id: var_id } = var_ty.kind(&db) else {
        unreachable!("fresh_var must create a UniVar")
    };
    let row_var = EffectVar { id: 7 };
    let substituted = EffectRow::new(
        &db,
        vec![Effect {
            ability_id: test_ability_id(&db, "State"),
            args: vec![var_ty],
        }],
        None,
    );
    solver.row_subst.insert(row_var.id, substituted);

    let int_ty = Type::new(&db, TypeKind::Int);
    let effect = EffectRow::open(&db, row_var);
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    let continuation_ty = Type::new(
        &db,
        TypeKind::Continuation {
            arg: int_ty,
            result: int_ty,
            effect,
        },
    );

    assert!(solver.occurs_in(*var_id, func_ty));
    assert!(solver.occurs_in(*var_id, continuation_ty));
}

#[test]
fn test_occurs_check_not_triggered_for_different_var_in_effect() {
    // Unifying ?a with fn() ->{State(?b)} Int should succeed,
    // because ?a does not appear in the effect row.
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_a = fresh_var(&db, 0);
    let var_b = fresh_var(&db, 1);
    let int_ty = Type::new(&db, TypeKind::Int);

    let effect = EffectRow::new(
        &db,
        vec![Effect {
            ability_id: test_ability_id(&db, "State"),
            args: vec![var_b],
        }],
        None,
    );

    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let result = solver.unify_types(var_a, func_ty);
    assert!(
        result.is_ok(),
        "Should not trigger occurs check when the var is different"
    );
}

#[test]
fn test_unify_tuple() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var1 = fresh_var(&db, 0);
    let var2 = fresh_var(&db, 1);
    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);

    let tuple1 = Type::new(&db, TypeKind::Tuple(vec![var1, var2]));
    let tuple2 = Type::new(&db, TypeKind::Tuple(vec![int_ty, bool_ty]));

    solver.unify_types(tuple1, tuple2).unwrap();

    assert_eq!(solver.type_subst.apply(&db, var1), int_ty);
    assert_eq!(solver.type_subst.apply(&db, var2), bool_ty);
}

// =========================================================================
// Row unification tests (adapted from tribute-passes)
// =========================================================================

#[test]
fn test_empty_row_unification() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let r1 = EffectRow::new(&db, vec![], None);
    let r2 = EffectRow::new(&db, vec![], None);

    let result = solver.unify_rows(r1, r2);
    assert!(result.is_ok());
}

#[test]
fn test_row_var_unification() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    // Create an open row with a variable
    let row_var = EffectVar { id: 42 };
    let r1 = EffectRow::new(&db, vec![], Some(row_var));
    let r2 = EffectRow::new(&db, vec![], None); // empty/pure row

    let result = solver.unify_rows(r1, r2);
    assert!(result.is_ok());

    // The row variable should now be bound to the empty row
    let resolved = solver.row_subst.get(row_var.id);
    assert!(resolved.is_some());
    assert!(resolved.unwrap().is_pure(&db));
}

#[test]
fn test_function_effect_unification() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    // Create two function types with the same (empty) effect row
    let empty_effect = EffectRow::new(&db, vec![], None);
    let int_ty = Type::new(&db, TypeKind::Int);

    let func1 = Type::new(
        &db,
        TypeKind::Func {
            params: vec![int_ty],
            result: int_ty,
            effect: empty_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    let func2 = Type::new(
        &db,
        TypeKind::Func {
            params: vec![int_ty],
            result: int_ty,
            effect: empty_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let result = solver.unify_types(func1, func2);
    assert!(result.is_ok(), "Same function types should unify");
}

#[test]
fn test_function_effect_unification_with_row_var() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);

    // Create a function with empty effect (pure)
    let empty_effect = EffectRow::new(&db, vec![], None);
    let func_pure = Type::new(
        &db,
        TypeKind::Func {
            params: vec![int_ty],
            result: int_ty,
            effect: empty_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    // Create a function with a row variable effect (polymorphic)
    let row_var = EffectVar { id: 99 };
    let poly_effect = EffectRow::new(&db, vec![], Some(row_var));
    let func_poly = Type::new(
        &db,
        TypeKind::Func {
            params: vec![int_ty],
            result: int_ty,
            effect: poly_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    // Unifying should bind the row variable to empty
    let result = solver.unify_types(func_pure, func_poly);
    assert!(
        result.is_ok(),
        "Pure function should unify with polymorphic function"
    );

    // Check that the row variable was bound to empty
    let resolved = solver.row_subst.get(row_var.id);
    assert!(resolved.is_some(), "Row variable should be bound");
    assert!(
        resolved.unwrap().is_pure(&db),
        "Row variable should be bound to empty"
    );
}

#[test]
fn test_pure_callee_in_effectful_context() {
    // Test that calling a pure function from an effectful context succeeds
    // without modifying the caller's effect row.
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    // Create a pure effect row (callee's effect)
    let pure_effect = EffectRow::new(&db, vec![], None);

    // Create an effectful row with State effect (caller's context)
    let row_var = EffectVar { id: 100 };
    let state_effect = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![Type::new(&db, TypeKind::Int)],
    };
    let effectful_row = EffectRow::new(&db, vec![state_effect], Some(row_var));

    // Unifying pure with effectful should succeed
    let result = solver.unify_rows(pure_effect, effectful_row);
    assert!(
        result.is_ok(),
        "Pure callee should be callable from effectful context"
    );

    // The row variable should NOT be bound - caller's effect stays unchanged
    let resolved = solver.row_subst.get(row_var.id);
    assert!(
        resolved.is_none(),
        "Caller's row variable should not be modified when calling pure function"
    );
}

#[test]
fn test_unify_named_types_with_args() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    let int_ty = Type::new(&db, TypeKind::Int);

    // List(var) and List(Int)
    let list_var = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: trunk_ir::Symbol::new("List"),
            args: vec![var_ty],
        },
    );
    let list_int = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: trunk_ir::Symbol::new("List"),
            args: vec![int_ty],
        },
    );

    solver.unify_types(list_var, list_int).unwrap();

    // var should be bound to Int
    assert_eq!(solver.type_subst.apply(&db, var_ty), int_ty);
}

#[test]
fn test_unify_named_types_mismatch() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);

    // List(Int) and Option(Int) should not unify
    let list_int = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: trunk_ir::Symbol::new("List"),
            args: vec![int_ty],
        },
    );
    let option_int = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option")),
            name: trunk_ir::Symbol::new("Option"),
            args: vec![int_ty],
        },
    );

    let result = solver.unify_types(list_int, option_int);
    assert!(matches!(result, Err(SolveError::TypeMismatch { .. })));
}

#[test]
fn test_unify_named_types_rejects_builtin_and_source_with_same_spelling() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);
    let name = Symbol::new("List");
    let int_ty = Type::new(&db, TypeKind::Int);
    let builtin = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name,
            args: vec![int_ty],
        },
    );
    let source = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::source(&db, name, crate::ast::NodeId::from_raw(1)),
            name,
            args: vec![int_ty],
        },
    );

    assert!(matches!(
        solver.unify_types(builtin, source),
        Err(SolveError::TypeMismatch { .. })
    ));
}

#[test]
fn test_unify_named_types_rejects_same_spelled_source_declarations() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);
    let name = Symbol::new("Thing");
    let int_ty = Type::new(&db, TypeKind::Int);
    let first = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::source(
                &db,
                Symbol::new("A::Thing"),
                crate::ast::NodeId::from_raw(1),
            ),
            name,
            args: vec![int_ty],
        },
    );
    let second = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::source(
                &db,
                Symbol::new("B::Thing"),
                crate::ast::NodeId::from_raw(2),
            ),
            name,
            args: vec![int_ty],
        },
    );

    assert!(matches!(
        solver.unify_types(first, second),
        Err(SolveError::TypeMismatch { .. })
    ));
}

#[test]
fn test_unify_app_types() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    let int_ty = Type::new(&db, TypeKind::Int);
    let ctor_ty = fresh_var(&db, 1);
    let list_ctor = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: trunk_ir::Symbol::new("List"),
            args: vec![],
        },
    );

    // App(ctor, [var]) and App(List, [Int])
    let app1 = Type::new(
        &db,
        TypeKind::App {
            ctor: ctor_ty,
            args: vec![var_ty],
        },
    );
    let app2 = Type::new(
        &db,
        TypeKind::App {
            ctor: list_ctor,
            args: vec![int_ty],
        },
    );

    solver.unify_types(app1, app2).unwrap();

    assert_eq!(solver.type_subst.apply(&db, var_ty), int_ty);
    assert_eq!(solver.type_subst.apply(&db, ctor_ty), list_ctor);
}

#[test]
fn test_error_type_unifies_with_anything() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let error_ty = Type::new(&db, TypeKind::Error);
    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);

    // Error should unify with any type
    assert!(solver.unify_types(error_ty, int_ty).is_ok());
    assert!(solver.unify_types(bool_ty, error_ty).is_ok());
}

#[test]
fn test_never_equality_is_strict() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let never_ty = Type::new(&db, TypeKind::Never);
    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);
    let nat_ty = Type::new(&db, TypeKind::Nat);

    assert!(solver.unify_types(never_ty, never_ty).is_ok());
    assert!(solver.unify_types(never_ty, int_ty).is_err());
    assert!(solver.unify_types(bool_ty, never_ty).is_err());
    assert!(solver.unify_types(never_ty, nat_ty).is_err());

    // UniVar should unify with Never, then resolve to Never
    let var = fresh_var(&db, 0);
    assert!(solver.unify_types(var, never_ty).is_ok());
    assert_eq!(solver.type_subst.apply(&db, var), never_ty);

    // A variable resolved to Never retains that exact identity.
    let var2 = fresh_var(&db, 1);
    assert!(solver.unify_types(var2, never_ty).is_ok());
    assert!(solver.unify_types(var2, int_ty).is_err());
}

#[test]
fn test_never_in_named_type_args() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let never_ty = Type::new(&db, TypeKind::Never);
    let int_ty = Type::new(&db, TypeKind::Int);

    // Expression elimination does not recurse into nominal arguments.
    let list_never = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: Symbol::new("List"),
            args: vec![never_ty],
        },
    );
    let list_int = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: Symbol::new("List"),
            args: vec![int_ty],
        },
    );
    assert!(solver.unify_types(list_never, list_int).is_err());
}

#[test]
fn test_never_coercion_preserves_expected_variable() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);
    let never = Type::new(&db, TypeKind::Never);
    let expected = fresh_var(&db, 0);
    let origin = ConstraintOrigin {
        node_id: crate::ast::NodeId::from_raw(0),
        kind: super::super::constraint::ConstraintOriginKind::Expression,
    };
    let mut constraints = ConstraintSet::new();
    constraints.add_type_coerce(never, expected, origin);
    solver.solve(constraints).unwrap();
    solver.finalize_relations().unwrap();
    assert_eq!(solver.type_subst.apply(&db, expected), expected);
    solver
        .unify_types(expected, Type::new(&db, TypeKind::Nat))
        .unwrap();
}

#[test]
fn deferred_producers_with_equal_results_retain_distinct_dependencies() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);
    let first = fresh_var(&db, 0);
    let second = fresh_var(&db, 1);
    let result = Type::new(&db, TypeKind::Bool);
    let first_node = NodeId::from_raw(1);
    let second_node = NodeId::from_raw(2);
    solver.defer_producer(first_node, result, vec![first]);
    solver.defer_producer(second_node, result, vec![second]);
    assert_eq!(solver.pending_variables().0.len(), 2);
    solver.resolve_producer(first_node);
    let TypeKind::UniVar { id } = second.kind(&db) else {
        unreachable!();
    };
    assert_eq!(solver.pending_variables().0, vec![*id]);
}

#[test]
fn deferred_join_keeps_actual_producer_and_reports_a_late_mismatch_once() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);
    let actual = fresh_var(&db, 0);
    let result = fresh_var(&db, 1);
    let nat = Type::new(&db, TypeKind::Nat);
    let origin = ConstraintOrigin {
        node_id: NodeId::from_raw(1),
        kind: super::super::constraint::ConstraintOriginKind::Expression,
    };
    solver.defer_producer(origin.node_id, actual, vec![]);
    let mut constraints = ConstraintSet::new();
    constraints.add(Constraint::TypeJoin {
        sources: vec![(actual, origin), (nat, origin)],
        result,
        origin,
        complete: true,
    });
    constraints.add_type_coerce(result, nat, origin);
    solver.solve(constraints).unwrap();
    solver.finalize_relations().unwrap();
    assert_eq!(solver.type_subst.apply(&db, actual), actual);
    assert_eq!(solver.type_subst.apply(&db, result), nat);
    solver.resolve_producer(origin.node_id);
    let mut late = ConstraintSet::new();
    late.add_type_eq(actual, Type::new(&db, TypeKind::Bool));
    let failure = solver.solve_with_origin(late).unwrap_err();
    assert_eq!(failure.origin, Some(origin));
    assert!(matches!(failure.error, SolveError::TypeMismatch { .. }));
    solver.finalize_relations().unwrap();
    solver.solve(ConstraintSet::new()).unwrap();
}

#[test]
fn test_transitive_unification() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var1 = fresh_var(&db, 0);
    let var2 = fresh_var(&db, 1);
    let int_ty = Type::new(&db, TypeKind::Int);

    // var1 = var2, var2 = Int => var1 = Int
    solver.unify_types(var1, var2).unwrap();
    solver.unify_types(var2, int_ty).unwrap();

    assert_eq!(solver.type_subst.apply(&db, var1), int_ty);
    assert_eq!(solver.type_subst.apply(&db, var2), int_ty);
}

#[test]
fn test_row_subst_apply() {
    let db = test_db();
    let mut row_subst = RowSubst::new();

    // Create a row variable and bind it to an empty row
    let row_var = EffectVar { id: 10 };
    let empty_row = EffectRow::new(&db, vec![], None);
    row_subst.insert(row_var.id, empty_row);

    // Apply substitution to an open row
    let open_row = EffectRow::new(&db, vec![], Some(row_var));
    let result = row_subst.apply(&db, open_row);

    assert!(result.is_pure(&db));
}

#[test]
fn test_type_subst_apply_with_rows() {
    let db = test_db();
    let mut type_subst = TypeSubst::new();
    let mut row_subst = RowSubst::new();

    let int_ty = Type::new(&db, TypeKind::Int);
    let var_ty = fresh_var(&db, 0);
    let var_id = match var_ty.kind(&db) {
        TypeKind::UniVar { id } => *id,
        _ => unreachable!(),
    };
    type_subst.insert(var_id, int_ty);

    // Create a function type with a row variable
    let row_var = EffectVar { id: 20 };
    let empty_row = EffectRow::new(&db, vec![], None);
    row_subst.insert(row_var.id, empty_row);

    let poly_effect = EffectRow::new(&db, vec![], Some(row_var));
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![var_ty],
            result: var_ty,
            effect: poly_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    // Apply both substitutions
    let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

    // Check params and result are substituted
    if let TypeKind::Func {
        params,
        result,
        effect,
        ..
    } = result.kind(&db)
    {
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], int_ty);
        assert_eq!(*result, int_ty);
        assert!(effect.is_pure(&db));
    } else {
        panic!("Expected Func type");
    }
}

#[test]
fn test_type_subst_applies_to_effect_args() {
    // State(?a) where ?a = Int should become State(Int)
    let db = test_db();
    let mut type_subst = TypeSubst::new();

    let int_ty = Type::new(&db, TypeKind::Int);
    let var_ty = fresh_var(&db, 0);
    let var_id = match var_ty.kind(&db) {
        TypeKind::UniVar { id } => *id,
        _ => unreachable!(),
    };
    type_subst.insert(var_id, int_ty);

    // fn() ->{State(?a)} Int
    let effect = EffectRow::new(
        &db,
        vec![Effect {
            ability_id: test_ability_id(&db, "State"),
            args: vec![var_ty],
        }],
        None,
    );
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let row_subst = RowSubst::new();
    let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

    if let TypeKind::Func { effect, .. } = result.kind(&db) {
        let effects = effect.effects(&db);
        assert_eq!(effects.len(), 1);
        assert_eq!(
            effects[0].ability_id.name(&db),
            trunk_ir::Symbol::new("State")
        );
        assert_eq!(effects[0].args.len(), 1);
        assert_eq!(
            effects[0].args[0], int_ty,
            "Effect arg ?a should be substituted to Int"
        );
    } else {
        panic!("Expected Func type");
    }
}

#[test]
fn test_type_subst_preserves_unchanged_effect_args() {
    // State(Int) with no relevant substitution should remain unchanged
    let db = test_db();
    let type_subst = TypeSubst::new();
    let int_ty = Type::new(&db, TypeKind::Int);

    let effect = EffectRow::new(
        &db,
        vec![Effect {
            ability_id: test_ability_id(&db, "State"),
            args: vec![int_ty],
        }],
        None,
    );
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let row_subst = RowSubst::new();
    let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

    if let TypeKind::Func {
        effect: result_effect,
        ..
    } = result.kind(&db)
    {
        let effects = result_effect.effects(&db);
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].args[0], int_ty);
    } else {
        panic!("Expected Func type");
    }
}

#[test]
#[should_panic(expected = "quantified type variable (BoundVar or LocalBoundVar) reached solver")]
fn test_bound_var_panics_in_debug() {
    // BoundVar and LocalBoundVar should never reach the solver — they must be instantiated first.
    // In debug mode, this triggers a debug_assert panic.
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let bound_var = Type::new(&db, TypeKind::BoundVar { index: 0 });
    let int_ty = Type::new(&db, TypeKind::Int);

    let _ = solver.unify_types(bound_var, int_ty);
}

// =========================================================================
// Generalization tests
// =========================================================================

#[test]
fn test_generalize_no_univars() {
    // Concrete type (Int) → no type params, type unchanged
    let db = test_db();
    let subst = TypeSubst::new();
    let row_subst = RowSubst::new();
    let int_ty = Type::new(&db, TypeKind::Int);

    let (generalized, params) = subst.generalize(&db, int_ty, &row_subst);
    assert_eq!(generalized, int_ty);
    assert!(params.is_empty());
}

#[test]
fn test_generalize_single_univar() {
    // fn(?a) -> ?a  →  fn(BoundVar(0)) -> BoundVar(0), 1 type param
    let db = test_db();
    let subst = TypeSubst::new();
    let row_subst = RowSubst::new();

    let var_ty = fresh_var(&db, 0);
    let effect = EffectRow::new(&db, vec![], None);
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![var_ty],
            result: var_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let (generalized, params) = subst.generalize(&db, func_ty, &row_subst);
    assert_eq!(params.len(), 1);

    if let TypeKind::Func {
        params: gen_params,
        result,
        ..
    } = generalized.kind(&db)
    {
        assert!(matches!(
            gen_params[0].kind(&db),
            TypeKind::BoundVar { index: 0 }
        ));
        assert!(matches!(result.kind(&db), TypeKind::BoundVar { index: 0 }));
    } else {
        panic!("Expected Func type");
    }
}

#[test]
fn test_generalize_two_univars() {
    // fn(?a) -> ?b  →  fn(BoundVar(0)) -> BoundVar(1), 2 type params
    let db = test_db();
    let subst = TypeSubst::new();
    let row_subst = RowSubst::new();

    let var_a = fresh_var(&db, 0);
    let var_b = fresh_var(&db, 1);
    let effect = EffectRow::new(&db, vec![], None);
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![var_a],
            result: var_b,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let (generalized, params) = subst.generalize(&db, func_ty, &row_subst);
    assert_eq!(params.len(), 2);

    if let TypeKind::Func {
        params: gen_params,
        result,
        ..
    } = generalized.kind(&db)
    {
        assert!(matches!(
            gen_params[0].kind(&db),
            TypeKind::BoundVar { index: 0 }
        ));
        assert!(matches!(result.kind(&db), TypeKind::BoundVar { index: 1 }));
    } else {
        panic!("Expected Func type");
    }
}

#[test]
fn test_generalize_resolved_univar_not_generalized() {
    // ?a resolved to Int → after apply + generalize: no type params, no BoundVars
    let db = test_db();
    let mut subst = TypeSubst::new();
    let row_subst = RowSubst::new();

    let var_ty = fresh_var(&db, 0);
    let int_ty = Type::new(&db, TypeKind::Int);
    let var_id = match var_ty.kind(&db) {
        TypeKind::UniVar { id } => *id,
        _ => unreachable!(),
    };
    subst.insert(var_id, int_ty);

    let effect = EffectRow::new(&db, vec![], None);
    let func_ty = Type::new(
        &db,
        TypeKind::Func {
            params: vec![var_ty],
            result: var_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    // Apply substitution first (as done in Phase 4)
    let applied = subst.apply_with_rows(&db, func_ty, &row_subst);
    let (generalized, params) = subst.generalize(&db, applied, &row_subst);
    assert!(params.is_empty());

    if let TypeKind::Func {
        params: gen_params,
        result,
        ..
    } = generalized.kind(&db)
    {
        assert_eq!(gen_params[0], int_ty);
        assert_eq!(*result, int_ty);
    } else {
        panic!("Expected Func type");
    }
}

// =========================================================================
// Advanced row unification tests
// =========================================================================

#[test]
fn test_row_unification_with_effects() {
    // {Console} unifies with {Console}
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let console = Effect {
        ability_id: test_ability_id(&db, "Console"),
        args: vec![],
    };
    let r1 = EffectRow::new(&db, vec![console.clone()], None);
    let r2 = EffectRow::new(&db, vec![console], None);

    let result = solver.unify_rows(r1, r2);
    assert!(result.is_ok());
}

#[test]
fn test_row_unification_closed_rows_mismatch() {
    // {Console} does not unify with {IO}
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let console = Effect {
        ability_id: test_ability_id(&db, "Console"),
        args: vec![],
    };
    let io = Effect {
        ability_id: test_ability_id(&db, "IO"),
        args: vec![],
    };
    let r1 = EffectRow::new(&db, vec![console], None);
    let r2 = EffectRow::new(&db, vec![io], None);

    let result = solver.unify_rows(r1, r2);
    assert!(matches!(result, Err(SolveError::RowMismatch { .. })));
}

#[test]
fn test_row_unification_open_row_binds_to_difference() {
    // {Console | e} unifies with {Console, IO}
    // Should bind e to {IO}
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let console = Effect {
        ability_id: test_ability_id(&db, "Console"),
        args: vec![],
    };
    let io = Effect {
        ability_id: test_ability_id(&db, "IO"),
        args: vec![],
    };
    let row_var = EffectVar { id: 50 };

    let r1 = EffectRow::new(&db, vec![console.clone()], Some(row_var));
    let r2 = EffectRow::new(&db, vec![console, io.clone()], None);

    let result = solver.unify_rows(r1, r2);
    assert!(result.is_ok());

    // e should be bound to {IO}
    let resolved = solver.row_subst.get(row_var.id).unwrap();
    let effects = resolved.effects(&db);
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0].ability_id.name(&db), trunk_ir::Symbol::new("IO"));
    assert!(resolved.rest(&db).is_none()); // Closed
}

#[test]
fn test_row_unification_two_open_rows() {
    // {Console | e1} unifies with {IO | e2}
    // Should create fresh e3:
    //   e1 = {IO | e3}
    //   e2 = {Console | e3}
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let console = Effect {
        ability_id: test_ability_id(&db, "Console"),
        args: vec![],
    };
    let io = Effect {
        ability_id: test_ability_id(&db, "IO"),
        args: vec![],
    };
    let e1 = EffectVar { id: 100 };
    let e2 = EffectVar { id: 200 };

    let r1 = EffectRow::new(&db, vec![console.clone()], Some(e1));
    let r2 = EffectRow::new(&db, vec![io.clone()], Some(e2));

    let result = solver.unify_rows(r1, r2);
    assert!(result.is_ok());

    // e1 should be bound to {IO | e3} for some fresh e3
    let e1_resolved = solver.row_subst.get(e1.id).unwrap();
    let e1_effects = e1_resolved.effects(&db);
    assert_eq!(e1_effects.len(), 1);
    assert_eq!(
        e1_effects[0].ability_id.name(&db),
        trunk_ir::Symbol::new("IO")
    );
    assert!(e1_resolved.rest(&db).is_some()); // Open with e3

    // e2 should be bound to {Console | e3}
    let e2_resolved = solver.row_subst.get(e2.id).unwrap();
    let e2_effects = e2_resolved.effects(&db);
    assert_eq!(e2_effects.len(), 1);
    assert_eq!(
        e2_effects[0].ability_id.name(&db),
        trunk_ir::Symbol::new("Console")
    );
    assert!(e2_resolved.rest(&db).is_some()); // Open with e3

    // Both should have the same fresh variable
    assert_eq!(e1_resolved.rest(&db), e2_resolved.rest(&db));
}

#[test]
fn test_row_unification_unifies_type_args() {
    // {State(?a)} unifies with {State(Int)}
    // Should bind ?a to Int
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    let int_ty = Type::new(&db, TypeKind::Int);

    let state_var = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![var_ty],
    };
    let state_int = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![int_ty],
    };

    let r1 = EffectRow::new(&db, vec![state_var], None);
    let r2 = EffectRow::new(&db, vec![state_int], None);

    let result = solver.unify_rows(r1, r2);
    assert!(result.is_ok());

    // ?a should be bound to Int
    assert_eq!(solver.type_subst.apply(&db, var_ty), int_ty);
}

#[test]
fn test_row_unification_same_var_different_effects_fails() {
    // {Console | e} and {IO | e} with the same e should fail
    // (because the concrete effects don't match)
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let console = Effect {
        ability_id: test_ability_id(&db, "Console"),
        args: vec![],
    };
    let io = Effect {
        ability_id: test_ability_id(&db, "IO"),
        args: vec![],
    };
    let row_var = EffectVar { id: 42 };

    let r1 = EffectRow::new(&db, vec![console], Some(row_var));
    let r2 = EffectRow::new(&db, vec![io], Some(row_var));

    let result = solver.unify_rows(r1, r2);
    assert!(matches!(result, Err(SolveError::RowMismatch { .. })));
}

#[test]
fn test_different_effect_arity_returns_arity_mismatch() {
    // State(Int) and State() have different arity - this is an arity mismatch error
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);

    let state_with_arg = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![int_ty],
    };
    let state_no_arg = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![],
    };

    let r1 = EffectRow::new(&db, vec![state_with_arg], None);
    let r2 = EffectRow::new(&db, vec![state_no_arg], None);

    let result = solver.unify_rows(r1, r2);
    // Same ability name but different arity is an arity mismatch error
    assert!(
        matches!(
            result,
            Err(SolveError::EffectArgArityMismatch {
                effect_name,
                expected: 1,
                found: 0,
            }) if effect_name == trunk_ir::Symbol::new("State")
        ),
        "Expected EffectArgArityMismatch error, got {:?}",
        result
    );
}

#[test]
fn test_different_effect_arg_types_returns_row_mismatch() {
    // State(Int) and State(Bool) are different parameterized abilities
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);

    let state_int = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![int_ty],
    };
    let state_bool = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![bool_ty],
    };

    let r1 = EffectRow::new(&db, vec![state_int], None);
    let r2 = EffectRow::new(&db, vec![state_bool], None);

    let result = solver.unify_rows(r1, r2);
    // State(Int) and State(Bool) are distinct abilities, so this is a row mismatch
    assert!(
        matches!(result, Err(SolveError::RowMismatch { .. })),
        "Expected RowMismatch error, got {:?}",
        result
    );
}

#[test]
fn test_same_effect_args_unifies_successfully() {
    // State(Int) and State(Int) are the same ability - should unify
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);

    let state_int1 = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![int_ty],
    };
    let state_int2 = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![int_ty],
    };

    let r1 = EffectRow::new(&db, vec![state_int1], None);
    let r2 = EffectRow::new(&db, vec![state_int2], None);

    let result = solver.unify_rows(r1, r2);
    assert!(
        result.is_ok(),
        "Same effects should unify, got {:?}",
        result
    );
}

// =========================================================================
// Parameterized ability unification with type variables
// =========================================================================

#[test]
fn test_effect_with_type_var_unifies_with_concrete() {
    // State(?a) and State(Int) should unify with ?a = Int
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_ty = fresh_var(&db, 0);
    let int_ty = Type::new(&db, TypeKind::Int);

    let state_var = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![var_ty],
    };
    let state_int = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![int_ty],
    };

    let r1 = EffectRow::new(&db, vec![state_var], None);
    let r2 = EffectRow::new(&db, vec![state_int], None);

    let result = solver.unify_rows(r1, r2);
    assert!(
        result.is_ok(),
        "State(?a) should unify with State(Int), got {:?}",
        result
    );

    // Check that ?a was unified to Int
    let resolved = solver.type_subst.apply(&db, var_ty);
    assert_eq!(resolved, int_ty, "Type variable should be unified to Int");
}

#[test]
fn test_effect_with_two_type_vars_unifies() {
    // State(?a) and State(?b) should unify with ?a = ?b
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_a = fresh_var(&db, 0);
    let var_b = fresh_var(&db, 1);

    let state_a = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![var_a],
    };
    let state_b = Effect {
        ability_id: test_ability_id(&db, "State"),
        args: vec![var_b],
    };

    let r1 = EffectRow::new(&db, vec![state_a], None);
    let r2 = EffectRow::new(&db, vec![state_b], None);

    let result = solver.unify_rows(r1, r2);
    assert!(
        result.is_ok(),
        "State(?a) should unify with State(?b), got {:?}",
        result
    );

    // Check that they are unified (both resolve to the same type)
    let resolved_a = solver.type_subst.apply(&db, var_a);
    let resolved_b = solver.type_subst.apply(&db, var_b);
    assert_eq!(resolved_a, resolved_b, "Type variables should be unified");
}

#[test]
fn test_effect_mixed_type_var_and_concrete_unifies() {
    // Pair(?a, Int) and Pair(Bool, ?b) should unify with ?a = Bool, ?b = Int
    let db = test_db();
    let mut solver = TypeSolver::new(&db);

    let var_a = fresh_var(&db, 0);
    let var_b = fresh_var(&db, 1);
    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);

    let pair1 = Effect {
        ability_id: test_ability_id(&db, "Pair"),
        args: vec![var_a, int_ty],
    };
    let pair2 = Effect {
        ability_id: test_ability_id(&db, "Pair"),
        args: vec![bool_ty, var_b],
    };

    let r1 = EffectRow::new(&db, vec![pair1], None);
    let r2 = EffectRow::new(&db, vec![pair2], None);

    let result = solver.unify_rows(r1, r2);
    assert!(
        result.is_ok(),
        "Pair(?a, Int) should unify with Pair(Bool, ?b), got {:?}",
        result
    );

    // Check that ?a = Bool and ?b = Int
    assert_eq!(solver.type_subst.apply(&db, var_a), bool_ty);
    assert_eq!(solver.type_subst.apply(&db, var_b), int_ty);
}

#[test]
fn test_types_unifiable_simple() {
    let db = test_db();
    let solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);
    let var_ty = fresh_var(&db, 0);

    // Same types are unifiable
    assert!(solver.types_unifiable(int_ty, int_ty));

    // Different concrete types are not unifiable
    assert!(!solver.types_unifiable(int_ty, bool_ty));

    // Type variable is unifiable with any type
    assert!(solver.types_unifiable(var_ty, int_ty));
    assert!(solver.types_unifiable(int_ty, var_ty));

    // Two type variables are unifiable
    let var_ty2 = fresh_var(&db, 1);
    assert!(solver.types_unifiable(var_ty, var_ty2));

    // Compatibility uses equality, including for effect arguments.
    let never_ty = Type::new(&db, TypeKind::Never);
    assert!(!solver.types_unifiable(never_ty, int_ty));
    assert!(!solver.types_unifiable(bool_ty, never_ty));
    assert!(solver.types_unifiable(never_ty, var_ty));
}

#[test]
fn test_types_unifiable_func() {
    let db = test_db();
    let solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);
    let effect = EffectRow::new(&db, vec![], None);

    // Same function types are unifiable
    let func1 = Type::new(
        &db,
        TypeKind::Func {
            params: vec![int_ty],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    let func2 = Type::new(
        &db,
        TypeKind::Func {
            params: vec![int_ty],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    assert!(solver.types_unifiable(func1, func2));

    // Different param types are not unifiable
    let func3 = Type::new(
        &db,
        TypeKind::Func {
            params: vec![bool_ty],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    assert!(!solver.types_unifiable(func1, func3));

    // Different result types are not unifiable
    let func4 = Type::new(
        &db,
        TypeKind::Func {
            params: vec![int_ty],
            result: bool_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    assert!(!solver.types_unifiable(func1, func4));

    // Function type with type variable in params is unifiable
    let var_ty = fresh_var(&db, 0);
    let func_with_var = Type::new(
        &db,
        TypeKind::Func {
            params: vec![var_ty],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    assert!(solver.types_unifiable(func1, func_with_var));
}

#[test]
fn test_types_unifiable_app() {
    let db = test_db();
    let solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);
    let list_ctor = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: trunk_ir::Symbol::new("List"),
            args: vec![],
        },
    );
    let option_ctor = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option")),
            name: trunk_ir::Symbol::new("Option"),
            args: vec![],
        },
    );

    // Same App types are unifiable
    let app1 = Type::new(
        &db,
        TypeKind::App {
            ctor: list_ctor,
            args: vec![int_ty],
        },
    );
    let app2 = Type::new(
        &db,
        TypeKind::App {
            ctor: list_ctor,
            args: vec![int_ty],
        },
    );
    assert!(solver.types_unifiable(app1, app2));

    // Different constructor is not unifiable
    let app3 = Type::new(
        &db,
        TypeKind::App {
            ctor: option_ctor,
            args: vec![int_ty],
        },
    );
    assert!(!solver.types_unifiable(app1, app3));

    // Different arg types are not unifiable
    let app4 = Type::new(
        &db,
        TypeKind::App {
            ctor: list_ctor,
            args: vec![bool_ty],
        },
    );
    assert!(!solver.types_unifiable(app1, app4));

    // App with type variable in args is unifiable
    let var_ty = fresh_var(&db, 0);
    let app_with_var = Type::new(
        &db,
        TypeKind::App {
            ctor: list_ctor,
            args: vec![var_ty],
        },
    );
    assert!(solver.types_unifiable(app1, app_with_var));
}

// =========================================================================
// row_occurs_in_type tests for params/result recursion
// =========================================================================

#[test]
fn test_row_occurs_in_func_params() {
    // row var in function parameter should be detected
    let db = test_db();
    let solver = TypeSolver::new(&db);

    let row_var = EffectVar { id: 42 };
    let int_ty = Type::new(&db, TypeKind::Int);

    // fn(fn() ->{e} Int) -> Int where we check for e in outer func
    let inner_effect = EffectRow::new(&db, vec![], Some(row_var));
    let inner_func = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect: inner_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    let outer_effect = EffectRow::new(&db, vec![], None);
    let outer_func = Type::new(
        &db,
        TypeKind::Func {
            params: vec![inner_func],
            result: int_ty,
            effect: outer_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    assert!(
        solver.row_occurs_in_type(row_var, outer_func),
        "Row variable in param's effect should be detected"
    );
}

#[test]
fn test_row_occurs_in_func_result() {
    // row var in function result should be detected
    let db = test_db();
    let solver = TypeSolver::new(&db);

    let row_var = EffectVar { id: 42 };
    let int_ty = Type::new(&db, TypeKind::Int);

    // fn() -> fn() ->{e} Int where we check for e in outer func
    let inner_effect = EffectRow::new(&db, vec![], Some(row_var));
    let inner_func = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect: inner_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    let outer_effect = EffectRow::new(&db, vec![], None);
    let outer_func = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: inner_func,
            effect: outer_effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    assert!(
        solver.row_occurs_in_type(row_var, outer_func),
        "Row variable in result's effect should be detected"
    );
}

#[test]
fn test_row_not_in_func_if_absent() {
    // row var not present should return false
    let db = test_db();
    let solver = TypeSolver::new(&db);

    let row_var = EffectVar { id: 42 };
    let other_var = EffectVar { id: 99 };
    let int_ty = Type::new(&db, TypeKind::Int);

    // fn() -> Int with empty effect
    let effect = EffectRow::new(&db, vec![], Some(other_var));
    let func = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: int_ty,
            effect,
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    assert!(
        !solver.row_occurs_in_type(row_var, func),
        "Row variable not present should not be detected"
    );
}

#[test]
fn test_types_unifiable_continuation() {
    let db = test_db();
    let solver = TypeSolver::new(&db);

    let int_ty = Type::new(&db, TypeKind::Int);
    let bool_ty = Type::new(&db, TypeKind::Bool);

    let pure = EffectRow::pure(&db);

    // Same continuation types should be unifiable
    let cont1 = Type::new(
        &db,
        TypeKind::Continuation {
            arg: int_ty,
            result: bool_ty,
            effect: pure,
        },
    );
    let cont2 = Type::new(
        &db,
        TypeKind::Continuation {
            arg: int_ty,
            result: bool_ty,
            effect: pure,
        },
    );
    assert!(solver.types_unifiable(cont1, cont2));

    // Different arg types should not be unifiable
    let cont3 = Type::new(
        &db,
        TypeKind::Continuation {
            arg: bool_ty,
            result: bool_ty,
            effect: pure,
        },
    );
    assert!(!solver.types_unifiable(cont1, cont3));

    // Different result types should not be unifiable
    let cont4 = Type::new(
        &db,
        TypeKind::Continuation {
            arg: int_ty,
            result: int_ty,
            effect: pure,
        },
    );
    assert!(!solver.types_unifiable(cont1, cont4));
}
