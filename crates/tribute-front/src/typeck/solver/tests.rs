use super::*;
use crate::ast::{AbilityId, Effect, EffectRow, TypeDefId, UniVarSource};
use trunk_ir::Symbol;

// Unification and row-unification laws (reflexivity, symmetry, soundness,
// completeness on generalizations, occurs check, set semantics of rows) are
// properties in `laws.rs`. The examples here document specific regressions
// and the intentional exceptions those laws exclude.

fn test_db() -> salsa::DatabaseImpl {
    salsa::DatabaseImpl::new()
}

/// Create an AbilityId for testing (with empty module path).
fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
    AbilityId::source(db, Symbol::new(name))
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
        },
    );

    let result = solver.unify_types(var_a, func_ty);
    assert!(
        result.is_ok(),
        "Should not trigger occurs check when the var is different"
    );
}

// =========================================================================
// Row unification tests (adapted from tribute-passes)
// =========================================================================

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
fn test_unify_named_types_rejects_builtin_and_source_with_same_spelling() {
    let db = test_db();
    let mut solver = TypeSolver::new(&db);
    let name = Symbol::new("List");
    let int_ty = Type::new(&db, TypeKind::Int);
    let builtin = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(&db),
            name: name.clone(),
            args: vec![int_ty],
        },
    );
    let source = Type::new(
        &db,
        TypeKind::Named {
            id: TypeDefId::source(&db, name.clone(), crate::ast::NodeId::from_raw(1)),
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
            name: name.clone(),
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
    solver.defer_producer(first_node, result, vec![first], EffectRow::pure(&db));
    solver.defer_producer(second_node, result, vec![second], EffectRow::pure(&db));
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
    solver.defer_producer(origin.node_id, actual, vec![], EffectRow::pure(&db));
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
                ref effect_name,
                expected: 1,
                found: 0,
            }) if *effect_name == trunk_ir::Symbol::new("State")
        ),
        "Expected EffectArgArityMismatch error, got {:?}",
        result
    );
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
        },
    );
    let outer_effect = EffectRow::new(&db, vec![], None);
    let outer_func = Type::new(
        &db,
        TypeKind::Func {
            params: vec![inner_func],
            result: int_ty,
            effect: outer_effect,
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
        },
    );
    let outer_effect = EffectRow::new(&db, vec![], None);
    let outer_func = Type::new(
        &db,
        TypeKind::Func {
            params: vec![],
            result: inner_func,
            effect: outer_effect,
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
        },
    );

    assert!(
        !solver.row_occurs_in_type(row_var, func),
        "Row variable not present should not be detected"
    );
}
