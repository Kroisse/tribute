use super::*;
use salsa_test_macros::salsa_test;

#[salsa_test]
fn row_removal_defers_ambiguous_types(db: &salsa::DatabaseImpl) {
    let ability = crate::ast::AbilityId::source(db, trunk_ir::Symbol::new("Writer"));
    let nat = Type::new(db, TypeKind::Nat);
    let int = Type::new(db, TypeKind::Int);
    let row = |ty| {
        EffectRow::single(
            db,
            Effect {
                ability_id: ability,
                args: vec![ty],
            },
        )
    };
    for reverse in [false, true] {
        let mut solver = TypeSolver::new(db);
        let result = EffectRow::open(db, solver.fresh_row_var());
        let mut labels = vec![
            row(nat).effects(db)[0].clone(),
            row(int).effects(db)[0].clone(),
        ];
        if reverse {
            labels.reverse();
        }
        solver.add_row_removals(vec![crate::ast::RowRemoval {
            source: EffectRow::new(db, labels, None),
            removed: row(nat),
            result,
        }]);
        solver.finalize_relations().unwrap();
        assert_eq!(solver.row_subst().apply(db, result), row(int));
    }
    let mut solver = TypeSolver::new(db);
    let unknown = solver.fresh_type_var(db);
    let result = EffectRow::open(db, solver.fresh_row_var());
    solver.add_row_removals(vec![crate::ast::RowRemoval {
        source: row(unknown),
        removed: row(nat),
        result,
    }]);
    solver.finalize_relations().unwrap();
    assert_eq!(solver.type_subst().apply(db, unknown), unknown);
    assert_eq!(solver.retained_row_removals().len(), 1);
    let mut constraints = ConstraintSet::new();
    constraints.add_type_eq(unknown, int);
    solver.solve(constraints).unwrap();
    assert_eq!(solver.row_subst().apply(db, result), row(int));
    assert!(solver.retained_row_removals().is_empty());
}

#[salsa_test]
fn row_removal_empty_result_does_not_close_its_source(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let source = EffectRow::open(db, solver.fresh_row_var());
    solver.add_row_removals(vec![crate::ast::RowRemoval {
        source,
        removed: label(db, "Ping"),
        result: EffectRow::pure(db),
    }]);
    solver.finalize_relations().unwrap();
    assert_eq!(solver.row_subst().apply(db, source), source);
    let mut constraints = ConstraintSet::new();
    constraints.add_row_eq(source, label(db, "Ping"));
    solver.solve(constraints).unwrap();
    assert!(solver.retained_row_removals().is_empty());
}

#[salsa_test]
fn row_removal_rejects_reintroduced_labels(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let source = EffectRow::open(db, solver.fresh_row_var());
    solver.add_row_removals(vec![crate::ast::RowRemoval {
        source,
        removed: label(db, "Ping"),
        result: label(db, "Ping"),
    }]);
    assert!(solver.finalize_relations().is_err());
}

#[salsa_test]
fn row_removal_uses_exact_ability_identity(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let name = trunk_ir::Symbol::new("Io");
    let builtin = crate::ast::AbilityId::new(
        db,
        crate::ast::AbilityOrigin::Builtin(crate::ast::BuiltinAbility::Io),
        name,
    );
    let source = crate::ast::AbilityId::source(db, name);
    let effect = |ability_id| Effect {
        ability_id,
        args: vec![],
    };
    let result = EffectRow::open(db, solver.fresh_row_var());
    solver.add_row_removals(vec![crate::ast::RowRemoval {
        source: EffectRow::new(db, vec![effect(builtin), effect(source)], None),
        removed: EffectRow::single(db, effect(builtin)),
        result,
    }]);
    solver.finalize_relations().unwrap();
    assert_eq!(
        solver.row_subst().apply(db, result),
        EffectRow::single(db, effect(source))
    );
}

fn label<'db>(db: &'db dyn salsa::Database, name: &str) -> EffectRow<'db> {
    EffectRow::single(
        db,
        Effect {
            ability_id: crate::ast::AbilityId::source(db, trunk_ir::Symbol::from_dynamic(name)),
            args: vec![],
        },
    )
}

#[salsa_test]
fn unrelated_row_union_stays_in_solver_without_entering_value_scheme(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let left = EffectRow::open(db, EffectVar { id: 1 });
    let right = EffectRow::open(db, EffectVar { id: 2 });
    solver.add_row_unions(vec![crate::ast::RowUnion {
        sources: vec![left, right],
        result: label(db, "Writer"),
    }]);
    solver.finalize_relations().unwrap();
    assert!(
        solver
            .row_unions_for_type(Type::new(db, TypeKind::Nat))
            .is_empty()
    );
    assert_eq!(solver.retained_row_unions().len(), 1);
    let mut constraints = ConstraintSet::new();
    constraints.add_row_eq(left, label(db, "Reader"));
    assert!(solver.solve(constraints).is_err());
}

#[salsa_test]
fn scheme_row_dependencies_follow_shared_effect_type_arguments(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let variable = Type::new(
        db,
        TypeKind::UniVar {
            id: UniVarId::new(db, crate::ast::UniVarSource::Anonymous(813), 0),
        },
    );
    let result = EffectRow::single(
        db,
        Effect {
            ability_id: crate::ast::AbilityId::source(db, trunk_ir::Symbol::new("Writer")),
            args: vec![variable],
        },
    );
    for first in [1, 3] {
        solver.add_row_unions(vec![crate::ast::RowUnion {
            sources: vec![
                EffectRow::open(db, EffectVar { id: first }),
                EffectRow::open(db, EffectVar { id: first + 1 }),
            ],
            result,
        }]);
    }
    solver.finalize_relations().unwrap();
    let nil = Type::new(db, TypeKind::Nil);
    let callable = Type::new(
        db,
        TypeKind::Func {
            params: vec![],
            result: nil,
            effect: EffectRow::open(db, EffectVar { id: 1 }),
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    // The second union is reachable only through the first union's
    // Writer argument, not through a shared row variable.
    assert_eq!(solver.row_unions_for_type(callable).len(), 2);
    assert_eq!(solver.row_unions_for_type(variable).len(), 2);
}

#[salsa_test]
fn row_union_retains_both_delayed_sources(db: &salsa::DatabaseImpl) {
    for reverse in [false, true] {
        let mut solver = TypeSolver::new(db);
        let left = EffectRow::open(db, EffectVar { id: 10 });
        let right = EffectRow::open(db, EffectVar { id: 11 });
        let result = EffectRow::open(db, EffectVar { id: 12 });
        let sources = if reverse {
            vec![right, left]
        } else {
            vec![left, right]
        };
        solver.add_row_unions(vec![crate::ast::RowUnion { sources, result }]);
        solver.finalize_relations().unwrap();
        assert_eq!(solver.retained_row_unions().len(), 1);
        assert!(solver.row_subst.get(10).is_none());
        assert!(solver.row_subst.get(11).is_none());
        let mut constraints = ConstraintSet::new();
        constraints.add_row_eq(left, label(db, "Reader"));
        solver.solve(constraints).unwrap();
        let mut constraints = ConstraintSet::new();
        constraints.add_row_eq(right, label(db, "Writer"));
        solver.solve(constraints).unwrap();
        let result = solver.row_subst.apply(db, result);
        assert!(result.rest(db).is_none());
        assert_eq!(result.effects(db).len(), 2);
        assert!(solver.retained_row_unions().is_empty());
    }
}

#[salsa_test]
fn row_union_closed_result_does_not_choose_a_source(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let left = EffectRow::open(db, EffectVar { id: 1 });
    let right = EffectRow::open(db, EffectVar { id: 2 });
    solver.add_row_unions(vec![crate::ast::RowUnion {
        sources: vec![left, right],
        result: label(db, "Writer"),
    }]);
    solver.finalize_relations().unwrap();
    assert!(solver.row_subst.get(1).is_none());
    assert!(solver.row_subst.get(2).is_none());
    let mut constraints = ConstraintSet::new();
    constraints.add_row_eq(left, label(db, "Reader"));
    assert!(solver.solve(constraints).is_err());
}

#[salsa_test]
fn row_union_pure_result_closes_every_source(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let left = EffectRow::open(db, EffectVar { id: 1 });
    let right = EffectRow::open(db, EffectVar { id: 2 });
    solver.add_row_unions(vec![crate::ast::RowUnion {
        sources: vec![left, right],
        result: EffectRow::pure(db),
    }]);
    solver.finalize_relations().unwrap();
    assert!(solver.row_subst.apply(db, left).is_pure(db));
    assert!(solver.row_subst.apply(db, right).is_pure(db));
}
#[salsa_test]
fn row_union_defers_ambiguous_instance_until_type_constraint_arrives(db: &salsa::DatabaseImpl) {
    let variable = Type::new(
        db,
        TypeKind::UniVar {
            id: UniVarId::new(db, crate::ast::UniVarSource::Anonymous(812), 0),
        },
    );
    let nat = Type::new(db, TypeKind::Nat);
    let int = Type::new(db, TypeKind::Int);
    let ability = crate::ast::AbilityId::source(db, trunk_ir::Symbol::new("Writer"));
    let effect = |arg| Effect {
        ability_id: ability,
        args: vec![arg],
    };
    for (reverse, swap) in [(false, false), (true, false), (false, true), (true, true)] {
        let source = EffectRow::new(db, vec![effect(variable), effect(nat)], None);
        let result = EffectRow::new(
            db,
            if reverse {
                vec![effect(int), effect(nat)]
            } else {
                vec![effect(nat), effect(int)]
            },
            None,
        );
        let mut solver = TypeSolver::new(db);
        let mut constraints = ConstraintSet::new();
        if swap {
            constraints.add_row_eq(result, source);
        } else {
            constraints.add_row_eq(source, result);
        }
        solver.solve(constraints).unwrap();
        assert_eq!(solver.type_subst().apply(db, variable), variable);
        assert!(!solver.retained_row_unions().is_empty());
        let mut constraints = ConstraintSet::new();
        constraints.add_type_eq(variable, int);
        solver.solve(constraints).unwrap();
        solver.finalize_relations().unwrap();
        assert!(solver.retained_row_unions().is_empty());
    }
}
