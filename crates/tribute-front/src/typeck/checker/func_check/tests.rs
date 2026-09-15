use std::collections::HashMap;

use salsa_test_macros::salsa_test;
use trunk_ir::Symbol;

use crate::ast::{EffectRow, NodeId, SpanMap, Type, TypeDefId, TypeKind};
use crate::typeck::{TypeChecker, TypeSolver};

use super::super::super::solver::SolveError;
use super::{ConstraintOriginKind, format_solve_error, solve_error_context};

#[test]
fn solve_error_context_describes_each_origin() {
    assert_eq!(
        [
            solve_error_context(Some(ConstraintOriginKind::Call)),
            solve_error_context(Some(ConstraintOriginKind::Lambda)),
            solve_error_context(Some(ConstraintOriginKind::HandlerBoundary)),
            solve_error_context(Some(ConstraintOriginKind::Expression)),
            solve_error_context(None),
        ],
        [
            " at call site",
            " in lambda",
            " at handler boundary",
            "",
            ""
        ]
    );
}

#[salsa_test]
fn format_solve_error_describes_builtin_actual_against_source_expected(db: &dyn salsa::Database) {
    let int_ty = Type::new(db, TypeKind::Int);
    let source = Type::new(
        db,
        TypeKind::Named {
            id: TypeDefId::source(db, Symbol::new("List"), NodeId::from_raw(1)),
            name: Symbol::new("List"),
            args: vec![int_ty],
        },
    );
    let builtin = Type::new(
        db,
        TypeKind::Named {
            id: TypeDefId::builtin_list(db),
            name: Symbol::new("List"),
            args: vec![int_ty],
        },
    );

    assert_eq!(
        format_solve_error(
            db,
            &SolveError::TypeMismatch {
                expected: source,
                actual: builtin,
            },
        ),
        "canonical compiler-owned type `List(Int)` is distinct from source-declared type `List(Int)`"
    );
}

#[salsa_test]
fn collect_deferred_resolution_univars_includes_callee_type(db: &salsa::DatabaseImpl) {
    let mut solver = TypeSolver::new(db);
    let first_solver_var = solver.fresh_type_var(db);
    let second_solver_var = solver.fresh_type_var(db);
    let first_callee_ty = Type::new(
        db,
        TypeKind::Func {
            params: vec![Type::new(db, TypeKind::Nat)],
            result: first_solver_var,
            effect: EffectRow::pure(db),
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );
    let second_callee_ty = Type::new(
        db,
        TypeKind::Func {
            params: vec![Type::new(db, TypeKind::Nat)],
            result: second_solver_var,
            effect: EffectRow::pure(db),
            minimum_convention: crate::ast::CallingConvention::Direct,
        },
    );

    let mut deferred_resolutions = HashMap::new();
    deferred_resolutions.insert(
        NodeId::from_raw(2),
        (
            crate::ast::FuncDefId::new(db, Symbol::new("Box::flat_map")),
            second_callee_ty,
        ),
    );
    deferred_resolutions.insert(
        NodeId::from_raw(1),
        (
            crate::ast::FuncDefId::new(db, Symbol::new("Box::map")),
            first_callee_ty,
        ),
    );

    let checker = TypeChecker::new(db, SpanMap::default());
    let mut collected = Vec::new();
    checker.collect_univars_from_deferred_resolutions(
        &deferred_resolutions,
        solver.type_subst(),
        solver.row_subst(),
        &mut collected,
    );

    let TypeKind::UniVar {
        id: first_solver_id,
    } = first_solver_var.kind(db)
    else {
        panic!("fresh solver variable should be a UniVar");
    };
    let TypeKind::UniVar {
        id: second_solver_id,
    } = second_solver_var.kind(db)
    else {
        panic!("fresh solver variable should be a UniVar");
    };
    assert_eq!(collected, vec![*first_solver_id, *second_solver_id]);
}
