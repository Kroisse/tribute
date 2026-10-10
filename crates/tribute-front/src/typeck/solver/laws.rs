//! Algebraic laws of type and row unification, checked on generated types.
//!
//! The example tests in `tests.rs` that remain document specific regressions
//! and the intentional exceptions these laws exclude:
//!
//! - `Error` unifies with every type, so generated types here omit it.
//! - A closed empty row unified against a row that names effects succeeds
//!   without binding (pure subsumption); [`RowRelation::PureSubsumes`]
//!   accounts for it.
//!
//! The tests at the end are minimal examples of cases the laws once found
//! failing, kept as regressions.

use proptest::prelude::*;

use super::{ConstraintSet, SolveError, TypeSolver};
use crate::ast::{EffectRow, Type};
use crate::typeck::prop::{
    RowRelation, RowShape, Sharing, TypeGen, TypeShape, context_with_hole, generalization,
    row_and_shuffle, row_shape, rows_equiv, type_shape, types_equiv, univar,
};

/// Unification variables and row tails for open types.
const OPEN: TypeGen = TypeGen::GROUND.univars(4).row_vars(3).higher_kinded(true);

/// Rows with open arguments and tails, one instance per ability.
const ORDERED_ROWS: TypeGen = TypeGen::GROUND
    .univars(3)
    .row_vars(2)
    .row_functions(false)
    .unique_abilities(true);

/// Closed ground types whose rows hold at most one instance of each ability,
/// so every effect of a generalization matches exactly one effect.
const GROUND_UNIQUE: TypeGen = TypeGen::GROUND.higher_kinded(true).unique_abilities(true);

fn solve_types<'db>(
    db: &'db dyn salsa::Database,
    left: Type<'db>,
    right: Type<'db>,
) -> (TypeSolver<'db>, Result<(), SolveError<'db>>) {
    let mut solver = TypeSolver::new(db);
    let mut constraints = ConstraintSet::new();
    constraints.add_type_eq(left, right);
    let result = solver.solve(constraints);
    (solver, result)
}

fn solve_rows<'db>(
    db: &'db dyn salsa::Database,
    left: EffectRow<'db>,
    right: EffectRow<'db>,
) -> (TypeSolver<'db>, Result<(), SolveError<'db>>) {
    let mut solver = TypeSolver::new(db);
    let mut constraints = ConstraintSet::new();
    constraints.add_row_eq(left, right);
    let result = solver.solve(constraints);
    (solver, result)
}

impl<'db> TypeSolver<'db> {
    fn resolved(&self, ty: Type<'db>) -> Type<'db> {
        self.type_subst
            .apply_with_rows(self.db, ty, &self.row_subst)
    }

    /// Whether no relation is left waiting for more information.
    fn settled(&self) -> bool {
        self.pending_row_unions.is_empty()
            && self.pending_row_removals.is_empty()
            && self.pending_row_eqs.is_empty()
    }

    fn has_bindings(&self) -> bool {
        !self.type_subst.map.is_empty() || !self.row_subst.map.is_empty()
    }
}

/// A ground type together with a generalization of it.
fn ground_and_generalization(sharing: Sharing) -> BoxedStrategy<(TypeShape, TypeShape)> {
    type_shape(GROUND_UNIQUE)
        .prop_flat_map(move |shape| (Just(shape.clone()), generalization(&shape, sharing)))
        .boxed()
}

/// Two generalizations of one ground type sharing a small variable pool, so
/// they unify often but not always. Effect arguments may hold function types
/// and share variables with the rest of the type, so unifying them can bind
/// a row tail being unified.
fn related_pair() -> BoxedStrategy<(TypeShape, TypeShape)> {
    let sharing = Sharing::Pool {
        univars: 3,
        row_vars: 2,
    };
    type_shape(GROUND_UNIQUE.row_functions(true))
        .prop_flat_map(move |shape| {
            (
                generalization(&shape, sharing),
                generalization(&shape, sharing),
            )
        })
        .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Reflexivity: every type unifies with itself and binds nothing.
    #[test]
    fn unify_is_reflexive(shape in type_shape(OPEN.error(true))) {
        let db = salsa::DatabaseImpl::new();
        let ty = shape.build(&db);
        let mut solver = TypeSolver::new(&db);
        prop_assert!(solver.unify_types(ty, ty).is_ok());
        prop_assert!(!solver.has_bindings());
        prop_assert_eq!(solver.resolved(ty), ty);
        let row_solver = solve_rows(&db, row_of(&db, ty), row_of(&db, ty)).0;
        prop_assert!(!row_solver.has_bindings());
    }

    /// Completeness on patterns: a ground type unifies with any linear
    /// generalization of it, in either order, and the solution maps the
    /// pattern onto the ground type (rows compared as sets).
    #[test]
    fn ground_type_unifies_with_its_generalization(
        (ground, pattern) in ground_and_generalization(Sharing::Linear),
        pattern_first in any::<bool>(),
    ) {
        let db = salsa::DatabaseImpl::new();
        let (ground, pattern) = (ground.build(&db), pattern.build(&db));
        let (left, right) = if pattern_first { (pattern, ground) } else { (ground, pattern) };
        let (solver, result) = solve_types(&db, left, right);
        prop_assert!(result.is_ok(), "{:?}", result);
        prop_assert!(solver.settled());
        prop_assert!(
            types_equiv(&db, solver.resolved(pattern), ground, RowRelation::SetEqual),
            "{:?} resolved to {:?}, expected {:?}",
            pattern,
            solver.resolved(pattern),
            ground,
        );
    }

    /// Soundness: when unification succeeds and settles, the substitution
    /// makes both sides equal, and applying it again changes nothing.
    #[test]
    fn successful_unification_equates_both_sides((left, right) in related_pair()) {
        let db = salsa::DatabaseImpl::new();
        let (left, right) = (left.build(&db), right.build(&db));
        let (solver, result) = solve_types(&db, left, right);
        if result.is_ok() && solver.settled() {
            let (left, right) = (solver.resolved(left), solver.resolved(right));
            prop_assert!(
                types_equiv(&db, left, right, RowRelation::PureSubsumes),
                "resolved sides differ: {:?} vs {:?}",
                left,
                right,
            );
            prop_assert_eq!(solver.resolved(left), left);
            prop_assert_eq!(solver.resolved(right), right);
        }
    }

    /// The quick compatibility check never rejects a pair that unifies.
    #[test]
    fn unifiable_pairs_pass_the_quick_check((left, right) in related_pair()) {
        let db = salsa::DatabaseImpl::new();
        let (left, right) = (left.build(&db), right.build(&db));
        let (_, result) = solve_types(&db, left, right);
        if result.is_ok() {
            prop_assert!(TypeSolver::new(&db).types_unifiable(left, right));
        }
    }

    /// Symmetry: when one order of the sides finds a unifier (both sides
    /// resolve equal, rows as sets), so does the other. Success alone is not
    /// symmetric: pure subsumption accepts a closed empty row on the left
    /// against a row naming effects, but not the reverse, and which side a
    /// row lands on can depend on bindings made earlier in the traversal
    /// (`test_pure_callee_in_effectful_context`).
    #[test]
    fn unification_is_symmetric((left, right) in related_pair()) {
        let db = salsa::DatabaseImpl::new();
        let (left, right) = (left.build(&db), right.build(&db));
        for (first, second) in [(left, right), (right, left)] {
            let (solver, result) = solve_types(&db, first, second);
            let unifier = result.is_ok()
                && solver.settled()
                && types_equiv(&db, solver.resolved(first), solver.resolved(second), RowRelation::SetEqual);
            if unifier {
                let reversed = solve_types(&db, second, first).1;
                prop_assert!(reversed.is_ok(), "{:?}", reversed);
            }
        }
    }

    /// Closed ground types unify exactly when they are structurally equal,
    /// with rows compared as sets; the quick check agrees when rows are
    /// ignored.
    #[test]
    fn ground_types_unify_iff_equal(
        left in type_shape(TypeGen::GROUND.higher_kinded(true)),
        right in type_shape(TypeGen::GROUND.higher_kinded(true)),
    ) {
        let db = salsa::DatabaseImpl::new();
        for (left, right) in [(left.build(&db), right.build(&db)), (left.build(&db), left.build(&db))] {
            let equal = types_equiv(&db, left, right, RowRelation::SetEqual);
            let (solver, result) = solve_types(&db, left, right);
            prop_assert_eq!(result.is_ok(), equal, "{:?} ~ {:?}: {:?}", left, right, result);
            prop_assert!(!solver.has_bindings());
            prop_assert_eq!(
                TypeSolver::new(&db).types_unifiable(left, right),
                types_equiv(&db, left, right, RowRelation::Ignore),
            );
        }
    }

    /// Occurs check: a variable never unifies with a type that strictly
    /// contains it, wherever it occurs, including inside an effect argument.
    #[test]
    fn variable_does_not_unify_with_a_type_containing_it(
        context in context_with_hole(TypeGen::GROUND.higher_kinded(true), TypeShape::UniVar(0)),
        var_first in any::<bool>(),
    ) {
        let db = salsa::DatabaseImpl::new();
        let var = univar(&db, 0);
        let context = context.build(&db);
        let mut solver = TypeSolver::new(&db);
        let result = if var_first {
            solver.unify_types(var, context)
        } else {
            solver.unify_types(context, var)
        };
        prop_assert!(matches!(result, Err(SolveError::OccursCheck { .. })), "{:?}", result);
        prop_assert!(!solver.has_bindings());
    }

    /// Closed rows with ground arguments unify exactly when they hold the
    /// same effects as sets. Effect identity includes the ability's origin
    /// and module and every type argument.
    #[test]
    fn closed_ground_rows_unify_iff_same_set(
        left in row_shape(TypeGen::GROUND),
        right in row_shape(TypeGen::GROUND),
        (row, shuffled) in row_and_shuffle(TypeGen::GROUND),
    ) {
        let db = salsa::DatabaseImpl::new();
        for (left, right) in [(&left, &right), (&row, &shuffled)] {
            let same = rows_equiv(&db, left.build(&db), right.build(&db), RowRelation::SetEqual);
            let (solver, result) = solve_rows(&db, left.build(&db), right.build(&db));
            prop_assert_eq!(result.is_ok(), same, "{:?} ~ {:?}: {:?}", left, right, result);
            prop_assert!(result.is_ok() || matches!(result, Err(SolveError::RowMismatch { .. })), "{:?}", result);
            prop_assert!(!solver.has_bindings());
        }
    }

    /// Two open rows over the same tail unify exactly when their effects
    /// are the same set.
    #[test]
    fn rows_sharing_a_tail_unify_iff_same_set(
        left in row_shape(TypeGen::GROUND.row_functions(false)),
        right in row_shape(TypeGen::GROUND.row_functions(false)),
    ) {
        let db = salsa::DatabaseImpl::new();
        let tail = Some(crate::typeck::prop::ROW_VAR_BASE);
        let left = RowShape { rest: tail, ..left };
        let right = RowShape { rest: tail, ..right };
        let (_, result) = solve_rows(&db, left.build(&db), right.build(&db));
        prop_assert_eq!(result.is_ok(), left.same_effect_set(&right), "{:?}", result);
    }

    /// An open row unifies with a closed ground row exactly when its effects
    /// are a subset; the tail is bound to the remaining effects.
    #[test]
    fn open_row_binds_tail_to_the_difference(
        open in row_shape(TypeGen::GROUND.row_functions(false)),
        closed in row_shape(TypeGen::GROUND.row_functions(false)),
        open_first in any::<bool>(),
    ) {
        let db = salsa::DatabaseImpl::new();
        let tail = crate::typeck::prop::ROW_VAR_BASE;
        let open = RowShape { rest: Some(tail), ..open };
        let (open_row, closed_row) = (open.build(&db), closed.build(&db));
        let (left, right) = if open_first { (open_row, closed_row) } else { (closed_row, open_row) };
        let (solver, result) = solve_rows(&db, left, right);
        let subset = open.effects.iter().all(|effect| closed.effects.contains(effect));
        // Pure subsumption: a closed empty row accepts any row naming effects.
        let subsumed = !open_first && closed.effects.is_empty() && !open.effects.is_empty();
        prop_assert_eq!(result.is_ok(), subset || subsumed, "{:?}", result);
        if subset && !subsumed {
            let bound = solver.row_subst.get(tail).expect("tail bound");
            let difference = RowShape::closed(
                closed
                    .effects
                    .iter()
                    .filter(|effect| !open.effects.contains(effect))
                    .cloned()
                    .collect(),
            );
            prop_assert!(rows_equiv(&db, bound, difference.build(&db), RowRelation::SetEqual));
            prop_assert!(rows_equiv(
                &db,
                solver.row_subst.apply(&db, open_row),
                closed_row,
                RowRelation::SetEqual,
            ));
        }
    }

    /// Two open rows with distinct tails always unify, and both resolve to
    /// the union of their effects over one common tail.
    #[test]
    fn distinct_tails_unify_to_the_union(
        left in row_shape(TypeGen::GROUND.row_functions(false)),
        right in row_shape(TypeGen::GROUND.row_functions(false)),
    ) {
        let db = salsa::DatabaseImpl::new();
        let base = crate::typeck::prop::ROW_VAR_BASE;
        let left = RowShape { rest: Some(base), ..left }.build(&db);
        let right = RowShape { rest: Some(base + 1), ..right }.build(&db);
        let (solver, result) = solve_rows(&db, left, right);
        prop_assert!(result.is_ok(), "{:?}", result);
        let (left, right) = (solver.row_subst.apply(&db, left), solver.row_subst.apply(&db, right));
        prop_assert!(rows_equiv(&db, left, right, RowRelation::SetEqual), "{:?} vs {:?}", left, right);
        prop_assert!(left.rest(&db).is_some_and(|tail| tail.id > base + 1));
    }

    /// Row unification does not depend on the order effects are listed in:
    /// permuting one side keeps the outcome and the type bindings. Rows here
    /// hold one instance per ability, so no effect has several candidates
    /// (see `row_equality_with_ambiguous_candidates_ignores_effect_order`).
    #[test]
    fn row_unification_ignores_effect_order(
        (left, shuffled) in row_and_shuffle(ORDERED_ROWS),
        right in row_shape(ORDERED_ROWS),
    ) {
        let db = salsa::DatabaseImpl::new();
        let right = right.build(&db);
        let (original, original_result) = solve_rows(&db, left.build(&db), right);
        let (permuted, permuted_result) = solve_rows(&db, shuffled.build(&db), right);
        prop_assert_eq!(original_result.is_ok(), permuted_result.is_ok());
        if original_result.is_ok() {
            for id in 0..3 {
                let var = univar(&db, id);
                prop_assert_eq!(original.resolved(var), permuted.resolved(var));
            }
        }
    }
}

/// A row whose only effect argument is `ty`, to check reflexivity of row
/// unification on rows with arbitrary arguments.
fn row_of<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> EffectRow<'db> {
    EffectRow::new(
        db,
        vec![crate::ast::Effect {
            ability_id: crate::typeck::prop::ability_id(db, 1),
            args: vec![ty],
        }],
        None,
    )
}

/// Closed-row equality does not depend on effect order when the other row
/// holds several candidates for one effect. `Console` has no candidate in
/// `{State(?1), State(?2)}`, so the rows cannot be equal, even though
/// `State(Int)` is ambiguous between the two.
#[test]
fn row_equality_with_ambiguous_candidates_ignores_effect_order() {
    use crate::typeck::prop::{EffectShape, Prim};
    let db = salsa::DatabaseImpl::new();
    let console = EffectShape {
        ability: 0,
        args: vec![],
    };
    let state = |arg| EffectShape {
        ability: 1,
        args: vec![arg],
    };
    let candidates = RowShape::closed(vec![
        state(TypeShape::UniVar(1)),
        state(TypeShape::UniVar(2)),
    ])
    .build(&db);
    let outcomes: Vec<_> = [
        vec![console.clone(), state(TypeShape::Prim(Prim::Int))],
        vec![state(TypeShape::Prim(Prim::Int)), console],
    ]
    .into_iter()
    .map(|effects| {
        let (mut solver, result) =
            solve_rows(&db, RowShape::closed(effects).build(&db), candidates);
        result.is_ok() && solver.finalize_relations().is_ok()
    })
    .collect();
    assert_eq!(outcomes, [false, false]);
}

/// An equality deferred for ambiguity that no type decides is reported once
/// relations settle. `State(?1)` could match either instance, and no `?1`
/// makes `{State(?1)}` equal to `{State(Int), State(Bool)}`.
#[test]
fn unsettled_ambiguous_row_equality_is_reported() {
    use crate::typeck::prop::{EffectShape, Prim};
    let db = salsa::DatabaseImpl::new();
    let state = |arg| EffectShape {
        ability: 1,
        args: vec![arg],
    };
    let one = RowShape::closed(vec![state(TypeShape::UniVar(1))]).build(&db);
    let two = RowShape::closed(vec![
        state(TypeShape::Prim(Prim::Int)),
        state(TypeShape::Prim(Prim::Bool)),
    ])
    .build(&db);
    let (mut solver, result) = solve_rows(&db, one, two);
    assert!(result.is_ok() && solver.finalize_relations().is_ok());
    assert!(matches!(
        solver.unsettled_ambiguity().map(|failure| failure.error),
        Some(SolveError::AmbiguousEffect { .. })
    ));
}

/// Binding an open row's tail to the remaining effects keeps a binding of
/// that tail made while unifying effect arguments. Equating
/// `{State(fn() ->{e1} Nil) | e1}` with `{State(fn() ->{e2} Nil)}` first
/// unifies the arguments, binding `e1` and `e2` to a fresh common tail; the
/// rows are then equated again under that binding, closing the common tail.
#[test]
fn open_row_tail_binding_keeps_argument_bindings() {
    use crate::typeck::prop::{EffectShape, Prim, ROW_VAR_BASE};
    let db = salsa::DatabaseImpl::new();
    let thunk = |tail| TypeShape::Func {
        params: vec![],
        result: Box::new(TypeShape::Prim(Prim::Nil)),
        effect: RowShape {
            effects: vec![],
            rest: Some(tail),
        },
    };
    let state = |tail| EffectShape {
        ability: 1,
        args: vec![thunk(tail)],
    };
    let (e1, e2) = (ROW_VAR_BASE, ROW_VAR_BASE + 1);
    let open = RowShape {
        effects: vec![state(e1)],
        rest: Some(e1),
    }
    .build(&db);
    let closed = RowShape::closed(vec![state(e2)]).build(&db);
    let (solver, result) = solve_rows(&db, open, closed);
    assert!(result.is_ok());
    let (open, closed) = (solver.normalize_row(open), solver.normalize_row(closed));
    assert!(rows_equiv(&db, open, closed, RowRelation::SetEqual));
}

/// A tail may occur in the effects bound to the other tail: equating
/// `{State(fn() ->{e1} Nil) | e1}` with `{|e2}` binds `e1` to a fresh tail
/// and `e2` to `State(fn() ->{e1} Nil)` over it, which is no cycle.
#[test]
fn open_rows_allow_a_tail_in_the_other_remainder() {
    use crate::typeck::prop::{EffectShape, Prim, ROW_VAR_BASE};
    let db = salsa::DatabaseImpl::new();
    let (e1, e2) = (ROW_VAR_BASE, ROW_VAR_BASE + 1);
    let state = EffectShape {
        ability: 1,
        args: vec![TypeShape::Func {
            params: vec![],
            result: Box::new(TypeShape::Prim(Prim::Nil)),
            effect: RowShape {
                effects: vec![],
                rest: Some(e1),
            },
        }],
    };
    let left = RowShape {
        effects: vec![state],
        rest: Some(e1),
    }
    .build(&db);
    let right = RowShape {
        effects: vec![],
        rest: Some(e2),
    }
    .build(&db);
    let (solver, result) = solve_rows(&db, left, right);
    assert!(result.is_ok(), "{result:?}");
    let (left, right) = (solver.normalize_row(left), solver.normalize_row(right));
    assert!(rows_equiv(&db, left, right, RowRelation::SetEqual));
}

/// Binding both tails must not form a cycle through each other: in
/// `{State(fn() ->{e1} Nil) | e1}` against `{Choice(fn() ->{e2} Nil, Nil) | e2}`,
/// `e1` would name `Choice` over `e2` and `e2` would name `State` over `e1`.
#[test]
fn open_rows_reject_a_cycle_through_both_tails() {
    use crate::typeck::prop::{EffectShape, Prim, ROW_VAR_BASE};
    let db = salsa::DatabaseImpl::new();
    let thunk = |tail| TypeShape::Func {
        params: vec![],
        result: Box::new(TypeShape::Prim(Prim::Nil)),
        effect: RowShape {
            effects: vec![],
            rest: Some(tail),
        },
    };
    let (e1, e2) = (ROW_VAR_BASE, ROW_VAR_BASE + 1);
    let left = RowShape {
        effects: vec![EffectShape {
            ability: 1,
            args: vec![thunk(e1)],
        }],
        rest: Some(e1),
    }
    .build(&db);
    let right = RowShape {
        effects: vec![EffectShape {
            ability: 3,
            args: vec![thunk(e2), TypeShape::Prim(Prim::Nil)],
        }],
        rest: Some(e2),
    }
    .build(&db);
    let (_, result) = solve_rows(&db, left, right);
    assert!(result.is_err());
}

/// Matching the effects of two open rows does not depend on effect order
/// when one variable occurs in several effects. In `{State(?1), Choice(?1) | e2}`
/// against `{State(Int), Choice(Bool) | e1}` each effect has one candidate,
/// and the two matches disagree on `?1`. Unifying `State` first must not
/// leave `Choice(Bool)` without a candidate, nor the reverse.
#[test]
fn open_row_matching_ignores_effect_order() {
    use crate::typeck::prop::{EffectShape, Prim, ROW_VAR_BASE};
    let db = salsa::DatabaseImpl::new();
    let effect = |ability, arg| EffectShape {
        ability,
        args: vec![arg],
    };
    let state = effect(1, TypeShape::Prim(Prim::Int));
    let choice = effect(2, TypeShape::Prim(Prim::Bool));
    let shared = RowShape {
        effects: vec![
            effect(1, TypeShape::UniVar(1)),
            effect(2, TypeShape::UniVar(1)),
        ],
        rest: Some(ROW_VAR_BASE + 1),
    }
    .build(&db);
    for effects in [vec![state.clone(), choice.clone()], vec![choice, state]] {
        let row = RowShape {
            effects,
            rest: Some(ROW_VAR_BASE),
        }
        .build(&db);
        for (left, right) in [(row, shared), (shared, row)] {
            let (_, result) = solve_rows(&db, left, right);
            assert!(
                matches!(result, Err(SolveError::TypeMismatch { .. })),
                "{result:?}"
            );
        }
    }
}
