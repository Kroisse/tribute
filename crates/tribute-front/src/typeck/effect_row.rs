//! Effect row utilities for row-polymorphic effect typing.
//!
//! This module provides utility functions for working with effect rows
//! during type checking, including:
//!
//! - Effect containment checks
//! - Effect row union operations
//! - Handler typing support (remove_with_constraint)
//! - Effect conflict detection

use trunk_ir::Symbol;

use crate::ast::{AbilityId, Effect, EffectRow, EffectVar, Type};

/// Result of attempting to remove an effect from an effect row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoveResult<'db> {
    /// Effect was directly present and removed.
    Removed(EffectRow<'db>),

    /// Effect was not in the concrete part, but might be in the row variable tail.
    /// Generates a constraint: `var = {must_contain | remainder}`.
    NeedsConstraint {
        /// The row variable that must be decomposed.
        var: EffectVar,
        /// The effect that must be contained in the row variable.
        must_contain: Effect<'db>,
        /// Fresh row variable for the remainder after removing the effect.
        remainder: EffectVar,
    },

    /// Effect was not found and the row is closed (no tail variable).
    NotFound,
}

/// Check if an effect row contains a specific effect.
pub fn contains<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    effect: &Effect<'db>,
) -> bool {
    row.effects(db).iter().any(|e| e == effect)
}

/// Check if an effect row contains an effect with a specific name (ignoring type args).
pub fn contains_by_name<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    name: &Symbol,
) -> bool {
    row.effects(db)
        .iter()
        .any(|e| e.ability_id.name(db) == *name)
}

/// Find all effects matching a given name (ignoring type parameters).
///
/// This is used for handler pattern matching where the pattern specifies
/// only the ability name but we need to find the fully parameterized ability.
pub fn find_by_name<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    name: &Symbol,
) -> Vec<Effect<'db>> {
    row.effects(db)
        .iter()
        .filter(|e| e.ability_id.name(db) == *name)
        .cloned()
        .collect()
}

/// Add an effect to an effect row, returning a new row.
pub fn add_effect<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    effect: Effect<'db>,
) -> EffectRow<'db> {
    let mut effects = row.effects(db).to_vec();
    if !effects.contains(&effect) {
        effects.push(effect);
    }
    EffectRow::new(db, effects, row.rest(db))
}

/// Remove an effect from an effect row, returning a new row.
///
/// Returns `Some(new_row)` if the effect was present, `None` otherwise.
pub fn remove_effect<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    effect: &Effect<'db>,
) -> Option<EffectRow<'db>> {
    let effects = row.effects(db);
    if let Some(pos) = effects.iter().position(|e| e == effect) {
        let mut new_effects = effects.to_vec();
        new_effects.remove(pos);
        Some(EffectRow::new(db, new_effects, row.rest(db)))
    } else {
        None
    }
}

/// Remove an effect with row variable decomposition support.
///
/// This method handles the case where the effect might be in a row variable tail.
/// Used by handler typing to correctly remove handled effects from effect rows.
pub fn remove_with_constraint<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    effect: &Effect<'db>,
    fresh_var: impl FnOnce() -> EffectVar,
) -> RemoveResult<'db> {
    if contains(db, row, effect) {
        // Direct removal
        let new_row = remove_effect(db, row, effect).unwrap();
        RemoveResult::Removed(new_row)
    } else if let Some(tail) = row.rest(db) {
        // Row variable decomposition
        let fresh = fresh_var();
        RemoveResult::NeedsConstraint {
            var: tail,
            must_contain: effect.clone(),
            remainder: fresh,
        }
    } else {
        // Closed row without the effect
        RemoveResult::NotFound
    }
}

/// Union two effect rows, combining their effects.
pub fn union<'db>(
    db: &'db dyn salsa::Database,
    row1: EffectRow<'db>,
    row2: EffectRow<'db>,
    fresh_var: impl FnOnce() -> EffectVar,
) -> (EffectRow<'db>, Option<crate::ast::RowUnion<'db>>) {
    let mut effects = row1.effects(db).to_vec();
    for effect in row2.effects(db) {
        if !effects.contains(effect) {
            effects.push(effect.clone());
        }
    }
    let independent = matches!((row1.rest(db), row2.rest(db)), (Some(a), Some(b)) if a != b);
    let rest = match (row1.rest(db), row2.rest(db)) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(a), Some(b)) if a == b => Some(a),
        (Some(_), Some(_)) => Some(fresh_var()),
    };
    let result = EffectRow::new(db, effects, rest);
    let constraint = independent.then(|| crate::ast::RowUnion {
        sources: vec![row1, row2],
        result,
    });
    (result, constraint)
}

/// Check for duplicate effects in an effect row.
///
/// Effects are considered duplicates only if they have both the same name
/// AND the same type arguments. For example:
/// - `State(Int) + State(Text)` = OK (different effects)
/// - `State(Int) + State(Int)` = conflict (duplicate)
///
/// Returns `Some((ability_id, effects))` if duplicates are found.
pub fn find_conflicting_effects<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
) -> Option<(AbilityId<'db>, Vec<Effect<'db>>)> {
    use rustc_hash::FxHashSet as HashSet;

    let effects = row.effects(db);
    let mut seen: HashSet<&Effect<'db>> = HashSet::default();

    for effect in effects.iter() {
        if !seen.insert(effect) {
            // Found a duplicate - return all instances of this effect
            let duplicates: Vec<Effect<'db>> =
                effects.iter().filter(|e| *e == effect).cloned().collect();
            return Some((effect.ability_id, duplicates));
        }
    }

    None
}

/// Create a simple effect with no type arguments.
pub fn simple_effect<'db>(
    _db: &'db dyn salsa::Database,
    ability_id: AbilityId<'db>,
) -> Effect<'db> {
    Effect {
        ability_id,
        args: Vec::new(),
    }
}

/// Create an effect with type arguments.
pub fn parameterized_effect<'db>(
    db: &'db dyn salsa::Database,
    ability_id: AbilityId<'db>,
    args: Vec<Type<'db>>,
) -> Effect<'db> {
    let _ = db; // db is needed for AbilityId but may not be used here
    Effect { ability_id, args }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    /// Helper to create a simple AbilityId with empty module path
    fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
        AbilityId::source(db, Symbol::new(name))
    }

    #[test]
    fn test_find_by_name() {
        let db = test_db();
        let int_ty = crate::ast::Type::new(&db, crate::ast::TypeKind::Int);
        let float_ty = crate::ast::Type::new(&db, crate::ast::TypeKind::Float);
        let state_id = test_ability_id(&db, "State");
        let console_id = test_ability_id(&db, "Console");

        let state_int = parameterized_effect(&db, state_id, vec![int_ty]);
        let state_float = parameterized_effect(&db, state_id, vec![float_ty]);
        let console = simple_effect(&db, console_id);

        let row = EffectRow::new(
            &db,
            vec![state_int.clone(), state_float.clone(), console.clone()],
            None,
        );

        let state_effects = find_by_name(&db, row, &Symbol::new("State"));
        assert_eq!(state_effects.len(), 2);
        assert!(state_effects.contains(&state_int));
        assert!(state_effects.contains(&state_float));

        let console_effects = find_by_name(&db, row, &Symbol::new("Console"));
        assert_eq!(console_effects.len(), 1);
    }

    // =========================================================================
    // AbilityId module path tests
    // =========================================================================

    /// Helper to create an AbilityId with a specific module path
    fn ability_id_with_path<'db>(
        db: &'db dyn salsa::Database,
        module_path: &[&str],
        name: &str,
    ) -> AbilityId<'db> {
        let mut qualified = module_path.join("::");
        if !qualified.is_empty() {
            qualified.push_str("::");
        }
        qualified.push_str(name);
        AbilityId::source(db, Symbol::new(&qualified))
    }

    #[test]
    fn test_same_name_different_module_are_distinct() {
        // mod1::State and mod2::State should be different abilities
        let db = test_db();

        let mod1_state = ability_id_with_path(&db, &["mod1"], "State");
        let mod2_state = ability_id_with_path(&db, &["mod2"], "State");

        // AbilityIds should be different
        assert_ne!(mod1_state, mod2_state);

        // Effects with these AbilityIds should be different
        let effect1 = simple_effect(&db, mod1_state);
        let effect2 = simple_effect(&db, mod2_state);
        assert_ne!(effect1, effect2);
    }

    #[test]
    fn test_qualified_name_display() {
        let db = test_db();

        // Test simple ability name
        let simple = test_ability_id(&db, "Console");
        assert_eq!(format!("{}", simple.qualified(&db)), "Console");

        // Test ability with module path
        let qualified = ability_id_with_path(&db, &["std", "io"], "Console");
        assert_eq!(format!("{}", qualified.qualified(&db)), "std::io::Console");
    }
}

/// Set laws of row operations, checked on generated rows. Generated rows
/// hold each effect once; effect identity covers the ability's origin and
/// module and every type argument (see `typeck::prop`).
#[cfg(test)]
mod laws {
    use proptest::prelude::*;

    use super::*;
    use crate::ast::RowUnion;
    use crate::typeck::prop::{EffectShape, RowShape, TypeGen, effect_shape, row_shape};

    const ROWS: TypeGen = TypeGen::GROUND.univars(2).row_vars(3);

    /// Fresh variables for union results, clear of generated tails.
    struct Fresh(u64);

    impl Fresh {
        fn new() -> Self {
            Self(1_000_000)
        }

        fn var(&mut self) -> EffectVar {
            self.0 += 1;
            EffectVar { id: self.0 }
        }

        fn used(&self) -> u64 {
            self.0 - 1_000_000
        }
    }

    fn has_effect(row: &RowShape, effect: &EffectShape) -> bool {
        row.effects.contains(effect)
    }

    /// The shape of `left ∪ right`: left's effects, then right's new ones.
    fn union_model(left: &RowShape, right: &RowShape) -> Vec<EffectShape> {
        let mut effects = left.effects.clone();
        effects.extend(
            right
                .effects
                .iter()
                .filter(|effect| !has_effect(left, effect))
                .cloned(),
        );
        effects
    }

    fn set_eq<'db>(
        db: &'db dyn salsa::Database,
        left: EffectRow<'db>,
        right: EffectRow<'db>,
    ) -> bool {
        let (left, right) = (left.effects(db), right.effects(db));
        left.iter().all(|effect| right.contains(effect))
            && right.iter().all(|effect| left.contains(effect))
    }

    proptest! {
        /// Union lists the left row's effects, then the right row's new
        /// ones, each once. A shared or single tail is kept; two distinct
        /// tails get one fresh tail and a retained `RowUnion` of the sources.
        #[test]
        fn union_matches_its_set_model(left in row_shape(ROWS), right in row_shape(ROWS)) {
            let db = salsa::DatabaseImpl::new();
            let (left_row, right_row) = (left.build(&db), right.build(&db));
            let mut fresh = Fresh::new();
            let (result, relation) = union(&db, left_row, right_row, || fresh.var());
            let effects: Vec<_> = union_model(&left, &right).iter().map(|e| e.build(&db)).collect();
            prop_assert_eq!(result.effects(&db), effects.as_slice());
            match (left.rest, right.rest) {
                (Some(a), Some(b)) if a != b => {
                    prop_assert_eq!(fresh.used(), 1);
                    prop_assert_eq!(result.rest(&db), Some(EffectVar { id: fresh.0 }));
                    prop_assert_eq!(
                        relation,
                        Some(RowUnion { sources: vec![left_row, right_row], result })
                    );
                }
                (a, b) => {
                    prop_assert_eq!(fresh.used(), 0);
                    prop_assert_eq!(result.rest(&db), a.or(b).map(|id| EffectVar { id }));
                    prop_assert_eq!(relation, None);
                }
            }
        }

        /// The closed empty row is the identity of union, and union is
        /// idempotent.
        #[test]
        fn union_has_identity_and_is_idempotent(row in row_shape(ROWS)) {
            let db = salsa::DatabaseImpl::new();
            let row = row.build(&db);
            let pure = EffectRow::pure(&db);
            let no_fresh = || -> EffectVar { panic!("no fresh variable needed") };
            prop_assert_eq!(union(&db, row, pure, no_fresh), (row, None));
            prop_assert_eq!(union(&db, pure, row, no_fresh), (row, None));
            prop_assert_eq!(union(&db, row, row, no_fresh), (row, None));
        }

        /// Union is commutative and associative on effect sets and on
        /// whether the result is open.
        #[test]
        fn union_is_commutative_and_associative(
            a in row_shape(ROWS),
            b in row_shape(ROWS),
            c in row_shape(ROWS),
        ) {
            let db = salsa::DatabaseImpl::new();
            let (a, b, c) = (a.build(&db), b.build(&db), c.build(&db));
            let mut fresh = Fresh::new();
            let mut join = |x, y| union(&db, x, y, || fresh.var()).0;
            let (ab, ba) = (join(a, b), join(b, a));
            prop_assert!(set_eq(&db, ab, ba));
            prop_assert_eq!(ab.rest(&db).is_some(), ba.rest(&db).is_some());
            if a.rest(&db) == b.rest(&db) || a.rest(&db).is_none() || b.rest(&db).is_none() {
                prop_assert_eq!(ab.rest(&db), ba.rest(&db));
            }
            let left = join(ab, c);
            let bc = join(b, c);
            let right = join(a, bc);
            prop_assert!(set_eq(&db, left, right));
            prop_assert_eq!(left.rest(&db).is_some(), right.rest(&db).is_some());
        }

        /// Adding an effect makes the row contain it; adding a present
        /// effect changes nothing; removing an added absent effect restores
        /// the row; removing keeps every other effect and the tail.
        #[test]
        fn add_and_remove_are_inverse(row in row_shape(ROWS), effect in effect_shape(ROWS)) {
            let db = salsa::DatabaseImpl::new();
            let present = has_effect(&row, &effect);
            let (row, effect) = (row.build(&db), effect.build(&db));
            let added = add_effect(&db, row, effect.clone());
            prop_assert!(contains(&db, added, &effect));
            prop_assert_eq!(contains(&db, row, &effect), present);
            if present {
                prop_assert_eq!(added, row);
                let removed = remove_effect(&db, row, &effect).expect("present effect");
                prop_assert!(!contains(&db, removed, &effect));
                prop_assert_eq!(removed.rest(&db), row.rest(&db));
                prop_assert_eq!(removed.effects(&db).len() + 1, row.effects(&db).len());
                prop_assert!(row.effects(&db).iter().all(|e| *e == effect || contains(&db, removed, e)));
                prop_assert!(set_eq(&db, add_effect(&db, removed, effect.clone()), row));
            } else {
                prop_assert_eq!(remove_effect(&db, row, &effect), None);
                prop_assert_eq!(remove_effect(&db, added, &effect), Some(row));
            }
        }

        /// Handler removal: a present effect is removed directly; an absent
        /// one decomposes the tail of an open row with one fresh variable,
        /// and is not found in a closed row.
        #[test]
        fn remove_with_constraint_classifies_rows(row in row_shape(ROWS), effect in effect_shape(ROWS)) {
            let db = salsa::DatabaseImpl::new();
            let present = has_effect(&row, &effect);
            let tail = row.rest;
            let (row, effect) = (row.build(&db), effect.build(&db));
            let mut fresh = Fresh::new();
            let result = remove_with_constraint(&db, row, &effect, || fresh.var());
            match (present, tail) {
                (true, _) => {
                    prop_assert_eq!(fresh.used(), 0);
                    prop_assert_eq!(
                        result,
                        RemoveResult::Removed(remove_effect(&db, row, &effect).unwrap())
                    );
                }
                (false, Some(id)) => {
                    prop_assert_eq!(fresh.used(), 1);
                    prop_assert_eq!(
                        result,
                        RemoveResult::NeedsConstraint {
                            var: EffectVar { id },
                            must_contain: effect,
                            remainder: EffectVar { id: fresh.0 },
                        }
                    );
                }
                (false, None) => {
                    prop_assert_eq!(fresh.used(), 0);
                    prop_assert_eq!(result, RemoveResult::NotFound);
                }
            }
        }

        /// A row holding each effect once has no conflict, whatever the
        /// abilities' names; repeating one effect is reported with both
        /// instances.
        #[test]
        fn conflicts_are_exactly_repeated_effects(
            row in row_shape(ROWS),
            pick in any::<proptest::sample::Index>(),
        ) {
            let db = salsa::DatabaseImpl::new();
            prop_assert_eq!(find_conflicting_effects(&db, row.build(&db)), None);
            if !row.effects.is_empty() {
                let repeated = row.effects[pick.index(row.effects.len())].clone();
                let mut effects = row.effects.clone();
                effects.push(repeated.clone());
                let doubled = RowShape { effects, rest: row.rest }.build(&db);
                let repeated = repeated.build(&db);
                prop_assert_eq!(
                    find_conflicting_effects(&db, doubled),
                    Some((repeated.ability_id, vec![repeated.clone(), repeated]))
                );
            }
        }
    }
}
