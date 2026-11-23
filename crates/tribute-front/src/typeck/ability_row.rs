//! Ability row utilities for row-polymorphic effect typing.
//!
//! This module provides the union operation on effect rows used during
//! type checking.

use crate::ast::{AbilityRow, RowVar};

/// Union two effect rows, combining their effects.
pub fn union<'db>(
    db: &'db dyn salsa::Database,
    row1: AbilityRow<'db>,
    row2: AbilityRow<'db>,
    fresh_var: impl FnOnce() -> RowVar,
) -> (AbilityRow<'db>, Option<crate::ast::RowUnion<'db>>) {
    let mut effects = row1.instances(db).to_vec();
    for effect in row2.instances(db) {
        if !effects.contains(effect) {
            effects.push(effect.clone());
        }
    }
    let independent = matches!((row1.rest(db), row2.rest(db)), (Some(a), Some(b)) if a != b);
    let rest = match (row1.rest(db), row2.rest(db)) {
        (None, None) => None,
        (Some(&v), None) | (None, Some(&v)) => Some(v),
        (Some(&a), Some(&b)) if a == b => Some(a),
        (Some(_), Some(_)) => Some(fresh_var()),
    };
    let result = AbilityRow::new(db, effects, rest);
    let constraint = independent.then(|| crate::ast::RowUnion {
        sources: vec![row1, row2],
        result,
    });
    (result, constraint)
}

#[cfg(test)]
mod tests {
    use crate::ast::{AbilityId, AbilityInstance};

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    /// Helper to create a simple AbilityId with empty module path
    fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
        AbilityId::source(db, trunk_ir::SymbolPath::new([name]))
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
        let path = trunk_ir::SymbolPath::new(module_path.iter().copied().chain([name]));
        AbilityId::source(db, path)
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
        let effect1 = AbilityInstance {
            ability_id: mod1_state,
            args: Vec::new(),
        };
        let effect2 = AbilityInstance {
            ability_id: mod2_state,
            args: Vec::new(),
        };
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
    use crate::typeck::prop::{AbilityInstanceShape, RowShape, TypeGen, row_shape};

    const ROWS: TypeGen = TypeGen::GROUND.univars(2).row_vars(3);

    /// Fresh variables for union results, clear of generated tails.
    struct Fresh(u64);

    impl Fresh {
        fn new() -> Self {
            Self(1_000_000)
        }

        fn var(&mut self) -> RowVar {
            self.0 += 1;
            RowVar { id: self.0 }
        }

        fn used(&self) -> u64 {
            self.0 - 1_000_000
        }
    }

    fn has_effect(row: &RowShape, effect: &AbilityInstanceShape) -> bool {
        row.effects.contains(effect)
    }

    /// The shape of `left ∪ right`: left's effects, then right's new ones.
    fn union_model(left: &RowShape, right: &RowShape) -> Vec<AbilityInstanceShape> {
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
        left: AbilityRow<'db>,
        right: AbilityRow<'db>,
    ) -> bool {
        let (left, right) = (left.instances(db), right.instances(db));
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
            prop_assert_eq!(result.instances(&db), effects.as_slice());
            match (left.rest, right.rest) {
                (Some(a), Some(b)) if a != b => {
                    prop_assert_eq!(fresh.used(), 1);
                    prop_assert_eq!(result.rest(&db), Some(&RowVar { id: fresh.0 }));
                    prop_assert_eq!(
                        relation,
                        Some(RowUnion { sources: vec![left_row, right_row], result })
                    );
                }
                (a, b) => {
                    prop_assert_eq!(fresh.used(), 0);
                    prop_assert_eq!(result.rest(&db).map(|var| var.id), a.or(b));
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
            let pure = AbilityRow::pure(&db);
            let no_fresh = || -> RowVar { panic!("no fresh variable needed") };
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
    }
}
