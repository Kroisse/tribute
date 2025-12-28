//! Effect row representation for row-polymorphic effect typing.
//!
//! Effect rows represent the set of effects a function may perform:
//!
//! ```text
//! Row ::= {}                    -- empty row (pure)
//!       | {A₁, A₂, ..., Aₙ}     -- concrete abilities
//!       | {e}                   -- row variable
//!       | {A₁, ..., Aₙ, e}      -- concrete + row variable
//!       | {e₁, e₂}              -- row variable union
//! ```
//!
//! This module provides:
//! - `EffectRow<'db>`: Type checker's internal representation
//! - Conversion to/from `core::EffectRowType<'db>` (IR type)

use std::collections::BTreeSet;

use trunk_ir::dialect::core::{AbilityRefType, EffectRowType};
use trunk_ir::{DialectType, IdVec, Symbol, Type};

/// A row variable identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RowVar(pub u64);

impl RowVar {
    /// Create a new row variable with the given ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// An ability reference in an effect row.
///
/// Abilities are identified by their name and optional type parameters.
/// For example: `State(Int)`, `Console`, `Http`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbilityRef<'db> {
    /// The ability name (e.g., "State", "Console").
    pub name: Symbol,
    /// Type parameters (e.g., `Int` in `State(Int)`).
    pub params: IdVec<Type<'db>>,
}

// Manual Ord implementation for AbilityRef, ordering by name and type parameters.
// This ensures `State(Int)` and `State(String)` are treated as distinct abilities.
impl<'db> PartialOrd for AbilityRef<'db> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'db> Ord for AbilityRef<'db> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // Compare by name first, then by type parameters
        // This ensures consistent ordering with Eq (name + params must match)
        match self.name.cmp(&other.name) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Compare params lexicographically using Type's Ord implementation
        let len_cmp = self.params.len().cmp(&other.params.len());
        for (a, b) in self.params.iter().zip(other.params.iter()) {
            match a.cmp(b) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        len_cmp
    }
}

impl<'db> AbilityRef<'db> {
    /// Create a new ability reference.
    pub fn new(name: Symbol, params: IdVec<Type<'db>>) -> Self {
        Self { name, params }
    }

    /// Create a simple ability reference with no type parameters.
    pub fn simple(name: Symbol) -> Self {
        Self {
            name,
            params: IdVec::new(),
        }
    }

    /// Convert from an IR type to an AbilityRef.
    ///
    /// Returns `None` if the type is not a `core.ability_ref`.
    pub fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self> {
        let ability_ref = AbilityRefType::from_type(db, ty)?;
        let name = ability_ref.name(db)?;
        let params: IdVec<Type<'db>> = ability_ref.params(db).iter().copied().collect();
        Some(Self { name, params })
    }

    /// Convert this AbilityRef to an IR type.
    pub fn to_type(&self, db: &'db dyn salsa::Database) -> Type<'db> {
        if self.params.is_empty() {
            AbilityRefType::simple(db, self.name).as_type()
        } else {
            AbilityRefType::with_params(db, self.name, self.params.clone()).as_type()
        }
    }
}

/// An effect row representing a set of abilities.
///
/// The row is represented as a set of concrete abilities plus an optional
/// row variable (tail) that represents "the rest" of the effects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectRow<'db> {
    /// Concrete abilities in this row.
    abilities: BTreeSet<AbilityRef<'db>>,
    /// Optional row variable tail (represents additional unknown effects).
    tail: Option<RowVar>,
}

impl<'db> EffectRow<'db> {
    /// Create an empty effect row (pure function).
    pub fn empty() -> Self {
        Self {
            abilities: BTreeSet::new(),
            tail: None,
        }
    }

    /// Create an effect row with just a row variable (polymorphic).
    pub fn var(v: RowVar) -> Self {
        Self {
            abilities: BTreeSet::new(),
            tail: Some(v),
        }
    }

    /// Create an effect row with concrete abilities and no tail.
    pub fn concrete(abilities: impl IntoIterator<Item = AbilityRef<'db>>) -> Self {
        Self {
            abilities: abilities.into_iter().collect(),
            tail: None,
        }
    }

    /// Create an effect row with concrete abilities and a row variable tail.
    pub fn with_tail(abilities: impl IntoIterator<Item = AbilityRef<'db>>, tail: RowVar) -> Self {
        Self {
            abilities: abilities.into_iter().collect(),
            tail: Some(tail),
        }
    }

    /// Check if this row is empty (pure).
    pub fn is_empty(&self) -> bool {
        self.abilities.is_empty() && self.tail.is_none()
    }

    /// Check if this row is a single row variable.
    pub fn is_var(&self) -> bool {
        self.abilities.is_empty() && self.tail.is_some()
    }

    /// Get the concrete abilities in this row.
    pub fn abilities(&self) -> &BTreeSet<AbilityRef<'db>> {
        &self.abilities
    }

    /// Get the row variable tail, if any.
    pub fn tail(&self) -> Option<RowVar> {
        self.tail
    }

    /// Check if this row contains a specific ability.
    pub fn contains(&self, ability: &AbilityRef<'db>) -> bool {
        self.abilities.contains(ability)
    }

    /// Add an ability to this row.
    pub fn add_ability(&mut self, ability: AbilityRef<'db>) {
        self.abilities.insert(ability);
    }

    /// Remove an ability from this row.
    ///
    /// Returns `true` if the ability was present and removed.
    pub fn remove_ability(&mut self, ability: &AbilityRef<'db>) -> bool {
        self.abilities.remove(ability)
    }

    /// Set the row variable tail.
    pub fn set_tail(&mut self, tail: Option<RowVar>) {
        self.tail = tail;
    }

    /// Union two effect rows, combining their abilities.
    ///
    /// If both rows have tails, creates a fresh row variable for the result.
    /// If only one has a tail, uses that tail.
    /// If neither has a tail, the result has no tail.
    pub fn union(&self, other: &Self, fresh_var: impl FnOnce() -> RowVar) -> Self {
        let mut abilities = self.abilities.clone();
        abilities.extend(other.abilities.iter().cloned());

        let tail = match (self.tail, other.tail) {
            (None, None) => None,
            (Some(v), None) | (None, Some(v)) => Some(v),
            (Some(_), Some(_)) => Some(fresh_var()),
        };

        Self { abilities, tail }
    }

    /// Check for duplicate abilities (same ability appearing twice).
    ///
    /// Returns `Some(ability)` if a duplicate is found.
    /// Note: This checks within `self`, not between two rows.
    pub fn find_duplicate(&self) -> Option<&AbilityRef<'db>> {
        // BTreeSet prevents duplicates, so we just need to check
        // if any ability appears with different type parameters
        // This is already prevented by the AbilityRef Eq implementation
        None
    }
}

impl<'db> EffectRow<'db> {
    /// Convert from an IR type to an EffectRow.
    ///
    /// Returns `None` if the type is not a `core.effect_row`.
    pub fn from_type(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<Self> {
        let effect_row = EffectRowType::from_type(db, ty)?;

        // Convert abilities
        let abilities: BTreeSet<AbilityRef<'db>> = effect_row
            .abilities(db)
            .iter()
            .filter_map(|&ability_ty| AbilityRef::from_type(db, ability_ty))
            .collect();

        // Get tail variable
        let tail = effect_row.tail_var(db).map(RowVar);

        Some(Self { abilities, tail })
    }

    /// Convert this EffectRow to an IR type.
    pub fn to_type(&self, db: &'db dyn salsa::Database) -> Type<'db> {
        let abilities: IdVec<Type<'db>> = self
            .abilities
            .iter()
            .map(|ability| ability.to_type(db))
            .collect();

        match self.tail {
            Some(RowVar(id)) => EffectRowType::with_tail(db, abilities, id).as_type(),
            None => {
                if abilities.is_empty() {
                    EffectRowType::empty(db).as_type()
                } else {
                    EffectRowType::concrete(db, abilities).as_type()
                }
            }
        }
    }
}

impl Default for EffectRow<'_> {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::core;

    #[salsa_test]
    fn test_parameterized_abilities_are_distinct(db: &salsa::DatabaseImpl) {
        // State(I64) and State(F64) should be treated as distinct abilities
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);
        let float_ty = *core::F64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let state_float = AbilityRef::new(state_name, IdVec::from(vec![float_ty]));

        // They should not be equal
        assert_ne!(state_int, state_float);

        // They should have different orderings
        assert_ne!(state_int.cmp(&state_float), std::cmp::Ordering::Equal);
    }

    #[salsa_test]
    fn test_parameterized_abilities_coexist_in_row(db: &salsa::DatabaseImpl) {
        // Both State(I64) and State(F64) should be able to coexist in the same row
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);
        let float_ty = *core::F64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let state_float = AbilityRef::new(state_name, IdVec::from(vec![float_ty]));

        let row = EffectRow::concrete([state_int.clone(), state_float.clone()]);

        // Both abilities should be in the row
        assert_eq!(row.abilities().len(), 2);
        assert!(row.contains(&state_int));
        assert!(row.contains(&state_float));
    }

    #[salsa_test]
    fn test_simple_abilities_with_same_name_are_equal(_db: &salsa::DatabaseImpl) {
        // Two simple (non-parameterized) abilities with the same name should be equal
        let console1 = AbilityRef::simple(Symbol::new("Console"));
        let console2 = AbilityRef::simple(Symbol::new("Console"));

        assert_eq!(console1, console2);
        assert_eq!(console1.cmp(&console2), std::cmp::Ordering::Equal);
    }

    #[salsa_test]
    fn test_ability_ref_ordering_is_consistent_with_eq(db: &salsa::DatabaseImpl) {
        // Ord and Eq must be consistent: a.cmp(&b) == Equal iff a == b
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);

        let state_int1 = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let state_int2 = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));

        // Same abilities should be equal in both Eq and Ord
        assert_eq!(state_int1, state_int2);
        assert_eq!(state_int1.cmp(&state_int2), std::cmp::Ordering::Equal);
    }

    #[salsa_test]
    fn test_effect_row_union_preserves_parameterized_abilities(db: &salsa::DatabaseImpl) {
        // Union of rows should preserve both State(I64) and State(F64)
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);
        let float_ty = *core::F64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let state_float = AbilityRef::new(state_name, IdVec::from(vec![float_ty]));

        let row1 = EffectRow::concrete([state_int.clone()]);
        let row2 = EffectRow::concrete([state_float.clone()]);

        let mut counter = 0u64;
        let union = row1.union(&row2, || {
            counter += 1;
            RowVar(counter)
        });

        // Both abilities should be in the union
        assert_eq!(union.abilities().len(), 2);
        assert!(union.contains(&state_int));
        assert!(union.contains(&state_float));
    }
}
