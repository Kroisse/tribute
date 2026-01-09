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

/// Result of attempting to remove an ability from an effect row.
///
/// This enum is used by `EffectRow::remove_with_constraint()` to handle
/// row variable decomposition during handler typing.
///
/// ## Type Theory Background
///
/// In row-polymorphic effect systems, effect rows can be either:
/// - **Closed**: `{A, B, C}` - a fixed set of abilities
/// - **Open**: `{A, B | e}` - some abilities plus a row variable `e`
///
/// When a handler handles ability `A`, we need to "subtract" it from the row:
/// - From closed `{A, B}` → `{B}` (direct removal)
/// - From open `{A | e}` → `{| e}` (direct removal from concrete part)
/// - From open `{B | e}` where `A ∉ {B}` → need constraint `e = {A | e'}`
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoveResult<'db> {
    /// Ability was directly present and removed.
    /// Contains the resulting effect row after removal.
    Removed(EffectRow<'db>),

    /// Ability was not in the concrete part, but might be in the row variable tail.
    /// Generates a constraint: `var = {must_contain | remainder}`.
    NeedsConstraint {
        /// The row variable that must be decomposed.
        var: RowVar,
        /// The ability that must be contained in the row variable.
        must_contain: AbilityRef<'db>,
        /// Fresh row variable for the remainder after removing the ability.
        remainder: RowVar,
    },

    /// Ability was not found and the row is closed (no tail variable).
    /// This indicates a type error: trying to handle an ability not in the effect row.
    NotFound,
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

    /// Remove an ability with row variable decomposition support.
    ///
    /// This method handles the case where the ability might be in a row variable tail.
    /// Used by handler typing to correctly remove handled abilities from effect rows.
    ///
    /// ## Effect Row Algebra
    ///
    /// Given an effect row `{A₁, A₂, ..., Aₙ | e}` where `e` is a row variable:
    ///
    /// - If ability `A` is directly in the row: remove and return `Removed`
    /// - If ability `A` is not in the row but `e` exists: return `NeedsConstraint`
    ///   (generates constraint `e = {A | e'}` where `e'` is fresh)
    /// - If ability `A` is not in the row and no tail: return `NotFound` (type error)
    ///
    /// ## Example
    ///
    /// ```text
    /// // Handler removes State from effect row {State(Int), Console | e}
    /// row.remove_with_constraint(&State(Int), fresh_var)
    /// // → Removed({Console | e})
    ///
    /// // Handler removes Reader from effect row {State(Int) | e}
    /// // Reader is not directly present, but e might contain it
    /// row.remove_with_constraint(&Reader(Int), fresh_var)
    /// // → NeedsConstraint { var: e, must_contain: Reader(Int), remainder: e' }
    /// ```
    pub fn remove_with_constraint(
        &self,
        ability: &AbilityRef<'db>,
        fresh_var: impl FnOnce() -> RowVar,
    ) -> RemoveResult<'db> {
        if self.contains(ability) {
            // Direct removal: ability is in the concrete part of the row
            let mut result = self.clone();
            result.remove_ability(ability);
            RemoveResult::Removed(result)
        } else if let Some(tail) = self.tail {
            // Row variable decomposition: ability might be in the tail
            // Generate constraint: tail = {ability | fresh}
            let fresh = fresh_var();
            RemoveResult::NeedsConstraint {
                var: tail,
                must_contain: ability.clone(),
                remainder: fresh,
            }
        } else {
            // Closed row without the ability: type error
            RemoveResult::NotFound
        }
    }

    /// Find abilities matching a given name (ignoring type parameters).
    ///
    /// This is used for handler pattern matching where the pattern specifies
    /// only the ability name (e.g., `State::get()`) but we need to find the
    /// fully parameterized ability (e.g., `State(Int)`) in the effect row.
    ///
    /// Returns all matching abilities since multiple parameterizations of the
    /// same ability can coexist in an effect row. When the effect row contains
    /// multiple parameterizations (e.g., `State(Int)` and `State(String)`), all
    /// are returned. This means a handler pattern without explicit type params
    /// will handle all matching parameterizations.
    ///
    /// Future work: Type inference from handler bodies could disambiguate which
    /// specific parameterization is being handled.
    pub fn find_by_name(&self, name: Symbol) -> Vec<AbilityRef<'db>> {
        self.abilities
            .iter()
            .filter(|ability| ability.name == name)
            .cloned()
            .collect()
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

    /// Check for conflicting abilities (same name with different type parameters).
    ///
    /// Returns `Some((name, abilities))` if a conflict is found, where `abilities`
    /// contains all the conflicting ability references with the same name.
    ///
    /// This restriction exists because handler patterns match abilities by name only
    /// (e.g., `State::get()`), so having both `State(Int)` and `State(Text)` would
    /// be ambiguous. Future work on named effects may lift this restriction.
    pub fn find_conflicting_abilities(&self) -> Option<(Symbol, Vec<&AbilityRef<'db>>)> {
        use std::collections::HashMap;

        // Group abilities by name
        let mut by_name: HashMap<Symbol, Vec<&AbilityRef<'db>>> = HashMap::new();
        for ability in &self.abilities {
            by_name.entry(ability.name).or_default().push(ability);
        }

        // Find any name with multiple different parameterizations
        for (name, abilities) in by_name {
            if abilities.len() > 1 {
                return Some((name, abilities));
            }
        }

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

    #[salsa_test]
    fn test_find_by_name_returns_all_matching_abilities(db: &salsa::DatabaseImpl) {
        // find_by_name should return all abilities matching the name, ignoring type params
        let state_name = Symbol::new("State");
        let console_name = Symbol::new("Console");
        let int_ty = *core::I64::new(db);
        let float_ty = *core::F64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let state_float = AbilityRef::new(state_name, IdVec::from(vec![float_ty]));
        let console = AbilityRef::simple(console_name);

        let row = EffectRow::concrete([state_int.clone(), state_float.clone(), console.clone()]);

        // find_by_name("State") should return both State(I64) and State(F64)
        let state_abilities = row.find_by_name(state_name);
        assert_eq!(state_abilities.len(), 2);
        assert!(state_abilities.contains(&state_int));
        assert!(state_abilities.contains(&state_float));

        // find_by_name("Console") should return only Console
        let console_abilities = row.find_by_name(console_name);
        assert_eq!(console_abilities.len(), 1);
        assert!(console_abilities.contains(&console));

        // find_by_name("Unknown") should return empty
        let unknown_name = Symbol::new("Unknown");
        let unknown_abilities = row.find_by_name(unknown_name);
        assert!(unknown_abilities.is_empty());
    }

    #[salsa_test]
    fn test_find_by_name_handler_pattern_matching(db: &salsa::DatabaseImpl) {
        // Simulates handler pattern matching: pattern has no type params,
        // effect row has parameterized ability
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);

        // Effect row contains State(Int)
        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let row = EffectRow::concrete([state_int.clone()]);

        // Handler pattern just says "State" (no params)
        let pattern_name = state_name;

        // find_by_name should find State(Int) when looking for "State"
        let matching = row.find_by_name(pattern_name);
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0], state_int);
    }

    #[salsa_test]
    fn test_find_by_name_with_multiple_parameterizations(db: &salsa::DatabaseImpl) {
        // When effect row contains multiple parameterizations of the same ability
        // (e.g., both State(Int) and State(String)), find_by_name returns all.
        //
        // Note: In the current implementation, a handler pattern without explicit
        // type parameters (e.g., `State::get()`) will handle ALL matching abilities.
        // Future work may add type inference from handler bodies to disambiguate.
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);
        let string_ty = *core::String::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let state_string = AbilityRef::new(state_name, IdVec::from(vec![string_ty]));
        let row = EffectRow::concrete([state_int.clone(), state_string.clone()]);

        // Handler pattern just says "State" (no params)
        let matching = row.find_by_name(state_name);

        // Both parameterizations are returned
        assert_eq!(matching.len(), 2);
        assert!(matching.contains(&state_int));
        assert!(matching.contains(&state_string));
    }

    #[salsa_test]
    fn test_find_conflicting_abilities_detects_conflicts(db: &salsa::DatabaseImpl) {
        // find_conflicting_abilities should detect when the same ability name
        // appears with different type parameters
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);
        let string_ty = *core::String::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let state_string = AbilityRef::new(state_name, IdVec::from(vec![string_ty]));
        let row = EffectRow::concrete([state_int.clone(), state_string.clone()]);

        // Should detect the conflict
        let conflict = row.find_conflicting_abilities();
        assert!(conflict.is_some());

        let (conflict_name, conflicting) = conflict.unwrap();
        assert_eq!(conflict_name, state_name);
        assert_eq!(conflicting.len(), 2);
    }

    #[salsa_test]
    fn test_find_conflicting_abilities_no_conflict_with_different_abilities(
        db: &salsa::DatabaseImpl,
    ) {
        // find_conflicting_abilities should NOT report a conflict when different
        // ability names are used
        let state_name = Symbol::new("State");
        let console_name = Symbol::new("Console");
        let int_ty = *core::I64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let console = AbilityRef::simple(console_name);
        let row = EffectRow::concrete([state_int, console]);

        // Should NOT detect a conflict
        let conflict = row.find_conflicting_abilities();
        assert!(conflict.is_none());
    }

    #[salsa_test]
    fn test_find_conflicting_abilities_no_conflict_with_single_parameterization(
        db: &salsa::DatabaseImpl,
    ) {
        // find_conflicting_abilities should NOT report a conflict when only
        // one parameterization of an ability is present
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let row = EffectRow::concrete([state_int]);

        // Should NOT detect a conflict
        let conflict = row.find_conflicting_abilities();
        assert!(conflict.is_none());
    }

    // =========================================================================
    // remove_with_constraint tests (Issue #201)
    // =========================================================================

    #[salsa_test]
    fn test_remove_with_constraint_direct_removal(db: &salsa::DatabaseImpl) {
        // When ability is directly in the row, it should be removed directly
        let state_name = Symbol::new("State");
        let console_name = Symbol::new("Console");
        let int_ty = *core::I64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let console = AbilityRef::simple(console_name);
        let row = EffectRow::concrete([state_int.clone(), console.clone()]);

        let mut counter = 0u64;
        let result = row.remove_with_constraint(&state_int, || {
            counter += 1;
            RowVar(counter)
        });

        match result {
            RemoveResult::Removed(new_row) => {
                // State should be removed, Console should remain
                assert!(!new_row.contains(&state_int));
                assert!(new_row.contains(&console));
                assert_eq!(new_row.abilities().len(), 1);
            }
            _ => panic!("Expected Removed, got {:?}", result),
        }

        // fresh_var should not have been called
        assert_eq!(counter, 0);
    }

    #[salsa_test]
    fn test_remove_with_constraint_from_open_row(db: &salsa::DatabaseImpl) {
        // When ability is in the concrete part of an open row, direct removal preserves tail
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let tail_var = RowVar(42);
        let row = EffectRow::with_tail([state_int.clone()], tail_var);

        let mut counter = 0u64;
        let result = row.remove_with_constraint(&state_int, || {
            counter += 1;
            RowVar(100 + counter)
        });

        match result {
            RemoveResult::Removed(new_row) => {
                // State should be removed
                assert!(!new_row.contains(&state_int));
                // Tail should be preserved
                assert_eq!(new_row.tail(), Some(tail_var));
            }
            _ => panic!("Expected Removed, got {:?}", result),
        }

        // fresh_var should not have been called
        assert_eq!(counter, 0);
    }

    #[salsa_test]
    fn test_remove_with_constraint_needs_constraint(db: &salsa::DatabaseImpl) {
        // When ability is NOT in concrete part but row has tail, generate constraint
        let state_name = Symbol::new("State");
        let reader_name = Symbol::new("Reader");
        let int_ty = *core::I64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let reader_int = AbilityRef::new(reader_name, IdVec::from(vec![int_ty]));
        let tail_var = RowVar(42);
        // Row has State(Int) and tail e, but not Reader(Int)
        let row = EffectRow::with_tail([state_int.clone()], tail_var);

        let mut counter = 0u64;
        let result = row.remove_with_constraint(&reader_int, || {
            counter += 1;
            RowVar(100 + counter)
        });

        match result {
            RemoveResult::NeedsConstraint {
                var,
                must_contain,
                remainder,
            } => {
                // The tail variable should be decomposed
                assert_eq!(var, tail_var);
                // Reader(Int) must be contained in the tail
                assert_eq!(must_contain, reader_int);
                // Fresh variable should be generated
                assert_eq!(remainder, RowVar(101));
            }
            _ => panic!("Expected NeedsConstraint, got {:?}", result),
        }

        // fresh_var should have been called once
        assert_eq!(counter, 1);
    }

    #[salsa_test]
    fn test_remove_with_constraint_not_found(db: &salsa::DatabaseImpl) {
        // When ability is NOT in row and row is closed (no tail), return NotFound
        let state_name = Symbol::new("State");
        let reader_name = Symbol::new("Reader");
        let int_ty = *core::I64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let reader_int = AbilityRef::new(reader_name, IdVec::from(vec![int_ty]));
        // Closed row with only State(Int), not Reader(Int)
        let row = EffectRow::concrete([state_int.clone()]);

        let mut counter = 0u64;
        let result = row.remove_with_constraint(&reader_int, || {
            counter += 1;
            RowVar(counter)
        });

        match result {
            RemoveResult::NotFound => {
                // Expected - Reader is not in the closed row
            }
            _ => panic!("Expected NotFound, got {:?}", result),
        }

        // fresh_var should not have been called
        assert_eq!(counter, 0);
    }

    #[salsa_test]
    fn test_remove_with_constraint_empty_open_row(db: &salsa::DatabaseImpl) {
        // When row is just a variable {e}, removing any ability generates constraint
        let state_name = Symbol::new("State");
        let int_ty = *core::I64::new(db);

        let state_int = AbilityRef::new(state_name, IdVec::from(vec![int_ty]));
        let tail_var = RowVar(42);
        // Row is just {| e} - no concrete abilities
        let row = EffectRow::var(tail_var);

        let mut counter = 0u64;
        let result = row.remove_with_constraint(&state_int, || {
            counter += 1;
            RowVar(100 + counter)
        });

        match result {
            RemoveResult::NeedsConstraint {
                var,
                must_contain,
                remainder,
            } => {
                assert_eq!(var, tail_var);
                assert_eq!(must_contain, state_int);
                assert_eq!(remainder, RowVar(101));
            }
            _ => panic!("Expected NeedsConstraint, got {:?}", result),
        }
    }

    #[salsa_test]
    fn test_remove_with_constraint_empty_closed_row(_db: &salsa::DatabaseImpl) {
        // When row is empty and closed {}, removing any ability returns NotFound
        let state_name = Symbol::new("State");
        let state = AbilityRef::simple(state_name);
        let row = EffectRow::empty();

        let mut counter = 0u64;
        let result = row.remove_with_constraint(&state, || {
            counter += 1;
            RowVar(counter)
        });

        assert!(matches!(result, RemoveResult::NotFound));
        assert_eq!(counter, 0);
    }
}
