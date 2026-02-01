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
    name: Symbol,
) -> bool {
    row.effects(db)
        .iter()
        .any(|e| e.ability_id.name(db) == name)
}

/// Find all effects matching a given name (ignoring type parameters).
///
/// This is used for handler pattern matching where the pattern specifies
/// only the ability name but we need to find the fully parameterized ability.
pub fn find_by_name<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    name: Symbol,
) -> Vec<Effect<'db>> {
    row.effects(db)
        .iter()
        .filter(|e| e.ability_id.name(db) == name)
        .cloned()
        .collect()
}

/// Add an effect to an effect row, returning a new row.
pub fn add_effect<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    effect: Effect<'db>,
) -> EffectRow<'db> {
    let mut effects = row.effects(db).clone();
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
        let mut new_effects = effects.clone();
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
) -> EffectRow<'db> {
    let mut effects = row1.effects(db).clone();
    for effect in row2.effects(db) {
        if !effects.contains(effect) {
            effects.push(effect.clone());
        }
    }

    let rest = match (row1.rest(db), row2.rest(db)) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(_), Some(_)) => Some(fresh_var()),
    };

    EffectRow::new(db, effects, rest)
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
    use std::collections::HashSet;

    let effects = row.effects(db);
    let mut seen: HashSet<&Effect<'db>> = HashSet::new();

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
    use trunk_ir::SymbolVec;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    /// Helper to create a simple AbilityId with empty module path
    fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
        AbilityId::new(db, SymbolVec::new(), Symbol::from_dynamic(name))
    }

    #[test]
    fn test_simple_effect_equality() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let console1 = simple_effect(&db, console_id);
        let console2 = simple_effect(&db, console_id);
        assert_eq!(console1, console2);
    }

    #[test]
    fn test_parameterized_effects_are_distinct() {
        let db = test_db();
        let int_ty = crate::ast::Type::new(&db, crate::ast::TypeKind::Int);
        let float_ty = crate::ast::Type::new(&db, crate::ast::TypeKind::Float);
        let state_id = test_ability_id(&db, "State");

        let state_int = parameterized_effect(&db, state_id, vec![int_ty]);
        let state_float = parameterized_effect(&db, state_id, vec![float_ty]);

        assert_ne!(state_int, state_float);
    }

    #[test]
    fn test_effect_row_contains() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let row = EffectRow::new(&db, vec![console.clone()], None);

        assert!(contains(&db, row, &console));
        assert!(!contains(&db, row, &io));
    }

    #[test]
    fn test_effect_row_add() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let row = EffectRow::new(&db, vec![console.clone()], None);
        let new_row = add_effect(&db, row, io.clone());

        assert!(contains(&db, new_row, &console));
        assert!(contains(&db, new_row, &io));
    }

    #[test]
    fn test_effect_row_remove() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let row = EffectRow::new(&db, vec![console.clone(), io.clone()], None);
        let new_row = remove_effect(&db, row, &console).unwrap();

        assert!(!contains(&db, new_row, &console));
        assert!(contains(&db, new_row, &io));
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

        let state_effects = find_by_name(&db, row, Symbol::new("State"));
        assert_eq!(state_effects.len(), 2);
        assert!(state_effects.contains(&state_int));
        assert!(state_effects.contains(&state_float));

        let console_effects = find_by_name(&db, row, Symbol::new("Console"));
        assert_eq!(console_effects.len(), 1);
    }

    #[test]
    fn test_find_conflicting_effects_with_different_type_args() {
        let db = test_db();
        let int_ty = crate::ast::Type::new(&db, crate::ast::TypeKind::Int);
        let float_ty = crate::ast::Type::new(&db, crate::ast::TypeKind::Float);
        let state_id = test_ability_id(&db, "State");
        let console_id = test_ability_id(&db, "Console");

        let state_int = parameterized_effect(&db, state_id, vec![int_ty]);
        let state_float = parameterized_effect(&db, state_id, vec![float_ty]);

        // State(Int) + State(Float) = OK (different effects, not a conflict)
        let row = EffectRow::new(&db, vec![state_int.clone(), state_float.clone()], None);
        assert!(find_conflicting_effects(&db, row).is_none());

        // State(Int) + Console = OK
        let console = simple_effect(&db, console_id);
        let row_ok = EffectRow::new(&db, vec![state_int.clone(), console], None);
        assert!(find_conflicting_effects(&db, row_ok).is_none());
    }

    #[test]
    fn test_find_conflicting_effects_with_duplicates() {
        let db = test_db();
        let int_ty = crate::ast::Type::new(&db, crate::ast::TypeKind::Int);
        let state_id = test_ability_id(&db, "State");
        let console_id = test_ability_id(&db, "Console");

        let state_int = parameterized_effect(&db, state_id, vec![int_ty]);

        // State(Int) + State(Int) = conflict (duplicate)
        let row_conflict = EffectRow::new(&db, vec![state_int.clone(), state_int.clone()], None);
        let conflict = find_conflicting_effects(&db, row_conflict);
        assert!(conflict.is_some());
        let (ability_id, effects) = conflict.unwrap();
        assert_eq!(ability_id.name(&db), Symbol::new("State"));
        assert_eq!(effects.len(), 2);

        // Console + Console = conflict (duplicate)
        let console = simple_effect(&db, console_id);
        let row_console = EffectRow::new(&db, vec![console.clone(), console.clone()], None);
        let console_conflict = find_conflicting_effects(&db, row_console);
        assert!(console_conflict.is_some());
    }

    #[test]
    fn test_union_closed_rows() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let row1 = EffectRow::new(&db, vec![console.clone()], None);
        let row2 = EffectRow::new(&db, vec![io.clone()], None);

        let mut counter = 0u64;
        let union_row = union(&db, row1, row2, || {
            counter += 1;
            EffectVar { id: counter }
        });

        assert!(contains(&db, union_row, &console));
        assert!(contains(&db, union_row, &io));
        assert!(union_row.rest(&db).is_none());
        assert_eq!(counter, 0); // No fresh var needed
    }

    #[test]
    fn test_union_open_rows() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let row1 = EffectRow::new(&db, vec![console.clone()], Some(EffectVar { id: 1 }));
        let row2 = EffectRow::new(&db, vec![io.clone()], Some(EffectVar { id: 2 }));

        let mut counter = 100u64;
        let union_row = union(&db, row1, row2, || {
            counter += 1;
            EffectVar { id: counter }
        });

        assert!(contains(&db, union_row, &console));
        assert!(contains(&db, union_row, &io));
        assert_eq!(union_row.rest(&db), Some(EffectVar { id: 101 }));
    }

    #[test]
    fn test_remove_with_constraint_direct() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let row = EffectRow::new(&db, vec![console.clone(), io.clone()], None);

        let mut counter = 0u64;
        let result = remove_with_constraint(&db, row, &console, || {
            counter += 1;
            EffectVar { id: counter }
        });

        match result {
            RemoveResult::Removed(new_row) => {
                assert!(!contains(&db, new_row, &console));
                assert!(contains(&db, new_row, &io));
            }
            _ => panic!("Expected Removed"),
        }
        assert_eq!(counter, 0);
    }

    #[test]
    fn test_remove_with_constraint_needs_constraint() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let tail = EffectVar { id: 42 };
        let row = EffectRow::new(&db, vec![console.clone()], Some(tail));

        let mut counter = 100u64;
        let result = remove_with_constraint(&db, row, &io, || {
            counter += 1;
            EffectVar { id: counter }
        });

        match result {
            RemoveResult::NeedsConstraint {
                var,
                must_contain,
                remainder,
            } => {
                assert_eq!(var, tail);
                assert_eq!(must_contain, io);
                assert_eq!(remainder, EffectVar { id: 101 });
            }
            _ => panic!("Expected NeedsConstraint"),
        }
    }

    #[test]
    fn test_remove_with_constraint_not_found() {
        let db = test_db();
        let console_id = test_ability_id(&db, "Console");
        let io_id = test_ability_id(&db, "IO");
        let console = simple_effect(&db, console_id);
        let io = simple_effect(&db, io_id);

        let row = EffectRow::new(&db, vec![console.clone()], None);

        let mut counter = 0u64;
        let result = remove_with_constraint(&db, row, &io, || {
            counter += 1;
            EffectVar { id: counter }
        });

        assert!(matches!(result, RemoveResult::NotFound));
        assert_eq!(counter, 0);
    }
}
