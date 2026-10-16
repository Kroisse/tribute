//! Calling-convention requirements derived from source effect rows.

use rustc_hash::FxHashMap as HashMap;

use super::{AbilityId, AbilityRow, OpDeclKind, RowVar, Type, TypeKind};
pub use tribute_core::CallingConvention;

/// The convention class of each class variable of one function instance.
pub type RowClasses = [(RowVar, CallingConvention)];

/// The convention each ability requires of a callable whose row names it:
/// `Cps` when it declares an `op`, `EvidenceDirect` otherwise.
pub fn ability_conventions<'db>(
    abilities: impl IntoIterator<Item = (AbilityId<'db>, impl IntoIterator<Item = OpDeclKind>)>,
) -> HashMap<AbilityId<'db>, CallingConvention> {
    abilities
        .into_iter()
        .map(|(ability, operations)| {
            let convention = if operations.into_iter().any(|kind| kind == OpDeclKind::Op) {
                CallingConvention::Cps
            } else {
                CallingConvention::EvidenceDirect
            };
            (ability, convention)
        })
        .collect()
}

/// Derive a convention from an ability row and ability-level requirements.
///
/// Unknown abilities and open row tails conservatively require CPS.
pub fn calling_convention_for_ability_row<'db>(
    db: &'db dyn salsa::Database,
    row: AbilityRow<'db>,
    abilities: &HashMap<AbilityId<'db>, CallingConvention>,
) -> CallingConvention {
    calling_convention_for_ability_row_in(db, row, abilities, &[])
}

/// Derive a convention from an ability row inside a function instance. A tail
/// that is a class variable of the instance requires its class; any other
/// tail requires CPS.
pub fn calling_convention_for_ability_row_in<'db>(
    db: &'db dyn salsa::Database,
    row: AbilityRow<'db>,
    abilities: &HashMap<AbilityId<'db>, CallingConvention>,
    classes: &RowClasses,
) -> CallingConvention {
    let mut convention = CallingConvention::Direct;
    for effect in row.instances(db) {
        let requirement = abilities
            .get(&effect.ability_id)
            .copied()
            .unwrap_or(CallingConvention::Cps);
        convention = convention.join(requirement);
    }
    if let Some(tail) = row.rest(db) {
        let class = classes
            .iter()
            .find(|(var, _)| *var == tail)
            .map_or(CallingConvention::Cps, |(_, class)| *class);
        convention = convention.join(class);
    }
    convention
}

/// Derive a convention for a function type from its ability row.
pub fn calling_convention_for_function_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    abilities: &HashMap<AbilityId<'db>, CallingConvention>,
) -> Option<CallingConvention> {
    calling_convention_for_function_type_in(db, ty, abilities, &[])
}

/// Derive a convention for a function type inside a function instance.
pub fn calling_convention_for_function_type_in<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    abilities: &HashMap<AbilityId<'db>, CallingConvention>,
    classes: &RowClasses,
) -> Option<CallingConvention> {
    let TypeKind::Func { effect, .. } = ty.kind(db) else {
        return None;
    };
    Some(calling_convention_for_ability_row_in(
        db, *effect, abilities, classes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityInstance, RowVar};

    #[test]
    fn joins_to_the_strongest_requirement() {
        assert_eq!(
            CallingConvention::Direct.join(CallingConvention::EvidenceDirect),
            CallingConvention::EvidenceDirect
        );
        assert_eq!(
            CallingConvention::EvidenceDirect.join(CallingConvention::Cps),
            CallingConvention::Cps
        );
    }

    #[test]
    fn ability_rows_use_ability_level_upper_bounds() {
        let db = salsa::DatabaseImpl::new();
        let logger = AbilityId::source(&db, trunk_ir::SymbolPath::new(["Logger"]));
        let state = AbilityId::source(&db, trunk_ir::SymbolPath::new(["State"]));
        let mut abilities = HashMap::default();
        abilities.insert(logger, CallingConvention::EvidenceDirect);
        abilities.insert(state, CallingConvention::Cps);

        let logger_effect = AbilityInstance {
            ability_id: logger,
            args: vec![],
        };
        let state_effect = AbilityInstance {
            ability_id: state,
            args: vec![],
        };
        let logger_only = AbilityRow::new(&db, vec![logger_effect.clone()], None);
        let mixed = AbilityRow::new(&db, vec![logger_effect, state_effect], None);

        assert_eq!(
            calling_convention_for_ability_row(&db, AbilityRow::pure(&db), &abilities),
            CallingConvention::Direct
        );

        assert_eq!(
            calling_convention_for_ability_row(&db, logger_only, &abilities),
            CallingConvention::EvidenceDirect
        );
        assert_eq!(
            calling_convention_for_ability_row(&db, mixed, &abilities),
            CallingConvention::Cps
        );
    }

    #[test]
    fn open_and_unknown_rows_are_cps() {
        let db = salsa::DatabaseImpl::new();
        let unknown = AbilityId::source(&db, trunk_ir::SymbolPath::new(["Unknown"]));
        let unknown_row = AbilityRow::new(
            &db,
            vec![AbilityInstance {
                ability_id: unknown,
                args: vec![],
            }],
            None,
        );
        let open_row = AbilityRow::open(&db, RowVar { id: 0 });
        let abilities = HashMap::default();

        assert_eq!(
            calling_convention_for_ability_row(&db, unknown_row, &abilities),
            CallingConvention::Cps
        );
        assert_eq!(
            calling_convention_for_ability_row(&db, open_row, &abilities),
            CallingConvention::Cps
        );
    }
}
