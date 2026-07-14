//! Calling-convention requirements derived from source effect rows.

use std::collections::HashMap;

use super::{AbilityId, EffectRow, Type, TypeKind};
pub use tribute_core::CallingConvention;

/// Derive a convention from an effect row and ability-level requirements.
///
/// Unknown abilities and open row tails conservatively require CPS.
pub fn calling_convention_for_effect_row<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    abilities: &HashMap<AbilityId<'db>, CallingConvention>,
) -> CallingConvention {
    let mut convention = CallingConvention::Direct;
    for effect in row.effects(db) {
        let requirement = abilities
            .get(&effect.ability_id)
            .copied()
            .unwrap_or(CallingConvention::Cps);
        convention = convention.join(requirement);
    }
    if row.rest(db).is_some() {
        convention = convention.join(CallingConvention::Cps);
    }
    convention
}

/// Derive a convention for a function type, including its explicit ABI lower bound.
pub fn calling_convention_for_function_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    abilities: &HashMap<AbilityId<'db>, CallingConvention>,
) -> Option<CallingConvention> {
    let TypeKind::Func {
        effect,
        minimum_convention,
        ..
    } = ty.kind(db)
    else {
        return None;
    };
    Some((*minimum_convention).join(calling_convention_for_effect_row(db, *effect, abilities)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Effect, EffectVar};
    use trunk_ir::Symbol;

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
    fn effect_rows_use_ability_level_upper_bounds() {
        let db = salsa::DatabaseImpl::new();
        let logger = AbilityId::new(&db, Symbol::new("Logger"));
        let state = AbilityId::new(&db, Symbol::new("State"));
        let mut abilities = HashMap::new();
        abilities.insert(logger, CallingConvention::EvidenceDirect);
        abilities.insert(state, CallingConvention::Cps);

        let logger_effect = Effect {
            ability_id: logger,
            args: vec![],
        };
        let state_effect = Effect {
            ability_id: state,
            args: vec![],
        };
        let logger_only = EffectRow::new(&db, vec![logger_effect.clone()], None);
        let mixed = EffectRow::new(&db, vec![logger_effect, state_effect], None);

        assert_eq!(
            calling_convention_for_effect_row(&db, EffectRow::pure(&db), &abilities),
            CallingConvention::Direct
        );

        assert_eq!(
            calling_convention_for_effect_row(&db, logger_only, &abilities),
            CallingConvention::EvidenceDirect
        );
        assert_eq!(
            calling_convention_for_effect_row(&db, mixed, &abilities),
            CallingConvention::Cps
        );
    }

    #[test]
    fn open_and_unknown_rows_are_cps() {
        let db = salsa::DatabaseImpl::new();
        let unknown = AbilityId::new(&db, Symbol::new("Unknown"));
        let unknown_row = EffectRow::new(
            &db,
            vec![Effect {
                ability_id: unknown,
                args: vec![],
            }],
            None,
        );
        let open_row = EffectRow::open(&db, EffectVar { id: 0 });
        let abilities = HashMap::new();

        assert_eq!(
            calling_convention_for_effect_row(&db, unknown_row, &abilities),
            CallingConvention::Cps
        );
        assert_eq!(
            calling_convention_for_effect_row(&db, open_row, &abilities),
            CallingConvention::Cps
        );
    }
}
