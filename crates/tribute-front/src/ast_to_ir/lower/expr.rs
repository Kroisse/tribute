//! Constructor coercion.

use super::IrBuilder;
use tribute_ir::dialect::adt::layout::get_enum_variants;
use trunk_ir::Symbol;
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Location;

/// Coerce constructor arguments to the representation recorded in the enum
/// layout. Generic enum fields are erased to `anyref`, so primitive payloads
/// must cross an explicit conversion boundary before `adt.variant_new`.
pub(super) fn cast_variant_args<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    args: Vec<ValueRef>,
    enum_ty: TypeRef,
    variant: &Symbol,
) -> Vec<ValueRef> {
    let field_types = get_enum_variants(builder.ir, enum_ty)
        .and_then(|variants| {
            variants
                .into_iter()
                .find_map(|(tag, fields)| (variant == builder.ir.str(tag)).then_some(fields))
        })
        .expect("resolved constructor must exist in enum metadata");

    assert_eq!(
        args.len(),
        field_types.len(),
        "type checking must enforce constructor arity"
    );

    args.into_iter()
        .zip(field_types)
        .map(|(arg, field_ty)| builder.cast_if_needed(location, arg, field_ty))
        .collect()
}
