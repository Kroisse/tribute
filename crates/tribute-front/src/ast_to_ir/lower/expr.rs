//! Constructor coercion.

use super::IrBuilder;
use tribute_ir::dialect::adt::layout::get_enum_variants;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Location;

/// The position and field types of the variant `variant` of the enum layout
/// `enum_ty`, which the frontend built with the source's variant names.
pub(super) fn constructor_variant(
    ir: &IrContext,
    enum_ty: TypeRef,
    variant: &Symbol,
) -> (u32, Vec<TypeRef>) {
    get_enum_variants(ir, enum_ty)
        .and_then(|variants| {
            variants
                .into_iter()
                .enumerate()
                .find_map(|(position, (tag, fields))| {
                    (variant == ir.str(tag)).then_some((position as u32, fields))
                })
        })
        .expect("resolved constructor must exist in enum metadata")
}

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
    let (_, field_types) = constructor_variant(builder.ir, enum_ty, variant);

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
