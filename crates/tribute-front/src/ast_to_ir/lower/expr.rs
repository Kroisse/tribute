//! Constructor coercion.

use super::IrBuilder;
use crate::ast_to_ir::context::IrLoweringCtx;
use tribute_ir::dialect::adt;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::types::Location;

/// The declaration position and field types of the variant `variant` of the
/// enum layout `enum_ty`.
pub(super) fn constructor_variant(
    ctx: &IrLoweringCtx<'_>,
    ir: &IrContext,
    enum_ty: TypeRef,
    variant: &Symbol,
) -> (u32, Vec<TypeRef>) {
    let position = ctx
        .variant_position(enum_ty, variant)
        .expect("resolved constructor must name a declared variant");
    let fields = adt::Enum::from_type_ref(ir, enum_ty)
        .and_then(|layout| layout.variant_fields(ir, position))
        .expect("a declared variant is in its enum layout")
        .to_vec();
    (position, fields)
}

/// Coerce constructor arguments to the representation recorded in the enum
/// layout. Generic enum fields are erased to `anyref`, so primitive payloads
/// must cross an explicit conversion boundary before `adt.variant_new`.
/// The layout's row binders take the row arguments of `value_ty`, the type
/// of the constructed value.
pub(super) fn cast_variant_args<'db>(
    builder: &mut IrBuilder<'_, 'db>,
    location: Location,
    args: Vec<ValueRef>,
    enum_ty: TypeRef,
    value_ty: TypeRef,
    variant: &Symbol,
) -> Vec<ValueRef> {
    let (_, field_types) = constructor_variant(builder.ctx, builder.ir, enum_ty, variant);
    let row_params = adt::Enum::from_type_ref(builder.ir, enum_ty)
        .expect("a constructor's layout is an adt.enum")
        .row_params(builder.ir);
    let field_types = super::case::fields_through(builder.ir, row_params, value_ty, field_types);

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
