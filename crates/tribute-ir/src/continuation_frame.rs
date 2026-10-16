//! Private immutable CPS continuation frames `ContinuationFrame<R>`.
//!
//! A frame is a nominal `adt` layout whose type carries its result type `R`.
//! The frame reference is an `adt.typeref` to the layout, so the layout may
//! refer to itself recursively.

use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{Attribute, AttributeMap, StringArg, TypeDataBuilder};

use crate::dialect::adt;

/// Result type carried by a continuation frame reference and its layout.
pub const RESULT_ATTR: &str = "tribute.cps_continuation_frame_result";
/// Name prefix of the compiler-generated continuation frame layouts.
pub const NAME_PREFIX: &str = "__tribute_continuation_frame_";

/// Make the nominal reference for one frame.
///
/// Its paired layout may recursively use this reference.
pub fn ref_type(ctx: &mut IrContext, name: impl Into<StringArg>, result: TypeRef) -> TypeRef {
    let name = ctx.intern_string_arg(name.into());
    TypeDataBuilder::new(crate::dialect::adt::Typeref::NAME)
        .attr("name", Attribute::String(name))
        .attr(RESULT_ATTR, Attribute::Type(result))
        .intern(ctx)
}

/// Read the result type only from an explicit frame reference or layout.
pub fn result_type(ctx: &IrContext, frame: TypeRef) -> Option<TypeRef> {
    use trunk_ir::ops::DialectType;
    (adt::Typeref::matches(ctx, frame) || adt::Struct::matches(ctx, frame))
        .then(|| ctx.type_attrs(frame).get_type(RESULT_ATTR))
        .flatten()
}

/// Make the exact immutable layout for [`ref_type`].
pub fn layout_type(
    ctx: &mut IrContext,
    name: impl Into<StringArg>,
    result: TypeRef,
    done: TypeRef,
    dispatch: TypeRef,
) -> TypeRef {
    let mut attrs = AttributeMap::new();
    attrs
        .view_mut(ctx)
        .insert(RESULT_ATTR, Attribute::Type(result));
    adt::struct_type(ctx, name, [("done", done), ("dispatch", dispatch)], attrs).as_type_ref()
}

/// A frame reference that carries its result type: the frame of a CPS
/// callable between frame expansion and physicalization. `Result` is the
/// answer `R`.
pub struct FrameRef;

impl trunk_ir::type_constraint::TypeConstraint for FrameRef {
    const DESC: &'static trunk_ir::type_constraint::ConstraintDesc = {
        use trunk_ir::type_constraint::{
            ConstraintDesc, Projected, ProjectionDesc, ProjectionKind,
        };
        &ConstraintDesc {
            name: "ContinuationFrame",
            exact: false,
            projections: &[ProjectionDesc {
                name: "Result",
                kind: ProjectionKind::One,
            }],
            matches: |ctx, ty| result_type(ctx, ty).is_some(),
            project: |ctx, ty, index| match index {
                0 => result_type(ctx, ty).map(Projected::One),
                _ => None,
            },
            fixed: None,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::ops::DialectType;

    #[test]
    fn frame_reference_and_layout_carry_the_result_type() {
        let mut ctx = IrContext::new();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let done = TypeDataBuilder::new("test.done").intern(&mut ctx);
        let dispatch = TypeDataBuilder::new("test.dispatch").intern(&mut ctx);
        let name = "ContinuationFrameI32";
        let frame = ref_type(&mut ctx, name, i32_ty);
        let layout = layout_type(&mut ctx, name, i32_ty, done, dispatch);

        assert_eq!(result_type(&ctx, frame), Some(i32_ty));
        assert_eq!(result_type(&ctx, layout), Some(i32_ty));
        assert_eq!(adt::nominal_name(&ctx, frame), Some(name));
        itertools::assert_equal(
            adt::Struct::from_type_ref(&ctx, layout)
                .unwrap()
                .fields(&ctx),
            [("done", done), ("dispatch", dispatch)],
        );
    }

    #[test]
    fn result_type_fails_closed_on_unmarked_or_foreign_types() {
        let mut ctx = IrContext::new();
        let name = ctx.string_attr("ContinuationFrame");
        let unmarked = TypeDataBuilder::new(crate::dialect::adt::Typeref::NAME)
            .attr("name", name)
            .intern(&mut ctx);
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let foreign = TypeDataBuilder::new("test.frame")
            .attr(RESULT_ATTR, Attribute::Type(i32_ty))
            .intern(&mut ctx);

        assert_eq!(result_type(&ctx, unmarked), None);
        assert_eq!(result_type(&ctx, foreign), None);
    }
}
