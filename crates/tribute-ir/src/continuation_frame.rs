//! Private immutable CPS continuation frames `ContinuationFrame<R>`.
//!
//! A frame is a nominal `adt` layout whose type carries its result type `R`.
//! The frame reference is an `adt.typeref` to the layout, so the layout may
//! refer to itself recursively.

use trunk_ir::Symbol;
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
    ctx.intern_type(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
            .attr("name", Attribute::String(name))
            .attr(RESULT_ATTR, Attribute::Type(result))
            .build(),
    )
}

/// Read the result type only from an explicit frame reference or layout.
pub fn result_type(ctx: &IrContext, frame: TypeRef) -> Option<TypeRef> {
    let data = ctx.get_type(frame);
    (data.dialect == "adt" && (data.name == "typeref" || data.name == "struct"))
        .then(|| data.attrs.get_type(RESULT_ATTR))
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
    attrs.insert(RESULT_ATTR, Attribute::Type(result));
    adt::struct_type(ctx, name, [("done", done), ("dispatch", dispatch)], attrs).as_type_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::ops::DialectType;

    #[test]
    fn frame_reference_and_layout_carry_the_result_type() {
        let mut ctx = IrContext::new();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let done = ctx.intern_type(TypeDataBuilder::new("test", "done").build());
        let dispatch = ctx.intern_type(TypeDataBuilder::new("test", "dispatch").build());
        let name = "ContinuationFrameI32";
        let frame = ref_type(&mut ctx, name, i32_ty);
        let layout = layout_type(&mut ctx, name, i32_ty, done, dispatch);

        assert_eq!(result_type(&ctx, frame), Some(i32_ty));
        assert_eq!(result_type(&ctx, layout), Some(i32_ty));
        assert_eq!(adt::nominal_name(&ctx, frame), Some(name));
        assert_eq!(
            adt::Struct::from_type_ref(&ctx, layout)
                .unwrap()
                .fields(&ctx)
                .collect::<Vec<_>>(),
            [("done", done), ("dispatch", dispatch)]
        );
    }

    #[test]
    fn result_type_fails_closed_on_unmarked_or_foreign_types() {
        let mut ctx = IrContext::new();
        let name = ctx.string_attr("ContinuationFrame");
        let unmarked = ctx.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", name)
                .build(),
        );
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let foreign = ctx.intern_type(
            TypeDataBuilder::new("test", "frame")
                .attr(RESULT_ATTR, Attribute::Type(i32_ty))
                .build(),
        );

        assert_eq!(result_type(&ctx, unmarked), None);
        assert_eq!(result_type(&ctx, foreign), None);
    }
}
