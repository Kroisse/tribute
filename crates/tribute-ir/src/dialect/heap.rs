//! Heap dialect — objects and references to them.
//!
//! `adt` operates on values; `heap` names the objects that hold them: boxes
//! of `adt` values and mutable cells. Both targets share it, and it assumes
//! no reclamation scheme. See `design/tribute-dialects.md` § "`heap`".

use crate::dialect::adt::{self, Array, Enum, Struct};
use crate::dialect::tribute_rtti::{
    DescriptorFieldsError, DescriptorLayoutType, ManagedRefType, OperandOwnership,
};
use std::fmt::{self, Write as _};

use trunk_ir::asm_format::{
    TypeAliasHint, TypeAsmFormat, parse_dict_tail, suggest_type_alias_name,
};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core::I1;
use trunk_ir::op_flag::Pure;
use trunk_ir::ops::{DialectType, VerificationError, Verify};
use trunk_ir::parser::{AsmParser, ParseResult};
use trunk_ir::printer::AsmPrinter;
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::type_verifier::VerifyType;
use trunk_ir::types::{Attribute, AttributeMap};

mod ownership;

#[trunk_ir::dialect]
mod heap {
    /// A managed reference to an immutable object of layout `Layout`.
    ///
    /// ```text
    /// heap.box<!Point>
    /// heap.box<adt.array<core.i8>>
    /// heap.box<nullable !Node>
    /// ```
    ///
    /// - **Layout:** `Layout` is an `adt.struct` or `adt.enum` layout (a
    ///   literal layout, a nominal layout's key, or a row instance) or an
    ///   `adt.array`. The object's descriptor is the layout's, or that of a
    ///   row instance's `def`. A box of an enum layout refers to an object
    ///   of any of its variants, which the object's descriptor tells apart.
    /// - **Meaning:** no reference observes the object change after it is
    ///   built, so its identity is not part of the program's meaning
    ///   (`design/tribute-dialects.md` § "`heap.box`").
    /// - **Null:** a box is never null unless it has `nullable = true`, as
    ///   for `core.ref`. Only compiler-made representations use null; see
    ///   `heap.null`, `heap.is_null`, and `heap.cast`. Retaining or releasing
    ///   null does nothing.
    /// - **Syntax:** a nullable box writes `nullable`, the name of its
    ///   attribute, before its layout. Other type attributes follow the
    ///   layout as `, {..}`.
    /// - **Alias:** a box of a layout that suggests an alias suggests that
    ///   name with `_box`, and a nullable one with `_nbox`: `!Point_box`,
    ///   `!Node_nbox`.
    #[verify]
    #[asm_format]
    #[alias_hint]
    #[implements(ManagedRefType)]
    struct Box<Layout> {
        nullable: Option<Attr<bool>>,
    }

    /// A managed reference to a mutable object holding one value of type
    /// `Contents`, like ML's `ref`.
    ///
    /// - **Meaning:** every reference to the same cell sees the value the
    ///   last `heap.cell_set` wrote. A cell is always an object, so
    ///   representation selection never unboxes it.
    /// - **Contents:** the storage representation of the held value. A
    ///   managed value is one unit the cell owns.
    /// - **Descriptor:** a cell allocation's descriptor is the cell type
    ///   without a tag, whose only field is `Contents`.
    /// - **Lowering:** each target stores the value as the only field of an
    ///   object, as it stores a field of a one-field struct.
    ///
    /// ```text
    /// heap.cell<core.i1>
    /// ```
    #[implements(ManagedRefType, DescriptorLayoutType)]
    struct Cell<Contents>;

    /// The `adt` value `value` as a box of an object holding it.
    ///
    /// ```text
    /// %p = heap.box_value %v : heap.box<!Point>
    /// ```
    ///
    /// - **Shape:** the result is a non-null box whose layout is the type
    ///   of `value`: an `adt.struct` or `adt.enum` layout with a body, or an
    ///   `adt.array`.
    /// - **Meaning:** representation selection writes every boxing of a
    ///   value with this operation (`design/tribute-dialects.md` §
    ///   "`heap`"). For a type it chose boxed, `value` already is the object
    ///   and the result is the same reference. For an unboxed type at a
    ///   forced position, target lowering allocates an object holding the
    ///   value's fields.
    /// - **Ownership:** for a boxed type, the result is `value` itself and
    ///   shares its unit (`Forward`). For an unboxed type, the unit of a
    ///   managed `value` moves into the new object (`Initialize`).
    /// - **Effects:** the object's identity is not observable, so the
    ///   operation is `Pure`. A cell's identity is observable, so
    ///   `heap.cell_new` is not.
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(value attr_dict_bare ":" type(result))]
    fn box_value<B: Box>(value: Value<B::Layout>) -> Value<B> {}

    /// The `adt` value the object `ref` refers to holds.
    ///
    /// ```text
    /// %v = heap.unbox_value %p : !Point
    /// ```
    ///
    /// - **Shape:** `ref` is a non-null box of an `adt.struct` or
    ///   `adt.enum` layout with a body, or of an `adt.array`, and the result
    ///   has the box's layout as its type.
    /// - **Meaning:** objects behind a box never change, so every unboxing
    ///   of a box yields the same value. For a type representation selection
    ///   chose boxed, the result is the same reference, and target lowering
    ///   lowers the `adt` operations on it to accesses of the object. For an
    ///   unboxed type at a forced position, target lowering reads the
    ///   object's fields.
    /// - **Ownership:** for a boxed type, the result is `ref` itself and
    ///   shares its unit (`Forward`). For an unboxed type, the result is a
    ///   borrow of the object's field (`Project`).
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn unbox_value<B: Box>(r#ref: Value<B>) -> Value<B::Layout> {}

    /// The null inhabitant of its result type, a nullable box.
    ///
    /// ```text
    /// %n = heap.null : heap.box<nullable !Node>
    /// ```
    ///
    /// - **Shape:** the result is a box with `nullable = true`.
    /// - **Ownership:** null owns nothing; retaining or releasing it does
    ///   nothing.
    #[flags(Pure)]
    #[verify]
    #[asm_format(attr_dict_bare ":" type(result))]
    fn null() -> Value<impl Box> {}

    /// Whether the nullable box `ref` is null.
    ///
    /// ```text
    /// %empty = heap.is_null %n : heap.box<nullable !Node>
    /// ```
    ///
    /// - **Shape:** `ref` is a box with `nullable = true`.
    /// - **Ownership:** borrows `ref`.
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(r#ref))]
    fn is_null<B: Box>(r#ref: Value<B>) -> Value<I1> {}

    /// `ref` as its result type, another box of the same object.
    ///
    /// ```text
    /// %n = heap.cast %b : heap.box<!Node> to heap.box<nullable !Node>
    /// ```
    ///
    /// - **Shape:** the two types are boxes that differ only in
    ///   `nullable`, or a box of a closure capture layout and a box of its
    ///   `closure_base`, in either direction and with the same `nullable`.
    ///   A cast from a nullable box to a non-null one is valid only where
    ///   `heap.is_null` has ruled null out. A cast between a box and
    ///   `tribute_rt.anyref` is a `core.unrealized_conversion_cast`, not
    ///   this operation (`design/tribute-dialects.md` § "`adt`").
    /// - **Ownership:** the result is `ref` itself and shares its unit
    ///   (`Forward`).
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(r#ref) "to" type(result))]
    fn cast(r#ref: Value<impl Box>) -> Value<impl Box> {}

    /// A new cell holding `value`.
    ///
    /// - **Ownership:** a managed `value` moves its unit into the cell
    ///   (`Initialize`).
    /// - **Effects:** each `cell_new` allocates a distinct cell, so two of
    ///   them are never merged; the operation is not `Pure`.
    ///
    /// ```text
    /// %c = heap.cell_new %v : heap.cell<core.i1>
    /// ```
    #[implements(OperandOwnership)]
    #[asm_format(value attr_dict_bare ":" type(result))]
    fn cell_new<C: Cell>(value: Value<C::Contents>) -> Value<C> {}

    /// The value `cell` holds now.
    ///
    /// - **Ownership:** the cell is borrowed for the read. A managed result
    ///   is never a borrow of the cell: it acquires its own unit right after
    ///   the read, because a later `heap.cell_set` reached through another
    ///   reference or a called function would release the value it replaces
    ///   (`Copy`; `design/rc.md` § "3. RC optimization pass").
    /// - **Effects:** a `heap.cell_set` between two reads may change the
    ///   value, so the operation is not `Pure`.
    ///
    /// ```text
    /// %v = heap.cell_get %c : core.i1
    /// ```
    #[implements(OperandOwnership)]
    #[asm_format(cell attr_dict_bare ":" type(result))]
    fn cell_get<C: Cell>(cell: Value<C>) -> Value<C::Contents> {}

    /// Replace the value `cell` holds with `value`.
    ///
    /// - **Ownership:** the cell is borrowed for the write. A managed
    ///   `value` moves its unit into the cell, and the unit of the replaced
    ///   value is released (`Assign`). A parameter written through is
    ///   retained, not borrowed (`design/rc.md` § "Borrowed parameter
    ///   elision").
    ///
    /// ```text
    /// heap.cell_set %c, %v
    /// ```
    #[implements(OperandOwnership)]
    #[asm_format(cell "," value attr_dict_bare)]
    fn cell_set<C: Cell>(cell: Value<C>, value: Value<C::Contents>) {}
}

impl ManagedRefType for Box {}
impl ManagedRefType for Cell {}

impl DescriptorLayoutType for Cell {
    fn descriptor_fields<'a>(
        &self,
        ctx: &'a IrContext,
        tag: Option<u32>,
    ) -> Result<&'a [TypeRef], DescriptorFieldsError> {
        match tag {
            // A cell's only parameter is the type of the value it holds.
            None => Ok(&ctx.get_type(self.as_type_ref()).params),
            Some(_) => Err(DescriptorFieldsError::UnexpectedTag {
                layout: "heap.cell",
            }),
        }
    }
}

/// A box type or a box operation that does not fit its layout.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub enum BoxError {
    #[display("the layout of a box is not an `adt.struct`, `adt.enum`, or `adt.array`")]
    Layout,
    #[display("the layout is a nominal layout without a body")]
    NoBody,
    #[display("expected a non-null box")]
    Nullable,
    #[display("a non-null box is written without `nullable`, not with `nullable = false`")]
    NotNullSpelling,
    #[display("expected a nullable box")]
    NotNullable,
    #[display("a cast between boxes of unrelated layouts")]
    CastLayouts,
}

impl VerifyType for Box {
    /// Validate a name-matching `heap.box`: its layout is an `adt.struct`,
    /// `adt.enum`, or `adt.array`.
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, VerificationError> {
        let Some(&layout) = ctx.get_type(ty).params.first() else {
            return Err(BoxError::Layout.into());
        };
        let is_layout = Struct::from_type_ref(ctx, layout).is_some()
            || Enum::from_type_ref(ctx, layout).is_some()
            || Array::from_type_ref(ctx, layout).is_some();
        if !is_layout {
            return Err(BoxError::Layout.into());
        }
        // Absence is the one spelling of a non-null box.
        if ctx.type_attrs(ty).get_bool("nullable") == Some(false) {
            return Err(BoxError::NotNullSpelling.into());
        }
        Ok(Self(ty))
    }
}

impl Box {
    /// Whether this box type has a null value.
    pub fn is_nullable(self, ctx: &IrContext) -> bool {
        self.nullable(ctx) == Some(true)
    }
}

/// The type attribute, and the keyword that writes it before the layout.
const NULLABLE: &str = "nullable";

impl TypeAsmFormat for Box {
    fn print(&self, p: &mut AsmPrinter<'_>) -> fmt::Result {
        let s = p.state();
        let ctx = s.ctx();
        let nullable = if self.is_nullable(ctx) {
            "nullable "
        } else {
            ""
        };
        write!(p, "heap.box<{nullable}{}", s.ty(self.layout(ctx)))?;
        let mut rest = ctx
            .get_type(self.as_type_ref())
            .attrs
            .iter()
            .filter(|&(key, _)| ctx.str(key) != NULLABLE)
            .peekable();
        if rest.peek().is_some() {
            write!(p, ", {}", s.attr_dict(rest))?;
        }
        p.write_char('>')
    }

    fn parse(p: &mut AsmParser<'_>, name: &str) -> ParseResult<Option<TypeRef>> {
        let nullable = p.parse_optional_keyword(NULLABLE);
        let layout = p.parse_type()?;
        let mut attrs = parse_dict_tail(p)?;
        p.parse_punct(">")?;
        if nullable {
            if attrs.view(p.ctx()).contains_key(NULLABLE) {
                return Err(p.expected("`nullable` once, as the keyword or the attribute"));
            }
            attrs
                .view_mut(p.ctx_mut())
                .insert(NULLABLE, Attribute::Bool(true));
        }
        p.build_type(name, vec![(layout, AttributeMap::new())], attrs)
            .map(Some)
    }
}

impl TypeAliasHint for Box {
    fn alias_hint(&self, ctx: &IrContext, out: &mut dyn fmt::Write) -> fmt::Result {
        let Some(layout) = suggest_type_alias_name(ctx, self.layout(ctx)) else {
            return Ok(());
        };
        let suffix = if self.is_nullable(ctx) { "nbox" } else { "box" };
        write!(out, "{layout}_{suffix}")
    }
}

/// The box type of `value`, required to be non-null.
fn non_null_box(ctx: &IrContext, value: ValueRef) -> Result<Box, BoxError> {
    let boxed = Box::from_type_ref(ctx, ctx.value_ty(value)).ok_or(BoxError::Layout)?;
    if boxed.is_nullable(ctx) {
        return Err(BoxError::Nullable);
    }
    Ok(boxed)
}

/// Check that `value` is a non-null box whose layout's value it can hold:
/// a struct or enum layout with a body, or an array.
fn verify_holds_value(ctx: &IrContext, value: ValueRef) -> Result<(), BoxError> {
    let layout = non_null_box(ctx, value)?.layout(ctx);
    if Array::from_type_ref(ctx, layout).is_some() {
        return Ok(());
    }
    // A row instance holds the value of its `def`.
    let def = adt::layout_of(ctx, layout).ok_or(BoxError::Layout)?;
    let key_without_body = if let Some(layout) = Struct::from_type_ref(ctx, def) {
        layout.is_nominal(ctx) && layout.body(ctx).is_none()
    } else if let Some(layout) = Enum::from_type_ref(ctx, def) {
        layout.is_nominal(ctx) && layout.body(ctx).is_none()
    } else {
        return Err(BoxError::Layout);
    };
    if key_without_body {
        return Err(BoxError::NoBody);
    }
    Ok(())
}

impl Verify for BoxValue {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        Ok(verify_holds_value(ctx, self.result(ctx))?)
    }
}

impl Verify for UnboxValue {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        Ok(verify_holds_value(ctx, self.r#ref(ctx))?)
    }
}

impl Verify for Null {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        nullable_box(ctx, ctx.value_ty(self.result(ctx)))?;
        Ok(())
    }
}

impl Verify for IsNull {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        nullable_box(ctx, ctx.value_ty(self.r#ref(ctx)))?;
        Ok(())
    }
}

/// The box type `ty`, required to be nullable.
fn nullable_box(ctx: &IrContext, ty: TypeRef) -> Result<Box, BoxError> {
    let boxed = Box::from_type_ref(ctx, ty).ok_or(BoxError::Layout)?;
    if !boxed.is_nullable(ctx) {
        return Err(BoxError::NotNullable);
    }
    Ok(boxed)
}

impl Verify for Cast {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        let from =
            Box::from_type_ref(ctx, ctx.value_ty(self.r#ref(ctx))).ok_or(BoxError::Layout)?;
        let to = Box::from_type_ref(ctx, ctx.value_ty(self.result(ctx))).ok_or(BoxError::Layout)?;
        let (from_layout, to_layout) = (from.layout(ctx), to.layout(ctx));
        let related = if from_layout == to_layout {
            from.is_nullable(ctx) != to.is_nullable(ctx)
        } else {
            from.is_nullable(ctx) == to.is_nullable(ctx)
                && (closure_base_of(ctx, from_layout) == Some(to_layout)
                    || closure_base_of(ctx, to_layout) == Some(from_layout))
        };
        if !related {
            return Err(BoxError::CastLayouts.into());
        }
        Ok(())
    }
}

/// The `closure_base` of the closure capture layout `layout`.
fn closure_base_of(ctx: &IrContext, layout: TypeRef) -> Option<TypeRef> {
    Struct::from_type_ref(ctx, layout)?;
    ctx.type_attrs(layout)
        .get_type(crate::runtime_layout::CLOSURE_BASE_ATTR)
}

#[cfg(test)]
mod tests;
