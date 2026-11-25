//! Heap dialect — objects and references to them.
//!
//! `adt` operates on values; `heap` operates on the objects that hold them.
//! Both targets share it, and it assumes no reclamation scheme. See
//! `design/tribute-dialects.md` § "`heap`".

use crate::dialect::adt::{self, Array, Enum, Struct};
use crate::dialect::tribute_rtti::{
    DescriptorFieldsError, DescriptorLayoutType, ManagedRefType, OperandOwnership,
};
use trunk_ir::attr_kind::Dense;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core::{I1, I8, Index};
use trunk_ir::op_flag::Pure;
use trunk_ir::ops::{DialectType, VerificationError, Verify};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::type_verifier::VerifyType;

mod ownership;

#[trunk_ir::dialect]
mod heap {
    /// A managed reference to an immutable object of layout `Layout`.
    ///
    /// ```text
    /// heap.box<!Point>
    /// heap.box<adt.array<core.i8>>
    /// heap.box<!Node, {nullable = true}>
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
    #[verify]
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

    /// A new object holding the `adt` value `value`; the result is its box.
    ///
    /// ```text
    /// %p = heap.box_value %v : heap.box<!Point>
    /// ```
    ///
    /// - **Shape:** the result is a non-null box whose layout is the type
    ///   of `value`: an `adt.struct` or `adt.enum` layout with a body, or an
    ///   `adt.array`.
    /// - **Meaning:** representation selection writes every boxing of a
    ///   value with this operation, and a target's object access fusion
    ///   turns it into the allocation or update that builds the value
    ///   (`design/tribute-dialects.md` § "Object 접근 fusion").
    /// - **Ownership:** it declares no operand roles. The native target
    ///   fuses it away before ownership planning.
    /// - **Effects:** the object's identity is not observable, so the
    ///   operation is `Pure`, as `heap.struct_new` is.
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
    ///   of a box yields the same value. A target's object access fusion
    ///   turns the reads of the result into reads of the object
    ///   (`design/tribute-dialects.md` § "Object 접근 fusion").
    /// - **Ownership:** as for `heap.box_value`.
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn unbox_value<B: Box>(r#ref: Value<B>) -> Value<B::Layout> {}

    /// A new struct object whose fields are `fields` in order, as
    /// `adt.struct_new` builds the value; the result is its box.
    ///
    /// ```text
    /// %p = heap.struct_new(%x, %y) : (core.f64, core.f64) -> heap.box<!Point>
    /// ```
    ///
    /// - **Shape:** the result is a non-null box of an `adt.struct` layout
    ///   with a body. The fields are one value of each of the layout's field
    ///   types, seen through the box's layout with its row arguments.
    /// - **Ownership:** each managed field operand moves its unit into its
    ///   field (`Initialize`). The result owns one unit.
    /// - **Effects:** the object's identity is not observable, so the
    ///   operation is `Pure`, as `adt.struct_new` is; so are the other
    ///   allocations and updates of boxes that cannot trap. A cell's
    ///   identity is observable, so `heap.cell_new` is not.
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format("(" fields ")" attr_dict_bare ":" functional_type(fields, result))]
    fn struct_new(fields: Variadic<_>) -> Value<impl Box> {}

    /// A new object of variant `tag` whose fields are `fields` in order, as
    /// `adt.variant_new` builds the value; the result is its box.
    ///
    /// ```text
    /// %d = heap.variant_new 1(%x, %y) : (core.f64, core.f64) -> heap.box<!Shape>
    /// ```
    ///
    /// - **Shape:** the result is a non-null box of an `adt.enum` layout
    ///   with a body, `tag` is a position among its variants, and the fields
    ///   are that variant's, as for `heap.struct_new`.
    /// - **Ownership:** as for `heap.struct_new` (`Initialize`).
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(tag "(" fields ")" attr_dict_bare ":" functional_type(fields, result))]
    fn variant_new(tag: Attr<Index>, fields: Variadic<_>) -> Value<impl Box> {}

    /// Field `field` of the struct object `ref` refers to.
    ///
    /// ```text
    /// %x = heap.struct_get %p {field = 0} : core.f64
    /// ```
    ///
    /// - **Shape:** `ref` is a non-null box of an `adt.struct` layout, and
    ///   the result has the type of field `field`, seen through the box's
    ///   layout with its row arguments.
    /// - **Ownership:** a managed result is a borrow derived from `ref`
    ///   (`Project`); objects behind a box never change.
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn struct_get<B: Box>(field: Attr<u32>, r#ref: Value<B>) -> Value<_> {}

    /// The struct `source` refers to with field `field` replaced by
    /// `value`, as `adt.struct_update` builds the value; the result is its
    /// box. The object `source` refers to does not change for any other
    /// reference.
    ///
    /// ```text
    /// %q = heap.struct_update %p, %y {field = 1} : heap.box<!Point>
    /// ```
    ///
    /// - **Shape:** `source` and the result are the same non-null box type
    ///   of an `adt.struct` layout, and `value` has the type of field
    ///   `field`, as for `heap.struct_get`.
    /// - **Ownership:** consumes the unit of `source` (`Consume`); a
    ///   managed `value` moves its unit into the field (`Initialize`). The
    ///   result owns one unit. Lowering may reuse the object when `source`
    ///   was its only reference, and otherwise copies it
    ///   (`design/rc.md` § "1. Typed ownership and RTTI plan").
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(source "," value attr_dict_bare ":" type(result))]
    fn struct_update<B: Box>(field: Attr<u32>, source: Value<B>, value: Value<_>) -> Value<B> {}

    /// Whether the enum object `ref` refers to is of variant `tag`, as
    /// `adt.variant_is` tests a value.
    ///
    /// ```text
    /// %is = heap.variant_is %s {tag = 1} : core.i1
    /// ```
    ///
    /// - **Shape:** `ref` is a non-null box of an `adt.enum` layout, and
    ///   `tag` is a position among its variants.
    /// - **Ownership:** borrows `ref`.
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn variant_is<B: Box>(tag: Attr<Index>, r#ref: Value<B>) -> Value<I1> {}

    /// The position of the variant of the enum object `ref` refers to, as
    /// `adt.variant_tag` reads a value's.
    ///
    /// ```text
    /// %tag = heap.variant_tag %s : heap.box<!Shape>
    /// ```
    ///
    /// - **Shape:** `ref` is a non-null box of an `adt.enum` layout.
    /// - **Ownership:** borrows `ref`.
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(r#ref))]
    fn variant_tag<B: Box>(r#ref: Value<B>) -> Value<Index> {}

    /// Field `field` of the object of variant `tag` that `ref` refers to,
    /// as `adt.variant_get` reads a value's.
    ///
    /// ```text
    /// %x = heap.variant_get %s {tag = 1, field = 0} : core.f64
    /// ```
    ///
    /// - **Shape:** `ref` is a non-null box of an `adt.enum` layout, `tag`
    ///   is a position among its variants, and the result has the type of
    ///   that variant's field `field`, seen through the box's layout.
    /// - **Ownership:** a managed result is a borrow derived from `ref`
    ///   (`Project`).
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(result))]
    fn variant_get<B: Box>(tag: Attr<Index>, field: Attr<u32>, r#ref: Value<B>) -> Value<_> {}

    /// A new `length`-element array object whose every element is
    /// `initial`, as `adt.array_new` builds the value; the result is its box.
    /// The target traps an execution whose `length` exceeds the largest
    /// array it can represent or allocate, so the operation is not `Pure`.
    ///
    /// ```text
    /// %a = heap.array_new %n, %x : heap.box<adt.array<core.i32>>
    /// ```
    ///
    /// - **Shape:** the result is a non-null box of an `adt.array` whose
    ///   element type is the type of `initial`.
    /// - **Ownership:** borrows `initial`; the operation acquires a unit
    ///   for every element it fills.
    #[implements(OperandOwnership)]
    #[verify]
    #[asm_format(length "," initial attr_dict_bare ":" type(result))]
    fn array_new<B, A>(length: Value<Index>, initial: Value<A::Element>) -> Value<B>
    where
        B: Box<Layout = A>,
        A: Array,
    {
    }

    /// A new array object of `elements` in order, as `adt.array_of` builds
    /// the value; the result is its box.
    ///
    /// ```text
    /// %a = heap.array_of %x, %y : heap.box<adt.array<core.i32>>
    /// ```
    ///
    /// - **Shape:** the result is a non-null box of an `adt.array` whose
    ///   element type each of `elements` has.
    /// - **Ownership:** each managed element moves its unit into the array
    ///   (`Initialize`).
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(elements attr_dict_bare ":" type(result))]
    fn array_of<B, A>(elements: Variadic<A::Element>) -> Value<B>
    where
        B: Box<Layout = A>,
        A: Array,
    {
    }

    /// A new array object of `source[0, start)`, `elements`, and
    /// `source[end, len)`, as `adt.array_splice` builds the value; the
    /// result is its box. It requires `start ≤ end ≤ len`; the target traps
    /// an execution that violates it, so the operation is not `Pure`.
    ///
    /// ```text
    /// %r = heap.array_splice %a[%s, %e](%x) : heap.box<adt.array<core.i32>>
    /// ```
    ///
    /// - **Shape:** `source` and the result are the same non-null box type
    ///   of an `adt.array`.
    /// - **Ownership:** borrows `source`; the operation acquires a unit for
    ///   every managed element it copies from it. Each managed operand of
    ///   `elements` moves its unit into the new array (`Initialize`).
    #[implements(OperandOwnership)]
    #[verify]
    #[asm_format(source "[" start "," end "]" "(" elements ")" attr_dict ":" type(result))]
    fn array_splice<B, A>(
        source: Value<B>,
        start: Value<Index>,
        end: Value<Index>,
        elements: Variadic<A::Element>,
    ) -> Value<B>
    where
        B: Box<Layout = A>,
        A: Array,
    {
    }

    /// The element at `index` of the array object `ref` refers to. It
    /// requires `index < len`; the target traps an execution that violates
    /// it, so the operation is not `Pure`.
    ///
    /// ```text
    /// %x = heap.array_get %a, %i : core.i32
    /// ```
    ///
    /// - **Shape:** `ref` is a non-null box of an `adt.array`.
    /// - **Ownership:** a managed result is a borrow derived from `ref`
    ///   (`Project`).
    #[implements(OperandOwnership)]
    #[verify]
    #[asm_format(r#ref "," index attr_dict_bare ":" type(result))]
    fn array_get<B, A>(r#ref: Value<B>, index: Value<Index>) -> Value<A::Element>
    where
        B: Box<Layout = A>,
        A: Array,
    {
    }

    /// The array `source` refers to with the element at `index` replaced
    /// by `value`, as `adt.array_set` builds the value; the result is its
    /// box. The object `source` refers to does not change for any other
    /// reference. It requires `index < len`; the target traps an execution
    /// that violates it, so the operation is not `Pure`.
    ///
    /// ```text
    /// %b = heap.array_set %a, %i, %x : heap.box<adt.array<core.i32>>
    /// ```
    ///
    /// - **Shape:** `source` and the result are the same non-null box type
    ///   of an `adt.array`.
    /// - **Ownership:** consumes the unit of `source` (`Consume`); a
    ///   managed `value` moves its unit into the element (`Initialize`). The
    ///   result owns one unit. Lowering may reuse the object as for
    ///   `heap.struct_update`.
    #[implements(OperandOwnership)]
    #[verify]
    #[asm_format(source "," index "," value attr_dict_bare ":" type(result))]
    fn array_set<B, A>(source: Value<B>, index: Value<Index>, value: Value<A::Element>) -> Value<B>
    where
        B: Box<Layout = A>,
        A: Array,
    {
    }

    /// The number of elements of the array object `ref` refers to.
    ///
    /// ```text
    /// %n = heap.array_len %a : heap.box<adt.array<core.i32>>
    /// ```
    ///
    /// - **Shape:** `ref` is a non-null box of an `adt.array`.
    /// - **Ownership:** borrows `ref`.
    #[implements(OperandOwnership)]
    #[flags(Pure)]
    #[verify]
    #[asm_format(r#ref attr_dict_bare ":" type(r#ref))]
    fn array_len<B, A>(r#ref: Value<B>) -> Value<Index>
    where
        B: Box<Layout = A>,
        A: Array,
    {
    }

    /// An array object holding the array constant `value`, as
    /// `adt.array_const` builds the value; the result is its box and owns
    /// one unit like a new allocation. Whether two evaluations produce the
    /// same object is not specified (`design/rc.md` § "정적 객체").
    ///
    /// ```text
    /// %b = heap.array_const dense<"0x0102"> : heap.box<adt.array<core.i8>>
    /// ```
    ///
    /// - **Shape:** the result is a non-null box of an `adt.array` of
    ///   `core.i8`, and `value` holds one byte per element.
    #[flags(Pure)]
    #[verify]
    #[asm_format(value attr_dict_bare ":" type(result))]
    fn array_const<B, A>(value: Attr<Dense>) -> Value<B>
    where
        B: Box<Layout = A>,
        A: Array<Element = I8>,
    {
    }

    /// The null inhabitant of its result type, a nullable box.
    ///
    /// ```text
    /// %n = heap.null : heap.box<!Node, {nullable = true}>
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
    /// %empty = heap.is_null %n : heap.box<!Node, {nullable = true}>
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
    /// %n = heap.cast %b : heap.box<!Node> to heap.box<!Node, {nullable = true}>
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
    #[display("expected a box of an `{kind}` layout")]
    LayoutKind { kind: &'static str },
    #[display("the layout is a nominal layout without a body")]
    NoBody,
    #[display("expected a non-null box")]
    Nullable,
    #[display("a non-null box is written without `nullable`, not with `nullable = false`")]
    NotNullSpelling,
    #[display("expected a nullable box")]
    NotNullable,
    #[display("field {field} is not a field of the layout")]
    NoField { field: u64 },
    #[display("tag {tag} is not a variant of an enum with {count} variant(s)")]
    NoVariant { tag: u64, count: usize },
    #[display("{operands} field operand(s) for {fields} field(s)")]
    FieldCount { operands: usize, fields: usize },
    #[display("{what} has type {actual}, but the layout declares {expected}")]
    FieldType {
        what: String,
        actual: String,
        expected: String,
    },
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

/// The box type of `value`, required to be non-null.
fn non_null_box(ctx: &IrContext, value: ValueRef) -> Result<Box, BoxError> {
    box_of_type(ctx, ctx.value_ty(value))
}

/// The box type `ty`, required to be non-null.
fn box_of_type(ctx: &IrContext, ty: TypeRef) -> Result<Box, BoxError> {
    let boxed = Box::from_type_ref(ctx, ty).ok_or(BoxError::Layout)?;
    if boxed.is_nullable(ctx) {
        return Err(BoxError::Nullable);
    }
    Ok(boxed)
}

/// The struct layout of `boxed`: a literal layout, a key, or a row
/// instance's `def`. A key must have a body to read its fields from.
fn struct_layout(ctx: &IrContext, boxed: Box) -> Result<Struct, BoxError> {
    let layout = adt::layout_of(ctx, boxed.layout(ctx))
        .and_then(|layout| Struct::from_type_ref(ctx, layout))
        .ok_or(BoxError::LayoutKind { kind: "adt.struct" })?;
    if layout.is_nominal(ctx) && layout.body(ctx).is_none() {
        return Err(BoxError::NoBody);
    }
    Ok(layout)
}

/// The enum layout of `boxed`: a key or a row instance's `def`, with a
/// body to read its variants from.
fn enum_layout(ctx: &IrContext, boxed: Box) -> Result<Enum, BoxError> {
    let layout = adt::layout_of(ctx, boxed.layout(ctx))
        .and_then(|layout| Enum::from_type_ref(ctx, layout))
        .ok_or(BoxError::LayoutKind { kind: "adt.enum" })?;
    if layout.is_nominal(ctx) && layout.body(ctx).is_none() {
        return Err(BoxError::NoBody);
    }
    Ok(layout)
}

/// The type of field `field` of the struct objects `boxed` refers to, as
/// its layout's row arguments see it.
fn struct_field(ctx: &IrContext, boxed: Box, field: u32) -> Result<TypeRef, BoxError> {
    let layout = struct_layout(ctx, boxed)?;
    adt::find_field_type(ctx, layout, boxed.layout(ctx), field as usize).ok_or(BoxError::NoField {
        field: field.into(),
    })
}

/// Check that `tag` is the position of a variant of `layout`.
fn variant_position(ctx: &IrContext, layout: Enum, tag: u64) -> Result<u32, BoxError> {
    let count = layout.variant_count(ctx);
    usize::try_from(tag)
        .ok()
        .filter(|&tag| tag < count)
        .and_then(|tag| u32::try_from(tag).ok())
        .ok_or(BoxError::NoVariant { tag, count })
}

/// The type of field `field` of variant `tag` of the enum objects `boxed`
/// refers to.
fn variant_field(ctx: &IrContext, boxed: Box, tag: u64, field: u32) -> Result<TypeRef, BoxError> {
    let layout = enum_layout(ctx, boxed)?;
    let tag = variant_position(ctx, layout, tag)?;
    adt::find_variant_field_type(ctx, layout, boxed.layout(ctx), tag, field as usize).ok_or(
        BoxError::NoField {
            field: field.into(),
        },
    )
}

/// Check that `value` has type `expected`.
fn expect_type(
    ctx: &IrContext,
    what: impl FnOnce() -> String,
    value: ValueRef,
    expected: TypeRef,
) -> Result<(), BoxError> {
    let actual = ctx.value_ty(value);
    if actual == expected {
        return Ok(());
    }
    Err(BoxError::FieldType {
        what: what(),
        actual: actual.display(ctx).to_string(),
        expected: expected.display(ctx).to_string(),
    })
}

/// Check that `operands` are one value of each of the `count` fields whose
/// types `field` gives.
fn verify_fields(
    ctx: &IrContext,
    operands: &[ValueRef],
    count: usize,
    field: impl Fn(u32) -> Result<TypeRef, BoxError>,
) -> Result<(), BoxError> {
    if operands.len() != count {
        return Err(BoxError::FieldCount {
            operands: operands.len(),
            fields: count,
        });
    }
    for (index, &operand) in (0u32..).zip(operands) {
        expect_type(ctx, || format!("field {index}"), operand, field(index)?)?;
    }
    Ok(())
}

/// Check that `value` is a non-null box whose layout's value it can hold:
/// a struct or enum layout with a body, or an array.
fn verify_holds_value(ctx: &IrContext, value: ValueRef) -> Result<(), BoxError> {
    let boxed = non_null_box(ctx, value)?;
    let layout = boxed.layout(ctx);
    if Array::from_type_ref(ctx, layout).is_some() {
        Ok(())
    } else if adt::layout_of(ctx, layout).is_some_and(|l| Struct::from_type_ref(ctx, l).is_some()) {
        struct_layout(ctx, boxed).map(drop)
    } else {
        enum_layout(ctx, boxed).map(drop)
    }
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

impl Verify for StructNew {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        let boxed = non_null_box(ctx, self.result(ctx))?;
        let count = struct_layout(ctx, boxed)?.fields(ctx).len();
        Ok(verify_fields(ctx, self.fields(ctx), count, |field| {
            struct_field(ctx, boxed, field)
        })?)
    }
}

impl Verify for VariantNew {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        let boxed = non_null_box(ctx, self.result(ctx))?;
        let layout = enum_layout(ctx, boxed)?;
        let tag = variant_position(ctx, layout, self.tag(ctx))?;
        let count = layout.variant_fields(ctx, tag).map_or(0, <[_]>::len);
        Ok(verify_fields(ctx, self.fields(ctx), count, |field| {
            variant_field(ctx, boxed, tag.into(), field)
        })?)
    }
}

impl Verify for StructGet {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        let boxed = non_null_box(ctx, self.r#ref(ctx))?;
        let field = self.field(ctx);
        let expected = struct_field(ctx, boxed, field)?;
        Ok(expect_type(
            ctx,
            || format!("the read of field {field}"),
            self.result(ctx),
            expected,
        )?)
    }
}

impl Verify for StructUpdate {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        let boxed = non_null_box(ctx, self.source(ctx))?;
        let field = self.field(ctx);
        let expected = struct_field(ctx, boxed, field)?;
        Ok(expect_type(
            ctx,
            || format!("the value of field {field}"),
            self.value(ctx),
            expected,
        )?)
    }
}

impl Verify for VariantIs {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        let layout = enum_layout(ctx, non_null_box(ctx, self.r#ref(ctx))?)?;
        variant_position(ctx, layout, self.tag(ctx))?;
        Ok(())
    }
}

impl Verify for VariantTag {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        enum_layout(ctx, non_null_box(ctx, self.r#ref(ctx))?)?;
        Ok(())
    }
}

impl Verify for VariantGet {
    fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
        let boxed = non_null_box(ctx, self.r#ref(ctx))?;
        let (tag, field) = (self.tag(ctx), self.field(ctx));
        let expected = variant_field(ctx, boxed, tag, field)?;
        Ok(expect_type(
            ctx,
            || format!("the read of field {field}"),
            self.result(ctx),
            expected,
        )?)
    }
}

/// Implement [`Verify`] for array operations whose box values, named by
/// their accessors, must be non-null.
macro_rules! verify_non_null_boxes {
    ($($op:ident: $($value:ident),+;)+) => {$(
        impl Verify for $op {
            fn verify(&self, ctx: &IrContext) -> Result<(), VerificationError> {
                $(non_null_box(ctx, self.$value(ctx))?;)+
                Ok(())
            }
        }
    )+};
}

verify_non_null_boxes! {
    ArrayNew: result;
    ArrayOf: result;
    ArraySplice: result;
    ArrayGet: r#ref;
    ArraySet: result;
    ArrayLen: r#ref;
    ArrayConst: result;
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
