//! Arena-based adt dialect.

// === Type alias hint registration ===
inventory::submit!(trunk_ir::asm_format::TypeAliasHint {
    dialect: "adt",
    suggest: |ctx, ty| { ctx.get_type(ty).attrs.get_str(ctx, "name") },
});

// === Managed reference registrations ===
//
// A value of a nominal layout type, or a reference to one by name, points to
// an allocation of that layout.
inventory::submit!(crate::dialect::tribute_rtti::ManagedRefType::new::<Struct>());
inventory::submit!(crate::dialect::tribute_rtti::ManagedRefType::named(
    "adt", "enum"
));
inventory::submit!(crate::dialect::tribute_rtti::ManagedRefType::named(
    "adt", "typeref"
));

// === Pure operation registrations ===
trunk_ir::register_pure_op!(StructNew);
trunk_ir::register_pure_op!(StructGet);

trunk_ir::register_pure_op!(VariantNew);
trunk_ir::register_pure_op!(VariantIs);
trunk_ir::register_pure_op!(VariantCast);
trunk_ir::register_pure_op!(VariantGet);

trunk_ir::register_pure_op!(ArrayNew);
trunk_ir::register_pure_op!(ArrayGet);
trunk_ir::register_pure_op!(ArrayLen);

trunk_ir::register_pure_op!(RefNull);
trunk_ir::register_pure_op!(RefIsNull);
trunk_ir::register_pure_op!(RefCast);

trunk_ir::register_pure_op!(StringConst);
trunk_ir::register_pure_op!(BytesConst);

#[trunk_ir::dialect]
mod adt {
    fn struct_new(r#type: Attr<Type>, fields: Variadic<_>) -> Value<_> {}

    fn struct_get(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    fn struct_set(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    fn variant_new(r#type: Attr<Type>, tag: Attr<String>, fields: Variadic<_>) -> Value<_> {}

    fn variant_is(r#type: Attr<Type>, tag: Attr<String>, r#ref: Value<_>) -> Value<_> {}

    fn variant_cast(r#type: Attr<Type>, tag: Attr<String>, r#ref: Value<_>) -> Value<_> {}

    fn variant_get(
        r#type: Attr<Type>,
        tag: Attr<String>,
        field: Attr<u32>,
        r#ref: Value<_>,
    ) -> Value<_> {
    }

    fn array_new(r#type: Attr<Type>, elements: Variadic<_>) -> Value<_> {}

    fn array_get(r#ref: Value<_>, index: Value<_>) -> Value<_> {}

    fn array_set(r#ref: Value<_>, index: Value<_>, value: Value<_>) {}

    fn array_len(r#ref: Value<_>) -> Value<_> {}

    fn ref_null(r#type: Attr<Type>) -> Value<_> {}

    fn ref_is_null(r#ref: Value<_>) -> Value<_> {}

    fn ref_cast(r#type: Attr<Type>, r#ref: Value<_>) -> Value<_> {}

    fn string_const(value: Attr<String>) -> Value<_> {}

    fn bytes_const(value: Attr<Bytes>) -> Value<_> {}
}

// === Nominal struct layout type ===

use trunk_ir::Symbol;
use trunk_ir::attr_kind::{Bytes, Type};
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::TypeRef;
use trunk_ir::types::{
    Attribute, AttributeMap, PARAM_ATTRS_ATTR, StringArg, StringRef, TypeDataBuilder,
};

pub mod layout;

/// Type attribute holding an `adt.struct`'s name, and the parameter attribute
/// holding each field's name. Both are strings.
pub const STRUCT_NAME_ATTR: &str = "name";

/// The nominal name of an `adt.struct`, `adt.enum` or `adt.typeref`: its
/// `name` type attribute.
pub fn nominal_name_ref(ctx: &IrContext, ty: TypeRef) -> Option<StringRef> {
    let data = ctx.get_type(ty);
    let nominal = data.dialect == Symbol::new("adt")
        && (data.name == Symbol::new("struct")
            || data.name == Symbol::new("enum")
            || data.name == Symbol::new("typeref"));
    nominal
        .then(|| data.attrs.get_string_ref(STRUCT_NAME_ATTR))
        .flatten()
}

/// The text of [`nominal_name_ref`].
pub fn nominal_name(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    nominal_name_ref(ctx, ty).map(|name| ctx.str(name))
}

/// A malformed `adt.struct` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum StructTypeError {
    #[display("missing `{STRUCT_NAME_ATTR}` string")]
    MissingName,
    #[display("field {_0} has no `{STRUCT_NAME_ATTR}` string")]
    MissingFieldName(#[error(not(source))] usize),
    #[display("duplicate field name {_0:?}")]
    DuplicateFieldName(#[error(not(source))] String),
    #[display("`fields` is not an `adt.struct` attribute; fields are type parameters")]
    FieldsAttribute,
}

/// Validated wrapper for a nominal `adt.struct` layout type.
///
/// Field types are the type's parameters, each field's name is its parameter's
/// `name` attribute, and the struct's name is the type's `name` attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Struct(TypeRef);

impl Struct {
    /// Validate a name-matching `adt.struct`.
    pub fn validate(ctx: &IrContext, ty: TypeRef) -> Result<Self, StructTypeError> {
        let data = ctx.get_type(ty);
        debug_assert!(Self::matches(ctx, ty));
        if data.attrs.contains_key("fields") {
            return Err(StructTypeError::FieldsAttribute);
        }
        if data.attrs.get_string_ref(STRUCT_NAME_ATTR).is_none() {
            return Err(StructTypeError::MissingName);
        }
        let mut names: smallvec::SmallVec<[StringRef; 8]> = smallvec::SmallVec::new();
        for index in 0..data.params.len() {
            let name = data
                .param_attrs(index)
                .get_string_ref(STRUCT_NAME_ATTR)
                .ok_or(StructTypeError::MissingFieldName(index))?;
            if names.contains(&name) {
                return Err(StructTypeError::DuplicateFieldName(
                    ctx.str(name).to_owned(),
                ));
            }
            names.push(name);
        }
        Ok(Self(ty))
    }

    pub fn as_type_ref(self) -> TypeRef {
        self.0
    }

    /// The struct's name.
    pub fn name(self, ctx: &IrContext) -> &str {
        ctx.str(self.name_ref(ctx))
    }

    /// The struct's name as a pooled handle.
    pub fn name_ref(self, ctx: &IrContext) -> StringRef {
        ctx.get_type(self.0)
            .attrs
            .get_string_ref(STRUCT_NAME_ATTR)
            .expect("validated adt.struct must retain its name")
    }

    /// The field types, in declaration order.
    pub fn field_types(self, ctx: &IrContext) -> &[TypeRef] {
        &ctx.get_type(self.0).params
    }

    pub fn field_count(self, ctx: &IrContext) -> usize {
        ctx.get_type(self.0).params.len()
    }

    /// The type of field `index`, if it exists.
    pub fn field_type(self, ctx: &IrContext, index: usize) -> Option<TypeRef> {
        ctx.get_type(self.0).params.get(index).copied()
    }

    /// The name of field `index`, if it exists.
    pub fn field_name(self, ctx: &IrContext, index: usize) -> Option<&str> {
        self.field_name_ref(ctx, index).map(|name| ctx.str(name))
    }

    /// The name of field `index` as a pooled handle, if it exists.
    pub fn field_name_ref(self, ctx: &IrContext, index: usize) -> Option<StringRef> {
        let data = ctx.get_type(self.0);
        (index < data.params.len()).then(|| field_name(data.param_attrs(index)))
    }

    /// The index of the field named `name`.
    pub fn field_index(self, ctx: &IrContext, name: &str) -> Option<usize> {
        self.fields(ctx).position(|(field, _)| field == name)
    }

    /// Each field's name and type, in declaration order.
    pub fn fields(self, ctx: &IrContext) -> impl ExactSizeIterator<Item = (&str, TypeRef)> + '_ {
        let data = ctx.get_type(self.0);
        data.params
            .iter()
            .enumerate()
            .map(|(index, &ty)| (ctx.str(field_name(data.param_attrs(index))), ty))
    }

    /// The attributes of field `index` other than its name.
    pub fn field_attrs(
        self,
        ctx: &IrContext,
        index: usize,
    ) -> impl Iterator<Item = (&Symbol, &Attribute)> + '_ {
        ctx.get_type(self.0)
            .param_attrs(index)
            .iter()
            .filter(|(key, _)| **key != Symbol::new(STRUCT_NAME_ATTR))
    }

    /// The type attributes other than the name and the field attributes.
    pub fn extra_attrs(self, ctx: &IrContext) -> impl Iterator<Item = (&Symbol, &Attribute)> + '_ {
        ctx.get_type(self.0).attrs.iter().filter(|(key, _)| {
            **key != Symbol::new(STRUCT_NAME_ATTR) && **key != Symbol::new(PARAM_ATTRS_ATTR)
        })
    }

    /// Each field as an owned `(name, type, attributes)` triple, for rebuilding
    /// the struct with [`struct_type_with_field_attrs`].
    pub fn fields_with_attrs(self, ctx: &IrContext) -> Vec<(StringRef, TypeRef, AttributeMap)> {
        (0..self.field_count(ctx))
            .map(|index| {
                let name = self
                    .field_name_ref(ctx, index)
                    .expect("field index is in range");
                let ty = self
                    .field_type(ctx, index)
                    .expect("field index is in range");
                let attrs = self
                    .field_attrs(ctx, index)
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                (name, ty, attrs)
            })
            .collect()
    }
}

fn field_name(attrs: &AttributeMap) -> StringRef {
    attrs
        .get_string_ref(STRUCT_NAME_ATTR)
        .expect("validated adt.struct fields must retain their names")
}

impl DialectType for Struct {
    const DIALECT_NAME: &'static str = "adt";
    const TYPE_NAME: &'static str = "struct";

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        if !Self::matches(ctx, ty) {
            return None;
        }
        Self::validate(ctx, ty).ok()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

inventory::submit! {
    trunk_ir::type_verifier::TypeVerifier::new::<Struct>(|ctx, ty| {
        Struct::validate(ctx, ty).map(|_| ()).map_err(|error| error.to_string())
    })
}

impl From<Struct> for TypeRef {
    fn from(ty: Struct) -> Self {
        ty.0
    }
}

/// Construct an `adt.struct` named `name` with `fields` in declaration order.
///
/// `attrs` holds the remaining type attributes, such as `layout`.
pub fn struct_type<N: Into<StringArg>>(
    ctx: &mut IrContext,
    name: impl Into<StringArg>,
    fields: impl IntoIterator<Item = (N, TypeRef)>,
    attrs: AttributeMap,
) -> Struct {
    struct_type_with_field_attrs(
        ctx,
        name,
        fields
            .into_iter()
            .map(|(field, ty)| (field, ty, AttributeMap::new())),
        attrs,
    )
}

/// Construct an `adt.struct` whose fields carry their own attributes.
///
/// # Panics
///
/// If the result is not a valid `adt.struct`; see [`try_struct_type`].
pub fn struct_type_with_field_attrs<N: Into<StringArg>>(
    ctx: &mut IrContext,
    name: impl Into<StringArg>,
    fields: impl IntoIterator<Item = (N, TypeRef, AttributeMap)>,
    attrs: AttributeMap,
) -> Struct {
    try_struct_type(ctx, name, fields, attrs).unwrap_or_else(|error| panic!("adt.struct: {error}"))
}

/// Construct an `adt.struct`, reporting a duplicate field name or a `fields`
/// attribute instead of panicking.
///
/// A `name` in `attrs` or in a field's attributes is replaced by `name` or the
/// field's name, so a caller may pass a source struct's attributes unchanged.
pub fn try_struct_type<N: Into<StringArg>>(
    ctx: &mut IrContext,
    name: impl Into<StringArg>,
    fields: impl IntoIterator<Item = (N, TypeRef, AttributeMap)>,
    attrs: AttributeMap,
) -> Result<Struct, StructTypeError> {
    let mut builder = TypeDataBuilder::new("adt", "struct");
    for (field, ty, mut field_attrs) in fields {
        let field = ctx.intern_string_arg(field.into());
        field_attrs.insert(STRUCT_NAME_ATTR, field);
        builder = builder.param_with_attrs(ty, field_attrs);
    }
    let name = ctx.intern_string_arg(name.into());
    finish_struct_type(ctx, name, builder, attrs)
}

/// The non-generic rest of [`try_struct_type`], once the fields are added.
fn finish_struct_type(
    ctx: &mut IrContext,
    name: StringRef,
    mut builder: TypeDataBuilder,
    mut attrs: AttributeMap,
) -> Result<Struct, StructTypeError> {
    attrs.remove(PARAM_ATTRS_ATTR);
    attrs.insert(STRUCT_NAME_ATTR, name);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let data = builder.build();
    if data.attrs.contains_key("fields") {
        return Err(StructTypeError::FieldsAttribute);
    }
    let mut seen: smallvec::SmallVec<[StringRef; 8]> = smallvec::SmallVec::new();
    for index in 0..data.params.len() {
        let field = field_name(data.param_attrs(index));
        if seen.contains(&field) {
            return Err(StructTypeError::DuplicateFieldName(
                ctx.str(field).to_owned(),
            ));
        }
        seen.push(field);
    }
    Ok(Struct(ctx.intern_type(data)))
}

// === Textual syntax: `adt.struct<Name(field: type {attrs}, ...), {attrs}>` ===
//
// Names are bare identifiers, or quoted strings when they are not identifiers.

inventory::submit! {
    trunk_ir::asm_format::TypeAsmFormat::new::<Struct>(print_struct_type, parse_struct_type)
}

fn print_struct_type(
    h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>,
    ty: TypeRef,
) -> Option<std::fmt::Result> {
    let adt_struct = Struct::from_type_ref(h.ctx(), ty)?;
    Some(write_struct_type(h, adt_struct))
}

fn write_struct_type(
    h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>,
    adt_struct: Struct,
) -> std::fmt::Result {
    use std::fmt::Write;

    let ctx = h.ctx();
    h.write_str("adt.struct<")?;
    h.write_name(adt_struct.name(ctx))?;
    h.write_char('(')?;
    for (index, (name, ty)) in adt_struct.fields(ctx).enumerate() {
        if index > 0 {
            h.write_str(", ")?;
        }
        h.write_name(name)?;
        h.write_str(": ")?;
        h.write_type(ty)?;
        let mut attrs = adt_struct.field_attrs(ctx, index).peekable();
        if attrs.peek().is_some() {
            h.write_char(' ')?;
            h.write_attr_dict(attrs)?;
        }
    }
    h.write_char(')')?;
    let mut attrs = adt_struct.extra_attrs(ctx).peekable();
    if attrs.peek().is_some() {
        h.write_str(", ")?;
        h.write_attr_dict(attrs)?;
    }
    h.write_char('>')
}

/// Parse the rest of `adt.struct<Name(field: type {attrs}, ...), {attrs}>`
/// after its opening bracket into the generic form: each field's name becomes
/// its parameter's `name` attribute, and the struct's name the type's.
///
/// The syntax owns both names, so `name` may not appear in either dictionary.
/// Anything else, including a generic spelling, backtracks to generic parsing.
fn parse_struct_type<'a>(
    input: &mut &'a str,
    dialect: &'a str,
    name: &'a str,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawType<'a>> {
    use trunk_ir::parser::raw::{
        RawAttribute, RawParam, RawType, name_token, raw_attr_dict, raw_param, ws,
    };
    use winnow::combinator::{delimited, opt, preceded, separated};
    use winnow::prelude::*;

    let backtrack = || winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new());
    let struct_name = name_token.parse_next(input)?;
    ws.parse_next(input)?;
    let fields: Vec<(String, RawParam<'a>)> = delimited(
        ('(', ws),
        separated(
            0..,
            (ws, name_token, ws, ':', ws, raw_param, ws)
                .map(|(_, field, _, _, _, param, _)| (field, param)),
            ',',
        ),
        (ws, ')'),
    )
    .parse_next(input)?;
    ws.parse_next(input)?;
    let mut attrs = opt(preceded((',', ws), raw_attr_dict))
        .parse_next(input)?
        .unwrap_or_default();
    ws.parse_next(input)?;
    '>'.parse_next(input)?;

    let is_name = |key: &str| key == STRUCT_NAME_ATTR;
    if attrs.iter().any(|(key, _)| is_name(key)) {
        return Err(backtrack());
    }
    attrs.push((STRUCT_NAME_ATTR.into(), RawAttribute::String(struct_name)));
    let mut params = Vec::with_capacity(fields.len());
    for (field, mut param) in fields {
        if param.attrs.iter().any(|(key, _)| is_name(key)) {
            return Err(backtrack());
        }
        param
            .attrs
            .push((STRUCT_NAME_ATTR.into(), RawAttribute::String(field)));
        params.push(param);
    }
    Ok(RawType::Concrete {
        dialect,
        name,
        params,
        attrs,
    })
}

// === Nominal enum layout type ===

/// A malformed `adt.enum` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum EnumTypeError {
    #[display("missing `{STRUCT_NAME_ATTR}` string")]
    MissingName,
    #[display("variant {_0} is not an `adt.variant` with a `{STRUCT_NAME_ATTR}` string")]
    MalformedVariant(#[error(not(source))] usize),
    #[display("duplicate variant name {_0:?}")]
    DuplicateVariantName(#[error(not(source))] String),
    #[display("`variants` is not an `adt.enum` attribute; variants are type parameters")]
    VariantsAttribute,
    #[display("an `adt.enum` variant parameter carries no parameter attributes")]
    VariantAttributes,
}

/// Validated wrapper for a nominal `adt.enum` layout type.
///
/// Each variant is one type parameter: an `adt.variant` whose parameters are
/// the variant's field types and whose `name` attribute is the variant's
/// name. A field may carry a `name` parameter attribute. The enum's name is
/// the type's `name` attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Enum(TypeRef);

impl Enum {
    /// Validate a name-matching `adt.enum`.
    pub fn validate(ctx: &IrContext, ty: TypeRef) -> Result<Self, EnumTypeError> {
        let data = ctx.get_type(ty);
        debug_assert!(Self::matches(ctx, ty));
        if data.attrs.contains_key("variants") {
            return Err(EnumTypeError::VariantsAttribute);
        }
        // The syntax has no place for them, so printing would drop them.
        if data.attrs.contains_key(PARAM_ATTRS_ATTR) {
            return Err(EnumTypeError::VariantAttributes);
        }
        if data.attrs.get_string_ref(STRUCT_NAME_ATTR).is_none() {
            return Err(EnumTypeError::MissingName);
        }
        let mut names: smallvec::SmallVec<[StringRef; 8]> = smallvec::SmallVec::new();
        for (index, &variant) in data.params.iter().enumerate() {
            let name = variant_name(ctx, variant).ok_or(EnumTypeError::MalformedVariant(index))?;
            if names.contains(&name) {
                return Err(EnumTypeError::DuplicateVariantName(
                    ctx.str(name).to_owned(),
                ));
            }
            names.push(name);
        }
        Ok(Self(ty))
    }

    pub fn as_type_ref(self) -> TypeRef {
        self.0
    }

    /// The enum's name.
    pub fn name(self, ctx: &IrContext) -> &str {
        ctx.str(self.name_ref(ctx))
    }

    /// The enum's name as a pooled handle.
    pub fn name_ref(self, ctx: &IrContext) -> StringRef {
        ctx.get_type(self.0)
            .attrs
            .get_string_ref(STRUCT_NAME_ATTR)
            .expect("validated adt.enum must retain its name")
    }

    pub fn variant_count(self, ctx: &IrContext) -> usize {
        ctx.get_type(self.0).params.len()
    }

    /// Each variant's name and field types, in declaration order.
    pub fn variants(
        self,
        ctx: &IrContext,
    ) -> impl ExactSizeIterator<Item = (StringRef, &[TypeRef])> + '_ {
        ctx.get_type(self.0).params.iter().map(|&variant| {
            let name = variant_name(ctx, variant).expect("validated adt.enum variant");
            (name, &ctx.get_type(variant).params[..])
        })
    }

    /// The field types of the variant named `tag`.
    pub fn variant_fields(self, ctx: &IrContext, tag: StringRef) -> Option<&[TypeRef]> {
        self.variants(ctx)
            .find_map(|(name, fields)| (name == tag).then_some(fields))
    }

    /// The type attributes other than the name.
    pub fn extra_attrs(self, ctx: &IrContext) -> impl Iterator<Item = (&Symbol, &Attribute)> + '_ {
        ctx.get_type(self.0)
            .attrs
            .iter()
            .filter(|(key, _)| **key != Symbol::new(STRUCT_NAME_ATTR))
    }
}

/// The name of an `adt.variant` type.
fn variant_name(ctx: &IrContext, variant: TypeRef) -> Option<StringRef> {
    let data = ctx.get_type(variant);
    (data.dialect == Symbol::new("adt") && data.name == Symbol::new("variant"))
        .then(|| data.attrs.get_string_ref(STRUCT_NAME_ATTR))
        .flatten()
}

impl DialectType for Enum {
    const DIALECT_NAME: &'static str = "adt";
    const TYPE_NAME: &'static str = "enum";

    fn from_type_ref(ctx: &IrContext, ty: TypeRef) -> Option<Self> {
        if !Self::matches(ctx, ty) {
            return None;
        }
        Self::validate(ctx, ty).ok()
    }

    fn as_type_ref(&self) -> TypeRef {
        self.0
    }
}

inventory::submit! {
    trunk_ir::type_verifier::TypeVerifier::new::<Enum>(|ctx, ty| {
        Enum::validate(ctx, ty).map(|_| ()).map_err(|error| error.to_string())
    })
}

impl From<Enum> for TypeRef {
    fn from(ty: Enum) -> Self {
        ty.0
    }
}

/// Construct an `adt.enum` named `name` with `variants` in declaration order,
/// each a name and its field types.
///
/// `attrs` holds the remaining type attributes.
///
/// # Panics
///
/// If two variants have the same name, or `attrs` has a `variants` or
/// parameter attribute entry.
pub fn enum_type<N: Into<StringArg>, F: IntoIterator<Item = TypeRef>>(
    ctx: &mut IrContext,
    name: impl Into<StringArg>,
    variants: impl IntoIterator<Item = (N, F)>,
    mut attrs: AttributeMap,
) -> Enum {
    let mut builder = TypeDataBuilder::new("adt", "enum");
    for (variant, fields) in variants {
        let variant = ctx.intern_string_arg(variant.into());
        let variant = ctx.intern_type(
            TypeDataBuilder::new("adt", "variant")
                .params(fields)
                .attr(STRUCT_NAME_ATTR, variant)
                .build(),
        );
        builder = builder.param(variant);
    }
    let name = ctx.intern_string_arg(name.into());
    attrs.insert(STRUCT_NAME_ATTR, name);
    for (key, value) in attrs {
        builder = builder.attr(key, value);
    }
    let ty = ctx.intern_type(builder.build());
    Enum::validate(ctx, ty).unwrap_or_else(|error| panic!("adt.enum: {error}"))
}

// === Textual syntax: `adt.enum<Name { Variant(field: type, ...), ... }, {attrs}>` ===
//
// A field is `name: type` or, without a name, `type`. Either may be followed
// by the field's other attributes.

inventory::submit! {
    trunk_ir::asm_format::TypeAsmFormat::new::<Enum>(print_enum_type, parse_enum_type)
}

fn print_enum_type(
    h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>,
    ty: TypeRef,
) -> Option<std::fmt::Result> {
    let adt_enum = Enum::from_type_ref(h.ctx(), ty)?;
    Some(write_enum_type(h, adt_enum))
}

fn write_enum_type(
    h: &mut trunk_ir::printer::TypePrintHelper<'_, '_>,
    adt_enum: Enum,
) -> std::fmt::Result {
    use std::fmt::Write;

    let ctx = h.ctx();
    let data = ctx.get_type(adt_enum.as_type_ref());
    h.write_str("adt.enum<")?;
    h.write_name(adt_enum.name(ctx))?;
    h.write_str(" {")?;
    for (index, &variant) in data.params.iter().enumerate() {
        h.write_str(if index > 0 { ", " } else { " " })?;
        let variant_data = ctx.get_type(variant);
        let name = variant_name(ctx, variant).expect("validated adt.enum variant");
        h.write_name(ctx.str(name))?;
        h.write_char('(')?;
        for (field, &field_ty) in variant_data.params.iter().enumerate() {
            if field > 0 {
                h.write_str(", ")?;
            }
            let field_attrs = variant_data.param_attrs(field);
            if let Some(field_name) = field_attrs.get_string_ref(STRUCT_NAME_ATTR) {
                h.write_name(ctx.str(field_name))?;
                h.write_str(": ")?;
            }
            h.write_type(field_ty)?;
            let mut attrs = field_attrs
                .iter()
                .filter(|(key, _)| **key != Symbol::new(STRUCT_NAME_ATTR))
                .peekable();
            if attrs.peek().is_some() {
                h.write_char(' ')?;
                h.write_attr_dict(attrs)?;
            }
        }
        h.write_char(')')?;
    }
    h.write_str(if data.params.is_empty() { "}" } else { " }" })?;
    let mut attrs = adt_enum.extra_attrs(ctx).peekable();
    if attrs.peek().is_some() {
        h.write_str(", ")?;
        h.write_attr_dict(attrs)?;
    }
    h.write_char('>')
}

/// Parse the rest of `adt.enum<Name { Variant(field: type, ...), ... }, {attrs}>`
/// after its opening bracket into the generic form: each variant becomes an
/// `adt.variant` parameter named by its `name` attribute.
///
/// The syntax owns the names, so `name` may not appear in a dictionary.
/// Anything else, including a generic spelling, backtracks to generic parsing.
fn parse_enum_type<'a>(
    input: &mut &'a str,
    dialect: &'a str,
    name: &'a str,
) -> winnow::ModalResult<trunk_ir::parser::raw::RawType<'a>> {
    use trunk_ir::parser::raw::{
        RawAttribute, RawParam, RawType, name_token, raw_attr_dict, raw_param, ws,
    };
    use winnow::combinator::{delimited, opt, preceded, separated, terminated};
    use winnow::prelude::*;

    type Field<'a> = (Option<String>, RawParam<'a>);

    let backtrack = || winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new());
    let field = |input: &mut &'a str| -> winnow::ModalResult<Field<'a>> {
        (
            ws,
            opt(terminated(name_token, (ws, ':', ws))),
            raw_param,
            ws,
        )
            .map(|(_, field, param, _)| (field, param))
            .parse_next(input)
    };
    let variant = |input: &mut &'a str| -> winnow::ModalResult<(String, Vec<Field<'a>>)> {
        (
            ws,
            name_token,
            ws,
            delimited('(', separated(0.., field, ','), (ws, ')')),
            ws,
        )
            .map(|(_, variant, _, fields, _)| (variant, fields))
            .parse_next(input)
    };

    let enum_name = name_token.parse_next(input)?;
    ws.parse_next(input)?;
    let variants: Vec<(String, Vec<Field<'a>>)> =
        delimited('{', separated(0.., variant, ','), (ws, '}')).parse_next(input)?;
    ws.parse_next(input)?;
    let mut attrs = opt(preceded((',', ws), raw_attr_dict))
        .parse_next(input)?
        .unwrap_or_default();
    ws.parse_next(input)?;
    '>'.parse_next(input)?;

    let is_name = |key: &str| key == STRUCT_NAME_ATTR;
    if attrs.iter().any(|(key, _)| is_name(key)) {
        return Err(backtrack());
    }
    attrs.push((STRUCT_NAME_ATTR.into(), RawAttribute::String(enum_name)));
    let mut params = Vec::with_capacity(variants.len());
    for (variant, fields) in variants {
        let mut field_params = Vec::with_capacity(fields.len());
        for (field, mut param) in fields {
            if param.attrs.iter().any(|(key, _)| is_name(key)) {
                return Err(backtrack());
            }
            if let Some(field) = field {
                param
                    .attrs
                    .push((STRUCT_NAME_ATTR.into(), RawAttribute::String(field)));
            }
            field_params.push(param);
        }
        params.push(RawParam::from(RawType::Concrete {
            dialect: "adt",
            name: "variant",
            params: field_params,
            attrs: vec![(STRUCT_NAME_ATTR.into(), RawAttribute::String(variant))],
        }));
    }
    Ok(RawType::Concrete {
        dialect,
        name,
        params,
        attrs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap as HashMap;
    use trunk_ir::parser::{parse_module, parse_test_module};
    use trunk_ir::printer::print_module;
    use trunk_ir::rewrite::Module;
    use trunk_ir::validation::validate_operation_verifiers;

    /// Print `module`, parse the output, and check that it prints the same.
    fn assert_roundtrip(ctx: &IrContext, module: trunk_ir::refs::OpRef) {
        let printed = print_module(ctx, module);
        let mut reparsed = IrContext::new();
        let module = parse_module(&mut reparsed, &printed).expect("printed IR must parse");
        assert_eq!(printed, print_module(&reparsed, module));
    }

    #[test]
    fn string_attribute_accessors_return_text_and_handle() {
        use trunk_ir::ops::DialectOp;

        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  %0 = adt.string_const {value = "hi"} : core.ptr
}"#,
        );
        let op = module.ops(&ctx)[0];
        let string_const = StringConst::from_op(&ctx, op).unwrap();
        assert_eq!(string_const.value(&ctx), "hi");
        assert_eq!(string_const.value_ref(&ctx), ctx.lookup_str("hi").unwrap());

        let loc = ctx.op(op).location;
        let ptr = ctx.op_result_types(op)[0];
        let copy = StringConst::operands()
            .value(string_const.value_ref(&ctx))
            .results(ptr)
            .build(&mut ctx, loc);
        assert_eq!(copy.value(&ctx), "hi");
    }

    #[test]
    fn test_roundtrip_adt_struct() {
        let input = r#"core.module @test {
  !point = adt.struct<Point(x: core.i32, y: core.i32 {k = @v})>
  !closure = adt.struct<"Nested::Closure"(func_ptr: func.func_sig<(core.i32) -> core.i32, {k = 1}> {}, env: core.tuple<core.ptr>), {layout = "closure"}>
  !empty = adt.struct<Empty()>
  !nested = adt.struct<Outer(inner: adt.struct<Inner(a: core.i32), {layout = "closure"}> {m = 1})>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("adt.struct syntax should parse");
        let aliases: HashMap<_, _> = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (name.to_string(), *ty))
            .collect();
        let point = Struct::from_type_ref(&ctx, aliases["point"]).unwrap();
        assert_eq!(point.name(&ctx), Symbol::new("Point"));
        assert_eq!(
            point.fields(&ctx).map(|(name, _)| name).collect::<Vec<_>>(),
            ["x", "y"]
        );
        assert_eq!(point.field_attrs(&ctx, 1).count(), 1);
        let empty = Struct::from_type_ref(&ctx, aliases["empty"]).unwrap();
        assert_eq!(empty.field_count(&ctx), 0);
        let printed = print_module(&ctx, module);
        for expected in [
            "!point = adt.struct<Point(x: core.i32, y: core.i32 {k = @v})>",
            "adt.struct<\"Nested::Closure\"(func_ptr: func.func_sig<(core.i32) -> core.i32, {k = 1}>, env: core.tuple<core.ptr>), {layout = \"closure\"}>",
            "!empty = adt.struct<Empty()>",
            "(inner: adt.struct<Inner(a: core.i32), {layout = \"closure\"}> {m = 1})>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn repeated_struct_is_aliased_by_its_name() {
        let input = r#"core.module @test {
  func.func @f1(%x: adt.struct<_Marker(a: core.i32)>) -> adt.struct<_Marker(a: core.i32)> {
    func.return %x
  }
  func.func @f2(%x: adt.struct<_Marker(a: core.i32)>) -> adt.struct<_Marker(a: core.i32)> {
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("struct types should parse");
        let printed = print_module(&ctx, module);
        assert!(
            printed.contains("!_Marker = adt.struct<_Marker(a: core.i32)>"),
            "{printed}"
        );
        assert!(
            printed.contains("func.func @f1(%0: !_Marker) -> !_Marker"),
            "{printed}"
        );
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_roundtrip_adt_enum() {
        let input = r#"core.module @test {
  !option = adt.enum<Option { None(), Some(value: core.i32) }>
  !shape = adt.enum<"geo::Shape" { Dot(), Rect(core.f64, core.f64 {k = 1}), Named(label: core.ptr {k = 2}) }, {layout = "x"}>
  !never = adt.enum<Never {}>
  !list = adt.enum<List { Empty(), Cons(core.i32, adt.typeref<{name = "List"}>) }>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("adt.enum syntax should parse");
        let aliases: HashMap<_, _> = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (name.to_string(), *ty))
            .collect();
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());

        let option = Enum::from_type_ref(&ctx, aliases["option"]).unwrap();
        assert_eq!(option.name(&ctx), "Option");
        let variants: Vec<_> = option
            .variants(&ctx)
            .map(|(name, fields)| (ctx.str(name), fields.to_vec()))
            .collect();
        assert_eq!(variants, [("None", vec![]), ("Some", vec![i32_ty])]);
        let some = ctx.intern_str("Some");
        assert_eq!(option.variant_fields(&ctx, some), Some(&[i32_ty][..]));
        let missing = ctx.intern_str("Missing");
        assert_eq!(option.variant_fields(&ctx, missing), None);
        // A variant is one type parameter, so type traversal reaches fields.
        assert_eq!(ctx.get_type(aliases["option"]).params.len(), 2);

        let never = Enum::from_type_ref(&ctx, aliases["never"]).unwrap();
        assert_eq!(never.variant_count(&ctx), 0);

        // The constructor builds the type the syntax denotes, apart from the
        // field name the syntax adds.
        let built = enum_type(
            &mut ctx,
            "List",
            [("Empty", vec![]), ("Cons", vec![i32_ty])],
            AttributeMap::new(),
        );
        assert_eq!(built.variant_count(&ctx), 2);

        let printed = print_module(&ctx, module);
        for expected in [
            "!option = adt.enum<Option { None(), Some(value: core.i32) }>",
            "adt.enum<\"geo::Shape\" { Dot(), Rect(core.f64, core.f64 {k = 1}), Named(label: core.ptr {k = 2}) }, {layout = \"x\"}>",
            "!never = adt.enum<Never {}>",
            "adt.enum<List { Empty(), Cons(core.i32, adt.typeref<{name = \"List\"}>) }>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_adt_enum_reserved_names_are_parse_errors() {
        for spelling in [
            "adt.enum<E { A() }, {name = @Q}>",
            "adt.enum<E { A(core.i32 {name = @x}) }>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_malformed_adt_enum_parses_but_fails_validation() {
        for spelling in [
            "adt.enum<E { A(), A(core.i32) }>",
            "adt.enum<{name = \"E\", variants = [[\"A\", []]]}>",
            "adt.enum<core.i32, {name = \"E\"}>",
            "adt.enum<adt.variant<core.i32>, {name = \"E\"}>",
            "adt.enum<adt.variant<{name = \"A\"}>>",
            "adt.enum<adt.variant<{name = \"A\"}>, {name = @E}>",
            // The syntax cannot print a variant parameter's attributes.
            "adt.enum<adt.variant<{name = \"A\"}> {k = 1}, {name = \"E\"}>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            let module = parse_module(&mut ctx, &input).expect(spelling);
            let alias = ctx.type_aliases()[0].1;
            assert!(Enum::from_type_ref(&ctx, alias).is_none(), "{spelling}");
            let result = validate_operation_verifiers(&ctx, Module::new(&ctx, module).unwrap());
            assert!(!result.is_ok(), "{spelling}");
        }
    }

    #[test]
    fn test_adt_struct_reserved_names_are_parse_errors() {
        for spelling in [
            "adt.struct<P(x: core.i32), {name = @Q}>",
            "adt.struct<P(x: core.i32), {param_attrs = [{}]}>",
            "adt.struct<P(x: core.i32 {name = @y})>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_malformed_adt_struct_parses_but_fails_validation() {
        for spelling in [
            "adt.struct<P(x: core.i32, x: core.i64)>",
            "adt.struct<P(), {fields = []}>",
            "adt.struct<core.i32, {name = @P}>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            let module = parse_module(&mut ctx, &input).expect(spelling);
            let alias = ctx.type_aliases()[0].1;
            assert!(Struct::from_type_ref(&ctx, alias).is_none(), "{spelling}");
            let result = validate_operation_verifiers(&ctx, Module::new(&ctx, module).unwrap());
            assert!(!result.is_ok(), "{spelling}");
        }
    }

    #[test]
    fn malformed_adt_struct_types_are_rejected_by_type_validation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, "core.module @test {}");
        let i32_ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let x = ctx.string_attr("x");
        let p = ctx.string_attr("P");
        let named_field = || {
            let mut attrs = AttributeMap::new();
            attrs.insert("name", x.clone());
            attrs
        };
        let cases = [
            (
                "missing `name` string",
                TypeDataBuilder::new("adt", "struct")
                    .param_with_attrs(i32_ty, named_field())
                    .build(),
            ),
            (
                "field 1 has no `name` string",
                TypeDataBuilder::new("adt", "struct")
                    .param_with_attrs(i32_ty, named_field())
                    .param(i32_ty)
                    .attr("name", p.clone())
                    .build(),
            ),
            (
                "duplicate field name \"x\"",
                TypeDataBuilder::new("adt", "struct")
                    .param_with_attrs(i32_ty, named_field())
                    .param_with_attrs(i32_ty, named_field())
                    .attr("name", p.clone())
                    .build(),
            ),
            (
                "`fields` is not an `adt.struct` attribute",
                TypeDataBuilder::new("adt", "struct")
                    .attr("name", p)
                    .attr("fields", Attribute::List(vec![]))
                    .build(),
            ),
        ];
        for (expected, data) in cases {
            let ty = ctx.intern_type(data);
            assert!(Struct::from_type_ref(&ctx, ty).is_none(), "{expected}");
            let result = validate_operation_verifiers(&ctx, module);
            let text = result.to_string();
            assert!(
                text.contains("adt.struct") && text.contains(expected),
                "{expected}: {text}"
            );
        }
        let valid = struct_type(&mut ctx, "Q", [("x", i32_ty)], AttributeMap::new());
        assert!(
            ctx.get_type(valid.as_type_ref())
                .attrs
                .contains_key(PARAM_ATTRS_ATTR)
        );
    }
}
