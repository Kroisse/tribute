//! Arena-based adt dialect.

// === Type alias hint registration ===
inventory::submit!(crate::op_interface::TypeAliasHint {
    dialect: "adt",
    suggest: |ctx, ty| { ctx.get_type(ty).attrs.get_symbol("name") },
});

// === Pure operation registrations ===
crate::register_pure_op!(StructNew);
crate::register_pure_op!(StructGet);

crate::register_pure_op!(VariantNew);
crate::register_pure_op!(VariantIs);
crate::register_pure_op!(VariantCast);
crate::register_pure_op!(VariantGet);

crate::register_pure_op!(ArrayNew);
crate::register_pure_op!(ArrayGet);
crate::register_pure_op!(ArrayLen);

crate::register_pure_op!(RefNull);
crate::register_pure_op!(RefIsNull);
crate::register_pure_op!(RefCast);

crate::register_pure_op!(StringConst);
crate::register_pure_op!(BytesConst);

#[trunk_ir::dialect]
mod adt {
    fn struct_new(r#type: Attr<Type>, fields: Variadic<_>) -> Value<_> {}

    fn struct_get(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    fn struct_set(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    fn variant_new(r#type: Attr<Type>, tag: Attr<Symbol>, fields: Variadic<_>) -> Value<_> {}

    fn variant_is(r#type: Attr<Type>, tag: Attr<Symbol>, r#ref: Value<_>) -> Value<_> {}

    fn variant_cast(r#type: Attr<Type>, tag: Attr<Symbol>, r#ref: Value<_>) -> Value<_> {}

    fn variant_get(
        r#type: Attr<Type>,
        tag: Attr<Symbol>,
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

use crate::Symbol;
use crate::context::IrContext;
use crate::ops::DialectType;
use crate::refs::TypeRef;
use crate::types::{Attribute, AttributeMap, PARAM_ATTRS_ATTR, TypeDataBuilder};

/// Type attribute holding an `adt.struct`'s name, and the parameter attribute
/// holding each field's name.
pub const STRUCT_NAME_ATTR: &str = "name";

/// A malformed `adt.struct` type.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum StructTypeError {
    #[display("missing `{STRUCT_NAME_ATTR}` symbol")]
    MissingName,
    #[display("field {_0} has no `{STRUCT_NAME_ATTR}` symbol")]
    MissingFieldName(#[error(not(source))] usize),
    #[display("duplicate field name @{_0}")]
    DuplicateFieldName(#[error(not(source))] Symbol),
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
        if data.attrs.get_symbol(STRUCT_NAME_ATTR).is_none() {
            return Err(StructTypeError::MissingName);
        }
        let mut names: smallvec::SmallVec<[Symbol; 8]> = smallvec::SmallVec::new();
        for index in 0..data.params.len() {
            let name = data
                .param_attrs(index)
                .get_symbol(STRUCT_NAME_ATTR)
                .ok_or(StructTypeError::MissingFieldName(index))?;
            if names.contains(&name) {
                return Err(StructTypeError::DuplicateFieldName(name));
            }
            names.push(name);
        }
        Ok(Self(ty))
    }

    pub fn as_type_ref(self) -> TypeRef {
        self.0
    }

    /// The struct's name.
    pub fn name(self, ctx: &IrContext) -> Symbol {
        ctx.get_type(self.0)
            .attrs
            .get_symbol(STRUCT_NAME_ATTR)
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
    pub fn field_name(self, ctx: &IrContext, index: usize) -> Option<Symbol> {
        let data = ctx.get_type(self.0);
        (index < data.params.len()).then(|| field_name(data.param_attrs(index)))
    }

    /// The index of the field named `name`.
    pub fn field_index(self, ctx: &IrContext, name: Symbol) -> Option<usize> {
        self.fields(ctx).position(|(field, _)| field == name)
    }

    /// Each field's name and type, in declaration order.
    pub fn fields(self, ctx: &IrContext) -> impl ExactSizeIterator<Item = (Symbol, TypeRef)> + '_ {
        let data = ctx.get_type(self.0);
        data.params
            .iter()
            .enumerate()
            .map(|(index, &ty)| (field_name(data.param_attrs(index)), ty))
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
    pub fn fields_with_attrs(self, ctx: &IrContext) -> Vec<(Symbol, TypeRef, AttributeMap)> {
        (0..self.field_count(ctx))
            .map(|index| {
                let name = self
                    .field_name(ctx, index)
                    .expect("field index is in range");
                let ty = self
                    .field_type(ctx, index)
                    .expect("field index is in range");
                let attrs = self
                    .field_attrs(ctx, index)
                    .map(|(key, value)| (*key, value.clone()))
                    .collect();
                (name, ty, attrs)
            })
            .collect()
    }
}

fn field_name(attrs: &AttributeMap) -> Symbol {
    attrs
        .get_symbol(STRUCT_NAME_ATTR)
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

impl From<Struct> for TypeRef {
    fn from(ty: Struct) -> Self {
        ty.0
    }
}

/// Construct an `adt.struct` named `name` with `fields` in declaration order.
///
/// `attrs` holds the remaining type attributes, such as `layout`.
pub fn struct_type<N: Into<Symbol>>(
    ctx: &mut IrContext,
    name: impl Into<Symbol>,
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
pub fn struct_type_with_field_attrs<N: Into<Symbol>>(
    ctx: &mut IrContext,
    name: impl Into<Symbol>,
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
pub fn try_struct_type<N: Into<Symbol>>(
    ctx: &mut IrContext,
    name: impl Into<Symbol>,
    fields: impl IntoIterator<Item = (N, TypeRef, AttributeMap)>,
    attrs: AttributeMap,
) -> Result<Struct, StructTypeError> {
    let mut builder = TypeDataBuilder::new("adt", "struct");
    for (field, ty, mut field_attrs) in fields {
        field_attrs.insert(STRUCT_NAME_ATTR, field.into());
        builder = builder.param_with_attrs(ty, field_attrs);
    }
    finish_struct_type(ctx, name.into(), builder, attrs)
}

/// The non-generic rest of [`try_struct_type`], once the fields are added.
fn finish_struct_type(
    ctx: &mut IrContext,
    name: Symbol,
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
    let mut seen: smallvec::SmallVec<[Symbol; 8]> = smallvec::SmallVec::new();
    for index in 0..data.params.len() {
        let field = field_name(data.param_attrs(index));
        if seen.contains(&field) {
            return Err(StructTypeError::DuplicateFieldName(field));
        }
        seen.push(field);
    }
    Ok(Struct(ctx.intern_type(data)))
}
