//! Arena-based adt dialect.

// === Type alias hint registration ===
inventory::submit!(crate::asm_format::TypeAliasHint {
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

// === Textual syntax: `adt.struct<@Name(@field: type {attrs}, ...), {attrs}>` ===

inventory::submit! {
    crate::asm_format::TypeAsmFormat::new::<Struct>(print_struct_type, parse_struct_type)
}

fn print_struct_type(
    h: &mut crate::printer::TypePrintHelper<'_, '_>,
    ty: TypeRef,
) -> Option<std::fmt::Result> {
    let adt_struct = Struct::from_type_ref(h.ctx(), ty)?;
    Some(write_struct_type(h, adt_struct))
}

fn write_struct_type(
    h: &mut crate::printer::TypePrintHelper<'_, '_>,
    adt_struct: Struct,
) -> std::fmt::Result {
    use std::fmt::Write;

    let ctx = h.ctx();
    h.write_str("adt.struct<")?;
    h.write_symbol(adt_struct.name(ctx))?;
    h.write_char('(')?;
    for (index, (name, ty)) in adt_struct.fields(ctx).enumerate() {
        if index > 0 {
            h.write_str(", ")?;
        }
        h.write_symbol(name)?;
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

/// Parse the rest of `adt.struct<@Name(@field: type {attrs}, ...), {attrs}>`
/// after its opening bracket into the generic form: each field's name becomes
/// its parameter's `name` attribute, and the struct's name the type's.
///
/// The syntax owns both names, so `name` may not appear in either dictionary.
/// Anything else, including a generic spelling, backtracks to generic parsing.
fn parse_struct_type<'a>(
    input: &mut &'a str,
    dialect: &'a str,
    name: &'a str,
) -> winnow::ModalResult<crate::parser::raw::RawType<'a>> {
    use crate::parser::raw::{
        RawAttribute, RawParam, RawType, raw_attr_dict, raw_param, symbol_ref, ws,
    };
    use winnow::combinator::{delimited, opt, preceded, separated};
    use winnow::prelude::*;

    let backtrack = || winnow::error::ErrMode::Backtrack(winnow::error::ContextError::new());
    let struct_name = symbol_ref.parse_next(input)?;
    ws.parse_next(input)?;
    let fields: Vec<(String, RawParam<'a>)> = delimited(
        ('(', ws),
        separated(
            0..,
            (ws, symbol_ref, ws, ':', ws, raw_param, ws)
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
    attrs.push((STRUCT_NAME_ATTR.into(), RawAttribute::Symbol(struct_name)));
    let mut params = Vec::with_capacity(fields.len());
    for (field, mut param) in fields {
        if param.attrs.iter().any(|(key, _)| is_name(key)) {
            return Err(backtrack());
        }
        param
            .attrs
            .push((STRUCT_NAME_ATTR.into(), RawAttribute::Symbol(field)));
        params.push(param);
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
    use crate::parser::{parse_module, parse_test_module};
    use crate::printer::print_module;
    use crate::rewrite::Module;
    use crate::validation::validate_operation_verifiers;

    /// Print `module`, parse the output, and check that it prints the same.
    fn assert_roundtrip(ctx: &IrContext, module: crate::refs::OpRef) {
        let printed = print_module(ctx, module);
        let mut reparsed = IrContext::new();
        let module = parse_module(&mut reparsed, &printed).expect("printed IR must parse");
        assert_eq!(printed, print_module(&reparsed, module));
    }

    #[test]
    fn string_attribute_accessors_return_text_and_handle() {
        use crate::ops::DialectOp;

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
  !point = adt.struct<@Point(@x: core.i32, @y: core.i32 {k = @v})>
  !closure = adt.struct<@"Nested::Closure"(@func_ptr: func.func_sig<(core.i32) -> core.i32, {k = 1}> {}, @env: core.tuple<core.ptr>), {layout = @closure}>
  !empty = adt.struct<@Empty()>
  !nested = adt.struct<@Outer(@inner: adt.struct<@Inner(@a: core.i32), {layout = @closure}> {m = 1})>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("adt.struct syntax should parse");
        let aliases: std::collections::HashMap<_, _> = ctx
            .type_aliases()
            .iter()
            .map(|&(name, ty)| (name.to_string(), ty))
            .collect();
        let point = Struct::from_type_ref(&ctx, aliases["point"]).unwrap();
        assert_eq!(point.name(&ctx), Symbol::new("Point"));
        assert_eq!(
            point.fields(&ctx).map(|(name, _)| name).collect::<Vec<_>>(),
            [Symbol::new("x"), Symbol::new("y")]
        );
        assert_eq!(point.field_attrs(&ctx, 1).count(), 1);
        let empty = Struct::from_type_ref(&ctx, aliases["empty"]).unwrap();
        assert_eq!(empty.field_count(&ctx), 0);
        let printed = print_module(&ctx, module);
        for expected in [
            "!point = adt.struct<@Point(@x: core.i32, @y: core.i32 {k = @v})>",
            "adt.struct<@\"Nested::Closure\"(@func_ptr: func.func_sig<(core.i32) -> core.i32, {k = 1}>, @env: core.tuple<core.ptr>), {layout = @closure}>",
            "!empty = adt.struct<@Empty()>",
            "(@inner: adt.struct<@Inner(@a: core.i32), {layout = @closure}> {m = 1})>",
        ] {
            assert!(printed.contains(expected), "{expected}\n{printed}");
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn repeated_struct_is_aliased_by_its_name() {
        let input = r#"core.module @test {
  func.func @f1(%x: adt.struct<@_Marker(@a: core.i32)>) -> adt.struct<@_Marker(@a: core.i32)> {
    func.return %x
  }
  func.func @f2(%x: adt.struct<@_Marker(@a: core.i32)>) -> adt.struct<@_Marker(@a: core.i32)> {
    func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("struct types should parse");
        let printed = print_module(&ctx, module);
        assert!(
            printed.contains("!_Marker = adt.struct<@_Marker(@a: core.i32)>"),
            "{printed}"
        );
        assert!(
            printed.contains("func.func @f1(%0: !_Marker) -> !_Marker"),
            "{printed}"
        );
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_adt_struct_reserved_names_are_parse_errors() {
        for spelling in [
            "adt.struct<@P(@x: core.i32), {name = @Q}>",
            "adt.struct<@P(@x: core.i32), {param_attrs = [{}]}>",
            "adt.struct<@P(@x: core.i32 {name = @y})>",
        ] {
            let mut ctx = IrContext::new();
            let input = format!("core.module @test {{ !bad = {spelling} }}");
            parse_module(&mut ctx, &input).expect_err(spelling);
        }
    }

    #[test]
    fn test_malformed_adt_struct_parses_but_fails_validation() {
        for spelling in [
            "adt.struct<@P(@x: core.i32, @x: core.i64)>",
            "adt.struct<@P(), {fields = []}>",
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
        let named_field = |name: &str| {
            let mut attrs = AttributeMap::new();
            attrs.insert("name", Symbol::from_dynamic(name));
            attrs
        };
        let cases = [
            (
                "missing `name` symbol",
                TypeDataBuilder::new("adt", "struct")
                    .param_with_attrs(i32_ty, named_field("x"))
                    .build(),
            ),
            (
                "field 1 has no `name` symbol",
                TypeDataBuilder::new("adt", "struct")
                    .param_with_attrs(i32_ty, named_field("x"))
                    .param(i32_ty)
                    .attr("name", Attribute::Symbol(Symbol::new("P")))
                    .build(),
            ),
            (
                "duplicate field name @x",
                TypeDataBuilder::new("adt", "struct")
                    .param_with_attrs(i32_ty, named_field("x"))
                    .param_with_attrs(i32_ty, named_field("x"))
                    .attr("name", Attribute::Symbol(Symbol::new("P")))
                    .build(),
            ),
            (
                "`fields` is not an `adt.struct` attribute",
                TypeDataBuilder::new("adt", "struct")
                    .attr("name", Attribute::Symbol(Symbol::new("P")))
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
