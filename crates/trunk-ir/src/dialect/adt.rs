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
