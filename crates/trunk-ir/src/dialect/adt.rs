//! Arena-based adt dialect.

// === Pure operation registrations ===
crate::register_pure_op!(adt.struct_new);
crate::register_pure_op!(adt.struct_get);

crate::register_pure_op!(adt.variant_new);
crate::register_pure_op!(adt.variant_is);
crate::register_pure_op!(adt.variant_cast);
crate::register_pure_op!(adt.variant_get);

crate::register_pure_op!(adt.array_new);
crate::register_pure_op!(adt.array_get);
crate::register_pure_op!(adt.array_len);

crate::register_pure_op!(adt.ref_null);
crate::register_pure_op!(adt.ref_is_null);
crate::register_pure_op!(adt.ref_cast);

crate::register_pure_op!(adt.string_const);
crate::register_pure_op!(adt.bytes_const);

#[crate::dialect(crate = crate)]
mod adt {
    #[attr(r#type: Type)]
    fn struct_new(#[rest] fields: ()) -> result {}

    #[attr(r#type: Type, field: u32)]
    fn struct_get(r#ref: ()) -> result {}

    #[attr(r#type: Type, field: u32)]
    fn struct_set(r#ref: (), value: ()) {}

    #[attr(r#type: Type, tag: Symbol)]
    fn variant_new(#[rest] fields: ()) -> result {}

    #[attr(r#type: Type, tag: Symbol)]
    fn variant_is(r#ref: ()) -> result {}

    #[attr(r#type: Type, tag: Symbol)]
    fn variant_cast(r#ref: ()) -> result {}

    #[attr(r#type: Type, tag: Symbol, field: u32)]
    fn variant_get(r#ref: ()) -> result {}

    #[attr(r#type: Type)]
    fn array_new(#[rest] elements: ()) -> result {}

    fn array_get(r#ref: (), index: ()) -> result {}

    fn array_set(r#ref: (), index: (), value: ()) {}

    fn array_len(r#ref: ()) -> result {}

    #[attr(r#type: Type)]
    fn ref_null() -> result {}

    fn ref_is_null(r#ref: ()) -> result {}

    #[attr(r#type: Type)]
    fn ref_cast(r#ref: ()) -> result {}

    #[attr(value: String)]
    fn string_const() -> result {}

    #[attr(value: any)]
    fn bytes_const() -> result {}
}
