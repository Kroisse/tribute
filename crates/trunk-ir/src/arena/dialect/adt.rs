//! Arena-based adt dialect.

use crate::arena_dialect;

arena_dialect! {
    mod adt {
        #[attr(r#type: Type)]
        fn struct_new(#[rest] fields) -> result;

        #[attr(r#type: Type, field: u32)]
        fn struct_get(r#ref) -> result;

        #[attr(r#type: Type, field: u32)]
        fn struct_set(r#ref, value);

        #[attr(r#type: Type, tag: Symbol)]
        fn variant_new(#[rest] fields) -> result;

        #[attr(r#type: Type, tag: Symbol)]
        fn variant_is(r#ref) -> result;

        #[attr(r#type: Type, tag: Symbol)]
        fn variant_cast(r#ref) -> result;

        #[attr(r#type: Type, tag: Symbol, field: u32)]
        fn variant_get(r#ref) -> result;

        #[attr(r#type: Type)]
        fn array_new(#[rest] elements) -> result;

        fn array_get(r#ref, index) -> result;

        fn array_set(r#ref, index, value);

        fn array_len(r#ref) -> result;

        #[attr(r#type: Type)]
        fn ref_null() -> result;

        fn ref_is_null(r#ref) -> result;

        #[attr(r#type: Type)]
        fn ref_cast(r#ref) -> result;

        #[attr(value: String)]
        fn string_const() -> result;

        #[attr(value: any)]
        fn bytes_const() -> result;
    }
}
