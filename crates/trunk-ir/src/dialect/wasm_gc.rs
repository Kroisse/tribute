//! Typed intermediate operations for WebAssembly GC lowering.
//!
//! Unlike the indexed `wasm` dialect, these operations identify nominal heap
//! types with `TypeRef`. A module-wide layout pass must fully convert them to
//! `wasm` operations before binary emission.

#[trunk_ir::dialect]
mod wasm_gc {
    fn struct_new(r#type: Attr<Type>, fields: Variadic<_>) -> Value<_> {}

    fn struct_get(r#type: Attr<Type>, field_idx: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    fn struct_set(r#type: Attr<Type>, field_idx: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    fn array_new(r#type: Attr<Type>, size: Value<_>, init: Value<_>) -> Value<_> {}

    fn array_new_default(r#type: Attr<Type>, size: Value<_>) -> Value<_> {}

    fn array_new_data(
        r#type: Attr<Type>,
        data_idx: Attr<u32>,
        offset: Value<_>,
        size: Value<_>,
    ) -> Value<_> {
    }

    fn array_get(r#type: Attr<Type>, r#ref: Value<_>, index: Value<_>) -> Value<_> {}

    fn array_get_s(r#type: Attr<Type>, r#ref: Value<_>, index: Value<_>) -> Value<_> {}

    fn array_get_u(r#type: Attr<Type>, r#ref: Value<_>, index: Value<_>) -> Value<_> {}

    fn array_set(r#type: Attr<Type>, r#ref: Value<_>, index: Value<_>, value: Value<_>) {}

    fn array_copy(
        dst_type: Attr<Type>,
        src_type: Attr<Type>,
        dst: Value<_>,
        dst_offset: Value<_>,
        src: Value<_>,
        src_offset: Value<_>,
        len: Value<_>,
    ) {
    }

    fn ref_null(target_type: Attr<Type>) -> Value<_> {}

    fn ref_cast(target_type: Attr<Type>, r#ref: Value<_>) -> Value<_> {}

    fn ref_test(target_type: Attr<Type>, r#ref: Value<_>) -> Value<_> {}
}
