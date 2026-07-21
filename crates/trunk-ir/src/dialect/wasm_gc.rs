//! Typed intermediate operations for WebAssembly GC lowering.
//!
//! Unlike the indexed `wasm` dialect, these operations identify nominal heap
//! types with `TypeRef`. A module-wide layout pass must fully convert them to
//! `wasm` operations before binary emission.

#[trunk_ir::dialect]
mod wasm_gc {
    #[attr(r#type: Type)]
    fn struct_new(#[rest] fields: ()) -> result {}

    #[attr(r#type: Type, field_idx: u32)]
    fn struct_get(r#ref: ()) -> result {}

    #[attr(r#type: Type, field_idx: u32)]
    fn struct_set(r#ref: (), value: ()) {}

    #[attr(r#type: Type)]
    fn array_new(size: (), init: ()) -> result {}

    #[attr(r#type: Type)]
    fn array_new_default(size: ()) -> result {}

    #[attr(r#type: Type, data_idx: u32)]
    fn array_new_data(offset: (), size: ()) -> result {}

    #[attr(r#type: Type)]
    fn array_get(r#ref: (), index: ()) -> result {}

    #[attr(r#type: Type)]
    fn array_get_s(r#ref: (), index: ()) -> result {}

    #[attr(r#type: Type)]
    fn array_get_u(r#ref: (), index: ()) -> result {}

    #[attr(r#type: Type)]
    fn array_set(r#ref: (), index: (), value: ()) {}

    #[attr(dst_type: Type, src_type: Type)]
    fn array_copy(dst: (), dst_offset: (), src: (), src_offset: (), len: ()) {}

    #[attr(target_type: Type)]
    fn ref_null() -> result {}

    #[attr(target_type: Type)]
    fn ref_cast(r#ref: ()) -> result {}

    #[attr(target_type: Type)]
    fn ref_test(r#ref: ()) -> result {}
}
