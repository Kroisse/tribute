//! Typed intermediate operations for WebAssembly GC lowering.
//!
//! Unlike the indexed `wasm` dialect, these operations identify heap types
//! with `TypeRef`. A module-wide layout pass must fully convert them to
//! `wasm` operations before binary emission.

use crate::attr_kind::Type;
use crate::dialect::core::{Array, IntegerLike};

#[trunk_ir::dialect]
mod wasm_gc {
    /// A structural GC struct: its field types in order, each a Wasm
    /// storage representation. Equal field lists are one GC type.
    struct Struct<#[rest] Fields>;

    fn struct_new(r#type: Attr<Type>, fields: Variadic<_>) -> Value<_> {}

    fn struct_get(r#type: Attr<Type>, field_idx: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    fn struct_set(r#type: Attr<Type>, field_idx: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    fn array_new<A: Array>(
        r#type: Attr<TypeOf<A>>,
        size: Value<impl IntegerLike>,
        init: Value<A::Element>,
    ) -> Value<A> {
    }

    fn array_new_default<A: Array>(
        r#type: Attr<TypeOf<A>>,
        size: Value<impl IntegerLike>,
    ) -> Value<A> {
    }

    fn array_new_data(
        r#type: Attr<Type>,
        data_idx: Attr<u32>,
        offset: Value<_>,
        size: Value<_>,
    ) -> Value<_> {
    }

    fn array_get<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<impl IntegerLike>,
    ) -> Value<A::Element> {
    }

    fn array_get_s<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<impl IntegerLike>,
    ) -> Value<_> {
    }

    fn array_get_u<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<impl IntegerLike>,
    ) -> Value<_> {
    }

    fn array_set<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<impl IntegerLike>,
        value: Value<A::Element>,
    ) {
    }

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
