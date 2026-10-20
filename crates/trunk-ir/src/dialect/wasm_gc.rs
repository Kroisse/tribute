//! Typed intermediate operations for WebAssembly GC lowering.
//!
//! Unlike the indexed `wasm` dialect, these operations identify heap types
//! with `TypeRef`. A module-wide layout pass must fully convert them to
//! `wasm` operations before binary emission.

use crate::attr_kind::Type;
use crate::dialect::core::{Array, I32};

#[trunk_ir::dialect]
mod wasm_gc {
    /// A structural GC struct: its field types in order, each a Wasm
    /// storage representation. Equal field lists with the same supertype
    /// are one GC type. `supertype`, when present, is the GC struct this
    /// struct is declared a subtype of; its fields are a prefix of these.
    struct Struct<#[rest] Fields> {
        supertype: Option<Attr<Type>>,
    }

    fn struct_new(r#type: Attr<Type>, fields: Variadic<_>) -> Value<_> {}

    fn struct_get(r#type: Attr<Type>, field_idx: Attr<u32>, r#ref: Value<_>) -> Value<_> {}

    fn struct_set(r#type: Attr<Type>, field_idx: Attr<u32>, r#ref: Value<_>, value: Value<_>) {}

    fn array_new<A: Array>(
        r#type: Attr<TypeOf<A>>,
        size: Value<I32>,
        init: Value<A::Element>,
    ) -> Value<A> {
    }

    fn array_new_default<A: Array>(r#type: Attr<TypeOf<A>>, size: Value<I32>) -> Value<A> {}

    fn array_new_fixed<A: Array>(
        r#type: Attr<TypeOf<A>>,
        elements: Variadic<A::Element>,
    ) -> Value<A> {
    }

    fn array_new_data<A: Array>(
        r#type: Attr<TypeOf<A>>,
        data_idx: Attr<u32>,
        offset: Value<I32>,
        size: Value<I32>,
    ) -> Value<A> {
    }

    fn array_get<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<I32>,
    ) -> Value<A::Element> {
    }

    fn array_get_s<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<I32>,
    ) -> Value<_> {
    }

    fn array_get_u<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<I32>,
    ) -> Value<_> {
    }

    fn array_set<A: Array>(
        r#type: Attr<TypeOf<A>>,
        r#ref: Value<A>,
        index: Value<I32>,
        value: Value<A::Element>,
    ) {
    }

    fn array_copy<D: Array, S: Array>(
        dst_type: Attr<TypeOf<D>>,
        src_type: Attr<TypeOf<S>>,
        dst: Value<D>,
        dst_offset: Value<I32>,
        src: Value<S>,
        src_offset: Value<I32>,
        len: Value<I32>,
    ) {
    }

    fn ref_null(target_type: Attr<Type>) -> Value<_> {}

    fn ref_cast(target_type: Attr<Type>, r#ref: Value<_>) -> Value<_> {}

    fn ref_test(target_type: Attr<Type>, r#ref: Value<_>) -> Value<_> {}
}

#[cfg(test)]
mod tests {
    use crate::context::IrContext;
    use crate::parser::parse_test_module;
    use crate::validation::validate_op_schemas;

    fn schema_ok(body: &str) -> bool {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                "core.module @test {{\n  !A = core.array<core.i32>\n  !B = core.array<core.i64>\n  func.func @f(%a: !A, %b: !B, %n: core.i32, %w: core.i64, %x: core.index) -> core.nil {{\n    {body}\n    func.return\n  }}\n}}"
            ),
        );
        validate_op_schemas(&ctx, module.op()).is_ok()
    }

    #[test]
    fn array_indices_and_lengths_are_i32() {
        assert!(schema_ok(
            "%e = wasm_gc.array_get %a, %n {type = !A} : core.i32"
        ));
        assert!(!schema_ok(
            "%e = wasm_gc.array_get %a, %w {type = !A} : core.i32"
        ));
        assert!(!schema_ok(
            "%e = wasm_gc.array_get %a, %x {type = !A} : core.i32"
        ));
        assert!(!schema_ok(
            "%e = wasm_gc.array_get_u %a, %x {type = !A} : core.i32"
        ));
        assert!(!schema_ok("wasm_gc.array_set %a, %x, %n {type = !A}"));
        assert!(schema_ok(
            "%v = wasm_gc.array_new_default %n {type = !A} : !A"
        ));
        assert!(!schema_ok(
            "%v = wasm_gc.array_new_default %w {type = !A} : !A"
        ));
        assert!(!schema_ok("%v = wasm_gc.array_new %x, %n {type = !A} : !A"));
        assert!(!schema_ok(
            "%v = wasm_gc.array_new_data %n, %w {type = !A, data_idx = 0} : !A"
        ));
    }

    #[test]
    fn array_new_fixed_elements_are_its_element_type() {
        assert!(schema_ok("%v = wasm_gc.array_new_fixed {type = !A} : !A"));
        assert!(schema_ok(
            "%v = wasm_gc.array_new_fixed %n, %n {type = !A} : !A"
        ));
        assert!(!schema_ok(
            "%v = wasm_gc.array_new_fixed %w {type = !A} : !A"
        ));
        assert!(!schema_ok(
            "%v = wasm_gc.array_new_fixed %n {type = !A} : !B"
        ));
    }

    #[test]
    fn array_copy_binds_its_array_types() {
        assert!(schema_ok(
            "wasm_gc.array_copy %a, %n, %b, %n, %n {dst_type = !A, src_type = !B}"
        ));
        assert!(!schema_ok(
            "wasm_gc.array_copy %a, %n, %b, %n, %n {dst_type = !A, src_type = !A}"
        ));
        assert!(!schema_ok(
            "wasm_gc.array_copy %a, %n, %b, %n, %w {dst_type = !A, src_type = !B}"
        ));
    }

    #[test]
    fn array_new_data_result_is_its_declared_array() {
        assert!(schema_ok(
            "%v = wasm_gc.array_new_data %n, %n {type = !A, data_idx = 0} : !A"
        ));
        assert!(!schema_ok(
            "%v = wasm_gc.array_new_data %n, %n {type = !A, data_idx = 0} : !B"
        ));
    }
}
