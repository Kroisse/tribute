//! Physical assignability of Wasm values.
//!
//! The backend's single rule for which value types may flow into a slot of
//! another type without a runtime cast. Validation, emission, target passes,
//! and callers outside this crate share it.

use trunk_ir::IrContext;
use trunk_ir::Symbol;
use trunk_ir::refs::TypeRef;

use crate::emit::helpers::is_type;

/// Whether this IR type is registered by the backend as a concrete WasmGC
/// struct reference, and is therefore assignable to the abstract `wasm.structref`
/// without a runtime cast.
///
/// Registration follows the same structure the rest of the backend uses: builtin
/// layouts at their reserved indices (`@bytes`, `_closure`,
/// `_Marker`, ...) and the ADT types that
/// `emit::gc_types_collection::normalize_type_for_gc` physicalizes as the
/// abstract struct supertype (`adt.typeref` and concrete variant instances
/// carrying `base_enum`). An ADT spelling without that registration evidence
/// proves nothing and stays rejected.
fn is_registered_gc_struct_reference(ctx: &IrContext, ty: TypeRef) -> bool {
    if let Some(index) = crate::passes::wasm_gc_to_wasm::builtin_type_idx(ctx, ty) {
        return crate::gc_types::is_builtin_struct_index(index);
    }
    let data = ctx.get_type(ty);
    data.dialect == Symbol::new("adt")
        && (data.name == Symbol::new("typeref") || data.attrs.get_type("base_enum").is_some())
}

/// Whether this IR type is registered by the backend as a concrete WasmGC array
/// reference, and is therefore assignable to the abstract `wasm.arrayref`
/// without a runtime cast. Only builtin array layouts (Bytes backing arrays and
/// the Evidence array) qualify; `core.array` spellings are handled by the
/// abstract array rule and never acquire a concrete index on their own.
fn is_registered_gc_array_reference(ctx: &IrContext, ty: TypeRef) -> bool {
    crate::passes::wasm_gc_to_wasm::builtin_type_idx(ctx, ty)
        .is_some_and(|index| !crate::gc_types::is_builtin_struct_index(index))
}

/// Whether an argument can satisfy an indirect-tail parameter after the Wasm
/// backend's physical type mapping.
///
/// This is the narrow physical assignability relation shared by argument,
/// result, exact indirect/tail signature, and CPS dispatch payload checks. It
/// only accepts widenings that emission performs without a runtime cast.
pub fn is_wasm_physical_argument_assignable(
    ctx: &IrContext,
    argument: TypeRef,
    parameter: TypeRef,
) -> bool {
    if argument == parameter {
        return true;
    }

    // Two spellings of one builtin layout are the same concrete GC type.
    let builtin = |ty| crate::passes::wasm_gc_to_wasm::builtin_type_idx(ctx, ty);
    if builtin(argument).is_some_and(|index| builtin(parameter) == Some(index)) {
        return true;
    }

    // `core.i1` is represented by an i32 in the Wasm value space.
    if is_type(ctx, argument, "core", "i1") && is_type(ctx, parameter, "core", "i32") {
        return true;
    }

    let argument_is_typeref = is_type(ctx, argument, "adt", "typeref");
    let argument_is_structref = is_type(ctx, argument, "wasm", "structref");
    let parameter_is_structref = is_type(ctx, parameter, "wasm", "structref");
    let parameter_is_arrayref = is_type(ctx, parameter, "wasm", "arrayref");
    let parameter_is_anyref = is_type(ctx, parameter, "wasm", "anyref");

    // `adt.typeref` is emitted as the abstract Wasm `structref` type.
    if argument_is_typeref && (parameter_is_structref || parameter_is_anyref) {
        return true;
    }

    // Registered concrete GC references widen to the abstract heap type the
    // emission chose for the slot. Reaching a concrete type from an abstract one
    // still requires `wasm.ref_cast`, so the reverse directions remain false.
    if parameter_is_structref && is_registered_gc_struct_reference(ctx, argument) {
        return true;
    }
    if parameter_is_arrayref && is_registered_gc_array_reference(ctx, argument) {
        return true;
    }
    // Every registered reference denotes a struct or array, and `anyref` is their
    // common supertype, so the same registration evidence satisfies an `anyref`
    // slot. `core.array` spellings are emitted as the abstract array reference.
    if parameter_is_anyref
        && (is_registered_gc_struct_reference(ctx, argument)
            || is_registered_gc_array_reference(ctx, argument)
            || is_type(ctx, argument, "core", "array"))
    {
        return true;
    }

    // An unregistered `adt.struct` spelling is emitted as the erased `anyref`
    // reference, so it satisfies an `anyref` slot. A variant-marked type is not
    // a second spelling of that erasure: it must carry registration evidence
    // (`base_enum`), which the registered-struct rule above already accepts, and
    // a variant instance without it is malformed rather than erased.
    if parameter_is_anyref && is_type(ctx, argument, "adt", "struct") {
        return true;
    }

    let argument_is_core_array = is_type(ctx, argument, "core", "array");
    if argument_is_core_array && parameter_is_arrayref {
        return true;
    }

    // WasmGC abstract-reference upcasts. The reverse directions are checked
    // downcasts and therefore intentionally remain false here.
    let argument_is_wasm_gc_ref = is_type(ctx, argument, "wasm", "i31ref")
        || argument_is_structref
        || is_type(ctx, argument, "wasm", "arrayref");
    argument_is_wasm_gc_ref && parameter_is_anyref
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn spellings_of_one_builtin_layout_are_the_same_type() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !data = core.array<core.i8, {layout = "bytes_data"}>
  !bytes = adt.struct<@_Bytes(@data: !data, @offset: core.i32, @len: core.i32), {layout = "bytes"}>
  !closure = adt.struct<@_closure(@table_idx: core.i32, @env: wasm.anyref), {layout = "closure"}>
  !view = adt.struct<@BytesView(@data: !data, @start: core.i32, @count: core.i32), {layout = "bytes"}>
  !plain = core.array<core.i8>
}"#,
        );
        let alias = |name: &'static str| {
            ctx.type_alias_by_name(Symbol::new(name))
                .expect("fixture alias")
        };

        assert!(is_wasm_physical_argument_assignable(
            &ctx,
            alias("view"),
            alias("bytes")
        ));
        assert!(is_wasm_physical_argument_assignable(
            &ctx,
            alias("bytes"),
            alias("view")
        ));
        assert!(!is_wasm_physical_argument_assignable(
            &ctx,
            alias("bytes"),
            alias("closure")
        ));
        assert!(!is_wasm_physical_argument_assignable(
            &ctx,
            alias("plain"),
            alias("data")
        ));
    }
}
