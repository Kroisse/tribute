//! Physical assignability of Wasm values.
//!
//! The backend's single rule for which value types may flow into a slot of
//! another type without a runtime cast. Validation, emission, target passes,
//! and callers outside this crate share it.

use trunk_ir::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::TypeRef;

/// Whether this IR type is registered by the backend as a concrete WasmGC
/// struct reference, and is therefore assignable to the abstract `wasm.structref`
/// without a runtime cast.
///
/// Registration follows the same structure the rest of the backend uses: builtin
/// layouts at their reserved indices (`@bytes`, `_closure`, ...) and the
/// structural `wasm_gc.struct` types, each of which
/// receives an index. Any other spelling proves nothing and stays rejected.
fn is_registered_gc_struct_reference(ctx: &IrContext, ty: TypeRef) -> bool {
    if let Some(index) = crate::passes::wasm_gc_to_wasm::builtin_type_idx(ctx, ty) {
        return crate::gc_types::is_builtin_struct_index(index);
    }
    trunk_ir::dialect::wasm_gc::Struct::matches(ctx, ty)
}

/// Whether this IR type is registered by the backend as a concrete WasmGC array
/// reference, and is therefore assignable to the abstract `wasm.arrayref`
/// without a runtime cast. Only builtin array layouts (Bytes backing arrays)
/// qualify; `wasm_gc.array` spellings are handled by the
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

    // A structural struct reaches each supertype it declares without a cast.
    let mut declared = argument;
    while let Some(supertype) = trunk_ir::dialect::wasm_gc::Struct::from_type_ref(ctx, declared)
        .and_then(|gc_struct| gc_struct.supertype(ctx))
    {
        if supertype == parameter
            || builtin(supertype).is_some_and(|index| builtin(parameter) == Some(index))
        {
            return true;
        }
        declared = supertype;
    }

    // `core.i1` is represented by an i32 in the Wasm value space.
    if trunk_ir::dialect::core::I1::matches(ctx, argument)
        && trunk_ir::dialect::core::I32::matches(ctx, parameter)
    {
        return true;
    }

    let argument_is_structref = trunk_ir::dialect::wasm::Structref::matches(ctx, argument);
    let parameter_is_structref = trunk_ir::dialect::wasm::Structref::matches(ctx, parameter);
    let parameter_is_arrayref = trunk_ir::dialect::wasm::Arrayref::matches(ctx, parameter);
    let parameter_is_anyref = trunk_ir::dialect::wasm::Anyref::matches(ctx, parameter);

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
    // slot. `wasm_gc.array` spellings are emitted as the abstract array reference.
    if parameter_is_anyref
        && (is_registered_gc_struct_reference(ctx, argument)
            || is_registered_gc_array_reference(ctx, argument)
            || trunk_ir::dialect::wasm_gc::Array::matches(ctx, argument))
    {
        return true;
    }

    let argument_is_gc_array = trunk_ir::dialect::wasm_gc::Array::matches(ctx, argument);
    if argument_is_gc_array && parameter_is_arrayref {
        return true;
    }

    // WasmGC abstract-reference upcasts. The reverse directions are checked
    // downcasts and therefore intentionally remain false here.
    let argument_is_wasm_gc_ref = trunk_ir::dialect::wasm::I31ref::matches(ctx, argument)
        || argument_is_structref
        || trunk_ir::dialect::wasm::Arrayref::matches(ctx, argument);
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
  !data = wasm_gc.array<core.i8, {layout = "bytes_data"}>
  !bytes = test.layout<!data, core.i32, core.i32, {layout = "bytes"}>
  !closure = test.layout<core.i32, wasm.anyref, {layout = "closure"}>
  !view = test.view<!data, core.i32, core.i32, {layout = "bytes"}>
  !plain = wasm_gc.array<core.i8>
}"#,
        );
        let alias = |name: &'static str| ctx.type_alias_by_text(name).expect("fixture alias");

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
