//! Structural Wasm GC struct types for user structs and variants.
//!
//! A user struct or variant is a `wasm_gc.struct` whose parameters are its
//! physical field types: the `i32` runtime type descriptor field, then each
//! source field in its Wasm storage representation, and whose supertype is
//! the builtin `Described`. Struct and variant layouts of equal physical
//! fields are therefore one GC type, whatever their source names. Builtin layouts keep their identifier and reserved index.
//!
//! `adt_to_wasm` builds variant types here. After every `adt` operation is
//! lowered, `convert` replaces each user `adt.struct` with its
//! `wasm_gc.struct` wherever it occurs. An `adt.typeref` becomes the
//! structural type of the struct it names, or the abstract struct reference
//! when it names an enum, whose variants have different types.

use tribute_ir::dialect::adt::layout::{get_enum_variants, get_struct_fields};
use trunk_ir::StringRef;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::wasm_gc;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::Module;

use super::descriptors::has_descriptor_field;
use trunk_ir::dialect::core;
use trunk_ir::dialect::wasm;
use trunk_ir::ops::DialectType;

/// The `wasm_gc.struct` of the user struct or closure capture layout `ty`,
/// or `None` when `ty` is neither.
pub fn struct_type(ctx: &mut IrContext, ty: TypeRef) -> Option<TypeRef> {
    if !ctx.get_type(ty).name.is(ctx, "adt.struct") {
        return None;
    }
    if let Some(base) = ctx
        .type_attrs(ty)
        .get_type(tribute_ir::runtime_layout::CLOSURE_BASE_ATTR)
    {
        let fields = get_struct_fields(ctx, ty)?;
        let params: Vec<_> = fields
            .into_iter()
            .map(|(_, field)| field_type(ctx, field))
            .collect();
        return Some(
            wasm_gc::Struct::params(params)
                .supertype(base)
                .intern(ctx)
                .as_type_ref(),
        );
    }
    if !has_descriptor_field(ctx, ty) {
        return None;
    }
    let fields = get_struct_fields(ctx, ty)?;
    Some(described_struct(
        ctx,
        fields.into_iter().map(|(_, field)| field),
    ))
}

/// The `wasm_gc.struct` of variant `tag` of the enum layout `enum_ty`.
pub fn variant_type(ctx: &mut IrContext, enum_ty: TypeRef, tag: StringRef) -> Option<TypeRef> {
    let (_, fields) = get_enum_variants(ctx, enum_ty)?
        .into_iter()
        .find(|(variant, _)| *variant == tag)?;
    Some(described_struct(ctx, fields))
}

/// Replace each user `adt.struct` in `module` with its `wasm_gc.struct`, and
/// each `adt.typeref` with the type of the layout it names.
pub(crate) fn convert(ctx: &mut IrContext, module: Module) {
    crate::closure_lower::substitute_module_types(ctx, module, |ctx, ty| {
        if ctx.get_type(ty).name == tribute_ir::dialect::adt::Typeref::NAME {
            return Some(reference_type(ctx, ty));
        }
        struct_type(ctx, ty)
    });
}

/// The Wasm type of a value of the `adt.typeref` `ty`: the structural type of
/// the user struct its exact alias names, or `wasm.structref` for an enum or
/// an unresolved name.
fn reference_type(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    let named = ctx
        .type_attrs(ty)
        .get_str("name")
        .and_then(|name| ctx.type_alias_by_text(name));
    named
        .and_then(|layout| struct_type(ctx, layout))
        .unwrap_or_else(|| wasm::Structref::params().intern(ctx).as_type_ref())
}

/// A struct of the descriptor field followed by `fields` in their Wasm
/// storage representation, declared a subtype of `Described`.
fn described_struct(ctx: &mut IrContext, fields: impl IntoIterator<Item = TypeRef>) -> TypeRef {
    let descriptor = core::I32::type_ref(ctx);
    let mut params = vec![descriptor];
    for field in fields {
        params.push(field_type(ctx, field));
    }
    let described = super::type_converter::described_adt_type(ctx);
    wasm_gc::Struct::params(params)
        .supertype(described)
        .intern(ctx)
        .as_type_ref()
}

/// The Wasm storage representation of a source field of type `ty`.
///
/// Bool storage is `i32`. A recursive reference is the abstract struct
/// supertype, an array the abstract array supertype, and an enum value the
/// erased reference it is emitted as; a read casts each back to its value's
/// type. Structural types cannot refer to themselves, and an array field may
/// hold its own struct, as a marker's `outer` evidence does. A nested user
/// struct is its own structural type; any other field keeps its type.
fn field_type(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    let data = ctx.get_type(ty);
    if data.name == core::I1::NAME {
        return core::I32::type_ref(ctx);
    }
    if tribute_ir::dialect::adt::Array::matches(ctx, ty) {
        return wasm::Arrayref::params().intern(ctx).as_type_ref();
    }
    if data.name == tribute_ir::dialect::adt::Typeref::NAME {
        return wasm::Structref::params().intern(ctx).as_type_ref();
    }
    if tribute_ir::dialect::adt::Enum::matches(ctx, ty) {
        return wasm::Anyref::params().intern(ctx).as_type_ref();
    }
    struct_type(ctx, ty).unwrap_or(ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::ops::DialectType;
    use trunk_ir::parser::parse_test_module;

    fn alias(ctx: &IrContext, name: &'static str) -> TypeRef {
        ctx.type_alias_by_text(name).expect("fixture alias")
    }

    fn fields(ctx: &IrContext, ty: TypeRef) -> Vec<TypeRef> {
        wasm_gc::Struct::from_type_ref(ctx, ty)
            .expect("structural struct")
            .fields(ctx)
            .to_vec()
    }

    #[test]
    fn structs_and_variants_of_equal_fields_are_one_type() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Point = adt.struct<Point(x: core.f64, y: core.f64)>
  !Size = adt.struct<Size(width: core.f64, height: core.f64)>
  !Shape = adt.enum<Shape { Rect(core.f64, core.f64), Dot() }>
}"#,
        );
        let [point, size, shape] = ["Point", "Size", "Shape"].map(|name| alias(&ctx, name));
        let point = struct_type(&mut ctx, point).expect("user struct");
        let size = struct_type(&mut ctx, size).expect("user struct");
        let rect_tag = ctx.intern_str("Rect");
        let rect = variant_type(&mut ctx, shape, rect_tag).expect("variant");

        assert_eq!(point, size);
        assert_eq!(point, rect);
        let [i32_ty, f64_ty] = [core::I32::type_ref(&mut ctx), core::F64::type_ref(&mut ctx)];
        assert_eq!(fields(&ctx, point), [i32_ty, f64_ty, f64_ty]);
        let described = crate::wasm::type_converter::described_adt_type(&mut ctx);
        let supertype = wasm_gc::Struct::from_type_ref(&ctx, point)
            .expect("structural struct")
            .supertype(&ctx);
        assert_eq!(supertype, Some(described));
    }

    #[test]
    fn convert_resolves_references_to_the_layout_they_name() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Pair = adt.struct<Pair(left: core.i64, right: core.i64)>
  !Shape = adt.enum<Shape { Dot() }>
  wasm.func @f(%pair: adt.typeref<{name = "Pair"}>, %shape: adt.typeref<{name = "Shape"}>, %unknown: adt.typeref<{name = "Missing"}>) -> core.nil {
    wasm.return
  }
}"#,
        );

        convert(&mut ctx, module);

        let function = module.ops(&ctx)[0];
        let entry = ctx.region(ctx.op_region(function, 0).unwrap()).blocks[0];
        let [pair, shape, unknown] = ctx.block_args(entry)[..] else {
            panic!("three parameters")
        };
        let [i32_ty, i64_ty, structref] = [
            core::I32::type_ref(&mut ctx),
            core::I64::type_ref(&mut ctx),
            wasm::Structref::params().intern(&mut ctx).as_type_ref(),
        ];
        assert_eq!(fields(&ctx, ctx.value_ty(pair)), [i32_ty, i64_ty, i64_ty]);
        assert_eq!(ctx.value_ty(shape), structref);
        assert_eq!(ctx.value_ty(unknown), structref);
    }

    #[test]
    fn fields_take_their_wasm_storage_representation() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Closure = adt.struct<_closure(entries: tribute_rt.convention_table_ref), {layout = "closure"}>
  !Inner = adt.struct<Inner(value: core.i64)>
  !E = adt.enum<E { None() }>
  !Outer = adt.struct<Outer(flag: core.i1, next: adt.typeref<{name = "Outer"}>, e: !E, inner: !Inner, f: !Closure)>
}"#,
        );
        let closure = alias(&ctx, "Closure");
        let [outer, inner] = ["Outer", "Inner"].map(|name| alias(&ctx, name));
        let outer = struct_type(&mut ctx, outer).expect("user struct");
        let inner = struct_type(&mut ctx, inner).expect("user struct");
        let i32_ty = core::I32::type_ref(&mut ctx);
        let structref = wasm::Structref::params().intern(&mut ctx).as_type_ref();
        let anyref = wasm::Anyref::params().intern(&mut ctx).as_type_ref();

        assert_eq!(
            fields(&ctx, outer),
            [i32_ty, i32_ty, structref, anyref, inner, closure]
        );
        assert_eq!(struct_type(&mut ctx, closure), None, "builtin layout");
    }
}
