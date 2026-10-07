//! Structural Wasm GC struct types for user structs and variants.
//!
//! A user struct or variant is a `wasm_gc.struct` whose parameters are its
//! physical field types: the `i32` runtime type descriptor field, then each
//! source field in its Wasm storage representation. Struct and variant
//! layouts of equal physical fields are therefore one GC type, whatever their
//! source names. Builtin layouts keep their identifier and reserved index.
//!
//! `adt_to_wasm` builds variant types here. After every `adt` operation is
//! lowered, [`convert`] replaces each user `adt.struct` with its
//! `wasm_gc.struct` wherever it occurs. An `adt.typeref` becomes the
//! structural type of the struct it names, or the abstract struct reference
//! when it names an enum, whose variants have different types.

use tribute_ir::dialect::adt::layout::{get_enum_variants, get_struct_fields};
use trunk_ir::StringRef;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::wasm_gc;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::Module;
use trunk_ir::types::TypeDataBuilder;

use super::descriptors::has_descriptor_field;

/// The `wasm_gc.struct` of the user struct layout `ty`, or `None` when `ty`
/// is not a user `adt.struct`.
pub fn struct_type(ctx: &mut IrContext, ty: TypeRef) -> Option<TypeRef> {
    if !ctx.types().is_dialect(ty, "adt", "struct") || !has_descriptor_field(ctx, ty) {
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
pub fn convert(ctx: &mut IrContext, module: Module) {
    crate::closure_lower::substitute_module_types(ctx, module, |ctx, ty| {
        if ctx.types().is_dialect(ty, "adt", "typeref") {
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
        .get_type(ty)
        .attrs
        .get_str(ctx, "name")
        .and_then(|name| ctx.type_alias_by_text(name));
    named
        .and_then(|layout| struct_type(ctx, layout))
        .unwrap_or_else(|| intern(ctx, "wasm", "structref"))
}

/// A struct of the descriptor field followed by `fields` in their Wasm
/// storage representation.
fn described_struct(ctx: &mut IrContext, fields: impl IntoIterator<Item = TypeRef>) -> TypeRef {
    let descriptor = intern(ctx, "core", "i32");
    let mut params = vec![descriptor];
    for field in fields {
        params.push(field_type(ctx, field));
    }
    wasm_gc::r#struct(ctx, params).as_type_ref()
}

/// The Wasm storage representation of a source field of type `ty`.
///
/// Bool storage is `i32`. A recursive reference is the abstract struct
/// supertype, and an enum value is the erased reference it is emitted as. A
/// nested user struct is its own structural type; any other field keeps its
/// type.
fn field_type(ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
    let data = ctx.get_type(ty);
    if data.dialect == "core" && data.name == "i1" {
        return intern(ctx, "core", "i32");
    }
    if data.dialect == "adt" {
        if data.name == "typeref" {
            return intern(ctx, "wasm", "structref");
        }
        if data.name == "enum" {
            return intern(ctx, "wasm", "anyref");
        }
    }
    struct_type(ctx, ty).unwrap_or(ty)
}

fn intern(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new(dialect, name).build())
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
        let [i32_ty, f64_ty] = [
            intern(&mut ctx, "core", "i32"),
            intern(&mut ctx, "core", "f64"),
        ];
        assert_eq!(fields(&ctx, point), [i32_ty, f64_ty, f64_ty]);
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
            intern(&mut ctx, "core", "i32"),
            intern(&mut ctx, "core", "i64"),
            intern(&mut ctx, "wasm", "structref"),
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
  !Closure = adt.struct<_closure(table_idx: core.i32, env: wasm.anyref), {layout = "closure"}>
  !Inner = adt.struct<Inner(value: core.i64)>
  !E = adt.enum<E { None() }>
  !Outer = adt.struct<Outer(flag: core.i1, next: adt.typeref<{name = "Outer"}>, e: !E, inner: !Inner, f: !Closure)>
}"#,
        );
        let closure = alias(&ctx, "Closure");
        let [outer, inner] = ["Outer", "Inner"].map(|name| alias(&ctx, name));
        let outer = struct_type(&mut ctx, outer).expect("user struct");
        let inner = struct_type(&mut ctx, inner).expect("user struct");
        let i32_ty = intern(&mut ctx, "core", "i32");
        let structref = intern(&mut ctx, "wasm", "structref");
        let anyref = intern(&mut ctx, "wasm", "anyref");

        assert_eq!(
            fields(&ctx, outer),
            [i32_ty, i32_ty, structref, anyref, inner, closure]
        );
        assert_eq!(struct_type(&mut ctx, closure), None, "builtin layout");
    }
}
