//! Wasm runtime type descriptor declarations.
//!
//! Right before `adt_to_wasm`, the last pass to read nominal layouts, this
//! step declares each user allocation descriptor as a `tribute_rtti.layout`:
//! its number and how the runtime reads each of its fields. `adt_to_wasm`
//! stores the number as the object's first field and erases the
//! declarations. Builtin layouts carry no descriptor field; their reserved GC
//! type index is their descriptor number, and user numbers follow it.
//!
//! An enum takes one number per declared variant, consecutive in
//! declaration order, when its first variant is allocated, so a variant's
//! position is its number minus the enum's first. A variant the module never
//! allocates keeps its number undeclared.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::ops::ControlFlow;

use tribute_ir::dialect::adt;
use tribute_ir::dialect::tribute_rtti::{
    self, FieldKind, allocation_descriptor, descriptor_field_types,
};
use trunk_ir::context::IrContext;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::Module;
use trunk_ir::walk::{WalkAction, walk_region};
use trunk_ir_wasm_backend::gc_types::FIRST_USER_TYPE_IDX;
use trunk_ir_wasm_backend::passes::wasm_gc_to_wasm::builtin_type_idx;

/// Whether objects of the struct or enum layout `ty` carry a leading
/// descriptor field. Builtin and closure layouts do not.
pub fn has_descriptor_field(ctx: &IrContext, ty: TypeRef) -> bool {
    builtin_type_idx(ctx, ty).is_none() && !crate::closure_lower::is_closure_layout(ctx, ty)
}

/// Declare the module's user allocation descriptors in allocation order,
/// each enum's variants in one block.
pub(crate) fn declare(ctx: &mut IrContext, module: Module) {
    let mut descriptors = Vec::new();
    let mut seen = HashSet::default();
    if let Some(body) = module.body(ctx) {
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            if let Some(descriptor) = allocation_descriptor(ctx, op)
                && has_descriptor_field(ctx, descriptor.0)
                && seen.insert(descriptor)
            {
                descriptors.push(descriptor);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    let Some(module_block) = module.first_block(ctx) else {
        return;
    };
    let location = ctx.op(module.op()).location;
    let mut next = FIRST_USER_TYPE_IDX;
    let mut enum_blocks: HashMap<TypeRef, u32> = HashMap::default();
    for (ty, tag) in descriptors {
        // An allocation whose type does not resolve to its layout stays
        // undeclared; `adt_to_wasm` leaves it for the backend boundary to reject.
        let Ok(field_types) = descriptor_field_types(ctx, ty, tag) else {
            continue;
        };
        let index = match tag {
            None => {
                next += 1;
                next - 1
            }
            Some(position) => {
                let count = adt::Enum::from_type_ref(ctx, ty)
                    .expect("a variant descriptor's layout is an adt.enum")
                    .variant_count(ctx) as u32;
                let first = *enum_blocks.entry(ty).or_insert_with(|| {
                    next += count;
                    next - count
                });
                first + position
            }
        };
        let fields = field_types
            .into_iter()
            .map(|field| field_kind(ctx, field))
            .collect::<Vec<_>>();
        let layout = tribute_rtti::Layout::declare(ctx, location, ty, tag, index, &fields);
        ctx.push_op(module_block, layout.op_ref());
    }
}

/// How the runtime reads a field of semantic type `ty`. Without an ownership
/// plan, a GC reference whose type names its layout is managed, and any other
/// GC reference is dynamic.
fn field_kind(ctx: &IrContext, ty: TypeRef) -> FieldKind {
    let data = ctx.get_type(ty);
    if data.name.dialect(ctx) == tribute_ir::dialect::adt::DIALECT_NAME {
        return FieldKind::Managed;
    }
    let dynamic = {
        use trunk_ir::dialect::wasm;
        [
            tribute_ir::dialect::tribute_rt::Anyref::NAME,
            tribute_ir::dialect::tribute_rt::Intref::NAME,
            wasm::Anyref::NAME,
            wasm::Eqref::NAME,
            wasm::Structref::NAME,
            wasm::Arrayref::NAME,
            wasm::I31ref::NAME,
        ]
        .contains(&data.name)
    };
    if dynamic {
        return FieldKind::Dynamic;
    }
    FieldKind::scalar(ctx, ty).unwrap_or(FieldKind::Raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn user_descriptors_are_declared_in_allocation_order() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<"S", (adt.struct<"S">,tribute_rt.anyref, tribute_rt.nat)>
  !Closure = adt.struct<(tribute_rt.convention_table_ref), {layout = "closure"}>
  !E = adt.enum<"E", [(), (core.f64)]>
  wasm.func @main(%next: adt.struct<"S">, %any: tribute_rt.anyref, %n: tribute_rt.nat, %x: core.f64) -> core.nil {
    %one = wasm.i32_const 1 : core.i32
    %some = adt.variant_new 1(%x) : (core.f64) -> !E
    %s = adt.struct_new(%next, %any, %n) : (adt.struct<"S">, tribute_rt.anyref, tribute_rt.nat) -> !S
    %closure = adt.struct_new(%one) : (core.i32) -> !Closure
    %again = adt.variant_new 1(%x) : (core.f64) -> !E
    wasm.return
  }
}"#,
        );

        declare(&mut ctx, module);

        let layouts = tribute_rtti::Layout::declared(&ctx, module);
        let summary = layouts
            .iter()
            .map(|layout| {
                (
                    layout.index(&ctx),
                    layout.tag(&ctx),
                    layout.field_kinds(&ctx),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                (
                    FIRST_USER_TYPE_IDX + 1,
                    Some(1),
                    vec![FieldKind::Float { width: 64 }]
                ),
                (
                    FIRST_USER_TYPE_IDX + 2,
                    None,
                    vec![
                        FieldKind::Managed,
                        FieldKind::Dynamic,
                        FieldKind::Int {
                            width: 32,
                            signed: false
                        }
                    ]
                ),
            ]
        );
    }

    #[test]
    fn unresolved_descriptors_are_skipped() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<"S", (core.f64)>
  !E = adt.enum<"E", [(), (core.f64)]>
  wasm.func @main(%x: core.f64) -> core.nil {
    %some = adt.variant_new 1(%x) : (core.f64) -> !E
    %s = adt.struct_new(%x) : (core.f64) -> !S
    wasm.return
  }
}"#,
        );

        declare(&mut ctx, module);

        let summary = tribute_rtti::Layout::declared(&ctx, module)
            .iter()
            .map(|layout| (layout.index(&ctx), layout.tag(&ctx)))
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                (FIRST_USER_TYPE_IDX + 1, Some(1)),
                (FIRST_USER_TYPE_IDX + 2, None),
            ]
        );
    }

    #[test]
    fn each_enum_numbers_its_variants_in_declaration_order() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<"S", (core.f64)>
  !E = adt.enum<"E", [(), (core.f64), (), (core.f64)]>
  wasm.func @main(%x: core.f64) -> core.nil {
    %d = adt.variant_new 3(%x) : (core.f64) -> !E
    %s = adt.struct_new(%x) : (core.f64) -> !S
    %b = adt.variant_new 1(%x) : (core.f64) -> !E
    wasm.return
  }
}"#,
        );

        declare(&mut ctx, module);

        let summary = tribute_rtti::Layout::declared(&ctx, module)
            .iter()
            .map(|layout| (layout.index(&ctx), layout.tag(&ctx)))
            .collect::<Vec<_>>();
        // `A` and `C` are never allocated but keep their numbers.
        assert_eq!(
            summary,
            [
                (FIRST_USER_TYPE_IDX + 3, Some(3)),
                (FIRST_USER_TYPE_IDX + 4, None),
                (FIRST_USER_TYPE_IDX + 1, Some(1)),
            ]
        );
    }
}
