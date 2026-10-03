//! Wasm runtime type descriptor declarations.
//!
//! Right before `adt_to_wasm`, the last pass to read nominal layouts, this
//! step declares each user allocation descriptor as a `tribute_rtti.layout`:
//! its number and how the runtime reads each of its fields. `adt_to_wasm`
//! stores the number as the object's first field and erases the
//! declarations. Builtin layouts carry no descriptor field; their reserved GC
//! type index is their descriptor number, and user numbers follow it.

use std::collections::HashSet;
use std::ops::ControlFlow;

use tribute_ir::dialect::tribute_rtti::{
    self, FieldKind, allocation_descriptor, descriptor_field_types,
};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::Module;
use trunk_ir::walk::{WalkAction, walk_region};
use trunk_ir_wasm_backend::gc_types::FIRST_USER_TYPE_IDX;
use trunk_ir_wasm_backend::passes::wasm_gc_to_wasm::builtin_type_idx;

/// Whether objects of the struct or enum layout `ty` carry a leading
/// descriptor field. Builtin layouts do not.
pub fn has_descriptor_field(ctx: &IrContext, ty: TypeRef) -> bool {
    builtin_type_idx(ctx, ty).is_none()
}

/// Declare the module's user allocation descriptors in allocation order.
pub fn declare(ctx: &mut IrContext, module: Module) {
    let mut descriptors = Vec::new();
    let mut seen = HashSet::new();
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
    let mut index = FIRST_USER_TYPE_IDX;
    for (ty, tag) in descriptors {
        // An allocation whose type does not resolve to its layout stays
        // undeclared; `adt_to_wasm` leaves it for the backend boundary to reject.
        let Ok(field_types) = descriptor_field_types(ctx, ty, tag) else {
            continue;
        };
        let fields = field_types
            .into_iter()
            .map(|field| field_kind(ctx, field))
            .collect::<Vec<_>>();
        let layout = tribute_rtti::Layout::declare(ctx, location, ty, tag, index, &fields);
        ctx.push_op(module_block, layout.op_ref());
        index += 1;
    }
}

/// How the runtime reads a field of semantic type `ty`. Without an ownership
/// plan, a GC reference whose type names its layout is managed, and any other
/// GC reference is dynamic.
fn field_kind(ctx: &IrContext, ty: TypeRef) -> FieldKind {
    let data = ctx.get_type(ty);
    if data.dialect == Symbol::new("adt")
        || (data.dialect == Symbol::new("core") && data.name == Symbol::new("array"))
    {
        return FieldKind::Managed;
    }
    let dynamic = (data.dialect == Symbol::new("tribute_rt")
        && (data.name == Symbol::new("anyref") || data.name == Symbol::new("intref")))
        || (data.dialect == Symbol::new("wasm")
            && data.name.with_str(|name| {
                matches!(
                    name,
                    "anyref" | "eqref" | "structref" | "arrayref" | "i31ref"
                )
            }));
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
  !S = adt.struct<S(next: adt.typeref<{name = "S"}>, any: tribute_rt.anyref, n: tribute_rt.nat)>
  !Closure = adt.struct<_closure(table_idx: core.i32, env: wasm.anyref), {layout = "closure"}>
  !E = adt.enum<{name = "E", variants = [["None", []], ["Some", [core.f64]]]}>
  wasm.func @main(%next: adt.typeref<{name = "S"}>, %any: tribute_rt.anyref, %n: tribute_rt.nat, %x: core.f64) -> core.nil {
    %one = wasm.i32_const {value = 1} : core.i32
    %some = adt.variant_new %x {type = !E, tag = "Some"} : !E
    %s = adt.struct_new %next, %any, %n {type = !S} : !S
    %closure = adt.struct_new %one, %any {type = !Closure} : !Closure
    %again = adt.variant_new %x {type = !E, tag = "Some"} : !E
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
                    FIRST_USER_TYPE_IDX,
                    Some("Some"),
                    vec![FieldKind::Float { width: 64 }]
                ),
                (
                    FIRST_USER_TYPE_IDX + 1,
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
    fn unresolved_descriptors_are_skipped_without_a_gap() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct<S(x: core.f64)>
  !E = adt.enum<{name = "E", variants = [["None", []], ["Some", [core.f64]]]}>
  wasm.func @main(%x: core.f64) -> core.nil {
    %some = adt.variant_new %x {type = !E, tag = "Some"} : !E
    %erased = adt.variant_new %x {type = tribute_rt.anyref, tag = "Some"} : tribute_rt.anyref
    %missing = adt.variant_new %x {type = !E, tag = "Other"} : !E
    %s = adt.struct_new %x {type = !S} : !S
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
                (FIRST_USER_TYPE_IDX, Some("Some")),
                (FIRST_USER_TYPE_IDX + 1, None),
            ]
        );
    }
}
