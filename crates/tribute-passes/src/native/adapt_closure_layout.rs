//! Adapt semantic closure allocations to the native closure layout.
//!
//! The semantic `_closure` struct stores a table index and a managed
//! environment. The native backend stores a function pointer and a raw
//! environment pointer instead. This step rewrites `adt.struct_new` and
//! `adt.struct_get` on the semantic layout, and the `tribute_rtti.layout`
//! declaring it, to the native layout.
//!
//! It runs after native ownership planning and RC materialization, which
//! read the semantic layout, and before `func_to_clif`.

use tribute_ir::dialect::tribute_rtti;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, core};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, AttributeMap, TypeDataBuilder};
use trunk_ir_cranelift_backend::passes::cf_to_clif::rebuild_op_as;

use crate::closure_lower::is_closure_struct_type_ref;

/// Rewrite semantic closure allocations and their RTTI declaration to the
/// native closure layout.
pub fn lower(ctx: &mut IrContext, module: Module) {
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(ClosureStructAdaptPattern)
        .apply_partial(ctx, module);

    let native_ty = native_closure_struct_type(ctx);
    let layouts = module
        .ops(ctx)
        .iter()
        .copied()
        .filter_map(|op| tribute_rtti::Layout::from_op(ctx, op).ok())
        .collect::<Vec<_>>();
    for layout in layouts {
        let ty = layout.r#type(ctx);
        if ty != native_ty && is_closure_struct_type_ref(ctx, ty) {
            layout.set_type(ctx, native_ty);
        }
    }
}

/// The native closure layout: `{ func_ptr: i64, env: ptr }`.
fn native_closure_struct_type(ctx: &mut IrContext) -> TypeRef {
    let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
    let ptr_ty = core::ptr(ctx).as_type_ref();
    let mut attrs = AttributeMap::new();
    attrs.insert(
        Symbol::new(tribute_core::runtime_layout::LAYOUT_ATTR),
        Attribute::Symbol(Symbol::new(tribute_core::runtime_layout::CLOSURE)),
    );
    adt::struct_type(
        ctx,
        Symbol::new("_closure"),
        [
            (Symbol::new("func_ptr"), i64_ty),
            (Symbol::new("env"), ptr_ty),
        ],
        attrs,
    )
    .as_type_ref()
}

/// Pattern: adapt `_closure` struct operations to the native layout.
struct ClosureStructAdaptPattern;

impl RewritePattern for ClosureStructAdaptPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let native_ty = native_closure_struct_type(ctx);

        if let Ok(struct_new) = adt::StructNew::from_op(ctx, op) {
            let ty = struct_new.r#type(ctx);
            if !is_closure_struct_type_ref(ctx, ty) || ty == native_ty {
                return false;
            }
            let new_op = rebuild_op_as(ctx, op, Symbol::new("adt"), Symbol::new("struct_new"));
            ctx.op_mut(new_op)
                .attributes
                .insert(Symbol::new("type"), Attribute::Type(native_ty));
            if !ctx.op_result_types(new_op).is_empty() {
                ctx.set_op_result_type(new_op, 0, native_ty);
            }
            rewriter.replace_op(new_op);
            return true;
        }

        if let Ok(struct_get) = adt::StructGet::from_op(ctx, op) {
            let ty = struct_get.r#type(ctx);
            if !is_closure_struct_type_ref(ctx, ty) || ty == native_ty {
                return false;
            }
            let field_idx = struct_get.field(ctx);
            let new_op = rebuild_op_as(ctx, op, Symbol::new("adt"), Symbol::new("struct_get"));
            ctx.op_mut(new_op)
                .attributes
                .insert(Symbol::new("type"), Attribute::Type(native_ty));
            if field_idx == 0 {
                let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
                ctx.set_op_result_type(new_op, 0, i64_ty);
            } else if field_idx == 1 {
                let ptr_ty = core::ptr(ctx).as_type_ref();
                ctx.set_op_result_type(new_op, 0, ptr_ty);
            }
            rewriter.replace_op(new_op);
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn adapts_closure_allocations_and_their_rtti_declaration() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !_closure = adt.struct<@_closure(@func_ptr: core.i32, @env: tribute_rt.anyref), {layout = @closure}>
  !Other = adt.struct<@Other(@value: tribute_rt.anyref)>
  tribute_rtti.layout {type = !_closure, index = 32, managed = [false, true]}
  tribute_rtti.layout {type = !Other, index = 33, managed = [true]}
  func.func @make(%table: core.i32, %env: tribute_rt.anyref) -> !_closure {
    %closure = adt.struct_new %table, %env {type = !_closure} : !_closure
    %loaded = adt.struct_get %closure {field = 1, type = !_closure} : tribute_rt.anyref
    func.return %closure
  }
}"#,
        );
        let semantic = ctx
            .type_aliases()
            .iter()
            .find_map(|(name, ty)| (*name == "_closure").then_some(*ty))
            .expect("closure alias");

        lower(&mut ctx, module);

        let native = native_closure_struct_type(&mut ctx);
        assert_ne!(native, semantic);
        let layouts = module
            .ops(&ctx)
            .iter()
            .copied()
            .filter_map(|op| tribute_rtti::Layout::from_op(&ctx, op).ok())
            .collect::<Vec<_>>();
        assert_eq!(layouts[0].r#type(&ctx), native);
        assert_eq!(
            layouts[0].managed_fields(&ctx),
            tribute_rtti::ManagedFieldBitmap::Struct(vec![false, true])
        );
        assert_ne!(layouts[1].r#type(&ctx), native);
        let output = print_module(&ctx, module.op());
        assert!(!output.contains("type = !_closure}"), "{output}");
    }

    /// Adapt closures, then lower functions as the native pipeline does.
    fn adapt_then_lower_func(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower(&mut ctx, module);
        trunk_ir_cranelift_backend::passes::func_to_clif::lower(
            &mut ctx,
            module,
            TypeConverter::new(),
        )
        .expect("func_to_clif");
        print_module(&ctx, module.op())
    }

    #[test]
    fn test_closure_struct_adaptation() {
        let result = adapt_then_lower_func(
            r#"core.module @test {
  func.func @test_fn() -> core.i32 {
    %0 = func.constant {func_ref = @lifted_fn} : core.i32
    %1 = mem.null : core.ptr
    %2 = adt.struct_new %0, %1 {type = adt.struct<@_closure(@table_idx: core.i32, @env: core.ptr), {layout = @closure}>} : adt.struct<@_closure(@table_idx: core.i32, @env: core.ptr), {layout = @closure}>
    %3 = adt.struct_get %2 {field = 0, type = adt.struct<@_closure(@table_idx: core.i32, @env: core.ptr), {layout = @closure}>} : core.i32
    %4 = adt.struct_get %2 {field = 1, type = adt.struct<@_closure(@table_idx: core.i32, @env: core.ptr), {layout = @closure}>} : core.ptr
    %5 = func.call_indirect %3, %4 {signature = func.func_sig<(core.ptr) -> core.i32>} : core.i32
    func.return %5
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }

    #[test]
    fn test_closure_struct_anyref_adaptation() {
        let result = adapt_then_lower_func(
            r#"core.module @test {
  func.func @test_fn(%1: wasm.anyref) -> core.i32 {
    %0 = func.constant {func_ref = @lifted_fn} : core.i32
    %2 = adt.struct_new %0, %1 {type = adt.struct<@_closure(@table_idx: core.i32, @env: wasm.anyref), {layout = @closure}>} : adt.struct<@_closure(@table_idx: core.i32, @env: wasm.anyref), {layout = @closure}>
    %3 = adt.struct_get %2 {field = 0, type = adt.struct<@_closure(@table_idx: core.i32, @env: wasm.anyref), {layout = @closure}>} : core.i32
    %4 = adt.struct_get %2 {field = 1, type = adt.struct<@_closure(@table_idx: core.i32, @env: wasm.anyref), {layout = @closure}>} : wasm.anyref
    %5 = func.call_indirect %3, %4 {signature = func.func_sig<(core.ptr) -> core.i32>} : core.i32
    func.return %5
  }
}"#,
        );
        insta::assert_snapshot!(result);
    }
}
