//! Lower mem dialect operations to clif dialect.
//!
//! - `mem.load(ptr, offset)` → `clif.load(ptr, offset)`
//! - `mem.store(ptr, value, offset)` → `clif.store(value, ptr, offset)`
//! - `mem.ptr_add(base, offset)` → `clif.iadd(base, offset)`

use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::mem;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, OpRewritePattern, PatternApplicator,
    PatternRewriter, TypeConverter,
};

/// Lower mem dialect to clif dialect.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    type_converter: TypeConverter,
) -> Result<(), ConversionError> {
    let applicator = PatternApplicator::new(type_converter)
        .with_auto_type_conversion(true)
        .add_pattern(MemLoadPattern)
        .add_pattern(MemStorePattern)
        .add_pattern(MemPtrAddPattern)
        .add_pattern(MemNullPattern)
        .with_target(mem_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "mem-to-clif")?;
    Ok(())
}

fn mem_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect(clif::DIALECT_NAME)
        .illegal_dialect(mem::DIALECT_NAME)
}

struct MemLoadPattern;

impl OpRewritePattern for MemLoadPattern {
    type Op = mem::Load;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        load_op: mem::Load,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = load_op.op_ref();
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let ptr = load_op.ptr(ctx);
        let Ok(offset) = i32::try_from(load_op.offset(ctx)) else {
            return false;
        };
        let new_op = clif::Load::operands(ptr)
            .offset(offset)
            .results(result_ty)
            .build(ctx, loc)
            .op_ref();
        rewriter.replace_op(new_op);
        true
    }
}

struct MemStorePattern;

impl OpRewritePattern for MemStorePattern {
    type Op = mem::Store;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        store_op: mem::Store,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = store_op.op_ref();
        let loc = ctx.op(op).location;
        let ptr = store_op.ptr(ctx);
        let value = store_op.value(ctx);
        let Ok(offset) = i32::try_from(store_op.offset(ctx)) else {
            return false;
        };
        // clif.store operand order: (value, addr)
        let new_op = clif::Store::operands(value, ptr)
            .offset(offset)
            .build(ctx, loc)
            .op_ref();
        rewriter.replace_op(new_op);
        true
    }
}

struct MemPtrAddPattern;

impl OpRewritePattern for MemPtrAddPattern {
    type Op = mem::PtrAdd;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        ptr_add: mem::PtrAdd,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = ptr_add.op_ref();
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;
        // Pointers are pointer-width integers in Cranelift.
        let new_op = clif::Iadd::operands(ptr_add.base(ctx), ptr_add.offset(ctx))
            .results(result_ty)
            .build(ctx, loc)
            .op_ref();
        rewriter.replace_op(new_op);
        true
    }
}

struct MemNullPattern;

impl OpRewritePattern for MemNullPattern {
    type Op = mem::Null;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        null: mem::Null,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let op = null.op_ref();
        let Some(result_ty) = rewriter.result_type(ctx, op, 0) else {
            return false;
        };
        let loc = ctx.op(op).location;
        // Pointers are pointer-width integers in Cranelift; null is zero.
        let new_op = clif::Iconst::operands()
            .value(0)
            .results(result_ty)
            .build(ctx, loc)
            .op_ref();
        rewriter.replace_op(new_op);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    #[test]
    fn ptr_add_lowers_to_integer_add_and_loads_keep_immediate_offsets() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @byte(%base: core.ptr, %index: core.index) -> core.i8 {
    %data = mem.load %base {offset = 0} : core.ptr
    %addr = mem.ptr_add %data, %index : core.ptr
    %byte = mem.load %addr {offset = 0} : core.i8
    func.return %byte
  }
}"#,
        );
        let i64_ty = trunk_ir::dialect::core::I64::type_ref(&mut ctx);
        let mut type_converter = TypeConverter::new();
        type_converter.add_conversion(move |ctx, ty| {
            trunk_ir::dialect::core::Index::matches(ctx, ty).then_some(i64_ty)
        });

        lower(&mut ctx, module, type_converter).expect("mem ops lower to clif");

        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("mem."), "{printed}");
        assert_eq!(printed.matches("clif.load").count(), 2, "{printed}");
        assert!(printed.contains("clif.iadd"), "{printed}");
        assert!(printed.contains(": core.i64"), "{printed}");
    }

    #[test]
    fn null_lowers_to_a_zero_pointer_constant() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @null() -> core.ptr {
    %null = mem.null : core.ptr
    func.return %null
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new()).expect("mem.null lowers to clif");

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("clif.iconst {value = 0} : core.ptr"),
            "{printed}"
        );
    }
}
