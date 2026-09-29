//! Lower mem dialect operations to clif dialect.
//!
//! - `mem.load(ptr, offset)` → `clif.load(ptr, offset)`
//! - `mem.store(ptr, value, offset)` → `clif.store(value, ptr, offset)`
//! - `mem.ptr_add(base, offset)` → `clif.iadd(base, offset)`

use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::dialect::mem;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
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
        .with_target(mem_to_clif_target());
    applicator.apply_partial_conversion(ctx, module, "mem-to-clif")?;
    Ok(())
}

fn mem_to_clif_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_dialect("clif")
        .illegal_dialect("mem")
}

struct MemLoadPattern;

impl RewritePattern for MemLoadPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(load_op) = mem::Load::from_op(ctx, op) else {
            return false;
        };
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

impl RewritePattern for MemStorePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(store_op) = mem::Store::from_op(ctx, op) else {
            return false;
        };
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

impl RewritePattern for MemPtrAddPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(ptr_add) = mem::PtrAdd::from_op(ctx, op) else {
            return false;
        };
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
  func.func @byte(%base: core.ptr, %index: core.i64) -> core.i8 {
    %data = mem.load %base {offset = 0} : core.ptr
    %addr = mem.ptr_add %data, %index : core.ptr
    %byte = mem.load %addr {offset = 0} : core.i8
    func.return %byte
  }
}"#,
        );

        lower(&mut ctx, module, TypeConverter::new()).expect("mem ops lower to clif");

        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("mem."), "{printed}");
        assert_eq!(printed.matches("clif.load").count(), 2, "{printed}");
        assert!(printed.contains("clif.iadd"), "{printed}");
    }
}
