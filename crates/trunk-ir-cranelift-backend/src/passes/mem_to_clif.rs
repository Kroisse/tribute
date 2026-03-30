//! Lower mem dialect operations to clif dialect.
//!
//! - `mem.load(ptr, offset)` → `clif.load(ptr, offset)`
//! - `mem.store(ptr, value, offset)` → `clif.store(value, ptr, offset)`

use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif as arena_clif;
use trunk_ir::dialect::mem;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

/// Lower mem dialect to clif dialect.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
    use trunk_ir::rewrite::ConversionTarget;

    let mut target = ConversionTarget::new();
    target.add_legal_dialect("clif");
    target.add_illegal_dialect("mem");

    let applicator = PatternApplicator::new(type_converter)
        .with_target(target)
        .add_pattern(MemLoadPattern)
        .add_pattern(MemStorePattern);
    applicator.apply_partial(ctx, module);
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
        let offset = load_op.offset(ctx) as i32;
        let new_op = arena_clif::load(ctx, loc, ptr, result_ty, offset).op_ref();
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
        let offset = store_op.offset(ctx) as i32;
        // clif.store operand order: (value, addr)
        let new_op = arena_clif::store(ctx, loc, value, ptr, offset).op_ref();
        rewriter.replace_op(new_op);
        true
    }
}
