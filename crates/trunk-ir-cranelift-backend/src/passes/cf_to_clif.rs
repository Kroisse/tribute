//! Lower cf dialect operations to clif dialect.
//!
//! This pass converts CFG-based control flow operations to Cranelift equivalents:
//! - `cf.br` -> `clif.jump`
//! - `cf.cond_br` -> `clif.brif`

use trunk_ir::OperationDataBuilder;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::cf as arena_cf;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

/// Lower cf dialect to clif dialect.
pub fn lower(ctx: &mut IrContext, module: Module, type_converter: TypeConverter) {
    use trunk_ir::rewrite::ConversionTarget;

    let mut target = ConversionTarget::new();
    target.add_legal_dialect("clif");
    target.add_illegal_dialect("cf");

    let applicator = PatternApplicator::new(type_converter)
        .with_target(target)
        .add_pattern(CfBrPattern)
        .add_pattern(CfCondBrPattern);
    applicator.apply_partial(ctx, module);
}

/// Pattern: `cf.br` -> `clif.jump`
struct CfBrPattern;

impl RewritePattern for CfBrPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_cf::Br::from_op(ctx, op).is_err() {
            return false;
        }

        let new_op = rebuild_op_as(ctx, op, Symbol::new("clif"), Symbol::new("jump"));
        rewriter.replace_op(new_op);
        true
    }
}

/// Pattern: `cf.cond_br` -> `clif.brif`
struct CfCondBrPattern;

impl RewritePattern for CfCondBrPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if arena_cf::CondBr::from_op(ctx, op).is_err() {
            return false;
        }

        let new_op = rebuild_op_as(ctx, op, Symbol::new("clif"), Symbol::new("brif"));
        rewriter.replace_op(new_op);
        true
    }
}

/// Rebuild an operation with a new dialect/name, transferring all operands,
/// results, attributes, regions, and successors from the original.
///
/// Regions are detached from the original operation and re-attached to the new one.
pub fn rebuild_op_as(ctx: &mut IrContext, op: OpRef, dialect: Symbol, name: Symbol) -> OpRef {
    let data = ctx.op(op);
    let loc = data.location;
    let attrs = data.attributes.clone();
    let regions = data.regions.to_vec();
    let successors = data.successors.to_vec();
    let operands: Vec<_> = ctx.op_operands(op).to_vec();
    let result_types: Vec<_> = ctx.op_result_types(op).to_vec();

    // Detach regions from the old operation so they can be owned by the new one
    for &r in &regions {
        ctx.detach_region(r);
    }

    let mut builder = OperationDataBuilder::new(loc, dialect, name)
        .operands(operands)
        .results(result_types);
    for (k, v) in attrs {
        builder = builder.attr(k, v);
    }
    for r in regions {
        builder = builder.region(r);
    }
    for s in successors {
        builder = builder.successor(s);
    }
    let data = builder.build(ctx);
    ctx.create_op(data)
}
