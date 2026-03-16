//! Wrap function returns in effectful functions with YieldResult::Done.

use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

use super::types::{YieldBubblingTypes, is_yield_result_type};

// ============================================================================
// Pattern: Wrap returns in effectful functions with YieldResult::Done
// ============================================================================

pub(crate) struct WrapReturnsPattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
    pub(crate) types: YieldBubblingTypes,
}

/// Walk up the parent chain from an op to find the enclosing `func.func` name.
fn find_parent_func_name(ctx: &IrContext, op: OpRef) -> Option<Symbol> {
    let mut current_block = ctx.op(op).parent_block?;
    loop {
        let region = ctx.block(current_block).parent_region?;
        let parent_op = ctx.region(region).parent_op?;
        if let Ok(func) = func::Func::from_op(ctx, parent_op) {
            return Some(func.sym_name(ctx));
        }
        current_block = ctx.op(parent_op).parent_block?;
    }
}

impl RewritePattern for WrapReturnsPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        // Match func.return
        if func::Return::from_op(ctx, op).is_err() {
            return false;
        }

        // Find enclosing function and check if it's effectful
        let Some(func_name) = find_parent_func_name(ctx, op) else {
            return false;
        };
        if !self.effectful_funcs.contains(&func_name) {
            return false;
        }

        // Check if return value is already YieldResult
        let Some(&value) = ctx.op_operands(op).first() else {
            return false;
        };
        if is_yield_result_type(ctx, ctx.value_ty(value)) {
            return false;
        }

        let location = ctx.op(op).location;

        // Cast to anyref
        let anyref_val = core::unrealized_conversion_cast(ctx, location, value, self.types.anyref);
        rewriter.insert_op(anyref_val.op_ref());

        // Create YieldResult::Done
        let done_op = adt::variant_new(
            ctx,
            location,
            [anyref_val.result(ctx)],
            self.types.yield_result,
            self.types.yield_result,
            Symbol::new("Done"),
        );
        rewriter.insert_op(done_op.op_ref());

        // Replace return with new return using Done value
        let new_return = func::r#return(ctx, location, [done_op.result(ctx)]);
        rewriter.replace_op(new_return.op_ref());

        true
    }
}

/// Wrap returns only in the specified target functions.
///
/// Used for post-processing newly generated resume/chain functions.
pub(crate) fn wrap_returns_for_funcs(
    ctx: &mut IrContext,
    module: Module,
    func_names: &[Symbol],
    types: &YieldBubblingTypes,
) {
    let target_funcs: HashSet<Symbol> = func_names.iter().copied().collect();
    let applicator = PatternApplicator::new(TypeConverter::new()).add_pattern(WrapReturnsPattern {
        effectful_funcs: Rc::new(target_funcs),
        types: *types,
    });
    applicator.apply_partial(ctx, module);
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::{BlockData, OperationDataBuilder, RegionData};
    use trunk_ir::dialect::arith;
    use trunk_ir::location::Span;
    use trunk_ir::smallvec::smallvec;
    use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn i32_type(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    /// Create a func.func with a body containing the given ops.
    fn make_func_with_body(
        ctx: &mut IrContext,
        loc: Location,
        name: &'static str,
        ret_ty: trunk_ir::refs::TypeRef,
        body_ops: Vec<trunk_ir::refs::OpRef>,
    ) -> trunk_ir::refs::OpRef {
        let func_ty = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(ret_ty)
                .build(),
        );
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: trunk_ir::smallvec::smallvec![],
            parent_region: None,
        });
        for op in body_ops {
            ctx.push_op(block, op);
        }
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let func_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::Symbol(Symbol::new(name)))
            .attr("type", Attribute::Type(func_ty))
            .region(body)
            .build(ctx);
        ctx.create_op(func_data)
    }

    fn make_module(ctx: &mut IrContext, loc: Location, ops: Vec<trunk_ir::refs::OpRef>) -> Module {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .region(body)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        Module::new(ctx, module_op).expect("should be a valid module")
    }

    #[test]
    fn find_parent_func_name_returns_enclosing_func() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        let c_val = c.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [c_val]);
        let func_op = make_func_with_body(
            &mut ctx,
            loc,
            "my_func",
            i32_ty,
            vec![c.op_ref(), ret.op_ref()],
        );

        // Place func in a module so parent chain is complete
        let _module = make_module(&mut ctx, loc, vec![func_op]);

        // The return op should find "my_func" as its parent
        let result = find_parent_func_name(&ctx, ret.op_ref());
        assert_eq!(result, Some(Symbol::new("my_func")));
    }

    #[test]
    fn find_parent_func_name_returns_none_for_top_level() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Op directly in module, no enclosing func
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        let _module = make_module(&mut ctx, loc, vec![c.op_ref()]);

        let result = find_parent_func_name(&ctx, c.op_ref());
        assert_eq!(result, None);
    }

    #[test]
    fn wrap_returns_for_funcs_wraps_specified_funcs() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);
        let types = super::super::types::YieldBubblingTypes::new(&mut ctx);

        // Create effectful func with a plain return
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        let c_val = c.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [c_val]);
        let effectful_func = make_func_with_body(
            &mut ctx,
            loc,
            "effectful",
            i32_ty,
            vec![c.op_ref(), ret.op_ref()],
        );

        // Create non-effectful func with a plain return
        let c2 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(99));
        let c2_val = c2.result(&ctx);
        let ret2 = func::r#return(&mut ctx, loc, [c2_val]);
        let pure_func = make_func_with_body(
            &mut ctx,
            loc,
            "pure",
            i32_ty,
            vec![c2.op_ref(), ret2.op_ref()],
        );

        let module = make_module(&mut ctx, loc, vec![effectful_func, pure_func]);

        // Wrap only "effectful"
        wrap_returns_for_funcs(&mut ctx, module, &[Symbol::new("effectful")], &types);

        // Check that "effectful" func's return now returns YieldResult
        let module_ops = module.ops(&ctx);
        let ef = func::Func::from_op(&ctx, module_ops[0]).unwrap();
        let ef_body = ef.body(&ctx);
        let ef_block = ctx.region(ef_body).blocks[0];
        let ef_ops = ctx.block(ef_block).ops.to_vec();
        let last_op = *ef_ops.last().unwrap();
        assert!(func::Return::from_op(&ctx, last_op).is_ok());
        let ret_operands = ctx.op_operands(last_op);
        let ret_val_ty = ctx.value_ty(ret_operands[0]);
        assert!(super::super::types::is_yield_result_type(&ctx, ret_val_ty));

        // Check that "pure" func's return is unchanged (i32, not YieldResult)
        let pf = func::Func::from_op(&ctx, module_ops[1]).unwrap();
        let pf_body = pf.body(&ctx);
        let pf_block = ctx.region(pf_body).blocks[0];
        let pf_ops = ctx.block(pf_block).ops.to_vec();
        let pf_last = *pf_ops.last().unwrap();
        assert!(func::Return::from_op(&ctx, pf_last).is_ok());
        let pf_ret_operands = ctx.op_operands(pf_last);
        let pf_ret_ty = ctx.value_ty(pf_ret_operands[0]);
        assert!(!super::super::types::is_yield_result_type(&ctx, pf_ret_ty));
    }
}
