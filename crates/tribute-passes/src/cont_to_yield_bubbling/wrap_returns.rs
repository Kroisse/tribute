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
    use trunk_ir::context::IrContext;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::walk;

    /// Find the func.return op inside a func.func by walking its body.
    fn find_return_op(ctx: &IrContext, func_op: trunk_ir::refs::OpRef) -> trunk_ir::refs::OpRef {
        let f = func::Func::from_op(ctx, func_op).unwrap();
        let body = f.body(ctx);
        let mut ret_op = None;
        let _: std::ops::ControlFlow<(), ()> = walk::walk_region(ctx, body, &mut |op| {
            if func::Return::from_op(ctx, op).is_ok() {
                ret_op = Some(op);
                return std::ops::ControlFlow::Break(());
            }
            std::ops::ControlFlow::Continue(walk::WalkAction::Advance)
        });
        ret_op.expect("func should contain a func.return")
    }

    #[test]
    fn find_parent_func_name_returns_enclosing_func() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @my_func() -> core.i32 {
    %c = arith.const {value = 42} : core.i32
    func.return %c
  }
}"#,
        );
        let func_op = module.ops(&ctx)[0];
        let ret_op = find_return_op(&ctx, func_op);

        let result = find_parent_func_name(&ctx, ret_op);
        assert_eq!(result, Some(Symbol::new("my_func")));
    }

    #[test]
    fn find_parent_func_name_returns_none_for_top_level() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  %c = arith.const {value = 42} : core.i32
}"#,
        );
        let top_level_op = module.ops(&ctx)[0];

        let result = find_parent_func_name(&ctx, top_level_op);
        assert_eq!(result, None);
    }

    #[test]
    fn wrap_returns_for_funcs_wraps_specified_funcs() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @effectful() -> core.i32 {
    %c = arith.const {value = 42} : core.i32
    func.return %c
  }
  func.func @pure() -> core.i32 {
    %c = arith.const {value = 99} : core.i32
    func.return %c
  }
}"#,
        );
        let types = super::super::types::YieldBubblingTypes::new(&mut ctx);

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
        let ret_val_ty = ctx.value_ty(ctx.op_operands(last_op)[0]);
        assert!(super::super::types::is_yield_result_type(&ctx, ret_val_ty));

        // Check that "pure" func's return is unchanged (i32, not YieldResult)
        let pf = func::Func::from_op(&ctx, module_ops[1]).unwrap();
        let pf_body = pf.body(&ctx);
        let pf_block = ctx.region(pf_body).blocks[0];
        let pf_ops = ctx.block(pf_block).ops.to_vec();
        let pf_last = *pf_ops.last().unwrap();
        assert!(func::Return::from_op(&ctx, pf_last).is_ok());
        let pf_ret_ty = ctx.value_ty(ctx.op_operands(pf_last)[0]);
        assert!(!super::super::types::is_yield_result_type(&ctx, pf_ret_ty));
    }
}
