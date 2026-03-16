//! Wrap function returns in effectful functions with YieldResult::Done.

use std::collections::HashSet;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt as arena_adt;
use trunk_ir::dialect::core as arena_core;
use trunk_ir::dialect::func as arena_func;
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
        if let Ok(func) = arena_func::Func::from_op(ctx, parent_op) {
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
        if arena_func::Return::from_op(ctx, op).is_err() {
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
        let anyref_val =
            arena_core::unrealized_conversion_cast(ctx, location, value, self.types.anyref);
        rewriter.insert_op(anyref_val.op_ref());

        // Create YieldResult::Done
        let done_op = arena_adt::variant_new(
            ctx,
            location,
            [anyref_val.result(ctx)],
            self.types.yield_result,
            self.types.yield_result,
            Symbol::new("Done"),
        );
        rewriter.insert_op(done_op.op_ref());

        // Replace return with new return using Done value
        let new_return = arena_func::r#return(ctx, location, [done_op.result(ctx)]);
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
