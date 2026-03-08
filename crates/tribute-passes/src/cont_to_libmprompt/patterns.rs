//! Simple lowering patterns for cont.shift, cont.resume, and cont.drop.
//!
//! These patterns transform continuation primitives into FFI calls to the
//! libmprompt-based runtime:
//! - `cont.shift` -> `func.call @__tribute_yield`
//! - `cont.resume` -> `func.call @__tribute_resume`
//! - `cont.drop` -> `func.call @__tribute_resume_drop`

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, cont as arena_cont, core as arena_core, func as arena_func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::types::{Attribute, TypeDataBuilder};

use crate::cont_util::compute_op_idx;

fn i32_ty(ctx: &mut IrContext) -> trunk_ir::refs::TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

/// Pattern: Lower `cont.shift` -> `func.call @__tribute_yield`
pub(crate) struct LowerShiftPattern;

impl RewritePattern for LowerShiftPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let i32_ty = i32_ty(ctx);
        let ptr_ty = arena_core::ptr(ctx).as_type_ref();

        let operands: Vec<_> = ctx.op_operands(op).to_vec();

        // First operand is the tag
        let tag = operands[0];

        // Second operand (if any) is the shift value
        let shift_value = operands.get(1).copied();

        // Compute op_idx from ability_ref and op_name
        let ability_ref_ty = shift_op.ability_ref(ctx);
        let ability_data = ctx.types.get(ability_ref_ty);
        let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
            Some(Attribute::Symbol(s)) => Some(*s),
            _ => panic!(
                "LowerShiftPattern: cont.shift has invalid ability_ref type (missing or non-Symbol 'name' attribute): {:?}",
                ability_data,
            ),
        };
        let op_name = Some(shift_op.op_name(ctx));
        let op_idx = compute_op_idx(ability_name, op_name);

        // %op_idx = arith.const <op_idx>
        let op_idx_const = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(op_idx as u64));
        rewriter.insert_op(op_idx_const.op_ref());

        // %shift_val = shift_value or null ptr
        let shift_val = if let Some(v) = shift_value {
            if ctx.value_ty(v) != ptr_ty {
                let cast = arena_core::unrealized_conversion_cast(ctx, loc, v, ptr_ty);
                rewriter.insert_op(cast.op_ref());
                cast.result(ctx)
            } else {
                v
            }
        } else {
            let null = arith::r#const(ctx, loc, ptr_ty, Attribute::IntBits(0));
            rewriter.insert_op(null.op_ref());
            null.result(ctx)
        };

        // %result = func.call @__tribute_yield(%tag, %op_idx, %shift_val)
        let call = arena_func::call(
            ctx,
            loc,
            [tag, op_idx_const.result(ctx), shift_val],
            ptr_ty,
            Symbol::new("__tribute_yield"),
        );

        // Cast ptr result back to original type if needed
        let result_types: Vec<_> = ctx.op_result_types(op).to_vec();
        let original_result_ty = result_types.first().copied();
        if let Some(result_ty) = original_result_ty
            && result_ty != ptr_ty
        {
            rewriter.insert_op(call.op_ref());
            let cast =
                arena_core::unrealized_conversion_cast(ctx, loc, call.result(ctx), result_ty);
            rewriter.replace_op(cast.op_ref());
        } else {
            rewriter.replace_op(call.op_ref());
        }
        true
    }

    fn name(&self) -> &'static str {
        "LowerShiftPattern"
    }
}

/// Pattern: Lower `cont.resume` -> `func.call @__tribute_resume`
pub(crate) struct LowerResumePattern;

impl RewritePattern for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_cont::Resume::matches(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;
        let ptr_ty = arena_core::ptr(ctx).as_type_ref();

        let operands: Vec<_> = ctx.op_operands(op).to_vec();
        let continuation = operands[0];
        let value = operands.get(1).copied();

        // Cast continuation to ptr if needed
        let cont_ptr = if ctx.value_ty(continuation) != ptr_ty {
            let cast = arena_core::unrealized_conversion_cast(ctx, loc, continuation, ptr_ty);
            rewriter.insert_op(cast.op_ref());
            cast.result(ctx)
        } else {
            continuation
        };

        // Cast value to ptr if needed
        let val_ptr = if let Some(v) = value {
            if ctx.value_ty(v) != ptr_ty {
                let cast = arena_core::unrealized_conversion_cast(ctx, loc, v, ptr_ty);
                rewriter.insert_op(cast.op_ref());
                cast.result(ctx)
            } else {
                v
            }
        } else {
            let null = arith::r#const(ctx, loc, ptr_ty, Attribute::IntBits(0));
            rewriter.insert_op(null.op_ref());
            null.result(ctx)
        };

        let call = arena_func::call(
            ctx,
            loc,
            [cont_ptr, val_ptr],
            ptr_ty,
            Symbol::new("__tribute_resume"),
        );

        // Cast result back if needed
        let result_types: Vec<_> = ctx.op_result_types(op).to_vec();
        let original_result_ty = result_types.first().copied();
        if let Some(result_ty) = original_result_ty
            && result_ty != ptr_ty
        {
            rewriter.insert_op(call.op_ref());
            let cast =
                arena_core::unrealized_conversion_cast(ctx, loc, call.result(ctx), result_ty);
            rewriter.replace_op(cast.op_ref());
        } else {
            rewriter.replace_op(call.op_ref());
        }
        true
    }

    fn name(&self) -> &'static str {
        "LowerResumePattern"
    }
}

/// Pattern: Lower `cont.drop` -> `func.call @__tribute_resume_drop`
pub(crate) struct LowerDropPattern;

impl RewritePattern for LowerDropPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !arena_cont::Drop::matches(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;
        let ptr_ty = arena_core::ptr(ctx).as_type_ref();
        let nil_ty = arena_core::nil(ctx).as_type_ref();

        let operands: Vec<_> = ctx.op_operands(op).to_vec();
        let continuation = operands[0];

        // Cast continuation to ptr if needed
        let cont_ptr = if ctx.value_ty(continuation) != ptr_ty {
            let cast = arena_core::unrealized_conversion_cast(ctx, loc, continuation, ptr_ty);
            rewriter.insert_op(cast.op_ref());
            cast.result(ctx)
        } else {
            continuation
        };

        let call = arena_func::call(
            ctx,
            loc,
            [cont_ptr],
            nil_ty,
            Symbol::new("__tribute_resume_drop"),
        );
        // cont.drop has 0 results, so use insert_op + erase_op instead of replace_op
        rewriter.insert_op(call.op_ref());
        rewriter.erase_op(vec![]);
        true
    }

    fn name(&self) -> &'static str {
        "LowerDropPattern"
    }
}
