//! Simple lowering patterns for cont.shift, cont.resume, and cont.drop.
//!
//! These patterns transform continuation primitives into FFI calls to the
//! libmprompt-based runtime:
//! - `cont.shift` → `func.call @__tribute_yield`
//! - `cont.resume` → `func.call @__tribute_resume`
//! - `cont.drop` → `func.call @__tribute_resume_drop`

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::{
    arith as arena_arith, cont as arena_cont, core as arena_core, func as arena_func,
};
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::OpRef;
use trunk_ir::arena::rewrite::{ArenaRewritePattern, PatternRewriter as ArenaPatternRewriter};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, TypeDataBuilder};
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{arith, cont};
use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol};

use crate::cont_util::compute_op_idx;

// ============================================================================
// Pattern: Lower cont.shift → func.call @__tribute_yield
// ============================================================================

pub(crate) struct LowerShiftPattern;

impl<'db> RewritePattern<'db> for LowerShiftPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(shift_op) = cont::Shift::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ptr_ty = core::Ptr::new(db).as_type();

        let operands = rewriter.operands();

        // First operand is the tag (prompt tag value)
        let tag = operands
            .first()
            .copied()
            .expect("cont.shift requires tag operand");

        // Second operand (if any) is the shift value
        let shift_value = operands.get(1).copied();

        let mut ops = Vec::new();

        // Compute op_idx from ability_ref and op_name
        let ability_ref_ty = shift_op.ability_ref(db);
        let ability_ref =
            core::AbilityRefType::from_type(db, ability_ref_ty).and_then(|ar| ar.name(db));
        let op_name = Some(shift_op.op_name(db));
        let op_idx = compute_op_idx(ability_ref, op_name);

        // %op_idx = arith.const <op_idx>
        let op_idx_const = arith::r#const(db, location, i32_ty, Attribute::IntBits(op_idx as u64));
        ops.push(op_idx_const.as_operation());

        // %shift_val = shift_value or null ptr
        let shift_val = if let Some(v) = shift_value {
            // Cast to ptr if needed
            if rewriter.get_value_type(db, v) != Some(ptr_ty) {
                let cast = core::unrealized_conversion_cast(db, location, v, ptr_ty);
                ops.push(cast.as_operation());
                cast.as_operation().result(db, 0)
            } else {
                v
            }
        } else {
            // No value — pass null ptr
            let null = arith::r#const(db, location, ptr_ty, Attribute::IntBits(0));
            ops.push(null.as_operation());
            null.as_operation().result(db, 0)
        };

        // %result = func.call @__tribute_yield(%tag, %op_idx, %shift_val)
        let call = func::call(
            db,
            location,
            vec![tag, op_idx_const.result(db), shift_val],
            ptr_ty,
            Symbol::new("__tribute_yield"),
        );
        ops.push(call.as_operation());

        // Cast ptr result back to the original result type if needed
        let original_result_ty = op.results(db).first().copied();
        if let Some(result_ty) = original_result_ty
            && result_ty != ptr_ty
        {
            let cast = core::unrealized_conversion_cast(db, location, call.result(db), result_ty);
            ops.push(cast.as_operation());
        }

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Pattern: Lower cont.resume → func.call @__tribute_resume
// ============================================================================

pub(crate) struct LowerResumePattern;

impl<'db> RewritePattern<'db> for LowerResumePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_) = cont::Resume::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();

        let operands = rewriter.operands();

        // First operand: continuation
        let continuation = operands
            .first()
            .copied()
            .expect("cont.resume requires continuation");

        // Second operand: value to send
        let value = operands.get(1).copied();

        let mut ops = Vec::new();

        // Cast continuation to ptr if needed
        let cont_ptr = if rewriter.get_value_type(db, continuation) != Some(ptr_ty) {
            let cast = core::unrealized_conversion_cast(db, location, continuation, ptr_ty);
            ops.push(cast.as_operation());
            cast.as_operation().result(db, 0)
        } else {
            continuation
        };

        // Cast value to ptr if needed
        let val_ptr = if let Some(v) = value {
            if rewriter.get_value_type(db, v) != Some(ptr_ty) {
                let cast = core::unrealized_conversion_cast(db, location, v, ptr_ty);
                ops.push(cast.as_operation());
                cast.as_operation().result(db, 0)
            } else {
                v
            }
        } else {
            // No value — pass null
            let null = arith::r#const(db, location, ptr_ty, Attribute::IntBits(0));
            ops.push(null.as_operation());
            null.as_operation().result(db, 0)
        };

        // %result = func.call @__tribute_resume(%cont, %val)
        let call = func::call(
            db,
            location,
            vec![cont_ptr, val_ptr],
            ptr_ty,
            Symbol::new("__tribute_resume"),
        );
        ops.push(call.as_operation());

        // Cast result back if needed
        let original_result_ty = op.results(db).first().copied();
        if let Some(result_ty) = original_result_ty
            && result_ty != ptr_ty
        {
            let cast = core::unrealized_conversion_cast(db, location, call.result(db), result_ty);
            ops.push(cast.as_operation());
        }

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Pattern: Lower cont.drop → func.call @__tribute_resume_drop
// ============================================================================

pub(crate) struct LowerDropPattern;

impl<'db> RewritePattern<'db> for LowerDropPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool {
        let Ok(_) = cont::Drop::from_operation(db, *op) else {
            return false;
        };

        let location = op.location(db);
        let ptr_ty = core::Ptr::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();

        let operands = rewriter.operands();

        // First operand: continuation
        let continuation = operands
            .first()
            .copied()
            .expect("cont.drop requires continuation");

        let mut ops = Vec::new();

        // Cast continuation to ptr if needed
        let cont_ptr = if rewriter.get_value_type(db, continuation) != Some(ptr_ty) {
            let cast = core::unrealized_conversion_cast(db, location, continuation, ptr_ty);
            ops.push(cast.as_operation());
            cast.as_operation().result(db, 0)
        } else {
            continuation
        };

        // func.call @__tribute_resume_drop(%cont)
        let call = func::call(
            db,
            location,
            vec![cont_ptr],
            nil_ty,
            Symbol::new("__tribute_resume_drop"),
        );
        ops.push(call.as_operation());

        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// ============================================================================
// Arena patterns
// ============================================================================

fn arena_ptr_ty(ctx: &mut IrContext) -> trunk_ir::arena::refs::TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build())
}

fn arena_i32_ty(ctx: &mut IrContext) -> trunk_ir::arena::refs::TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

fn arena_nil_ty(ctx: &mut IrContext) -> trunk_ir::arena::refs::TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
}

/// Arena pattern: Lower `cont.shift` → `func.call @__tribute_yield`
pub(crate) struct ArenaLowerShiftPattern;

impl ArenaRewritePattern for ArenaLowerShiftPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        let Ok(shift_op) = arena_cont::Shift::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let i32_ty = arena_i32_ty(ctx);
        let ptr_ty = arena_ptr_ty(ctx);

        let operands: Vec<_> = ctx.op_operands(op).to_vec();

        // First operand is the tag
        let tag = operands[0];

        // Second operand (if any) is the shift value
        let shift_value = operands.get(1).copied();

        // Compute op_idx from ability_ref and op_name
        let ability_ref_ty = shift_op.ability_ref(ctx);
        let ability_data = ctx.types.get(ability_ref_ty);
        let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
            Some(ArenaAttribute::Symbol(s)) => Some(*s),
            _ => None,
        };
        let op_name = Some(shift_op.op_name(ctx));
        let op_idx = compute_op_idx(ability_name, op_name);

        // %op_idx = arith.const <op_idx>
        let op_idx_const =
            arena_arith::r#const(ctx, loc, i32_ty, ArenaAttribute::IntBits(op_idx as u64));
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
            let null = arena_arith::r#const(ctx, loc, ptr_ty, ArenaAttribute::IntBits(0));
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
        "ArenaLowerShiftPattern"
    }
}

/// Arena pattern: Lower `cont.resume` → `func.call @__tribute_resume`
pub(crate) struct ArenaLowerResumePattern;

impl ArenaRewritePattern for ArenaLowerResumePattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if !arena_cont::Resume::matches(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;
        let ptr_ty = arena_ptr_ty(ctx);

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
            let null = arena_arith::r#const(ctx, loc, ptr_ty, ArenaAttribute::IntBits(0));
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
        "ArenaLowerResumePattern"
    }
}

/// Arena pattern: Lower `cont.drop` → `func.call @__tribute_resume_drop`
pub(crate) struct ArenaLowerDropPattern;

impl ArenaRewritePattern for ArenaLowerDropPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut ArenaPatternRewriter<'_>,
    ) -> bool {
        if !arena_cont::Drop::matches(ctx, op) {
            return false;
        }

        let loc = ctx.op(op).location;
        let ptr_ty = arena_ptr_ty(ctx);
        let nil_ty = arena_nil_ty(ctx);

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
        rewriter.replace_op(call.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "ArenaLowerDropPattern"
    }
}
