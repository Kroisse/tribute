//! Simple lowering patterns for cont.shift, cont.resume, and cont.drop.
//!
//! These patterns transform continuation primitives into FFI calls to the
//! libmprompt-based runtime:
//! - `cont.shift` → `func.call @__tribute_yield`
//! - `cont.resume` → `func.call @__tribute_resume`
//! - `cont.drop` → `func.call @__tribute_resume_drop`

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
