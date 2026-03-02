//! Lower tribute_rt dialect operations to clif dialect for native backend.
//!
//! This pass converts boxing/unboxing operations to their native equivalents:
//! - `tribute_rt.box_int` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_int` → `clif.load`
//! - `tribute_rt.box_nat` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_nat` → `clif.load`
//! - `tribute_rt.box_float` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_float` → `clif.load`
//! - `tribute_rt.box_bool` → `clif.call @__tribute_alloc` + `clif.store`
//! - `tribute_rt.unbox_bool` → `clif.load`
//!
//! ## Allocation Strategy
//!
//! Each boxed value is heap-allocated via `__tribute_alloc(size)` and the
//! primitive value is stored at offset 0. Phase 3 (RC) will prepend an 8-byte
//! header (refcount + rtti_idx) before the payload.

use tribute_ir::arena::dialect::tribute_rt;
use tribute_ir::dialect::tribute_rt::RC_HEADER_SIZE;
use trunk_ir::Symbol;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::clif;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::rewriter::PatternRewriter;
use trunk_ir::arena::rewrite::type_converter::ArenaTypeConverter;
use trunk_ir::arena::rewrite::{
    ArenaConversionTarget, ArenaModule, ArenaRewritePattern, PatternApplicator,
};
use trunk_ir::arena::types::{Location, TypeDataBuilder};

/// Name of the runtime allocation function.
const ALLOC_FN: &str = "__tribute_alloc";

/// Generate boxing operations: allocate + store RC header + store value.
///
/// Returns a list of OpRefs where the last op produces the payload pointer result.
#[allow(clippy::too_many_arguments)]
fn box_value(
    ctx: &mut IrContext,
    loc: Location,
    value: ValueRef,
    payload_size: u64,
    rtti_idx: u32,
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
) -> Vec<OpRef> {
    let mut ops = Vec::new();

    // 1. Allocation size (payload + RC header)
    let alloc_size = payload_size
        .checked_add(RC_HEADER_SIZE)
        .expect("allocation size overflow: payload_size + RC_HEADER_SIZE exceeds u64::MAX");
    let alloc_size_i64 = i64::try_from(alloc_size).expect("allocation size does not fit in i64");
    let size_op = clif::iconst(ctx, loc, i64_ty, alloc_size_i64);
    let size_val = size_op.result(ctx);
    ops.push(size_op.op_ref());

    // 2. Allocate heap memory
    let call_op = clif::call(ctx, loc, [size_val], ptr_ty, Symbol::new(ALLOC_FN));
    let raw_ptr = call_op.result(ctx);
    ops.push(call_op.op_ref());

    // 3. Store refcount = 1 at raw_ptr + 0
    let rc_one = clif::iconst(ctx, loc, i32_ty, 1);
    let rc_one_val = rc_one.result(ctx);
    ops.push(rc_one.op_ref());
    let store_rc = clif::store(ctx, loc, rc_one_val, raw_ptr, 0);
    ops.push(store_rc.op_ref());

    // 4. Store rtti_idx at raw_ptr + 4
    let rtti_val = clif::iconst(ctx, loc, i32_ty, rtti_idx as i64);
    let rtti_val_v = rtti_val.result(ctx);
    ops.push(rtti_val.op_ref());
    let store_rtti = clif::store(ctx, loc, rtti_val_v, raw_ptr, 4);
    ops.push(store_rtti.op_ref());

    // 5. Compute payload pointer = raw_ptr + 8
    let hdr_size = clif::iconst(ctx, loc, i64_ty, RC_HEADER_SIZE as i64);
    let hdr_size_val = hdr_size.result(ctx);
    ops.push(hdr_size.op_ref());
    let payload_ptr = clif::iadd(ctx, loc, raw_ptr, hdr_size_val, ptr_ty);
    let payload_ptr_val = payload_ptr.result(ctx);
    ops.push(payload_ptr.op_ref());

    // 6. Store value at payload offset 0
    let store_val = clif::store(ctx, loc, value, payload_ptr_val, 0);
    ops.push(store_val.op_ref());

    // 7. Identity pass-through so the last op produces the payload ptr result.
    //    Cranelift will optimize away iadd(ptr, 0).
    let zero_op = clif::iconst(ctx, loc, ptr_ty, 0);
    let zero_val = zero_op.result(ctx);
    ops.push(zero_op.op_ref());

    let identity_op = clif::iadd(ctx, loc, payload_ptr_val, zero_val, ptr_ty);
    ops.push(identity_op.op_ref());

    ops
}

/// Lower tribute_rt boxing/unboxing operations to clif dialect.
///
/// This is a partial lowering: only box/unbox operations are converted.
/// `retain`/`release` ops pass through (handled by a future RC lowering pass).
pub fn lower(ctx: &mut IrContext, module: ArenaModule, type_converter: ArenaTypeConverter) {
    // Pre-intern types for patterns
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    let f64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build());

    let applicator = PatternApplicator::new(type_converter)
        .add_pattern(BoxIntPattern {
            ptr_ty,
            i64_ty,
            i32_ty,
        })
        .add_pattern(UnboxIntPattern { i32_ty })
        .add_pattern(BoxNatPattern {
            ptr_ty,
            i64_ty,
            i32_ty,
        })
        .add_pattern(UnboxNatPattern { i32_ty })
        .add_pattern(BoxFloatPattern {
            ptr_ty,
            i64_ty,
            i32_ty,
        })
        .add_pattern(UnboxFloatPattern { f64_ty })
        .add_pattern(BoxBoolPattern {
            ptr_ty,
            i64_ty,
            i32_ty,
        })
        .add_pattern(UnboxBoolPattern { i32_ty });

    applicator.apply_partial(ctx, module);

    // Verify: tribute_rt.* ops (except retain/release) should be gone
    let mut target = ArenaConversionTarget::new();
    target.add_illegal_dialect("tribute_rt");
    target.add_legal_op("tribute_rt", "retain");
    target.add_legal_op("tribute_rt", "release");

    if let Some(body) = module.body(ctx) {
        let illegal = target.verify(ctx, body);
        assert!(
            illegal.is_empty(),
            "lower (tribute_rt_to_clif): unconverted tribute_rt.* ops remain: {:?}",
            illegal,
        );
    }
}

// =============================================================================
// Boxing Patterns (primitive → heap pointer)
// =============================================================================

struct BoxIntPattern {
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for BoxIntPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = tribute_rt::BoxInt::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = box_op.value(ctx);
        let mut ops = box_value(
            ctx,
            loc,
            value,
            4,
            super::rtti::RTTI_INT,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );
        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

struct BoxNatPattern {
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for BoxNatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = tribute_rt::BoxNat::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = box_op.value(ctx);
        let mut ops = box_value(
            ctx,
            loc,
            value,
            4,
            super::rtti::RTTI_NAT,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );
        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

struct BoxBoolPattern {
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for BoxBoolPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = tribute_rt::BoxBool::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = box_op.value(ctx);
        let mut ops = box_value(
            ctx,
            loc,
            value,
            4,
            super::rtti::RTTI_BOOL,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );
        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

struct BoxFloatPattern {
    ptr_ty: TypeRef,
    i64_ty: TypeRef,
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for BoxFloatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = tribute_rt::BoxFloat::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = box_op.value(ctx);
        let mut ops = box_value(
            ctx,
            loc,
            value,
            8,
            super::rtti::RTTI_FLOAT,
            self.ptr_ty,
            self.i64_ty,
            self.i32_ty,
        );
        let last = ops.pop().unwrap();
        for o in ops {
            rewriter.insert_op(o);
        }
        rewriter.replace_op(last);
        true
    }
}

// =============================================================================
// Unboxing Patterns (heap pointer → primitive)
// =============================================================================

struct UnboxIntPattern {
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for UnboxIntPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = tribute_rt::UnboxInt::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = unbox_op.value(ctx);

        // If the input is already i32, the unbox is a no-op (value was stored raw).
        if ctx.value_ty(value) == self.i32_ty {
            rewriter.erase_op(vec![value]);
            return true;
        }

        let load_op = clif::load(ctx, loc, value, self.i32_ty, 0);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}

struct UnboxNatPattern {
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for UnboxNatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = tribute_rt::UnboxNat::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = unbox_op.value(ctx);

        if ctx.value_ty(value) == self.i32_ty {
            rewriter.erase_op(vec![value]);
            return true;
        }

        let load_op = clif::load(ctx, loc, value, self.i32_ty, 0);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}

struct UnboxBoolPattern {
    i32_ty: TypeRef,
}

impl ArenaRewritePattern for UnboxBoolPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = tribute_rt::UnboxBool::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = unbox_op.value(ctx);

        if ctx.value_ty(value) == self.i32_ty {
            rewriter.erase_op(vec![value]);
            return true;
        }

        let load_op = clif::load(ctx, loc, value, self.i32_ty, 0);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}

struct UnboxFloatPattern {
    f64_ty: TypeRef,
}

impl ArenaRewritePattern for UnboxFloatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = tribute_rt::UnboxFloat::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let value = unbox_op.value(ctx);

        if ctx.value_ty(value) == self.f64_ty {
            rewriter.erase_op(vec![value]);
            return true;
        }

        let load_op = clif::load(ctx, loc, value, self.f64_ty, 0);
        rewriter.replace_op(load_op.op_ref());
        true
    }
}
