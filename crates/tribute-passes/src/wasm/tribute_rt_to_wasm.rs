//! Lower tribute_rt dialect operations to wasm dialect.
//!
//! This pass converts boxing/unboxing operations to their wasm equivalents:
//! - `tribute_rt.box_int` -> `wasm.ref_i31` (i32 -> i31ref)
//! - `tribute_rt.unbox_int` -> `wasm.ref_cast` (i31ref) + `wasm.i31_get_s`
//! - `tribute_rt.box_nat` -> `wasm.ref_i31` (i32 -> i31ref)
//! - `tribute_rt.unbox_nat` -> `wasm.ref_cast` (i31ref) + `wasm.i31_get_u`
//! - `tribute_rt.box_float` -> `adt.struct_new` (f64 -> BoxedF64 struct)
//! - `tribute_rt.unbox_float` -> `adt.ref_cast` (BoxedF64) + `adt.struct_get`
//! - `tribute_rt.box_bool` -> `wasm.ref_i31` (i32 -> i31ref)
//! - `tribute_rt.unbox_bool` -> `wasm.ref_cast` (i31ref) + `wasm.i31_get_u`
//!
//! ## Type Mappings
//!
//! - `tribute_rt.int` -> `core.i32`
//! - `tribute_rt.nat` -> `core.i32`
//! - `tribute_rt.float` -> `core.f64`
//! - `tribute_rt.bool` -> `core.i32`
//! - `tribute_rt.intref` -> `wasm.i31ref`
//! - `tribute_rt.any` -> `wasm.anyref`

use tribute_ir::arena::dialect::tribute_rt as arena_tribute_rt;
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, Location, TypeDataBuilder};
use trunk_ir::ir::Symbol;

/// Helper to create arena type refs for common types.
fn i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

fn f64_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("f64")).build())
}

fn i31ref_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("i31ref")).build())
}

fn anyref_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("wasm"), Symbol::new("anyref")).build())
}

/// Get the BoxedF64 struct type: `adt.struct(f64, name="_BoxedF64")`
fn boxed_f64_type(ctx: &mut IrContext) -> TypeRef {
    let f64_ty = f64_type(ctx);
    ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .param(f64_ty)
            .attr("name", ArenaAttribute::Symbol(Symbol::new("_BoxedF64")))
            .build(),
    )
}

/// Create i31 unbox operations (ref_cast to i31ref + i31_get_s/u).
///
/// Returns `(prefix_ops, final_op)` where prefix_ops should be inserted before
/// and final_op is the replacement.
fn create_i31_unbox(
    ctx: &mut IrContext,
    location: Location,
    value: ValueRef,
    is_signed: bool,
) -> (Vec<OpRef>, OpRef) {
    let i31ref_ty = i31ref_type(ctx);
    let i32_ty = i32_type(ctx);

    // Cast anyref to i31ref first (abstract type, no type_idx needed)
    let cast_op = arena_wasm::ref_cast(ctx, location, value, i31ref_ty, i31ref_ty, None);
    let cast_result = cast_op.result(ctx);

    // Extract value: signed or unsigned
    let get_op_ref = if is_signed {
        arena_wasm::i31_get_s(ctx, location, cast_result, i32_ty).op_ref()
    } else {
        arena_wasm::i31_get_u(ctx, location, cast_result, i32_ty).op_ref()
    };

    (vec![cast_op.op_ref()], get_op_ref)
}

/// Lower tribute_rt dialect to wasm dialect.
pub fn lower(ctx: &mut IrContext, module: ArenaModule) {
    let applicator = PatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(BoxIntPattern)
        .add_pattern(UnboxIntPattern)
        .add_pattern(BoxNatPattern)
        .add_pattern(UnboxNatPattern)
        .add_pattern(BoxFloatPattern)
        .add_pattern(UnboxFloatPattern)
        .add_pattern(BoxBoolPattern)
        .add_pattern(UnboxBoolPattern);
    applicator.apply_partial(ctx, module);
}

/// Pattern for `tribute_rt.box_int` -> `wasm.ref_i31`
///
/// Boxing converts an unboxed i32 to an i31ref via `wasm.ref_i31`.
struct BoxIntPattern;

impl ArenaRewritePattern for BoxIntPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = arena_tribute_rt::BoxInt::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let value = box_op.value(ctx);

        let i31ref_ty = i31ref_type(ctx);

        // wasm.ref_i31: i32 -> i31ref
        let new_op = arena_wasm::ref_i31(ctx, location, value, i31ref_ty);

        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxIntPattern"
    }
}

/// Pattern for `tribute_rt.unbox_int` -> `wasm.ref_cast` + `wasm.i31_get_s`
///
/// Unboxing extracts the signed i32 from an i31ref.
struct UnboxIntPattern;

impl ArenaRewritePattern for UnboxIntPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = arena_tribute_rt::UnboxInt::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let (prefix_ops, final_op) = create_i31_unbox(ctx, location, unbox_op.value(ctx), true);
        for prefix in prefix_ops {
            rewriter.insert_op(prefix);
        }
        rewriter.replace_op(final_op);
        true
    }

    fn name(&self) -> &'static str {
        "UnboxIntPattern"
    }
}

/// Pattern for `tribute_rt.box_nat` -> `wasm.ref_i31`
///
/// Boxing converts an unboxed nat (u32) to an i31ref.
struct BoxNatPattern;

impl ArenaRewritePattern for BoxNatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = arena_tribute_rt::BoxNat::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let value = box_op.value(ctx);
        let i31ref_ty = i31ref_type(ctx);

        let new_op = arena_wasm::ref_i31(ctx, location, value, i31ref_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxNatPattern"
    }
}

/// Pattern for `tribute_rt.unbox_nat` -> `wasm.ref_cast` + `wasm.i31_get_u`
///
/// Unboxing extracts the unsigned i32 from an i31ref.
struct UnboxNatPattern;

impl ArenaRewritePattern for UnboxNatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = arena_tribute_rt::UnboxNat::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let (prefix_ops, final_op) = create_i31_unbox(ctx, location, unbox_op.value(ctx), false);
        for prefix in prefix_ops {
            rewriter.insert_op(prefix);
        }
        rewriter.replace_op(final_op);
        true
    }

    fn name(&self) -> &'static str {
        "UnboxNatPattern"
    }
}

/// Pattern for `tribute_rt.box_float` -> `adt.struct_new(BoxedF64, anyref)`
///
/// Boxing converts an f64 to a BoxedF64 struct typed as anyref via
/// `adt.struct_new` with `anyref_ty` as the result type.
struct BoxFloatPattern;

impl ArenaRewritePattern for BoxFloatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = arena_tribute_rt::BoxFloat::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let value = box_op.value(ctx);
        let anyref_ty = anyref_type(ctx);
        let boxed_f64_ty = boxed_f64_type(ctx);

        // adt.struct_new creates BoxedF64 struct with the f64 value
        let struct_op = arena_adt::struct_new(ctx, location, vec![value], anyref_ty, boxed_f64_ty);

        rewriter.replace_op(struct_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxFloatPattern"
    }
}

/// Pattern for `tribute_rt.unbox_float` -> `adt.ref_cast(BoxedF64)` + `adt.struct_get`
///
/// Unboxing extracts the f64 from a BoxedF64 struct.
struct UnboxFloatPattern;

impl ArenaRewritePattern for UnboxFloatPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = arena_tribute_rt::UnboxFloat::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let value = unbox_op.value(ctx);

        let boxed_f64_ty = boxed_f64_type(ctx);
        let f64_ty = f64_type(ctx);

        // Cast anyref to BoxedF64 struct first
        let anyref_ty = anyref_type(ctx);
        let cast_op = arena_adt::ref_cast(ctx, location, value, anyref_ty, boxed_f64_ty);
        let cast_result = cast_op.result(ctx);

        // adt.struct_get extracts field 0 (the f64 value) from BoxedF64
        let get_op = arena_adt::struct_get(ctx, location, cast_result, f64_ty, boxed_f64_ty, 0);

        rewriter.insert_op(cast_op.op_ref());
        rewriter.replace_op(get_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "UnboxFloatPattern"
    }
}

/// Pattern for `tribute_rt.box_bool` -> `wasm.ref_i31`
///
/// Boxing converts an i32 boolean to an i31ref.
struct BoxBoolPattern;

impl ArenaRewritePattern for BoxBoolPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(box_op) = arena_tribute_rt::BoxBool::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let value = box_op.value(ctx);
        let i31ref_ty = i31ref_type(ctx);

        let new_op = arena_wasm::ref_i31(ctx, location, value, i31ref_ty);
        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BoxBoolPattern"
    }
}

/// Pattern for `tribute_rt.unbox_bool` -> `wasm.ref_cast` + `wasm.i31_get_u`
///
/// Unboxing extracts the boolean (0 or 1) from an i31ref.
struct UnboxBoolPattern;

impl ArenaRewritePattern for UnboxBoolPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(unbox_op) = arena_tribute_rt::UnboxBool::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        // Use unsigned extraction since bool is 0 or 1
        let (prefix_ops, final_op) = create_i31_unbox(ctx, location, unbox_op.value(ctx), false);
        for prefix in prefix_ops {
            rewriter.insert_op(prefix);
        }
        rewriter.replace_op(final_op);
        true
    }

    fn name(&self) -> &'static str {
        "UnboxBoolPattern"
    }
}
