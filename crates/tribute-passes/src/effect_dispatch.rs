//! Target-shared structure of `effect.*` dispatch lowering.
//!
//! Every target lowers `effect.dispatch_tail` and `effect.dispatch_cps` inside
//! the representation/ABI boundary with the same shape: an evidence runtime
//! call selects the handler, the canonical closure is decomposed into its
//! function and environment, and control transfers through an ordinary
//! `func.call_indirect` or a proper-tail `func.tail_call_indirect`. Only how
//! evidence runtime values are typed differs per target; callers supply those.

use tribute_ir::dialect::ability::{self, compute_op_idx};
use tribute_ir::dialect::{closure, effect, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{PatternRewriter, TypeConverter};
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

pub(crate) fn i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
}

/// Insert the stable ability id of `ability_ref` as an `i32` constant.
pub(crate) fn insert_ability_id(
    ctx: &mut IrContext,
    loc: Location,
    ability_ref: TypeRef,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let i32_ty = i32_type(ctx);
    let ability_id = ability::ability_id_const(ctx, loc, i32_ty, ability_ref);
    rewriter.insert_op(ability_id.op_ref());
    ability_id.result(ctx)
}

fn insert_op_idx(
    ctx: &mut IrContext,
    loc: Location,
    ability_ref: TypeRef,
    op_name: Symbol,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let i32_ty = i32_type(ctx);
    let op_idx = compute_op_idx(ability::ability_name(ctx, ability_ref), Some(op_name));
    let constant = arith::Const::operands()
        .value(Attribute::Int(op_idx as i128))
        .results(i32_ty)
        .build(ctx, loc);
    rewriter.insert_op(constant.op_ref());
    constant.result(ctx)
}

/// Decompose a canonical closure into its function reference and environment.
fn insert_closure_parts(
    ctx: &mut IrContext,
    loc: Location,
    closure: ValueRef,
    rewriter: &mut PatternRewriter<'_>,
) -> (ValueRef, ValueRef) {
    let i32_ty = i32_type(ctx);
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let function = adt::StructGet::operands(closure)
        .r#type(closure_ty)
        .field(0)
        .results(i32_ty)
        .build(ctx, loc);
    rewriter.insert_op(function.op_ref());
    let env = adt::StructGet::operands(closure)
        .r#type(closure_ty)
        .field(1)
        .results(anyref_ty)
        .build(ctx, loc);
    rewriter.insert_op(env.op_ref());
    (function.result(ctx), env.result(ctx))
}

/// Whether `op` is a well-formed tail dispatch for a target whose converter
/// is `converter`: evidence and payload operands, one `anyref` result.
pub(crate) fn is_valid_tail_dispatch(
    ctx: &mut IrContext,
    op: OpRef,
    converter: &TypeConverter,
) -> bool {
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let operands = ctx.op_operands(op).to_vec();
    let results = ctx.op_result_types(op).to_vec();
    if operands.len() != 2
        || results.len() != 1
        || ctx.op(op).attributes.get_type("ability_ref").is_none()
        || ctx.op(op).attributes.get_symbol("op_name").is_none()
    {
        return false;
    }
    let same = |ctx: &mut IrContext, actual: TypeRef, expected: TypeRef| {
        converter.convert_type_or_identity(ctx, actual)
            == converter.convert_type_or_identity(ctx, expected)
    };
    same(ctx, ctx.value_ty(operands[0]), evidence_ty)
        && same(ctx, ctx.value_ty(operands[1]), anyref_ty)
        && same(ctx, results[0], anyref_ty)
}

/// Replace a tail dispatch with an ordinary indirect call through the
/// tail-resumptive dispatch closure selected by the evidence runtime.
pub(crate) fn lower_tail_dispatch(
    ctx: &mut IrContext,
    op: OpRef,
    dispatch_closure: ValueRef,
    rewriter: &mut PatternRewriter<'_>,
) {
    let dispatch_op = effect::DispatchTail::from_op(ctx, op).expect("tail dispatch");
    let loc = ctx.op(op).location;
    let evidence = dispatch_op.evidence(ctx);
    let payload = dispatch_op.payload(ctx);
    let op_idx = insert_op_idx(
        ctx,
        loc,
        dispatch_op.ability_ref(ctx),
        dispatch_op.op_name(ctx),
        rewriter,
    );
    let (function, env) = insert_closure_parts(ctx, loc, dispatch_closure, rewriter);
    let result_ty = ctx.op_result_types(op)[0];
    let args = [evidence, env, op_idx, payload];
    let parameters = args.map(|value| ctx.value_ty(value));
    let signature = func::func_sig(ctx, parameters, [result_ty]).as_type_ref();
    let call = func::CallIndirect::operands(function, args)
        .signature(signature)
        .build(ctx, loc);
    rewriter.insert_op(call.op_ref());
    rewriter.erase_op(vec![call.result(ctx)]);
}

/// The fixed physical signature of a CPS handler dispatch closure.
pub(crate) fn cps_dispatch_signature(ctx: &mut IrContext) -> TypeRef {
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let i32_ty = i32_type(ctx);
    func::func_sig(
        ctx,
        [
            evidence_ty,
            anyref_ty,
            closure_ty,
            i32_ty,
            i32_ty,
            i32_ty,
            anyref_ty,
        ],
        [],
    )
    .with_call_conv(ctx, func::CallConv::Tail)
    .as_type_ref()
}

/// Whether `op` is a well-formed final CPS dispatch for a target whose
/// converter is `converter`. A malformed one is left for the conversion
/// target to report, before any helper op is inserted.
pub(crate) fn is_valid_cps_dispatch(
    ctx: &mut IrContext,
    op: OpRef,
    converter: &TypeConverter,
) -> bool {
    if !ctx.op_result_types(op).is_empty()
        || ctx.op(op).attributes.get_type("answer_type").is_none()
        || ctx.op(op).attributes.get_type("ability_ref").is_none()
        || ctx.op(op).attributes.get_symbol("op_name").is_none()
        || ctx.op_operands(op).len() != 4
    {
        return false;
    }
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let expected = [evidence_ty, closure_ty, closure_ty, anyref_ty];
    ctx.op_operands(op)
        .to_vec()
        .into_iter()
        .zip(expected)
        .all(|(value, expected)| {
            converter.convert_type_or_identity(ctx, ctx.value_ty(value))
                == converter.convert_type_or_identity(ctx, expected)
        })
}

/// Replace a final CPS dispatch with a proper-tail transfer to its handler
/// dispatch closure, passing the handler's `prompt` and the ability id.
pub(crate) fn lower_cps_dispatch(
    ctx: &mut IrContext,
    op: OpRef,
    ability_id: ValueRef,
    prompt: ValueRef,
    rewriter: &mut PatternRewriter<'_>,
) {
    let dispatch_op = effect::DispatchCps::from_op(ctx, op).expect("CPS dispatch");
    let loc = ctx.op(op).location;
    let op_idx = insert_op_idx(
        ctx,
        loc,
        dispatch_op.ability_ref(ctx),
        dispatch_op.op_name(ctx),
        rewriter,
    );
    let (function, env) = insert_closure_parts(ctx, loc, dispatch_op.dispatch(ctx), rewriter);

    // Closure lowering keeps a packed continuation at its semantic closure
    // type until storage finalization; retype it to the dispatch slot.
    let mut resume = dispatch_op.resume(ctx);
    if closure::Closure::matches(ctx, ctx.value_ty(resume)) {
        let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
        let cast = core::UnrealizedConversionCast::operands(resume)
            .results(closure_ty)
            .build(ctx, loc);
        rewriter.insert_op(cast.op_ref());
        resume = cast.result(ctx);
    }

    let signature = cps_dispatch_signature(ctx);
    let tail = func::TailCallIndirect::operands(
        function,
        [
            dispatch_op.evidence(ctx),
            env,
            resume,
            prompt,
            ability_id,
            op_idx,
            dispatch_op.payload(ctx),
        ],
    )
    .signature(signature)
    .build(ctx, loc);
    rewriter.replace_op(tail.op_ref());
}
