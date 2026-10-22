//! Target-shared structure of `effect.*` dispatch lowering.
//!
//! Every target lowers `effect.dispatch_tail` and `effect.dispatch_cps` inside
//! the representation/ABI boundary with the same shape: an evidence runtime
//! call selects the handler, the canonical closure's function is read and the
//! closure itself passed as its environment, and control transfers through an ordinary
//! `func.call_indirect` or a proper-tail `func.tail_call_indirect`. Only how
//! evidence runtime values are typed differs per target; callers supply those.

use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::ability::{self, compute_op_idx};

use crate::target_abi::physical_parameter_attrs;
use tribute_ir::dialect::adt;
use tribute_ir::dialect::{closure, effect, tribute_rt};
use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::PatternRewriter;
use trunk_ir::types::{Attribute, AttributeMap, Location, StringRef, TypeDataBuilder};

pub(crate) fn i32_type(ctx: &mut IrContext) -> TypeRef {
    TypeDataBuilder::new("core.i32").intern(ctx)
}

/// Insert the stable ability id of `ability` as an `i32` constant.
pub(crate) fn insert_ability_id(
    ctx: &mut IrContext,
    loc: Location,
    ability: &SymbolPath,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let i32_ty = i32_type(ctx);
    let ability_id = ability::ability_id_const(ctx, loc, i32_ty, ability);
    rewriter.insert_op(ability_id.op_ref());
    ability_id.result(ctx)
}

fn insert_op_idx(
    ctx: &mut IrContext,
    loc: Location,
    ability: &SymbolPath,
    op_name: StringRef,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let i32_ty = i32_type(ctx);
    let op_idx = compute_op_idx(ability, ctx.str(op_name));
    let constant = arith::Const::operands()
        .value(Attribute::Int(op_idx as i128))
        .results(i32_ty)
        .build(ctx, loc);
    rewriter.insert_op(constant.op_ref());
    constant.result(ctx)
}

/// Read a canonical closure's function reference. The closure itself is the
/// function's environment.
fn insert_closure_function(
    ctx: &mut IrContext,
    loc: Location,
    closure: ValueRef,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    let i32_ty = i32_type(ctx);
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let function = adt::StructGet::operands(closure)
        .r#type(closure_ty)
        .field(0)
        .results(i32_ty)
        .build(ctx, loc);
    rewriter.insert_op(function.op_ref());
    function.result(ctx)
}

/// `closure` at the canonical closure storage type.
fn canonical_closure(
    ctx: &mut IrContext,
    loc: Location,
    closure: ValueRef,
    rewriter: &mut PatternRewriter<'_>,
) -> ValueRef {
    if !closure::Closure::matches(ctx, ctx.value_ty(closure)) {
        return closure;
    }
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let cast = core::UnrealizedConversionCast::operands(closure)
        .results(closure_ty)
        .build(ctx, loc);
    rewriter.insert_op(cast.op_ref());
    cast.result(ctx)
}

/// Whether `op` is a well-formed tail dispatch: evidence and `anyref`
/// payload operands, one `anyref` result.
pub(crate) fn is_valid_tail_dispatch(ctx: &mut IrContext, op: OpRef) -> bool {
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let anyref_ty = tribute_rt::Anyref::params().intern(ctx).as_type_ref();
    let operands = ctx.op_operands(op).to_vec();
    let results = ctx.op_result_types(op).to_vec();
    if operands.len() != 2
        || results.len() != 1
        || ctx.op_attrs(op).get_symbol_ref("ability").is_none()
        || ctx.op_attrs(op).get_string_ref("op_name").is_none()
    {
        return false;
    }
    ctx.value_ty(operands[0]) == evidence_ty
        && ctx.value_ty(operands[1]) == anyref_ty
        && results[0] == anyref_ty
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
    let ability = dispatch_op.ability(ctx).clone();
    let op_idx = insert_op_idx(ctx, loc, &ability, dispatch_op.op_name_ref(ctx), rewriter);
    let function = insert_closure_function(ctx, loc, dispatch_closure, rewriter);
    let result_ty = ctx.op_result_types(op)[0];
    let args = [evidence, dispatch_closure, op_idx, payload];
    // The dispatch closure is a module-internal EvidenceDirect callable, so
    // it has the internal `tail` machine convention and parameter contract.
    let contract = physical_parameter_attrs(ctx, CallingConvention::EvidenceDirect);
    let parameters = args.map(|value| (ctx.value_ty(value), contract.clone()));
    let signature = func::func_sig_with_param_attrs(
        ctx,
        parameters,
        [(result_ty, AttributeMap::new())],
        AttributeMap::new(),
    )
    .with_call_conv(ctx, func::CallConv::Tail)
    .as_type_ref();
    let call = func::CallIndirect::operands(function, args)
        .signature(signature)
        .build(ctx, loc);
    rewriter.insert_op(call.op_ref());
    rewriter.erase_op(vec![call.result(ctx)]);
}

/// The fixed physical signature of a CPS handler dispatch closure.
pub(crate) fn cps_dispatch_signature(ctx: &mut IrContext) -> TypeRef {
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let anyref_ty = tribute_rt::Anyref::params().intern(ctx).as_type_ref();
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let i32_ty = i32_type(ctx);
    let contract = physical_parameter_attrs(ctx, CallingConvention::Cps);
    func::func_sig_with_param_attrs(
        ctx,
        [
            evidence_ty,
            closure_ty,
            closure_ty,
            i32_ty,
            i32_ty,
            i32_ty,
            anyref_ty,
        ]
        .map(|ty| (ty, contract.clone())),
        [],
        AttributeMap::new(),
    )
    .with_call_conv(ctx, func::CallConv::Tail)
    .as_type_ref()
}

/// Whether `op` is a well-formed final CPS dispatch: evidence, two closures,
/// and an `anyref` payload. A malformed one is left for the conversion target
/// to report, before any helper op is inserted.
pub(crate) fn is_valid_cps_dispatch(ctx: &mut IrContext, op: OpRef) -> bool {
    if !ctx.op_result_types(op).is_empty()
        || ctx.op_attrs(op).get_type("answer_type").is_none()
        || ctx.op_attrs(op).get_symbol_ref("ability").is_none()
        || ctx.op_attrs(op).get_string_ref("op_name").is_none()
        || ctx.op_operands(op).len() != 4
    {
        return false;
    }
    let evidence_ty = ability::evidence_adt_type_ref(ctx);
    let anyref_ty = tribute_rt::Anyref::params().intern(ctx).as_type_ref();
    let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
    let [evidence, dispatch, resume, payload] = ctx.op_operands(op)[..] else {
        return false;
    };
    // Closure lowering keeps a packed closure at its semantic closure type
    // until storage finalization.
    let is_closure = |ty: TypeRef| ty == closure_ty || closure::Closure::matches(ctx, ty);
    ctx.value_ty(evidence) == evidence_ty
        && is_closure(ctx.value_ty(dispatch))
        && is_closure(ctx.value_ty(resume))
        && ctx.value_ty(payload) == anyref_ty
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
    let ability = dispatch_op.ability(ctx).clone();
    let op_idx = insert_op_idx(ctx, loc, &ability, dispatch_op.op_name_ref(ctx), rewriter);
    // Closure lowering keeps packed closures at their semantic closure type
    // until storage finalization; retype them to the dispatch slots.
    let dispatch = canonical_closure(ctx, loc, dispatch_op.dispatch(ctx), rewriter);
    let function = insert_closure_function(ctx, loc, dispatch, rewriter);
    let resume = canonical_closure(ctx, loc, dispatch_op.resume(ctx), rewriter);

    let signature = cps_dispatch_signature(ctx);
    let tail = func::TailCallIndirect::operands(
        function,
        [
            dispatch_op.evidence(ctx),
            dispatch,
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

pub(crate) fn pack_payload(
    ctx: &mut IrContext,
    rewriter: &mut PatternRewriter<'_>,
    location: trunk_ir::types::Location,
    ability: &SymbolPath,
    op_name: StringRef,
    values: &[ValueRef],
    anyref: TypeRef,
) -> ValueRef {
    let payload_type =
        ability::operation_payload_type_ref(ctx, ability, op_name, values.iter().map(|_| anyref));
    let dynamic_values = values
        .iter()
        .map(|&value| {
            let cast = tribute_rt::convert_value(ctx, location, value, anyref);
            rewriter.insert_op(cast);
            ctx.op_result(cast, 0)
        })
        .collect::<Vec<_>>();
    let payload = adt::StructNew::operands(dynamic_values)
        .r#type(payload_type)
        .results(payload_type)
        .build(ctx, location);
    rewriter.insert_op(payload.op_ref());
    let erased = core::UnrealizedConversionCast::operands(payload.result(ctx))
        .results(anyref)
        .build(ctx, location);
    rewriter.insert_op(erased.op_ref());
    erased.result(ctx)
}
