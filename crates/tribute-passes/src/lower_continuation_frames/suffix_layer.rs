//! Builders of a suffix layer of a continuation frame: its done adapter,
//! dispatch adapter factory, and rebound resumption, and the frame struct
//! that holds a layer's `Done` and dispatcher.

use tribute_core::calling_convention::{cps_completion_type, cps_done_type, cps_resume_type};
use tribute_core::{CallingConvention, set_calling_convention};
use tribute_ir::dialect::{ability, adt, tribute_control, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::PatternRewriter;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

use super::{ExpandFrameOperations, FrameTypes, detach_into};
use crate::cps_builders::{
    closure_over, emit_cps_tail_call_indirect, make_block, set_evidence_plan, single_block_region,
};
use crate::tribute_control_to_cps::TributeControlToCpsError;

/// The frame types a suffix layer builds around: its own and the frame
/// around it.
#[derive(Clone, Copy)]
pub(super) struct LayerFrames {
    pub(super) value: FrameTypes,
    pub(super) boundary: FrameTypes,
}

/// A call, resume, or structured suffix layer of a continuation.
#[derive(Clone)]
pub(super) struct SuffixLayer {
    pub(super) value_type: TypeRef,
    /// Builds the dispatcher that rebuilds this layer when it is resumed.
    pub(super) dispatch_factory: Symbol,
    /// The `evidence_plan` that selects the evidence of the computation the
    /// layer continues.
    pub(super) plan: Option<Attribute>,
}

pub(super) fn unpack_frame(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    frame: &FrameTypes,
    frame_value: ValueRef,
) -> (ValueRef, ValueRef) {
    let done = adt::StructGet::operands(frame_value)
        .r#type(frame.layout)
        .field(0)
        .results(frame.done)
        .build(ctx, location);
    ctx.push_op(block, done.op_ref());
    let dispatch = adt::StructGet::operands(frame_value)
        .r#type(frame.layout)
        .field(1)
        .results(frame.dispatch)
        .build(ctx, location);
    ctx.push_op(block, dispatch.op_ref());
    (done.result(ctx), dispatch.result(ctx))
}

pub(super) fn pack_frame(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    frame: &FrameTypes,
    done: ValueRef,
    dispatch: ValueRef,
) -> ValueRef {
    let packed = adt::StructNew::operands([done, dispatch])
        .r#type(frame.layout)
        .results(frame.reference)
        .build(ctx, location);
    ctx.push_op(block, packed.op_ref());
    packed.result(ctx)
}

pub(super) fn build_done_adapter(
    ctx: &mut IrContext,
    value_type: TypeRef,
    completion: ValueRef,
    evidence: ValueRef,
    outer_frame: ValueRef,
    location: Location,
) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
    let done_block = make_block(ctx, location, &[value_type]);
    let value = ctx.block_args(done_block)[0];
    emit_cps_tail_call_indirect(
        ctx,
        done_block,
        location,
        completion,
        [evidence, outer_frame, value],
    )?;
    let region = single_block_region(ctx, location, done_block);
    let done_type = cps_done_type(ctx, value_type);
    let done = closure_over(ctx, location, region, done_type, CallingConvention::Cps);
    Ok((done.op_ref(), done.result(ctx)))
}

/// Build the resumption of a call, resume, or structured suffix layer.
///
/// The suffix keeps the evidence the layer is resumed with; the resumed
/// computation receives that evidence after the selection of `plan`.
pub(super) fn build_suffix_rebound(
    ctx: &mut IrContext,
    location: Location,
    layer: &SuffixLayer,
    frames: &LayerFrames,
    resume_body: ValueRef,
    completion: ValueRef,
) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
    let SuffixLayer {
        value_type,
        dispatch_factory,
        plan,
    } = layer.clone();
    let LayerFrames { value, boundary } = *frames;
    let evidence_type = ability::evidence_adt_type_ref(ctx);
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let block = make_block(ctx, location, &[evidence_type, boundary.reference, anyref]);
    let args = ctx.block_args(block).to_vec();
    let (done_op, done) =
        build_done_adapter(ctx, value_type, completion, args[0], args[1], location)?;
    ctx.push_op(block, done_op);
    let (_, outer_dispatch) = unpack_frame(ctx, block, location, &boundary, args[1]);
    let dispatch = func::Call::operands([completion, outer_dispatch])
        .callee(dispatch_factory.into())
        .results([value.dispatch])
        .build(ctx, location);
    set_calling_convention(ctx, dispatch.op_ref(), CallingConvention::Direct);
    ctx.push_op(block, dispatch.op_ref());
    let frame = pack_frame(ctx, block, location, &value, done, dispatch.result(ctx));
    let transfer =
        emit_cps_tail_call_indirect(ctx, block, location, resume_body, [args[0], frame, args[2]])?;
    set_evidence_plan(ctx, transfer, plan);
    Ok(finish_rebound(ctx, location, &boundary, block))
}

/// Wrap a rebound resumption block into its `Resume` closure.
pub(super) fn finish_rebound(
    ctx: &mut IrContext,
    location: Location,
    boundary: &FrameTypes,
    block: BlockRef,
) -> (OpRef, ValueRef) {
    let evidence_type = ability::evidence_adt_type_ref(ctx);
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let region = single_block_region(ctx, location, block);
    let resume_type = cps_resume_type(ctx, evidence_type, boundary.reference, anyref);
    let resume = closure_over(ctx, location, region, resume_type, CallingConvention::Cps);
    (resume.op_ref(), resume.result(ctx))
}

/// Build the function that makes the dispatcher of a suffix layer from its
/// completion and the dispatcher of the frame around it.
pub(super) fn build_dispatch_adapter_factory(
    ctx: &mut IrContext,
    location: Location,
    symbol: Symbol,
    frames: &LayerFrames,
    plan: Option<Attribute>,
) -> Result<OpRef, TributeControlToCpsError> {
    let LayerFrames { value, boundary } = *frames;
    let evidence_type = ability::evidence_adt_type_ref(ctx);
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let i32_type = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
    let completion_type = cps_completion_type(ctx, evidence_type, value.answer, boundary.reference);
    let factory_type =
        func::func_sig(ctx, [completion_type, boundary.dispatch], [value.dispatch]).as_type_ref();
    let factory_block = make_block(ctx, location, &[completion_type, boundary.dispatch]);
    let factory_args = ctx.block_args(factory_block).to_vec();
    let resume_type = cps_resume_type(ctx, evidence_type, value.reference, anyref);
    let dispatch_block = make_block(
        ctx,
        location,
        &[
            evidence_type,
            resume_type,
            i32_type,
            i32_type,
            i32_type,
            anyref,
        ],
    );
    let dispatch_args = ctx.block_args(dispatch_block).to_vec();
    let layer = SuffixLayer {
        value_type: value.answer,
        dispatch_factory: symbol.clone(),
        plan,
    };
    let (resume_op, rebound_resume) = build_suffix_rebound(
        ctx,
        location,
        &layer,
        frames,
        dispatch_args[1],
        factory_args[0],
    )?;
    ctx.push_op(dispatch_block, resume_op);
    emit_cps_tail_call_indirect(
        ctx,
        dispatch_block,
        location,
        factory_args[1],
        [
            dispatch_args[0],
            rebound_resume,
            dispatch_args[2],
            dispatch_args[3],
            dispatch_args[4],
            dispatch_args[5],
        ],
    )?;
    let dispatch_region = single_block_region(ctx, location, dispatch_block);
    let dispatch = closure_over(
        ctx,
        location,
        dispatch_region,
        value.dispatch,
        CallingConvention::Cps,
    );
    ctx.push_op(factory_block, dispatch.op_ref());
    let ret = func::Return::operands([dispatch.result(ctx)]).build(ctx, location);
    ctx.push_op(factory_block, ret.op_ref());
    let factory_region = single_block_region(ctx, location, factory_block);
    let factory = func::Func::operands()
        .sym_name(ctx.intern_symbol_text(&symbol))
        .r#type(factory_type)
        .regions(factory_region)
        .build(ctx, location);
    set_calling_convention(ctx, factory.op_ref(), CallingConvention::Direct);
    Ok(factory.op_ref())
}

impl ExpandFrameOperations {
    /// Replace `ability.suffix_frame` with the done adapter, the dispatch
    /// adapter factory over the outer frame's dispatcher, and the frame that
    /// holds both.
    pub(super) fn expand_suffix_frame(
        &self,
        ctx: &mut IrContext,
        suffix: ability::SuffixFrame,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let location = ctx.op(suffix.op_ref()).location;
        let evidence = suffix.evidence(ctx);
        let outer = suffix.outer(ctx);
        let continuation = suffix.continuation(ctx);
        let plan = ctx
            .op(suffix.op_ref())
            .attributes
            .get(tribute_control::EVIDENCE_PLAN_ATTR)
            .cloned();
        let value = self.frame_of(ctx, suffix.result_ty(ctx))?;
        let boundary = self.frame_of(ctx, ctx.value_ty(outer))?;
        let block = make_block(ctx, location, &[]);
        let (done_op, done) =
            build_done_adapter(ctx, value.answer, continuation, evidence, outer, location).ok()?;
        ctx.push_op(block, done_op);
        let (_, outer_dispatch) = unpack_frame(ctx, block, location, &boundary, outer);
        let symbol = self.fresh_helper("make_dispatch_adapter");
        let frames = LayerFrames { value, boundary };
        let factory =
            build_dispatch_adapter_factory(ctx, location, symbol.clone(), &frames, plan).ok()?;
        let evidence_type = ctx.value_ty(evidence);
        let completion_type =
            cps_completion_type(ctx, evidence_type, value.answer, boundary.reference);
        let completion = core::UnrealizedConversionCast::operands(continuation)
            .results(completion_type)
            .build(ctx, location);
        ctx.push_op(block, completion.op_ref());
        let dispatch = func::Call::operands([completion.result(ctx), outer_dispatch])
            .callee(symbol.into())
            .results([value.dispatch])
            .build(ctx, location);
        tribute_core::set_calling_convention(
            ctx,
            dispatch.op_ref(),
            tribute_core::CallingConvention::Direct,
        );
        ctx.push_op(block, dispatch.op_ref());
        let frame = pack_frame(ctx, block, location, &value, done, dispatch.result(ctx));
        detach_into(ctx, block, rewriter);
        rewriter.add_module_op(factory);
        rewriter.erase_op(vec![frame]);
        Some(())
    }

    /// Replace `ability.exit` with the transfer to the frame's `Done<R>`.
    pub(super) fn expand_exit(
        &self,
        ctx: &mut IrContext,
        exit: ability::Exit,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let location = ctx.op(exit.op_ref()).location;
        let frame = exit.frame(ctx);
        let value = exit.value(ctx);
        let types = self.frame_of(ctx, ctx.value_ty(frame))?;
        let block = make_block(ctx, location, &[]);
        let (done, _) = unpack_frame(ctx, block, location, &types, frame);
        let transfer = emit_cps_tail_call_indirect(ctx, block, location, done, [value]).ok()?;
        ctx.remove_op_from_block(block, transfer);
        detach_into(ctx, block, rewriter);
        rewriter.replace_op(transfer);
        Some(())
    }
}
