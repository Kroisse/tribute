//! Builders of the done adapter, dispatch adapter factory, and rebound
//! resumption of a continuation frame layer. They take the frame types they
//! build over, so `tribute_control_to_cps` and `lower_continuation_frames`
//! share them.

use super::*;

pub(crate) fn make_block(ctx: &mut IrContext, location: Location, types: &[TypeRef]) -> BlockRef {
    ctx.create_block(BlockData {
        location,
        args: types
            .iter()
            .copied()
            .map(|ty| BlockArgData {
                ty,
                attrs: AttributeMap::new(),
            })
            .collect(),
        ops: Default::default(),
        parent_region: None,
    })
}

pub(crate) fn single_block_region(
    ctx: &mut IrContext,
    location: Location,
    block: BlockRef,
) -> RegionRef {
    ctx.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![block],
        parent_op: None,
    })
}

/// The name of the `index`th helper function of kind `prefix`.
pub(crate) fn helper_symbol(prefix: &str, index: u32) -> Symbol {
    Symbol::new(&format!("__tribute_{prefix}_{index}"))
}

/// Build the closure of `region`, capturing the values it uses from outside
/// in their order of first use.
pub(crate) fn closure_over(
    ctx: &mut IrContext,
    location: Location,
    region: RegionRef,
    closure_type: TypeRef,
    convention: CallingConvention,
) -> closure::Lambda {
    let lambda = closure::Lambda::operands(ordered_external_values(ctx, region))
        .results(closure_type)
        .regions(region)
        .build(ctx, location);
    set_calling_convention(ctx, lambda.op_ref(), convention);
    lambda
}

/// Emit a final CPS tail transfer from the callee's exact typed closure
/// contract. This must not reconstruct a signature from physical operands:
/// closure extraction interposes an environment later and preserves this
/// contract on the resulting indirect transfer.
pub(crate) fn emit_cps_tail_call_indirect(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    callee: ValueRef,
    args: impl IntoIterator<Item = ValueRef>,
) -> Result<OpRef, TributeControlToCpsError> {
    let args = args.into_iter().collect::<Vec<_>>();
    let closure_type = ctx.value_ty(callee);
    let signature = cps_closure_function_type(ctx, closure_type).ok_or_else(|| {
        TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail callee has no exact provenance-bearing closure contract",
        )
    })?;
    let callable = func::FuncSig::from_type_ref(ctx, signature).ok_or_else(|| {
        TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail callee contract is not func.func_sig",
        )
    })?;
    let never = core::never(ctx).as_type_ref();
    if callable.results(ctx) != [never]
        || callable.inputs(ctx).len() != args.len()
        || callable
            .inputs(ctx)
            .iter()
            .zip(&args)
            .any(|(expected, actual)| *expected != ctx.value_ty(*actual))
    {
        return Err(TributeControlToCpsError::post_at(
            location,
            "CPS indirect tail operands differ from the exact closure contract",
        ));
    }
    let tail = func::TailCallIndirect::operands(callee, args)
        .signature(signature)
        .build(ctx, location);
    set_calling_convention(ctx, tail.op_ref(), CallingConvention::Cps);
    ctx.push_op(block, tail.op_ref());
    Ok(tail.op_ref())
}

pub(crate) fn unpack_frame(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    frame: &FrameTypes,
    frame_value: ValueRef,
) -> (ValueRef, ValueRef) {
    let frame_value = if ctx.value_ty(frame_value) == frame.reference {
        frame_value
    } else {
        let cast = core::UnrealizedConversionCast::operands(frame_value)
            .results(frame.reference)
            .build(ctx, location);
        ctx.push_op(block, cast.op_ref());
        cast.result(ctx)
    };
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

pub(crate) fn pack_frame(
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

pub(crate) fn build_done_adapter(
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
pub(crate) fn build_suffix_rebound(
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
pub(crate) fn finish_rebound(
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
pub(crate) fn build_dispatch_adapter_factory(
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

fn collect_defined_values(ctx: &IrContext, region: RegionRef, defined: &mut HashSet<ValueRef>) {
    for block in ctx.region(region).blocks.iter().copied() {
        defined.extend(ctx.block_args(block).iter().copied());
        for op in ctx.block(block).ops.iter().copied() {
            defined.extend(ctx.op_results(op).iter().copied());
            for nested in ctx.op_regions(op) {
                collect_defined_values(ctx, nested, defined);
            }
        }
    }
}

fn collect_external_in_order(
    ctx: &IrContext,
    region: RegionRef,
    defined: &HashSet<ValueRef>,
    seen: &mut HashSet<ValueRef>,
    external: &mut Vec<ValueRef>,
) {
    for block in ctx.region(region).blocks.iter().copied() {
        for op in ctx.block(block).ops.iter().copied() {
            for operand in ctx.op_operands(op).iter().copied() {
                if !defined.contains(&operand) && seen.insert(operand) {
                    external.push(operand);
                }
            }
            for nested in ctx.op_regions(op) {
                collect_external_in_order(ctx, nested, defined, seen, external);
            }
        }
    }
}

fn ordered_external_values(ctx: &IrContext, region: RegionRef) -> Vec<ValueRef> {
    let mut defined = HashSet::default();
    collect_defined_values(ctx, region, &mut defined);
    let mut seen = HashSet::default();
    let mut external = Vec::new();
    collect_external_in_order(ctx, region, &defined, &mut seen, &mut external);
    external
}
