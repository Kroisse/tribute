//! Builders of an installed handle layer: its dispatcher and resumption
//! factories, resume tokens, marker dispatchers, and delimiter.

use std::ops::ControlFlow;

use rustc_hash::FxHashMap as HashMap;

use tribute_core::calling_convention::{
    cps_closure_function_type, cps_resume_exact_type, cps_resume_type,
    physical_closure_function_type,
};
use tribute_core::{CallingConvention, physical_closure_type, set_calling_convention};
use tribute_ir::dialect::ability::{HandlerBinding, OperationKind};
use tribute_ir::dialect::{ability, adt, effect, tribute_control, tribute_rt};
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, func, scf};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{Module, PatternRewriter, RewritePattern};
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{Symbol, SymbolPath};

use super::suffix_layer::{
    DispatchAdapters, LayerFrames, SuffixLayer, build_done_adapter, build_suffix_rebound,
    finish_rebound, frame_dispatch, pack_frame,
};
use super::{FrameLayouts, detach_into};
use crate::cps_builders::{
    closure_over, emit_cps_tail_call_indirect, helper_symbol, make_block, set_evidence_plan,
    single_block_region,
};
use crate::tribute_control_to_cps::TributeControlToCpsError;

/// One arm of a handle: the operation it handles and the closure it takes.
struct HandlerArm {
    binding: HandlerBinding,
    closure_type: TypeRef,
    /// The operation's parameters, followed by the source resume token of a
    /// resumptive arm.
    params: Vec<TypeRef>,
    resumptive: bool,
}

impl HandlerArm {
    fn new(ctx: &IrContext, binding: HandlerBinding, closure_type: TypeRef) -> Option<Self> {
        let resumptive = binding.is_resumptive(ctx);
        let params = match binding.kind {
            OperationKind::Op => {
                let function = cps_closure_function_type(ctx, closure_type)?;
                let inputs = func::FuncSig::from_type_ref(ctx, function)?.inputs(ctx);
                inputs.get(2..inputs.len() - usize::from(resumptive))?
            }
            OperationKind::Fn => {
                let function = physical_closure_function_type(
                    ctx,
                    closure_type,
                    CallingConvention::EvidenceDirect,
                )?;
                func::FuncSig::from_type_ref(ctx, function)?
                    .inputs(ctx)
                    .get(1..)?
            }
        }
        .to_vec();
        Some(Self {
            binding,
            closure_type,
            params,
            resumptive,
        })
    }

    fn is_resumptive(&self) -> bool {
        self.resumptive
    }

    fn general(&self) -> bool {
        self.binding.kind == OperationKind::Op
    }

    fn op_index(&self, ctx: &IrContext) -> u32 {
        ability::compute_op_idx(&self.binding.ability_ref, ctx.str(self.binding.op_name))
    }
}

/// The static description of one `ability.handle`, shared by every layer that
/// installs it: the first installation and each rebuilt continuation layer.
struct HandleLayer {
    arms: Vec<HandlerArm>,
    /// The frame of the body and the frame the handle exits to.
    frames: LayerFrames,
    completion_type: TypeRef,
    /// The `evidence_plan` of each installation.
    plan: Option<Attribute>,
    /// Builds the dispatcher of an installed layer.
    dispatch_factory: Symbol,
    /// Builds the dispatcher of a layer resumed from a lambda, which keeps
    /// only the handle's completion.
    passthrough_factory: Symbol,
    /// Builds the resumption that installs a layer again.
    installed_resume_factory: Symbol,
    /// Builds the resumption of a layer resumed from a lambda.
    passthrough_resume_factory: Symbol,
}

/// The values one installed layer of a handle runs with.
struct LayerValues {
    completion: ValueRef,
    prompt: ValueRef,
    /// The handler arm closures, in `HandleLayer::arms` order.
    arms: Vec<ValueRef>,
}

fn evidence_type(ctx: &mut IrContext) -> TypeRef {
    ability::evidence_adt_type_ref(ctx)
}

fn anyref_type(ctx: &mut IrContext) -> TypeRef {
    tribute_rt::anyref(ctx).as_type_ref()
}

fn i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
}

/// The block arguments of a `Dispatch<R>` over `frame`.
fn dispatch_block(ctx: &mut IrContext, location: Location, frame: TypeRef) -> BlockRef {
    let evidence_type = evidence_type(ctx);
    let anyref = anyref_type(ctx);
    let i32_type = i32_type(ctx);
    let resume_type = cps_resume_type(ctx, evidence_type, frame, anyref);
    make_block(
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
    )
}

/// Wrap `block` into the `Direct` function `symbol`.
fn direct_function(
    ctx: &mut IrContext,
    location: Location,
    symbol: &Symbol,
    signature: TypeRef,
    block: BlockRef,
) -> OpRef {
    let region = single_block_region(ctx, location, block);
    let function = func::Func::operands()
        .sym_name(ctx.intern_symbol_text(symbol))
        .r#type(signature)
        .regions(region)
        .build(ctx, location);
    set_calling_convention(ctx, function.op_ref(), CallingConvention::Direct);
    function.op_ref()
}

/// Build the resumption of an installed handle layer.
///
/// The evidence it is resumed with is the layer's outer evidence: the
/// completion and the arms run with it, and the handle is installed on it
/// again under the same prompt for the resumed computation.
fn build_handle_rebound(
    ctx: &mut IrContext,
    location: Location,
    layer: &HandleLayer,
    values: &LayerValues,
    resume_body: ValueRef,
) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
    let boundary = layer.frames.boundary;
    let evidence_type = evidence_type(ctx);
    let anyref = anyref_type(ctx);
    let block = make_block(ctx, location, &[evidence_type, boundary.reference, anyref]);
    let args = ctx.block_args(block).to_vec();
    let body_block = make_block(ctx, location, &[evidence_type]);
    let body_evidence = ctx.block_args(body_block)[0];
    let frame = push_layer_frame(ctx, block, location, layer, values, (args[0], args[1]))?;
    emit_cps_tail_call_indirect(
        ctx,
        body_block,
        location,
        resume_body,
        [body_evidence, frame, args[2]],
    )?;
    let body_region = single_block_region(ctx, location, body_block);
    push_handle_dispatch(ctx, block, location, layer, values, args[0], body_region)?;
    Ok(finish_rebound(ctx, location, &boundary, block))
}

/// Build the frame of one installed layer: its `Done` enters the completion
/// with `outer_evidence` and `exit_frame`, and its dispatcher is the layer's.
fn push_layer_frame(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    layer: &HandleLayer,
    values: &LayerValues,
    (outer_evidence, exit_frame): (ValueRef, ValueRef),
) -> Result<ValueRef, TributeControlToCpsError> {
    let value = layer.frames.value;
    let (done_op, done) = build_done_adapter(
        ctx,
        value.answer,
        values.completion,
        outer_evidence,
        exit_frame,
        location,
    )?;
    ctx.push_op(block, done_op);
    let mut args = vec![values.completion, exit_frame, values.prompt, outer_evidence];
    args.extend(values.arms.iter().copied());
    let dispatch = func::Call::operands(args)
        .callee(layer.dispatch_factory.clone().into())
        .results([value.dispatch])
        .build(ctx, location);
    set_calling_convention(ctx, dispatch.op_ref(), CallingConvention::Direct);
    ctx.push_op(block, dispatch.op_ref());
    Ok(pack_frame(
        ctx,
        block,
        location,
        &value,
        done,
        dispatch.result(ctx),
    ))
}

/// The ability instances a handle handles, in first-arm order.
fn layer_ability_refs(layer: &HandleLayer) -> Vec<SymbolPath> {
    let mut ability_refs = Vec::new();
    for arm in &layer.arms {
        if !ability_refs.contains(&arm.binding.ability_ref) {
            ability_refs.push(arm.binding.ability_ref.clone());
        }
    }
    ability_refs
}

/// Install a handle layer: extend `outer_evidence` for `body`, with marker
/// dispatchers whose arms run with `outer_evidence`. For each handled
/// instance, the marker's dispatcher runs its `fn` arms. General operations
/// read only the marker's prompt and dispatch through the continuation frame.
fn push_handle_dispatch(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    layer: &HandleLayer,
    values: &LayerValues,
    outer_evidence: ValueRef,
    body: RegionRef,
) -> Result<(), TributeControlToCpsError> {
    let ability_refs = layer_ability_refs(layer);
    let mut dispatchers = Vec::new();
    for ability_ref in &ability_refs {
        let arms: Vec<_> = layer
            .arms
            .iter()
            .zip(values.arms.iter().copied())
            .filter(|(arm, _)| arm.binding.ability_ref == *ability_ref && !arm.general())
            .collect();
        let (dispatcher_op, dispatcher) =
            build_tail_dispatcher(ctx, location, &arms, outer_evidence)?;
        ctx.push_op(block, dispatcher_op);
        dispatchers.push(dispatcher);
    }
    let dispatch = ability::HandleDispatch::operands(outer_evidence, values.prompt, dispatchers)
        .ability_refs(ability_refs)
        .regions(body)
        .build(ctx, location);
    set_evidence_plan(ctx, dispatch.op_ref(), layer.plan.clone());
    ctx.push_op(block, dispatch.op_ref());
    Ok(())
}

/// Build the function that makes a resumption of one handle's layer, so
/// every token and dispatcher of the handle shares one copy of it.
///
/// The function's parameters are `(completion, prompt, resume_body,
/// arms...)` for a resumption that installs the layer again, and
/// `(completion, resume_body)` for one resumed from a lambda.
fn build_layer_resume_factory(
    ctx: &mut IrContext,
    location: Location,
    layer: &HandleLayer,
    installed: bool,
) -> Result<OpRef, TributeControlToCpsError> {
    let LayerFrames { value, boundary } = layer.frames;
    let evidence_type = evidence_type(ctx);
    let anyref = anyref_type(ctx);
    let resume_body_type = cps_resume_type(ctx, evidence_type, value.reference, anyref);
    let resume_type = cps_resume_type(ctx, evidence_type, boundary.reference, anyref);
    let params = if installed {
        let mut params = vec![layer.completion_type, i32_type(ctx), resume_body_type];
        params.extend(layer.arms.iter().map(|arm| arm.closure_type));
        params
    } else {
        vec![layer.completion_type, resume_body_type]
    };
    let block = make_block(ctx, location, &params);
    let args = ctx.block_args(block).to_vec();
    let (resume_op, resume) = if installed {
        let values = LayerValues {
            completion: args[0],
            prompt: args[1],
            arms: args[3..].to_vec(),
        };
        build_handle_rebound(ctx, location, layer, &values, args[2])?
    } else {
        // A resume in a lambda carries the lambda's evidence, whose
        // handlers the resumed computation keeps: the layer leaves only
        // its completion behind.
        let completion_only = SuffixLayer {
            value_type: value.answer,
            dispatch_factory: layer.passthrough_factory.clone(),
            plan: None,
        };
        build_suffix_rebound(
            ctx,
            location,
            &completion_only,
            &layer.frames,
            args[1],
            args[0],
        )?
    };
    ctx.push_op(block, resume_op);
    let ret = func::Return::operands([resume]).build(ctx, location);
    ctx.push_op(block, ret.op_ref());
    let symbol = if installed {
        &layer.installed_resume_factory
    } else {
        &layer.passthrough_resume_factory
    };
    let signature = func::func_sig(ctx, params, [resume_type]).as_type_ref();
    Ok(direct_function(ctx, location, symbol, signature, block))
}

/// Call a handle's resumption factory for `resume_body`: the resumption
/// that installs the layer again, or the one resumed from a lambda.
fn call_layer_resume_factory(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    layer: &HandleLayer,
    values: &LayerValues,
    resume_body: ValueRef,
    installed: bool,
) -> ValueRef {
    let (symbol, args) = if installed {
        let mut args = vec![values.completion, values.prompt, resume_body];
        args.extend(values.arms.iter().copied());
        (layer.installed_resume_factory.clone(), args)
    } else {
        (
            layer.passthrough_resume_factory.clone(),
            vec![values.completion, resume_body],
        )
    };
    let evidence_type = evidence_type(ctx);
    let anyref = anyref_type(ctx);
    let resume_type = cps_resume_type(ctx, evidence_type, layer.frames.boundary.reference, anyref);
    let resume = func::Call::operands(args)
        .callee(symbol.into())
        .results([resume_type])
        .build(ctx, location);
    set_calling_convention(ctx, resume.op_ref(), CallingConvention::Direct);
    ctx.push_op(block, resume.op_ref());
    resume.result(ctx)
}

/// Build one resume token of a general arm: the exact-typed entry to a
/// resumption of its handle layer, made by the handle's factory when the
/// token is used.
fn build_exact_handler_token(
    ctx: &mut IrContext,
    location: Location,
    layer: &HandleLayer,
    values: &LayerValues,
    input_type: TypeRef,
    resume_body: ValueRef,
    installed: bool,
) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
    let evidence_type = evidence_type(ctx);
    let frame_type = layer.frames.boundary.reference;
    let block = make_block(ctx, location, &[evidence_type, frame_type, input_type]);
    let args = ctx.block_args(block).to_vec();
    let anyref = anyref_type(ctx);
    let erased_input = core::UnrealizedConversionCast::operands(args[2])
        .results(anyref)
        .build(ctx, location);
    ctx.push_op(block, erased_input.op_ref());
    let rebound =
        call_layer_resume_factory(ctx, block, location, layer, values, resume_body, installed);
    emit_cps_tail_call_indirect(
        ctx,
        block,
        location,
        rebound,
        [args[0], args[1], erased_input.result(ctx)],
    )?;
    let region = single_block_region(ctx, location, block);
    let token_type = cps_resume_exact_type(ctx, evidence_type, input_type, frame_type);
    let lambda = closure_over(ctx, location, region, token_type, CallingConvention::Cps);
    Ok((lambda.op_ref(), lambda.result(ctx)))
}

/// Build an immutable factory for the dispatcher of one handle. Each
/// installed layer of the handle calls it with the frame the handle exits
/// to and the layer's outer evidence, so the dispatcher's arms run with
/// the evidence and continuation of that layer.
///
/// The factory's parameters are `(completion, exit_frame, prompt,
/// outer_evidence, arms...)`.
fn build_local_dispatcher_factory(
    ctx: &mut IrContext,
    location: Location,
    layer: &HandleLayer,
) -> Result<OpRef, TributeControlToCpsError> {
    let LayerFrames { value, boundary } = layer.frames;
    let mut params = vec![
        layer.completion_type,
        boundary.reference,
        i32_type(ctx),
        evidence_type(ctx),
    ];
    params.extend(layer.arms.iter().map(|arm| arm.closure_type));
    let signature = func::func_sig(ctx, params.clone(), [value.dispatch]).as_type_ref();
    let block = make_block(ctx, location, &params);
    let args = ctx.block_args(block).to_vec();
    let values = LayerValues {
        completion: args[0],
        prompt: args[2],
        arms: args[4..].to_vec(),
    };
    let parent_dispatch = frame_dispatch(ctx, block, location, &boundary, args[1]);
    let (dispatcher_op, dispatcher) = build_local_dispatcher_instance(
        ctx,
        location,
        layer,
        &values,
        args[1],
        parent_dispatch,
        args[3],
    )?;
    ctx.push_op(block, dispatcher_op);
    let ret = func::Return::operands([dispatcher]).build(ctx, location);
    ctx.push_op(block, ret.op_ref());
    Ok(direct_function(
        ctx,
        location,
        &layer.dispatch_factory,
        signature,
        block,
    ))
}

/// Build the dispatch of an operation this handle layer does not handle:
/// the operation goes to the parent dispatcher with a resumption that
/// installs this layer again.
fn build_local_foreign_dispatch(
    ctx: &mut IrContext,
    location: Location,
    layer: &HandleLayer,
    values: &LayerValues,
    parent_dispatch: ValueRef,
) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
    let value = layer.frames.value;
    let block = dispatch_block(ctx, location, value.reference);
    let args = ctx.block_args(block).to_vec();
    let rebound = call_layer_resume_factory(ctx, block, location, layer, values, args[1], true);
    emit_cps_tail_call_indirect(
        ctx,
        block,
        location,
        parent_dispatch,
        [args[0], rebound, args[2], args[3], args[4], args[5]],
    )?;
    let region = single_block_region(ctx, location, block);
    let lambda = closure_over(
        ctx,
        location,
        region,
        value.dispatch,
        CallingConvention::Cps,
    );
    Ok((lambda.op_ref(), lambda.result(ctx)))
}

/// Build the dispatcher of one installed handle layer. A general
/// arm runs with the layer's outer evidence and exits to `exit_frame`.
fn build_local_dispatcher_instance(
    ctx: &mut IrContext,
    location: Location,
    layer: &HandleLayer,
    values: &LayerValues,
    exit_frame: ValueRef,
    parent_dispatch: ValueRef,
    outer_evidence: ValueRef,
) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
    let value = layer.frames.value;
    let block = dispatch_block(ctx, location, value.reference);
    let args = ctx.block_args(block).to_vec();
    let (foreign_op, foreign_dispatch) =
        build_local_foreign_dispatch(ctx, location, layer, values, parent_dispatch)?;
    ctx.push_op(block, foreign_op);
    let never = core::never(ctx).as_type_ref();
    let switch_block = make_block(ctx, location, &[]);
    let general_arms = layer
        .arms
        .iter()
        .zip(&values.arms)
        .filter(|(arm, _)| arm.general());
    for (arm, &arm_value) in general_arms {
        let case_block = make_block(ctx, location, &[]);
        let same_prompt = arith::Cmpi::operands(args[2], values.prompt)
            .predicate("eq")
            .build(ctx, location);
        ctx.push_op(case_block, same_prompt.op_ref());

        let local_block = make_block(ctx, location, &[]);
        let mut call_args = vec![outer_evidence, exit_frame];
        call_args.extend(unpack_handler_payload(
            ctx,
            local_block,
            location,
            args[5],
            arm,
        ));
        if arm.resumptive {
            let input_type = *arm.params.last().expect("resumptive arm has a token");
            let token_input = cps_closure_function_type(ctx, input_type)
                .and_then(|function| func::FuncSig::from_type_ref(ctx, function))
                .and_then(|function| function.inputs(ctx).get(2).copied())
                .ok_or_else(|| {
                    TributeControlToCpsError::post_at(
                        location,
                        "handler resume token lacks an exact callable input",
                    )
                })?;
            // The source token resumes from a lambda; the second token
            // resumes from the arm body, under this handle installed
            // again on the arm's evidence at the resume.
            for installed in [false, true] {
                let (token_op, token) = build_exact_handler_token(
                    ctx,
                    location,
                    layer,
                    values,
                    token_input,
                    args[1],
                    installed,
                )?;
                ctx.push_op(local_block, token_op);
                call_args.push(token);
            }
        }
        emit_cps_tail_call_indirect(ctx, local_block, location, arm_value, call_args)?;
        let local_region = single_block_region(ctx, location, local_block);
        let fallback_block = make_block(ctx, location, &[]);
        emit_cps_tail_call_indirect(
            ctx,
            fallback_block,
            location,
            foreign_dispatch,
            [args[0], args[1], args[2], args[3], args[4], args[5]],
        )?;
        let fallback_region = single_block_region(ctx, location, fallback_block);
        let choose = scf::If::operands(same_prompt.result(ctx))
            .results(never)
            .regions(local_region, fallback_region)
            .build(ctx, location);
        ctx.push_op(case_block, choose.op_ref());
        let case_region = single_block_region(ctx, location, case_block);
        let case = scf::Case::operands()
            .value(Attribute::Int(arm.op_index(ctx) as i128))
            .regions(case_region)
            .build(ctx, location);
        ctx.push_op(switch_block, case.op_ref());
    }
    let default_block = make_block(ctx, location, &[]);
    emit_cps_tail_call_indirect(
        ctx,
        default_block,
        location,
        foreign_dispatch,
        [args[0], args[1], args[2], args[3], args[4], args[5]],
    )?;
    let foreign_region = single_block_region(ctx, location, default_block);
    let default = scf::Default::operands()
        .regions(foreign_region)
        .build(ctx, location);
    ctx.push_op(switch_block, default.op_ref());
    let switch_region = single_block_region(ctx, location, switch_block);
    let switch = scf::Switch::operands(args[4])
        .regions(switch_region)
        .build(ctx, location);
    ctx.push_op(block, switch.op_ref());
    let region = single_block_region(ctx, location, block);
    let lambda = closure_over(
        ctx,
        location,
        region,
        value.dispatch,
        CallingConvention::Cps,
    );
    Ok((lambda.op_ref(), lambda.result(ctx)))
}

fn unpack_handler_payload(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    payload: ValueRef,
    arm: &HandlerArm,
) -> Vec<ValueRef> {
    let value_params = &arm.params[..arm.params.len() - usize::from(arm.resumptive)];
    let anyref = anyref_type(ctx);
    let payload_type = ability::operation_payload_type_ref(
        ctx,
        &arm.binding.ability_ref,
        arm.binding.op_name,
        value_params.iter().map(|_| anyref),
    );
    let cast = core::UnrealizedConversionCast::operands(payload)
        .results(payload_type)
        .build(ctx, location);
    ctx.push_op(block, cast.op_ref());
    value_params
        .iter()
        .copied()
        .enumerate()
        .map(|(index, ty)| {
            let get = adt::StructGet::operands(cast.result(ctx))
                .r#type(payload_type)
                .field(index as u32)
                .results(anyref)
                .build(ctx, location);
            ctx.push_op(block, get.op_ref());
            let recovered = core::UnrealizedConversionCast::operands(get.result(ctx))
                .results(ty)
                .build(ctx, location);
            ctx.push_op(block, recovered.op_ref());
            recovered.result(ctx)
        })
        .collect()
}

/// Build the marker's dispatcher for the `fn` arms of one ability
/// instance. The arms run with `outer_evidence`, the evidence of the
/// layer that installed the marker, not with the evidence of the
/// operation that reaches them.
fn build_tail_dispatcher(
    ctx: &mut IrContext,
    location: Location,
    arms: &[(&HandlerArm, ValueRef)],
    outer_evidence: ValueRef,
) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
    let anyref = anyref_type(ctx);
    let params = vec![evidence_type(ctx), i32_type(ctx), anyref];
    let block = make_block(ctx, location, &params);
    let args = ctx.block_args(block).to_vec();
    let (op_idx, payload) = (args[1], args[2]);

    let switch_block = make_block(ctx, location, &[]);
    for &(arm, arm_value) in arms {
        let case_block = make_block(ctx, location, &[]);
        let mut call_args = vec![outer_evidence];
        call_args.extend(unpack_handler_payload(
            ctx, case_block, location, payload, arm,
        ));
        let signature = physical_closure_function_type(
            ctx,
            arm.closure_type,
            CallingConvention::EvidenceDirect,
        )
        .ok_or_else(|| {
            TributeControlToCpsError::post_at(
                location,
                "fn handler indirect callee has no exact provenance-bearing closure contract",
            )
        })?;
        let call = func::CallIndirect::operands(arm_value, call_args)
            .signature(signature)
            .build(ctx, location);
        set_calling_convention(ctx, call.op_ref(), CallingConvention::EvidenceDirect);
        ctx.push_op(case_block, call.op_ref());
        let erased = core::UnrealizedConversionCast::operands(call.result(ctx))
            .results(anyref)
            .build(ctx, location);
        ctx.push_op(case_block, erased.op_ref());
        let ret = func::Return::operands([erased.result(ctx)]).build(ctx, location);
        ctx.push_op(case_block, ret.op_ref());
        let case_region = single_block_region(ctx, location, case_block);
        let case = scf::Case::operands()
            .value(Attribute::Int(arm.op_index(ctx) as i128))
            .regions(case_region)
            .build(ctx, location);
        ctx.push_op(switch_block, case.op_ref());
    }
    let reject_block = make_block(ctx, location, &[]);
    let unreachable = func::Unreachable::operands().build(ctx, location);
    ctx.push_op(reject_block, unreachable.op_ref());
    let reject_region = single_block_region(ctx, location, reject_block);
    let default = scf::Default::operands()
        .regions(reject_region)
        .build(ctx, location);
    ctx.push_op(switch_block, default.op_ref());
    let switch_region = single_block_region(ctx, location, switch_block);
    let switch = scf::Switch::operands(op_idx)
        .regions(switch_region)
        .build(ctx, location);
    ctx.push_op(block, switch.op_ref());

    let region = single_block_region(ctx, location, block);
    let function = func::func_sig(ctx, params, [anyref]).as_type_ref();
    let closure_type = physical_closure_type(ctx, function, CallingConvention::EvidenceDirect);
    let lambda = closure_over(
        ctx,
        location,
        region,
        closure_type,
        CallingConvention::EvidenceDirect,
    );
    Ok((lambda.op_ref(), lambda.result(ctx)))
}

/// Number the module's `ability.handle` operations in walk order. The number
/// names a handle's factories.
pub(super) fn number_handles(ctx: &IrContext, module: Module) -> HashMap<OpRef, u32> {
    let mut handles = HashMap::default();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if ability::Handle::matches(ctx, op) {
            handles.insert(op, handles.len() as u32);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    handles
}

/// Expands `ability.handle`.
pub(super) struct ExpandHandles {
    pub(super) frames: FrameLayouts,
    pub(super) adapters: DispatchAdapters,
    pub(super) handles: HashMap<OpRef, u32>,
}

impl RewritePattern for ExpandHandles {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        ability::Handle::from_op(ctx, op)
            .is_ok_and(|handle| self.expand_handle(ctx, handle, rewriter).is_some())
    }
}

impl ExpandHandles {
    /// Replace `ability.handle` with the first installation of its layer:
    /// the layer's factories, a fresh prompt, and the `ability.handle_dispatch`
    /// whose body starts by building the layer's frame.
    fn expand_handle(
        &self,
        ctx: &mut IrContext,
        handle: ability::Handle,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = handle.op_ref();
        let location = ctx.op(op).location;
        let evidence = handle.evidence(ctx);
        let exit = handle.exit(ctx);
        let completion = handle.completion(ctx);
        let arm_values = handle.arms(ctx).to_vec();
        let bindings: Vec<_> = handle.handlers(ctx).collect();
        let [source_block] = ctx.region(handle.body(ctx)).blocks[..] else {
            return None;
        };
        let [source_evidence, source_frame] = ctx.block_args(source_block)[..] else {
            return None;
        };
        if bindings.len() != arm_values.len() {
            return None;
        }
        let arms = bindings
            .into_iter()
            .zip(&arm_values)
            .map(|(binding, arm)| HandlerArm::new(ctx, binding, ctx.value_ty(*arm)))
            .collect::<Option<Vec<_>>>()?;
        let frames = LayerFrames {
            value: self.frames.of(ctx, ctx.value_ty(source_frame))?,
            boundary: self.frames.of(ctx, ctx.value_ty(exit))?,
        };
        let passthrough_factory = self.adapters.factory(&frames, None)?;
        let index = *self.handles.get(&op)?;
        let layer = HandleLayer {
            arms,
            frames,
            completion_type: ctx.value_ty(completion),
            plan: ctx
                .op(op)
                .attributes
                .get(tribute_control::EVIDENCE_PLAN_ATTR)
                .cloned(),
            dispatch_factory: helper_symbol("make_local_dispatch", index),
            passthrough_factory,
            installed_resume_factory: helper_symbol("make_installed_resume", index),
            passthrough_resume_factory: helper_symbol("make_passthrough_resume", index),
        };
        let mut factories = vec![
            build_local_dispatcher_factory(ctx, location, &layer).ok()?,
            build_layer_resume_factory(ctx, location, &layer, true).ok()?,
        ];
        if layer.arms.iter().any(HandlerArm::is_resumptive) {
            factories.push(build_layer_resume_factory(ctx, location, &layer, false).ok()?);
        }

        let block = make_block(ctx, location, &[]);
        let prompt = effect::FreshPromptTag::operands().build(ctx, location);
        ctx.push_op(block, prompt.op_ref());
        let values = LayerValues {
            completion,
            prompt: prompt.result(ctx),
            arms: arm_values,
        };
        let body_block = make_block(ctx, location, &[ctx.value_ty(source_evidence)]);
        let frame =
            push_layer_frame(ctx, body_block, location, &layer, &values, (evidence, exit)).ok()?;
        for moved in ctx.block(source_block).ops.clone() {
            ctx.remove_op_from_block(source_block, moved);
            ctx.push_op(body_block, moved);
        }
        ctx.replace_all_uses(source_evidence, ctx.block_args(body_block)[0]);
        ctx.replace_all_uses(source_frame, frame);
        let body = single_block_region(ctx, location, body_block);
        push_handle_dispatch(ctx, block, location, &layer, &values, evidence, body).ok()?;
        let delimiter = *ctx.block(block).ops.last()?;
        ctx.remove_op_from_block(block, delimiter);
        detach_into(ctx, block, rewriter);
        for factory in factories {
            rewriter.add_module_op(factory);
        }
        rewriter.replace_op(delimiter);
        Some(())
    }
}
