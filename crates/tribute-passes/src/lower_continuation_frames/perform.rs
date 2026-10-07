//! Builders of the resumption an expanded `ability.perform` or
//! `ability.abort` passes to the dispatcher.

use tribute_core::calling_convention::{cps_closure_function_type, cps_resume_type};
use tribute_core::{CallingConvention, set_calling_convention};
use tribute_ir::dialect::{ability, adt, closure, effect, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, func, scf};
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::PatternRewriter;
use trunk_ir::types::{Attribute, AttributeMap, Location, TypeDataBuilder};

use super::suffix_layer::unpack_frame;
use super::{ExpandFrameOperations, detach_into};
use crate::cps_builders::{
    FrameTypes, closure_over, emit_cps_tail_call_indirect, make_block, single_block_region,
};
use crate::effect_dispatch::pack_payload;
use crate::tribute_control_to_cps::TributeControlToCpsError;

/// The block of a `Resume<R>` over `frame` and its closure type.
fn resume_block(
    ctx: &mut IrContext,
    location: Location,
    frame: &FrameTypes,
) -> (BlockRef, TypeRef) {
    let evidence_type = ability::evidence_adt_type_ref(ctx);
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let block = make_block(ctx, location, &[evidence_type, frame.reference, anyref]);
    let resume_type = cps_resume_type(ctx, evidence_type, frame.reference, anyref);
    (block, resume_type)
}

/// Wrap the raw resumption of an `ability.perform` into a `Resume<R>` that
/// enters it once: a second call traps before the continuation is re-entered.
/// The state struct is named `state_name`. The built operations go to `block`.
fn push_one_shot_resume(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    frame: &FrameTypes,
    raw_resumption: ValueRef,
    state_name: &Symbol,
) -> Result<ValueRef, TributeControlToCpsError> {
    let input_type = cps_closure_function_type(ctx, ctx.value_ty(raw_resumption))
        .and_then(|function| func::FuncSig::from_type_ref(ctx, function))
        .and_then(|function| function.inputs(ctx).get(2).copied())
        .ok_or_else(|| {
            TributeControlToCpsError::post_at(
                location,
                "ability.perform resumption lacks an exact callable input",
            )
        })?;
    let i1_type = ctx.intern_type(TypeDataBuilder::new("core", "i1").build());
    let state_name = ctx.intern_symbol_text(state_name);
    let state_type = adt::struct_type(
        ctx,
        state_name,
        [("consumed", i1_type)],
        AttributeMap::new(),
    )
    .as_type_ref();
    let not_consumed = arith::Const::operands()
        .value(Attribute::Int(0))
        .results(i1_type)
        .build(ctx, location);
    ctx.push_op(block, not_consumed.op_ref());
    let state = adt::StructNew::operands([not_consumed.result(ctx)])
        .r#type(state_type)
        .results(state_type)
        .build(ctx, location);
    ctx.push_op(block, state.op_ref());
    let state = state.result(ctx);

    // The dispatcher ABI is existential only at this boundary. Keep the
    // captured continuation exact, recover this operation's declared input,
    // then transfer in proper tail position.
    let (wrapper_block, resume_type) = resume_block(ctx, location, frame);
    let args = ctx.block_args(wrapper_block).to_vec();
    let input_data = ctx.get_type(input_type);
    let input = if input_data.dialect == "core" && input_data.name == "nil" {
        // Nil has no physical payload: its exact resumption receives the
        // canonical unit instead of an erased runtime value.
        let unit = core::NilValue::operands().build(ctx, location);
        ctx.push_op(wrapper_block, unit.op_ref());
        unit.result(ctx)
    } else if input_data.dialect == "adt" && input_data.name == "typeref" {
        // Dynamic effect values recover nominal references only through
        // their declared type, preserving the typed ownership boundary.
        let recovered = adt::RefCast::operands(args[2])
            .r#type(input_type)
            .results(input_type)
            .build(ctx, location);
        ctx.push_op(wrapper_block, recovered.op_ref());
        recovered.result(ctx)
    } else {
        let recovered = core::UnrealizedConversionCast::operands(args[2])
            .results(input_type)
            .build(ctx, location);
        ctx.push_op(wrapper_block, recovered.op_ref());
        recovered.result(ctx)
    };
    let consumed = adt::StructGet::operands(state)
        .r#type(state_type)
        .field(0)
        .results(i1_type)
        .build(ctx, location);
    ctx.push_op(wrapper_block, consumed.op_ref());

    let reject_block = make_block(ctx, location, &[]);
    let unreachable = func::Unreachable::operands().build(ctx, location);
    ctx.push_op(reject_block, unreachable.op_ref());
    let reject_region = single_block_region(ctx, location, reject_block);

    let enter_block = make_block(ctx, location, &[]);
    let consumed_true = arith::Const::operands()
        .value(Attribute::Int(1))
        .results(i1_type)
        .build(ctx, location);
    ctx.push_op(enter_block, consumed_true.op_ref());
    let mark = adt::StructSet::operands(state, consumed_true.result(ctx))
        .r#type(state_type)
        .field(0)
        .build(ctx, location);
    ctx.push_op(enter_block, mark.op_ref());
    emit_cps_tail_call_indirect(
        ctx,
        enter_block,
        location,
        raw_resumption,
        [args[0], args[1], input],
    )?;
    let enter_region = single_block_region(ctx, location, enter_block);

    let never = core::never(ctx).as_type_ref();
    let guard = scf::If::operands(consumed.result(ctx))
        .results(never)
        .regions(reject_region, enter_region)
        .build(ctx, location);
    ctx.push_op(wrapper_block, guard.op_ref());
    let region = single_block_region(ctx, location, wrapper_block);
    let wrapper = closure_over(ctx, location, region, resume_type, CallingConvention::Cps);
    ctx.push_op(block, wrapper.op_ref());
    Ok(wrapper.result(ctx))
}

/// Build the `Resume<R>` of an `ability.abort`: it captures nothing and traps
/// when called. The built operation goes to `block`.
fn push_reject_resume(
    ctx: &mut IrContext,
    block: BlockRef,
    location: Location,
    frame: &FrameTypes,
) -> ValueRef {
    let (reject_block, resume_type) = resume_block(ctx, location, frame);
    let unreachable = func::Unreachable::operands().build(ctx, location);
    ctx.push_op(reject_block, unreachable.op_ref());
    let region = single_block_region(ctx, location, reject_block);
    let lambda = closure::Lambda::operands(std::iter::empty::<ValueRef>())
        .results(resume_type)
        .regions(region)
        .build(ctx, location);
    set_calling_convention(ctx, lambda.op_ref(), CallingConvention::Cps);
    ctx.push_op(block, lambda.op_ref());
    lambda.result(ctx)
}

impl ExpandFrameOperations {
    /// Replace `ability.perform` or `ability.abort` with `effect.dispatch_cps`
    /// through the frame's dispatcher. A perform passes its raw resumption
    /// behind a one-shot check, and an abort a resumption that traps.
    pub(super) fn expand_dispatch(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        frame: ValueRef,
        resumption: Option<ValueRef>,
        values: &[ValueRef],
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let location = ctx.op(op).location;
        let evidence = ctx.op_operands(op)[0];
        let ability_ref = ctx.op(op).attributes.get_type("ability_ref")?;
        let op_name = ctx.op(op).attributes.get_string_ref("op_name")?;
        let types = self.frame_of(ctx, ctx.value_ty(frame))?;
        let block = make_block(ctx, location, &[]);
        let resume = match resumption {
            Some(raw) => {
                let state = self.fresh_helper("one_shot_state");
                push_one_shot_resume(ctx, block, location, &types, raw, &state).ok()?
            }
            None => push_reject_resume(ctx, block, location, &types),
        };
        let (_, dispatch) = unpack_frame(ctx, block, location, &types, frame);
        detach_into(ctx, block, rewriter);
        let anyref = tribute_rt::anyref(ctx).as_type_ref();
        let payload = pack_payload(
            ctx,
            rewriter,
            location,
            ability_ref,
            op_name,
            values,
            anyref,
        );
        let dispatch = effect::DispatchCps::operands(evidence, dispatch, resume, payload)
            .ability_ref(ability_ref)
            .op_name(op_name)
            .answer_type(types.answer)
            .build(ctx, location);
        rewriter.replace_op(dispatch.op_ref());
        Some(())
    }
}
