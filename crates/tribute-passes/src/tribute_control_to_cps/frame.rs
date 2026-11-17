//! Continuation frames: suffix continuations and their frames, and the exits
//! that transfer through a frame.

use super::*;

/// The parent operation of `op`.
fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    ctx.region(ctx.block(block).parent_region?).parent_op
}

/// Whether `op` is the `tribute_control.yield` that ends the body of a
/// handle.
fn is_handle_body_yield(ctx: &IrContext, op: OpRef) -> bool {
    tribute_control::Yield::matches(ctx, op)
        && parent_op(ctx, op).is_some_and(|parent| {
            tribute_control::Handle::from_op(ctx, parent).is_ok_and(|handle| {
                ctx.op(op)
                    .parent_block
                    .and_then(|block| ctx.block(block).parent_region)
                    == Some(handle.body(ctx))
            })
        })
}

impl Converter<'_> {
    pub(super) fn fresh_helper(&mut self, prefix: &str) -> Symbol {
        let index = self.helper_index;
        self.helper_index += 1;
        helper_symbol(prefix, index)
    }

    /// The opaque `ability.frame<R>` that callables, completions, and
    /// resumptions take in the frame position.
    pub(super) fn frame_type(&mut self, answer: TypeRef) -> TypeRef {
        ability::Frame::params(answer)
            .intern(self.ctx)
            .as_type_ref()
    }

    /// The terminator of the flow that the result of `source`, followed by
    /// `rest`, flows into unchanged: a `return` or `tribute_control.yield`
    /// that takes the result right after `source`, or the `scf.yield` of an
    /// arm of a structured operation in tail position (`join`), whose own
    /// result flows the same way.
    pub(super) fn flow_exit(
        &self,
        source: OpRef,
        rest: Rest<'_>,
        join: Option<&TailJoin>,
    ) -> Option<OpRef> {
        let &[terminator] = rest.ops.get(rest.start..)? else {
            return None;
        };
        if self.ctx.op_operands(terminator) != self.ctx.op_results(source) {
            return None;
        }
        if tribute_control::Return::matches(self.ctx, terminator)
            || tribute_control::Yield::matches(self.ctx, terminator)
        {
            return Some(terminator);
        }
        let join = join?;
        if !scf::Yield::matches(self.ctx, terminator)
            || parent_op(self.ctx, terminator) != Some(join.structured)
        {
            return None;
        }
        let rest = Rest {
            ops: &join.ops,
            start: join.start,
        };
        self.flow_exit(join.structured, rest, join.outer.as_deref())
    }

    /// The frame of `flow` when `source`, a handle or a resume without an
    /// `evidence_plan`, or a call without one that makes the result of a
    /// handle body, can pass it on instead of a suffix frame: its result
    /// flows into the flow's terminator unchanged
    /// (`design/cps-effects.md`, "Frame을 그대로 넘기는 자리").
    pub(super) fn tail_exit_frame(
        &self,
        source: OpRef,
        rest: Rest<'_>,
        flow: &Flow,
    ) -> Option<ValueRef> {
        if flow.convention != CallingConvention::Cps {
            return None;
        }
        let handle = tribute_control::Handle::matches(self.ctx, source);
        // A selection is applied again on the resumed path by the layer of
        // its suffix frame.
        if !handle && evidence_plan_of(self.ctx, source).is_some() {
            return None;
        }
        let exit = self.flow_exit(source, rest, flow.tail_join.as_deref())?;
        let passes_frame = handle
            || tribute_control::Resume::matches(self.ctx, source)
            || is_handle_body_yield(self.ctx, exit);
        // The result is the terminator's operand, so its type is the
        // flow's answer type and the flow's frame takes it.
        passes_frame.then_some(flow.exit_k).flatten()
    }

    /// Build the frame a suffix continuation is entered through. `plan` is
    /// the `evidence_plan` that selects the evidence of the computation the
    /// frame is passed to.
    pub(super) fn frame_for_suffix(
        &mut self,
        block: BlockRef,
        location: Location,
        value_type: TypeRef,
        flow: &Flow,
        suffix: ValueRef,
        plan: Option<Attribute>,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let outer = flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::post_at(
                location,
                "CPS transfer has no verified ContinuationFrame",
            )
        })?;
        let evidence = flow.evidence.ok_or_else(|| {
            TributeControlToCpsError::post_at(location, "CPS transfer has no verified evidence")
        })?;
        let completion_type = self.completion_type(value_type, flow.answer_type);
        let suffix = if self.ctx.value_ty(suffix) == completion_type {
            suffix
        } else {
            let typed = core::UnrealizedConversionCast::operands(suffix)
                .results(completion_type)
                .build(self.ctx, location);
            self.ctx.push_op(block, typed.op_ref());
            typed.result(self.ctx)
        };
        let frame_type = self.frame_type(value_type);
        let frame = ability::SuffixFrame::operands(evidence, outer, suffix)
            .results(frame_type)
            .build(self.ctx, location);
        set_evidence_plan(self.ctx, frame.op_ref(), plan);
        self.ctx.push_op(block, frame.op_ref());
        Ok(frame.result(self.ctx))
    }

    pub(super) fn emit_exit(
        &mut self,
        block: BlockRef,
        location: Location,
        value: ValueRef,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention == CallingConvention::Cps {
            let exit_k = flow.exit_k.ok_or_else(|| {
                TributeControlToCpsError::post_at(
                    location,
                    "CPS region has no verified exit continuation",
                )
            })?;
            let exit = ability::Exit::operands(exit_k, value).build(self.ctx, location);
            self.ctx.push_op(block, exit.op_ref());
        } else {
            let ret = func::Return::operands([value]).build(self.ctx, location);
            self.ctx.push_op(block, ret.op_ref());
        }
        Ok(())
    }

    pub(super) fn emit_void_exit(
        &mut self,
        block: BlockRef,
        location: Location,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if let Some(void_exit) = flow.void_exit_k {
            let evidence = flow.evidence.ok_or_else(|| {
                TributeControlToCpsError::post_at(
                    location,
                    "zero-result suffix has no verified evidence",
                )
            })?;
            let frame = flow.exit_k.ok_or_else(|| {
                TributeControlToCpsError::post_at(
                    location,
                    "zero-result suffix has no verified ContinuationFrame",
                )
            })?;
            emit_cps_tail_call_indirect(self.ctx, block, location, void_exit, [evidence, frame])?;
            return Ok(());
        }
        Err(TributeControlToCpsError::post_at(
            location,
            "zero-result structured exit has no suffix continuation",
        ))
    }

    pub(super) fn build_void_suffix_continuation(
        &mut self,
        rest: Rest<'_>,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_type(flow.answer_type);
        let block = make_block(self.ctx, location, &[evidence_type, frame_type]);
        let mut suffix_mapping = mapping.clone();
        let suffix_flow = Flow {
            evidence: Some(self.ctx.block_args(block)[0]),
            exit_k: Some(self.ctx.block_args(block)[1]),
            root_exit_k: Some(self.ctx.block_args(block)[1]),
            void_exit_k: None,
            ..flow.clone()
        };
        self.convert_sequence(
            OpList::from(rest.ops),
            rest.start,
            block,
            &mut suffix_mapping,
            &suffix_flow,
        )?;
        let region = single_block_region(self.ctx, location, block);
        let never = self.never_type();
        let function = func::func_sig(self.ctx, [evidence_type, frame_type], [never]).as_type_ref();
        let closure_type = self.generated_continuation_type(function);
        let lambda = closure_over(
            self.ctx,
            location,
            region,
            closure_type,
            CallingConvention::Cps,
        );
        Ok(lambda.result(self.ctx))
    }

    pub(super) fn build_suffix_continuation(
        &mut self,
        rest: Rest<'_>,
        source_result: ValueRef,
        result_type: TypeRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let result_type = self.convert_type(result_type);
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_type(flow.answer_type);
        let block = make_block(
            self.ctx,
            location,
            &[evidence_type, frame_type, result_type],
        );
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_result, self.ctx.block_args(block)[2]);
        let suffix_flow = Flow {
            evidence: Some(self.ctx.block_args(block)[0]),
            exit_k: Some(self.ctx.block_args(block)[1]),
            root_exit_k: Some(self.ctx.block_args(block)[1]),
            ..flow.clone()
        };
        self.convert_sequence(
            OpList::from(rest.ops),
            rest.start,
            block,
            &mut body_mapping,
            &suffix_flow,
        )?;
        let region = single_block_region(self.ctx, location, block);
        let closure_ty = self.completion_type(result_type, flow.answer_type);
        let lambda = closure_over(
            self.ctx,
            location,
            region,
            closure_ty,
            CallingConvention::Cps,
        );
        Ok(lambda.result(self.ctx))
    }

    /// Build the suffix continuation of the rest of a block after `source`
    /// and the frame it is entered through, pushing both to `block`. The
    /// suffix receives the result of `source`.
    pub(super) fn push_suffix_frame(
        &mut self,
        source: OpRef,
        rest: Rest<'_>,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        plan: Option<Attribute>,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let old_result = self.ctx.op_result(source, 0);
        let result_type = self.ctx.op_result_types(source)[0];
        let continuation =
            self.build_suffix_continuation(rest, old_result, result_type, mapping, flow, location)?;
        let continuation_op = match self.ctx.value_def(continuation) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("continuation is produced by closure.lambda"),
        };
        self.ctx.push_op(block, continuation_op);
        let converted_result = self.convert_type(result_type);
        self.frame_for_suffix(block, location, converted_result, flow, continuation, plan)
    }
}
