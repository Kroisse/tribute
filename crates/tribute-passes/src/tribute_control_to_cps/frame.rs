//! Continuation frames: the layouts and helper names a conversion allocates,
//! suffix continuations and their frames, and the exits that transfer through
//! a frame.

use super::*;

impl FrameState {
    pub(super) fn fresh_helper(&mut self, prefix: &str) -> Symbol {
        let index = self.helper_index;
        self.helper_index += 1;
        helper_symbol(prefix, index)
    }

    /// The opaque `ability.frame<R>` that callables, completions, and
    /// resumptions take in the frame position.
    pub(super) fn frame_type(&mut self, ctx: &mut IrContext, answer: TypeRef) -> TypeRef {
        self.frame_types(ctx, answer);
        ability::frame(ctx, answer).as_type_ref()
    }

    /// The layout `lower_continuation_frames` gives `ability.frame<R>`,
    /// registered on first use.
    pub(super) fn frame_types(&mut self, ctx: &mut IrContext, answer: TypeRef) -> FrameTypes {
        if let Some(frame) = self.layouts.get(&answer).copied() {
            return frame;
        }
        // Number frames in order of first use. A `TypeRef` is an interner
        // index, which changes with the types interned before this pass.
        let name_text = format!("{}{}", continuation_frame::NAME_PREFIX, self.layouts.len());
        let name = ctx.intern_str(&name_text);
        let reference = continuation_frame::ref_type(ctx, name, answer);
        let done = cps_done_type(ctx, answer);
        let evidence = ability::evidence_adt_type_ref(ctx);
        let anyref = tribute_rt::anyref(ctx).as_type_ref();
        let i32 = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let abstract_frame = ability::frame(ctx, answer).as_type_ref();
        let dispatch = tribute_core::calling_convention::cps_dispatch_type(
            ctx,
            evidence,
            abstract_frame,
            anyref,
            i32,
        );
        let layout = continuation_frame::layout_type(ctx, name, answer, done, dispatch);
        let frame = FrameTypes {
            answer,
            reference,
            layout,
            done,
            dispatch,
        };
        self.layouts.insert(answer, frame);
        self.layout_aliases.push((Symbol::new(&name_text), layout));
        frame
    }
}

impl Converter<'_> {
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
        let frame_type = self.frames.frame_type(self.ctx, value_type);
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
        let exit_k = flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::post_at(
                location,
                "structured region has no verified exit continuation",
            )
        })?;
        let frame = self.frames.frame_types(self.ctx, flow.answer_type);
        let (done, _) = unpack_frame(self.ctx, block, location, &frame, exit_k);
        emit_cps_tail_call_indirect(
            self.ctx,
            block,
            location,
            done,
            std::iter::empty::<ValueRef>(),
        )?;
        Ok(())
    }

    pub(super) fn build_void_suffix_continuation(
        &mut self,
        rest: Rest<'_>,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let frame_type = self.frames.frame_type(self.ctx, flow.answer_type);
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
        let frame_type = self.frames.frame_type(self.ctx, flow.answer_type);
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
