//! Continuation frames: suffix continuations, the `Done` adapter and dispatch
//! adapter of a suffix layer, and the exits that transfer through a frame.

use super::*;

/// The operations that replace one abstract frame operation.
pub(crate) struct FrameExpansion {
    /// Operations to place where the frame operation was, in order.
    pub(crate) body: Vec<OpRef>,
    /// Module-level helper functions the expansion defines.
    pub(crate) helpers: Vec<OpRef>,
    /// The frame value that replaces the result of `ability.suffix_frame`.
    pub(crate) frame: Option<ValueRef>,
}

/// Expands abstract frame operations over the frame layouts of a module with
/// the builders `tribute_control_to_cps` uses for its own frames.
pub(crate) struct FrameExpander {
    frames: HashMap<TypeRef, FrameTypes>,
    helper_index: u32,
}

impl FrameExpander {
    pub(crate) fn new(frames: HashMap<TypeRef, FrameTypes>, helper_index: u32) -> Self {
        Self {
            frames,
            helper_index,
        }
    }

    pub(crate) fn suffix_frame(
        &mut self,
        ctx: &mut IrContext,
        module_block: BlockRef,
        op: ability::SuffixFrame,
    ) -> Result<FrameExpansion, TributeControlToCpsError> {
        self.expand(ctx, module_block, op.op_ref(), |converter, scratch| {
            converter.expand_suffix_frame(scratch, op).map(Some)
        })
    }

    pub(crate) fn exit(
        &mut self,
        ctx: &mut IrContext,
        module_block: BlockRef,
        op: ability::Exit,
    ) -> Result<FrameExpansion, TributeControlToCpsError> {
        self.expand(ctx, module_block, op.op_ref(), |converter, scratch| {
            converter.expand_exit(scratch, op).map(|()| None)
        })
    }

    fn expand(
        &mut self,
        ctx: &mut IrContext,
        module_block: BlockRef,
        op: OpRef,
        build: impl FnOnce(
            &mut Converter<'_>,
            BlockRef,
        ) -> Result<Option<ValueRef>, TributeControlToCpsError>,
    ) -> Result<FrameExpansion, TributeControlToCpsError> {
        let location = ctx.op(op).location;
        let known = ctx.block(module_block).ops.len();
        let mut converter = Converter::new(ctx, module_block, HashMap::default());
        converter.frames = std::mem::take(&mut self.frames);
        converter.helper_index = self.helper_index;
        let scratch = converter.make_block(location, &[]);
        let frame = build(&mut converter, scratch);
        self.frames = std::mem::take(&mut converter.frames);
        self.helper_index = converter.helper_index;
        let frame = frame?;
        let body = ctx.block(scratch).ops.to_vec();
        for built in &body {
            ctx.remove_op_from_block(scratch, *built);
        }
        let helpers = ctx.block(module_block).ops[known..].to_vec();
        for helper in &helpers {
            ctx.remove_op_from_block(module_block, *helper);
        }
        Ok(FrameExpansion {
            body,
            helpers,
            frame,
        })
    }
}

impl Converter<'_> {
    /// Emit a final CPS tail transfer from the callee's exact typed closure
    /// contract. This must not reconstruct a signature from physical operands:
    /// closure extraction interposes an environment later and preserves this
    /// contract on the resulting indirect transfer.
    pub(super) fn emit_cps_tail_call_indirect(
        &mut self,
        block: BlockRef,
        location: Location,
        callee: ValueRef,
        args: impl IntoIterator<Item = ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let args = args.into_iter().collect::<Vec<_>>();
        let closure_type = self.ctx.value_ty(callee);
        let signature = cps_closure_function_type(self.ctx, closure_type).ok_or_else(|| {
            TributeControlToCpsError::post_at(
                location,
                "CPS indirect tail callee has no exact provenance-bearing closure contract",
            )
        })?;
        let callable = func::FuncSig::from_type_ref(self.ctx, signature).ok_or_else(|| {
            TributeControlToCpsError::post_at(
                location,
                "CPS indirect tail callee contract is not func.func_sig",
            )
        })?;
        let never = self.never_type();
        if callable.results(self.ctx) != [never]
            || callable.inputs(self.ctx).len() != args.len()
            || callable
                .inputs(self.ctx)
                .iter()
                .zip(&args)
                .any(|(expected, actual)| *expected != self.ctx.value_ty(*actual))
        {
            return Err(TributeControlToCpsError::post_at(
                location,
                "CPS indirect tail operands differ from the exact closure contract",
            ));
        }
        let tail = func::TailCallIndirect::operands(callee, args)
            .signature(signature)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
        self.ctx.push_op(block, tail.op_ref());
        Ok(tail.op_ref())
    }

    pub(super) fn unpack_frame(
        &mut self,
        block: BlockRef,
        location: Location,
        answer: TypeRef,
        frame_value: ValueRef,
    ) -> (ValueRef, ValueRef) {
        let frame = self.frame_types(answer);
        let frame_value = if self.ctx.value_ty(frame_value) == frame.reference {
            frame_value
        } else {
            let cast = core::UnrealizedConversionCast::operands(frame_value)
                .results(frame.reference)
                .build(self.ctx, location);
            self.ctx.push_op(block, cast.op_ref());
            cast.result(self.ctx)
        };
        let done = adt::StructGet::operands(frame_value)
            .r#type(frame.layout)
            .field(0)
            .results(frame.done)
            .build(self.ctx, location);
        self.ctx.push_op(block, done.op_ref());
        let dispatch = adt::StructGet::operands(frame_value)
            .r#type(frame.layout)
            .field(1)
            .results(frame.dispatch)
            .build(self.ctx, location);
        self.ctx.push_op(block, dispatch.op_ref());
        (done.result(self.ctx), dispatch.result(self.ctx))
    }

    pub(super) fn pack_frame(
        &mut self,
        block: BlockRef,
        location: Location,
        answer: TypeRef,
        done: ValueRef,
        dispatch: ValueRef,
    ) -> ValueRef {
        let frame = self.frame_types(answer);
        let packed = adt::StructNew::operands([done, dispatch])
            .r#type(frame.layout)
            .results(frame.reference)
            .build(self.ctx, location);
        self.ctx.push_op(block, packed.op_ref());
        if frame.abstract_frame == frame.reference {
            return packed.result(self.ctx);
        }
        let opaque = core::UnrealizedConversionCast::operands(packed.result(self.ctx))
            .results(frame.abstract_frame)
            .build(self.ctx, location);
        self.ctx.push_op(block, opaque.op_ref());
        opaque.result(self.ctx)
    }

    /// Build the closure of `region`, capturing the values it uses from
    /// outside in their order of first use.
    pub(super) fn closure_over(
        &mut self,
        location: Location,
        region: RegionRef,
        closure_type: TypeRef,
        convention: CallingConvention,
    ) -> closure::Lambda {
        let lambda = closure::Lambda::operands(ordered_external_values(self.ctx, region))
            .results(closure_type)
            .regions(region)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, lambda.op_ref(), convention);
        lambda
    }

    pub(super) fn build_done_adapter(
        &mut self,
        value_type: TypeRef,
        completion: ValueRef,
        evidence: ValueRef,
        outer_frame: ValueRef,
        location: Location,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let done_block = self.make_block(location, &[value_type]);
        let value = self.ctx.block_args(done_block)[0];
        self.emit_cps_tail_call_indirect(
            done_block,
            location,
            completion,
            [evidence, outer_frame, value],
        )?;
        let region = self.single_block_region(location, done_block);
        let done_type = self.done_k_type(value_type);
        let done = self.closure_over(location, region, done_type, CallingConvention::Cps);
        Ok((done.op_ref(), done.result(self.ctx)))
    }

    /// Build the resumption of a call, resume, or structured suffix layer.
    ///
    /// The suffix keeps the evidence the layer is resumed with; the resumed
    /// computation receives that evidence after the selection of `plan`.
    pub(super) fn build_suffix_rebound(
        &mut self,
        location: Location,
        layer: &SuffixLayer,
        resume_body: ValueRef,
        completion: ValueRef,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let SuffixLayer {
            value_type,
            boundary,
            dispatch_factory,
            plan,
        } = layer.clone();
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let boundary_frame = self.frame_type(boundary);
        let block = self.make_block(location, &[evidence_type, boundary_frame, anyref]);
        let args = self.ctx.block_args(block).to_vec();
        let (done_op, done) =
            self.build_done_adapter(value_type, completion, args[0], args[1], location)?;
        self.ctx.push_op(block, done_op);
        let (_, outer_dispatch) = self.unpack_frame(block, location, boundary, args[1]);
        let dispatch_type = self.frame_types(value_type).dispatch;
        let dispatch = func::Call::operands([completion, outer_dispatch])
            .callee(dispatch_factory.into())
            .results([dispatch_type])
            .build(self.ctx, location);
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(block, dispatch.op_ref());
        let frame = self.pack_frame(block, location, value_type, done, dispatch.result(self.ctx));
        let transfer = self.emit_cps_tail_call_indirect(
            block,
            location,
            resume_body,
            [args[0], frame, args[2]],
        )?;
        set_evidence_plan(self.ctx, transfer, plan);
        self.finish_rebound(location, boundary, block)
    }

    /// Wrap a rebound resumption block into its `Resume` closure.
    pub(super) fn finish_rebound(
        &mut self,
        location: Location,
        boundary: TypeRef,
        block: BlockRef,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let boundary_frame = self.frame_type(boundary);
        let region = self.single_block_region(location, block);
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            boundary_frame,
            anyref,
        );
        let resume = self.closure_over(location, region, resume_type, CallingConvention::Cps);
        Ok((resume.op_ref(), resume.result(self.ctx)))
    }

    pub(super) fn build_dispatch_adapter_factory(
        &mut self,
        location: Location,
        value_type: TypeRef,
        boundary: TypeRef,
        plan: Option<Attribute>,
    ) -> Result<Symbol, TributeControlToCpsError> {
        let symbol = self.fresh_helper("make_dispatch_adapter");
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let completion_type = self.completion_type(value_type, boundary);
        let outer_dispatch_type = self.frame_types(boundary).dispatch;
        let dispatch_type = self.frame_types(value_type).dispatch;
        let factory_type = func::func_sig(
            self.ctx,
            [completion_type, outer_dispatch_type],
            [dispatch_type],
        )
        .as_type_ref();
        let factory_block = self.make_block(location, &[completion_type, outer_dispatch_type]);
        let factory_args = self.ctx.block_args(factory_block).to_vec();
        let value_frame = self.frame_type(value_type);
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            value_frame,
            anyref,
        );
        let dispatch_block = self.make_block(
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
        let dispatch_args = self.ctx.block_args(dispatch_block).to_vec();
        let layer = SuffixLayer {
            value_type,
            boundary,
            dispatch_factory: symbol.clone(),
            plan,
        };
        let (resume_op, rebound_resume) =
            self.build_suffix_rebound(location, &layer, dispatch_args[1], factory_args[0])?;
        self.ctx.push_op(dispatch_block, resume_op);
        self.emit_cps_tail_call_indirect(
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
        let dispatch_region = self.single_block_region(location, dispatch_block);
        let dispatch = self.closure_over(
            location,
            dispatch_region,
            dispatch_type,
            CallingConvention::Cps,
        );
        self.ctx.push_op(factory_block, dispatch.op_ref());
        let ret = func::Return::operands([dispatch.result(self.ctx)]).build(self.ctx, location);
        self.ctx.push_op(factory_block, ret.op_ref());
        let factory_region = self.single_block_region(location, factory_block);
        let factory = func::Func::operands()
            .sym_name(self.ctx.intern_symbol_text(&symbol))
            .r#type(factory_type)
            .regions(factory_region)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, factory.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(self.module_block, factory.op_ref());
        Ok(symbol)
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

    /// Expand `ability.suffix_frame` into the frame the suffix is entered
    /// through, pushing the operations to `block`.
    fn expand_suffix_frame(
        &mut self,
        block: BlockRef,
        frame_op: ability::SuffixFrame,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let location = self.ctx.op(frame_op.op_ref()).location;
        let evidence = frame_op.evidence(self.ctx);
        let outer = frame_op.outer(self.ctx);
        let suffix = frame_op.continuation(self.ctx);
        let plan = evidence_plan_of(self.ctx, frame_op.op_ref());
        let value_type = continuation_frame::result_type(self.ctx, frame_op.result_ty(self.ctx))
            .ok_or_else(|| {
                TributeControlToCpsError::post_at(location, "suffix frame has no frame result")
            })?;
        let answer_type = continuation_frame::result_type(self.ctx, self.ctx.value_ty(outer))
            .ok_or_else(|| {
                TributeControlToCpsError::post_at(location, "suffix frame has no outer frame")
            })?;
        let (done_op, done) =
            self.build_done_adapter(value_type, suffix, evidence, outer, location)?;
        self.ctx.push_op(block, done_op);
        let (_, outer_dispatch) = self.unpack_frame(block, location, answer_type, outer);
        let dispatch_factory =
            self.build_dispatch_adapter_factory(location, value_type, answer_type, plan)?;
        let completion_type = self.completion_type(value_type, answer_type);
        let typed_completion = core::UnrealizedConversionCast::operands(suffix)
            .results(completion_type)
            .build(self.ctx, location);
        self.ctx.push_op(block, typed_completion.op_ref());
        let dispatch_type = self.frame_types(value_type).dispatch;
        let dispatch = func::Call::operands([typed_completion.result(self.ctx), outer_dispatch])
            .callee(dispatch_factory.into())
            .results([dispatch_type])
            .build(self.ctx, location);
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(block, dispatch.op_ref());
        Ok(self.pack_frame(block, location, value_type, done, dispatch.result(self.ctx)))
    }

    /// Expand `ability.exit` into the transfer to the frame's `Done<R>`.
    fn expand_exit(
        &mut self,
        block: BlockRef,
        exit: ability::Exit,
    ) -> Result<(), TributeControlToCpsError> {
        let location = self.ctx.op(exit.op_ref()).location;
        let frame = exit.frame(self.ctx);
        let value = exit.value(self.ctx);
        let answer = continuation_frame::result_type(self.ctx, self.ctx.value_ty(frame))
            .ok_or_else(|| {
                TributeControlToCpsError::post_at(location, "exit has no frame operand")
            })?;
        let (done, _) = self.unpack_frame(block, location, answer, frame);
        self.emit_cps_tail_call_indirect(block, location, done, [value])?;
        Ok(())
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
            self.emit_cps_tail_call_indirect(block, location, void_exit, [evidence, frame])?;
            return Ok(());
        }
        let exit_k = flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::post_at(
                location,
                "structured region has no verified exit continuation",
            )
        })?;
        let (done, _) = self.unpack_frame(block, location, flow.answer_type, exit_k);
        self.emit_cps_tail_call_indirect(block, location, done, std::iter::empty::<ValueRef>())?;
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
        let frame_type = self.frame_type(flow.answer_type);
        let block = self.make_block(location, &[evidence_type, frame_type]);
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
        let region = self.single_block_region(location, block);
        let never = self.never_type();
        let function = func::func_sig(self.ctx, [evidence_type, frame_type], [never]).as_type_ref();
        let closure_type = self.generated_continuation_type(function);
        let lambda = self.closure_over(location, region, closure_type, CallingConvention::Cps);
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
        let block = self.make_block(location, &[evidence_type, frame_type, result_type]);
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
        let region = self.single_block_region(location, block);
        let closure_ty = self.completion_type(result_type, flow.answer_type);
        let lambda = self.closure_over(location, region, closure_ty, CallingConvention::Cps);
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
