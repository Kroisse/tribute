//! Continuation frames: suffix continuations, the `Done` adapter and dispatch
//! adapter of a suffix layer, and the exits that transfer through a frame.

use super::*;

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
        packed.result(self.ctx)
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
        let boundary_frame = self.frame_types(boundary).reference;
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
        let boundary_frame = self.frame_types(boundary).reference;
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
        let value_frame = self.frame_types(value_type).reference;
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
        let (done_op, done) =
            self.build_done_adapter(value_type, suffix, evidence, outer, location)?;
        self.ctx.push_op(block, done_op);
        let (_, outer_dispatch) = self.unpack_frame(block, location, flow.answer_type, outer);
        let dispatch_factory =
            self.build_dispatch_adapter_factory(location, value_type, flow.answer_type, plan)?;
        let completion_type = self.completion_type(value_type, flow.answer_type);
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
            let (done, _) = self.unpack_frame(block, location, flow.answer_type, exit_k);
            self.emit_cps_tail_call_indirect(block, location, done, [value])?;
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
        let frame_type = self.frame_types(flow.answer_type).reference;
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
        let frame_type = self.frame_types(flow.answer_type).reference;
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
