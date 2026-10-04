//! Conversion of `handle`: handler arms, the layers that install a handle,
//! their dispatchers and resume tokens, and `perform` and `resume`.

use super::*;

impl Converter<'_> {
    /// Build the resumption of an installed handle layer.
    ///
    /// The evidence it is resumed with is the layer's outer evidence: the
    /// completion and the arms run with it, and the handle is installed on it
    /// again under the same prompt for the resumed computation.
    pub(super) fn build_handle_rebound(
        &mut self,
        location: Location,
        layer: &HandleLayer,
        values: &LayerValues,
        resume_body: ValueRef,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let boundary_frame = self.frame_types(layer.answer_type).reference;
        let block = self.make_block(location, &[evidence_type, boundary_frame, anyref]);
        let args = self.ctx.block_args(block).to_vec();
        let (done_op, done) = self.build_done_adapter(
            layer.body_type,
            values.completion,
            args[0],
            args[1],
            location,
        )?;
        self.ctx.push_op(block, done_op);
        let dispatch = self.call_dispatch_factory(block, location, layer, values, args[1], args[0]);
        let frame = self.pack_frame(block, location, layer.body_type, done, dispatch);
        let body_block = self.make_block(location, &[evidence_type]);
        let body_evidence = self.ctx.block_args(body_block)[0];
        self.emit_cps_tail_call_indirect(
            body_block,
            location,
            resume_body,
            [body_evidence, frame, args[2]],
        )?;
        let body_region = self.single_block_region(location, body_block);
        self.push_handle_dispatch(block, location, layer, values, args[0], body_region)?;
        self.finish_rebound(location, layer.answer_type, block)
    }

    /// Call a handle's dispatcher factory for one installed layer.
    pub(super) fn call_dispatch_factory(
        &mut self,
        block: BlockRef,
        location: Location,
        layer: &HandleLayer,
        values: &LayerValues,
        exit_frame: ValueRef,
        outer_evidence: ValueRef,
    ) -> ValueRef {
        let mut args = vec![values.completion, exit_frame, values.prompt, outer_evidence];
        args.extend(values.arms.iter().copied());
        let dispatch_type = self.frame_types(layer.body_type).dispatch;
        let dispatch = func::Call::operands(args)
            .callee(layer.dispatch_factory.clone().into())
            .results([dispatch_type])
            .build(self.ctx, location);
        set_calling_convention(self.ctx, dispatch.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(block, dispatch.op_ref());
        dispatch.result(self.ctx)
    }

    /// The ability instances a handle handles, in first-arm order.
    pub(super) fn layer_ability_refs(layer: &HandleLayer) -> Vec<TypeRef> {
        let mut ability_refs = Vec::new();
        for arm in &layer.arms {
            if !ability_refs.contains(&arm.ability_ref) {
                ability_refs.push(arm.ability_ref);
            }
        }
        ability_refs
    }

    /// Build the marker dispatch closures of one installed layer: for each
    /// handled instance, the `fn` dispatcher running its arms with
    /// `outer_evidence` and the marker's general slot.
    pub(super) fn build_marker_dispatchers(
        &mut self,
        block: BlockRef,
        location: Location,
        layer: &HandleLayer,
        arm_values: &[ValueRef],
        outer_evidence: ValueRef,
    ) -> Result<Vec<ValueRef>, TributeControlToCpsError> {
        let mut dispatchers = Vec::new();
        for ability_ref in Self::layer_ability_refs(layer) {
            let ability_arms: Vec<_> = layer
                .arms
                .iter()
                .zip(arm_values)
                .filter(|(arm, _)| arm.ability_ref == ability_ref && !arm.general)
                .map(|(arm, value)| HandlerArmInfo {
                    value: *value,
                    ..arm.clone()
                })
                .collect();
            let (tr_op, tr_value) =
                self.build_tail_dispatcher(location, &ability_arms, outer_evidence)?;
            self.ctx.push_op(block, tr_op);
            dispatchers.push(tr_value);
            // General operations dispatch through the continuation frame's
            // handle-layer dispatcher, so the marker's general slot only rejects.
            let (handler_op, handler_value) = self.build_general_reject_dispatcher(location);
            self.ctx.push_op(block, handler_op);
            dispatchers.push(handler_value);
        }
        Ok(dispatchers)
    }

    /// Install a handle layer: extend `outer_evidence` for `body`, with
    /// marker dispatchers whose arms run with `outer_evidence`.
    pub(super) fn push_handle_dispatch(
        &mut self,
        block: BlockRef,
        location: Location,
        layer: &HandleLayer,
        values: &LayerValues,
        outer_evidence: ValueRef,
        body: RegionRef,
    ) -> Result<(), TributeControlToCpsError> {
        let dispatchers =
            self.build_marker_dispatchers(block, location, layer, &values.arms, outer_evidence)?;
        let dispatch =
            ability::HandleDispatch::operands(outer_evidence, values.prompt, dispatchers)
                .ability_refs(Self::layer_ability_refs(layer))
                .regions(body)
                .build(self.ctx, location);
        carry_evidence_plan(self.ctx, layer.source, dispatch.op_ref());
        self.ctx.push_op(block, dispatch.op_ref());
        Ok(())
    }

    pub(super) fn build_completion_continuation(
        &mut self,
        source_region: RegionRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let source_block = self.ctx.region(source_region).blocks[0];
        let source_arg = self.ctx.block_args(source_block)[0];
        let arg_type = self.convert_type(self.ctx.value_ty(source_arg));
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(flow.answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, arg_type]);
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_arg, self.ctx.block_args(block)[2]);
        let completion_flow = Flow {
            convention: CallingConvention::Cps,
            evidence: Some(self.ctx.block_args(block)[0]),
            exit_k: Some(self.ctx.block_args(block)[1]),
            root_exit_k: Some(self.ctx.block_args(block)[1]),
            void_exit_k: None,
            answer_type: flow.answer_type,
            preserve_scf_yield: false,
            arm: flow.arm.clone(),
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.clone(),
            0,
            block,
            &mut body_mapping,
            &completion_flow,
        )?;
        let body = self.single_block_region(location, block);
        let closure_type = self.completion_type(arg_type, flow.answer_type);
        let lambda = self.closure_over(location, body, closure_type, CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    pub(super) fn build_raw_resumption(
        &mut self,
        rest: Rest<'_>,
        source_result: ValueRef,
        input_type: TypeRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
        location: Location,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let input_type = self.convert_type(input_type);
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(flow.answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, input_type]);
        let resume_evidence = self.ctx.block_args(block)[0];
        let resume_frame = self.ctx.block_args(block)[1];
        let resume_input = self.ctx.block_args(block)[2];
        let mut body_mapping = mapping.clone();
        body_mapping.insert(source_result, resume_input);
        let mut suffix_flow = flow.clone();
        // A resumed suffix receives the dynamic frame directly. It carries
        // both the completion and handle-layer dispatch provenance, so no closure
        // graph cloning or ambient-evidence reconstruction is permitted.
        suffix_flow.evidence = Some(resume_evidence);
        suffix_flow.exit_k = Some(resume_frame);
        suffix_flow.root_exit_k = Some(resume_frame);
        self.convert_sequence(
            OpList::from(rest.ops),
            rest.start,
            block,
            &mut body_mapping,
            &suffix_flow,
        )?;
        let region = self.single_block_region(location, block);
        let closure_type = self.resumption_type(input_type, flow.answer_type);
        let lambda = self.closure_over(location, region, closure_type, CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    /// Build one resume token of a general arm: the exact-typed entry to the
    /// `rebound` resumption of its handle layer.
    pub(super) fn build_exact_handler_token(
        &mut self,
        location: Location,
        input_type: TypeRef,
        answer_type: TypeRef,
        (rebound_op, rebound): (OpRef, ValueRef),
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(answer_type).reference;
        let block = self.make_block(location, &[evidence_type, frame_type, input_type]);
        let args = self.ctx.block_args(block).to_vec();
        let anyref = self.anyref_type();
        let erased_input = core::UnrealizedConversionCast::operands(args[2])
            .results(anyref)
            .build(self.ctx, location);
        self.ctx.push_op(block, erased_input.op_ref());
        self.ctx.push_op(block, rebound_op);
        self.emit_cps_tail_call_indirect(
            block,
            location,
            rebound,
            [args[0], args[1], erased_input.result(self.ctx)],
        )?;
        let region = self.single_block_region(location, block);
        let token_type = self.resumption_type(input_type, answer_type);
        let lambda = self.closure_over(location, region, token_type, CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    /// Build an immutable factory for the dispatcher of one handle. Each
    /// installed layer of the handle calls it with the frame the handle exits
    /// to and the layer's outer evidence, so the dispatcher's arms run with
    /// the evidence and continuation of that layer.
    ///
    /// The factory's parameters are `(completion, exit_frame, prompt,
    /// outer_evidence, arms...)`.
    pub(super) fn build_local_dispatcher_factory(
        &mut self,
        location: Location,
        layer: &HandleLayer,
    ) -> Result<(), TributeControlToCpsError> {
        let completion_type = self.completion_type(layer.body_type, layer.answer_type);
        let i32_type = self.i32_type();
        let evidence_type = self.evidence_type();
        let exit_frame_type = self.frame_types(layer.answer_type).reference;
        let dispatch_type = self.frame_types(layer.body_type).dispatch;
        let mut params = vec![completion_type, exit_frame_type, i32_type, evidence_type];
        params.extend(layer.arms.iter().map(|arm| self.ctx.value_ty(arm.value)));
        let factory_type = func::func_sig(self.ctx, params.clone(), [dispatch_type]).as_type_ref();
        let factory_block = self.make_block(location, &params);
        let factory_args = self.ctx.block_args(factory_block).to_vec();
        let values = LayerValues {
            completion: factory_args[0],
            prompt: factory_args[2],
            arms: factory_args[4..].to_vec(),
        };
        let (_, parent_dispatch) =
            self.unpack_frame(factory_block, location, layer.answer_type, factory_args[1]);
        let (dispatcher_op, dispatcher) = self.build_local_dispatcher_instance(
            location,
            layer,
            &values,
            factory_args[1],
            parent_dispatch,
            factory_args[3],
        )?;
        self.ctx.push_op(factory_block, dispatcher_op);
        let ret = func::Return::operands([dispatcher]).build(self.ctx, location);
        self.ctx.push_op(factory_block, ret.op_ref());
        let region = self.single_block_region(location, factory_block);
        let factory = func::Func::operands()
            .sym_name(layer.dispatch_factory.clone())
            .r#type(factory_type)
            .regions(region)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, factory.op_ref(), CallingConvention::Direct);
        self.ctx.push_op(self.module_block, factory.op_ref());
        Ok(())
    }

    /// Build the dispatch of an operation this handle layer does not handle:
    /// the operation goes to the parent dispatcher with a resumption that
    /// installs this layer again.
    pub(super) fn build_local_foreign_dispatch(
        &mut self,
        location: Location,
        layer: &HandleLayer,
        values: &LayerValues,
        parent_dispatch: ValueRef,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let body_frame = self.frame_types(layer.body_type).reference;
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            body_frame,
            anyref,
        );
        let block = self.make_block(
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
        let args = self.ctx.block_args(block).to_vec();
        let (rebound_op, rebound) = self.build_handle_rebound(location, layer, values, args[1])?;
        self.ctx.push_op(block, rebound_op);
        self.emit_cps_tail_call_indirect(
            block,
            location,
            parent_dispatch,
            [args[0], rebound, args[2], args[3], args[4], args[5]],
        )?;
        let region = self.single_block_region(location, block);
        let dispatch_type = self.frame_types(layer.body_type).dispatch;
        let lambda = self.closure_over(location, region, dispatch_type, CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    /// Build the dispatcher of one installed handle layer. A general
    /// arm runs with the layer's outer evidence and exits to `exit_frame`.
    pub(super) fn build_local_dispatcher_instance(
        &mut self,
        location: Location,
        layer: &HandleLayer,
        values: &LayerValues,
        exit_frame: ValueRef,
        parent_dispatch: ValueRef,
        outer_evidence: ValueRef,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let body_frame = self.frame_types(layer.body_type).reference;
        let resume_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            body_frame,
            anyref,
        );
        let block = self.make_block(
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
        let args = self.ctx.block_args(block).to_vec();
        let (foreign_op, foreign_dispatch) =
            self.build_local_foreign_dispatch(location, layer, values, parent_dispatch)?;
        self.ctx.push_op(block, foreign_op);
        let switch_block = self.make_block(location, &[]);
        let general_arms = layer
            .arms
            .iter()
            .zip(&values.arms)
            .filter(|(arm, _)| arm.general);
        for (arm, &arm_value) in general_arms {
            let case_block = self.make_block(location, &[]);
            let same_prompt = arith::Cmpi::operands(args[2], values.prompt)
                .predicate("eq")
                .build(self.ctx, location);
            self.ctx.push_op(case_block, same_prompt.op_ref());

            let local_block = self.make_block(location, &[]);
            let mut call_args = vec![outer_evidence, exit_frame];
            call_args.extend(self.unpack_handler_payload(local_block, location, args[5], arm));
            if arm.has_resume_token {
                let input_type = *arm.params.last().expect("resumptive arm has a token");
                let token_input = cps_closure_function_type(self.ctx, input_type)
                    .and_then(|function| func::FuncSig::from_type_ref(self.ctx, function))
                    .and_then(|function| function.inputs(self.ctx).get(2).copied())
                    .ok_or_else(|| {
                        TributeControlToCpsError::post_at(
                            location,
                            "handler resume token lacks an exact callable input",
                        )
                    })?;
                // A resume in a lambda carries the lambda's evidence, whose
                // handlers the resumed computation keeps: the layer leaves
                // only its completion behind.
                let completion_only = SuffixLayer {
                    value_type: layer.body_type,
                    boundary: layer.answer_type,
                    dispatch_factory: layer.passthrough_factory.clone(),
                    plan: None,
                };
                let passthrough = self.build_suffix_rebound(
                    location,
                    &completion_only,
                    args[1],
                    values.completion,
                )?;
                let (token_op, token) = self.build_exact_handler_token(
                    location,
                    token_input,
                    layer.answer_type,
                    passthrough,
                )?;
                self.ctx.push_op(local_block, token_op);
                call_args.push(token);
                // A resume in the arm body continues under this handle,
                // installed again on the arm's evidence at the resume.
                let installed = self.build_handle_rebound(location, layer, values, args[1])?;
                let (token_op, token) = self.build_exact_handler_token(
                    location,
                    token_input,
                    layer.answer_type,
                    installed,
                )?;
                self.ctx.push_op(local_block, token_op);
                call_args.push(token);
            }
            self.emit_cps_tail_call_indirect(local_block, location, arm_value, call_args)?;
            let local_region = self.single_block_region(location, local_block);
            let fallback_block = self.make_block(location, &[]);
            self.emit_cps_tail_call_indirect(
                fallback_block,
                location,
                foreign_dispatch,
                [args[0], args[1], args[2], args[3], args[4], args[5]],
            )?;
            let fallback_region = self.single_block_region(location, fallback_block);
            let never = self.never_type();
            let choose = scf::If::operands(same_prompt.result(self.ctx))
                .results(never)
                .regions(local_region, fallback_region)
                .build(self.ctx, location);
            self.ctx.push_op(case_block, choose.op_ref());
            let case_region = self.single_block_region(location, case_block);
            let op_index = ability::compute_op_idx(
                ability::ability_name(self.ctx, arm.ability_ref),
                Some(self.ctx.str(arm.op_name)),
            );
            let case = scf::Case::operands()
                .value(Attribute::Int(op_index as i128))
                .regions(case_region)
                .build(self.ctx, location);
            self.ctx.push_op(switch_block, case.op_ref());
        }
        let default_block = self.make_block(location, &[]);
        self.emit_cps_tail_call_indirect(
            default_block,
            location,
            foreign_dispatch,
            [args[0], args[1], args[2], args[3], args[4], args[5]],
        )?;
        let foreign_region = self.single_block_region(location, default_block);
        let default = scf::Default::operands()
            .regions(foreign_region)
            .build(self.ctx, location);
        self.ctx.push_op(switch_block, default.op_ref());
        let switch_region = self.single_block_region(location, switch_block);
        let switch = scf::Switch::operands(args[4])
            .regions(switch_region)
            .build(self.ctx, location);
        self.ctx.push_op(block, switch.op_ref());
        let region = self.single_block_region(location, block);
        let dispatch_type = self.frame_types(layer.body_type).dispatch;
        let lambda = self.closure_over(location, region, dispatch_type, CallingConvention::Cps);
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    pub(super) fn build_one_shot_wrapper(
        &mut self,
        raw_continuation: ValueRef,
        input_type: TypeRef,
        answer_type: TypeRef,
        location: Location,
    ) -> Result<(Vec<OpRef>, ValueRef), TributeControlToCpsError> {
        let i1_type = self
            .ctx
            .intern_type(TypeDataBuilder::new("core", "i1").build());
        let state_name = self.fresh_helper("one_shot_state");
        let state_name = self.ctx.intern_symbol_text(&state_name);
        let state_type = adt::struct_type(
            self.ctx,
            state_name,
            [("consumed", i1_type)],
            AttributeMap::new(),
        )
        .as_type_ref();
        let not_consumed = arith::Const::operands()
            .value(Attribute::Int(0))
            .results(i1_type)
            .build(self.ctx, location);
        let state = adt::StructNew::operands([not_consumed.result(self.ctx)])
            .r#type(state_type)
            .results(state_type)
            .build(self.ctx, location);

        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(answer_type).reference;
        let anyref = self.anyref_type();
        // The dispatcher ABI is existential only at this boundary. Keep the
        // captured continuation exact, recover this operation's declared input,
        // then transfer in proper tail position.
        let block = self.make_block(location, &[evidence_type, frame_type, anyref]);
        let args = self.ctx.block_args(block).to_vec();
        let input = if type_is(self.ctx, input_type, "core", "nil") {
            // Nil has no physical payload: its exact resumption receives the
            // canonical unit instead of an erased runtime value.
            let unit = core::NilValue::operands().build(self.ctx, location);
            self.ctx.push_op(block, unit.op_ref());
            unit.result(self.ctx)
        } else if type_is(self.ctx, input_type, "adt", "typeref") {
            // Dynamic effect values recover nominal references only through
            // their declared type, preserving the typed ownership boundary.
            let recovered = adt::RefCast::operands(args[2])
                .r#type(input_type)
                .results(input_type)
                .build(self.ctx, location);
            self.ctx.push_op(block, recovered.op_ref());
            recovered.result(self.ctx)
        } else {
            let recovered = core::UnrealizedConversionCast::operands(args[2])
                .results(input_type)
                .build(self.ctx, location);
            self.ctx.push_op(block, recovered.op_ref());
            recovered.result(self.ctx)
        };
        let consumed = adt::StructGet::operands(state.result(self.ctx))
            .r#type(state_type)
            .field(0)
            .results(i1_type)
            .build(self.ctx, location);
        self.ctx.push_op(block, consumed.op_ref());

        let reject_block = self.make_block(location, &[]);
        let unreachable = func::Unreachable::operands().build(self.ctx, location);
        self.ctx.push_op(reject_block, unreachable.op_ref());
        let reject_region = self.single_block_region(location, reject_block);

        let enter_block = self.make_block(location, &[]);
        let consumed_true = arith::Const::operands()
            .value(Attribute::Int(1))
            .results(i1_type)
            .build(self.ctx, location);
        self.ctx.push_op(enter_block, consumed_true.op_ref());
        let mark = adt::StructSet::operands(state.result(self.ctx), consumed_true.result(self.ctx))
            .r#type(state_type)
            .field(0)
            .build(self.ctx, location);
        self.ctx.push_op(enter_block, mark.op_ref());
        self.emit_cps_tail_call_indirect(
            enter_block,
            location,
            raw_continuation,
            [args[0], args[1], input],
        )?;
        let enter_region = self.single_block_region(location, enter_block);

        let never = self.never_type();
        let guard = scf::If::operands(consumed.result(self.ctx))
            .results(never)
            .regions(reject_region, enter_region)
            .build(self.ctx, location);
        self.ctx.push_op(block, guard.op_ref());
        let region = self.single_block_region(location, block);
        let closure_type = tribute_core::calling_convention::cps_resume_type(
            self.ctx,
            evidence_type,
            frame_type,
            anyref,
        );
        let wrapper = self.closure_over(location, region, closure_type, CallingConvention::Cps);
        Ok((
            vec![not_consumed.op_ref(), state.op_ref(), wrapper.op_ref()],
            wrapper.result(self.ctx),
        ))
    }

    pub(super) fn build_reject_continuation(
        &mut self,
        answer_type: TypeRef,
        location: Location,
    ) -> (OpRef, ValueRef) {
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_types(answer_type).reference;
        let anyref = self.anyref_type();
        let block = self.make_block(location, &[evidence_type, frame_type, anyref]);
        let unreachable = func::Unreachable::operands().build(self.ctx, location);
        self.ctx.push_op(block, unreachable.op_ref());
        let region = self.single_block_region(location, block);
        let closure_type = cps_resume_type(self.ctx, evidence_type, frame_type, anyref);
        let lambda = closure::Lambda::operands(std::iter::empty::<ValueRef>())
            .results(closure_type)
            .regions(region)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        (lambda.op_ref(), lambda.result(self.ctx))
    }

    pub(super) fn lower_general_perform(
        &mut self,
        source: OpRef,
        rest: Rest<'_>,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps {
            return Err(TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                "general operation appears in a non-CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let old_result = self.ctx.op_result(source, 0);
        let input_type = self.ctx.op_result_types(source)[0];
        let continuation = if type_is(self.ctx, input_type, "core", "never") {
            let (op, value) = self.build_reject_continuation(flow.answer_type, location);
            self.ctx.push_op(block, op);
            value
        } else {
            let (raw_op, raw) =
                self.build_raw_resumption(rest, old_result, input_type, mapping, flow, location)?;
            self.ctx.push_op(block, raw_op);
            let converted_input = self.convert_type(input_type);
            let (ops, one_shot) =
                self.build_one_shot_wrapper(raw, converted_input, flow.answer_type, location)?;
            for op in ops {
                self.ctx.push_op(block, op);
            }
            one_shot
        };
        let args: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|arg| mapping.get(arg).copied().unwrap_or(*arg))
            .collect();
        let ability_ref = self
            .ctx
            .op(source)
            .attributes
            .get_type("ability_ref")
            .expect("pre-CPS validation checked perform ability");
        let op_name = self
            .ctx
            .op(source)
            .attributes
            .get_string_ref("op_name")
            .expect("pre-CPS validation checked perform operation");
        let evidence = self.current_evidence(source, flow)?;
        let frame = flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::post_op(
                source,
                location,
                "general operation has no verified Dispatch boundary",
            )
        })?;
        // The frame's dispatcher is the dispatcher of the nearest
        // handle layer as it is installed now.
        let (_, dispatch) = self.unpack_frame(block, location, flow.answer_type, frame);
        let perform = ability::Perform::operands(evidence, dispatch, continuation, args)
            .ability_ref(ability_ref)
            .op_name(op_name)
            .build(self.ctx, location);
        self.ctx.push_op(block, perform.op_ref());
        Ok(())
    }

    pub(super) fn lower_resume(
        &mut self,
        source: OpRef,
        rest: Rest<'_>,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps {
            return Err(TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                "resume appears outside a CPS handler arm",
            ));
        }
        let location = self.ctx.op(source).location;
        let token_source = self.ctx.op_operands(source)[0];
        let value_source = self.ctx.op_operands(source)[1];
        // A resume in the arm body continues under the arm's own handle; any
        // other resume passes the evidence of the lambda it is in.
        // The selection yields the evidence the resumed computation's handle
        // is installed on, or its own evidence when resumed from a lambda.
        let arm = flow.arm.clone().filter(|arm| arm.token == token_source);
        let (token, evidence, plan) = match arm {
            Some(arm) => {
                let (evidence, plan) = match arm.evidence {
                    ArmEvidence::Flow => (self.current_evidence(source, flow)?, None),
                    ArmEvidence::Beneath(instances) => (
                        self.current_evidence(source, flow)?,
                        tribute_control::EvidenceStep::plan_attribute(
                            instances
                                .into_iter()
                                .map(tribute_control::EvidenceStep::Mask),
                        ),
                    ),
                    ArmEvidence::Captured(evidence) => (evidence, None),
                };
                (arm.installed, evidence, plan)
            }
            None => (
                mapping.get(&token_source).copied().unwrap_or(token_source),
                self.current_evidence(source, flow)?,
                evidence_plan_of(self.ctx, source),
            ),
        };
        let value = mapping.get(&value_source).copied().unwrap_or(value_source);
        let old_result = self.ctx.op_result(source, 0);
        let result_type = self.ctx.op_result_types(source)[0];
        let suffix =
            self.build_suffix_continuation(rest, old_result, result_type, mapping, flow, location)?;
        let suffix_op = match self.ctx.value_def(suffix) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("suffix is produced by closure.lambda"),
        };
        self.ctx.push_op(block, suffix_op);
        let resume_result = self.convert_type(result_type);
        let resume_frame =
            self.frame_for_suffix(block, location, resume_result, flow, suffix, plan.clone())?;
        let transfer = self.emit_cps_tail_call_indirect(
            block,
            location,
            token,
            [evidence, resume_frame, value],
        )?;
        set_evidence_plan(self.ctx, transfer, plan);
        Ok(())
    }

    pub(super) fn lower_handler_arm(
        &mut self,
        source: OpRef,
        outer_mapping: &HashMap<ValueRef, ValueRef>,
        handle_answer: TypeRef,
    ) -> Result<HandlerArmInfo, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let ability_ref = self
            .ctx
            .op(source)
            .attributes
            .get_type("ability_ref")
            .unwrap();
        let op_name = self
            .ctx
            .op(source)
            .attributes
            .get_string_ref("op_name")
            .unwrap();
        let general = self.ctx.op(source).attributes.get_str(self.ctx, "kind") == Some("op");
        let operation_result = self
            .ctx
            .op(source)
            .attributes
            .get_type("operation_result_type")
            .unwrap();
        let source_region = self.ctx.op_region(source, 0).ok_or_else(|| {
            self.malformed_source(source, "tribute_control.handler requires a body region")
        })?;
        let source_block = self.ctx.region(source_region).blocks[0];
        let source_args = self.ctx.block_args(source_block).to_vec();
        let has_resume_token = source_args.last().is_some_and(|arg| {
            type_is(
                self.ctx,
                self.ctx.value_ty(*arg),
                "tribute_control",
                "resume_token",
            )
        });
        let evidence_type = self.evidence_type();
        let converted_args: Vec<_> = source_args
            .iter()
            .map(|arg| self.convert_type(self.ctx.value_ty(*arg)))
            .collect();
        let convention = if general {
            CallingConvention::Cps
        } else {
            CallingConvention::EvidenceDirect
        };
        // A general arm receives the outer evidence and exit frame of the
        // layer that dispatched to it. A resumptive arm ends with a second
        // token, which resumes under the arm's own handle; the source token
        // resumes from a lambda.
        let mut params = vec![evidence_type];
        if general {
            params.push(self.frame_types(handle_answer).reference);
        }
        let source_offset = params.len();
        params.extend_from_slice(&converted_args);
        if has_resume_token {
            params.push(*converted_args.last().expect("resumptive arm has a token"));
        }
        let block = self.make_block(location, &params);
        let block_args = self.ctx.block_args(block).to_vec();
        let mut mapping = outer_mapping.clone();
        for (old, new) in source_args
            .iter()
            .copied()
            .zip(block_args[source_offset..].iter().copied())
        {
            mapping.insert(old, new);
        }
        let evidence = block_args[0];
        let handle_exit = general.then(|| block_args[1]);
        let arm = has_resume_token.then(|| ArmResume {
            token: *source_args.last().expect("resumptive arm has a token"),
            installed: *block_args.last().expect("resumptive arm has a token"),
            evidence: ArmEvidence::Flow,
        });
        let flow = Flow {
            convention,
            evidence: Some(evidence),
            exit_k: handle_exit,
            root_exit_k: handle_exit,
            void_exit_k: None,
            answer_type: handle_answer,
            preserve_scf_yield: false,
            arm,
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.clone(),
            0,
            block,
            &mut mapping,
            &flow,
        )?;
        let region = self.single_block_region(location, block);
        let result = if convention == CallingConvention::Cps {
            self.never_type()
        } else {
            self.convert_type(operation_result)
        };
        let function = func::func_sig(self.ctx, params, [result]).as_type_ref();
        let closure_type = physical_closure_type(self.ctx, function, convention);
        let lambda = self.closure_over(location, region, closure_type, convention);
        Ok(HandlerArmInfo {
            op: lambda.op_ref(),
            value: lambda.result(self.ctx),
            ability_ref,
            op_name,
            general,
            params: converted_args,
            has_resume_token,
        })
    }

    pub(super) fn unpack_handler_payload(
        &mut self,
        block: BlockRef,
        location: Location,
        payload: ValueRef,
        arm: &HandlerArmInfo,
    ) -> Vec<ValueRef> {
        let value_params = if arm.has_resume_token {
            &arm.params[..arm.params.len() - 1]
        } else {
            arm.params.as_slice()
        };
        let anyref = self.anyref_type();
        let payload_type = ability::operation_payload_type_ref(
            self.ctx,
            arm.ability_ref,
            arm.op_name,
            value_params.iter().map(|_| anyref),
        );
        let cast = core::UnrealizedConversionCast::operands(payload)
            .results(payload_type)
            .build(self.ctx, location);
        self.ctx.push_op(block, cast.op_ref());
        value_params
            .iter()
            .copied()
            .enumerate()
            .map(|(index, ty)| {
                let get = adt::StructGet::operands(cast.result(self.ctx))
                    .r#type(payload_type)
                    .field(index as u32)
                    .results(anyref)
                    .build(self.ctx, location);
                self.ctx.push_op(block, get.op_ref());
                let recovered = core::UnrealizedConversionCast::operands(get.result(self.ctx))
                    .results(ty)
                    .build(self.ctx, location);
                self.ctx.push_op(block, recovered.op_ref());
                recovered.result(self.ctx)
            })
            .collect()
    }

    /// Build the marker's dispatcher for the `fn` arms of one ability
    /// instance. The arms run with `outer_evidence`, the evidence of the
    /// layer that installed the marker, not with the evidence of the
    /// operation that reaches them.
    pub(super) fn build_tail_dispatcher(
        &mut self,
        location: Location,
        arms: &[HandlerArmInfo],
        outer_evidence: ValueRef,
    ) -> Result<(OpRef, ValueRef), TributeControlToCpsError> {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let params = vec![evidence_type, i32_type, anyref];
        let block = self.make_block(location, &params);
        let args = self.ctx.block_args(block).to_vec();
        let (op_idx, payload) = (args[1], args[2]);

        let switch_block = self.make_block(location, &[]);
        for arm in arms {
            let case_block = self.make_block(location, &[]);
            let mut call_args = vec![outer_evidence];
            call_args.extend(self.unpack_handler_payload(case_block, location, payload, arm));
            let signature = physical_closure_function_type(
                self.ctx,
                self.ctx.value_ty(arm.value),
                CallingConvention::EvidenceDirect,
            )
            .ok_or_else(|| {
                TributeControlToCpsError::post_op(
                    arm.op,
                    location,
                    "fn handler indirect callee has no exact provenance-bearing closure contract",
                )
            })?;
            let call = func::CallIndirect::operands(arm.value, call_args)
                .signature(signature)
                .build(self.ctx, location);
            set_calling_convention(self.ctx, call.op_ref(), CallingConvention::EvidenceDirect);
            self.ctx.push_op(case_block, call.op_ref());
            let erased = core::UnrealizedConversionCast::operands(call.result(self.ctx))
                .results(anyref)
                .build(self.ctx, location);
            self.ctx.push_op(case_block, erased.op_ref());
            let ret = func::Return::operands([erased.result(self.ctx)]).build(self.ctx, location);
            self.ctx.push_op(case_block, ret.op_ref());
            let case_region = self.single_block_region(location, case_block);
            let op_index = ability::compute_op_idx(
                ability::ability_name(self.ctx, arm.ability_ref),
                Some(self.ctx.str(arm.op_name)),
            );
            let case = scf::Case::operands()
                .value(Attribute::Int(op_index as i128))
                .regions(case_region)
                .build(self.ctx, location);
            self.ctx.push_op(switch_block, case.op_ref());
        }
        let reject_block = self.make_block(location, &[]);
        let unreachable = func::Unreachable::operands().build(self.ctx, location);
        self.ctx.push_op(reject_block, unreachable.op_ref());
        let reject_region = self.single_block_region(location, reject_block);
        let default = scf::Default::operands()
            .regions(reject_region)
            .build(self.ctx, location);
        self.ctx.push_op(switch_block, default.op_ref());
        let switch_region = self.single_block_region(location, switch_block);
        let switch = scf::Switch::operands(op_idx)
            .regions(switch_region)
            .build(self.ctx, location);
        self.ctx.push_op(block, switch.op_ref());

        let region = self.single_block_region(location, block);
        let function = func::func_sig(self.ctx, params, [anyref]).as_type_ref();
        let closure_type =
            physical_closure_type(self.ctx, function, CallingConvention::EvidenceDirect);
        let lambda = self.closure_over(
            location,
            region,
            closure_type,
            CallingConvention::EvidenceDirect,
        );
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
    }

    /// Build the closure for a marker's general dispatch slot, which nothing
    /// transfers to.
    pub(super) fn build_general_reject_dispatcher(
        &mut self,
        location: Location,
    ) -> (OpRef, ValueRef) {
        let evidence_type = self.evidence_type();
        let anyref = self.anyref_type();
        let i32_type = self.i32_type();
        let params = vec![evidence_type, anyref, i32_type, anyref];
        let block = self.make_block(location, &params);
        let unreachable = func::Unreachable::operands().build(self.ctx, location);
        self.ctx.push_op(block, unreachable.op_ref());
        let region = self.single_block_region(location, block);
        let never = self.never_type();
        let function = func::func_sig(self.ctx, params, [never]).as_type_ref();
        let closure_type = physical_closure_type(self.ctx, function, CallingConvention::Cps);
        let lambda = closure::Lambda::operands(std::iter::empty::<ValueRef>())
            .results(closure_type)
            .regions(region)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, lambda.op_ref(), CallingConvention::Cps);
        (lambda.op_ref(), lambda.result(self.ctx))
    }

    pub(super) fn lower_handle(
        &mut self,
        source: OpRef,
        rest: Rest<'_>,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps {
            return Err(TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                "handle appears in a non-CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let handle_result_source = self.ctx.op_result(source, 0);
        let handle_answer_source = self.ctx.op_result_types(source)[0];
        let handle_answer = self.convert_type(handle_answer_source);
        let after_handle = self.build_suffix_continuation(
            rest,
            handle_result_source,
            handle_answer_source,
            mapping,
            flow,
            location,
        )?;
        let after_handle_op = match self.ctx.value_def(after_handle) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("handle continuation is produced by closure.lambda"),
        };
        self.ctx.push_op(block, after_handle_op);
        let handle_frame =
            self.frame_for_suffix(block, location, handle_answer, flow, after_handle, None)?;

        let Some((body_source, completion_source, handlers_region)) =
            self.ctx.op_regions(source).collect_tuple()
        else {
            unreachable!("pre-CPS validation checked handle regions");
        };
        let handlers_block = self.ctx.region(handlers_region).blocks[0];
        let mut handler_arms = Vec::new();
        let source_handlers = self.ctx.block(handlers_block).ops.clone();
        for handler in source_handlers {
            let arm = self.lower_handler_arm(handler, mapping, handle_answer)?;
            self.ctx.push_op(block, arm.op);
            handler_arms.push(arm);
        }

        let completion_input = self.convert_type(
            self.ctx.value_ty(
                self.ctx
                    .block_args(self.ctx.region(completion_source).blocks[0])[0],
            ),
        );
        let layer = HandleLayer {
            source,
            arms: handler_arms,
            body_type: completion_input,
            answer_type: handle_answer,
            dispatch_factory: self.fresh_helper("make_local_dispatch"),
            passthrough_factory: self.build_dispatch_adapter_factory(
                location,
                completion_input,
                handle_answer,
                None,
            )?,
        };
        self.build_local_dispatcher_factory(location, &layer)?;
        let arm_values: Vec<_> = layer.arms.iter().map(|arm| arm.value).collect();

        // The arms and the `do` arm run with the evidence the handle is
        // installed on, not with the body's extended evidence.
        let outer_evidence = self.current_evidence(source, flow)?;

        let evidence_type = self.evidence_type();
        let i32_type = self.i32_type();
        let prompt = effect::FreshPromptTag::operands()
            .results(i32_type)
            .build(self.ctx, location);
        self.ctx.push_op(block, prompt.op_ref());
        let body_block = self.make_block(location, &[evidence_type]);
        let extended_evidence = self.ctx.block_args(body_block)[0];
        let mut body_mapping = mapping.clone();
        let outer_flow = Flow {
            convention: CallingConvention::Cps,
            evidence: Some(outer_evidence),
            exit_k: Some(handle_frame),
            root_exit_k: Some(handle_frame),
            void_exit_k: None,
            answer_type: handle_answer,
            preserve_scf_yield: false,
            arm: flow.arm.clone(),
        };
        // Inside the body, an enclosing arm's evidence lies beneath this
        // handle's markers.
        let masks = evidence_plan_of(self.ctx, source).is_some();
        let body_arm = flow.arm.clone().map(|arm| ArmResume {
            evidence: match arm.evidence {
                ArmEvidence::Flow if !masks => {
                    ArmEvidence::Beneath(Self::layer_ability_refs(&layer))
                }
                ArmEvidence::Beneath(mut instances) if !masks => {
                    instances.extend(Self::layer_ability_refs(&layer));
                    ArmEvidence::Beneath(instances)
                }
                ArmEvidence::Captured(evidence) => ArmEvidence::Captured(evidence),
                ArmEvidence::Flow | ArmEvidence::Beneath(_) => {
                    ArmEvidence::Captured(outer_evidence)
                }
            },
            ..arm
        });
        let (completion_op, completion_k) = self.build_completion_continuation(
            completion_source,
            &body_mapping,
            &outer_flow,
            location,
        )?;
        self.ctx.push_op(body_block, completion_op);
        let (done_op, done) = self.build_done_adapter(
            completion_input,
            completion_k,
            outer_evidence,
            handle_frame,
            location,
        )?;
        self.ctx.push_op(body_block, done_op);
        let values = LayerValues {
            completion: completion_k,
            prompt: prompt.result(self.ctx),
            arms: arm_values,
        };
        let local_dispatch = self.call_dispatch_factory(
            body_block,
            location,
            &layer,
            &values,
            handle_frame,
            outer_evidence,
        );
        // The body's frame carries the layer's dispatcher, as the frame of a
        // rebuilt layer does.
        let completion_frame =
            self.pack_frame(body_block, location, completion_input, done, local_dispatch);
        let body_flow = Flow {
            arm: body_arm,
            evidence: Some(extended_evidence),
            exit_k: Some(completion_frame),
            root_exit_k: Some(completion_frame),
            answer_type: completion_input,
            ..outer_flow
        };
        let source_body_block = self.ctx.region(body_source).blocks[0];
        self.convert_sequence(
            self.ctx.block(source_body_block).ops.clone(),
            0,
            body_block,
            &mut body_mapping,
            &body_flow,
        )?;
        let body_region = self.single_block_region(location, body_block);
        self.push_handle_dispatch(
            block,
            location,
            &layer,
            &values,
            outer_evidence,
            body_region,
        )
    }
}
