//! Conversion of `handle` into `ability.handle` with its completion and
//! handler arms, and of `perform` and `resume`.

use super::*;

impl Converter<'_> {
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
        let frame_type = self.frame_type(flow.answer_type);
        let block = make_block(self.ctx, location, &[evidence_type, frame_type, arg_type]);
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
            tail_join: flow.tail_join.clone(),
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.clone(),
            0,
            block,
            &mut body_mapping,
            &completion_flow,
        )?;
        let body = single_block_region(self.ctx, location, block);
        let closure_type = self.completion_type(arg_type, flow.answer_type);
        let lambda = closure_over(
            self.ctx,
            location,
            body,
            closure_type,
            CallingConvention::Cps,
        );
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
        let frame_type = self.frame_type(flow.answer_type);
        let block = make_block(self.ctx, location, &[evidence_type, frame_type, input_type]);
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
        let region = single_block_region(self.ctx, location, block);
        let closure_type = self.resumption_type(input_type, flow.answer_type);
        let lambda = closure_over(
            self.ctx,
            location,
            region,
            closure_type,
            CallingConvention::Cps,
        );
        Ok((lambda.op_ref(), lambda.result(self.ctx)))
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
            .get_symbol_ref("ability_ref")
            .cloned()
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
        let perform = if type_is(self.ctx, input_type, "core", "never") {
            ability::Abort::operands(evidence, frame, args)
                .ability_ref(ability_ref)
                .op_name(op_name)
                .build(self.ctx, location)
                .op_ref()
        } else {
            let (raw_op, raw) =
                self.build_raw_resumption(rest, old_result, input_type, mapping, flow, location)?;
            self.ctx.push_op(block, raw_op);
            ability::Perform::operands(evidence, frame, raw, args)
                .ability_ref(ability_ref)
                .op_name(op_name)
                .build(self.ctx, location)
                .op_ref()
        };
        self.ctx.push_op(block, perform);
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
                    // Leave the nested handles innermost first.
                    ArmEvidence::Outside(instances) => (
                        self.current_evidence(source, flow)?,
                        tribute_control::EvidenceStep::plan_attribute(
                            instances
                                .into_iter()
                                .rev()
                                .map(tribute_control::EvidenceStep::Outer),
                        ),
                    ),
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
        let resume_frame =
            self.push_suffix_frame(source, rest, block, mapping, flow, plan.clone())?;
        let transfer = emit_cps_tail_call_indirect(
            self.ctx,
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
    ) -> Result<(closure::Lambda, ability::HandlerBinding), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let ability_ref = self
            .ctx
            .op(source)
            .attributes
            .get_symbol_ref("ability_ref")
            .cloned()
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
            params.push(self.frame_type(handle_answer));
        }
        let source_offset = params.len();
        params.extend_from_slice(&converted_args);
        if has_resume_token {
            params.push(*converted_args.last().expect("resumptive arm has a token"));
        }
        let block = make_block(self.ctx, location, &params);
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
            tail_join: None,
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.clone(),
            0,
            block,
            &mut mapping,
            &flow,
        )?;
        let region = single_block_region(self.ctx, location, block);
        let result = if convention == CallingConvention::Cps {
            self.never_type()
        } else {
            self.convert_type(operation_result)
        };
        let function = func::func_sig(self.ctx, params, [result]).as_type_ref();
        let closure_type = physical_closure_type(self.ctx, function, convention);
        let lambda = closure_over(self.ctx, location, region, closure_type, convention);
        let binding = ability::HandlerBinding {
            ability_ref,
            op_name,
            kind: if general {
                ability::OperationKind::Op
            } else {
                ability::OperationKind::Fn
            },
            operation_result_type: self.convert_type(operation_result),
        };
        Ok((lambda, binding))
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
        let mut arms = Vec::new();
        let mut handlers = Vec::new();
        let source_handlers = self.ctx.block(handlers_block).ops.clone();
        for handler in source_handlers {
            let (arm, binding) = self.lower_handler_arm(handler, mapping, handle_answer)?;
            self.ctx.push_op(block, arm.op_ref());
            arms.push(arm.result(self.ctx));
            handlers.push(binding);
        }

        let completion_input = self.convert_type(
            self.ctx.value_ty(
                self.ctx
                    .block_args(self.ctx.region(completion_source).blocks[0])[0],
            ),
        );
        // The arms and the `do` arm run with the evidence the handle is
        // installed on, not with the body's extended evidence.
        let outer_evidence = self.current_evidence(source, flow)?;

        let evidence_type = self.evidence_type();
        let body_frame_type = self.frame_type(completion_input);
        let body_block = make_block(self.ctx, location, &[evidence_type, body_frame_type]);
        let extended_evidence = self.ctx.block_args(body_block)[0];
        let body_frame = self.ctx.block_args(body_block)[1];
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
            tail_join: flow.tail_join.clone(),
        };
        // Inside the body, an enclosing arm's evidence is the evidence this
        // handle is installed on. A handle that handles nothing installs no
        // marker and leaves the evidence as it is.
        let handled = handlers.first().map(|binding| binding.ability_ref.clone());
        let body_arm = flow.arm.clone().map(|arm| ArmResume {
            evidence: match (arm.evidence, handled) {
                (evidence, None) => evidence,
                (ArmEvidence::Flow, Some(instance)) => ArmEvidence::Outside(vec![instance]),
                (ArmEvidence::Outside(mut instances), Some(instance)) => {
                    instances.push(instance);
                    ArmEvidence::Outside(instances)
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
        self.ctx.push_op(block, completion_op);
        let body_flow = Flow {
            arm: body_arm,
            evidence: Some(extended_evidence),
            exit_k: Some(body_frame),
            root_exit_k: Some(body_frame),
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
        let body_region = single_block_region(self.ctx, location, body_block);
        let handle = ability::Handle::operands(outer_evidence, handle_frame, completion_k, arms)
            .handlers(handlers)
            .regions(body_region)
            .build(self.ctx, location);
        carry_evidence_plan(self.ctx, source, handle.op_ref());
        self.ctx.push_op(block, handle.op_ref());
        Ok(())
    }

    /// Convert a `fn` operation: an `ability.call` whose result flows to the
    /// rest of the block, with no continuation captured.
    pub(super) fn lower_tail_perform(
        &mut self,
        source: OpRef,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        self.current_evidence(source, flow)?;
        let args: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|arg| mapping.get(arg).copied().unwrap_or(*arg))
            .collect();
        let result_type = self.convert_type(self.ctx.op_result_types(source)[0]);
        let ability_ref = self
            .ctx
            .op(source)
            .attributes
            .get_symbol_ref("ability_ref")
            .cloned()
            .expect("pre-CPS validation checked perform ability");
        let op_name = self
            .ctx
            .op(source)
            .attributes
            .get_string_ref("op_name")
            .expect("pre-CPS validation checked perform operation");
        let call = ability::Call::operands(args)
            .ability_ref(ability_ref)
            .op_name(op_name)
            .results(result_type)
            .build(self.ctx, location);
        self.ctx.push_op(block, call.op_ref());
        mapping.insert(self.ctx.op_result(source, 0), call.result(self.ctx));
        Ok(())
    }
}
