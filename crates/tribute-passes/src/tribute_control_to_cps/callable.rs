//! Conversion of source callables: function definitions, lambdas, function
//! references, and direct calls.

use super::*;

impl Converter<'_> {
    pub(super) fn lower_lambda(
        &mut self,
        source: OpRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let logical_ty = self.ctx.op_result_types(source)[0];
        let callable = tribute_control::FuncSig::from_type_ref(self.ctx, logical_ty).unwrap();
        let convention =
            convert_convention(tribute_control::func_sig_convention(self.ctx, logical_ty).unwrap());
        let source_result = callable.result(self.ctx);
        let source_params = callable.inputs(self.ctx).to_vec();
        let result = self.convert_type(source_result);
        let source_param_types: Vec<_> = source_params
            .iter()
            .copied()
            .map(|ty| self.convert_type(ty))
            .collect();
        let evidence = self.evidence_type();
        let frame = self.frame_type(result);
        let abi = CallableAbi::new(convention, source_param_types, result);
        let params = abi.lowered_params(evidence, frame);
        let body_source = self.ctx.op_region(source, 0).ok_or_else(|| {
            self.malformed_source(source, "tribute_control.lambda requires a body region")
        })?;
        let entry_source = self.ctx.region(body_source).blocks[0];
        let block = self.make_block(location, &params);
        let mut body_mapping = mapping.clone();
        let offset = abi.source_param_offset();
        for (old, new) in self
            .ctx
            .block_args(entry_source)
            .to_vec()
            .into_iter()
            .zip(self.ctx.block_args(block)[offset..].iter().copied())
        {
            body_mapping.insert(old, new);
        }
        let evidence_value = convention
            .needs_evidence()
            .then(|| self.ctx.block_args(block)[0]);
        let exit_k = convention
            .needs_continuation_frame()
            .then(|| self.ctx.block_args(block)[usize::from(convention.needs_evidence())]);
        let flow = Flow {
            convention,
            evidence: evidence_value,
            exit_k,
            root_exit_k: exit_k,
            void_exit_k: None,
            answer_type: result,
            preserve_scf_yield: false,
            arm: None,
            tail_join: None,
        };
        self.convert_sequence(
            self.ctx.block(entry_source).ops.clone(),
            0,
            block,
            &mut body_mapping,
            &flow,
        )?;
        let body = self.single_block_region(location, block);
        let captures: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|capture| mapping.get(capture).copied().unwrap_or(*capture))
            .collect();
        let physical_ty = self.convert_type(logical_ty);
        let lambda = closure::Lambda::operands(captures)
            .results(physical_ty)
            .regions(body)
            .build(self.ctx, location);
        self.copy_extra_attrs(source, lambda.op_ref(), &[CALLING_CONVENTION_ATTR]);
        set_calling_convention(self.ctx, lambda.op_ref(), convention);
        Ok(lambda.op_ref())
    }

    pub(super) fn lower_direct_call(
        &mut self,
        source: OpRef,
        target: &CallableInfo,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<func::Call, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let mut args = Vec::new();
        if target.convention.needs_evidence() {
            args.push(self.current_evidence(source, flow)?);
        }
        args.extend(
            self.ctx
                .op_operands(source)
                .iter()
                .map(|value| mapping.get(value).copied().unwrap_or(*value)),
        );
        let result_ty = self.convert_type(target.source_result);
        let call = func::Call::operands(args)
            .callee(target.symbol.clone())
            .results([result_ty])
            .build(self.ctx, location);
        set_calling_convention(self.ctx, call.op_ref(), target.convention);
        if target.convention.needs_evidence() {
            carry_evidence_plan(self.ctx, source, call.op_ref());
        }
        Ok(call)
    }

    pub(super) fn lower_func_ref(
        &mut self,
        source: OpRef,
    ) -> Result<(Vec<OpRef>, ValueRef), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let target_symbol = self
            .ctx
            .op(source)
            .attributes
            .get_symbol_ref("func_ref")
            .expect("pre-CPS validation checked func_ref target");
        let target = self
            .current_func(target_symbol)
            .expect("pre-CPS validation resolved func_ref target in this module");
        let result_logical_ty = self.ctx.op_result_types(source)[0];
        let result_callable = tribute_control::FuncSig::from_type_ref(self.ctx, result_logical_ty)
            .expect("pre-CPS validation checked func_ref result type");
        let result_convention = tribute_control::func_sig_convention(self.ctx, result_logical_ty)
            .map(convert_convention)
            .expect("pre-CPS validation checked func_ref convention");
        debug_assert!(
            !target.convention.needs_continuation_frame()
                || result_convention.needs_continuation_frame(),
            "pre-CPS validation rejects a weaker func_ref result convention"
        );
        let result = self.convert_type(result_callable.result(self.ctx));
        // The adapter is the closure's callable with the environment
        // interposed, so both share parameter attributes and metadata.
        let closure_ty = self.convert_type(result_logical_ty);
        let function = closure::Closure::from_type_ref(self.ctx, closure_ty)
            .expect("func_ref result lowers to a closure")
            .func_type(self.ctx);
        let callable = func::FuncSig::from_type_ref(self.ctx, function)
            .expect("func_ref result lowers to a func.func_sig callable");
        let env_ty = self.anyref_type();
        let adapter = callable.rebuild(self.ctx, |inputs, _| {
            inputs.insert(
                usize::from(result_convention.needs_evidence()),
                (env_ty, AttributeMap::new()),
            );
        });
        let physical_params = adapter.inputs(self.ctx).to_vec();
        let adapter_ty = adapter.as_type_ref();
        let block = self.make_block(location, &physical_params);
        let args = self.ctx.block_args(block).to_vec();
        let evidence_offset = usize::from(result_convention.needs_evidence());
        // The environment is interposed immediately after optional evidence,
        // so a present ContinuationFrame is the following slot.
        let frame_offset = evidence_offset + 1;
        let source_offset = usize::from(result_convention.needs_evidence())
            + 1
            + usize::from(result_convention.needs_continuation_frame());
        let mut target_args = Vec::new();
        if target.convention.needs_evidence() {
            target_args.push(args[0]);
        }
        if target.convention.needs_continuation_frame() {
            target_args.push(args[frame_offset]);
        }
        target_args.extend_from_slice(&args[source_offset..]);
        if target.convention == CallingConvention::Cps {
            let tail = func::TailCall::operands(target_args)
                .callee(target.symbol)
                .build(self.ctx, location);
            set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
            self.ctx.push_op(block, tail.op_ref());
        } else {
            let target_result = self.convert_type(target.source_result);
            let call = func::Call::operands(target_args)
                .callee(target.symbol)
                .results([target_result])
                .build(self.ctx, location);
            set_calling_convention(self.ctx, call.op_ref(), target.convention);
            self.ctx.push_op(block, call.op_ref());
            if result_convention == CallingConvention::Cps {
                let frame = args[frame_offset];
                let (done_k, _) = self.unpack_frame(block, location, result, frame);
                self.emit_cps_tail_call_indirect(block, location, done_k, [call.result(self.ctx)])?;
            } else {
                let ret = func::Return::operands([call.result(self.ctx)]).build(self.ctx, location);
                self.ctx.push_op(block, ret.op_ref());
            }
        }
        let region = self.single_block_region(location, block);
        let adapter_symbol = self.fresh_helper("func_ref_adapter");
        let adapter = func::Func::operands()
            .sym_name(self.ctx.intern_symbol_text(&adapter_symbol))
            .r#type(adapter_ty)
            .regions(region)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, adapter.op_ref(), result_convention);
        self.ctx.push_op(self.module_block, adapter.op_ref());

        let empty_env_ty = adt::struct_type::<String>(
            self.ctx,
            format!("{adapter_symbol}::env"),
            [],
            AttributeMap::new(),
        )
        .as_type_ref();
        let empty_env = adt::StructNew::operands(std::iter::empty::<ValueRef>())
            .r#type(empty_env_ty)
            .results(empty_env_ty)
            .build(self.ctx, location);
        let closure_new = closure::New::operands(empty_env.result(self.ctx))
            .func_ref(adapter_symbol.into())
            .results(closure_ty)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, closure_new.op_ref(), result_convention);
        Ok((
            vec![empty_env.op_ref(), closure_new.op_ref()],
            closure_new.result(self.ctx),
        ))
    }

    pub(super) fn convert_func(
        &mut self,
        source: OpRef,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let symbol = self
            .ctx
            .op(source)
            .attributes
            .get_str(self.ctx, "sym_name")
            .map(Symbol::new)
            .expect("pre-CPS validation checked function symbol");
        let logical_type = self
            .ctx
            .op(source)
            .attributes
            .get_type("type")
            .expect("pre-CPS validation checked function type");
        let qualified =
            qualified_name(self.ctx, source).expect("pre-CPS validation checked function symbol");
        let info = self
            .current_func(&qualified)
            .expect("validated function is present in the callable graph");
        let physical_type = self.physical_function_type(logical_type);
        if !self.ctx.op_has_regions(source) {
            let mut builder =
                OperationDataBuilder::new(location, Symbol::new("func"), Symbol::new("func"))
                    .attr(
                        "sym_name",
                        Attribute::String(self.ctx.intern_symbol_text(&symbol)),
                    )
                    .attr("type", Attribute::Type(physical_type));
            for (key, value) in self.convert_attrs(&self.ctx.op(source).attributes.clone()) {
                if !matches!(key.as_str(), "sym_name" | "type") {
                    builder = builder.attr(key, value);
                }
            }
            let data = builder.build(self.ctx);
            let declaration = self.ctx.create_op(data);
            set_calling_convention(self.ctx, declaration, info.convention);
            return Ok(declaration);
        }

        let source_region = self.ctx.op_region(source, 0).ok_or_else(|| {
            self.malformed_source(
                source,
                "tribute_control.func definition requires a body region",
            )
        })?;
        let source_block = self.ctx.region(source_region).blocks[0];
        let source_result = self.convert_type(info.source_result);
        let source_params: Vec<_> = info
            .source_params
            .iter()
            .copied()
            .map(|ty| self.convert_type(ty))
            .collect();
        let abi = CallableAbi::new(info.convention, source_params, source_result);
        let evidence_ty = self.evidence_type();
        let frame_ty = self.frame_type(source_result);
        let params = abi.lowered_params(evidence_ty, frame_ty);
        let block = self.make_block(location, &params);
        let mut mapping = HashMap::default();
        for (old, new) in self.ctx.block_args(source_block).to_vec().into_iter().zip(
            self.ctx.block_args(block)[abi.source_param_offset()..]
                .iter()
                .copied(),
        ) {
            mapping.insert(old, new);
        }
        let evidence = info
            .convention
            .needs_evidence()
            .then(|| self.ctx.block_args(block)[0]);
        let exit_k = info
            .convention
            .needs_continuation_frame()
            .then(|| self.ctx.block_args(block)[usize::from(info.convention.needs_evidence())]);
        let flow = Flow {
            convention: info.convention,
            evidence,
            exit_k,
            root_exit_k: exit_k,
            void_exit_k: None,
            answer_type: source_result,
            preserve_scf_yield: false,
            arm: None,
            tail_join: None,
        };
        self.convert_sequence(
            self.ctx.block(source_block).ops.clone(),
            0,
            block,
            &mut mapping,
            &flow,
        )?;
        let region = self.single_block_region(location, block);
        let function = func::Func::operands()
            .sym_name(self.ctx.intern_symbol_text(&symbol))
            .r#type(physical_type)
            .regions(region)
            .build(self.ctx, location);
        self.copy_extra_attrs(
            source,
            function.op_ref(),
            &["sym_name", "type", CALLING_CONVENTION_ATTR],
        );
        set_calling_convention(self.ctx, function.op_ref(), info.convention);
        Ok(function.op_ref())
    }

    /// Convert a direct call. A CPS call ends its block: the rest of the
    /// block becomes the suffix the callee continues with.
    pub(super) fn lower_call(
        &mut self,
        source: OpRef,
        rest: Rest<'_>,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<ControlFlow<()>, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let target_symbol = self
            .ctx
            .op(source)
            .attributes
            .get_symbol_ref("callee")
            .expect("pre-CPS validation checked direct callee")
            .clone();
        let target = self
            .current_func(&target_symbol)
            .expect("pre-CPS validation resolved direct callee in this module");
        if target.convention == CallingConvention::Cps {
            if flow.convention != CallingConvention::Cps {
                return Err(TributeControlToCpsError::post_op(
                    source,
                    location,
                    "a non-CPS callable cannot call a CPS target",
                ));
            }
            let plan = evidence_plan_of(self.ctx, source);
            let frame = self.push_suffix_frame(source, rest, block, mapping, flow, plan)?;
            let mut args = vec![self.current_evidence(source, flow)?, frame];
            args.extend(
                self.ctx
                    .op_operands(source)
                    .iter()
                    .map(|arg| mapping.get(arg).copied().unwrap_or(*arg)),
            );
            let tail = func::TailCall::operands(args)
                .callee(target_symbol)
                .build(self.ctx, location);
            set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
            carry_evidence_plan(self.ctx, source, tail.op_ref());
            self.ctx.push_op(block, tail.op_ref());
            return Ok(ControlFlow::Break(()));
        }
        let call = self.lower_direct_call(source, &target, mapping, flow)?;
        self.ctx.push_op(block, call.op_ref());
        mapping.insert(self.ctx.op_result(source, 0), call.result(self.ctx));
        Ok(ControlFlow::Continue(()))
    }

    /// The ContinuationFrame a source `become` in a CPS callable passes on:
    /// the callable's own.
    fn tail_frame(&self, source: OpRef, flow: &Flow) -> Result<ValueRef, TributeControlToCpsError> {
        flow.exit_k.ok_or_else(|| {
            TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                "a tail call in a CPS callable has no verified ContinuationFrame",
            )
        })
    }

    /// Convert a source `become` of a named callable into a proper tail
    /// transfer. It ends its block.
    pub(super) fn lower_tail_call(
        &mut self,
        source: OpRef,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let target_symbol = self
            .ctx
            .op(source)
            .attributes
            .get_symbol_ref("callee")
            .expect("pre-CPS validation checked tail callee")
            .clone();
        let target = self
            .current_func(&target_symbol)
            .expect("pre-CPS validation resolved tail callee in this module");
        let source_args = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|arg| mapping.get(arg).copied().unwrap_or(*arg))
            .collect::<Vec<_>>();
        match (flow.convention, target.convention) {
            (_, CallingConvention::Cps) => {
                if flow.convention != CallingConvention::Cps {
                    return Err(TributeControlToCpsError::post_op(
                        source,
                        location,
                        "a non-CPS callable cannot tail call a CPS target",
                    ));
                }
                // The callee continues with the caller's continuation.
                let mut args = vec![
                    self.current_evidence(source, flow)?,
                    self.tail_frame(source, flow)?,
                ];
                args.extend(source_args);
                let tail = func::TailCall::operands(args)
                    .callee(target_symbol)
                    .build(self.ctx, location);
                set_calling_convention(self.ctx, tail.op_ref(), CallingConvention::Cps);
                carry_evidence_plan(self.ctx, source, tail.op_ref());
                self.ctx.push_op(block, tail.op_ref());
            }
            (CallingConvention::Cps, _) => {
                // A value-returning callee cannot reach a CPS callable, so
                // calling it adds one bounded activation before the caller's
                // `Done` takes its result.
                let call = self.lower_direct_call(source, &target, mapping, flow)?;
                self.ctx.push_op(block, call.op_ref());
                self.emit_exit(block, location, call.result(self.ctx), flow)?;
            }
            _ => {
                let mut args = Vec::new();
                if target.convention.needs_evidence() {
                    args.push(self.current_evidence(source, flow)?);
                }
                args.extend(source_args);
                let tail = func::TailCall::operands(args)
                    .callee(target.symbol.clone())
                    .build(self.ctx, location);
                set_calling_convention(self.ctx, tail.op_ref(), target.convention);
                if target.convention.needs_evidence() {
                    carry_evidence_plan(self.ctx, source, tail.op_ref());
                }
                self.ctx.push_op(block, tail.op_ref());
            }
        }
        Ok(())
    }

    /// Convert a source `become` of a callable value into a proper tail
    /// transfer. It ends its block.
    pub(super) fn lower_tail_call_indirect(
        &mut self,
        source: OpRef,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let source_callee = self.ctx.op_operands(source)[0];
        let logical_type = self.ctx.value_ty(source_callee);
        let convention = tribute_control::func_sig_convention(self.ctx, logical_type)
            .map(convert_convention)
            .expect("pre-CPS validation checked indirect tail callee convention");
        let callee = mapping
            .get(&source_callee)
            .copied()
            .unwrap_or(source_callee);
        let source_args = self.ctx.op_operands(source)[1..]
            .iter()
            .map(|arg| mapping.get(arg).copied().unwrap_or(*arg))
            .collect::<Vec<_>>();
        if convention == CallingConvention::Cps {
            if flow.convention != CallingConvention::Cps {
                return Err(TributeControlToCpsError::post_op(
                    source,
                    location,
                    "a non-CPS callable cannot tail call a CPS callable value",
                ));
            }
            let mut args = vec![
                self.current_evidence(source, flow)?,
                self.tail_frame(source, flow)?,
            ];
            args.extend(source_args);
            let transfer = self.emit_cps_tail_call_indirect(block, location, callee, args)?;
            carry_evidence_plan(self.ctx, source, transfer);
            return Ok(());
        }
        let mut args = Vec::new();
        if convention.needs_evidence() {
            args.push(self.current_evidence(source, flow)?);
        }
        args.extend(source_args);
        let converted_callee = self.convert_type(logical_type);
        let signature = physical_closure_function_type(self.ctx, converted_callee, convention)
            .ok_or_else(|| {
                TributeControlToCpsError::post_op(
                    source,
                    location,
                    "indirect tail callee has no exact provenance-bearing closure contract",
                )
            })?;
        if flow.convention == CallingConvention::Cps {
            let call = func::CallIndirect::operands(callee, args)
                .signature(signature)
                .build(self.ctx, location);
            set_calling_convention(self.ctx, call.op_ref(), convention);
            if convention.needs_evidence() {
                carry_evidence_plan(self.ctx, source, call.op_ref());
            }
            self.ctx.push_op(block, call.op_ref());
            let result = call.result(self.ctx);
            self.emit_exit(block, location, result, flow)?;
            return Ok(());
        }
        let tail = func::TailCallIndirect::operands(callee, args)
            .signature(signature)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, tail.op_ref(), convention);
        if convention.needs_evidence() {
            carry_evidence_plan(self.ctx, source, tail.op_ref());
        }
        self.ctx.push_op(block, tail.op_ref());
        Ok(())
    }

    /// Convert an indirect call. A CPS call ends its block: the rest of the
    /// block becomes the suffix the callee continues with.
    pub(super) fn lower_call_indirect(
        &mut self,
        source: OpRef,
        rest: Rest<'_>,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<ControlFlow<()>, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let source_callee = self.ctx.op_operands(source)[0];
        let logical_type = self.ctx.value_ty(source_callee);
        let convention = tribute_control::func_sig_convention(self.ctx, logical_type)
            .map(convert_convention)
            .expect("pre-CPS validation checked indirect callee convention");
        let callee = mapping
            .get(&source_callee)
            .copied()
            .unwrap_or(source_callee);
        let source_args = self.ctx.op_operands(source)[1..].to_vec();
        if convention == CallingConvention::Cps {
            if flow.convention != CallingConvention::Cps {
                return Err(TributeControlToCpsError::post_op(
                    source,
                    location,
                    "a non-CPS callable cannot make a CPS indirect call",
                ));
            }
            let plan = evidence_plan_of(self.ctx, source);
            let frame = self.push_suffix_frame(source, rest, block, mapping, flow, plan)?;
            let mut args = vec![self.current_evidence(source, flow)?, frame];
            args.extend(
                source_args
                    .iter()
                    .map(|arg| mapping.get(arg).copied().unwrap_or(*arg)),
            );
            let transfer = self.emit_cps_tail_call_indirect(block, location, callee, args)?;
            carry_evidence_plan(self.ctx, source, transfer);
            return Ok(ControlFlow::Break(()));
        }
        let mut args = Vec::new();
        if convention.needs_evidence() {
            args.push(self.current_evidence(source, flow)?);
        }
        args.extend(
            source_args
                .iter()
                .map(|arg| mapping.get(arg).copied().unwrap_or(*arg)),
        );
        // The source-data callee keeps its exact callable contract in its
        // converted closure type. Carry that contract onto the transfer
        // instead of inferring it from the physical operands later.
        let converted_callee = self.convert_type(logical_type);
        let signature = physical_closure_function_type(self.ctx, converted_callee, convention)
            .ok_or_else(|| {
                TributeControlToCpsError::post_op(
                    source,
                    location,
                    "indirect callee has no exact provenance-bearing closure contract",
                )
            })?;
        let call = func::CallIndirect::operands(callee, args)
            .signature(signature)
            .build(self.ctx, location);
        set_calling_convention(self.ctx, call.op_ref(), convention);
        if convention.needs_evidence() {
            carry_evidence_plan(self.ctx, source, call.op_ref());
        }
        self.ctx.push_op(block, call.op_ref());
        mapping.insert(self.ctx.op_result(source, 0), call.result(self.ctx));
        Ok(ControlFlow::Continue(()))
    }
}
