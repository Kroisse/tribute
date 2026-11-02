//! Conversion of source callables: function definitions, lambdas, function
//! references, and direct calls.

use super::*;
use tribute_ir::calling_convention::{ConventionBearing, DynConventionBearing};

impl Converter<'_> {
    pub(super) fn lower_lambda(
        &mut self,
        source: OpRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let logical_ty = self.ctx.op_result_types(source)[0];
        let callable = tribute_control::FuncSig::from_type_ref(self.ctx, logical_ty).unwrap();
        let conventions = tribute_cps::body_conventions(self.ctx, source)
            .expect("pre-CPS validation checked lambda conventions");
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
        let sources: Vec<_> = self.ctx.op_regions(source).collect();
        let mut bodies = Vec::with_capacity(sources.len());
        for (&convention, body_source) in conventions.iter().zip(sources) {
            let abi = CallableAbi::new(convention, source_param_types.clone(), result);
            let params = abi.lowered_params(evidence, frame);
            let entry_source = self.ctx.region(body_source).blocks[0];
            let block = make_block(self.ctx, location, &params);
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
            bodies.push(single_block_region(self.ctx, location, block));
        }
        let captures: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|capture| mapping.get(capture).copied().unwrap_or(*capture))
            .collect();
        let physical_ty = self.convert_type(logical_ty);
        let signatures: Vec<_> = conventions
            .iter()
            .map(|&convention| self.lower_callable_signature(logical_ty, convention))
            .collect();
        let adapters = self.tables.adapters(
            self.ctx,
            self.module_block,
            location,
            physical_ty,
            &conventions,
        );
        let adapters = (!adapters.is_empty()).then(|| {
            adapters
                .into_iter()
                .map(|(convention, adapter)| {
                    (
                        convention.keyword().into(),
                        trunk_ir::SymbolPath::new([adapter]),
                    )
                })
                .collect::<Vec<_>>()
        });
        let lambda = closure::Lambda::operands(captures)
            .signatures(signatures)
            .conventions(conventions)
            .adapters(adapters)
            .results(physical_ty)
            .regions(bodies)
            .build(self.ctx, location);
        self.copy_extra_attrs(
            source,
            lambda.op_ref(),
            &[
                tribute_control::QUANTIFIED_ROWS_ATTR,
                tribute_cps::CONVENTIONS_ATTR,
            ],
        );
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
            .callee(self.written_callee(source))
            .results([result_ty])
            .build(self.ctx, location);
        call.set_calling_convention(self.ctx, target.convention);
        if target.convention.needs_evidence() {
            carry_evidence_plan(self.ctx, source, call.op_ref());
        }
        Ok(call)
    }

    /// The callee `source` names, as it is written there. A call that
    /// replaces `source` in the same module refers to it the same way.
    fn written_callee(&self, source: OpRef) -> SymbolPath {
        self.ctx
            .op_attrs(source)
            .get_symbol_ref("callee")
            .expect("pre-CPS validation checked direct callee")
            .clone()
    }

    pub(super) fn lower_func_ref(
        &mut self,
        source: OpRef,
    ) -> Result<(Vec<OpRef>, ValueRef), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let table = self
            .func_tables
            .get(&source)
            .cloned()
            .expect("pre-CPS validation resolved func_ref table in this module");
        let result_logical_ty = self.ctx.op_result_types(source)[0];
        let closure_ty = self.convert_type(result_logical_ty);
        let entries = match self.table_entries.get(&(table.op, closure_ty)) {
            Some(entries) => entries.clone(),
            None => {
                let entries = self.build_table_entries(&table, result_logical_ty, closure_ty)?;
                self.table_entries
                    .insert((table.op, closure_ty), entries.clone());
                entries
            }
        };
        let closure_new = closure::New::operands(std::iter::empty::<ValueRef>())
            .entries(entries)
            .results(closure_ty)
            .build(self.ctx, location);
        Ok((vec![closure_new.op_ref()], closure_new.result(self.ctx)))
    }

    /// The convention entries of `table` for closures of `closure_ty`: an
    /// entry function for each target, then adapters and traps for the other
    /// conventions.
    fn build_table_entries(
        &mut self,
        table: &FuncTable,
        logical_ty: TypeRef,
        closure_ty: TypeRef,
    ) -> Result<[SymbolPath; 3], TributeControlToCpsError> {
        let location = self.ctx.op(table.op).location;
        let mut bodies = Vec::new();
        for (convention, callable, target) in &table.entries {
            let info = self
                .funcs
                .get(callable)
                .cloned()
                .expect("pre-CPS validation resolved table targets");
            if *convention != info.convention {
                return Err(self.malformed_source(
                    table.op,
                    "a function table entry weaker than its target is not supported",
                ));
            }
            let entry = self.func_ref_entry(location, logical_ty, &info, target);
            bodies.push((*convention, entry));
        }
        let conventions: Vec<_> = bodies.iter().map(|&(convention, _)| convention).collect();
        let adapters = self.tables.adapters(
            self.ctx,
            self.module_block,
            location,
            closure_ty,
            &conventions,
        );
        Ok(self.tables.table(
            self.ctx,
            self.module_block,
            location,
            closure_ty,
            bodies
                .into_iter()
                .chain(adapters)
                .map(|(convention, entry)| (convention, SymbolPath::new([entry]))),
        ))
    }

    /// The entry function of a closure of `logical_ty` that enters `target`,
    /// the callable `info` describes, through its own convention.
    fn func_ref_entry(
        &mut self,
        location: Location,
        logical_ty: TypeRef,
        info: &CallableInfo,
        target: &SymbolPath,
    ) -> Symbol {
        let convention = info.convention;
        // The entry is the closure's entry with the environment interposed,
        // so both share parameter attributes and metadata.
        let closure_ty = self.convert_type(logical_ty);
        let function = self.lower_callable_signature(logical_ty, convention);
        let callable = func::FuncSig::from_type_ref(self.ctx, function)
            .expect("an entry signature is a func.func_sig");
        // The entry's environment is the capture-free closure itself, right
        // after the optional evidence.
        let evidence = usize::from(convention.needs_evidence());
        let entry = callable.rebuild(self.ctx, |inputs, _| {
            inputs.insert(evidence, (closure_ty, AttributeMap::new()));
        });
        let physical_params = entry.inputs(self.ctx).to_vec();
        let entry_ty = entry.as_type_ref();
        let block = make_block(self.ctx, location, &physical_params);
        let args = self.ctx.block_args(block).to_vec();
        let target_args: Vec<_> = args[..evidence]
            .iter()
            .chain(&args[evidence + 1..])
            .copied()
            .collect();
        let target_result = self.convert_type(info.source_result);
        if convention == CallingConvention::Cps
            || (!info.platform && entry.results(self.ctx) == [target_result])
        {
            // The entry transfers its frame to the target, so a source
            // `become` through the callable value stays a proper tail call.
            let tail = func::TailCall::operands(target_args)
                .callee(target.clone())
                .build(self.ctx, location);
            tail.set_calling_convention(self.ctx, convention);
            self.ctx.push_op(block, tail.op_ref());
        } else {
            let call = func::Call::operands(target_args)
                .callee(target.clone())
                .results([target_result])
                .build(self.ctx, location);
            call.set_calling_convention(self.ctx, convention);
            self.ctx.push_op(block, call.op_ref());
            let ret = func::Return::operands([call.result(self.ctx)]).build(self.ctx, location);
            self.ctx.push_op(block, ret.op_ref());
        }
        let region = single_block_region(self.ctx, location, block);
        let symbol = self.fresh_helper("func_ref_adapter");
        let entry = func::Func::operands()
            .sym_name(self.ctx.intern_symbol_text(&symbol))
            .r#type(entry_ty)
            .regions(region)
            .build(self.ctx, location);
        entry.set_calling_convention(self.ctx, convention);
        self.ctx.push_op(self.module_block, entry.op_ref());
        symbol
    }

    pub(super) fn convert_func(
        &mut self,
        source: OpRef,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let symbol = self
            .ctx
            .op_attrs(source)
            .get_str("sym_name")
            .map(Symbol::new)
            .expect("pre-CPS validation checked function symbol");
        let logical_type = self
            .ctx
            .op_attrs(source)
            .get_type("type")
            .expect("pre-CPS validation checked function type");
        let info = self
            .funcs
            .get(&source)
            .cloned()
            .expect("validated function is present in the callable graph");
        let physical_type = self.lower_callable_signature(logical_type, info.convention);
        if !self.ctx.op_has_regions(source) {
            let mut builder = OperationDataBuilder::new(location, "func.func")
                .attr(
                    "sym_name",
                    Attribute::String(self.ctx.intern_symbol_text(&symbol)),
                )
                .attr("type", Attribute::Type(physical_type));
            for (key, value) in self.convert_attrs(&self.ctx.op(source).attributes.clone()) {
                if !matches!(
                    self.ctx.str(key),
                    "sym_name"
                        | "type"
                        | tribute_control::ROW_BINDERS_ATTR
                        | tribute_control::ROW_UNIONS_ATTR
                        | tribute_control::INFERRED_ROW_ATTR
                ) {
                    builder = builder.attr(key, value);
                }
            }
            let data = builder.build(self.ctx);
            let declaration = self.ctx.create_op(data);
            DynConventionBearing::set(self.ctx, declaration, info.convention);
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
        let block = make_block(self.ctx, location, &params);
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
        let region = single_block_region(self.ctx, location, block);
        let function = func::Func::operands()
            .sym_name(self.ctx.intern_symbol_text(&symbol))
            .r#type(physical_type)
            .regions(region)
            .build(self.ctx, location);
        self.copy_extra_attrs(
            source,
            function.op_ref(),
            &[
                "sym_name",
                "type",
                tribute_control::ROW_BINDERS_ATTR,
                tribute_control::ROW_UNIONS_ATTR,
                tribute_control::INFERRED_ROW_ATTR,
            ],
        );
        function.set_calling_convention(self.ctx, info.convention);
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
            .op_attrs(source)
            .get_symbol_ref("callee")
            .expect("pre-CPS validation checked direct callee")
            .clone();
        let target = self
            .current_func(source)
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
            tail.set_calling_convention(self.ctx, CallingConvention::Cps);
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
            .op_attrs(source)
            .get_symbol_ref("callee")
            .expect("pre-CPS validation checked tail callee")
            .clone();
        let target = self
            .current_func(source)
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
                tail.set_calling_convention(self.ctx, CallingConvention::Cps);
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
                    .callee(self.written_callee(source))
                    .build(self.ctx, location);
                tail.set_calling_convention(self.ctx, target.convention);
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
        let convention = tribute_cps::declared_convention(self.ctx, source)
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
            let signature = self.lower_callable_signature(logical_type, convention);
            let transfer =
                emit_cps_entry_transfer(self.ctx, block, location, callee, signature, args)?;
            carry_evidence_plan(self.ctx, source, transfer);
            return Ok(());
        }
        let mut args = Vec::new();
        if convention.needs_evidence() {
            args.push(self.current_evidence(source, flow)?);
        }
        args.extend(source_args);
        let signature = self.lower_callable_signature(logical_type, convention);
        if flow.convention == CallingConvention::Cps {
            let call = func::CallIndirect::operands(callee, args)
                .signature(signature)
                .build(self.ctx, location);
            call.set_calling_convention(self.ctx, convention);
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
        tail.set_calling_convention(self.ctx, convention);
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
        let convention = tribute_cps::declared_convention(self.ctx, source)
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
            let signature = self.lower_callable_signature(logical_type, convention);
            let transfer =
                emit_cps_entry_transfer(self.ctx, block, location, callee, signature, args)?;
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
        // The transfer carries the entry ABI of its convention; the
        // callee's closure type does not.
        let signature = self.lower_callable_signature(logical_type, convention);
        let call = func::CallIndirect::operands(callee, args)
            .signature(signature)
            .build(self.ctx, location);
        call.set_calling_convention(self.ctx, convention);
        if convention.needs_evidence() {
            carry_evidence_plan(self.ctx, source, call.op_ref());
        }
        self.ctx.push_op(block, call.op_ref());
        mapping.insert(self.ctx.op_result(source, 0), call.result(self.ctx));
        Ok(ControlFlow::Continue(()))
    }
}
