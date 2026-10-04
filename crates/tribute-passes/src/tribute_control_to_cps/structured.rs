//! Plain operation cloning and the conversion of effectful `scf` regions.

use super::*;

impl Converter<'_> {
    pub(super) fn clone_plain_op(
        &mut self,
        source: OpRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let data = self.ctx.op(source);
        if data.dialect == Symbol::new("tribute_control") {
            return Err(TributeControlToCpsError::post_op(
                source,
                data.location,
                format!(
                    "unsupported tribute_control operation {} reached plain cloning",
                    data.name
                ),
            ));
        }
        let location = data.location;
        let dialect = data.dialect.clone();
        let name = data.name.clone();
        let operands: Vec<_> = self
            .ctx
            .op_operands(source)
            .iter()
            .map(|value| mapping.get(value).copied().unwrap_or(*value))
            .collect();
        let result_types: Vec<_> = self.ctx.op_result_types(source).to_vec();
        let attrs = data.attributes.clone();
        let regions: trunk_ir::RegionList = self.ctx.op_regions(source).collect();
        let successors: trunk_ir::BlockList = self.ctx.op_successors(source).collect();
        if !successors.is_empty() {
            return Err(self.malformed_source(
                source,
                "plain operation cloning does not support block successors",
            ));
        }

        let mut builder = OperationDataBuilder::new(location, dialect.clone(), name.clone());
        for operand in operands {
            builder = builder.operand(operand);
        }
        for ty in result_types {
            let converted = self.convert_type(ty);
            builder = builder.result(converted);
        }
        for (key, value) in self.convert_attrs(&attrs) {
            builder = builder.attr(key, value);
        }
        for region in regions {
            let converted = if dialect == Symbol::new("core") && name == Symbol::new("module") {
                self.clone_module_region(region)?
            } else {
                self.clone_plain_region(region, mapping)?
            };
            builder = builder.region(converted);
        }
        let data = builder.build(self.ctx);
        let cloned = self.ctx.create_op(data);
        for (old, new) in self
            .ctx
            .op_results(source)
            .to_vec()
            .into_iter()
            .zip(self.ctx.op_results(cloned).to_vec())
        {
            mapping.insert(old, new);
        }
        Ok(cloned)
    }

    pub(super) fn clone_module_region(
        &mut self,
        source: RegionRef,
    ) -> Result<RegionRef, TributeControlToCpsError> {
        let location = self.ctx.region(source).location;
        let source_blocks = self.ctx.region(source).blocks.clone();
        let mut blocks = Vec::with_capacity(source_blocks.len());
        for source_block in source_blocks {
            let logical_arg_types = self
                .ctx
                .block_args(source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect::<Vec<_>>();
            let source_arg_types = logical_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect::<Vec<_>>();
            let block = self.make_block(self.ctx.block(source_block).location, &source_arg_types);
            let previous_module_block = self.module_block;
            self.module_block = block;
            let conversion = (|| {
                let mut mapping = HashMap::new();
                for (old, new) in self
                    .ctx
                    .block_args(source_block)
                    .iter()
                    .copied()
                    .zip(self.ctx.block_args(block).iter().copied())
                {
                    mapping.insert(old, new);
                }
                let source_ops = self.ctx.block(source_block).ops.clone();
                for source_op in source_ops {
                    let converted = if tribute_control::Func::matches(self.ctx, source_op) {
                        self.convert_func(source_op)?
                    } else if self.ctx.op(source_op).dialect == Symbol::new("tribute_control") {
                        return Err(TributeControlToCpsError::one(
                            PRE_CPS_BOUNDARY,
                            Some(source_op),
                            Some(self.ctx.op(source_op).location),
                            "only tribute_control.func may appear directly in a module block",
                        ));
                    } else {
                        self.clone_plain_op(source_op, &mut mapping)?
                    };
                    self.ctx.push_op(block, converted);
                }
                Ok(())
            })();
            self.module_block = previous_module_block;
            conversion?;
            blocks.push(block);
        }
        Ok(self.ctx.create_region(RegionData {
            location,
            blocks: blocks.into(),
            parent_op: None,
        }))
    }

    pub(super) fn clone_plain_region(
        &mut self,
        source: RegionRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<RegionRef, TributeControlToCpsError> {
        let location = self.ctx.region(source).location;
        let source_blocks = self.ctx.region(source).blocks.clone();
        let mut blocks = Vec::with_capacity(source_blocks.len());
        for source_block in source_blocks {
            let source_arg_types: Vec<_> = self
                .ctx
                .block_args(source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect();
            let arg_types: Vec<_> = source_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect();
            let block = self.make_block(self.ctx.block(source_block).location, &arg_types);
            for (old, new) in self
                .ctx
                .block_args(source_block)
                .to_vec()
                .into_iter()
                .zip(self.ctx.block_args(block).to_vec())
            {
                mapping.insert(old, new);
            }
            let source_ops = self.ctx.block(source_block).ops.clone();
            for op in source_ops {
                let cloned = self.clone_plain_op(op, mapping)?;
                self.ctx.push_op(block, cloned);
            }
            blocks.push(block);
        }
        Ok(self.ctx.create_region(RegionData {
            location,
            blocks: blocks.into(),
            parent_op: None,
        }))
    }

    pub(super) fn contains_tribute_control(&self, op: OpRef) -> bool {
        self.ctx.op_regions(op).any(|region| {
            self.ctx.region(region).blocks.iter().copied().any(|block| {
                self.ctx.block(block).ops.iter().copied().any(|child| {
                    self.ctx.op(child).dialect == Symbol::new("tribute_control")
                        || self.contains_tribute_control(child)
                })
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_structured_if(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let result_types = self.ctx.op_result_types(source).to_vec();
        let is_cps = flow.convention == CallingConvention::Cps;
        if is_cps && result_types.len() > 1 {
            return Err(TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                "effectful scf.if requires zero or one result inside a CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let continuation = if is_cps {
            Some(if let [source_result_type] = result_types.as_slice() {
                self.build_suffix_continuation(
                    source_ops,
                    index + 1,
                    self.ctx.op_result(source, 0),
                    *source_result_type,
                    mapping,
                    flow,
                    location,
                )?
            } else {
                self.build_void_suffix_continuation(source_ops, index + 1, mapping, flow, location)?
            })
        } else {
            None
        };
        if let Some(continuation) = continuation {
            let continuation_op = match self.ctx.value_def(continuation) {
                trunk_ir::ValueDef::OpResult(op, _) => op,
                _ => unreachable!("structured continuation is a closure.lambda"),
            };
            self.ctx.push_op(block, continuation_op);
        }

        let mut converted_regions = Vec::new();
        let source_regions = self
            .ctx
            .op_regions(source)
            .collect::<trunk_ir::RegionList>();
        for source_region in source_regions {
            let source_blocks = self.ctx.region(source_region).blocks.clone();
            let [source_block] = source_blocks.as_slice() else {
                return Err(TributeControlToCpsError::post_op(
                    source,
                    location,
                    "effectful scf.if regions must contain one block",
                ));
            };
            let source_arg_types = self
                .ctx
                .block_args(*source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect::<Vec<_>>();
            let arg_types = source_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect::<Vec<_>>();
            let converted_block =
                self.make_block(self.ctx.block(*source_block).location, &arg_types);
            let mut branch_mapping = mapping.clone();
            for (old, new) in self
                .ctx
                .block_args(*source_block)
                .iter()
                .copied()
                .zip(self.ctx.block_args(converted_block).iter().copied())
            {
                branch_mapping.insert(old, new);
            }
            let branch_flow = match continuation {
                Some(continuation) if result_types.is_empty() => Flow {
                    void_exit_k: Some(continuation),
                    ..flow.clone()
                },
                Some(continuation) => {
                    let [source_result_type] = result_types.as_slice() else {
                        unreachable!("non-void CPS branch has one result")
                    };
                    let result_type = self.convert_type(*source_result_type);
                    let frame = self.frame_for_suffix(
                        block,
                        location,
                        result_type,
                        flow,
                        continuation,
                        None,
                    )?;
                    Flow {
                        exit_k: Some(frame),
                        root_exit_k: Some(frame),
                        answer_type: result_type,
                        ..flow.clone()
                    }
                }
                None => Flow {
                    preserve_scf_yield: true,
                    ..flow.clone()
                },
            };
            self.convert_sequence(
                self.ctx.block(*source_block).ops.clone(),
                0,
                converted_block,
                &mut branch_mapping,
                &branch_flow,
            )?;
            converted_regions.push(self.single_block_region(location, converted_block));
        }
        let [then_region, else_region] = converted_regions.as_slice() else {
            return Err(TributeControlToCpsError::post_op(
                source,
                location,
                "scf.if requires exactly two regions",
            ));
        };
        let condition = self.ctx.op_operands(source)[0];
        let condition = mapping.get(&condition).copied().unwrap_or(condition);
        if continuation.is_some() {
            let never = self.never_type();
            let lowered = scf::If::operands(condition)
                .results(never)
                .regions(*then_region, *else_region)
                .build(self.ctx, location);
            self.copy_extra_attrs(source, lowered.op_ref(), &[]);
            self.ctx.push_op(block, lowered.op_ref());
            return Ok(());
        }

        let mut builder =
            OperationDataBuilder::new(location, Symbol::new("scf"), Symbol::new("if"))
                .operand(condition);
        for result_type in result_types {
            builder = builder.result(self.convert_type(result_type));
        }
        let data = builder
            .region(*then_region)
            .region(*else_region)
            .build(self.ctx);
        let lowered = self.ctx.create_op(data);
        self.copy_extra_attrs(source, lowered, &[]);
        self.ctx.push_op(block, lowered);
        for (old, new) in self
            .ctx
            .op_results(source)
            .to_vec()
            .into_iter()
            .zip(self.ctx.op_results(lowered).to_vec())
        {
            mapping.insert(old, new);
        }
        self.convert_sequence(OpList::from(source_ops), index + 1, block, mapping, flow)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_structured_switch(
        &mut self,
        source: OpRef,
        source_ops: &[OpRef],
        index: usize,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        if flow.convention != CallingConvention::Cps || !self.ctx.op_result_types(source).is_empty()
        {
            return Err(TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                "effectful scf.switch must be resultless inside a CPS callable",
            ));
        }
        let location = self.ctx.op(source).location;
        let continuation =
            self.build_void_suffix_continuation(source_ops, index + 1, mapping, flow, location)?;
        let continuation_op = match self.ctx.value_def(continuation) {
            trunk_ir::ValueDef::OpResult(op, _) => op,
            _ => unreachable!("structured continuation is a closure.lambda"),
        };
        self.ctx.push_op(block, continuation_op);

        let Ok(source_body) = self.ctx.op_regions(source).exactly_one() else {
            return Err(TributeControlToCpsError::post_op(
                source,
                location,
                "scf.switch requires exactly one body region",
            ));
        };
        let source_body_blocks = self.ctx.region(source_body).blocks.clone();
        let [source_body_block] = source_body_blocks.as_slice() else {
            return Err(TributeControlToCpsError::post_op(
                source,
                location,
                "scf.switch body requires exactly one block",
            ));
        };
        let switch_block = self.make_block(location, &[]);
        let source_cases = self.ctx.block(*source_body_block).ops.clone();
        for case in source_cases {
            let case_data = self.ctx.op(case);
            let case_location = case_data.location;
            let is_case =
                case_data.dialect == Symbol::new("scf") && case_data.name == Symbol::new("case");
            let is_default =
                case_data.dialect == Symbol::new("scf") && case_data.name == Symbol::new("default");
            if !is_case && !is_default {
                return Err(TributeControlToCpsError::post_op(
                    case,
                    case_location,
                    "scf.switch body may contain only scf.case and scf.default",
                ));
            }
            let case_value = case_data.attributes.get("value").cloned();
            let Ok(source_region) = self.ctx.op_regions(case).exactly_one() else {
                return Err(TributeControlToCpsError::post_op(
                    case,
                    case_location,
                    "scf switch arm requires exactly one region",
                ));
            };
            let source_case_blocks = &self.ctx.region(source_region).blocks;
            let [source_case_block] = source_case_blocks.as_slice() else {
                return Err(TributeControlToCpsError::post_op(
                    case,
                    case_location,
                    "scf switch arm requires exactly one block",
                ));
            };
            let source_case_ops = self.ctx.block(*source_case_block).ops.clone();
            let converted_block = self.make_block(case_location, &[]);
            let mut case_mapping = mapping.clone();
            let case_flow = Flow {
                void_exit_k: Some(continuation),
                ..flow.clone()
            };
            self.convert_sequence(
                source_case_ops,
                0,
                converted_block,
                &mut case_mapping,
                &case_flow,
            )?;
            let converted_region = self.single_block_region(case_location, converted_block);
            let converted = if is_case {
                let Some(case_value) = case_value else {
                    return Err(self.malformed_source(case, "scf.case requires a value attribute"));
                };
                scf::Case::operands()
                    .value(case_value)
                    .regions(converted_region)
                    .build(self.ctx, case_location)
                    .op_ref()
            } else {
                scf::Default::operands()
                    .regions(converted_region)
                    .build(self.ctx, case_location)
                    .op_ref()
            };
            self.ctx.push_op(switch_block, converted);
        }
        let switch_region = self.single_block_region(location, switch_block);
        let discriminant = self.ctx.op_operands(source)[0];
        let discriminant = mapping.get(&discriminant).copied().unwrap_or(discriminant);
        let lowered = scf::Switch::operands(discriminant)
            .regions(switch_region)
            .build(self.ctx, location);
        self.copy_extra_attrs(source, lowered.op_ref(), &[]);
        self.ctx.push_op(block, lowered.op_ref());
        Ok(())
    }
}
