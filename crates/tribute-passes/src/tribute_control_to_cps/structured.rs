//! Plain operation cloning and the conversion of effectful `scf` regions.

use super::*;

impl Converter<'_> {
    pub(super) fn clone_plain_op(
        &mut self,
        source: OpRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
    ) -> Result<OpRef, TributeControlToCpsError> {
        let data = self.ctx.op(source);
        if is_source_logical(self.ctx, source) {
            return Err(TributeControlToCpsError::post_op(
                source,
                data.location,
                format!(
                    "unsupported tribute_control operation {} reached plain cloning",
                    data.name.display(self.ctx)
                ),
            ));
        }
        let location = data.location;
        let name = data.name;
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

        let mut builder = OperationDataBuilder::new(location, name);
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
            let converted = if name == core::Module::NAME {
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
            let block = make_block(
                self.ctx,
                self.ctx.block(source_block).location,
                &source_arg_types,
            );
            let previous_module_block = self.module_block;
            self.module_block = block;
            let conversion = (|| {
                let mut mapping = HashMap::default();
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
                    let converted = if tribute_cps::Func::matches(self.ctx, source_op) {
                        self.convert_func(source_op)?
                    } else if tribute_cps::FuncTable::matches(self.ctx, source_op) {
                        // Its references build its entries when they are
                        // converted.
                        continue;
                    } else if is_source_logical(self.ctx, source_op) {
                        return Err(TributeControlToCpsError::one(
                            PRE_CPS_BOUNDARY,
                            Some(source_op),
                            Some(self.ctx.op(source_op).location),
                            "only tribute_cps.func may appear directly in a module block",
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
            let block = make_block(self.ctx, self.ctx.block(source_block).location, &arg_types);
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
                    is_source_logical(self.ctx, child) || self.contains_tribute_control(child)
                })
            })
        })
    }

    /// Whether a source `become` in `op`'s regions leaves the enclosing
    /// callable. A lambda's tail calls leave the lambda.
    fn contains_tail_call(&self, op: OpRef) -> bool {
        self.ctx.op_regions(op).any(|region| {
            self.ctx.region(region).blocks.iter().copied().any(|block| {
                self.ctx.block(block).ops.iter().copied().any(|child| {
                    tribute_control::TailCall::matches(self.ctx, child)
                        || tribute_cps::TailCallIndirect::matches(self.ctx, child)
                        || (!tribute_cps::Lambda::matches(self.ctx, child)
                            && self.contains_tail_call(child))
                })
            })
        })
    }

    /// The tail join of a structured operation that a `become` in its arms
    /// leaves. Pre-CPS validation placed such an operation where only casts
    /// and the block's `return` or `scf.yield` follow it.
    fn tail_join(
        &self,
        source: OpRef,
        rest: Rest<'_>,
        flow: &Flow,
    ) -> Result<Option<TailJoin>, TributeControlToCpsError> {
        if !self.contains_tail_call(source) {
            return Ok(None);
        }
        let following = &rest.ops[rest.start..];
        let in_tail_position = self.ctx.op_results(source).len() == 1
            && following.split_last().is_some_and(|(&terminator, casts)| {
                (tribute_control::Return::matches(self.ctx, terminator)
                    || scf::Yield::matches(self.ctx, terminator))
                    && casts
                        .iter()
                        .all(|&cast| core::UnrealizedConversionCast::matches(self.ctx, cast))
            });
        if !in_tail_position {
            return Err(self.malformed_source(
                source,
                "a structured operation containing a tail call must be in tail position",
            ));
        }
        Ok(Some(TailJoin {
            structured: source,
            ops: OpList::from(rest.ops),
            start: rest.start,
            outer: flow.tail_join.clone(),
        }))
    }

    /// Lower an `scf.if` or `scf.index_switch` whose arms each end the
    /// callable: by a tail call, or by running the tail join's continuation
    /// in place of their yield.
    fn lower_tail_structured(
        &mut self,
        source: OpRef,
        shape: StructuredArms,
        join: TailJoin,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let mut converted_regions = trunk_ir::RegionList::new();
        for source_region in shape.source_regions(self.ctx, source) {
            let source_block = self.single_arm_block(source, source_region)?;
            let block_location = self.ctx.block(source_block).location;
            let converted_block = make_block(self.ctx, block_location, &[]);
            let mut branch_mapping = mapping.clone();
            let branch_flow = Flow {
                preserve_scf_yield: false,
                tail_join: Some(Box::new(join.clone())),
                ..flow.clone()
            };
            self.convert_sequence(
                self.ctx.block(source_block).ops.clone(),
                0,
                converted_block,
                &mut branch_mapping,
                &branch_flow,
            )?;
            converted_regions.push(single_block_region(self.ctx, location, converted_block));
        }
        let never = self.never_type();
        let lowered = shape.build(self.ctx, location, mapping, [never], converted_regions);
        self.copy_extra_attrs(source, lowered, &[]);
        self.ctx.push_op(block, lowered);
        Ok(())
    }

    /// The single block of one arm of an effectful structured operation.
    fn single_arm_block(
        &self,
        source: OpRef,
        region: RegionRef,
    ) -> Result<BlockRef, TributeControlToCpsError> {
        match self.ctx.region(region).blocks.as_slice() {
            [block] => Ok(*block),
            _ => Err(TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                format!(
                    "effectful {} regions must contain one block",
                    self.ctx.op(source).name.display(self.ctx)
                ),
            )),
        }
    }

    /// Lower an `scf.if` or `scf.index_switch` whose arms contain source
    /// control. Inside a CPS callable, each arm continues with the suffix
    /// after the operation; elsewhere the operation keeps its results and
    /// its arms keep their `scf.yield`.
    pub(super) fn lower_structured(
        &mut self,
        source: OpRef,
        rest: Rest<'_>,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        let shape = StructuredArms::of(self.ctx, source)
            .ok_or_else(|| self.malformed_source(source, "expected scf.if or scf.index_switch"))?;
        if let Some(join) = self.tail_join(source, rest, flow)? {
            return self.lower_tail_structured(source, shape, join, block, mapping, flow);
        }
        let result_types = self.ctx.op_result_types(source).to_vec();
        let is_cps = flow.convention == CallingConvention::Cps;
        let location = self.ctx.op(source).location;
        if is_cps && result_types.len() > 1 {
            return Err(TributeControlToCpsError::post_op(
                source,
                location,
                format!(
                    "effectful {} requires zero or one result inside a CPS callable",
                    self.ctx.op(source).name.display(self.ctx)
                ),
            ));
        }
        let continuation = if is_cps {
            Some(if let [source_result_type] = result_types.as_slice() {
                self.build_suffix_continuation(
                    rest,
                    self.ctx.op_result(source, 0),
                    *source_result_type,
                    mapping,
                    flow,
                    location,
                )?
            } else {
                self.build_void_suffix_continuation(rest, mapping, flow, location)?
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

        let mut converted_regions = trunk_ir::RegionList::new();
        for source_region in shape.source_regions(self.ctx, source) {
            let source_block = self.single_arm_block(source, source_region)?;
            let source_arg_types = self
                .ctx
                .block_args(source_block)
                .iter()
                .map(|arg| self.ctx.value_ty(*arg))
                .collect::<Vec<_>>();
            let arg_types = source_arg_types
                .into_iter()
                .map(|ty| self.convert_type(ty))
                .collect::<Vec<_>>();
            let converted_block =
                make_block(self.ctx, self.ctx.block(source_block).location, &arg_types);
            let mut branch_mapping = mapping.clone();
            for (old, new) in self
                .ctx
                .block_args(source_block)
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
                self.ctx.block(source_block).ops.clone(),
                0,
                converted_block,
                &mut branch_mapping,
                &branch_flow,
            )?;
            converted_regions.push(single_block_region(self.ctx, location, converted_block));
        }
        if continuation.is_some() {
            let never = self.never_type();
            let lowered = shape.build(self.ctx, location, mapping, [never], converted_regions);
            self.copy_extra_attrs(source, lowered, &[]);
            self.ctx.push_op(block, lowered);
            return Ok(());
        }

        let converted_types = result_types
            .into_iter()
            .map(|ty| self.convert_type(ty))
            .collect::<Vec<_>>();
        let lowered = shape.build(
            self.ctx,
            location,
            mapping,
            converted_types,
            converted_regions,
        );
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
        self.convert_sequence(OpList::from(rest.ops), rest.start, block, mapping, flow)
    }
}

/// The structured operations whose arms CPS conversion lowers one by one.
enum StructuredArms {
    If {
        condition: ValueRef,
    },
    Switch {
        discriminant: ValueRef,
        cases: Vec<i64>,
    },
}

impl StructuredArms {
    fn of(ctx: &IrContext, op: OpRef) -> Option<Self> {
        if let Ok(if_op) = scf::If::from_op(ctx, op) {
            return Some(Self::If {
                condition: if_op.cond(ctx),
            });
        }
        let switch = scf::IndexSwitch::from_op(ctx, op).ok()?;
        Some(Self::Switch {
            discriminant: switch.discriminant(ctx),
            cases: switch.cases(ctx).collect(),
        })
    }

    /// The arms in the order `build` takes them: then and else, or the
    /// cases and then the default.
    fn source_regions(&self, ctx: &IrContext, op: OpRef) -> trunk_ir::RegionList {
        match self {
            Self::If { .. } => ctx.op_regions(op).collect(),
            Self::Switch { .. } => scf::IndexSwitch::from_op(ctx, op)
                .expect("StructuredArms::Switch is built from an scf.index_switch")
                .arm_regions(ctx),
        }
    }

    /// The same operation over the converted arms, its operand remapped.
    fn build(
        self,
        ctx: &mut IrContext,
        location: Location,
        mapping: &HashMap<ValueRef, ValueRef>,
        results: impl IntoIterator<Item = TypeRef>,
        mut regions: trunk_ir::RegionList,
    ) -> OpRef {
        let remap = |value: ValueRef| mapping.get(&value).copied().unwrap_or(value);
        match self {
            Self::If { condition } => {
                let [then_region, else_region] = regions[..] else {
                    unreachable!("scf.if has a then and an else region")
                };
                let mut builder =
                    OperationDataBuilder::new(location, scf::If::NAME).operand(remap(condition));
                for result in results {
                    builder = builder.result(result);
                }
                let data = builder.region(then_region).region(else_region).build(ctx);
                ctx.create_op(data)
            }
            Self::Switch {
                discriminant,
                cases,
            } => {
                let default_region = regions
                    .pop()
                    .expect("an scf.index_switch ends with its default region");
                scf::IndexSwitch::build_with_arms(
                    ctx,
                    location,
                    remap(discriminant),
                    results,
                    cases.into_iter().zip(regions),
                    default_region,
                )
                .op_ref()
            }
        }
    }
}
