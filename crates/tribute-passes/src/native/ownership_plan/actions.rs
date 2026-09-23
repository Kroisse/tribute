use super::facts::{
    FlowKind, NativeOwnershipFunctionFacts, borrowed_owner, is_core_ptr_type,
    is_internal_closure_layout, root_value,
};
use super::*;
pub(super) fn plan_function_actions(
    ir: &IrContext,
    facts: &NativeOwnershipFunctionFacts,
    entries: &[EntryOwnership],
    entry_contracts: &HashMap<Symbol, Vec<EntryOwnership>>,
    definitions: &HashMap<Symbol, OpRef>,
    managed_layouts: &HashSet<TypeRef>,
    elide_proven_field_borrows: bool,
) -> Result<Vec<OwnershipAction>, OwnershipPlanError> {
    ActionPlanner::new(
        ir,
        facts,
        entries,
        entry_contracts,
        definitions,
        managed_layouts,
        elide_proven_field_borrows,
    )
    .plan()
}

struct ActionPlanner<'a> {
    ir: &'a IrContext,
    facts: &'a NativeOwnershipFunctionFacts,
    entries: &'a [EntryOwnership],
    entry_contracts: &'a HashMap<Symbol, Vec<EntryOwnership>>,
    definitions: &'a HashMap<Symbol, OpRef>,
    managed_layouts: &'a HashSet<TypeRef>,
    borrowed: HashMap<ValueRef, ValueRef>,
    owned: HashSet<ValueRef>,
    liveness: Liveness,
    actions: Vec<OwnershipAction>,
}

impl<'a> ActionPlanner<'a> {
    fn new(
        ir: &'a IrContext,
        facts: &'a NativeOwnershipFunctionFacts,
        entries: &'a [EntryOwnership],
        entry_contracts: &'a HashMap<Symbol, Vec<EntryOwnership>>,
        definitions: &'a HashMap<Symbol, OpRef>,
        managed_layouts: &'a HashSet<TypeRef>,
        elide_proven_field_borrows: bool,
    ) -> Self {
        // The temporary-borrow policy selects which policy-neutral projection
        // facts participate; it never changes the facts themselves.
        let borrowed = if elide_proven_field_borrows {
            facts.projection_owners().clone()
        } else {
            HashMap::new()
        };
        let liveness = compute_liveness(facts, &borrowed);
        let mut owned = facts.managed_values().clone();
        for (&value, entry) in ir.block_args(facts.cfg().entry()).iter().zip(entries) {
            if *entry == EntryOwnership::Borrowed {
                owned.remove(&value);
            }
        }
        Self {
            ir,
            facts,
            entries,
            entry_contracts,
            definitions,
            managed_layouts,
            borrowed,
            owned,
            liveness,
            actions: Vec::new(),
        }
    }

    fn plan(mut self) -> Result<Vec<OwnershipAction>, OwnershipPlanError> {
        self.plan_entries();
        let blocks = self.facts.cfg().blocks().to_vec();
        for block in blocks {
            self.plan_block(block)?;
        }
        Ok(self.actions)
    }

    fn plan_entries(&mut self) {
        let entry_block = self.facts.cfg().entry();
        for (index, (&value, entry)) in self
            .ir
            .block_args(entry_block)
            .iter()
            .zip(self.entries)
            .enumerate()
        {
            if *entry != EntryOwnership::Retained {
                continue;
            }
            self.actions.push(OwnershipAction {
                kind: ActionKind::EntryAcquire,
                value,
                anchor: ActionAnchor::BlockStart(entry_block),
                destination: index as u32,
            });
        }
    }

    fn plan_block(&mut self, block: BlockRef) -> Result<(), OwnershipPlanError> {
        let ops = self.ir.block(block).ops.to_vec();
        let mut transferred = HashSet::new();
        for &op in &ops {
            self.plan_operation(op, &mut transferred)?;
            if let Some(&result) = self.ir.op_results(op).first()
                && self.borrowed.contains_key(&result)
            {
                self.actions.push(OwnershipAction {
                    kind: ActionKind::BorrowLoad,
                    value: result,
                    anchor: ActionAnchor::After(op),
                    destination: 0,
                });
            } else if let Some(&result) = self.ir.op_results(op).first()
                && self.facts.projection_owners().contains_key(&result)
            {
                // Preserving the temporary-borrow policy gives the projected
                // semantic value its own unit at the exact typed projection.
                // The normal final-release planner balances this acquire.
                self.actions.push(OwnershipAction {
                    kind: ActionKind::CopyAcquire,
                    value: result,
                    anchor: ActionAnchor::After(op),
                    destination: 0,
                });
            }
        }
        self.plan_final_releases(block, &ops, &transferred);
        Ok(())
    }
}

struct Liveness {
    defs: HashMap<BlockRef, HashSet<ValueRef>>,
    live_in: HashMap<BlockRef, HashSet<ValueRef>>,
    live_out: HashMap<BlockRef, HashSet<ValueRef>>,
}

fn compute_liveness(
    facts: &NativeOwnershipFunctionFacts,
    borrowed: &HashMap<ValueRef, ValueRef>,
) -> Liveness {
    let cfg = facts.cfg();
    let managed = facts.managed_values();
    let aliases = facts.aliases();
    let blocks = cfg.blocks();
    let mut uses = HashMap::new();
    let mut defs = HashMap::new();
    for &block in blocks {
        let mut block_uses = HashSet::new();
        let mut block_defs = HashSet::new();
        // Replaying the recorded policy-neutral events keeps the scan order
        // identical to the original direct walk: arguments, then each
        // operation's operands followed by its results.
        for event in facts.block_flow(block).events() {
            match event.kind {
                FlowKind::Def => {
                    if managed.contains(&event.root) {
                        block_defs.insert(event.root);
                    }
                }
                FlowKind::Use => {
                    let root = event.root;
                    if managed.contains(&root) && !block_defs.contains(&root) {
                        block_uses.insert(root);
                    }
                    if let Some(owner) = borrowed_owner(borrowed, aliases, root)
                        && managed.contains(&owner)
                        && !block_defs.contains(&owner)
                    {
                        block_uses.insert(owner);
                    }
                }
            }
        }
        uses.insert(block, block_uses);
        defs.insert(block, block_defs);
    }
    let mut live_in = blocks
        .iter()
        .map(|&b| (b, HashSet::new()))
        .collect::<HashMap<_, _>>();
    let mut live_out = live_in.clone();
    loop {
        let mut changed = false;
        for &block in blocks.iter().rev() {
            let mut out = HashSet::new();
            for successor in cfg.successors(block) {
                out.extend(live_in[successor].iter().copied());
            }
            let mut input = uses[&block].clone();
            input.extend(
                out.iter()
                    .filter(|value| !defs[&block].contains(value))
                    .copied(),
            );
            if input != live_in[&block] {
                live_in.insert(block, input);
                changed = true;
            }
            if out != live_out[&block] {
                live_out.insert(block, out);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Liveness {
        defs,
        live_in,
        live_out,
    }
}

impl ActionPlanner<'_> {
    fn plan_operation(
        &mut self,
        op: OpRef,
        transferred: &mut HashSet<ValueRef>,
    ) -> Result<(), OwnershipPlanError> {
        if tribute_ir::dialect::tribute_rt::IntoRaw::matches(self.ir, op) {
            return self.plan_into_raw(op, transferred);
        }
        if let Ok(null) = adt::RefNull::from_op(self.ir, op) {
            let [result] = self.ir.op_results(op) else {
                return Err(OwnershipPlanError::new(
                    "adt.ref_null has malformed result arity",
                ));
            };
            let ty = null.r#type(self.ir);
            let result_managed = is_managed_value(self.ir, *result, self.managed_layouts);
            let declared_managed = is_typed_managed_reference(self.ir, ty, self.managed_layouts);
            if result_managed
                && (!declared_managed
                    || !types_compatible(
                        self.ir,
                        self.ir.value_ty(*result),
                        ty,
                        self.managed_layouts,
                    ))
            {
                return Err(OwnershipPlanError::new(
                    "adt.ref_null must inhabit a compatible managed reference type",
                ));
            }
            return Ok(());
        }
        if let Ok(new) = adt::StructNew::from_op(self.ir, op) {
            let fields = get_struct_fields(self.ir, new.r#type(self.ir)).ok_or_else(|| {
                OwnershipPlanError::new(format!(
                    "struct_new {op:?} has invalid layout {} ({:?})",
                    new.r#type(self.ir),
                    self.ir.types.get(new.r#type(self.ir))
                ))
            })?;
            validate_allocation_result(self.ir, op, new.r#type(self.ir), self.managed_layouts)?;
            return self.plan_owning_operands(op, fields.iter().map(|(_, ty)| *ty));
        }
        if let Ok(new) = adt::VariantNew::from_op(self.ir, op) {
            let variants = get_enum_variants(self.ir, new.r#type(self.ir))
                .ok_or_else(|| OwnershipPlanError::new("variant_new has invalid layout"))?;
            let fields = variants
                .iter()
                .find(|(tag, _)| *tag == new.tag(self.ir))
                .map(|(_, fields)| fields.as_slice())
                .ok_or_else(|| OwnershipPlanError::new("variant_new tag is stale"))?;
            validate_allocation_result(self.ir, op, new.r#type(self.ir), self.managed_layouts)?;
            return self.plan_owning_operands(op, fields.iter().copied());
        }
        if let Ok(set) = adt::StructSet::from_op(self.ir, op) {
            let fields = get_struct_fields(self.ir, set.r#type(self.ir))
                .ok_or_else(|| OwnershipPlanError::new("struct_set has invalid layout"))?;
            let index = set.field(self.ir) as usize;
            let (_, field_ty) = fields
                .get(index)
                .ok_or_else(|| OwnershipPlanError::new("struct_set field is stale"))?;
            if !types_compatible(
                self.ir,
                self.ir.value_ty(set.r#ref(self.ir)),
                set.r#type(self.ir),
                self.managed_layouts,
            ) {
                return Err(OwnershipPlanError::new(
                    "struct_set typed contract is malformed",
                ));
            }
            if is_typed_managed_reference(self.ir, *field_ty, self.managed_layouts) {
                if !types_compatible(
                    self.ir,
                    self.ir.value_ty(set.value(self.ir)),
                    *field_ty,
                    self.managed_layouts,
                ) {
                    return Err(OwnershipPlanError::new(
                        "struct_set value type is malformed",
                    ));
                }
                self.actions.push(OwnershipAction {
                    kind: ActionKind::StoreAcquire,
                    value: set.value(self.ir),
                    anchor: ActionAnchor::Before(op),
                    destination: index as u32,
                });
                self.actions.push(OwnershipAction {
                    kind: ActionKind::ReleaseReplacedField,
                    value: set.r#ref(self.ir),
                    anchor: ActionAnchor::Before(op),
                    destination: index as u32,
                });
            }
            return Ok(());
        }
        if func::Call::matches(self.ir, op)
            || func::CallIndirect::matches(self.ir, op)
            || func::TailCall::matches(self.ir, op)
            || func::TailCallIndirect::matches(self.ir, op)
        {
            return self.plan_call(op, transferred);
        }
        if func::Return::matches(self.ir, op) {
            for (index, &operand) in self.ir.op_operands(op).iter().enumerate() {
                if is_managed_value(self.ir, operand, self.managed_layouts) {
                    let root = root_value(self.facts.aliases(), operand);
                    if self.borrowed.contains_key(&root) {
                        self.actions.push(OwnershipAction {
                            kind: ActionKind::CopyAcquire,
                            value: operand,
                            anchor: ActionAnchor::Before(op),
                            destination: index as u32,
                        });
                    }
                    transferred.insert(root);
                    self.actions.push(OwnershipAction {
                        kind: ActionKind::ReturnTransfer,
                        value: operand,
                        anchor: ActionAnchor::Before(op),
                        destination: index as u32,
                    });
                }
            }
        } else if let Some(transfers) = self.facts.cfg().branch_transfers(op) {
            let mut counts = HashMap::<ValueRef, u32>::new();
            for (index, transfer) in transfers.enumerate() {
                if is_managed_value(self.ir, transfer.destination, self.managed_layouts) {
                    let root = root_value(self.facts.aliases(), transfer.source);
                    let count = counts.entry(root).or_default();
                    if *count > 0 || self.borrowed.contains_key(&root) {
                        self.actions.push(OwnershipAction {
                            kind: ActionKind::CopyAcquire,
                            value: transfer.source,
                            anchor: ActionAnchor::Before(op),
                            destination: index as u32,
                        });
                    }
                    *count += 1;
                    transferred.insert(root);
                }
            }
        }
        Ok(())
    }

    fn plan_into_raw(
        &mut self,
        op: OpRef,
        transferred: &mut HashSet<ValueRef>,
    ) -> Result<(), OwnershipPlanError> {
        let ([input], [result]) = (self.ir.op_operands(op), self.ir.op_results(op)) else {
            return Err(OwnershipPlanError::new(
                "tribute_rt.into_raw has malformed arity",
            ));
        };
        if !is_internal_closure_layout(self.ir, self.ir.value_ty(*input), self.managed_layouts)
            || !is_core_ptr_type(self.ir, self.ir.value_ty(*result))
        {
            return Err(OwnershipPlanError::new(
                "tribute_rt.into_raw requires the exact managed closure layout and a core.ptr result",
            ));
        }
        let root = root_value(self.facts.aliases(), *input);
        if root != *input {
            return Err(OwnershipPlanError::new(
                "tribute_rt.into_raw requires an exact closure ownership value",
            ));
        }
        let (_, transfers) = exact_into_raw_transfers(
            self.ir,
            *input,
            self.ir.value_ty(*input),
            self.managed_layouts,
        )?;
        if op == transfers[0] {
            if !transferred.insert(root) {
                return Err(OwnershipPlanError::new(
                    "tribute_rt.into_raw consumes a non-final closure ownership unit",
                ));
            }
            for destination in 1..transfers.len() {
                self.actions.push(OwnershipAction {
                    kind: ActionKind::CopyAcquire,
                    value: *input,
                    anchor: ActionAnchor::Before(op),
                    destination: destination as u32,
                });
            }
        } else if !transferred.contains(&root) {
            return Err(OwnershipPlanError::new(
                "tribute_rt.into_raw consumes a non-final closure ownership unit",
            ));
        }
        self.actions.push(OwnershipAction {
            kind: ActionKind::IntoRawTransfer,
            value: *input,
            anchor: ActionAnchor::Before(op),
            destination: 0,
        });
        Ok(())
    }

    fn plan_owning_operands(
        &mut self,
        op: OpRef,
        field_types: impl IntoIterator<Item = TypeRef>,
    ) -> Result<(), OwnershipPlanError> {
        let field_types = field_types.into_iter().collect::<Vec<_>>();
        if field_types.len() != self.ir.op_operands(op).len() {
            return Err(OwnershipPlanError::new(
                "aggregate constructor arity is malformed",
            ));
        }
        for (index, (&operand, &field_ty)) in
            self.ir.op_operands(op).iter().zip(&field_types).enumerate()
        {
            if is_typed_managed_reference(self.ir, field_ty, self.managed_layouts) {
                if !types_compatible(
                    self.ir,
                    self.ir.value_ty(operand),
                    field_ty,
                    self.managed_layouts,
                ) {
                    return Err(OwnershipPlanError::new(format!(
                        "aggregate managed field type is stale at {op:?}: expected {field_ty}, got {}",
                        self.ir.value_ty(operand)
                    )));
                }
                self.actions.push(OwnershipAction {
                    kind: ActionKind::StoreAcquire,
                    value: operand,
                    anchor: ActionAnchor::Before(op),
                    destination: index as u32,
                });
            }
        }
        Ok(())
    }
}

pub(super) fn exact_into_raw_transfers(
    ctx: &IrContext,
    source: ValueRef,
    closure_layout: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<(BlockRef, Vec<OpRef>), OwnershipPlanError> {
    if ctx.value_ty(source) != closure_layout
        || !is_internal_closure_layout(ctx, closure_layout, managed_layouts)
        || is_closure_alias(ctx, source)
    {
        return Err(OwnershipPlanError::new(
            "tribute_rt.into_raw requires an exact closure ownership value",
        ));
    }
    let mut block = None;
    let mut users = HashSet::new();
    for use_ in ctx.uses(source) {
        let user = use_.user;
        if !tribute_ir::dialect::tribute_rt::IntoRaw::matches(ctx, user) {
            return Err(OwnershipPlanError::new(
                "tribute_rt.into_raw requires all direct closure uses to be exact same-block transfers",
            ));
        }
        let ([input], [result]) = (ctx.op_operands(user), ctx.op_results(user)) else {
            return Err(OwnershipPlanError::new(
                "tribute_rt.into_raw has malformed arity",
            ));
        };
        let user_block = ctx
            .op(user)
            .parent_block
            .ok_or_else(|| OwnershipPlanError::new("tribute_rt.into_raw has a stale anchor"))?;
        if *input != source
            || !is_core_ptr_type(ctx, ctx.value_ty(*result))
            || block.is_some_and(|current| user_block != current)
        {
            return Err(OwnershipPlanError::new(
                "tribute_rt.into_raw requires all direct closure uses to be exact same-block transfers",
            ));
        }
        block = Some(user_block);
        users.insert(user);
    }
    let block = block.ok_or_else(|| OwnershipPlanError::new("into_raw transfer group is empty"))?;
    let transfers = ctx
        .block(block)
        .ops
        .iter()
        .copied()
        .filter(|op| users.contains(op))
        .collect();
    Ok((block, transfers))
}

fn is_closure_alias(ctx: &IrContext, value: ValueRef) -> bool {
    let ValueDef::OpResult(op, _) = ctx.value_def(value) else {
        return false;
    };
    adt::RefCast::matches(ctx, op)
        || adt::VariantCast::matches(ctx, op)
        || core::UnrealizedConversionCast::matches(ctx, op)
}

impl ActionPlanner<'_> {
    fn plan_call(
        &mut self,
        op: OpRef,
        transferred: &mut HashSet<ValueRef>,
    ) -> Result<(), OwnershipPlanError> {
        let indirect = func::CallIndirect::matches(self.ir, op)
            || func::TailCallIndirect::matches(self.ir, op);
        let tail =
            func::TailCall::matches(self.ir, op) || func::TailCallIndirect::matches(self.ir, op);
        let operands = self.ir.op_operands(op);
        let args = if indirect {
            trunk_ir::op_interface::IndirectCallLikeOps::arguments(self.ir, op)
                .ok_or_else(|| OwnershipPlanError::new("indirect call has malformed operands"))?
        } else {
            operands
        };
        let entries = if indirect {
            let signature = trunk_ir::op_interface::IndirectCallLikeOps::exact_signature(self.ir, op)
                .and_then(|ty| func::FuncSig::from_type_ref(self.ir, ty))
                .ok_or_else(|| {
                    OwnershipPlanError::new(format!(
                        "indirect call {op:?} lacks exact signature; attrs = {:?}, operand types = {:?}",
                        self.ir.op(op).attributes,
                        self.ir
                            .op_operands(op)
                            .iter()
                            .map(|&value| self.ir.value_ty(value))
                            .collect::<Vec<_>>()
                    ))
                })?;
            if signature.inputs(self.ir).len() != args.len() {
                return Err(OwnershipPlanError::new("indirect call arity is malformed"));
            }
            validate_call_contract(self.ir, op, signature, args, self.managed_layouts)?;
            signature
                .inputs(self.ir)
                .iter()
                .map(|&ty| {
                    if is_typed_managed_reference(self.ir, ty, self.managed_layouts) {
                        if tail {
                            EntryOwnership::Consumed
                        } else {
                            EntryOwnership::Retained
                        }
                    } else {
                        EntryOwnership::Plain
                    }
                })
                .collect::<Vec<_>>()
        } else {
            let callee = self
                .ir
                .op(op)
                .attributes
                .get_symbol("callee")
                .ok_or_else(|| OwnershipPlanError::new("direct call lacks callee identity"))?;
            if !self.definitions.contains_key(&callee) {
                if args
                    .iter()
                    .any(|&value| is_managed_value(self.ir, value, self.managed_layouts))
                {
                    return Err(OwnershipPlanError::new(format!(
                        "unclassified direct call {op:?} to @{callee} carries a managed reference"
                    )));
                }
                return Ok(());
            }
            let callee_op = self.definitions[&callee];
            let signature = self
                .ir
                .op(callee_op)
                .attributes
                .get_type("type")
                .and_then(|ty| func::FuncSig::from_type_ref(self.ir, ty))
                .ok_or_else(|| OwnershipPlanError::new("direct callee lacks exact signature"))?;
            validate_call_contract(self.ir, op, signature, args, self.managed_layouts)?;
            self.entry_contracts
                .get(&callee)
                .cloned()
                .ok_or_else(|| OwnershipPlanError::new("callee has no trusted entry contract"))?
        };
        if entries.len() != args.len() {
            return Err(OwnershipPlanError::new(
                "call arity differs from entry contract",
            ));
        }
        let mut transfers = HashMap::<ValueRef, u32>::new();
        for (index, (&argument, entry)) in args.iter().zip(entries).enumerate() {
            let managed = is_managed_value(self.ir, argument, self.managed_layouts);
            if managed != (entry != EntryOwnership::Plain) {
                return Err(OwnershipPlanError::new(
                    "call type differs from ownership contract",
                ));
            }
            let kind = match (tail, entry) {
                (_, EntryOwnership::Plain) => continue,
                (false, EntryOwnership::Borrowed) => ActionKind::CallBorrow,
                (false, EntryOwnership::Retained) => ActionKind::CallRetain,
                (false, EntryOwnership::Consumed) => ActionKind::CallAcquire,
                (true, EntryOwnership::Consumed) => ActionKind::TailTransfer,
                (true, EntryOwnership::Borrowed | EntryOwnership::Retained) => {
                    return Err(OwnershipPlanError::new(
                        "proper-tail managed parameter is not consumed",
                    ));
                }
            };
            if kind == ActionKind::TailTransfer {
                let root = root_value(self.facts.aliases(), argument);
                let count = transfers.entry(root).or_default();
                if *count > 0 || self.borrowed.contains_key(&root) {
                    self.actions.push(OwnershipAction {
                        kind: ActionKind::CopyAcquire,
                        value: argument,
                        anchor: ActionAnchor::Before(op),
                        destination: index as u32,
                    });
                }
                *count += 1;
                transferred.insert(root);
            }
            self.actions.push(OwnershipAction {
                kind,
                value: argument,
                anchor: ActionAnchor::Before(op),
                destination: index as u32,
            });
        }
        Ok(())
    }
}

fn validate_allocation_result(
    ctx: &IrContext,
    op: OpRef,
    layout: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<(), OwnershipPlanError> {
    let [result] = ctx.op_results(op) else {
        return Err(OwnershipPlanError::new(
            "ADT allocation has malformed result arity",
        ));
    };
    if !types_compatible(ctx, ctx.value_ty(*result), layout, managed_layouts) {
        return Err(OwnershipPlanError::new(
            "ADT allocation result differs from its nominal managed layout",
        ));
    }
    Ok(())
}

fn validate_call_contract(
    ctx: &IrContext,
    op: OpRef,
    signature: func::FuncSig,
    args: &[ValueRef],
    managed_layouts: &HashSet<TypeRef>,
) -> Result<(), OwnershipPlanError> {
    if args.len() != signature.inputs(ctx).len()
        || args
            .iter()
            .zip(signature.inputs(ctx))
            .any(|(&argument, &expected)| {
                let actual = ctx.value_ty(argument);
                (is_typed_managed_reference(ctx, actual, managed_layouts)
                    || is_typed_managed_reference(ctx, expected, managed_layouts))
                    && !types_compatible(ctx, actual, expected, managed_layouts)
            })
    {
        return Err(OwnershipPlanError::new(
            "call arguments differ from the exact callable signature",
        ));
    }
    validate_result_contract(
        ctx,
        ctx.op_results(op),
        signature.results(ctx),
        managed_layouts,
        "call result",
    )
}

pub(super) fn validate_result_contract(
    ctx: &IrContext,
    values: &[ValueRef],
    expected: &[TypeRef],
    managed_layouts: &HashSet<TypeRef>,
    subject: &str,
) -> Result<(), OwnershipPlanError> {
    let [expected] = expected else {
        return if expected.is_empty() && values.is_empty() {
            Ok(())
        } else {
            Err(OwnershipPlanError::new(format!(
                "{subject} differs from the exact callable signature"
            )))
        };
    };
    let expected = *expected;
    let expected_data = ctx.types.get(expected);
    let physically_empty = expected_data.dialect == Symbol::new("core")
        && (expected_data.name == Symbol::new("nil") || expected_data.name == Symbol::new("never"));
    if physically_empty {
        if values.is_empty() || matches!(values, [value] if ctx.value_ty(*value) == expected) {
            return Ok(());
        }
        return Err(OwnershipPlanError::new(format!(
            "{subject} differs from the exact callable signature"
        )));
    }
    if values.len() == 1
        && !is_typed_managed_reference(ctx, expected, managed_layouts)
        && !values
            .iter()
            .any(|&value| is_managed_value(ctx, value, managed_layouts))
    {
        return Ok(());
    }
    let valid = matches!(values, [value] if types_compatible(ctx, ctx.value_ty(*value), expected, managed_layouts));
    if !valid {
        return Err(OwnershipPlanError::new(format!(
            "{subject} differs from the exact callable signature"
        )));
    }
    Ok(())
}

impl ActionPlanner<'_> {
    fn plan_final_releases(
        &mut self,
        block: BlockRef,
        ops: &[OpRef],
        transferred: &HashSet<ValueRef>,
    ) {
        let mut last_use = HashMap::new();
        for (index, &op) in ops.iter().enumerate() {
            for &operand in self.ir.op_operands(op) {
                let root = root_value(self.facts.aliases(), operand);
                if self.owned.contains(&root) {
                    last_use.insert(root, index);
                }
                if let Some(owner) = borrowed_owner(&self.borrowed, self.facts.aliases(), root)
                    && self.owned.contains(&owner)
                {
                    last_use.insert(owner, index);
                }
            }
        }
        let mut dying = HashSet::new();
        for value in &self.liveness.live_in[&block] {
            if self.owned.contains(value)
                && !self.liveness.live_out[&block].contains(value)
                && !transferred.contains(value)
                && !self.borrowed.contains_key(value)
            {
                dying.insert(*value);
            }
        }
        for value in &self.liveness.defs[&block] {
            if self.owned.contains(value)
                && !self.liveness.live_out[&block].contains(value)
                && !transferred.contains(value)
                && !self.borrowed.contains_key(value)
            {
                dying.insert(*value);
            }
        }
        let mut dying = dying.into_iter().collect::<Vec<_>>();
        dying.sort_unstable();
        for (destination, value) in dying.into_iter().enumerate() {
            let anchor = if let Some(&index) = last_use.get(&value) {
                let op = ops[index];
                if self.facts.cfg().is_terminator(op) {
                    ActionAnchor::Before(op)
                } else {
                    ActionAnchor::After(op)
                }
            } else if self.liveness.live_in[&block].contains(&value) {
                ActionAnchor::BlockStart(block)
            } else if let ValueDef::OpResult(op, _) = self.ir.value_def(value) {
                ActionAnchor::After(op)
            } else {
                ActionAnchor::BlockStart(block)
            };
            self.actions.push(OwnershipAction {
                kind: ActionKind::FinalRelease,
                value,
                anchor,
                destination: destination as u32,
            });
        }
    }
}
