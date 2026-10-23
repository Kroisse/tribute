use super::allocation_layout;
use super::facts::{
    NativeOwnershipFunctionFacts, NativeOwnershipModuleFacts, borrowed_owner, is_core_ptr_type,
    is_internal_closure_layout, root_value,
};
use super::*;
use trunk_ir::op_interface::DynIndirectCallLike;

pub(super) struct ActionInputs<'a> {
    pub facts: &'a NativeOwnershipFunctionFacts,
    pub liveness: &'a BlockLiveness,
}

pub(super) fn plan_function_actions(
    ir: &IrContext,
    inputs: ActionInputs<'_>,
    entries: &[EntryOwnership],
    entry_contracts: &HashMap<OpRef, Vec<EntryOwnership>>,
    module: &NativeOwnershipModuleFacts,
    managed_layouts: &HashSet<TypeRef>,
    options: NativeOwnershipPlanOptions,
) -> Result<Vec<OwnershipAction>, OwnershipPlanError> {
    ActionPlanner::new(
        ir,
        inputs,
        entries,
        entry_contracts,
        module,
        managed_layouts,
        options,
    )
    .plan()
}

struct ActionPlanner<'a> {
    ir: &'a IrContext,
    facts: &'a NativeOwnershipFunctionFacts,
    entries: &'a [EntryOwnership],
    entry_contracts: &'a HashMap<OpRef, Vec<EntryOwnership>>,
    module: &'a NativeOwnershipModuleFacts,
    managed_layouts: &'a HashSet<TypeRef>,
    borrowed: HashMap<ValueRef, ValueRef>,
    owned: HashSet<ValueRef>,
    move_units_at_last_use: bool,
    move_fields_from_dying_owners: bool,
    /// Projections that take the unit of their field from their owner's
    /// final release, with that owner and the field index.
    moved_fields: HashMap<ValueRef, (ValueRef, u32)>,
    liveness: &'a BlockLiveness,
    /// Position of each value's definition in program order, which orders
    /// actions independently of arena numbering.
    definition_order: HashMap<ValueRef, usize>,
    actions: Vec<OwnershipAction>,
}

impl<'a> ActionPlanner<'a> {
    fn new(
        ir: &'a IrContext,
        inputs: ActionInputs<'a>,
        entries: &'a [EntryOwnership],
        entry_contracts: &'a HashMap<OpRef, Vec<EntryOwnership>>,
        module: &'a NativeOwnershipModuleFacts,
        managed_layouts: &'a HashSet<TypeRef>,
        options: NativeOwnershipPlanOptions,
    ) -> Self {
        let facts = inputs.facts;
        // The temporary-borrow policy selects which policy-neutral projection
        // facts participate; it never changes the facts themselves.
        let borrowed = if options.elide_proven_field_borrows {
            facts.borrowable_projection_owners().clone()
        } else {
            HashMap::default()
        };
        let mut owned = facts.managed_values().clone();
        for (&value, entry) in ir.block_args(facts.cfg().entry()).iter().zip(entries) {
            if *entry == EntryOwnership::Borrowed {
                owned.remove(&value);
            }
        }
        let definition_order = facts
            .cfg()
            .blocks()
            .iter()
            .flat_map(|&block| {
                let results = ir
                    .block(block)
                    .ops
                    .iter()
                    .flat_map(|&op| ir.op_results(op).iter().copied());
                ir.block_args(block).iter().copied().chain(results)
            })
            .enumerate()
            .map(|(index, value)| (value, index))
            .collect();
        Self {
            ir,
            facts,
            entries,
            entry_contracts,
            module,
            managed_layouts,
            borrowed,
            owned,
            move_units_at_last_use: options.move_units_at_last_use,
            move_fields_from_dying_owners: options.move_fields_from_dying_owners,
            moved_fields: HashMap::default(),
            liveness: inputs.liveness,
            definition_order,
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

    /// Plan `block`, then plan it again for each round of field moves it
    /// admits. A moved projection is an owned root from its owner's release,
    /// so a projection read from it may move in the next round.
    fn plan_block(&mut self, block: BlockRef) -> Result<(), OwnershipPlanError> {
        loop {
            let first_action = self.actions.len();
            self.plan_block_once(block)?;
            if !self.move_fields_from_dying_owners {
                return Ok(());
            }
            let moves = self.field_moves(block, first_action);
            if moves.is_empty() {
                return Ok(());
            }
            self.actions.truncate(first_action);
            for (projection, owner, field) in moves {
                self.borrowed.remove(&projection);
                self.owned.insert(projection);
                self.moved_fields.insert(projection, (owner, field));
            }
        }
    }

    fn plan_block_once(&mut self, block: BlockRef) -> Result<(), OwnershipPlanError> {
        let ops = &self.ir.block(block).ops;
        let last_use = self.last_uses(ops);
        let mut transferred = HashSet::default();
        for (index, &op) in ops.iter().enumerate() {
            let first_action = self.actions.len();
            self.plan_operation(block, op, &mut transferred)?;
            if self.move_units_at_last_use {
                self.move_last_use_units(
                    block,
                    op,
                    index,
                    first_action,
                    &last_use,
                    &mut transferred,
                );
            }
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
                && !self.moved_fields.contains_key(&result)
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
        self.plan_final_releases(block, ops, &last_use, &transferred)
    }

    /// The projections in `block` whose field unit can move out of their
    /// owner at the owner's final release, as `(projection, owner, field)`.
    /// They are chosen from the actions the latest plan of `block` made,
    /// starting at `first_action`.
    fn field_moves(&self, block: BlockRef, first_action: usize) -> Vec<(ValueRef, ValueRef, u32)> {
        let aliases = self.facts.aliases();
        let actions = &self.actions[first_action..];
        let dying = actions
            .iter()
            .filter(|action| {
                action.kind == ActionKind::FinalRelease
                    && !matches!(action.anchor, ActionAnchor::BlockStart(_))
            })
            .map(|action| action.value)
            .collect::<HashSet<_>>();
        // A borrow needs a unit when it, or a borrow read from it, acquires
        // one for a use.
        let mut needs_unit = HashSet::default();
        for action in actions {
            if matches!(
                action.kind,
                ActionKind::StoreAcquire | ActionKind::CallAcquire | ActionKind::CopyAcquire
            ) {
                let mut value = root_value(aliases, action.value);
                while let Some(&owner) = self.borrowed.get(&value) {
                    if !needs_unit.insert(value) {
                        break;
                    }
                    value = root_value(aliases, owner);
                }
            }
        }
        let ops = &self.ir.block(block).ops;
        let live_out = &self.liveness.live_out[&block];
        let mut candidates = HashMap::<ValueRef, Vec<(ValueRef, u32)>>::default();
        for &op in ops {
            let Ok(get) = adt::StructGet::from_op(self.ir, op) else {
                continue;
            };
            let [projection] = self.ir.op_results(op) else {
                continue;
            };
            let owner = root_value(aliases, get.r#ref(self.ir));
            let field = get.field(self.ir);
            if self.borrowed.get(projection) == Some(&owner)
                && needs_unit.contains(projection)
                && !live_out.contains(projection)
                && dying.contains(&owner)
                && self.owned.contains(&owner)
                && self.has_static_size(owner)
                && self.field_reads(owner, field) == 1
            {
                candidates
                    .entry(owner)
                    .or_default()
                    .push((*projection, field));
            }
        }
        let first_use = |value: ValueRef| {
            ops.iter().position(|&op| {
                self.ir
                    .op_operands(op)
                    .iter()
                    .any(|&operand| self.derives_from(operand, value))
            })
        };
        let mut moves = Vec::new();
        for (owner, mut moved) in candidates {
            // The owner's release must follow every use of the owner and of
            // the borrows that stay, and precede every use of a moved one.
            loop {
                let last_owner_use = ops.iter().rposition(|&op| {
                    self.ir.op_operands(op).iter().any(|&operand| {
                        self.derives_from(operand, owner)
                            && !moved
                                .iter()
                                .any(|&(projection, _)| self.derives_from(operand, projection))
                    })
                });
                let before = moved.len();
                moved.retain(|&(projection, _)| first_use(projection) > last_owner_use);
                if moved.len() == before {
                    break;
                }
            }
            moves.extend(
                moved
                    .into_iter()
                    .map(|(projection, field)| (projection, owner, field)),
            );
        }
        moves.sort_unstable_by_key(|&(projection, _, _)| self.definition_order[&projection]);
        moves
    }

    /// Whether `value` is `ancestor` or a borrow read from it.
    fn derives_from(&self, value: ValueRef, ancestor: ValueRef) -> bool {
        let aliases = self.facts.aliases();
        let mut value = root_value(aliases, value);
        loop {
            if value == ancestor {
                return true;
            }
            match self.borrowed.get(&value) {
                Some(&owner) if root_value(aliases, owner) != value => {
                    value = root_value(aliases, owner);
                }
                _ => return false,
            }
        }
    }

    /// Whether the release of `owner` frees an exact struct layout by a
    /// static size, rather than through its RTTI entry. A closure's size
    /// depends on its capture layout, so its header RTTI frees it.
    fn has_static_size(&self, owner: ValueRef) -> bool {
        let ty = self.ir.value_ty(owner);
        allocation_layout(self.ir, self.managed_layouts, ty).is_ok_and(|layout| {
            adt::Struct::from_type_ref(self.ir, layout).is_some()
                && !crate::closure_lower::is_closure_layout(self.ir, layout)
        })
    }

    /// How many projections in the function read `field` of `owner`.
    fn field_reads(&self, owner: ValueRef, field: u32) -> usize {
        self.facts
            .projection_owners()
            .iter()
            .filter(|&(&projection, &projection_owner)| {
                projection_owner == owner
                    && matches!(
                        self.ir.value_def(projection),
                        ValueDef::OpResult(op, _)
                            if adt::StructGet::from_op(self.ir, op)
                                .is_ok_and(|get| get.field(self.ir) == field)
                    )
            })
            .count()
    }

    /// Index in `ops` of the last operation that uses each owned root, either
    /// directly or through a borrow derived from it.
    fn last_uses(&self, ops: &[OpRef]) -> HashMap<ValueRef, usize> {
        let mut last_use = HashMap::default();
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
        last_use
    }

    /// Turn the acquires planned for `op` into moves of the value's own unit
    /// when `op` is the last use of an owned root. Every operand of `op` that
    /// reaches the root must be a destination that acquires: any other use
    /// relies on the caller's unit for the duration of the operation. Only
    /// the first destination receives the unit; the rest keep acquiring.
    fn move_last_use_units(
        &mut self,
        block: BlockRef,
        op: OpRef,
        index: usize,
        first_action: usize,
        last_use: &HashMap<ValueRef, usize>,
        transferred: &mut HashSet<ValueRef>,
    ) {
        let aliases = self.facts.aliases();
        let mut uses = HashMap::<ValueRef, usize>::default();
        let mut borrowed_from = HashSet::default();
        for &operand in self.ir.op_operands(op) {
            let root = root_value(aliases, operand);
            *uses.entry(root).or_default() += 1;
            if let Some(owner) = borrowed_owner(&self.borrowed, aliases, root) {
                borrowed_from.insert(owner);
            }
        }
        let mut acquires = HashMap::<ValueRef, usize>::default();
        for action in &self.actions[first_action..] {
            if matches!(
                action.kind,
                ActionKind::StoreAcquire | ActionKind::CallAcquire
            ) {
                *acquires
                    .entry(root_value(aliases, action.value))
                    .or_default() += 1;
            }
        }
        let live_out = &self.liveness.live_out[&block];
        for action in &mut self.actions[first_action..] {
            let moved = match action.kind {
                ActionKind::StoreAcquire => ActionKind::StoreTransfer,
                ActionKind::CallAcquire => ActionKind::CallTransfer,
                _ => continue,
            };
            let root = root_value(aliases, action.value);
            if self.owned.contains(&root)
                && !self.borrowed.contains_key(&root)
                && !borrowed_from.contains(&root)
                && !live_out.contains(&root)
                && last_use.get(&root) == Some(&index)
                && uses.get(&root) == acquires.get(&root)
                && transferred.insert(root)
            {
                action.kind = moved;
            }
        }
    }

    /// Owned values that die on every edge into `block`: live out of each
    /// predecessor and not live into `block`.
    fn values_dying_on_entry(&self, block: BlockRef) -> Result<Vec<ValueRef>, OwnershipPlanError> {
        let predecessors = self.facts.cfg().predecessors(block);
        let live_in = &self.liveness.live_in[&block];
        let mut holders = HashMap::<ValueRef, usize>::default();
        for predecessor in predecessors {
            for &value in &self.liveness.live_out[predecessor] {
                if self.owned.contains(&value)
                    && !self.borrowed.contains_key(&value)
                    && !live_in.contains(&value)
                {
                    *holders.entry(value).or_default() += 1;
                }
            }
        }
        // The block start releases a value for every predecessor, so a
        // predecessor that no longer holds it would release it twice. The
        // function entry is such a predecessor of the entry block, and a
        // block that redefines the value is entered before it holds one.
        let defs = &self.liveness.defs[&block];
        if holders.iter().any(|(value, &count)| {
            count != predecessors.len() || block == self.facts.cfg().entry() || defs.contains(value)
        }) {
            return Err(OwnershipPlanError::new(
                "managed value dies on a control-flow edge whose successor has another predecessor",
            ));
        }
        Ok(holders.into_keys().collect())
    }
}

impl ActionPlanner<'_> {
    fn plan_operation(
        &mut self,
        block: BlockRef,
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
                    self.ir.get_type(new.r#type(self.ir))
                ))
            })?;
            validate_allocation_result(self.ir, op, new.r#type(self.ir), self.managed_layouts)?;
            return self.plan_owning_operands(op, 0, fields.iter().map(|(_, ty)| *ty));
        }
        if let Ok(new) = adt::VariantNew::from_op(self.ir, op) {
            let variants = get_enum_variants(self.ir, new.r#type(self.ir))
                .ok_or_else(|| OwnershipPlanError::new("variant_new has invalid layout"))?;
            let fields = variants
                .iter()
                .find(|(tag, _)| *tag == new.tag_ref(self.ir))
                .map(|(_, fields)| fields.as_slice())
                .ok_or_else(|| OwnershipPlanError::new("variant_new tag is stale"))?;
            validate_allocation_result(self.ir, op, new.r#type(self.ir), self.managed_layouts)?;
            return self.plan_owning_operands(op, 0, fields.iter().copied());
        }
        if let Ok(of) = adt::ArrayOf::from_op(self.ir, op) {
            let element = array_element(self.ir, of.r#type(self.ir))?;
            validate_allocation_result(self.ir, op, of.r#type(self.ir), self.managed_layouts)?;
            let count = self.ir.op_operands(op).len();
            return self.plan_owning_operands(op, 0, std::iter::repeat_n(element, count));
        }
        if let Ok(splice) = adt::ArraySplice::from_op(self.ir, op) {
            // `source` is borrowed for the operation, which acquires the units
            // of the elements it copies; only the new elements are stores.
            let array_ty = splice.r#type(self.ir);
            let element = array_element(self.ir, array_ty)?;
            validate_allocation_result(self.ir, op, array_ty, self.managed_layouts)?;
            if self.ir.value_ty(splice.source(self.ir)) != array_ty {
                return Err(OwnershipPlanError::new(
                    "array_splice source differs from its array type",
                ));
            }
            let first = SPLICE_ELEMENTS_OPERAND;
            let count = self.ir.op_operands(op).len().saturating_sub(first);
            return self.plan_owning_operands(op, first, std::iter::repeat_n(element, count));
        }
        if let Ok(new) = adt::ArrayNew::from_op(self.ir, op) {
            // `initial` is borrowed for the operation, which acquires a unit
            // for every element it fills.
            let array_ty = new.r#type(self.ir);
            let element = array_element(self.ir, array_ty)?;
            validate_allocation_result(self.ir, op, array_ty, self.managed_layouts)?;
            if is_typed_managed_reference(self.ir, element, self.managed_layouts)
                && !types_compatible(
                    self.ir,
                    self.ir.value_ty(new.initial(self.ir)),
                    element,
                    self.managed_layouts,
                )
            {
                return Err(OwnershipPlanError::new(
                    "array_new initial value type is malformed",
                ));
            }
            return Ok(());
        }
        if let Ok(set) = adt::ArraySet::from_op(self.ir, op) {
            let element = array_element(self.ir, self.ir.value_ty(set.r#ref(self.ir)))?;
            if is_typed_managed_reference(self.ir, element, self.managed_layouts) {
                if !types_compatible(
                    self.ir,
                    self.ir.value_ty(set.value(self.ir)),
                    element,
                    self.managed_layouts,
                ) {
                    return Err(OwnershipPlanError::new("array_set value type is malformed"));
                }
                self.actions.push(OwnershipAction {
                    kind: ActionKind::StoreAcquire,
                    value: set.value(self.ir),
                    anchor: ActionAnchor::Before(op),
                    destination: 0,
                });
                self.actions.push(OwnershipAction {
                    kind: ActionKind::ReleaseReplacedField,
                    value: set.r#ref(self.ir),
                    anchor: ActionAnchor::Before(op),
                    destination: 0,
                });
            }
            return Ok(());
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
            let mut counts = HashMap::<ValueRef, u32>::default();
            for (index, transfer) in transfers.enumerate() {
                if is_managed_value(self.ir, transfer.destination, self.managed_layouts) {
                    let root = root_value(self.facts.aliases(), transfer.source);
                    let count = counts.entry(root).or_default();
                    // A source that stays live after the branch keeps its
                    // unit, so the destination needs one of its own.
                    if *count > 0
                        || self.borrowed.contains_key(&root)
                        || self.liveness.live_out[&block].contains(&root)
                    {
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
        if exact_closure_root(self.ir, *input) != root {
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

    /// Plan the stores of the operands of `op` from position `first` on, one
    /// per entry of `field_types`.
    fn plan_owning_operands(
        &mut self,
        op: OpRef,
        first: usize,
        field_types: impl IntoIterator<Item = TypeRef>,
    ) -> Result<(), OwnershipPlanError> {
        let field_types = field_types.into_iter().collect::<Vec<_>>();
        let operands = self.ir.op_operands(op);
        if operands.len() < first || field_types.len() != operands.len() - first {
            return Err(OwnershipPlanError::new(
                "aggregate constructor arity is malformed",
            ));
        }
        for (offset, &field_ty) in field_types.iter().enumerate() {
            let index = first + offset;
            let operand = operands[index];
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
        || (exact_closure_root(ctx, source) == source && is_closure_alias(ctx, source))
    {
        return Err(OwnershipPlanError::new(
            "tribute_rt.into_raw requires an exact closure ownership value",
        ));
    }
    let mut block = None;
    let mut users = HashSet::default();
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

/// The allocation an exact closure ownership value starts from: the value
/// itself, or the closure allocation it is an upcast of. Each upcast takes a
/// closure layout to another and is the only use of its input, so the
/// allocation's unit has one path to `value`.
fn exact_closure_root(ctx: &IrContext, value: ValueRef) -> ValueRef {
    let mut current = value;
    loop {
        let ValueDef::OpResult(op, _) = ctx.value_def(current) else {
            return current;
        };
        let Ok(cast) = core::UnrealizedConversionCast::from_op(ctx, op) else {
            return current;
        };
        let input = cast.value(ctx);
        if !crate::closure_lower::is_closure_layout(ctx, ctx.value_ty(input))
            || !crate::closure_lower::is_closure_layout(ctx, ctx.value_ty(current))
            || ctx.uses(input).len() != 1
        {
            return current;
        }
        current = input;
    }
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
        let indirect = DynIndirectCallLike::get(self.ir, op);
        let tail =
            func::TailCall::matches(self.ir, op) || func::TailCallIndirect::matches(self.ir, op);
        let operands = self.ir.op_operands(op);
        let args = if let Some(call) = indirect {
            call.indirect_arguments(self.ir)
                .ok_or_else(|| OwnershipPlanError::new("indirect call has malformed operands"))?
        } else {
            operands
        };
        let entries = if let Some(call) = indirect {
            let signature = call
                .exact_signature(self.ir)
                .and_then(|ty| func::FuncSig::from_type_ref(self.ir, ty))
                .ok_or_else(|| {
                    OwnershipPlanError::new(format!(
                        "indirect call {op:?} lacks exact signature; attrs = {:?}, operand types = {:?}",
                        self.ir.op_attrs(op),
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
            // The exact signature states the callee's entry contract; an
            // unmarked managed input has the retained callable contract.
            signature
                .inputs(self.ir)
                .iter()
                .zip(consumed_inputs(self.ir, signature)?)
                .map(|(&ty, consumed)| {
                    if !is_typed_managed_reference(self.ir, ty, self.managed_layouts) {
                        EntryOwnership::Plain
                    } else if consumed {
                        EntryOwnership::Consumed
                    } else {
                        EntryOwnership::Retained
                    }
                })
                .collect::<Vec<_>>()
        } else {
            let callee = self
                .ir
                .op_attrs(op)
                .get_symbol_ref("callee")
                .ok_or_else(|| OwnershipPlanError::new("direct call lacks callee identity"))?;
            let Some(callee_op) = self.module.callee(self.ir, op, callee) else {
                if args
                    .iter()
                    .any(|&value| is_managed_value(self.ir, value, self.managed_layouts))
                {
                    return Err(OwnershipPlanError::new(format!(
                        "unclassified direct call {op:?} to @{callee} carries a managed reference"
                    )));
                }
                return Ok(());
            };
            let signature = self
                .ir
                .op_attrs(callee_op)
                .get_type("type")
                .and_then(|ty| func::FuncSig::from_type_ref(self.ir, ty))
                .ok_or_else(|| OwnershipPlanError::new("direct callee lacks exact signature"))?;
            validate_call_contract(self.ir, op, signature, args, self.managed_layouts)?;
            self.entry_contracts
                .get(&callee_op)
                .cloned()
                .ok_or_else(|| OwnershipPlanError::new("callee has no trusted entry contract"))?
        };
        if entries.len() != args.len() {
            return Err(OwnershipPlanError::new(
                "call arity differs from entry contract",
            ));
        }
        let mut transfers = HashMap::<ValueRef, u32>::default();
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

/// The operand position of the first new element of an `adt.array_splice`,
/// after `source`, `start`, and `end`.
const SPLICE_ELEMENTS_OPERAND: usize = 3;

/// The element type of the array type `array_ty`.
fn array_element(ctx: &IrContext, array_ty: TypeRef) -> Result<TypeRef, OwnershipPlanError> {
    tribute_ir::dialect::adt::Array::from_type_ref(ctx, array_ty)
        .map(|array| array.element(ctx))
        .ok_or_else(|| OwnershipPlanError::new("array operation has a non-array type"))
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
    // A tail transfer has no result here: the callee returns its result to
    // the caller's caller.
    if func::TailCall::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op) {
        return Ok(());
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
    let physically_empty = trunk_ir::dialect::core::Nil::matches(ctx, expected)
        || trunk_ir::dialect::core::Never::matches(ctx, expected);
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
        last_use: &HashMap<ValueRef, usize>,
        transferred: &HashSet<ValueRef>,
    ) -> Result<(), OwnershipPlanError> {
        let dying_on_entry = self.values_dying_on_entry(block)?;
        let mut dying = HashSet::default();
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
        dying.extend(dying_on_entry.iter().copied());
        dying.sort_unstable_by_key(|value| self.definition_order[value]);
        for (destination, value) in dying.into_iter().enumerate() {
            let anchor = if dying_on_entry.contains(&value) {
                ActionAnchor::BlockStart(block)
            } else if let Some(&index) = last_use.get(&value) {
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
            // An owner may die in several blocks; only the projections read
            // in this one move at this release.
            let mut moved = self
                .moved_fields
                .iter()
                .filter(|&(&projection, &(owner, _))| {
                    owner == value
                        && matches!(
                            self.ir.value_def(projection),
                            ValueDef::OpResult(op, _) if self.ir.op(op).parent_block == Some(block)
                        )
                })
                .map(|(&projection, &(_, field))| (projection, field))
                .collect::<Vec<_>>();
            moved.sort_unstable_by_key(|&(_, field)| field);
            self.actions.push(OwnershipAction {
                kind: if moved.is_empty() {
                    ActionKind::FinalRelease
                } else {
                    ActionKind::ReleaseMoving
                },
                value,
                anchor,
                destination: destination as u32,
            });
            for (projection, field) in moved {
                self.actions.push(OwnershipAction {
                    kind: ActionKind::FieldMove,
                    value: projection,
                    anchor,
                    destination: field,
                });
            }
        }
        Ok(())
    }
}
