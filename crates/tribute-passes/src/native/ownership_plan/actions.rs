use super::*;

pub(super) fn plan_function_actions(
    ctx: &IrContext,
    body: RegionRef,
    entries: &[EntryOwnership],
    entry_contracts: &HashMap<Symbol, Vec<EntryOwnership>>,
    definitions: &HashMap<Symbol, OpRef>,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<Vec<OwnershipAction>, OwnershipPlanError> {
    let mut managed = collect_managed_values(ctx, body, managed_layouts);
    let aliases = build_aliases(ctx, body, &mut managed, managed_layouts)?;
    let borrowed_loads = collect_borrowed_loads(ctx, body, managed_layouts, &aliases)?;
    let liveness = compute_liveness(ctx, body, &managed, &aliases, &borrowed_loads);
    let entry_block = ctx.region(body).blocks[0];
    let mut owned = managed.clone();
    for (&value, entry) in ctx.block_args(entry_block).iter().zip(entries) {
        if *entry == EntryOwnership::Borrowed {
            owned.remove(&value);
        }
    }
    let mut actions = Vec::new();

    for (index, (&value, entry)) in ctx.block_args(entry_block).iter().zip(entries).enumerate() {
        if *entry == EntryOwnership::Retained {
            actions.push(OwnershipAction {
                kind: ActionKind::EntryAcquire,
                value,
                anchor: ActionAnchor::BlockStart(entry_block),
                destination: index as u32,
            });
        }
    }

    for &block in &ctx.region(body).blocks {
        let ops = ctx.block(block).ops.to_vec();
        let mut transferred = HashSet::new();
        for &op in &ops {
            plan_operation_actions(
                ctx,
                op,
                entry_contracts,
                definitions,
                managed_layouts,
                &aliases,
                &borrowed_loads,
                &mut transferred,
                &mut actions,
            )?;
            if let Some(&borrowed) = ctx
                .op_results(op)
                .first()
                .filter(|result| borrowed_loads.contains_key(result))
            {
                actions.push(OwnershipAction {
                    kind: ActionKind::BorrowLoad,
                    value: borrowed,
                    anchor: ActionAnchor::After(op),
                    destination: 0,
                });
            }
        }
        plan_final_releases(
            ctx,
            block,
            &ops,
            &owned,
            &aliases,
            &borrowed_loads,
            &liveness,
            &transferred,
            &mut actions,
        );
    }
    Ok(actions)
}

fn collect_managed_values(
    ctx: &IrContext,
    body: RegionRef,
    managed_layouts: &HashSet<TypeRef>,
) -> HashSet<ValueRef> {
    let mut values = HashSet::new();
    for &block in &ctx.region(body).blocks {
        for &value in ctx.block_args(block) {
            if is_managed_value(ctx, value, managed_layouts) {
                values.insert(value);
            }
        }
        for &op in &ctx.block(block).ops {
            for &value in ctx.op_results(op) {
                if is_managed_value(ctx, value, managed_layouts) {
                    values.insert(value);
                }
            }
        }
    }
    values
}

fn build_aliases(
    ctx: &IrContext,
    body: RegionRef,
    managed: &mut HashSet<ValueRef>,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<HashMap<ValueRef, ValueRef>, OwnershipPlanError> {
    let mut aliases = HashMap::new();
    for &block in &ctx.region(body).blocks {
        for &op in &ctx.block(block).ops {
            if !(adt::RefCast::matches(ctx, op) || core::UnrealizedConversionCast::matches(ctx, op))
            {
                continue;
            }
            let ([input], [output]) = (ctx.op_operands(op), ctx.op_results(op)) else {
                return Err(OwnershipPlanError::new("managed alias has malformed arity"));
            };
            let input_managed = is_managed_value(ctx, *input, managed_layouts);
            let output_managed = is_managed_value(ctx, *output, managed_layouts);
            let input_data = ctx.types.get(ctx.value_ty(*input));
            let output_data = ctx.types.get(ctx.value_ty(*output));
            if input_data.dialect == Symbol::new("core")
                && input_data.name == Symbol::new("ptr")
                && output_data.dialect == Symbol::new("adt")
                && output_data.name == Symbol::new("typeref")
            {
                return Err(OwnershipPlanError::new(format!(
                    "raw pointer alias {op:?} masquerades as managed adt.typeref: {} -> {}",
                    ctx.value_ty(*input),
                    ctx.value_ty(*output)
                )));
            }
            if let Ok(cast) = adt::RefCast::from_op(ctx, op) {
                let target = cast.r#type(ctx);
                if ctx.value_ty(*output) != target
                    || !input_managed
                    || !output_managed
                    || (!is_anyref_type(ctx, ctx.value_ty(*input))
                        && !is_anyref_type(ctx, target)
                        && !nominal_types_compatible(ctx, ctx.value_ty(*input), target))
                {
                    return Err(OwnershipPlanError::new(
                        "adt.ref_cast does not preserve a compatible managed reference",
                    ));
                }
            }
            if input_managed && output_managed {
                let root = aliases.get(input).copied().unwrap_or(*input);
                aliases.insert(*output, root);
                managed.remove(output);
            }
        }
    }
    Ok(aliases)
}

fn root_value(aliases: &HashMap<ValueRef, ValueRef>, value: ValueRef) -> ValueRef {
    aliases.get(&value).copied().unwrap_or(value)
}

fn collect_borrowed_loads(
    ctx: &IrContext,
    body: RegionRef,
    managed_layouts: &HashSet<TypeRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
) -> Result<HashMap<ValueRef, ValueRef>, OwnershipPlanError> {
    let mut borrowed = HashMap::new();
    for &block in &ctx.region(body).blocks {
        for &op in &ctx.block(block).ops {
            let source = if let Ok(get) = adt::StructGet::from_op(ctx, op) {
                Some(get.r#ref(ctx))
            } else if let Ok(get) = adt::VariantGet::from_op(ctx, op) {
                Some(get.r#ref(ctx))
            } else {
                None
            };
            let Some(source) = source else { continue };
            let Some(&result) = ctx.op_results(op).first() else {
                return Err(OwnershipPlanError::new("ADT projection has no result"));
            };
            validate_projection_contract(ctx, op, source, result, managed_layouts)?;
            if is_managed_value(ctx, result, managed_layouts) {
                borrowed.insert(result, root_value(aliases, source));
            }
        }
    }
    Ok(borrowed)
}

fn validate_projection_contract(
    ctx: &IrContext,
    op: OpRef,
    source: ValueRef,
    result: ValueRef,
    managed_layouts: &HashSet<TypeRef>,
) -> Result<(), OwnershipPlanError> {
    let (layout, field_ty) = if let Ok(get) = adt::StructGet::from_op(ctx, op) {
        let fields = get_struct_fields(ctx, get.r#type(ctx))
            .ok_or_else(|| OwnershipPlanError::new("struct_get has invalid layout"))?;
        let (_, field_ty) = fields
            .get(get.field(ctx) as usize)
            .ok_or_else(|| OwnershipPlanError::new("struct_get field is stale"))?;
        (get.r#type(ctx), *field_ty)
    } else {
        let get = adt::VariantGet::from_op(ctx, op)
            .map_err(|_| OwnershipPlanError::new("unsupported ADT projection"))?;
        let variants = get_enum_variants(ctx, get.r#type(ctx))
            .ok_or_else(|| OwnershipPlanError::new("variant_get has invalid layout"))?;
        let fields = variants
            .iter()
            .find(|(tag, _)| *tag == get.tag(ctx))
            .map(|(_, fields)| fields)
            .ok_or_else(|| OwnershipPlanError::new("variant_get tag is stale"))?;
        let field_ty = fields
            .get(get.field(ctx) as usize)
            .ok_or_else(|| OwnershipPlanError::new("variant_get field is stale"))?;
        (get.r#type(ctx), *field_ty)
    };
    let source_ty = ctx.value_ty(source);
    let source_data = ctx.types.get(source_ty);
    let raw_source =
        source_data.dialect == Symbol::new("core") && source_data.name == Symbol::new("ptr");
    let result_managed = is_managed_value(ctx, result, managed_layouts);
    if (!raw_source && !types_compatible(ctx, source_ty, layout, managed_layouts))
        || (result_managed
            && !types_compatible(ctx, ctx.value_ty(result), field_ty, managed_layouts))
    {
        return Err(OwnershipPlanError::new(
            "ADT projection managed type contract is malformed",
        ));
    }
    Ok(())
}

struct Liveness {
    defs: HashMap<BlockRef, HashSet<ValueRef>>,
    live_in: HashMap<BlockRef, HashSet<ValueRef>>,
    live_out: HashMap<BlockRef, HashSet<ValueRef>>,
}

fn compute_liveness(
    ctx: &IrContext,
    body: RegionRef,
    managed: &HashSet<ValueRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
    borrowed: &HashMap<ValueRef, ValueRef>,
) -> Liveness {
    let blocks = ctx.region(body).blocks.to_vec();
    let mut uses = HashMap::new();
    let mut defs = HashMap::new();
    for &block in &blocks {
        let mut block_uses = HashSet::new();
        let mut block_defs = HashSet::new();
        for &value in ctx.block_args(block) {
            if managed.contains(&value) {
                block_defs.insert(value);
            }
        }
        for &op in &ctx.block(block).ops {
            for &operand in ctx.op_operands(op) {
                let root = root_value(aliases, operand);
                if managed.contains(&root) && !block_defs.contains(&root) {
                    block_uses.insert(root);
                }
                if let Some(owner) = borrowed
                    .get(&operand)
                    .copied()
                    .map(|v| root_value(aliases, v))
                    && managed.contains(&owner)
                    && !block_defs.contains(&owner)
                {
                    block_uses.insert(owner);
                }
            }
            for &result in ctx.op_results(op) {
                let root = root_value(aliases, result);
                if managed.contains(&root) {
                    block_defs.insert(root);
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
            if let Some(&terminator) = ctx.block(block).ops.last() {
                for successor in &ctx.op(terminator).successors {
                    out.extend(live_in[successor].iter().copied());
                }
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

#[allow(clippy::too_many_arguments)]
fn plan_operation_actions(
    ctx: &IrContext,
    op: OpRef,
    entry_contracts: &HashMap<Symbol, Vec<EntryOwnership>>,
    definitions: &HashMap<Symbol, OpRef>,
    managed_layouts: &HashSet<TypeRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
    borrowed: &HashMap<ValueRef, ValueRef>,
    transferred: &mut HashSet<ValueRef>,
    actions: &mut Vec<OwnershipAction>,
) -> Result<(), OwnershipPlanError> {
    if let Ok(null) = adt::RefNull::from_op(ctx, op) {
        let [result] = ctx.op_results(op) else {
            return Err(OwnershipPlanError::new(
                "adt.ref_null has malformed result arity",
            ));
        };
        let ty = null.r#type(ctx);
        let result_managed = is_managed_value(ctx, *result, managed_layouts);
        let declared_managed = is_typed_managed_reference(ctx, ty, managed_layouts);
        if result_managed
            && (!declared_managed
                || !types_compatible(ctx, ctx.value_ty(*result), ty, managed_layouts))
        {
            return Err(OwnershipPlanError::new(
                "adt.ref_null must inhabit a compatible managed reference type",
            ));
        }
        return Ok(());
    }
    if let Ok(new) = adt::StructNew::from_op(ctx, op) {
        let fields = get_struct_fields(ctx, new.r#type(ctx)).ok_or_else(|| {
            OwnershipPlanError::new(format!(
                "struct_new {op:?} has invalid layout {} ({:?})",
                new.r#type(ctx),
                ctx.types.get(new.r#type(ctx))
            ))
        })?;
        validate_allocation_result(ctx, op, new.r#type(ctx), managed_layouts)?;
        return plan_owning_operands(
            ctx,
            op,
            fields.iter().map(|(_, ty)| *ty),
            managed_layouts,
            actions,
        );
    }
    if let Ok(new) = adt::VariantNew::from_op(ctx, op) {
        let variants = get_enum_variants(ctx, new.r#type(ctx))
            .ok_or_else(|| OwnershipPlanError::new("variant_new has invalid layout"))?;
        let fields = variants
            .iter()
            .find(|(tag, _)| *tag == new.tag(ctx))
            .map(|(_, fields)| fields.as_slice())
            .ok_or_else(|| OwnershipPlanError::new("variant_new tag is stale"))?;
        validate_allocation_result(ctx, op, new.r#type(ctx), managed_layouts)?;
        return plan_owning_operands(ctx, op, fields.iter().copied(), managed_layouts, actions);
    }
    if let Ok(set) = adt::StructSet::from_op(ctx, op) {
        let fields = get_struct_fields(ctx, set.r#type(ctx))
            .ok_or_else(|| OwnershipPlanError::new("struct_set has invalid layout"))?;
        let index = set.field(ctx) as usize;
        let (_, field_ty) = fields
            .get(index)
            .ok_or_else(|| OwnershipPlanError::new("struct_set field is stale"))?;
        if !types_compatible(
            ctx,
            ctx.value_ty(set.r#ref(ctx)),
            set.r#type(ctx),
            managed_layouts,
        ) {
            return Err(OwnershipPlanError::new(
                "struct_set typed contract is malformed",
            ));
        }
        if is_typed_managed_reference(ctx, *field_ty, managed_layouts) {
            if !types_compatible(
                ctx,
                ctx.value_ty(set.value(ctx)),
                *field_ty,
                managed_layouts,
            ) {
                return Err(OwnershipPlanError::new(
                    "struct_set value type is malformed",
                ));
            }
            actions.push(OwnershipAction {
                kind: ActionKind::StoreAcquire,
                value: set.value(ctx),
                anchor: ActionAnchor::Before(op),
                destination: index as u32,
            });
            actions.push(OwnershipAction {
                kind: ActionKind::ReleaseReplacedField,
                value: set.r#ref(ctx),
                anchor: ActionAnchor::Before(op),
                destination: index as u32,
            });
        }
        return Ok(());
    }
    if func::Call::matches(ctx, op)
        || func::CallIndirect::matches(ctx, op)
        || func::TailCall::matches(ctx, op)
        || func::TailCallIndirect::matches(ctx, op)
    {
        return plan_call_actions(
            ctx,
            op,
            entry_contracts,
            definitions,
            managed_layouts,
            aliases,
            borrowed,
            transferred,
            actions,
        );
    }
    if func::Return::matches(ctx, op) {
        for (index, &operand) in ctx.op_operands(op).iter().enumerate() {
            if is_managed_value(ctx, operand, managed_layouts) {
                let root = root_value(aliases, operand);
                if borrowed.contains_key(&root) {
                    actions.push(OwnershipAction {
                        kind: ActionKind::CopyAcquire,
                        value: operand,
                        anchor: ActionAnchor::Before(op),
                        destination: index as u32,
                    });
                }
                transferred.insert(root);
                actions.push(OwnershipAction {
                    kind: ActionKind::ReturnTransfer,
                    value: operand,
                    anchor: ActionAnchor::Before(op),
                    destination: index as u32,
                });
            }
        }
    } else if let Ok(branch) = cf::Br::from_op(ctx, op) {
        let destination = branch.dest(ctx);
        let mut counts = HashMap::<ValueRef, u32>::new();
        for (index, (&operand, &argument)) in ctx
            .op_operands(op)
            .iter()
            .zip(ctx.block_args(destination))
            .enumerate()
        {
            if is_managed_value(ctx, argument, managed_layouts) {
                let root = root_value(aliases, operand);
                let count = counts.entry(root).or_default();
                if *count > 0 || borrowed.contains_key(&root) {
                    actions.push(OwnershipAction {
                        kind: ActionKind::CopyAcquire,
                        value: operand,
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

fn plan_owning_operands(
    ctx: &IrContext,
    op: OpRef,
    field_types: impl IntoIterator<Item = TypeRef>,
    managed_layouts: &HashSet<TypeRef>,
    actions: &mut Vec<OwnershipAction>,
) -> Result<(), OwnershipPlanError> {
    let field_types = field_types.into_iter().collect::<Vec<_>>();
    if field_types.len() != ctx.op_operands(op).len() {
        return Err(OwnershipPlanError::new(
            "aggregate constructor arity is malformed",
        ));
    }
    for (index, (&operand, &field_ty)) in ctx.op_operands(op).iter().zip(&field_types).enumerate() {
        if is_typed_managed_reference(ctx, field_ty, managed_layouts) {
            if !types_compatible(ctx, ctx.value_ty(operand), field_ty, managed_layouts) {
                return Err(OwnershipPlanError::new(format!(
                    "aggregate managed field type is stale at {op:?}: expected {field_ty}, got {}",
                    ctx.value_ty(operand)
                )));
            }
            actions.push(OwnershipAction {
                kind: ActionKind::StoreAcquire,
                value: operand,
                anchor: ActionAnchor::Before(op),
                destination: index as u32,
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn plan_call_actions(
    ctx: &IrContext,
    op: OpRef,
    entry_contracts: &HashMap<Symbol, Vec<EntryOwnership>>,
    definitions: &HashMap<Symbol, OpRef>,
    managed_layouts: &HashSet<TypeRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
    borrowed: &HashMap<ValueRef, ValueRef>,
    transferred: &mut HashSet<ValueRef>,
    actions: &mut Vec<OwnershipAction>,
) -> Result<(), OwnershipPlanError> {
    let indirect = func::CallIndirect::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op);
    let tail = func::TailCall::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op);
    let operands = ctx.op_operands(op);
    let args = operands.get(usize::from(indirect)..).unwrap_or_default();
    let entries = if indirect {
        let signature = get_indirect_call_signature(ctx, op)
            .and_then(|ty| core::Func::from_type_ref(ctx, ty))
            .ok_or_else(|| {
                OwnershipPlanError::new(format!(
                    "indirect call {op:?} lacks exact signature; attrs = {:?}, operand types = {:?}",
                    ctx.op(op).attributes,
                    ctx.op_operands(op)
                        .iter()
                        .map(|&value| ctx.value_ty(value))
                        .collect::<Vec<_>>()
                ))
            })?;
        if signature.params(ctx).len() != args.len() {
            return Err(OwnershipPlanError::new("indirect call arity is malformed"));
        }
        validate_call_contract(ctx, op, signature, args, managed_layouts)?;
        signature
            .params(ctx)
            .iter()
            .map(|&ty| {
                if is_typed_managed_reference(ctx, ty, managed_layouts) {
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
        let callee = ctx
            .op(op)
            .attributes
            .get_symbol("callee")
            .ok_or_else(|| OwnershipPlanError::new("direct call lacks callee identity"))?;
        if !definitions.contains_key(&callee) {
            if args
                .iter()
                .any(|&value| is_managed_value(ctx, value, managed_layouts))
            {
                return Err(OwnershipPlanError::new(format!(
                    "unclassified direct call {op:?} to @{callee} carries a managed reference"
                )));
            }
            return Ok(());
        }
        let callee_op = definitions[&callee];
        let signature = ctx
            .op(callee_op)
            .attributes
            .get_type("type")
            .and_then(|ty| core::Func::from_type_ref(ctx, ty))
            .ok_or_else(|| OwnershipPlanError::new("direct callee lacks exact signature"))?;
        validate_call_contract(ctx, op, signature, args, managed_layouts)?;
        entry_contracts
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
        let managed = is_managed_value(ctx, argument, managed_layouts);
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
            let root = root_value(aliases, argument);
            let count = transfers.entry(root).or_default();
            if *count > 0 || borrowed.contains_key(&root) {
                actions.push(OwnershipAction {
                    kind: ActionKind::CopyAcquire,
                    value: argument,
                    anchor: ActionAnchor::Before(op),
                    destination: index as u32,
                });
            }
            *count += 1;
            transferred.insert(root);
        }
        actions.push(OwnershipAction {
            kind,
            value: argument,
            anchor: ActionAnchor::Before(op),
            destination: index as u32,
        });
    }
    Ok(())
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
    signature: core::Func,
    args: &[ValueRef],
    managed_layouts: &HashSet<TypeRef>,
) -> Result<(), OwnershipPlanError> {
    if args.len() != signature.params(ctx).len()
        || args
            .iter()
            .zip(signature.params(ctx))
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
        signature.r#return(ctx),
        managed_layouts,
        "call result",
    )
}

pub(super) fn validate_result_contract(
    ctx: &IrContext,
    values: &[ValueRef],
    expected: TypeRef,
    managed_layouts: &HashSet<TypeRef>,
    subject: &str,
) -> Result<(), OwnershipPlanError> {
    if !is_typed_managed_reference(ctx, expected, managed_layouts)
        && !values
            .iter()
            .any(|&value| is_managed_value(ctx, value, managed_layouts))
    {
        return Ok(());
    }
    let expected_data = ctx.types.get(expected);
    let physically_empty = expected_data.dialect == Symbol::new("core")
        && (expected_data.name == Symbol::new("nil") || expected_data.name == Symbol::new("never"));
    let valid = if physically_empty {
        values.is_empty()
    } else {
        matches!(values, [value] if types_compatible(ctx, ctx.value_ty(*value), expected, managed_layouts))
    };
    if !valid {
        return Err(OwnershipPlanError::new(format!(
            "{subject} differs from the exact callable signature"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn plan_final_releases(
    ctx: &IrContext,
    block: BlockRef,
    ops: &[OpRef],
    owned: &HashSet<ValueRef>,
    aliases: &HashMap<ValueRef, ValueRef>,
    borrowed: &HashMap<ValueRef, ValueRef>,
    liveness: &Liveness,
    transferred: &HashSet<ValueRef>,
    actions: &mut Vec<OwnershipAction>,
) {
    let mut last_use = HashMap::new();
    for (index, &op) in ops.iter().enumerate() {
        for &operand in ctx.op_operands(op) {
            let root = root_value(aliases, operand);
            if owned.contains(&root) {
                last_use.insert(root, index);
            }
            if let Some(owner) = borrowed
                .get(&operand)
                .copied()
                .map(|v| root_value(aliases, v))
                && owned.contains(&owner)
            {
                last_use.insert(owner, index);
            }
        }
    }
    let mut dying = HashSet::new();
    for value in &liveness.live_in[&block] {
        if owned.contains(value)
            && !liveness.live_out[&block].contains(value)
            && !transferred.contains(value)
        {
            dying.insert(*value);
        }
    }
    for value in &liveness.defs[&block] {
        if owned.contains(value)
            && !liveness.live_out[&block].contains(value)
            && !transferred.contains(value)
            && !borrowed.contains_key(value)
        {
            dying.insert(*value);
        }
    }
    let mut dying = dying.into_iter().collect::<Vec<_>>();
    dying.sort_unstable();
    for (destination, value) in dying.into_iter().enumerate() {
        let anchor = if let Some(&index) = last_use.get(&value) {
            let op = ops[index];
            if is_terminator(ctx, op) {
                ActionAnchor::Before(op)
            } else {
                ActionAnchor::After(op)
            }
        } else if liveness.live_in[&block].contains(&value) {
            ActionAnchor::BlockStart(block)
        } else if let ValueDef::OpResult(op, _) = ctx.value_def(value) {
            ActionAnchor::After(op)
        } else {
            ActionAnchor::BlockStart(block)
        };
        actions.push(OwnershipAction {
            kind: ActionKind::FinalRelease,
            value,
            anchor,
            destination: destination as u32,
        });
    }
}
