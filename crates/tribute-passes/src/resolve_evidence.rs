//! Evidence-based dispatch resolution pass.
//!
//! This pass transforms static `cont.shift` operations into evidence-based dispatch
//! using runtime function calls.
//!
//! ## Transformations
//!
//! ```text
//! // Before (static placeholder tag)
//! %result = cont.shift(%placeholder_tag, args...) { ability_ref, op_name }
//!
//! // After (dynamic evidence-based tag)
//! %marker = func.call @__tribute_evidence_lookup(%ev, ability_id)
//! %tag = func.call @__tribute_marker_prompt(%marker)
//! %result = cont.shift(%tag, args...) { ability_ref, op_name }
//! ```
//!
//! This pass must run AFTER `add_evidence_params` so that effectful functions
//! have evidence as their first parameter.

use std::collections::{HashMap, HashSet};

use im::HashMap as ImHashMap;
use tribute_ir::dialect::ability;
use trunk_ir::dialect::{cont, core, func};
use trunk_ir::{
    Attribute, Block, BlockArg, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type,
    Value,
};

/// Sentinel value used for unresolved cont.shift tags.
/// When a shift is generated without an enclosing handler, this value is used
/// as a placeholder. The evidence pass should transform all such shifts.
pub const UNRESOLVED_SHIFT_TAG: u32 = u32::MAX;

/// Resolve evidence-based dispatch for `cont.shift` operations.
///
/// Transforms `cont.shift` with placeholder tags into `cont.shift` with
/// dynamically resolved tags via evidence lookup. This enables proper
/// handler dispatch at runtime.
#[salsa::tracked]
pub fn resolve_evidence_dispatch<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // First, ensure runtime helper functions are declared
    let module = ensure_runtime_functions(db, module);

    // Collect functions with evidence parameters
    let fns_with_evidence = collect_functions_with_evidence(db, &module);

    // Transform shifts in functions with evidence (if any)
    let module = if fns_with_evidence.is_empty() {
        module
    } else {
        transform_shifts_in_module(db, module, &fns_with_evidence)
    };

    // Validate that all shifts have been resolved (no sentinel tags remain).
    // This must run regardless of whether evidence functions exist, to catch
    // any unresolved shifts that may have been introduced incorrectly.
    validate_no_unresolved_shifts(db, &module);

    module
}

/// Ensure runtime helper functions are declared in the module.
///
/// These functions are:
/// - `__tribute_evidence_lookup(ev: Evidence, ability_id: i32) -> Marker`
/// - `__tribute_marker_prompt(marker: Marker) -> PromptTag`
/// - `__tribute_evidence_extend(ev: Evidence, ability_id: i32, tag: PromptTag) -> Evidence`
fn ensure_runtime_functions<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    // Module body should always have exactly one block
    assert!(
        blocks.len() <= 1,
        "ICE: Module body should have at most one block, found {}",
        blocks.len()
    );

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    // Check if functions already exist
    let mut has_evidence_lookup = false;
    let mut has_marker_prompt = false;
    let mut has_evidence_extend = false;

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            let name = func_op.sym_name(db);
            if name == Symbol::new("__tribute_evidence_lookup") {
                has_evidence_lookup = true;
            } else if name == Symbol::new("__tribute_marker_prompt") {
                has_marker_prompt = true;
            } else if name == Symbol::new("__tribute_evidence_extend") {
                has_evidence_extend = true;
            }
        }
    }

    if has_evidence_lookup && has_marker_prompt && has_evidence_extend {
        return module;
    }

    // Add missing function declarations
    let location = module.location(db);
    let mut new_ops: Vec<Operation<'db>> = entry_block.operations(db).iter().copied().collect();

    if !has_evidence_lookup {
        let evidence_ty = ability::EvidencePtr::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let marker_ty = ability::Marker::new(db).as_type();

        // fn __tribute_evidence_lookup(ev: Evidence, ability_id: i32) -> Marker
        let func_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, i32_ty]), marker_ty);

        // Empty body with unreachable (extern function pattern)
        let unreachable_op = func::unreachable(db, location);
        let body_block = Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::from(vec![
                BlockArg::of_type(db, evidence_ty),
                BlockArg::of_type(db, i32_ty),
            ]),
            IdVec::from(vec![unreachable_op.as_operation()]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));

        let func_op = func::func(
            db,
            location,
            Symbol::new("__tribute_evidence_lookup"),
            *func_ty,
            body,
        );
        new_ops.insert(0, func_op.as_operation());
    }

    if !has_marker_prompt {
        let marker_ty = ability::Marker::new(db).as_type();
        let prompt_tag_ty = ability::PromptTag::new(db).as_type();

        // fn __tribute_marker_prompt(marker: Marker) -> PromptTag
        let func_ty = core::Func::new(db, IdVec::from(vec![marker_ty]), prompt_tag_ty);

        // Empty body with unreachable
        let unreachable_op = func::unreachable(db, location);
        let body_block = Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::from(vec![BlockArg::of_type(db, marker_ty)]),
            IdVec::from(vec![unreachable_op.as_operation()]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));

        let func_op = func::func(
            db,
            location,
            Symbol::new("__tribute_marker_prompt"),
            *func_ty,
            body,
        );
        new_ops.insert(0, func_op.as_operation());
    }

    if !has_evidence_extend {
        let evidence_ty = ability::EvidencePtr::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let prompt_tag_ty = ability::PromptTag::new(db).as_type();

        // fn __tribute_evidence_extend(ev: Evidence, ability_id: i32, tag: PromptTag) -> Evidence
        let func_ty = core::Func::new(
            db,
            IdVec::from(vec![evidence_ty, i32_ty, prompt_tag_ty]),
            evidence_ty,
        );

        // Empty body with unreachable
        let unreachable_op = func::unreachable(db, location);
        let body_block = Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::from(vec![
                BlockArg::of_type(db, evidence_ty),
                BlockArg::of_type(db, i32_ty),
                BlockArg::of_type(db, prompt_tag_ty),
            ]),
            IdVec::from(vec![unreachable_op.as_operation()]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));

        let func_op = func::func(
            db,
            location,
            Symbol::new("__tribute_evidence_extend"),
            *func_ty,
            body,
        );
        new_ops.insert(0, func_op.as_operation());
    }

    // Rebuild module with new functions
    let new_entry_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        new_ops.into_iter().collect(),
    );

    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_entry_block]));
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Collect all function names that have `ability.evidence_ptr` as their first parameter.
fn collect_functions_with_evidence<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
) -> HashSet<Symbol> {
    let mut fns_with_evidence = HashSet::new();

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let func_ty = func_op.r#type(db);
                if let Some(core_func) = core::Func::from_type(db, func_ty) {
                    let params = core_func.params(db);
                    if !params.is_empty()
                        && ability::EvidencePtr::from_type(db, params[0]).is_some()
                    {
                        fns_with_evidence.insert(func_op.sym_name(db));
                    }
                }
            }
        }
    }

    fns_with_evidence
}

/// Transform cont.shift operations in all functions with evidence.
fn transform_shifts_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    fns_with_evidence: &HashSet<Symbol>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    // Module body should always have exactly one block
    assert!(
        blocks.len() <= 1,
        "ICE: Module body should have at most one block, found {}",
        blocks.len()
    );

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    let mut changed = false;
    let new_ops: IdVec<Operation<'db>> = entry_block
        .operations(db)
        .iter()
        .map(|op| {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let func_name = func_op.sym_name(db);
                if fns_with_evidence.contains(&func_name) {
                    let (new_func, func_changed) =
                        transform_shifts_in_function(db, func_op.as_operation());
                    if func_changed {
                        changed = true;
                        return new_func;
                    }
                }
            }
            *op
        })
        .collect();

    if !changed {
        return module;
    }

    let new_entry_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        new_ops,
    );

    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_entry_block]));
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Transform cont.shift operations in a single function.
fn transform_shifts_in_function<'db>(
    db: &'db dyn salsa::Database,
    func_op: Operation<'db>,
) -> (Operation<'db>, bool) {
    let Ok(func) = func::Func::from_operation(db, func_op) else {
        return (func_op, false);
    };

    let func_body = func.body(db);
    let blocks = func_body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return (func_op, false);
    };

    // Get evidence value from first block argument
    let ev_value = entry_block.arg(db, 0);

    let mut changed = false;
    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .map(|block| {
            // Precompute handled abilities for all prompt tags in this block
            let handled_by_tag = collect_handled_abilities_by_tag(db, block);
            let (new_block, block_changed) =
                transform_shifts_in_block(db, block, ev_value, &handled_by_tag);
            if block_changed {
                changed = true;
            }
            new_block
        })
        .collect();

    if !changed {
        return (func_op, false);
    }

    let new_body = Region::new(db, func_body.location(db), new_blocks);
    let new_func = func::func(
        db,
        func_op.location(db),
        func.sym_name(db),
        func.r#type(db),
        new_body,
    );
    (new_func.as_operation(), true)
}

/// Transform cont.shift operations in a block.
fn transform_shifts_in_block<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    ev_value: Value<'db>,
    handled_by_tag: &HashMap<u32, Vec<Type<'db>>>,
) -> (Block<'db>, bool) {
    transform_shifts_in_block_with_remap(db, block, ev_value, handled_by_tag, ImHashMap::new())
}

/// Transform shifts in a block with an initial value remap.
///
/// The `initial_remap` provides a seed for the value_map, allowing operand
/// references to be remapped before processing. This is used when processing
/// push_prompt body/handlers regions to remap the outer evidence to the
/// extended evidence.
fn transform_shifts_in_block_with_remap<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    ev_value: Value<'db>,
    handled_by_tag: &HashMap<u32, Vec<Type<'db>>>,
    initial_remap: ImHashMap<Value<'db>, Value<'db>>,
) -> (Block<'db>, bool) {
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    let mut changed = !initial_remap.is_empty(); // If we have remaps, consider it a change
    let mut value_map: ImHashMap<Value<'db>, Value<'db>> = initial_remap;

    for op in block.operations(db).iter() {
        // Remap operands using value map
        let remapped_operands: Vec<Value<'db>> = op
            .operands(db)
            .iter()
            .map(|v| *value_map.get(v).unwrap_or(v))
            .collect();

        // Check if this is a cont.push_prompt - transform to use evidence_extend
        if let Ok(push_prompt_op) = cont::PushPrompt::from_operation(db, *op) {
            let location = op.location(db);
            let tag = push_prompt_op.tag(db);

            // Look up abilities handled by this handler from the precomputed map
            let abilities = handled_by_tag.get(&tag).cloned().unwrap_or_default();

            // If no abilities found, just recurse into regions without evidence_extend
            if abilities.is_empty() {
                let body_region = push_prompt_op.body(db);
                let handlers_region = push_prompt_op.handlers(db);
                let (new_body, body_changed) = transform_shifts_in_region_with_remap(
                    db,
                    &body_region,
                    ev_value,
                    value_map.clone(),
                );
                let (new_handlers, handlers_changed) = transform_shifts_in_region_with_remap(
                    db,
                    &handlers_region,
                    ev_value,
                    value_map.clone(),
                );

                if body_changed || handlers_changed {
                    changed = true;
                    let result_ty = op
                        .results(db)
                        .first()
                        .copied()
                        .unwrap_or_else(|| *core::Nil::new(db));
                    let new_push_prompt =
                        cont::push_prompt(db, location, result_ty, tag, new_body, new_handlers);
                    let new_op = new_push_prompt.as_operation();
                    if !op.results(db).is_empty() {
                        value_map.insert(op.result(db, 0), new_op.result(db, 0));
                    }
                    new_ops.push(new_op);
                } else {
                    new_ops.push(*op);
                }
                continue;
            }

            // Generate evidence_extend calls for each ability
            let i32_ty = core::I32::new(db).as_type();
            let prompt_tag_ty = ability::PromptTag::new(db).as_type();
            let evidence_ty = ability::EvidencePtr::new(db).as_type();

            // Create tag constant
            let tag_const = trunk_ir::dialect::arith::r#const(
                db,
                location,
                prompt_tag_ty,
                Attribute::IntBits(tag as u64),
            );
            let tag_val = tag_const.result(db);
            new_ops.push(tag_const.as_operation());

            // Extend evidence for each ability
            let mut current_ev = ev_value;
            for ability_ref in &abilities {
                let ability_id = compute_ability_id(db, *ability_ref);

                // Create ability_id constant
                let ability_id_const = trunk_ir::dialect::arith::r#const(
                    db,
                    location,
                    i32_ty,
                    Attribute::IntBits(ability_id as u64),
                );
                let ability_id_val = ability_id_const.result(db);
                new_ops.push(ability_id_const.as_operation());

                // Call __tribute_evidence_extend
                let extend_call = func::call(
                    db,
                    location,
                    vec![current_ev, ability_id_val, tag_val],
                    evidence_ty,
                    Symbol::new("__tribute_evidence_extend"),
                );
                current_ev = extend_call.result(db);
                new_ops.push(extend_call.as_operation());
            }

            // Transform body region with new evidence.
            // We need to create a remap from the original evidence (ev_value) to the
            // extended evidence (current_ev) so that any operands in the body/handlers
            // that reference ev_value will be correctly remapped to current_ev.
            let body_region = push_prompt_op.body(db);
            let handlers_region = push_prompt_op.handlers(db);
            let mut evidence_remap: ImHashMap<Value<'db>, Value<'db>> = value_map.clone();
            if ev_value != current_ev {
                evidence_remap.insert(ev_value, current_ev);
            }
            let (new_body, _) = transform_shifts_in_region_with_remap(
                db,
                &body_region,
                current_ev,
                evidence_remap.clone(),
            );
            let (new_handlers, _) = transform_shifts_in_region_with_remap(
                db,
                &handlers_region,
                current_ev,
                evidence_remap,
            );

            // Create new push_prompt with transformed regions
            let result_ty = op
                .results(db)
                .first()
                .copied()
                .unwrap_or_else(|| *core::Nil::new(db));
            let new_push_prompt =
                cont::push_prompt(db, location, result_ty, tag, new_body, new_handlers);
            let new_op = new_push_prompt.as_operation();

            if !op.results(db).is_empty() {
                value_map.insert(op.result(db, 0), new_op.result(db, 0));
            }
            new_ops.push(new_op);
            changed = true;
            continue;
        }

        // Check if this is a cont.shift that needs tag resolution
        // cont.shift now takes tag as first operand - we replace it with evidence lookup
        if let Ok(shift_op) = cont::Shift::from_operation(db, *op) {
            let location = op.location(db);
            let ability_ref = shift_op.ability_ref(db);
            let op_name = shift_op.op_name(db);

            // Generate ability ID from ability reference (use hash of ability name)
            let ability_id = compute_ability_id(db, ability_ref);

            // Create evidence lookup call
            let marker_ty = ability::Marker::new(db).as_type();
            let i32_ty = core::I32::new(db).as_type();

            // %ability_id_const = arith.const ability_id
            let ability_id_const = trunk_ir::dialect::arith::r#const(
                db,
                location,
                i32_ty,
                Attribute::IntBits(ability_id as u64),
            );
            let ability_id_val = ability_id_const.result(db);
            new_ops.push(ability_id_const.as_operation());

            // %marker = func.call @__tribute_evidence_lookup(%ev, %ability_id)
            let lookup_call = func::call(
                db,
                location,
                vec![ev_value, ability_id_val],
                marker_ty,
                Symbol::new("__tribute_evidence_lookup"),
            );
            let marker_val = lookup_call.result(db);
            new_ops.push(lookup_call.as_operation());

            // %tag = func.call @__tribute_marker_prompt(%marker)
            let prompt_tag_ty = ability::PromptTag::new(db).as_type();
            let prompt_call = func::call(
                db,
                location,
                vec![marker_val],
                prompt_tag_ty,
                Symbol::new("__tribute_marker_prompt"),
            );
            let tag_val = prompt_call.result(db);
            new_ops.push(prompt_call.as_operation());

            // Get result type
            let result_ty = op
                .results(db)
                .first()
                .copied()
                .unwrap_or_else(|| *core::Nil::new(db));

            // Transform regions
            let handler_region = shift_op.handler(db);
            let (new_handler_region, _) = transform_shifts_in_region_with_remap(
                db,
                &handler_region,
                ev_value,
                value_map.clone(),
            );

            // Get value operands (skip the old tag operand at index 0)
            let value_operands: Vec<Value<'db>> =
                remapped_operands.iter().skip(1).copied().collect();

            // Create new cont.shift with the resolved tag
            let new_shift = cont::shift(
                db,
                location,
                tag_val,
                value_operands,
                result_ty,
                ability_ref,
                op_name,
                new_handler_region,
            );

            // Map old result to new result
            if !op.results(db).is_empty() {
                let old_result = op.result(db, 0);
                let new_result = new_shift.as_operation().result(db, 0);
                value_map.insert(old_result, new_result);
            }

            new_ops.push(new_shift.as_operation());
            changed = true;
            continue;
        }

        // Recursively transform nested regions
        let regions = op.regions(db);
        if !regions.is_empty() {
            let mut region_changed = false;
            let new_regions: IdVec<Region<'db>> = regions
                .iter()
                .map(|region| {
                    let (new_region, r_changed) = transform_shifts_in_region_with_remap(
                        db,
                        region,
                        ev_value,
                        value_map.clone(),
                    );
                    if r_changed {
                        region_changed = true;
                    }
                    new_region
                })
                .collect();

            if region_changed {
                changed = true;
                let new_op = op
                    .modify(db)
                    .operands(IdVec::from(remapped_operands))
                    .regions(new_regions)
                    .build();

                // Map old results to new results
                for i in 0..op.results(db).len() {
                    let old_result = op.result(db, i);
                    let new_result = new_op.result(db, i);
                    value_map.insert(old_result, new_result);
                }
                new_ops.push(new_op);
                continue;
            }
        }

        // If operands were remapped, rebuild the operation
        let operands_changed = op
            .operands(db)
            .iter()
            .zip(remapped_operands.iter())
            .any(|(old, new)| old != new);

        if operands_changed {
            let new_op = op
                .modify(db)
                .operands(IdVec::from(remapped_operands))
                .build();

            for i in 0..op.results(db).len() {
                let old_result = op.result(db, i);
                let new_result = new_op.result(db, i);
                value_map.insert(old_result, new_result);
            }
            new_ops.push(new_op);
        } else {
            new_ops.push(*op);
        }
    }

    let new_block = Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops.into_iter().collect(),
    );

    (new_block, changed)
}

/// Transform shifts in a region with an initial value remap.
///
/// The `initial_remap` is used to remap operand references before processing.
/// This is needed for push_prompt body/handlers regions where the outer evidence
/// (`ev_value`) needs to be remapped to the extended evidence (`current_ev`)
/// created by `evidence_extend`.
fn transform_shifts_in_region_with_remap<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    ev_value: Value<'db>,
    initial_remap: ImHashMap<Value<'db>, Value<'db>>,
) -> (Region<'db>, bool) {
    let mut changed = false;
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            // Precompute handled abilities for each block in the region
            let handled_by_tag = collect_handled_abilities_by_tag(db, block);
            let (new_block, block_changed) = transform_shifts_in_block_with_remap(
                db,
                block,
                ev_value,
                &handled_by_tag,
                initial_remap.clone(),
            );
            if block_changed {
                changed = true;
            }
            new_block
        })
        .collect();

    let new_region = Region::new(db, region.location(db), new_blocks);
    (new_region, changed)
}

/// Precompute handled abilities for all prompt tags in a block.
///
/// Scans the block once for all handler_dispatch operations and builds a map
/// from tag to the list of abilities handled by that handler. This avoids
/// O(n²) scanning when processing multiple push_prompt operations in the same block.
fn collect_handled_abilities_by_tag<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
) -> HashMap<u32, Vec<Type<'db>>> {
    let mut map: HashMap<u32, Vec<Type<'db>>> = HashMap::new();

    for op in block.operations(db).iter() {
        if let Ok(dispatch_op) = cont::HandlerDispatch::from_operation(db, *op) {
            let tag = dispatch_op.tag(db);
            let mut abilities = Vec::new();

            // Extract ability_ref from body blocks (skip block 0 which is the "done" case)
            let body = dispatch_op.body(db);
            for (i, body_block) in body.blocks(db).iter().enumerate() {
                if i == 0 {
                    continue; // Skip "done" block
                }

                // Check block arguments for ability_ref attribute
                for arg in body_block.args(db).iter() {
                    if let Some(Attribute::Type(ability_ref)) =
                        arg.get_attr(db, Symbol::new("ability_ref"))
                    {
                        // Avoid duplicates
                        if !abilities.contains(ability_ref) {
                            abilities.push(*ability_ref);
                        }
                    }
                }
            }

            map.insert(tag, abilities);
        }
    }

    map
}

/// Validate that no unresolved cont.shift operations remain in the module.
///
/// This function scans all cont.shift operations in the module and panics if any
/// have the sentinel tag value (u32::MAX), indicating they were not transformed
/// by the evidence pass.
fn validate_no_unresolved_shifts(db: &dyn salsa::Database, module: &core::Module<'_>) {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                validate_no_unresolved_shifts_in_region(db, &func_op.body(db));
            }
        }
    }
}

/// Recursively validate a region for unresolved shifts.
fn validate_no_unresolved_shifts_in_region(db: &dyn salsa::Database, region: &Region<'_>) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Check if this is a cont.shift with the sentinel tag in its first operand
            if let Ok(shift_op) = cont::Shift::from_operation(db, *op) {
                // Get the tag operand (first operand)
                let tag_operand = shift_op.tag(db);

                // Check if the tag operand is a constant with the sentinel value
                // ValueDef::OpResult contains the defining operation; index is checked separately
                if let trunk_ir::ValueDef::OpResult(def_op) = tag_operand.def(db)
                    && tag_operand.index(db) == 0
                    && let Ok(const_op) =
                        trunk_ir::dialect::arith::Const::from_operation(db, def_op)
                    && let Attribute::IntBits(value) = const_op.value(db)
                    && value == UNRESOLVED_SHIFT_TAG as u64
                {
                    let ability_ref = shift_op.ability_ref(db);
                    let op_name = shift_op.op_name(db);
                    panic!(
                        "ICE: Unresolved cont.shift found after evidence pass.\n\
                         Ability: {:?}, Op: {:?}\n\
                         This indicates the shift was not inside a function with evidence parameter.",
                        ability_ref, op_name
                    );
                }
            }

            // Recursively check nested regions
            for region in op.regions(db).iter() {
                validate_no_unresolved_shifts_in_region(db, region);
            }
        }
    }
}

/// Compute a stable ability ID from an ability reference type.
///
/// This generates a consistent integer ID from the ability name and type parameters
/// for use in evidence lookup. The ID must be deterministic so that push_prompt and
/// shift use the same ID for the same ability.
///
/// # Panics
/// Panics if the type is not a valid ability reference type. This indicates a
/// compiler internal error (ICE) - ability types should always be well-formed
/// by the time they reach this pass.
fn compute_ability_id(db: &dyn salsa::Database, ability_ref: Type<'_>) -> u32 {
    // Extract ability type - panic if not valid
    let ability_type = core::AbilityRefType::from_type(db, ability_ref).unwrap_or_else(|| {
        panic!(
            "ICE: compute_ability_id called with non-ability type: {:?}",
            ability_ref
        )
    });

    let ability_name = ability_type.name(db).unwrap_or_else(|| {
        panic!(
            "ICE: AbilityRefType has no name attribute: {:?}",
            ability_ref
        )
    });

    // Start hash with the ability name
    let mut hash: u32 = ability_name.with_str(|s| {
        let mut h: u32 = 0;
        for byte in s.bytes() {
            h = h.wrapping_mul(31).wrapping_add(byte as u32);
        }
        h
    });

    // Include type parameters in the hash for parameterized abilities like State(Int)
    for param in ability_type.params(db) {
        // Hash the type by including its dialect, name, and recursively its params
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_add(hash_type(db, *param));
    }

    hash
}

/// Helper to hash a type for ability ID computation.
fn hash_type(db: &dyn salsa::Database, ty: Type<'_>) -> u32 {
    let mut hash: u32 = 0;

    // Hash dialect
    let dialect = ty.dialect(db);
    dialect.with_str(|s| {
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
    });

    // Hash name
    let name = ty.name(db);
    name.with_str(|s| {
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
    });

    // Recursively hash params
    for param in ty.params(db) {
        hash = hash.wrapping_mul(37);
        hash = hash.wrapping_add(hash_type(db, *param));
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{BlockId, Location, PathId, Span, ValueDef, idvec};

    #[salsa_test]
    fn test_compute_ability_id(db: &salsa::DatabaseImpl) {
        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        let console_ref = core::AbilityRefType::simple(db, Symbol::new("Console"));

        let state_id = compute_ability_id(db, state_ref.as_type());
        let console_id = compute_ability_id(db, console_ref.as_type());

        // Same ability should have same ID
        let state_ref2 = core::AbilityRefType::simple(db, Symbol::new("State"));
        let state_id2 = compute_ability_id(db, state_ref2.as_type());
        assert_eq!(state_id, state_id2);

        // Different abilities should have different IDs
        assert_ne!(state_id, console_id);
    }

    /// Helper to create test location
    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Test that transform_shifts_in_block correctly remaps ev_value to current_ev
    /// in push_prompt body when abilities are handled.
    ///
    /// This tracked function contains all the test logic because Salsa requires
    /// tracked struct creation to happen within tracked functions.
    ///
    /// Returns Ok(()) if the test passes, Err(message) if it fails.
    #[salsa::tracked]
    fn run_evidence_remap_test(db: &dyn salsa::Database) -> Result<(), String> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();
        let evidence_ty = ability::EvidencePtr::new(db).as_type();

        // Entry block with evidence parameter
        let entry_block_id = BlockId::fresh();
        let ev_arg = BlockArg::of_type(db, evidence_ty);

        // Create a func.call inside push_prompt body that uses the evidence
        // This simulates: func.call @effectful_fn(%ev)
        let ev_value = Value::new(db, ValueDef::BlockArg(entry_block_id), 0);

        // Create the call operation that uses ev_value as operand
        let inner_call = func::call(
            db,
            location,
            vec![ev_value], // Uses original evidence
            i64_ty,
            Symbol::new("effectful_fn"),
        );

        // Create yield operation
        let yield_op = trunk_ir::dialect::scf::r#yield(db, location, vec![inner_call.result(db)]);

        // Create the body region for push_prompt
        let body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![inner_call.as_operation(), yield_op.as_operation()],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // Create empty handlers region
        let handlers_region = Region::new(db, location, idvec![]);

        // Create push_prompt with tag 0
        let push_prompt = cont::push_prompt(db, location, i64_ty, 0, body_region, handlers_region);

        // Create handler_dispatch that handles an ability for tag 0
        // This is needed so that evidence_extend is generated
        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        let handler_block_arg = BlockArg::with_attr(
            db,
            core::Nil::new(db).as_type(),
            Symbol::new("ability_ref"),
            Attribute::Type(state_ref.as_type()),
        );
        let handler_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![handler_block_arg],
            idvec![],
        );
        let done_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![BlockArg::of_type(db, i64_ty)],
            idvec![],
        );
        let dispatch_body = Region::new(db, location, idvec![done_block, handler_block]);
        let handler_dispatch = cont::handler_dispatch(
            db,
            location,
            push_prompt.result(db),
            i64_ty,
            0,
            i64_ty,
            dispatch_body,
        );

        // Create return
        let ret = func::r#return(db, location, Some(handler_dispatch.result(db)));

        // Create the entry block
        let entry_block = Block::new(
            db,
            entry_block_id,
            location,
            idvec![ev_arg],
            idvec![
                push_prompt.as_operation(),
                handler_dispatch.as_operation(),
                ret.as_operation()
            ],
        );

        // Now call transform_shifts_in_block with this block
        let handled_by_tag = collect_handled_abilities_by_tag(db, &entry_block);

        // Verify that tag 0 has State ability
        if !handled_by_tag.contains_key(&0) {
            return Err("Should find handler for tag 0".to_string());
        }
        let abilities = handled_by_tag.get(&0).unwrap();
        if abilities.len() != 1 {
            return Err(format!("Should have one ability, got {}", abilities.len()));
        }

        // Run the transformation
        let (new_block, changed) =
            transform_shifts_in_block(db, &entry_block, ev_value, &handled_by_tag);

        if !changed {
            return Err("Block should be transformed".to_string());
        }

        // Find the new push_prompt in the transformed block
        let mut found_evidence_extend = false;
        let mut new_push_prompt_op = None;

        for op in new_block.operations(db).iter() {
            if let Ok(call) = func::Call::from_operation(db, *op)
                && call.callee(db) == "__tribute_evidence_extend"
            {
                found_evidence_extend = true;
            }
            if cont::PushPrompt::from_operation(db, *op).is_ok() {
                new_push_prompt_op = Some(*op);
            }
        }

        if !found_evidence_extend {
            return Err("Should generate evidence_extend call".to_string());
        }

        let Some(push_prompt_op) = new_push_prompt_op else {
            return Err("Should find new push_prompt".to_string());
        };
        let new_push_prompt = cont::PushPrompt::from_operation(db, push_prompt_op).unwrap();
        let new_body = new_push_prompt.body(db);

        // Find the func.call in the new body
        let new_body_block = new_body.blocks(db).first().unwrap();
        let mut inner_call_operands = None;

        for op in new_body_block.operations(db).iter() {
            if let Ok(call) = func::Call::from_operation(db, *op)
                && call.callee(db) == "effectful_fn"
            {
                inner_call_operands = Some(op.operands(db).to_vec());
            }
        }

        let Some(operands) = inner_call_operands else {
            return Err("Should find effectful_fn call in body".to_string());
        };

        // THE KEY ASSERTION:
        // The evidence operand should NOT be the original ev_value (block arg 0)
        // It should be the result of evidence_extend (or remapped to current_ev)
        if operands.is_empty() {
            return Err("Call should have evidence operand".to_string());
        }

        let evidence_operand = operands[0];

        // The original ev_value is BlockArg(entry_block_id, 0)
        // If the bug exists, evidence_operand == ev_value
        // If fixed, evidence_operand should be the result of evidence_extend
        if evidence_operand == ev_value {
            return Err(format!(
                "BUG CONFIRMED: Handler body effectful call still uses original evidence \
                 parameter ({:?}) instead of extended evidence. \
                 The ev_value -> current_ev remap is missing in transform_shifts_in_region.",
                ev_value
            ));
        }

        Ok(())
    }

    /// Test that transform_shifts_in_block correctly remaps ev_value to current_ev
    /// in push_prompt body when abilities are handled.
    ///
    /// This test verifies the fix for the evidence remap issue where effectful
    /// function calls inside handler body/handlers regions need to use the
    /// extended evidence (from evidence_extend) rather than the original evidence.
    #[salsa_test]
    fn test_push_prompt_body_remaps_evidence(db: &salsa::DatabaseImpl) {
        let result = run_evidence_remap_test(db);
        if let Err(msg) = result {
            panic!("{}", msg);
        }
    }
}
