//! Evidence-based dispatch resolution pass.
//!
//! This pass transforms static `cont.shift` operations into evidence-based dispatch
//! using runtime function calls.
//!
//! ## Transformations
//!
//! ```text
//! // Before (static placeholder tag)
//! %result = cont.shift(args...) { tag: 0, ability_ref, op_name }
//!
//! // After (dynamic evidence-based tag)
//! %marker = func.call @__tribute_evidence_lookup(%ev, ability_id)
//! %tag = func.call @__tribute_marker_prompt(%marker)
//! %result = cont.shift_dynamic(%tag, args...) { ability_ref, op_name }
//! ```
//!
//! This pass must run AFTER `add_evidence_params` so that effectful functions
//! have evidence as their first parameter.

use std::collections::{HashMap, HashSet};

use tribute_ir::dialect::ability;
use trunk_ir::dialect::{cont, core, func};
use trunk_ir::{
    Attribute, Block, BlockArg, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type,
    Value,
};

/// Resolve evidence-based dispatch for `cont.shift` operations.
///
/// Transforms `cont.shift` with static tags into `cont.shift_dynamic` with
/// evidence lookup. This enables proper handler dispatch at runtime.
#[salsa::tracked]
pub fn resolve_evidence_dispatch<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    // First, ensure runtime helper functions are declared
    let module = ensure_runtime_functions(db, module);

    // Collect functions with evidence parameters
    let fns_with_evidence = collect_functions_with_evidence(db, &module);

    if fns_with_evidence.is_empty() {
        return module;
    }

    // Transform shifts in functions with evidence
    transform_shifts_in_module(db, module, &fns_with_evidence)
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
            let (new_block, block_changed) = transform_shifts_in_block(db, block, ev_value);
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
) -> (Block<'db>, bool) {
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    let mut changed = false;
    let mut value_map: HashMap<Value<'db>, Value<'db>> = HashMap::new();

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

            // Find abilities handled by this handler by scanning forward for handler_dispatch
            let abilities = collect_handled_abilities(db, block, tag);

            // If no abilities found, just recurse into regions without evidence_extend
            if abilities.is_empty() {
                let body_region = push_prompt_op.body(db);
                let handlers_region = push_prompt_op.handlers(db);
                let (new_body, body_changed) =
                    transform_shifts_in_region(db, &body_region, ev_value);
                let (new_handlers, handlers_changed) =
                    transform_shifts_in_region(db, &handlers_region, ev_value);

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

            // Transform body region with new evidence
            let body_region = push_prompt_op.body(db);
            let handlers_region = push_prompt_op.handlers(db);
            let (new_body, _) = transform_shifts_in_region(db, &body_region, current_ev);
            let (new_handlers, _) = transform_shifts_in_region(db, &handlers_region, current_ev);

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

        // Check if this is a cont.shift
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
            let (new_handler_region, _) = transform_shifts_in_region(db, &handler_region, ev_value);

            // Create cont.shift_dynamic
            let shift_dynamic = cont::shift_dynamic(
                db,
                location,
                tag_val,
                remapped_operands,
                result_ty,
                ability_ref,
                op_name,
                new_handler_region,
            );

            // Map old result to new result
            if !op.results(db).is_empty() {
                let old_result = op.result(db, 0);
                let new_result = shift_dynamic.as_operation().result(db, 0);
                value_map.insert(old_result, new_result);
            }

            new_ops.push(shift_dynamic.as_operation());
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
                    let (new_region, r_changed) = transform_shifts_in_region(db, region, ev_value);
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

/// Transform cont.shift operations in a region.
fn transform_shifts_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    ev_value: Value<'db>,
) -> (Region<'db>, bool) {
    let mut changed = false;
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let (new_block, block_changed) = transform_shifts_in_block(db, block, ev_value);
            if block_changed {
                changed = true;
            }
            new_block
        })
        .collect();

    let new_region = Region::new(db, region.location(db), new_blocks);
    (new_region, changed)
}

/// Collect abilities handled by a push_prompt by finding the associated handler_dispatch.
///
/// Scans the block for a handler_dispatch with the same tag and extracts ability_ref
/// attributes from its body blocks.
fn collect_handled_abilities<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    tag: u32,
) -> Vec<Type<'db>> {
    let mut abilities = Vec::new();

    for op in block.operations(db).iter() {
        // Look for handler_dispatch with matching tag
        if let Ok(dispatch_op) = cont::HandlerDispatch::from_operation(db, *op) {
            if dispatch_op.tag(db) != tag {
                continue;
            }

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
            break; // Found the matching handler_dispatch
        }
    }

    abilities
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
    use salsa::Database;

    #[test]
    fn test_compute_ability_id() {
        let db = salsa::DatabaseImpl::default();
        db.attach(|db| {
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
        });
    }
}
