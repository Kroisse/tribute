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
//! %tag = adt.struct_get(%marker, 1)  // field 1 = prompt_tag
//! %result = cont.shift(%tag, args...) { ability_ref, op_name }
//! ```
//!
//! This pass must run AFTER `add_evidence_params` so that effectful functions
//! have evidence as their first parameter.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use im::HashMap as ImHashMap;
use tribute_ir::dialect::ability;
use trunk_ir::dialect::{adt, cont, core, func};
use trunk_ir::{
    Attribute, Block, BlockArg, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type,
    Value,
};

// ============================================================================
// OpTable Registry - assigns unique indices to handler dispatch points
// ============================================================================

/// Entry in the operation table registry.
/// Each entry represents a handler (push_prompt) with its handled abilities and operations.
#[derive(Debug, Clone)]
pub struct OpTableEntry<'db> {
    /// Abilities handled by this handler
    pub abilities: Vec<Type<'db>>,
    /// Operations in this handler, ordered by (ability_ref, op_name)
    /// The order determines op_offset for each operation
    pub operations: Vec<(Type<'db>, Symbol)>,
    /// Location of the handler for debugging
    pub location: trunk_ir::Location<'db>,
}

/// Registry for managing op_table_index assignments.
/// Thread-safe via RefCell for use during IR transformation.
#[derive(Debug, Default)]
pub struct OpTableRegistry<'db> {
    entries: Vec<OpTableEntry<'db>>,
}

impl<'db> OpTableRegistry<'db> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a handler and return its op_table_index.
    ///
    /// The abilities list specifies which abilities this handler handles.
    /// Operations are extracted from the handler dispatch body.
    pub fn register(
        &mut self,
        abilities: Vec<Type<'db>>,
        operations: Vec<(Type<'db>, Symbol)>,
        location: trunk_ir::Location<'db>,
    ) -> u32 {
        let index = self.entries.len() as u32;
        self.entries.push(OpTableEntry {
            abilities,
            operations,
            location,
        });
        index
    }

    /// Get the number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by index.
    pub fn get(&self, index: u32) -> Option<&OpTableEntry<'db>> {
        self.entries.get(index as usize)
    }

    /// Get all entries.
    pub fn entries(&self) -> &[OpTableEntry<'db>] {
        &self.entries
    }

    /// Compute the op_offset for a given ability and operation within this registry.
    ///
    /// Returns the index of the operation within the handler's operation list,
    /// or None if the operation is not found.
    pub fn compute_op_offset(
        &self,
        op_table_index: u32,
        ability_ref: Type<'db>,
        op_name: Symbol,
    ) -> Option<u32> {
        let entry = self.get(op_table_index)?;
        entry
            .operations
            .iter()
            .position(|(a, o)| *a == ability_ref && *o == op_name)
            .map(|pos| pos as u32)
    }
}

/// Shared reference to OpTableRegistry for use during IR transformation.
pub type SharedOpTableRegistry<'db> = Rc<RefCell<OpTableRegistry<'db>>>;

/// Sentinel value used for unresolved cont.shift tags.
/// When a shift is generated without an enclosing handler, this value is used
/// as a placeholder. The evidence pass should transform all such shifts.
pub const UNRESOLVED_SHIFT_TAG: u32 = u32::MAX;

/// Resolve evidence-based dispatch for `cont.shift` operations.
///
/// Transforms `cont.shift` with placeholder tags into `cont.shift` with
/// dynamically resolved tags via evidence lookup. This enables proper
/// handler dispatch at runtime.
///
/// Returns the transformed module along with the OpTableRegistry containing
/// information about all registered handlers and their operations.
#[salsa::tracked]
pub fn resolve_evidence_dispatch<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    resolve_evidence_dispatch_with_registry(db, module).0
}

/// Resolve evidence-based dispatch and return the OpTableRegistry.
///
/// This is the internal implementation that also returns the registry,
/// which can be used by downstream passes for table-based dispatch.
pub fn resolve_evidence_dispatch_with_registry<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> (core::Module<'db>, OpTableRegistry<'db>) {
    // Create shared registry for this pass
    let registry: SharedOpTableRegistry<'db> = Rc::new(RefCell::new(OpTableRegistry::new()));

    // First, ensure runtime helper functions are declared
    let module = ensure_runtime_functions(db, module);

    // Validate no ability ID hash collisions before proceeding
    validate_ability_id_uniqueness(db, &module);

    // Collect functions with evidence parameters
    let fns_with_evidence = collect_functions_with_evidence(db, &module);

    // Handle "handler-root" functions first: functions that contain cont.push_prompt
    // but do not have evidence as their first parameter. These are top-level handler
    // functions (e.g., main) that need to create initial empty evidence.
    // This must run BEFORE transform_shifts so the OpTableRegistry is populated
    // when shifts look up op_offset.
    let handler_root_fns = collect_handler_root_functions(db, &module, &fns_with_evidence);
    let module = if handler_root_fns.is_empty() {
        module
    } else {
        transform_handler_roots_in_module(
            db,
            module,
            &handler_root_fns,
            &fns_with_evidence,
            Rc::clone(&registry),
        )
    };

    // Transform shifts in functions with evidence (if any).
    // Registry is now populated by handler-root processing above.
    let module = if fns_with_evidence.is_empty() {
        module
    } else {
        transform_shifts_in_module(db, module, &fns_with_evidence, Rc::clone(&registry))
    };

    // Validate that all shifts have been resolved (no sentinel tags remain).
    // This must run regardless of whether evidence functions exist, to catch
    // any unresolved shifts that may have been introduced incorrectly.
    validate_no_unresolved_shifts(db, &module);

    // Extract the registry from the RefCell
    let final_registry = Rc::try_unwrap(registry)
        .expect("OpTableRegistry should have no other references")
        .into_inner();

    (module, final_registry)
}

/// Ensure runtime helper functions are declared in the module.
///
/// These functions are:
/// - `__tribute_evidence_lookup(ev: Evidence, ability_id: i32) -> Marker`
/// - `__tribute_evidence_extend(ev: Evidence, marker: Marker) -> Evidence`
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
    let mut has_evidence_extend = false;

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            let name = func_op.sym_name(db);
            if name == Symbol::new("__tribute_evidence_lookup") {
                has_evidence_lookup = true;
            } else if name == Symbol::new("__tribute_evidence_extend") {
                has_evidence_extend = true;
            }
        }
    }

    if has_evidence_lookup && has_evidence_extend {
        return module;
    }

    // Add missing function declarations
    let location = module.location(db);
    let mut new_ops: Vec<Operation<'db>> = entry_block.operations(db).iter().copied().collect();

    if !has_evidence_lookup {
        let evidence_ty = ability::evidence_adt_type(db);
        let i32_ty = core::I32::new(db).as_type();
        let marker_ty = ability::marker_adt_type(db);

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

    if !has_evidence_extend {
        let evidence_ty = ability::evidence_adt_type(db);
        let marker_ty = ability::marker_adt_type(db);

        // fn __tribute_evidence_extend(ev: Evidence, marker: Marker) -> Evidence
        let func_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, marker_ty]), evidence_ty);

        // Empty body with unreachable
        let unreachable_op = func::unreachable(db, location);
        let body_block = Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            location,
            IdVec::from(vec![
                BlockArg::of_type(db, evidence_ty),
                BlockArg::of_type(db, marker_ty),
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

/// Collect all function names that have evidence (`core.array(Marker)`) as their first parameter.
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
                    if !params.is_empty() && ability::is_evidence_type(db, params[0]) {
                        fns_with_evidence.insert(func_op.sym_name(db));
                    }
                }
            }
        }
    }

    fns_with_evidence
}

/// Collect functions that contain `cont.push_prompt` but do NOT have evidence
/// as their first parameter. These are "handler-root" functions (e.g., `main`)
/// that establish handlers without being effectful themselves.
fn collect_handler_root_functions<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
    fns_with_evidence: &HashSet<Symbol>,
) -> HashSet<Symbol> {
    let mut handler_roots = HashSet::new();

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let func_name = func_op.sym_name(db);
                // Skip functions that already have evidence (handled by transform_shifts_in_module)
                if fns_with_evidence.contains(&func_name) {
                    continue;
                }
                // Check if the function body contains cont.push_prompt
                if function_contains_push_prompt(db, &func_op) {
                    handler_roots.insert(func_name);
                }
            }
        }
    }

    handler_roots
}

/// Check if a function contains any `cont.push_prompt` operation in its body.
fn function_contains_push_prompt<'db>(
    db: &'db dyn salsa::Database,
    func_op: &func::Func<'db>,
) -> bool {
    let body = func_op.body(db);
    region_contains_push_prompt(db, &body)
}

/// Recursively check if a region contains `cont.push_prompt`.
fn region_contains_push_prompt(db: &dyn salsa::Database, region: &Region) -> bool {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if cont::PushPrompt::from_operation(db, *op).is_ok() {
                return true;
            }
            // Check nested regions
            for region in op.regions(db).iter() {
                if region_contains_push_prompt(db, region) {
                    return true;
                }
            }
        }
    }
    false
}

/// Transform handler-root functions by creating initial empty evidence.
///
/// For functions that contain `cont.push_prompt` but have no evidence parameter,
/// this creates an empty evidence array at the start of the function body and
/// processes push_prompt operations to use evidence_extend.
fn transform_handler_roots_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    handler_root_fns: &HashSet<Symbol>,
    fns_with_evidence: &HashSet<Symbol>,
    registry: SharedOpTableRegistry<'db>,
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
                if handler_root_fns.contains(&func_name) {
                    let (new_func, func_changed) = transform_handler_root_function(
                        db,
                        func_op.as_operation(),
                        fns_with_evidence,
                        Rc::clone(&registry),
                    );
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

/// Transform a handler-root function by inserting initial empty evidence.
fn transform_handler_root_function<'db>(
    db: &'db dyn salsa::Database,
    func_op: Operation<'db>,
    fns_with_evidence: &HashSet<Symbol>,
    registry: SharedOpTableRegistry<'db>,
) -> (Operation<'db>, bool) {
    let Ok(func_def) = func::Func::from_operation(db, func_op) else {
        return (func_op, false);
    };

    let func_body = func_def.body(db);
    let blocks = func_body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return (func_op, false);
    };

    let location = func_op.location(db);

    // Create empty evidence array: adt.array_new(0) with evidence type
    let evidence_ty = ability::evidence_adt_type(db);
    let i32_ty = core::I32::new(db).as_type();
    let zero_const = trunk_ir::dialect::arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
    let empty_evidence = adt::array_new(
        db,
        location,
        vec![zero_const.result(db)],
        evidence_ty,
        evidence_ty,
    );
    let ev_value = empty_evidence.as_operation().result(db, 0);

    // Transform entry block using the empty evidence.
    // This handles push_prompt -> evidence_extend AND calls to effectful functions.
    let (new_entry_block, block_changed) = transform_block_with_evidence(
        db,
        entry_block,
        ev_value,
        fns_with_evidence,
        ImHashMap::new(),
        Rc::clone(&registry),
    );

    if !block_changed {
        return (func_op, false);
    }

    // Prepend the empty evidence creation operations to the transformed block
    let mut prefix_ops: Vec<Operation<'db>> =
        vec![zero_const.as_operation(), empty_evidence.as_operation()];
    prefix_ops.extend(new_entry_block.operations(db).iter().copied());

    let final_entry_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        prefix_ops.into_iter().collect(),
    );

    // Rebuild function with new body
    let mut new_blocks: Vec<Block<'db>> = vec![final_entry_block];
    for block in blocks.iter().skip(1) {
        new_blocks.push(*block);
    }
    let new_body = Region::new(db, func_body.location(db), new_blocks.into_iter().collect());
    let new_func = func::func(
        db,
        location,
        func_def.sym_name(db),
        func_def.r#type(db),
        new_body,
    );
    (new_func.as_operation(), true)
}

/// If `op` is a `func.call` to an effectful function, prepend `ev_value` as the first argument.
/// Returns the new operation if the call was rewritten, or `None` if not applicable.
fn try_prepend_evidence_to_call<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    remapped_operands: &[Value<'db>],
    ev_value: Value<'db>,
    fns_with_evidence: &HashSet<Symbol>,
    value_map: &mut ImHashMap<Value<'db>, Value<'db>>,
) -> Option<Operation<'db>> {
    let call_op = func::Call::from_operation(db, *op).ok()?;
    let callee = call_op.callee(db);
    if !fns_with_evidence.contains(&callee) {
        return None;
    }
    // Check if evidence is already the first argument
    let first_arg = remapped_operands.first().copied();
    if first_arg == Some(ev_value) {
        return None;
    }
    let location = op.location(db);
    let result_ty = op
        .results(db)
        .first()
        .copied()
        .unwrap_or_else(|| *core::Nil::new(db));
    let mut new_args = vec![ev_value];
    new_args.extend(remapped_operands.iter().copied());
    let new_call = func::call(db, location, new_args, result_ty, callee);
    let new_call_op = new_call.as_operation();
    if !op.results(db).is_empty() {
        value_map.insert(op.result(db, 0), new_call_op.result(db, 0));
    }
    Some(new_call_op)
}

/// If `op` is a `func.call_indirect`, replace the evidence argument (index 1) with `ev_value`.
/// Returns the new operation if the call was rewritten, or `None` if not applicable.
fn try_replace_call_indirect_evidence<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    remapped_operands: &[Value<'db>],
    ev_value: Value<'db>,
    value_map: &mut ImHashMap<Value<'db>, Value<'db>>,
) -> Option<Operation<'db>> {
    func::CallIndirect::from_operation(db, *op).ok()?;
    if remapped_operands.len() < 2 {
        return None;
    }
    let current_ev = remapped_operands[1];
    if current_ev == ev_value {
        return None;
    }
    let location = op.location(db);
    let result_ty = op
        .results(db)
        .first()
        .copied()
        .unwrap_or_else(|| *core::Nil::new(db));
    let table_idx = remapped_operands[0];
    let mut new_args: Vec<Value<'db>> = vec![ev_value];
    new_args.extend(remapped_operands[2..].iter().copied());
    let new_call = func::call_indirect(db, location, table_idx, new_args, result_ty);
    let new_call_op = new_call.as_operation();
    if !op.results(db).is_empty() {
        value_map.insert(op.result(db, 0), new_call_op.result(db, 0));
    }
    Some(new_call_op)
}

/// Transform a block with evidence, handling both shifts and calls.
///
/// Unlike transform_shifts_in_block_with_remap, this also transforms
/// func.call to effectful functions by prepending evidence.
fn transform_block_with_evidence<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    ev_value: Value<'db>,
    fns_with_evidence: &HashSet<Symbol>,
    initial_remap: ImHashMap<Value<'db>, Value<'db>>,
    registry: SharedOpTableRegistry<'db>,
) -> (Block<'db>, bool) {
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    let mut changed = !initial_remap.is_empty();
    let mut value_map: ImHashMap<Value<'db>, Value<'db>> = initial_remap;

    for op in block.operations(db).iter() {
        // Remap operands using value map
        let remapped_operands: Vec<Value<'db>> = op
            .operands(db)
            .iter()
            .map(|v| *value_map.get(v).unwrap_or(v))
            .collect();

        // Handle cont.push_prompt: delegate to existing transform
        if cont::PushPrompt::from_operation(db, *op).is_ok() {
            // Use existing transform for push_prompt but with evidence call support.
            // Build a temporary single-op block for the push_prompt.
            let single_block = Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                IdVec::from(vec![*op]),
            );
            // Collect handled abilities from the ORIGINAL block (not single_block),
            // because handler_dispatch ops are siblings of push_prompt in the parent block.
            let handled = collect_handled_abilities_by_tag(db, block);
            let (transformed, was_changed) = transform_shifts_in_block_with_remap(
                db,
                &single_block,
                ev_value,
                &handled,
                value_map.clone(),
                Rc::clone(&registry),
                fns_with_evidence,
                Some(block),
            );
            if was_changed {
                changed = true;
                for transformed_op in transformed.operations(db).iter() {
                    new_ops.push(*transformed_op);
                }
                // Map old results to new results
                if !op.results(db).is_empty()
                    && let Some(last_op) = transformed.operations(db).last()
                    && !last_op.results(db).is_empty()
                {
                    value_map.insert(op.result(db, 0), last_op.result(db, 0));
                }
                continue;
            }
        }

        // Handle func.call to effectful functions: prepend evidence
        if let Some(new_call_op) = try_prepend_evidence_to_call(
            db,
            op,
            &remapped_operands,
            ev_value,
            fns_with_evidence,
            &mut value_map,
        ) {
            new_ops.push(new_call_op);
            changed = true;
            continue;
        }

        // Handle func.call_indirect: replace evidence argument with current ev_value.
        if let Some(new_call_op) =
            try_replace_call_indirect_evidence(db, op, &remapped_operands, ev_value, &mut value_map)
        {
            new_ops.push(new_call_op);
            changed = true;
            continue;
        }

        // Default: remap operands and recurse into regions
        let regions = op.regions(db);
        if !regions.is_empty() {
            let mut region_changed = false;
            let new_regions: IdVec<Region<'db>> = regions
                .iter()
                .map(|region| {
                    let (new_region, r_changed) = transform_region_with_evidence(
                        db,
                        region,
                        ev_value,
                        fns_with_evidence,
                        value_map.clone(),
                        Rc::clone(&registry),
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
                for i in 0..op.results(db).len() {
                    value_map.insert(op.result(db, i), new_op.result(db, i));
                }
                new_ops.push(new_op);
                continue;
            }
        }

        // If operands were remapped, rebuild
        let operands_changed = op
            .operands(db)
            .iter()
            .zip(remapped_operands.iter())
            .any(|(old, new)| old != new);

        if operands_changed {
            changed = true;
            let new_op = op
                .modify(db)
                .operands(IdVec::from(remapped_operands))
                .build();
            for i in 0..op.results(db).len() {
                value_map.insert(op.result(db, i), new_op.result(db, i));
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

/// Transform a region with evidence for handler-root functions.
fn transform_region_with_evidence<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    ev_value: Value<'db>,
    fns_with_evidence: &HashSet<Symbol>,
    initial_remap: ImHashMap<Value<'db>, Value<'db>>,
    registry: SharedOpTableRegistry<'db>,
) -> (Region<'db>, bool) {
    let mut changed = false;
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let (new_block, block_changed) = transform_block_with_evidence(
                db,
                block,
                ev_value,
                fns_with_evidence,
                initial_remap.clone(),
                Rc::clone(&registry),
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

/// Transform cont.shift operations in all functions with evidence.
fn transform_shifts_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    fns_with_evidence: &HashSet<Symbol>,
    registry: SharedOpTableRegistry<'db>,
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
                    let (new_func, func_changed) = transform_shifts_in_function(
                        db,
                        func_op.as_operation(),
                        Rc::clone(&registry),
                    );
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
    registry: SharedOpTableRegistry<'db>,
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
            let (new_block, block_changed) = transform_shifts_in_block(
                db,
                block,
                ev_value,
                &handled_by_tag,
                Rc::clone(&registry),
            );
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
    registry: SharedOpTableRegistry<'db>,
) -> (Block<'db>, bool) {
    let empty = HashSet::new();
    transform_shifts_in_block_with_remap(
        db,
        block,
        ev_value,
        handled_by_tag,
        ImHashMap::new(),
        registry,
        &empty,
        None,
    )
}

/// Transform shifts in a block with an initial value remap.
///
/// The `initial_remap` provides a seed for the value_map, allowing operand
/// references to be remapped before processing. This is used when processing
/// push_prompt body/handlers regions to remap the outer evidence to the
/// extended evidence.
#[allow(clippy::too_many_arguments)]
fn transform_shifts_in_block_with_remap<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    ev_value: Value<'db>,
    handled_by_tag: &HashMap<u32, Vec<Type<'db>>>,
    initial_remap: ImHashMap<Value<'db>, Value<'db>>,
    registry: SharedOpTableRegistry<'db>,
    fns_with_evidence: &HashSet<Symbol>,
    dispatch_block: Option<&Block<'db>>,
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
                    Rc::clone(&registry),
                    fns_with_evidence,
                );
                let (new_handlers, handlers_changed) = transform_shifts_in_region_with_remap(
                    db,
                    &handlers_region,
                    ev_value,
                    value_map.clone(),
                    Rc::clone(&registry),
                    fns_with_evidence,
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
            let evidence_ty = ability::evidence_adt_type(db);
            let marker_ty = ability::marker_adt_type(db);

            // Collect operations from the handler_dispatch for this tag.
            // Use dispatch_block if provided (for cases where block is a
            // single-op slice that doesn't contain handler_dispatch siblings).
            let operations = collect_operations_for_tag(db, dispatch_block.unwrap_or(block), tag);

            // Register this handler in the OpTableRegistry and get the assigned index
            let op_table_idx =
                registry
                    .borrow_mut()
                    .register(abilities.clone(), operations, location);

            // Create tag constant (prompt_tag for the handler)
            let tag_const = trunk_ir::dialect::arith::r#const(
                db,
                location,
                prompt_tag_ty,
                Attribute::IntBits(tag as u64),
            );
            let tag_val = tag_const.result(db);
            new_ops.push(tag_const.as_operation());

            // Create op_table_index constant with the assigned index from registry
            let op_table_idx_const = trunk_ir::dialect::arith::r#const(
                db,
                location,
                i32_ty,
                Attribute::IntBits(op_table_idx as u64),
            );
            let op_table_idx_val = op_table_idx_const.result(db);
            new_ops.push(op_table_idx_const.as_operation());

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

                // Create Marker struct: { ability_id, prompt_tag, op_table_index }
                let marker_struct = adt::struct_new(
                    db,
                    location,
                    vec![ability_id_val, tag_val, op_table_idx_val],
                    marker_ty,
                    marker_ty,
                );
                let marker_val = marker_struct.result(db);
                new_ops.push(marker_struct.as_operation());

                // Call __tribute_evidence_extend with the Marker
                let extend_call = func::call(
                    db,
                    location,
                    vec![current_ev, marker_val],
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
                Rc::clone(&registry),
                fns_with_evidence,
            );
            let (new_handlers, _) = transform_shifts_in_region_with_remap(
                db,
                &handlers_region,
                current_ev,
                evidence_remap,
                Rc::clone(&registry),
                fns_with_evidence,
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
            let marker_ty = ability::marker_adt_type(db);
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

            // %tag = adt.struct_get(%marker, field=1)  -- prompt_tag is at field index 1
            let prompt_tag_ty = ability::PromptTag::new(db).as_type();
            let struct_get_tag =
                adt::struct_get(db, location, marker_val, prompt_tag_ty, marker_ty, 1);
            let tag_val = struct_get_tag.result(db);
            new_ops.push(struct_get_tag.as_operation());

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
                Rc::clone(&registry),
                fns_with_evidence,
            );

            // Get value operands (skip the old tag operand at index 0)
            let value_operands: Vec<Value<'db>> =
                remapped_operands.iter().skip(1).copied().collect();

            // op_offset is always None: the actual dispatch index is computed at
            // the handler side via hash-based `compute_op_idx(ability, op_name)`.
            // This ensures shift and handler always agree on the op index regardless
            // of handler registration order.
            let op_offset = None;

            // op_table_index is a runtime value (from Marker), so we pass None here.
            // The actual dispatch will use the runtime value from trampoline.get_yield_op_idx.
            let new_shift = cont::shift(
                db,
                location,
                tag_val,
                value_operands,
                result_ty,
                ability_ref,
                op_name,
                None, // op_table_index is dynamic (from Marker at runtime)
                op_offset,
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

        // Handle func.call to effectful functions: prepend evidence
        if !fns_with_evidence.is_empty()
            && let Some(new_call_op) = try_prepend_evidence_to_call(
                db,
                op,
                &remapped_operands,
                ev_value,
                fns_with_evidence,
                &mut value_map,
            )
        {
            new_ops.push(new_call_op);
            changed = true;
            continue;
        }

        // Handle func.call_indirect: replace evidence argument with current ev_value.
        if let Some(new_call_op) =
            try_replace_call_indirect_evidence(db, op, &remapped_operands, ev_value, &mut value_map)
        {
            new_ops.push(new_call_op);
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
                        Rc::clone(&registry),
                        fns_with_evidence,
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
    registry: SharedOpTableRegistry<'db>,
    fns_with_evidence: &HashSet<Symbol>,
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
                Rc::clone(&registry),
                fns_with_evidence,
                None,
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

/// Collect operations (ability_ref, op_name) pairs for a given tag from handler_dispatch.
///
/// This is used by the OpTableRegistry to track which operations each handler handles,
/// enabling table-based dispatch. The operations are ordered by their appearance in the
/// handler body blocks.
fn collect_operations_for_tag<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    target_tag: u32,
) -> Vec<(Type<'db>, Symbol)> {
    let mut operations = Vec::new();

    for op in block.operations(db).iter() {
        if let Ok(dispatch_op) = cont::HandlerDispatch::from_operation(db, *op) {
            let tag = dispatch_op.tag(db);
            if tag != target_tag {
                continue;
            }

            // Extract (ability_ref, op_name) pairs from body blocks
            // Skip block 0 which is the "done" case
            let body = dispatch_op.body(db);
            for (i, body_block) in body.blocks(db).iter().enumerate() {
                if i == 0 {
                    continue; // Skip "done" block
                }

                // Check block arguments for ability_ref and op_name attributes
                for arg in body_block.args(db).iter() {
                    let attrs = arg.attrs(db);
                    let ability_ref = attrs.get(&Symbol::new("ability_ref")).and_then(|a| {
                        if let Attribute::Type(ty) = a {
                            Some(*ty)
                        } else {
                            None
                        }
                    });
                    let op_name = attrs.get(&Symbol::new("op_name")).and_then(|a| {
                        if let Attribute::Symbol(s) = a {
                            Some(*s)
                        } else {
                            None
                        }
                    });

                    if let (Some(ability), Some(name)) = (ability_ref, op_name) {
                        // Avoid duplicates
                        if !operations.contains(&(ability, name)) {
                            operations.push((ability, name));
                        }
                    }
                }
            }

            break; // Found the target tag, no need to continue
        }
    }

    operations
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

/// Validate that all ability IDs in the module are unique (no hash collisions).
///
/// Scans all `cont.shift` operations to collect ability references, computes their
/// IDs, and panics if two distinct ability types map to the same 32-bit hash.
fn validate_ability_id_uniqueness(db: &dyn salsa::Database, module: &core::Module<'_>) {
    let mut seen: HashMap<u32, Type<'_>> = HashMap::new();
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                collect_ability_ids_in_region(db, &func_op.body(db), &mut seen);
            }
        }
    }
}

fn collect_ability_ids_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    seen: &mut HashMap<u32, Type<'db>>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(shift_op) = cont::Shift::from_operation(db, *op) {
                let ability_ref = shift_op.ability_ref(db);
                let id = compute_ability_id(db, ability_ref);
                if let Some(existing) = seen.get(&id) {
                    if *existing != ability_ref {
                        panic!(
                            "ICE: ability ID hash collision: {existing:?} and {ability_ref:?} \
                             both hash to {id:#010x}. Consider using sequential IDs."
                        );
                    }
                } else {
                    seen.insert(id, ability_ref);
                }
            }
            for nested in op.regions(db).iter() {
                collect_ability_ids_in_region(db, nested, seen);
            }
        }
    }
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

// ============================================================================
// Handler Table Emission
// ============================================================================

/// Maximum number of operations per handler for table sizing.
/// This constant determines the stride in the flattened handler dispatch table.
#[cfg(test)]
pub(crate) const MAX_OPS_PER_HANDLER: u32 = 8;

/// Emit `ability.handler_table` operation to capture the handler dispatch table structure.
///
/// No longer called in the main pipeline — handler dispatch is done via
/// inline `scf.if` chains in `cont_to_trampoline`. Kept for tests.
#[cfg(test)]
pub(crate) fn emit_handler_table<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    registry: &OpTableRegistry<'db>,
) -> core::Module<'db> {
    if registry.is_empty() {
        return module;
    }

    let location = module.location(db);
    let body = module.body(db);
    let blocks = body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    // Build handler_entry operations for each registered handler
    let mut entry_ops: Vec<Operation<'db>> = Vec::new();
    for (idx, entry) in registry.entries().iter().enumerate() {
        let entry_location = entry.location;

        // Create funcs region with func.constant operations
        // For now, we just record the function names; actual function references
        // will be resolved during WASM lowering
        // Use core.ptr type as placeholder for funcref (will be lowered to wasm.funcref)
        let funcref_ty = core::Ptr::new(db).as_type();
        let func_ops: Vec<Operation<'db>> = entry
            .operations
            .iter()
            .enumerate()
            .map(|(op_idx, (_ability_ref, _op_name))| {
                // Generate handler function name: __handler_{tag}_op_{idx}
                let func_name = format!("__handler_{}_op_{}", idx, op_idx);
                func::constant(
                    db,
                    entry_location,
                    funcref_ty,
                    Symbol::from_dynamic(&func_name),
                )
                .as_operation()
            })
            .collect();

        let funcs_block = Block::new(
            db,
            trunk_ir::BlockId::fresh(),
            entry_location,
            IdVec::new(),
            IdVec::from(func_ops),
        );
        let funcs_region = Region::new(db, entry_location, IdVec::from(vec![funcs_block]));

        let entry_op = ability::handler_entry(
            db,
            entry_location,
            idx as u32,                    // tag
            entry.operations.len() as u32, // op_count
            funcs_region,
        );
        entry_ops.push(entry_op.as_operation());
    }

    // Create entries region containing all handler_entry ops
    let entries_block = Block::new(
        db,
        trunk_ir::BlockId::fresh(),
        location,
        IdVec::new(),
        IdVec::from(entry_ops),
    );
    let entries_region = Region::new(db, location, IdVec::from(vec![entries_block]));

    // Create handler_table operation
    let handler_table_op =
        ability::handler_table(db, location, MAX_OPS_PER_HANDLER, entries_region);

    // Prepend handler_table to module body
    let mut new_ops: Vec<Operation<'db>> = vec![handler_table_op.as_operation()];
    new_ops.extend(entry_block.operations(db).iter().copied());

    let new_entry_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        IdVec::from(new_ops),
    );

    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_entry_block]));
    core::Module::create(db, module.location(db), module.name(db), new_body)
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
        let evidence_ty = ability::evidence_adt_type(db);

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
        let registry = Rc::new(RefCell::new(OpTableRegistry::new()));
        let (new_block, changed) =
            transform_shifts_in_block(db, &entry_block, ev_value, &handled_by_tag, registry);

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

    // ========================================================================
    // OpTableRegistry Tests
    // ========================================================================

    #[salsa_test]
    fn test_op_table_registry_basic(db: &salsa::DatabaseImpl) {
        let mut registry = OpTableRegistry::new();
        let location = test_location(db);

        // Initially empty
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        // Register a handler with one ability and two operations
        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        let abilities = vec![state_ref.as_type()];
        let operations = vec![
            (state_ref.as_type(), Symbol::new("get")),
            (state_ref.as_type(), Symbol::new("set")),
        ];

        let idx = registry.register(abilities.clone(), operations.clone(), location);
        assert_eq!(idx, 0);
        assert_eq!(registry.len(), 1);

        // Verify entry contents
        let entry = registry.get(0).expect("Entry should exist");
        assert_eq!(entry.abilities.len(), 1);
        assert_eq!(entry.operations.len(), 2);
    }

    #[salsa_test]
    fn test_op_table_registry_compute_op_offset(db: &salsa::DatabaseImpl) {
        let mut registry = OpTableRegistry::new();
        let location = test_location(db);

        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        let operations = vec![
            (state_ref.as_type(), Symbol::new("get")),
            (state_ref.as_type(), Symbol::new("set")),
            (state_ref.as_type(), Symbol::new("modify")),
        ];

        let idx = registry.register(vec![state_ref.as_type()], operations, location);

        // Test op_offset lookup
        let get_offset = registry.compute_op_offset(idx, state_ref.as_type(), Symbol::new("get"));
        assert_eq!(get_offset, Some(0));

        let set_offset = registry.compute_op_offset(idx, state_ref.as_type(), Symbol::new("set"));
        assert_eq!(set_offset, Some(1));

        let modify_offset =
            registry.compute_op_offset(idx, state_ref.as_type(), Symbol::new("modify"));
        assert_eq!(modify_offset, Some(2));

        // Non-existent operation
        let unknown_offset =
            registry.compute_op_offset(idx, state_ref.as_type(), Symbol::new("unknown"));
        assert_eq!(unknown_offset, None);

        // Non-existent handler index
        let invalid_idx_offset =
            registry.compute_op_offset(999, state_ref.as_type(), Symbol::new("get"));
        assert_eq!(invalid_idx_offset, None);
    }

    #[salsa_test]
    fn test_op_table_registry_multiple_handlers(db: &salsa::DatabaseImpl) {
        let mut registry = OpTableRegistry::new();
        let location = test_location(db);

        // Register first handler (State)
        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        let state_ops = vec![
            (state_ref.as_type(), Symbol::new("get")),
            (state_ref.as_type(), Symbol::new("set")),
        ];
        let state_idx = registry.register(vec![state_ref.as_type()], state_ops, location);

        // Register second handler (Console)
        let console_ref = core::AbilityRefType::simple(db, Symbol::new("Console"));
        let console_ops = vec![
            (console_ref.as_type(), Symbol::new("print")),
            (console_ref.as_type(), Symbol::new("read")),
        ];
        let console_idx = registry.register(vec![console_ref.as_type()], console_ops, location);

        // Handlers should get sequential indices
        assert_eq!(state_idx, 0);
        assert_eq!(console_idx, 1);
        assert_eq!(registry.len(), 2);

        // Each handler should have correct op_offsets
        assert_eq!(
            registry.compute_op_offset(state_idx, state_ref.as_type(), Symbol::new("get")),
            Some(0)
        );
        assert_eq!(
            registry.compute_op_offset(state_idx, state_ref.as_type(), Symbol::new("set")),
            Some(1)
        );

        assert_eq!(
            registry.compute_op_offset(console_idx, console_ref.as_type(), Symbol::new("print")),
            Some(0)
        );
        assert_eq!(
            registry.compute_op_offset(console_idx, console_ref.as_type(), Symbol::new("read")),
            Some(1)
        );

        // Cross-handler lookups should fail
        assert_eq!(
            registry.compute_op_offset(state_idx, console_ref.as_type(), Symbol::new("print")),
            None
        );
        assert_eq!(
            registry.compute_op_offset(console_idx, state_ref.as_type(), Symbol::new("get")),
            None
        );
    }

    #[salsa_test]
    fn test_op_table_registry_entries_accessor(db: &salsa::DatabaseImpl) {
        let mut registry = OpTableRegistry::new();
        let location = test_location(db);

        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        registry.register(
            vec![state_ref.as_type()],
            vec![(state_ref.as_type(), Symbol::new("get"))],
            location,
        );

        let entries = registry.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].abilities.len(), 1);
        assert_eq!(entries[0].operations.len(), 1);
    }

    // ========================================================================
    // op_offset soundness test
    // ========================================================================

    /// Demonstrate that two handlers registering the same ability operations
    /// in different order produce different `compute_op_offset` results.
    ///
    /// This is the fundamental reason the first-match heuristic is unsound:
    /// at compile time we don't know which handler will be active at runtime,
    /// so picking the offset from an arbitrary handler can be wrong.
    #[salsa_test]
    fn test_op_offset_differs_across_handlers(db: &salsa::DatabaseImpl) {
        let mut registry = OpTableRegistry::new();
        let location = test_location(db);

        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));

        // Handler A: [get, set]
        let ops_a = vec![
            (state_ref.as_type(), Symbol::new("get")),
            (state_ref.as_type(), Symbol::new("set")),
        ];
        let idx_a = registry.register(vec![state_ref.as_type()], ops_a, location);

        // Handler B: [set, get] — reversed order
        let ops_b = vec![
            (state_ref.as_type(), Symbol::new("set")),
            (state_ref.as_type(), Symbol::new("get")),
        ];
        let idx_b = registry.register(vec![state_ref.as_type()], ops_b, location);

        // get has offset 0 in handler A, but offset 1 in handler B
        let get_offset_a =
            registry.compute_op_offset(idx_a, state_ref.as_type(), Symbol::new("get"));
        let get_offset_b =
            registry.compute_op_offset(idx_b, state_ref.as_type(), Symbol::new("get"));
        assert_eq!(get_offset_a, Some(0));
        assert_eq!(get_offset_b, Some(1));
        assert_ne!(
            get_offset_a, get_offset_b,
            "Same operation 'get' has different offsets across handlers — \
             first-match heuristic would pick one arbitrarily"
        );

        // set has offset 1 in handler A, but offset 0 in handler B
        let set_offset_a =
            registry.compute_op_offset(idx_a, state_ref.as_type(), Symbol::new("set"));
        let set_offset_b =
            registry.compute_op_offset(idx_b, state_ref.as_type(), Symbol::new("set"));
        assert_eq!(set_offset_a, Some(1));
        assert_eq!(set_offset_b, Some(0));
        assert_ne!(set_offset_a, set_offset_b);
    }

    // ========================================================================
    // emit_handler_table Tests
    // ========================================================================

    #[salsa::tracked]
    fn run_emit_handler_table_test(db: &dyn salsa::Database) -> Result<(), String> {
        let location = test_location(db);

        // Create an empty module
        let entry_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        let body = Region::new(db, location, idvec![entry_block]);
        let module = core::Module::create(db, location, Symbol::new("test"), body);

        // Create a registry with one handler
        let mut registry = OpTableRegistry::new();
        let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));
        registry.register(
            vec![state_ref.as_type()],
            vec![
                (state_ref.as_type(), Symbol::new("get")),
                (state_ref.as_type(), Symbol::new("set")),
            ],
            location,
        );

        // Emit handler_table
        let new_module = emit_handler_table(db, module, &registry);

        // Verify handler_table was added to module body
        let new_body = new_module.body(db);
        let blocks = new_body.blocks(db);
        if blocks.is_empty() {
            return Err("Module body should have blocks".to_string());
        }

        let entry = blocks.first().unwrap();
        let ops = entry.operations(db);
        if ops.is_empty() {
            return Err("Module body should have operations".to_string());
        }

        // First operation should be handler_table
        let first_op = ops[0];
        let handler_table_op = ability::HandlerTable::from_operation(db, first_op)
            .map_err(|_| "First operation should be handler_table")?;

        // Verify max_ops_per_handler
        if handler_table_op.max_ops_per_handler(db) != MAX_OPS_PER_HANDLER {
            return Err(format!(
                "max_ops_per_handler should be {}, got {}",
                MAX_OPS_PER_HANDLER,
                handler_table_op.max_ops_per_handler(db)
            ));
        }

        // Verify entries region has one handler_entry
        let entries_region = handler_table_op.entries(db);
        let entries_blocks = entries_region.blocks(db);
        if entries_blocks.is_empty() {
            return Err("Entries region should have blocks".to_string());
        }

        let entries_block = entries_blocks.first().unwrap();
        let entry_ops = entries_block.operations(db);
        if entry_ops.len() != 1 {
            return Err(format!(
                "Should have 1 handler_entry, got {}",
                entry_ops.len()
            ));
        }

        // Verify handler_entry
        let entry_op = entry_ops[0];
        let handler_entry = ability::HandlerEntry::from_operation(db, entry_op)
            .map_err(|_| "Should be handler_entry operation")?;

        if handler_entry.tag(db) != 0 {
            return Err(format!(
                "Handler tag should be 0, got {}",
                handler_entry.tag(db)
            ));
        }

        if handler_entry.op_count(db) != 2 {
            return Err(format!(
                "Handler op_count should be 2, got {}",
                handler_entry.op_count(db)
            ));
        }

        // Verify funcs region has func.constant operations
        let funcs_region = handler_entry.funcs(db);
        let funcs_blocks = funcs_region.blocks(db);
        if funcs_blocks.is_empty() {
            return Err("Funcs region should have blocks".to_string());
        }

        let funcs_block = funcs_blocks.first().unwrap();
        let func_ops = funcs_block.operations(db);
        if func_ops.len() != 2 {
            return Err(format!(
                "Should have 2 func.constant ops, got {}",
                func_ops.len()
            ));
        }

        // Verify func.constant names
        let const_0 = func::Constant::from_operation(db, func_ops[0])
            .map_err(|_| "Should be func.constant operation")?;
        let const_1 = func::Constant::from_operation(db, func_ops[1])
            .map_err(|_| "Should be func.constant operation")?;

        if const_0.func_ref(db) != Symbol::new("__handler_0_op_0") {
            return Err(format!(
                "First func.constant should reference __handler_0_op_0, got {}",
                const_0.func_ref(db)
            ));
        }

        if const_1.func_ref(db) != Symbol::new("__handler_0_op_1") {
            return Err(format!(
                "Second func.constant should reference __handler_0_op_1, got {}",
                const_1.func_ref(db)
            ));
        }

        Ok(())
    }

    #[salsa_test]
    fn test_emit_handler_table(db: &salsa::DatabaseImpl) {
        let result = run_emit_handler_table_test(db);
        if let Err(msg) = result {
            panic!("{}", msg);
        }
    }

    #[salsa::tracked]
    fn run_emit_handler_table_empty_registry_test(db: &dyn salsa::Database) -> Result<(), String> {
        let location = test_location(db);

        // Create an empty module with one operation
        let ret_op = func::r#return(db, location, None).as_operation();
        let entry_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), idvec![ret_op]);
        let body = Region::new(db, location, idvec![entry_block]);
        let module = core::Module::create(db, location, Symbol::new("test"), body);

        // Empty registry
        let registry = OpTableRegistry::new();

        // Emit handler_table (should be no-op)
        let new_module = emit_handler_table(db, module, &registry);

        // Verify module is unchanged (no handler_table added)
        let new_body = new_module.body(db);
        let blocks = new_body.blocks(db);
        let entry = blocks.first().unwrap();
        let ops = entry.operations(db);

        // Should only have the original return op
        if ops.len() != 1 {
            return Err(format!(
                "Module should have only 1 op (return), got {}",
                ops.len()
            ));
        }

        // Verify it's not a handler_table
        if ability::HandlerTable::from_operation(db, ops[0]).is_ok() {
            return Err("Should not have handler_table when registry is empty".to_string());
        }

        Ok(())
    }

    #[salsa_test]
    fn test_emit_handler_table_empty_registry(db: &salsa::DatabaseImpl) {
        let result = run_emit_handler_table_empty_registry_test(db);
        if let Err(msg) = result {
            panic!("{}", msg);
        }
    }

    // ========================================================================
    // transform_block_with_evidence / transform_region_with_evidence tests
    // ========================================================================

    /// Test that `transform_block_with_evidence` correctly applies non-empty
    /// `initial_remap` to operands in the block.
    #[salsa::tracked]
    fn run_initial_remap_test(db: &dyn salsa::Database) -> Result<(), String> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();
        let evidence_ty = ability::evidence_adt_type(db);

        // Create an evidence value that will serve as ev_value
        let ev_block_id = BlockId::fresh();
        let ev_value = Value::new(db, ValueDef::BlockArg(ev_block_id), 0);

        // Create old_val and new_val for the remap
        let old_val_id = BlockId::fresh();
        let old_val = Value::new(db, ValueDef::BlockArg(old_val_id), 0);
        let new_val_id = BlockId::fresh();
        let new_val = Value::new(db, ValueDef::BlockArg(new_val_id), 0);

        // Create a func.call @some_fn(%old_val) — a non-effectful call
        let call_op = func::call(db, location, vec![old_val], i64_ty, Symbol::new("some_fn"));

        let ret = func::r#return(db, location, Some(call_op.result(db)));

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![BlockArg::of_type(db, evidence_ty)],
            idvec![call_op.as_operation(), ret.as_operation()],
        );

        // Build initial_remap: old_val → new_val
        let mut initial_remap = ImHashMap::new();
        initial_remap.insert(old_val, new_val);

        let fns_with_evidence: HashSet<Symbol> = HashSet::new();
        let registry = Rc::new(RefCell::new(OpTableRegistry::new()));

        let (new_block, changed) = transform_block_with_evidence(
            db,
            &block,
            ev_value,
            &fns_with_evidence,
            initial_remap,
            registry,
        );

        // changed should be true since initial_remap is non-empty
        if !changed {
            return Err("Block should be marked as changed when initial_remap is non-empty".into());
        }

        // Find the call operation and check its operand was remapped
        let mut found_remapped = false;
        for op in new_block.operations(db).iter() {
            if let Ok(call) = func::Call::from_operation(db, *op)
                && call.callee(db) == "some_fn"
            {
                let operands = op.operands(db);
                if operands.is_empty() {
                    return Err("Call should have operands".into());
                }
                if operands[0] == old_val {
                    return Err("Operand should have been remapped from old_val to new_val".into());
                }
                if operands[0] != new_val {
                    return Err(format!("Operand should be new_val, got {:?}", operands[0]));
                }
                found_remapped = true;
            }
        }

        if !found_remapped {
            return Err("Should find remapped call to some_fn".into());
        }

        Ok(())
    }

    #[salsa_test]
    fn test_transform_block_with_evidence_uses_initial_remap(db: &salsa::DatabaseImpl) {
        let result = run_initial_remap_test(db);
        if let Err(msg) = result {
            panic!("{}", msg);
        }
    }

    /// Test that `transform_region_with_evidence` propagates `initial_remap`
    /// to each block in the region.
    #[salsa::tracked]
    fn run_region_remap_propagation_test(db: &dyn salsa::Database) -> Result<(), String> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();
        let evidence_ty = ability::evidence_adt_type(db);

        // Create ev_value
        let ev_block_id = BlockId::fresh();
        let ev_value = Value::new(db, ValueDef::BlockArg(ev_block_id), 0);

        // Create old_val / new_val for the remap
        let old_val_id = BlockId::fresh();
        let old_val = Value::new(db, ValueDef::BlockArg(old_val_id), 0);
        let new_val_id = BlockId::fresh();
        let new_val = Value::new(db, ValueDef::BlockArg(new_val_id), 0);

        // Create a block with func.call @fn(%old_val)
        let call_op = func::call(db, location, vec![old_val], i64_ty, Symbol::new("fn"));
        let ret = func::r#return(db, location, Some(call_op.result(db)));
        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![BlockArg::of_type(db, evidence_ty)],
            idvec![call_op.as_operation(), ret.as_operation()],
        );

        let region = Region::new(db, location, idvec![block]);

        // Build initial_remap: old_val → new_val
        let mut initial_remap = ImHashMap::new();
        initial_remap.insert(old_val, new_val);

        let fns_with_evidence: HashSet<Symbol> = HashSet::new();
        let registry = Rc::new(RefCell::new(OpTableRegistry::new()));

        let (new_region, changed) = transform_region_with_evidence(
            db,
            &region,
            ev_value,
            &fns_with_evidence,
            initial_remap,
            registry,
        );

        if !changed {
            return Err("Region should be marked as changed".into());
        }

        // Verify the first block's call operand was remapped
        let blocks = new_region.blocks(db);
        if blocks.is_empty() {
            return Err("Region should have blocks".into());
        }

        let first_block = &blocks[0];
        let mut found_remapped = false;
        for op in first_block.operations(db).iter() {
            if let Ok(call) = func::Call::from_operation(db, *op)
                && call.callee(db) == "fn"
            {
                let operands = op.operands(db);
                if operands.is_empty() {
                    return Err("Call should have operands".into());
                }
                if operands[0] == old_val {
                    return Err(
                        "Region should propagate initial_remap to block: operand not remapped"
                            .into(),
                    );
                }
                if operands[0] != new_val {
                    return Err(format!("Operand should be new_val, got {:?}", operands[0]));
                }
                found_remapped = true;
            }
        }

        if !found_remapped {
            return Err("Should find remapped call in region's block".into());
        }

        Ok(())
    }

    #[salsa_test]
    fn test_transform_region_with_evidence_propagates_remap(db: &salsa::DatabaseImpl) {
        let result = run_region_remap_propagation_test(db);
        if let Err(msg) = result {
            panic!("{}", msg);
        }
    }
}
