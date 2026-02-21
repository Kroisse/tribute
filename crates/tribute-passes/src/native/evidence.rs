//! Evidence runtime lowering for the native backend.
//!
//! This pass adapts evidence-related IR from `resolve_evidence` for native
//! (Cranelift) code generation.  The WASM backend (`evidence_to_wasm.rs`)
//! replaces the stubs with inline binary-search IR; the native backend instead
//! delegates to `extern "C"` functions in `tribute-runtime`.
//!
//! ## Transformations
//!
//! 1. **Stub replacement** — `func.func @__tribute_evidence_lookup` /
//!    `@__tribute_evidence_extend` stubs (with `unreachable` body) are replaced
//!    by extern declarations with native signatures.
//!
//! 2. **Empty evidence** — `adt.array_new(0, evidence_ty)` →
//!    `func.call @__tribute_evidence_empty()`.
//!
//! 3. **Extend call-site rewrite** — the 2-arg call
//!    `func.call @__tribute_evidence_extend(ev, marker)` where `marker` is
//!    produced by `adt.struct_new(ability_id, prompt_tag, op_table_index)` is
//!    rewritten to a 4-arg call passing the fields directly, and the now-dead
//!    `adt.struct_new` is removed.

use tribute_ir::dialect::ability;
use trunk_ir::dialect::{adt, core, func};
use trunk_ir::{Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Value};

/// Lower evidence operations for the native backend.
///
/// Must run AFTER `cont_to_libmprompt` and BEFORE DCE.
#[salsa::tracked]
pub fn lower_evidence_to_native<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let module = replace_stubs_and_add_empty(db, module);
    rewrite_evidence_ops_in_module(db, module)
}

// =============================================================================
// Phase 1: Replace stubs + add __tribute_evidence_empty declaration
// =============================================================================

/// Replace evidence stub functions with extern declarations and add
/// `__tribute_evidence_empty` if not already present.
fn replace_stubs_and_add_empty<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    let location = module.location(db);
    let mut changed = false;
    let mut has_evidence_empty = false;
    let mut new_ops: Vec<Operation<'db>> = Vec::new();

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            let name = func_op.sym_name(db);

            if name == Symbol::new("__tribute_evidence_lookup") {
                new_ops.push(make_evidence_lookup_extern(db, location));
                changed = true;
                continue;
            }
            if name == Symbol::new("__tribute_evidence_extend") {
                new_ops.push(make_evidence_extend_extern(db, location));
                changed = true;
                continue;
            }
            if name == Symbol::new("__tribute_evidence_empty") {
                has_evidence_empty = true;
            }
        }
        new_ops.push(*op);
    }

    if !has_evidence_empty {
        // Insert the empty declaration at the front so it's visible to all callers.
        new_ops.insert(0, make_evidence_empty_extern(db, location));
        changed = true;
    }

    if !changed {
        return module;
    }

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

/// `fn __tribute_evidence_empty() -> ptr`  (extern declaration)
fn make_evidence_empty_extern<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
) -> Operation<'db> {
    let ptr_ty = core::Ptr::new(db).as_type();

    func::Func::build_extern(
        db,
        location,
        "__tribute_evidence_empty",
        None,
        [],
        ptr_ty,
        None,
        Some("C"),
    )
    .as_operation()
}

/// `fn __tribute_evidence_lookup(ev: ptr, ability_id: i32) -> Marker`  (extern)
///
/// Marker is kept as `ability.marker_adt_type` — the native type converter
/// leaves it as-is (pass-through), and downstream `adt_to_clif` lowers
/// field accesses (`adt.struct_get`) to `clif.load` with correct offsets.
fn make_evidence_lookup_extern<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
) -> Operation<'db> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();
    let marker_ty = ability::marker_adt_type(db);

    func::Func::build_extern(
        db,
        location,
        "__tribute_evidence_lookup",
        None,
        [(ptr_ty, None), (i32_ty, None)],
        marker_ty,
        None,
        Some("C"),
    )
    .as_operation()
}

/// `fn __tribute_evidence_extend(ev: ptr, ability_id: i32, prompt_tag: i32, op_table_index: i32) -> ptr`
fn make_evidence_extend_extern<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
) -> Operation<'db> {
    let ptr_ty = core::Ptr::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();

    func::Func::build_extern(
        db,
        location,
        "__tribute_evidence_extend",
        None,
        [
            (ptr_ty, None),
            (i32_ty, None),
            (i32_ty, None),
            (i32_ty, None),
        ],
        ptr_ty,
        None,
        Some("C"),
    )
    .as_operation()
}

// =============================================================================
// Phase 2: Rewrite evidence operations inside function bodies
// =============================================================================

/// Walk every function in the module and rewrite evidence-related operations.
fn rewrite_evidence_ops_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    let mut changed = false;
    let mut new_ops: Vec<Operation<'db>> = Vec::new();

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            let name = func_op.sym_name(db);
            // Skip extern stubs (they have no real body to rewrite)
            if is_evidence_runtime_fn(name) {
                new_ops.push(*op);
                continue;
            }

            let func_body = func_op.body(db);
            let (new_body, body_changed) = rewrite_evidence_ops_in_region(db, &func_body);
            if body_changed {
                let rebuilt = func::func(
                    db,
                    op.location(db),
                    func_op.sym_name(db),
                    func_op.r#type(db),
                    new_body,
                );
                new_ops.push(rebuilt.as_operation());
                changed = true;
            } else {
                new_ops.push(*op);
            }
        } else {
            new_ops.push(*op);
        }
    }

    if !changed {
        return module;
    }

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

fn is_evidence_runtime_fn(name: Symbol) -> bool {
    name == Symbol::new("__tribute_evidence_lookup")
        || name == Symbol::new("__tribute_evidence_extend")
        || name == Symbol::new("__tribute_evidence_empty")
}

/// Rewrite evidence operations in a region (recursive over nested regions).
fn rewrite_evidence_ops_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> (Region<'db>, bool) {
    let blocks = region.blocks(db);
    let mut any_changed = false;
    let mut new_blocks: Vec<Block<'db>> = Vec::with_capacity(blocks.len());

    for block in blocks.iter() {
        let (new_block, block_changed) = rewrite_evidence_ops_in_block(db, block);
        any_changed |= block_changed;
        new_blocks.push(new_block);
    }

    if !any_changed {
        return (*region, false);
    }

    let new_region = Region::new(db, region.location(db), IdVec::from(new_blocks));
    (new_region, true)
}

/// Rewrite evidence operations in a single block.
///
/// Handles:
/// - `adt.array_new(0, evidence_ty)` → `func.call @__tribute_evidence_empty()`
/// - `adt.struct_new(a, b, c) : Marker` + `func.call @__tribute_evidence_extend(ev, marker)`
///   → `func.call @__tribute_evidence_extend(ev, a, b, c)`
fn rewrite_evidence_ops_in_block<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
) -> (Block<'db>, bool) {
    let ops = block.operations(db);
    let mut new_ops: Vec<Operation<'db>> = Vec::with_capacity(ops.len());
    let mut changed = false;

    // Track adt.struct_new results that produce Marker values, so we can
    // decompose them at the extend call site and skip the struct_new itself.
    let mut marker_struct_operands: std::collections::HashMap<Value<'db>, Vec<Value<'db>>> =
        std::collections::HashMap::new();

    // Track value remapping: when an operation is replaced, its result Value
    // changes. Subsequent operations referencing the old result must use the
    // new one.
    let mut value_remap: std::collections::HashMap<Value<'db>, Value<'db>> =
        std::collections::HashMap::new();

    for op in ops.iter() {
        // --- adt.array_new(0, evidence_ty) → func.call @__tribute_evidence_empty() ---
        if let Ok(array_new) = adt::ArrayNew::from_operation(db, *op) {
            let result_ty = array_new.as_operation().results(db);
            if !result_ty.is_empty() && ability::is_evidence_type(db, result_ty[0]) {
                let ptr_ty = core::Ptr::new(db).as_type();
                let call = func::call(
                    db,
                    op.location(db),
                    vec![],
                    ptr_ty,
                    Symbol::new("__tribute_evidence_empty"),
                );
                new_ops.push(call.as_operation());
                // Remap old result → new result
                let old_result = op.result(db, 0);
                let new_result = call.as_operation().result(db, 0);
                value_remap.insert(old_result, new_result);
                changed = true;
                continue;
            }
        }

        // --- Track adt.struct_new that produces a Marker ---
        if let Ok(struct_new) = adt::StructNew::from_operation(db, *op) {
            let result_ty = struct_new.as_operation().results(db);
            if !result_ty.is_empty() && ability::is_marker_type(db, result_ty[0]) {
                let operands: Vec<Value<'db>> = op
                    .operands(db)
                    .iter()
                    .map(|v| remap_value(*v, &value_remap))
                    .collect();
                let result_val = op.result(db, 0);
                marker_struct_operands.insert(result_val, operands);
                // Don't emit the struct_new — it will be inlined into the extend call.
                changed = true;
                continue;
            }
        }

        // --- Rewrite func.call @__tribute_evidence_extend(ev, marker) → (ev, a, b, c) ---
        if let Ok(call_op) = func::Call::from_operation(db, *op)
            && call_op.callee(db) == "__tribute_evidence_extend"
        {
            let operands = op.operands(db);
            // Original: (ev, marker)  — 2 operands
            if operands.len() == 2 {
                let ev_val = remap_value(operands[0], &value_remap);
                let marker_val = operands[1];

                let fields = marker_struct_operands.get(&marker_val).unwrap_or_else(|| {
                    unreachable!(
                        "ICE: __tribute_evidence_extend marker operand has no matching \
                         adt.struct_new in the same block (IR invariant violated)"
                    )
                });
                // fields = [ability_id, prompt_tag, op_table_index]
                let ptr_ty = core::Ptr::new(db).as_type();
                let mut args = vec![ev_val];
                args.extend_from_slice(fields);
                let new_call = func::call(
                    db,
                    op.location(db),
                    args,
                    ptr_ty,
                    Symbol::new("__tribute_evidence_extend"),
                );
                new_ops.push(new_call.as_operation());
                // Remap old result → new result
                let old_result = op.result(db, 0);
                let new_result = new_call.as_operation().result(db, 0);
                value_remap.insert(old_result, new_result);
                changed = true;
                continue;
            }
        }

        // --- Apply value remapping to operands if needed ---
        let needs_remap = op.operands(db).iter().any(|v| value_remap.contains_key(v));

        if needs_remap {
            let new_operands: IdVec<Value<'db>> = op
                .operands(db)
                .iter()
                .map(|v| remap_value(*v, &value_remap))
                .collect();

            // Recurse into nested regions
            let regions = op.regions(db);
            let mut new_regions: Vec<Region<'db>> = Vec::with_capacity(regions.len());
            let mut region_changed = false;
            for region in regions.iter() {
                let (new_region, rc) = rewrite_evidence_ops_in_region(db, region);
                region_changed |= rc;
                new_regions.push(new_region);
            }

            let rebuilt = Operation::of(db, op.location(db), op.dialect(db), op.name(db))
                .operands(new_operands)
                .results(op.results(db).clone())
                .successors(op.successors(db).clone())
                .attrs(op.attributes(db).clone())
                .regions(if region_changed {
                    IdVec::from(new_regions)
                } else {
                    regions.clone()
                })
                .build();
            // Remap results of rebuilt operation so downstream ops see the new Values
            for (i, _) in op.results(db).iter().enumerate() {
                value_remap.insert(op.result(db, i), rebuilt.result(db, i));
            }
            new_ops.push(rebuilt);
            changed = true;
            continue;
        }

        // --- Recurse into nested regions ---
        let regions = op.regions(db);
        if !regions.is_empty() {
            let mut region_changed = false;
            let mut new_regions: Vec<Region<'db>> = Vec::with_capacity(regions.len());
            for region in regions.iter() {
                let (new_region, rc) = rewrite_evidence_ops_in_region(db, region);
                region_changed |= rc;
                new_regions.push(new_region);
            }
            if region_changed {
                let rebuilt = Operation::of(db, op.location(db), op.dialect(db), op.name(db))
                    .operands(op.operands(db).clone())
                    .results(op.results(db).clone())
                    .successors(op.successors(db).clone())
                    .attrs(op.attributes(db).clone())
                    .regions(IdVec::from(new_regions))
                    .build();
                new_ops.push(rebuilt);
                changed = true;
                continue;
            }
        }

        new_ops.push(*op);
    }

    if !changed {
        return (*block, false);
    }

    let new_block = Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        IdVec::from(new_ops),
    );
    (new_block, true)
}

/// Remap a value through the value_remap table, returning the original if
/// no remapping exists.
fn remap_value<'db>(
    value: Value<'db>,
    remap: &std::collections::HashMap<Value<'db>, Value<'db>>,
) -> Value<'db> {
    remap.get(&value).copied().unwrap_or(value)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Attribute, BlockArg, BlockId, PathId, Span, Value, ValueDef, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Create a module with __tribute_evidence_lookup and __tribute_evidence_extend stubs.
    #[salsa::tracked]
    fn make_module_with_stubs(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let evidence_ty = ability::evidence_adt_type(db);
        let i32_ty = core::I32::new(db).as_type();
        let marker_ty = ability::marker_adt_type(db);

        // Stub: fn __tribute_evidence_lookup(ev: Evidence, ability_id: i32) -> Marker
        let lookup_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, i32_ty]), marker_ty);
        let unreachable1 = func::unreachable(db, location);
        let lookup_body = Region::new(
            db,
            location,
            IdVec::from(vec![Block::new(
                db,
                BlockId::fresh(),
                location,
                IdVec::from(vec![
                    BlockArg::of_type(db, evidence_ty),
                    BlockArg::of_type(db, i32_ty),
                ]),
                IdVec::from(vec![unreachable1.as_operation()]),
            )]),
        );
        let lookup_func = func::func(
            db,
            location,
            Symbol::new("__tribute_evidence_lookup"),
            *lookup_ty,
            lookup_body,
        );

        // Stub: fn __tribute_evidence_extend(ev: Evidence, marker: Marker) -> Evidence
        let extend_ty = core::Func::new(db, IdVec::from(vec![evidence_ty, marker_ty]), evidence_ty);
        let unreachable2 = func::unreachable(db, location);
        let extend_body = Region::new(
            db,
            location,
            IdVec::from(vec![Block::new(
                db,
                BlockId::fresh(),
                location,
                IdVec::from(vec![
                    BlockArg::of_type(db, evidence_ty),
                    BlockArg::of_type(db, marker_ty),
                ]),
                IdVec::from(vec![unreachable2.as_operation()]),
            )]),
        );
        let extend_func = func::func(
            db,
            location,
            Symbol::new("__tribute_evidence_extend"),
            *extend_ty,
            extend_body,
        );

        let entry_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            IdVec::from(vec![lookup_func.as_operation(), extend_func.as_operation()]),
        );
        let module_body = Region::new(db, location, IdVec::from(vec![entry_block]));
        core::Module::create(db, location, "test".into(), module_body)
    }

    #[salsa_test]
    fn test_stubs_replaced_with_native_signatures(db: &salsa::DatabaseImpl) {
        let module = make_module_with_stubs(db);
        let lowered = lower_evidence_to_native(db, module);

        let body = lowered.body(db);
        let entry = body.blocks(db).first().unwrap();
        // Should have 3 functions: empty (added) + lookup + extend
        assert_eq!(entry.operations(db).len(), 3);

        let ptr_ty = core::Ptr::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();

        // First should be __tribute_evidence_empty: () -> ptr
        let empty_op = &entry.operations(db)[0];
        let empty_func = func::Func::from_operation(db, *empty_op).unwrap();
        assert_eq!(
            empty_func.sym_name(db),
            Symbol::new("__tribute_evidence_empty")
        );
        let empty_ty = core::Func::from_type(db, empty_func.r#type(db)).unwrap();
        assert!(empty_ty.params(db).is_empty());
        assert_eq!(empty_ty.result(db), ptr_ty);

        // Second: __tribute_evidence_lookup: (ptr, i32) -> Marker
        let lookup_op = &entry.operations(db)[1];
        let lookup_func = func::Func::from_operation(db, *lookup_op).unwrap();
        assert_eq!(
            lookup_func.sym_name(db),
            Symbol::new("__tribute_evidence_lookup")
        );
        let lookup_ty = core::Func::from_type(db, lookup_func.r#type(db)).unwrap();
        assert_eq!(lookup_ty.params(db).as_slice(), &[ptr_ty, i32_ty]);

        // Third: __tribute_evidence_extend: (ptr, i32, i32, i32) -> ptr
        let extend_op = &entry.operations(db)[2];
        let extend_func = func::Func::from_operation(db, *extend_op).unwrap();
        assert_eq!(
            extend_func.sym_name(db),
            Symbol::new("__tribute_evidence_extend")
        );
        let extend_ty = core::Func::from_type(db, extend_func.r#type(db)).unwrap();
        assert_eq!(
            extend_ty.params(db).as_slice(),
            &[ptr_ty, i32_ty, i32_ty, i32_ty]
        );
        assert_eq!(extend_ty.result(db), ptr_ty);
    }

    /// Create a module with a function that creates empty evidence via adt.array_new.
    #[salsa::tracked]
    fn make_module_with_empty_evidence(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let evidence_ty = ability::evidence_adt_type(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();

        // Build a user function that creates empty evidence
        let func_ty = core::Func::new(db, IdVec::new(), nil_ty);
        let body_block_id = BlockId::fresh();

        let zero_const =
            trunk_ir::dialect::arith::r#const(db, location, i32_ty, Attribute::IntBits(0));
        let array_new = adt::array_new(
            db,
            location,
            vec![zero_const.result(db)],
            evidence_ty,
            evidence_ty,
        );
        let return_op = func::r#return(db, location, vec![]);

        let body_block = Block::new(
            db,
            body_block_id,
            location,
            IdVec::new(),
            IdVec::from(vec![
                zero_const.as_operation(),
                array_new.as_operation(),
                return_op.as_operation(),
            ]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));
        let user_func = func::func(db, location, Symbol::new("test_fn"), *func_ty, body);

        let entry_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            IdVec::from(vec![user_func.as_operation()]),
        );
        let module_body = Region::new(db, location, IdVec::from(vec![entry_block]));
        core::Module::create(db, location, "test".into(), module_body)
    }

    #[salsa_test]
    fn test_array_new_replaced_with_evidence_empty(db: &salsa::DatabaseImpl) {
        let module = make_module_with_empty_evidence(db);
        let lowered = lower_evidence_to_native(db, module);

        let body = lowered.body(db);
        let entry = body.blocks(db).first().unwrap();

        // Find the user function
        let mut found_empty_call = false;
        for op in entry.operations(db).iter() {
            if let Ok(f) = func::Func::from_operation(db, *op)
                && f.sym_name(db) == Symbol::new("test_fn")
            {
                let func_body = f.body(db);
                let func_entry = func_body.blocks(db).first().unwrap();
                for inner_op in func_entry.operations(db).iter() {
                    if let Ok(call) = func::Call::from_operation(db, *inner_op)
                        && call.callee(db) == "__tribute_evidence_empty"
                    {
                        found_empty_call = true;
                    }
                }
            }
        }
        assert!(
            found_empty_call,
            "adt.array_new should be replaced with func.call @__tribute_evidence_empty"
        );
    }

    /// Create a module with a function that calls __tribute_evidence_extend(ev, struct_new(...)).
    #[salsa::tracked]
    fn make_module_with_extend_call(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let evidence_ty = ability::evidence_adt_type(db);
        let i32_ty = core::I32::new(db).as_type();
        let marker_ty = ability::marker_adt_type(db);

        // Build a function with evidence as first param
        let func_ty = core::Func::new(db, IdVec::from(vec![evidence_ty]), evidence_ty);
        let body_block_id = BlockId::fresh();
        let ev_val = Value::new(db, ValueDef::BlockArg(body_block_id), 0);

        // ability_id = 42
        let ability_id_const =
            trunk_ir::dialect::arith::r#const(db, location, i32_ty, Attribute::IntBits(42));
        // prompt_tag = 1
        let tag_const =
            trunk_ir::dialect::arith::r#const(db, location, i32_ty, Attribute::IntBits(1));
        // op_table_index = 0
        let idx_const =
            trunk_ir::dialect::arith::r#const(db, location, i32_ty, Attribute::IntBits(0));

        // adt.struct_new(ability_id, prompt_tag, op_table_index) : Marker
        let marker_struct = adt::struct_new(
            db,
            location,
            vec![
                ability_id_const.result(db),
                tag_const.result(db),
                idx_const.result(db),
            ],
            marker_ty,
            marker_ty,
        );

        // func.call @__tribute_evidence_extend(ev, marker)
        let extend_call = func::call(
            db,
            location,
            vec![ev_val, marker_struct.result(db)],
            evidence_ty,
            Symbol::new("__tribute_evidence_extend"),
        );

        let return_op = func::r#return(db, location, vec![extend_call.result(db)]);

        let body_block = Block::new(
            db,
            body_block_id,
            location,
            IdVec::from(vec![BlockArg::of_type(db, evidence_ty)]),
            IdVec::from(vec![
                ability_id_const.as_operation(),
                tag_const.as_operation(),
                idx_const.as_operation(),
                marker_struct.as_operation(),
                extend_call.as_operation(),
                return_op.as_operation(),
            ]),
        );
        let body = Region::new(db, location, IdVec::from(vec![body_block]));
        let user_func = func::func(db, location, Symbol::new("test_fn"), *func_ty, body);

        let entry_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            IdVec::from(vec![user_func.as_operation()]),
        );
        let module_body = Region::new(db, location, IdVec::from(vec![entry_block]));
        core::Module::create(db, location, "test".into(), module_body)
    }

    #[salsa_test]
    fn test_extend_call_rewritten_to_4_args(db: &salsa::DatabaseImpl) {
        let module = make_module_with_extend_call(db);
        let lowered = lower_evidence_to_native(db, module);

        let body = lowered.body(db);
        let entry = body.blocks(db).first().unwrap();

        for op in entry.operations(db).iter() {
            if let Ok(f) = func::Func::from_operation(db, *op)
                && f.sym_name(db) == Symbol::new("test_fn")
            {
                let func_body = f.body(db);
                let func_entry = func_body.blocks(db).first().unwrap();

                // Should NOT have adt.struct_new
                let has_struct_new = func_entry
                    .operations(db)
                    .iter()
                    .any(|inner_op| adt::StructNew::from_operation(db, *inner_op).is_ok());
                assert!(
                    !has_struct_new,
                    "adt.struct_new for Marker should be removed"
                );

                // The extend call should now have 4 operands
                for inner_op in func_entry.operations(db).iter() {
                    if let Ok(call) = func::Call::from_operation(db, *inner_op)
                        && call.callee(db) == "__tribute_evidence_extend"
                    {
                        assert_eq!(
                            inner_op.operands(db).len(),
                            4,
                            "extend call should have 4 operands: ev, ability_id, prompt_tag, op_table_index"
                        );
                        return;
                    }
                }
                panic!("should find __tribute_evidence_extend call");
            }
        }
        panic!("should find test_fn function");
    }
}
