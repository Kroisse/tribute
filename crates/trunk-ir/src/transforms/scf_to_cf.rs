//! Lower structured control flow (scf) to CFG-based control flow (cf).
//!
//! This pass converts `scf.if`, `scf.loop`, and `scf.switch` operations into
//! explicit `cf.br` and `cf.cond_br` branch operations with basic blocks.
//!
//! Unlike the scf→wasm lowering which uses PatternApplicator (1:1 op mapping),
//! this pass operates at the block level because scf→cf requires block splitting:
//! a single block containing `scf.if` becomes multiple blocks (entry, then, else, merge).
//!
//! ## Transformations
//!
//! ### scf.if
//! ```text
//! ^bb0:
//!   %0 = op_before(...)
//!   %1 = scf.if(%cond) -> T { scf.yield(%a) } { scf.yield(%b) }
//!   %2 = op_after(%1)
//! ```
//! becomes:
//! ```text
//! ^bb0:
//!   %0 = op_before(...)
//!   cf.cond_br(%cond) -> ^then, ^else
//! ^then:
//!   cf.br(%a) -> ^merge
//! ^else:
//!   cf.br(%b) -> ^merge
//! ^merge(%1: T):
//!   %2 = op_after(%1)
//! ```
//!
//! ### scf.loop
//! ```text
//! ^bb0:
//!   %0 = scf.loop(%init) -> T { body(%x): ... scf.continue(%next) / scf.break(%val) }
//!   %1 = op_after(%0)
//! ```
//! becomes:
//! ```text
//! ^bb0:
//!   cf.br(%init) -> ^header
//! ^header(%x: T):
//!   ... body ops (scf.continue → cf.br -> ^header, scf.break → cf.br -> ^exit) ...
//! ^exit(%0: T):
//!   %1 = op_after(%0)
//! ```
//!
//! ### scf.switch
//! Lowered to chained `cf.cond_br` + `arith.cmp_eq` comparisons.

use std::collections::HashMap;

use crate::dialect::{arith, cf, core, scf};
use crate::{
    Block, BlockArg, BlockBuilder, BlockId, DialectOp, DialectType, IdVec, Location, Operation,
    Region, Type, Value, ValueDef, idvec,
};

/// Lower all `scf` operations in a module to `cf` operations.
///
/// Transforms `scf.if`, `scf.loop`, and `scf.switch` into explicit
/// `cf.br` and `cf.cond_br` with block splitting.
pub fn lower_scf_to_cf<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let new_body = transform_region(db, &body);
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Transform all blocks in a region, recursively processing nested regions.
fn transform_region<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> Region<'db> {
    let mut new_blocks = IdVec::new();
    for block in region.blocks(db).iter() {
        let transformed = transform_block(db, block);
        new_blocks.extend(transformed);
    }
    Region::new(db, region.location(db), new_blocks)
}

/// Transform a single block.
///
/// Scans operations for scf ops. When one is found, the block is split:
/// - Operations before the scf op stay in the current block
/// - The scf op is expanded into multiple blocks
/// - Operations after the scf op go into a merge/exit block
///
/// Non-scf operations with nested regions are recursively transformed.
fn transform_block<'db>(db: &'db dyn salsa::Database, block: &Block<'db>) -> Vec<Block<'db>> {
    let ops = block.operations(db);
    let location = block.location(db);

    // Find the first scf operation in the block
    let scf_idx = ops.iter().position(|op| {
        let dialect = op.dialect(db);
        dialect == scf::DIALECT_NAME()
            && (op.name(db) == scf::IF()
                || op.name(db) == scf::LOOP()
                || op.name(db) == scf::SWITCH())
    });

    let Some(scf_idx) = scf_idx else {
        // No scf operations — just recursively transform nested regions
        let new_ops: IdVec<Operation<'db>> =
            ops.iter().map(|op| transform_op_regions(db, op)).collect();
        return vec![Block::new(
            db,
            block.id(db),
            location,
            block.args(db).clone(),
            new_ops,
        )];
    };

    let scf_op = &ops[scf_idx];

    // Operations before the scf op (recursively transform their regions)
    let before_ops: IdVec<Operation<'db>> = ops[..scf_idx]
        .iter()
        .map(|op| transform_op_regions(db, op))
        .collect();

    // Operations after the scf op (will go into merge block)
    let after_ops: Vec<&Operation<'db>> = ops[scf_idx + 1..].iter().collect();

    // Value map for remapping scf.if/loop results to merge block args
    let mut value_map: HashMap<Value<'db>, Value<'db>> = HashMap::new();

    if scf_op.name(db) == scf::IF() {
        let if_op = scf::If::from_operation(db, *scf_op).unwrap();
        lower_scf_if(
            db,
            block,
            &if_op,
            before_ops,
            &after_ops,
            &mut value_map,
            location,
        )
    } else if scf_op.name(db) == scf::LOOP() {
        let loop_op = scf::Loop::from_operation(db, *scf_op).unwrap();
        lower_scf_loop(
            db,
            block,
            &loop_op,
            before_ops,
            &after_ops,
            &mut value_map,
            location,
        )
    } else if scf_op.name(db) == scf::SWITCH() {
        let switch_op = scf::Switch::from_operation(db, *scf_op).unwrap();
        lower_scf_switch(
            db,
            block,
            &switch_op,
            before_ops,
            &after_ops,
            &mut value_map,
            location,
        )
    } else {
        unreachable!()
    }
}

/// Recursively transform nested regions within an operation.
fn transform_op_regions<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> Operation<'db> {
    let regions = op.regions(db);
    if regions.is_empty() {
        return *op;
    }

    let new_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|region| transform_region(db, region))
        .collect();

    if new_regions == *regions {
        *op
    } else {
        op.modify(db).regions(new_regions).build()
    }
}

/// Recursively transform nested regions within an operation, propagating a value map.
///
/// This is used by `remap_op` so that nested regions also have their captured
/// values remapped (e.g., when an scf.if result is replaced by a merge block arg,
/// operations inside nested func regions must see the updated value).
fn transform_op_regions_with_map<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    value_map: &HashMap<Value<'db>, Value<'db>>,
) -> Operation<'db> {
    let regions = op.regions(db);
    if regions.is_empty() {
        return *op;
    }

    let new_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|region| transform_region_with_map(db, region, value_map))
        .collect();

    if new_regions == *regions {
        *op
    } else {
        op.modify(db).regions(new_regions).build()
    }
}

/// Transform all blocks in a region, remapping values via `value_map` and
/// recursively processing nested regions.
fn transform_region_with_map<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    value_map: &HashMap<Value<'db>, Value<'db>>,
) -> Region<'db> {
    let mut new_blocks = IdVec::new();
    for block in region.blocks(db).iter() {
        let new_ops: IdVec<Operation<'db>> = block
            .operations(db)
            .iter()
            .map(|op| remap_op(db, op, value_map))
            .collect();
        new_blocks.push(Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            new_ops,
        ));
    }
    Region::new(db, region.location(db), new_blocks)
}

/// Remap operands of an operation using the value map.
fn remap_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    value_map: &HashMap<Value<'db>, Value<'db>>,
) -> Operation<'db> {
    if value_map.is_empty() {
        return transform_op_regions(db, op);
    }

    let operands = op.operands(db);
    let mut new_operands = IdVec::with_capacity(operands.len());
    let mut changed = false;

    for &operand in operands.iter() {
        if let Some(&mapped) = value_map.get(&operand) {
            new_operands.push(mapped);
            changed = true;
        } else {
            new_operands.push(operand);
        }
    }

    let remapped = if changed {
        op.modify(db).operands(new_operands).build()
    } else {
        *op
    };
    transform_op_regions_with_map(db, &remapped, value_map)
}

/// Build the merge block with remaining ops, remapping values.
fn build_merge_block<'db>(
    db: &'db dyn salsa::Database,
    merge_id: BlockId,
    location: Location<'db>,
    merge_args: IdVec<BlockArg<'db>>,
    after_ops: &[&Operation<'db>],
    value_map: &HashMap<Value<'db>, Value<'db>>,
) -> Block<'db> {
    let remapped_ops: IdVec<Operation<'db>> = after_ops
        .iter()
        .map(|op| remap_op(db, op, value_map))
        .collect();

    Block::new(db, merge_id, location, merge_args, remapped_ops)
}

/// Lower `scf.if` to cf.cond_br + merge block.
fn lower_scf_if<'db>(
    db: &'db dyn salsa::Database,
    orig_block: &Block<'db>,
    if_op: &scf::If<'db>,
    before_ops: IdVec<Operation<'db>>,
    after_ops: &[&Operation<'db>],
    value_map: &mut HashMap<Value<'db>, Value<'db>>,
    location: Location<'db>,
) -> Vec<Block<'db>> {
    let results = if_op.as_operation().results(db);
    assert!(
        results.len() <= 1,
        "scf.if must have at most 1 result, got {}",
        results.len()
    );
    let result_ty = results.first().copied();

    // Create merge block (will be the target of then/else branches)
    let merge_id = BlockId::fresh();
    let merge_args: IdVec<BlockArg<'db>> = if let Some(ty) = result_ty {
        idvec![BlockArg::of_type(db, ty)]
    } else {
        IdVec::new()
    };

    // Map scf.if result to merge block argument
    if result_ty.is_some() {
        let if_result = if_op.as_operation().result(db, 0);
        let merge_arg = Value::new(db, ValueDef::BlockArg(merge_id), 0);
        value_map.insert(if_result, merge_arg);
    }

    // Build merge block with remaining ops
    let merge_block = build_merge_block(db, merge_id, location, merge_args, after_ops, value_map);

    // Process then region: extract ops and replace scf.yield with cf.br to merge
    let then_blocks = lower_region_to_br(db, &if_op.then(db), merge_block, location);

    // Process else region
    let else_blocks = lower_region_to_br(db, &if_op.r#else(db), merge_block, location);

    // Get the entry blocks of then/else for cond_br targets
    let then_entry = then_blocks[0];
    let else_entry = else_blocks[0];

    // Build entry block: before_ops + cf.cond_br
    let cond_br_op = cf::cond_br(db, location, if_op.cond(db), then_entry, else_entry);
    let mut entry_ops = before_ops;
    entry_ops.push(cond_br_op.as_operation());

    let entry_block = Block::new(
        db,
        orig_block.id(db),
        location,
        orig_block.args(db).clone(),
        entry_ops,
    );

    // Assemble all blocks: entry, then blocks, else blocks, merge
    let mut blocks = vec![entry_block];
    blocks.extend(then_blocks);
    blocks.extend(else_blocks);

    // Recursively transform the merge block (it may contain more scf ops)
    let merge_transformed = transform_block(db, &merge_block);
    blocks.extend(merge_transformed);

    blocks
}

/// Lower a region's blocks to end with cf.br to a merge block.
///
/// Replaces `scf.yield(values)` with `cf.br(values) -> merge_block`.
/// Recursively transforms any nested scf ops within the region.
fn lower_region_to_br<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    merge_block: Block<'db>,
    location: Location<'db>,
) -> Vec<Block<'db>> {
    let mut all_blocks = Vec::new();

    for block in region.blocks(db).iter() {
        let ops = block.operations(db);
        let mut new_ops = IdVec::new();

        for op in ops.iter() {
            if scf::Yield::matches(db, *op) {
                // Replace scf.yield with cf.br to merge block
                let yield_op = scf::Yield::from_operation(db, *op).unwrap();
                let values: Vec<Value<'db>> = yield_op.values(db).to_vec();
                let expected = merge_block.args(db).len();
                assert_eq!(
                    values.len(),
                    expected,
                    "scf.yield must pass {} value(s) to match merge block args, got {}",
                    expected,
                    values.len(),
                );
                let br_op = cf::br(db, location, values, merge_block);
                new_ops.push(br_op.as_operation());
            } else {
                new_ops.push(transform_op_regions(db, op));
            }
        }

        let new_block = Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            new_ops,
        );

        // The block itself may still contain scf ops (not yield)
        // We need to recursively transform it
        let transformed = transform_block_no_yield(db, &new_block);
        all_blocks.extend(transformed);
    }

    all_blocks
}

/// Transform a block that may contain scf ops (but yield has already been handled).
/// This is the same as transform_block but called after yield replacement.
fn transform_block_no_yield<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
) -> Vec<Block<'db>> {
    let ops = block.operations(db);

    // Check if there are any remaining scf ops (if/loop/switch, not yield)
    let has_scf = ops.iter().any(|op| {
        let dialect = op.dialect(db);
        dialect == scf::DIALECT_NAME()
            && (op.name(db) == scf::IF()
                || op.name(db) == scf::LOOP()
                || op.name(db) == scf::SWITCH())
    });

    if has_scf {
        transform_block(db, block)
    } else {
        vec![*block]
    }
}

/// Lower `scf.loop` to cf header + exit blocks.
fn lower_scf_loop<'db>(
    db: &'db dyn salsa::Database,
    orig_block: &Block<'db>,
    loop_op: &scf::Loop<'db>,
    before_ops: IdVec<Operation<'db>>,
    after_ops: &[&Operation<'db>],
    value_map: &mut HashMap<Value<'db>, Value<'db>>,
    location: Location<'db>,
) -> Vec<Block<'db>> {
    let results = loop_op.as_operation().results(db);
    assert!(
        results.len() <= 1,
        "scf.loop must have at most 1 result, got {}",
        results.len()
    );
    let result_ty = results.first().copied();

    // Create exit block
    let exit_id = BlockId::fresh();
    let exit_args: IdVec<BlockArg<'db>> = if let Some(ty) = result_ty {
        idvec![BlockArg::of_type(db, ty)]
    } else {
        IdVec::new()
    };

    // Map loop result to exit block argument
    if result_ty.is_some() {
        let loop_result = loop_op.as_operation().result(db, 0);
        let exit_arg = Value::new(db, ValueDef::BlockArg(exit_id), 0);
        value_map.insert(loop_result, exit_arg);
    }

    let exit_block = Block::new(db, exit_id, location, exit_args, IdVec::new());

    // Get the loop body region
    let body = loop_op.body(db);
    let body_blocks = body.blocks(db);

    // The body's first block has the loop-carried arguments
    let body_entry = &body_blocks[0];
    let header_id = body_entry.id(db);

    // Create a placeholder header block for successor references
    let header_block_template = Block::new(
        db,
        header_id,
        location,
        body_entry.args(db).clone(),
        IdVec::new(),
    );

    // Deep-replace scf.continue/scf.break with cf.br throughout the body
    // (including inside nested regions like scf.if's then/else)
    let replaced_body =
        replace_continue_break_deep(db, &body, header_block_template, exit_block, location);

    // Now recursively transform the replaced body for remaining scf ops (scf.if, etc.)
    let transformed_body = transform_region(db, &replaced_body);

    // Build entry block: before_ops + cf.br to header with init values
    let init_values: Vec<Value<'db>> = loop_op.init(db).to_vec();
    let br_to_header = cf::br(db, location, init_values, header_block_template);
    let mut entry_ops = before_ops;
    entry_ops.push(br_to_header.as_operation());

    let entry_block = Block::new(
        db,
        orig_block.id(db),
        location,
        orig_block.args(db).clone(),
        entry_ops,
    );

    // Build final exit block with after_ops
    let remapped_after: IdVec<Operation<'db>> = after_ops
        .iter()
        .map(|op| remap_op(db, op, value_map))
        .collect();
    let final_exit = Block::new(
        db,
        exit_id,
        location,
        exit_block.args(db).clone(),
        remapped_after,
    );

    // Assemble: entry, body blocks (from transformed_body), exit
    let mut blocks = vec![entry_block];
    blocks.extend(transformed_body.blocks(db).iter().copied());

    // Recursively transform exit block (may contain more scf ops)
    let exit_transformed = transform_block(db, &final_exit);
    blocks.extend(exit_transformed);

    blocks
}

/// Deep-replace `scf.continue` and `scf.break` with `cf.br` throughout a region,
/// including inside nested regions.
fn replace_continue_break_deep<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    header_block: Block<'db>,
    exit_block: Block<'db>,
    location: Location<'db>,
) -> Region<'db> {
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| {
            let new_ops: IdVec<Operation<'db>> = block
                .operations(db)
                .iter()
                .map(|op| replace_continue_break_op(db, op, header_block, exit_block, location))
                .collect();
            Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                new_ops,
            )
        })
        .collect();
    Region::new(db, region.location(db), new_blocks)
}

/// Replace scf.continue/scf.break in a single op, recursing into nested regions.
fn replace_continue_break_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    header_block: Block<'db>,
    exit_block: Block<'db>,
    location: Location<'db>,
) -> Operation<'db> {
    if scf::Continue::matches(db, *op) {
        let cont_op = scf::Continue::from_operation(db, *op).unwrap();
        let values: Vec<Value<'db>> = cont_op.values(db).to_vec();
        return cf::br(db, location, values, header_block).as_operation();
    }

    if scf::Break::matches(db, *op) {
        let break_op = scf::Break::from_operation(db, *op).unwrap();
        return cf::br(db, location, [break_op.value(db)], exit_block).as_operation();
    }

    // Don't rewrite inside nested loops; their continue/break are handled
    // when that loop is lowered.
    if scf::Loop::matches(db, *op) {
        return *op;
    }

    // Recurse into nested regions (e.g., scf.if's then/else bodies)
    let regions = op.regions(db);
    if regions.is_empty() {
        return *op;
    }

    let new_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|r| replace_continue_break_deep(db, r, header_block, exit_block, location))
        .collect();

    if new_regions == *regions {
        *op
    } else {
        op.modify(db).regions(new_regions).build()
    }
}

/// Lower `scf.switch` to chained cond_br comparisons.
fn lower_scf_switch<'db>(
    db: &'db dyn salsa::Database,
    orig_block: &Block<'db>,
    switch_op: &scf::Switch<'db>,
    before_ops: IdVec<Operation<'db>>,
    after_ops: &[&Operation<'db>],
    value_map: &mut HashMap<Value<'db>, Value<'db>>,
    location: Location<'db>,
) -> Vec<Block<'db>> {
    let discriminant = switch_op.discriminant(db);

    // Collect cases and default from the body region
    let body = switch_op.body(db);
    let body_block = &body.blocks(db)[0];
    let body_ops = body_block.operations(db);

    let mut cases: Vec<(crate::Attribute<'db>, Region<'db>)> = Vec::new();
    let mut default_region: Option<Region<'db>> = None;

    for op in body_ops.iter() {
        if let Ok(case_op) = scf::Case::from_operation(db, *op) {
            cases.push((case_op.value(db).clone(), case_op.body(db)));
        } else if scf::Default::matches(db, *op) {
            let default_op = scf::Default::from_operation(db, *op).unwrap();
            default_region = Some(default_op.body(db));
        }
    }

    // Determine the result type from the switch op
    // scf.switch doesn't have results, but scf.yield inside cases provides them.
    // For now, we handle the case where scf.switch has no direct result.
    // The result comes from the merge block with values from scf.yield in each case.

    // Find result type from first case's yield (if any)
    let result_ty = find_yield_type(db, cases.first().map(|(_, r)| r))
        .or_else(|| find_yield_type(db, default_region.as_ref()));

    // Create merge block
    let merge_id = BlockId::fresh();
    let merge_args: IdVec<BlockArg<'db>> = if let Some(ty) = result_ty {
        idvec![BlockArg::of_type(db, ty)]
    } else {
        IdVec::new()
    };

    let merge_block = Block::new(db, merge_id, location, merge_args.clone(), IdVec::new());

    // Lower each case region to end with cf.br to merge
    let mut case_blocks: Vec<Vec<Block<'db>>> = Vec::new();
    for (_, region) in &cases {
        let blocks = lower_region_to_br(db, region, merge_block, location);
        case_blocks.push(blocks);
    }

    // Lower default region
    let default_blocks = if let Some(ref region) = default_region {
        lower_region_to_br(db, region, merge_block, location)
    } else {
        // No default — create a block that just branches to merge with no args
        assert!(
            result_ty.is_none(),
            "scf.switch with yielding cases must have a default region"
        );
        let default_block = BlockBuilder::new(db, location).build();
        let br_op = cf::br(db, location, std::iter::empty::<Value<'db>>(), merge_block);
        vec![Block::new(
            db,
            default_block.id(db),
            location,
            IdVec::new(),
            idvec![br_op.as_operation()],
        )]
    };

    // Get discriminant type for comparisons
    // Use i32 as default comparison type
    let cmp_ty = core::I1::new(db).as_type();

    // Build chained cond_br blocks
    // For N cases, we need N check blocks (the first reuses orig_block's ID)
    let mut all_blocks = Vec::new();

    if cases.is_empty() {
        // No cases, just branch to default
        let default_entry = default_blocks[0];
        let br_op = cf::br(
            db,
            location,
            std::iter::empty::<Value<'db>>(),
            default_entry,
        );
        let mut entry_ops = before_ops;
        entry_ops.push(br_op.as_operation());

        let entry_block = Block::new(
            db,
            orig_block.id(db),
            location,
            orig_block.args(db).clone(),
            entry_ops,
        );
        all_blocks.push(entry_block);
    } else {
        // Pre-allocate check block IDs for cases after the first.
        // check_block_ids[j] is the ID for the block that checks case j+1.
        let check_block_ids: Vec<BlockId> = (1..cases.len()).map(|_| BlockId::fresh()).collect();

        let mut current_entry_ops = before_ops;

        for (i, ((case_attr, _), case_blks)) in cases.iter().zip(case_blocks.iter()).enumerate() {
            let case_entry = case_blks[0];
            let is_last = i == cases.len() - 1;

            // Create comparison: discriminant == case_value
            let case_const = arith::r#const(
                db,
                location,
                discriminant_type(db, discriminant, orig_block),
                case_attr.clone(),
            );
            let cmp = arith::cmp_eq(db, location, discriminant, case_const.result(db), cmp_ty);

            current_entry_ops.push(case_const.as_operation());
            current_entry_ops.push(cmp.as_operation());

            // Branch: match → case entry, no match → next check (or default for last)
            let else_target = if is_last {
                default_blocks[0]
            } else {
                Block::new(db, check_block_ids[i], location, IdVec::new(), IdVec::new())
            };
            let cond_br_op = cf::cond_br(db, location, cmp.result(db), case_entry, else_target);
            current_entry_ops.push(cond_br_op.as_operation());

            // Finish current check block
            let (block_id, block_args) = if i == 0 {
                (orig_block.id(db), orig_block.args(db).clone())
            } else {
                (check_block_ids[i - 1], IdVec::new())
            };
            let check_block = Block::new(db, block_id, location, block_args, current_entry_ops);
            all_blocks.push(check_block);

            // Add case blocks
            all_blocks.extend(case_blks.iter().copied());

            // Start next check block ops
            current_entry_ops = IdVec::new();
        }
    }

    // Add default blocks
    all_blocks.extend(default_blocks);

    // Build merge block with after ops
    let remapped_after: IdVec<Operation<'db>> = after_ops
        .iter()
        .map(|op| remap_op(db, op, value_map))
        .collect();
    let final_merge = Block::new(db, merge_id, location, merge_args, remapped_after);

    let merge_transformed = transform_block(db, &final_merge);
    all_blocks.extend(merge_transformed);

    all_blocks
}

/// Find the type of values yielded from a region (if any).
fn find_yield_type<'db>(
    db: &'db dyn salsa::Database,
    region: Option<&Region<'db>>,
) -> Option<Type<'db>> {
    let region = region?;
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if scf::Yield::matches(db, *op) {
                let yield_op = scf::Yield::from_operation(db, *op).unwrap();
                let values = yield_op.values(db);
                if !values.is_empty() {
                    // Get the type from the yielded value
                    let val = values[0];
                    return match val.def(db) {
                        ValueDef::OpResult(op) => op.results(db).get(val.index(db)).copied(),
                        ValueDef::BlockArg(block_id) => {
                            // Search for the block with this ID in the region.
                            // NOTE: This only searches the immediate region. BlockArgs from
                            // parent regions won't be found and will return None. This is
                            // acceptable because yield values typically reference local values.
                            for b in region.blocks(db).iter() {
                                if b.id(db) == block_id {
                                    return Some(b.arg_ty(db, val.index(db)));
                                }
                            }
                            None
                        }
                    };
                }
            }
        }
    }
    None
}

/// Get the type of a discriminant value.
fn discriminant_type<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    block: &Block<'db>,
) -> Type<'db> {
    match value.def(db) {
        ValueDef::OpResult(op) => op
            .results(db)
            .get(value.index(db))
            .copied()
            .expect("discriminant result index out of bounds"),
        ValueDef::BlockArg(block_id) => {
            assert_eq!(
                block.id(db),
                block_id,
                "discriminant BlockArg must belong to the current block"
            );
            block.arg_ty(db, value.index(db))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core, func, scf};
    use crate::parser::parse_test_module;
    use crate::{Attribute, DialectOp, DialectType, PathId, Span};
    use salsa_test_macros::salsa_test;

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Recursively collect all operation names from a region.
    fn collect_all_op_names<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
    ) -> Vec<String> {
        let mut names = Vec::new();
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                names.push(op.full_name(db));
                for nested_region in op.regions(db).iter() {
                    names.extend(collect_all_op_names(db, nested_region));
                }
            }
        }
        names
    }

    // === scf.if tests ===

    /// Create module: scf.if with simple then/else yielding constants.
    #[salsa::tracked]
    fn make_scf_if_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test {
  %0 = arith.const {value = true} : core.i1
  %1 = scf.if %0 : core.i32 {
    %2 = arith.const {value = 42} : core.i32
    scf.yield %2
  } {
    %3 = arith.const {value = 0} : core.i32
    scf.yield %3
  }
  func.return %1
}"#,
        )
    }

    /// Lower module and return (block_count, all_op_names, per-block last op names, merge_block_arg_count).
    #[salsa::tracked]
    fn lower_and_check_if(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, Vec<String>, Vec<String>, usize) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);
        let block_count = blocks.len();
        let all_ops = collect_all_op_names(db, &body);
        let last_ops: Vec<String> = blocks
            .iter()
            .filter_map(|b| b.operations(db).last().map(|op| op.full_name(db)))
            .collect();
        let merge_arg_count = blocks.last().map(|b| b.args(db).len()).unwrap_or(0);
        (block_count, all_ops, last_ops, merge_arg_count)
    }

    #[salsa_test]
    fn test_scf_if_basic(db: &salsa::DatabaseImpl) {
        let module = make_scf_if_module(db);
        let (block_count, all_ops, last_ops, merge_arg_count) = lower_and_check_if(db, module);

        // Should have 4 blocks: entry, then, else, merge
        assert_eq!(block_count, 4, "Expected 4 blocks, got {}", block_count);

        // Entry block should end with cf.cond_br
        assert_eq!(
            last_ops[0], "cf.cond_br",
            "Entry should end with cf.cond_br"
        );

        // Then and else blocks should end with cf.br
        assert_eq!(last_ops[1], "cf.br", "Then block should end with cf.br");
        assert_eq!(last_ops[2], "cf.br", "Else block should end with cf.br");

        // Merge block should end with func.return
        assert_eq!(
            last_ops[3], "func.return",
            "Merge should end with func.return"
        );

        // Merge block should have one block argument (the if result)
        assert_eq!(
            merge_arg_count, 1,
            "Merge block should have 1 block argument"
        );

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain, but found: {:?}",
            all_ops
        );
    }

    // === scf.if with ops before and after ===

    #[salsa::tracked]
    fn make_scf_if_with_surrounding_ops(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        // op_before: const 10
        let before_const = arith::Const::i32(db, location, 10);

        // Condition
        let cond = arith::r#const(db, location, i1_ty, Attribute::Bool(true));

        // Then: yield 42
        let then_const = arith::Const::i32(db, location, 42);
        let then_yield = scf::r#yield(db, location, [then_const.result(db)]);
        let then_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![then_const.as_operation(), then_yield.as_operation()],
        );
        let then_region = Region::new(db, location, idvec![then_block]);

        // Else: yield 0
        let else_const = arith::Const::i32(db, location, 0);
        let else_yield = scf::r#yield(db, location, [else_const.result(db)]);
        let else_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![else_const.as_operation(), else_yield.as_operation()],
        );
        let else_region = Region::new(db, location, idvec![else_block]);

        let if_op = scf::r#if(
            db,
            location,
            cond.result(db),
            i32_ty,
            then_region,
            else_region,
        );

        // op_after: add(before_const, if_result)
        let add_op = arith::add(
            db,
            location,
            before_const.result(db),
            if_op.result(db),
            i32_ty,
        );
        let ret = func::r#return(db, location, [add_op.result(db)]);

        let entry = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                before_const.as_operation(),
                cond.as_operation(),
                if_op.as_operation(),
                add_op.as_operation(),
                ret.as_operation()
            ],
        );
        let body = Region::new(db, location, idvec![entry]);
        core::Module::create(db, location, "test".into(), body)
    }

    /// Lower module and return detailed info for surrounding ops test.
    /// Returns (block_count, entry_op_count, entry_last_op, merge_op_names, add_rhs_is_block_arg).
    #[salsa::tracked]
    fn lower_and_check_surrounding(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, usize, String, Vec<String>, bool) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);

        let entry_ops = blocks[0].operations(db);
        let entry_last = entry_ops
            .last()
            .map(|op| op.full_name(db))
            .unwrap_or_default();

        let merge_ops = blocks[3].operations(db);
        let merge_names: Vec<String> = merge_ops.iter().map(|op| op.full_name(db)).collect();

        // Check if the add op's rhs references a merge block arg
        let add_rhs_is_block_arg = merge_ops
            .iter()
            .find(|op| arith::Add::matches(db, **op))
            .map(|op| {
                let add = arith::Add::from_operation(db, *op).unwrap();
                matches!(add.rhs(db).def(db), ValueDef::BlockArg(_))
            })
            .unwrap_or(false);

        (
            blocks.len(),
            entry_ops.len(),
            entry_last,
            merge_names,
            add_rhs_is_block_arg,
        )
    }

    #[salsa_test]
    fn test_scf_if_with_ops_before_and_after(db: &salsa::DatabaseImpl) {
        let module = make_scf_if_with_surrounding_ops(db);
        let (block_count, entry_op_count, entry_last, merge_names, add_rhs_is_block_arg) =
            lower_and_check_surrounding(db, module);

        assert_eq!(block_count, 4, "Expected 4 blocks");
        assert_eq!(entry_op_count, 3, "Entry: before_const, cond, cf.cond_br");
        assert_eq!(entry_last, "cf.cond_br");

        assert!(
            merge_names.iter().any(|n| n == "arith.add"),
            "Merge block should have arith.add, got: {:?}",
            merge_names
        );
        assert!(
            merge_names.iter().any(|n| n == "func.return"),
            "Merge block should have func.return, got: {:?}",
            merge_names
        );

        assert!(
            add_rhs_is_block_arg,
            "Add rhs should reference merge block arg, not old scf.if result"
        );
    }

    // === scf.loop tests ===

    #[salsa::tracked]
    fn make_scf_loop_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_loop {
  %init = arith.const {value = 0} : core.i32
  %0 = scf.loop %init : core.i32 {
    ^bb0(%loop_var: core.i32):
      %limit = arith.const {value = 10} : core.i32
      %cond = arith.cmp_lt %loop_var, %limit : core.i1
      %1 = scf.if %cond : core.i32 {
        %one = arith.const {value = 1} : core.i32
        %next = arith.add %loop_var, %one : core.i32
        scf.continue %next
      } {
        scf.break %loop_var
      }
  }
  func.return %0
}"#,
        )
    }

    /// Lower loop module and return (block_count, all_op_names).
    #[salsa::tracked]
    fn lower_and_check_loop(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, Vec<String>) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);
        let all_ops = collect_all_op_names(db, &body);
        (blocks.len(), all_ops)
    }

    #[salsa_test]
    fn test_scf_loop_basic(db: &salsa::DatabaseImpl) {
        let module = make_scf_loop_module(db);
        let (block_count, all_ops) = lower_and_check_loop(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain, found: {:?}",
            all_ops
        );

        // Should have cf.br and cf.cond_br operations
        assert!(
            all_ops.iter().any(|n| n == "cf.br"),
            "Should have cf.br, got: {:?}",
            all_ops
        );
        assert!(
            all_ops.iter().any(|n| n == "cf.cond_br"),
            "Should have cf.cond_br, got: {:?}",
            all_ops
        );

        // Should have at least: entry, header, then, else, merge(for if), exit
        assert!(
            block_count >= 4,
            "Expected at least 4 blocks, got {}",
            block_count
        );
    }

    // === Nested scf.if tests ===

    /// Create module with nested scf.if: outer if → then has inner if.
    /// Simulates case matching patterns like `if x then (if y then a else b) else c`.
    #[salsa::tracked]
    fn make_nested_scf_if_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_nested_if {
  %outer_cond = arith.const {value = true} : core.i1
  %0 = scf.if %outer_cond : core.i32 {
    %inner_cond = arith.const {value = false} : core.i1
    %1 = scf.if %inner_cond : core.i32 {
      %2 = arith.const {value = 1} : core.i32
      scf.yield %2
    } {
      %3 = arith.const {value = 2} : core.i32
      scf.yield %3
    }
    scf.yield %1
  } {
    %4 = arith.const {value = 3} : core.i32
    scf.yield %4
  }
  func.return %0
}"#,
        )
    }

    #[salsa::tracked]
    fn lower_and_check_nested_if(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, Vec<String>, Vec<String>) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);
        let all_ops = collect_all_op_names(db, &body);
        let last_ops: Vec<String> = blocks
            .iter()
            .filter_map(|b| b.operations(db).last().map(|op| op.full_name(db)))
            .collect();
        (blocks.len(), all_ops, last_ops)
    }

    #[salsa_test]
    fn test_scf_if_nested(db: &salsa::DatabaseImpl) {
        let module = make_nested_scf_if_module(db);
        let (block_count, all_ops, last_ops) = lower_and_check_nested_if(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain after nested lowering, found: {:?}",
            all_ops
        );

        // Should have at least 7 blocks:
        // outer_entry, outer_then_entry(=inner_entry), inner_then, inner_else,
        // inner_merge(→outer_then_merge), outer_else, outer_merge
        assert!(
            block_count >= 7,
            "Expected at least 7 blocks for nested if, got {}",
            block_count
        );

        // All blocks should end with a branch op or func.return
        for (i, last_op) in last_ops.iter().enumerate() {
            assert!(
                last_op == "cf.cond_br" || last_op == "cf.br" || last_op == "func.return",
                "Block {} should end with a branch or return, got: {}",
                i,
                last_op
            );
        }

        // Exactly one func.return should exist
        let return_count = all_ops
            .iter()
            .filter(|n| n.as_str() == "func.return")
            .count();
        assert_eq!(return_count, 1, "Should have exactly 1 func.return");
    }

    // === scf.loop break/continue test ===

    /// Lower loop module and return detailed info:
    /// (block_count, all_op_names, per_block_last_ops, header_has_block_arg).
    #[salsa::tracked]
    fn lower_and_check_loop_detail(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, Vec<String>, Vec<String>, bool) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);
        let all_ops = collect_all_op_names(db, &body);
        let last_ops: Vec<String> = blocks
            .iter()
            .filter_map(|b| b.operations(db).last().map(|op| op.full_name(db)))
            .collect();

        // The header block is the one with a block argument (loop-carried variable)
        // It should be the second block (after entry which has cf.br to header)
        let header_has_block_arg = blocks
            .get(1)
            .map(|b| !b.args(db).is_empty())
            .unwrap_or(false);

        (blocks.len(), all_ops, last_ops, header_has_block_arg)
    }

    #[salsa_test]
    fn test_scf_loop_break_continue(db: &salsa::DatabaseImpl) {
        let module = make_scf_loop_module(db);
        let (block_count, all_ops, last_ops, header_has_block_arg) =
            lower_and_check_loop_detail(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain, found: {:?}",
            all_ops
        );

        // Header block should have a block argument (loop-carried variable)
        assert!(
            header_has_block_arg,
            "Header block should have a block argument for loop-carried variable"
        );

        // Entry block should end with cf.br (to header)
        assert_eq!(
            last_ops[0], "cf.br",
            "Entry block should end with cf.br to header"
        );

        // Count cf.br ops: at least 3 (entry→header, continue→header, break→exit)
        let br_count = all_ops.iter().filter(|n| n.as_str() == "cf.br").count();
        assert!(
            br_count >= 3,
            "Expected at least 3 cf.br (entry→header, continue→header, break→exit), got {}",
            br_count
        );

        // Should have cf.cond_br for the loop condition
        let cond_br_count = all_ops
            .iter()
            .filter(|n| n.as_str() == "cf.cond_br")
            .count();
        assert!(
            cond_br_count >= 1,
            "Expected at least 1 cf.cond_br for loop condition, got {}",
            cond_br_count
        );

        // Last block should be the exit block with func.return
        assert_eq!(
            last_ops.last().unwrap(),
            "func.return",
            "Last block should end with func.return"
        );

        // Verify block count is reasonable (entry, header, then/else from if, merge, exit)
        assert!(
            block_count >= 5,
            "Expected at least 5 blocks, got {}",
            block_count
        );
    }

    // === scf.switch test ===

    /// Create module with scf.switch: 2 cases + default.
    #[salsa::tracked]
    fn make_scf_switch_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_switch {
  %disc = arith.const {value = 1} : core.i32
  scf.switch %disc {
    scf.case {value = 0} {
      %0 = arith.const {value = 10} : core.i32
      scf.yield %0
    }
    scf.case {value = 1} {
      %1 = arith.const {value = 20} : core.i32
      scf.yield %1
    }
    scf.default {
      %2 = arith.const {value = 99} : core.i32
      scf.yield %2
    }
  }
  func.return
}"#,
        )
    }

    #[salsa::tracked]
    fn lower_and_check_switch(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, Vec<String>, Vec<String>) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);
        let all_ops = collect_all_op_names(db, &body);
        let last_ops: Vec<String> = blocks
            .iter()
            .filter_map(|b| b.operations(db).last().map(|op| op.full_name(db)))
            .collect();
        (blocks.len(), all_ops, last_ops)
    }

    // === Consecutive scf.if test (edge case: scf ops in after_ops) ===

    /// Create module with two consecutive scf.if ops in the same block.
    /// The second scf.if appears in the after_ops of the first, testing that
    /// merge block successor references remain valid after transformation.
    #[salsa::tracked]
    fn make_consecutive_scf_if_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_consecutive_if {
  %cond1 = arith.const {value = true} : core.i1
  %0 = scf.if %cond1 : core.i32 {
    %1 = arith.const {value = 1} : core.i32
    scf.yield %1
  } {
    %2 = arith.const {value = 2} : core.i32
    scf.yield %2
  }
  %cond2 = arith.const {value = false} : core.i1
  %3 = scf.if %cond2 : core.i32 {
    %4 = arith.const {value = 10} : core.i32
    scf.yield %4
  } {
    %5 = arith.const {value = 20} : core.i32
    scf.yield %5
  }
  %6 = arith.add %0, %3 : core.i32
  func.return %6
}"#,
        )
    }

    #[salsa::tracked]
    fn lower_and_check_consecutive_if(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, Vec<String>, Vec<String>) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);
        let all_ops = collect_all_op_names(db, &body);
        let last_ops: Vec<String> = blocks
            .iter()
            .filter_map(|b| b.operations(db).last().map(|op| op.full_name(db)))
            .collect();
        (blocks.len(), all_ops, last_ops)
    }

    #[salsa_test]
    fn test_consecutive_scf_if(db: &salsa::DatabaseImpl) {
        let module = make_consecutive_scf_if_module(db);
        let (block_count, all_ops, last_ops) = lower_and_check_consecutive_if(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain after consecutive if lowering, found: {:?}",
            all_ops
        );

        // Should have at least 8 blocks:
        // entry, then1, else1, merge1(=entry2), then2, else2, merge2(with add+return)
        // (merge1 may split further)
        assert!(
            block_count >= 7,
            "Expected at least 7 blocks for consecutive ifs, got {}",
            block_count
        );

        // All blocks should end with branch or return
        for (i, last_op) in last_ops.iter().enumerate() {
            assert!(
                last_op == "cf.cond_br" || last_op == "cf.br" || last_op == "func.return",
                "Block {} should end with a branch or return, got: {}",
                i,
                last_op
            );
        }

        // Exactly one func.return should exist
        let return_count = all_ops
            .iter()
            .filter(|n| n.as_str() == "func.return")
            .count();
        assert_eq!(return_count, 1, "Should have exactly 1 func.return");

        // Should have arith.add in the final merge block
        assert!(
            all_ops.iter().any(|n| n == "arith.add"),
            "Should have arith.add for combining results, got: {:?}",
            all_ops
        );
    }

    // === scf.switch with 3 cases (exercises pre-allocated BlockId chain) ===

    /// Create module with scf.switch: 3 cases + default.
    /// This tests that the pre-allocated check block IDs form a valid chain
    /// across multiple intermediate cond_br blocks.
    #[salsa::tracked]
    fn make_scf_switch_3cases_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_switch_3cases {
  %disc = arith.const {value = 2} : core.i32
  scf.switch %disc {
    scf.case {value = 0} {
      %0 = arith.const {value = 100} : core.i32
      scf.yield %0
    }
    scf.case {value = 1} {
      %1 = arith.const {value = 200} : core.i32
      scf.yield %1
    }
    scf.case {value = 2} {
      %2 = arith.const {value = 300} : core.i32
      scf.yield %2
    }
    scf.default {
      %3 = arith.const {value = 999} : core.i32
      scf.yield %3
    }
  }
  func.return
}"#,
        )
    }

    #[salsa_test]
    fn test_scf_switch_3cases(db: &salsa::DatabaseImpl) {
        let module = make_scf_switch_3cases_module(db);
        let (block_count, all_ops, last_ops) = lower_and_check_switch(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain after 3-case switch lowering, found: {:?}",
            all_ops
        );

        // Should have 3 arith.cmp_eq (one per case)
        let cmp_eq_count = all_ops
            .iter()
            .filter(|n| n.as_str() == "arith.cmp_eq")
            .count();
        assert_eq!(
            cmp_eq_count, 3,
            "Expected 3 arith.cmp_eq (one per case), got {}",
            cmp_eq_count
        );

        // Should have 3 cf.cond_br (one per case check)
        let cond_br_count = all_ops
            .iter()
            .filter(|n| n.as_str() == "cf.cond_br")
            .count();
        assert_eq!(
            cond_br_count, 3,
            "Expected 3 cf.cond_br (one per case check), got {}",
            cond_br_count
        );

        // Should have cf.br for each case + default → merge (4)
        let br_count = all_ops.iter().filter(|n| n.as_str() == "cf.br").count();
        assert!(br_count >= 4, "Expected at least 4 cf.br, got {}", br_count);

        // Should have at least 8 blocks:
        // entry(check0), case0, check1, case1, check2, case2, default, merge
        assert!(
            block_count >= 8,
            "Expected at least 8 blocks for 3-case switch, got {}",
            block_count
        );

        // All blocks should end with branch or return
        for (i, last_op) in last_ops.iter().enumerate() {
            assert!(
                last_op == "cf.cond_br" || last_op == "cf.br" || last_op == "func.return",
                "Block {} should end with a branch or return, got: {}",
                i,
                last_op
            );
        }

        // Last block should end with func.return
        assert_eq!(
            last_ops.last().unwrap(),
            "func.return",
            "Last block should end with func.return"
        );
    }

    // === scf.switch with block argument discriminant ===

    /// Create module where switch discriminant is a block argument,
    /// testing that discriminant_type resolves the actual type.
    #[salsa::tracked]
    fn make_scf_switch_block_arg_disc_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_switch_block_arg {
  ^bb0(%disc: core.i32):
    scf.switch %disc {
      scf.case {value = 0} {
        %0 = arith.const {value = 10} : core.i32
        scf.yield %0
      }
      scf.default {
        %1 = arith.const {value = 99} : core.i32
        scf.yield %1
      }
    }
    func.return
}"#,
        )
    }

    #[salsa_test]
    fn test_scf_switch_block_arg_discriminant(db: &salsa::DatabaseImpl) {
        let module = make_scf_switch_block_arg_disc_module(db);
        let (block_count, all_ops, last_ops) = lower_and_check_switch(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain, found: {:?}",
            all_ops
        );

        // Should have arith.const for comparison value (discriminant_type resolved correctly)
        assert!(
            all_ops.iter().any(|n| n == "arith.const"),
            "Should have arith.const for case comparison, got: {:?}",
            all_ops
        );

        // Should have arith.cmp_eq
        assert!(
            all_ops.iter().any(|n| n == "arith.cmp_eq"),
            "Should have arith.cmp_eq, got: {:?}",
            all_ops
        );

        // Should have at least 4 blocks: entry, case0, default, merge
        assert!(
            block_count >= 4,
            "Expected at least 4 blocks, got {}",
            block_count
        );

        // Last block should end with func.return
        assert_eq!(
            last_ops.last().unwrap(),
            "func.return",
            "Last block should end with func.return"
        );
    }

    #[salsa_test]
    fn test_scf_switch_basic(db: &salsa::DatabaseImpl) {
        let module = make_scf_switch_module(db);
        let (block_count, all_ops, last_ops) = lower_and_check_switch(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain after switch lowering, found: {:?}",
            all_ops
        );

        // Should have arith.cmp_eq for each case comparison (2 cases)
        let cmp_eq_count = all_ops
            .iter()
            .filter(|n| n.as_str() == "arith.cmp_eq")
            .count();
        assert_eq!(
            cmp_eq_count, 2,
            "Expected 2 arith.cmp_eq (one per case), got {}",
            cmp_eq_count
        );

        // Should have cf.cond_br for each case check (2)
        let cond_br_count = all_ops
            .iter()
            .filter(|n| n.as_str() == "cf.cond_br")
            .count();
        assert_eq!(
            cond_br_count, 2,
            "Expected 2 cf.cond_br (one per case check), got {}",
            cond_br_count
        );

        // Should have cf.br for each case + default → merge (3)
        let br_count = all_ops.iter().filter(|n| n.as_str() == "cf.br").count();
        assert!(
            br_count >= 3,
            "Expected at least 3 cf.br (case0→merge, case1→merge, default→merge), got {}",
            br_count
        );

        // Should have multiple blocks:
        // entry(check0), check1, case0, case1, default, merge
        assert!(
            block_count >= 6,
            "Expected at least 6 blocks for 2-case switch, got {}",
            block_count
        );

        // Last block should end with func.return
        assert_eq!(
            last_ops.last().unwrap(),
            "func.return",
            "Last block should end with func.return"
        );
    }

    // === value_map propagation into nested regions ===

    /// Create module where an scf.if result is captured inside a func.func body
    /// (nested region). After lowering, the reference inside the func body must
    /// be remapped from the old scf.if result to the merge block argument.
    ///
    /// ```text
    /// %cond  = arith.const true
    /// %x     = scf.if(%cond) { yield 42 } { yield 0 }
    /// func.func @inner() {
    ///     func.return(%x)          // captures %x from outer scope
    /// }
    /// func.return(%x)
    /// ```
    #[salsa::tracked]
    fn make_scf_if_with_nested_capture_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_nested_capture {
  %cond = arith.const {value = true} : core.i1
  %x = scf.if %cond : core.i32 {
    %0 = arith.const {value = 42} : core.i32
    scf.yield %0
  } {
    %1 = arith.const {value = 0} : core.i32
    scf.yield %1
  }
  func.func @inner() -> core.nil {
    func.return %x
  }
  func.return %x
}"#,
        )
    }

    /// Recursively collect all values referenced as operands in a region.
    fn collect_all_operand_values<'db>(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
    ) -> Vec<Value<'db>> {
        let mut values = Vec::new();
        for block in region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                values.extend(op.operands(db).iter().copied());
                for nested in op.regions(db).iter() {
                    values.extend(collect_all_operand_values(db, nested));
                }
            }
        }
        values
    }

    /// Lower the nested-capture module and verify value_map propagation.
    /// Returns (has_no_scf, merge_has_arg, inner_body_refs_merge_arg, outer_ret_refs_merge_arg).
    #[salsa::tracked]
    fn lower_and_check_nested_capture(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (bool, bool, bool, bool) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);

        let all_ops = collect_all_op_names(db, &body);
        let has_no_scf = !all_ops.iter().any(|n| n.starts_with("scf."));

        // Find the merge block (last block, should contain func.func + func.return)
        let merge_block = blocks.last().unwrap();
        let merge_has_arg = !merge_block.args(db).is_empty();

        if !merge_has_arg {
            return (has_no_scf, false, false, false);
        }

        let merge_arg = Value::new(db, ValueDef::BlockArg(merge_block.id(db)), 0);

        // Check func.func body references merge block arg
        let inner_body_refs_merge_arg = merge_block
            .operations(db)
            .iter()
            .find(|op| func::Func::matches(db, **op))
            .map(|op| {
                let inner_func = func::Func::from_operation(db, *op).unwrap();
                let inner_body = inner_func.body(db);
                let inner_operands = collect_all_operand_values(db, &inner_body);
                inner_operands.contains(&merge_arg)
            })
            .unwrap_or(false);

        // Check outer func.return references merge block arg
        let outer_ret_refs_merge_arg = merge_block
            .operations(db)
            .iter()
            .find(|op| func::Return::matches(db, **op))
            .map(|op| op.operands(db).contains(&merge_arg))
            .unwrap_or(false);

        (
            has_no_scf,
            merge_has_arg,
            inner_body_refs_merge_arg,
            outer_ret_refs_merge_arg,
        )
    }

    #[salsa_test]
    fn test_value_map_propagates_into_nested_regions(db: &salsa::DatabaseImpl) {
        let module = make_scf_if_with_nested_capture_module(db);
        let (has_no_scf, merge_has_arg, inner_refs, outer_refs) =
            lower_and_check_nested_capture(db, module);

        assert!(has_no_scf, "No scf ops should remain after lowering");
        assert!(
            merge_has_arg,
            "Merge block should have a block argument for the if result"
        );
        assert!(
            inner_refs,
            "func.func body should reference merge block arg after value_map propagation"
        );
        assert!(
            outer_refs,
            "Outer func.return should reference merge block arg"
        );
    }

    // === switch with default only (no cases) ===

    /// Create module with scf.switch that has no cases, only a default branch
    /// that yields a value. Tests that result_ty falls back to default_region.
    ///
    /// ```text
    /// %disc   = arith.const 0
    /// %result = scf.switch(%disc) { default { yield 99 } }
    /// func.return(%result)
    /// ```
    #[salsa::tracked]
    fn make_scf_switch_default_only_module(db: &dyn salsa::Database) -> core::Module<'_> {
        parse_test_module(
            db,
            r#"core.module @test_switch_default_only {
  %disc = arith.const {value = 0} : core.i32
  scf.switch %disc {
    scf.default {
      %0 = arith.const {value = 99} : core.i32
      scf.yield %0
    }
  }
  func.return
}"#,
        )
    }

    /// Lower the default-only switch module and return check results.
    /// Returns (block_count, all_ops, has_cmp_eq, has_cond_br, merge_arg_count, last_op_name).
    #[salsa::tracked]
    fn lower_and_check_switch_default_only(
        db: &dyn salsa::Database,
        module: core::Module<'_>,
    ) -> (usize, Vec<String>, bool, bool, usize, String) {
        let lowered = lower_scf_to_cf(db, module);
        let body = lowered.body(db);
        let blocks = body.blocks(db);
        let all_ops = collect_all_op_names(db, &body);

        let has_cmp_eq = all_ops.iter().any(|n| n == "arith.cmp_eq");
        let has_cond_br = all_ops.iter().any(|n| n == "cf.cond_br");

        let merge_block = blocks.last().unwrap();
        let merge_arg_count = merge_block.args(db).len();
        let last_op_name = merge_block
            .operations(db)
            .last()
            .map(|op| op.full_name(db))
            .unwrap_or_default();

        (
            blocks.len(),
            all_ops,
            has_cmp_eq,
            has_cond_br,
            merge_arg_count,
            last_op_name,
        )
    }

    #[salsa_test]
    fn test_scf_switch_default_only(db: &salsa::DatabaseImpl) {
        let module = make_scf_switch_default_only_module(db);
        let (block_count, all_ops, has_cmp_eq, has_cond_br, merge_arg_count, last_op_name) =
            lower_and_check_switch_default_only(db, module);

        // No scf ops should remain
        assert!(
            !all_ops.iter().any(|n| n.starts_with("scf.")),
            "No scf ops should remain, found: {:?}",
            all_ops
        );

        // With no cases, entry should branch directly to default
        // Blocks: entry, default, merge
        assert!(
            block_count >= 3,
            "Expected at least 3 blocks (entry, default, merge), got {}",
            block_count
        );

        // Should have no cmp_eq or cond_br (no cases to compare)
        assert!(!has_cmp_eq, "Should have no arith.cmp_eq with no cases");
        assert!(!has_cond_br, "Should have no cf.cond_br with no cases");

        // Default block should yield via cf.br to merge
        assert!(
            all_ops.iter().any(|n| n == "cf.br"),
            "Should have cf.br from default to merge"
        );

        // Merge block should have a block argument (from default's yield type)
        assert_eq!(
            merge_arg_count, 1,
            "Merge block should have 1 block arg from default yield type, got {}",
            merge_arg_count
        );

        // Last block should end with func.return
        assert_eq!(
            last_op_name, "func.return",
            "Last block should end with func.return"
        );
    }

    // === discriminant_type: cross-block BlockArg should panic ===

    #[salsa::tracked]
    fn call_discriminant_type_cross_block(db: &dyn salsa::Database) {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let block_id = BlockId::fresh();
        let other_id = BlockId::fresh();

        let block = Block::new(
            db,
            block_id,
            location,
            idvec![BlockArg::of_type(db, i32_ty)],
            idvec![],
        );

        // Value referring to a different block's argument
        let cross_block_value = Value::new(db, ValueDef::BlockArg(other_id), 0);

        // Should panic because the BlockArg doesn't belong to `block`
        let _ = discriminant_type(db, cross_block_value, &block);
    }

    #[salsa_test]
    #[should_panic(expected = "discriminant BlockArg must belong to the current block")]
    fn test_discriminant_type_panics_on_cross_block_arg(db: &salsa::DatabaseImpl) {
        call_discriminant_type_cross_block(db);
    }

    // === find_yield_type: BlockArg from outside region returns None ===

    #[salsa::tracked]
    fn call_find_yield_type_outside_block_arg(db: &dyn salsa::Database) -> bool {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Create a BlockArg value referring to a block outside the region
        let outer_block_id = BlockId::fresh();
        let outer_value = Value::new(db, ValueDef::BlockArg(outer_block_id), 0);

        // Build a region whose only block yields that outer value
        let yield_op = scf::r#yield(db, location, [outer_value]);
        let inner_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![BlockArg::of_type(db, i32_ty)],
            idvec![yield_op.as_operation()],
        );
        let region = Region::new(db, location, idvec![inner_block]);

        // find_yield_type cannot resolve the type because the defining block
        // is not in this region
        find_yield_type(db, Some(&region)).is_none()
    }

    #[salsa_test]
    fn test_find_yield_type_outside_block_arg_returns_none(db: &salsa::DatabaseImpl) {
        assert!(
            call_find_yield_type_outside_block_arg(db),
            "find_yield_type should return None for BlockArg from outside the region"
        );
    }
}
