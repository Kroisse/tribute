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
    transform_op_regions(db, &remapped)
}

/// Build the merge block with remaining ops, remapping values.
fn build_merge_block<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    merge_args: IdVec<BlockArg<'db>>,
    after_ops: &[&Operation<'db>],
    value_map: &HashMap<Value<'db>, Value<'db>>,
) -> Block<'db> {
    let merge_id = BlockId::fresh();

    // Build value map for merge block args from the scf result values
    // The value_map already maps scf result → placeholder, but we need to update
    // it to map to the actual merge block args
    let full_map = value_map.clone();

    // Map any values that point to merge block arg placeholders
    // Actually, the caller sets up value_map to map scf result → merge block arg Value
    // We just need to use the map as-is

    let remapped_ops: IdVec<Operation<'db>> = after_ops
        .iter()
        .map(|op| remap_op(db, op, &full_map))
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
    let result_ty = if_op.as_operation().results(db).first().copied();

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
    let merge_block_inner = build_merge_block(db, location, merge_args, after_ops, value_map);
    // We need to set the merge_id on it
    let merge_block = Block::new(
        db,
        merge_id,
        location,
        merge_block_inner.args(db).clone(),
        merge_block_inner.operations(db).clone(),
    );

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
    let result_ty = loop_op.as_operation().results(db).first().copied();

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
    let result_ty = find_yield_type(db, cases.first().map(|(_, r)| r));

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
    // For N cases, we need N check blocks (the first is the entry block)
    let mut all_blocks = Vec::new();

    // Build from the last case backwards so each check block knows its "else" target
    // Actually, let's build forward with entry block containing before_ops + first check

    let mut current_entry_ops = before_ops;

    for (i, ((case_attr, _), case_blks)) in cases.iter().zip(case_blocks.iter()).enumerate() {
        let case_entry = case_blks[0];
        let is_last = i == cases.len() - 1;

        // Create comparison: discriminant == case_value
        let case_const = arith::r#const(
            db,
            location,
            discriminant_type(db, discriminant),
            case_attr.clone(),
        );
        let cmp = arith::cmp_eq(db, location, discriminant, case_const.result(db), cmp_ty);

        current_entry_ops.push(case_const.as_operation());
        current_entry_ops.push(cmp.as_operation());

        if is_last {
            // Last case: else goes to default
            let default_entry = default_blocks[0];
            let cond_br_op = cf::cond_br(db, location, cmp.result(db), case_entry, default_entry);
            current_entry_ops.push(cond_br_op.as_operation());
        } else {
            // Not the last case: else goes to next check block
            let next_check_id = BlockId::fresh();
            let next_check_placeholder =
                Block::new(db, next_check_id, location, IdVec::new(), IdVec::new());

            let cond_br_op = cf::cond_br(
                db,
                location,
                cmp.result(db),
                case_entry,
                next_check_placeholder,
            );
            current_entry_ops.push(cond_br_op.as_operation());

            // Finish current check block
            if i == 0 {
                // First check block is the entry block
                let entry_block = Block::new(
                    db,
                    orig_block.id(db),
                    location,
                    orig_block.args(db).clone(),
                    current_entry_ops,
                );
                all_blocks.push(entry_block);
            } else {
                let check_block = Block::new(
                    db,
                    all_blocks
                        .last()
                        .map(|_| BlockId::fresh())
                        .unwrap_or(orig_block.id(db)),
                    location,
                    IdVec::new(),
                    current_entry_ops,
                );
                all_blocks.push(check_block);
            }

            // Start next check block
            current_entry_ops = IdVec::new();

            // Add case blocks
            all_blocks.extend(case_blks.iter().copied());
        }
    }

    if cases.is_empty() {
        // No cases, just branch to default
        let default_entry = default_blocks[0];
        let br_op = cf::br(
            db,
            location,
            std::iter::empty::<Value<'db>>(),
            default_entry,
        );
        current_entry_ops.push(br_op.as_operation());

        let entry_block = Block::new(
            db,
            orig_block.id(db),
            location,
            orig_block.args(db).clone(),
            current_entry_ops,
        );
        all_blocks.push(entry_block);
    } else {
        // Add the last entry/check block
        let last_block_id = if all_blocks.is_empty() {
            orig_block.id(db)
        } else {
            BlockId::fresh()
        };
        let last_check = Block::new(
            db,
            last_block_id,
            location,
            if all_blocks.is_empty() {
                orig_block.args(db).clone()
            } else {
                IdVec::new()
            },
            current_entry_ops,
        );
        all_blocks.push(last_check);

        // Add last case blocks
        if let Some(last_case_blks) = case_blocks.last() {
            all_blocks.extend(last_case_blks.iter().copied());
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
                            // Search for the block with this ID in the region
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
fn discriminant_type<'db>(db: &'db dyn salsa::Database, value: Value<'db>) -> Type<'db> {
    match value.def(db) {
        ValueDef::OpResult(op) => op
            .results(db)
            .get(value.index(db))
            .copied()
            .unwrap_or_else(|| core::I32::new(db).as_type()),
        ValueDef::BlockArg(_) => {
            // Fallback to i32
            core::I32::new(db).as_type()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core, func, scf};
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
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        // Condition
        let cond = arith::r#const(db, location, i1_ty, Attribute::Bool(true));

        // Then region: yield 42
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

        // Else region: yield 0
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

        // scf.if
        let if_op = scf::r#if(
            db,
            location,
            cond.result(db),
            i32_ty,
            then_region,
            else_region,
        );

        // Return the if result
        let ret = func::r#return(db, location, [if_op.result(db)]);

        let entry = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                cond.as_operation(),
                if_op.as_operation(),
                ret.as_operation()
            ],
        );
        let body = Region::new(db, location, idvec![entry]);
        core::Module::create(db, location, "test".into(), body)
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
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        // Init value
        let init = arith::Const::i32(db, location, 0);

        // Loop body: receives loop var, condition check, break or continue
        let body_id = BlockId::fresh();
        let body_arg = BlockArg::of_type(db, i32_ty);
        let loop_var = Value::new(db, ValueDef::BlockArg(body_id), 0);

        // Condition: loop_var < 10
        let limit = arith::Const::i32(db, location, 10);
        let cond = arith::cmp_lt(db, location, loop_var, limit.result(db), i1_ty);

        // Then: continue with loop_var + 1
        let one = arith::Const::i32(db, location, 1);
        let next = arith::add(db, location, loop_var, one.result(db), i32_ty);
        let cont_op = scf::r#continue(db, location, vec![next.result(db)]);
        let then_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                one.as_operation(),
                next.as_operation(),
                cont_op.as_operation()
            ],
        );
        let then_region = Region::new(db, location, idvec![then_block]);

        // Else: break with loop_var
        let break_op = scf::r#break(db, location, loop_var);
        let else_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![break_op.as_operation()],
        );
        let else_region = Region::new(db, location, idvec![else_block]);

        // Inner if
        let if_op = scf::r#if(
            db,
            location,
            cond.result(db),
            i32_ty,
            then_region,
            else_region,
        );

        let body_block = Block::new(
            db,
            body_id,
            location,
            idvec![body_arg],
            idvec![
                limit.as_operation(),
                cond.as_operation(),
                if_op.as_operation()
            ],
        );
        let body_region = Region::new(db, location, idvec![body_block]);

        // scf.loop
        let loop_op = scf::r#loop(db, location, vec![init.result(db)], i32_ty, body_region);

        let ret = func::r#return(db, location, [loop_op.result(db)]);

        let entry = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                init.as_operation(),
                loop_op.as_operation(),
                ret.as_operation()
            ],
        );
        let body = Region::new(db, location, idvec![entry]);
        core::Module::create(db, location, "test_loop".into(), body)
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
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let i1_ty = core::I1::new(db).as_type();

        // Outer condition
        let outer_cond = arith::r#const(db, location, i1_ty, Attribute::Bool(true));
        // Inner condition
        let inner_cond = arith::r#const(db, location, i1_ty, Attribute::Bool(false));

        // Inner then: yield 1
        let inner_then_const = arith::Const::i32(db, location, 1);
        let inner_then_yield = scf::r#yield(db, location, [inner_then_const.result(db)]);
        let inner_then_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                inner_then_const.as_operation(),
                inner_then_yield.as_operation()
            ],
        );
        let inner_then_region = Region::new(db, location, idvec![inner_then_block]);

        // Inner else: yield 2
        let inner_else_const = arith::Const::i32(db, location, 2);
        let inner_else_yield = scf::r#yield(db, location, [inner_else_const.result(db)]);
        let inner_else_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                inner_else_const.as_operation(),
                inner_else_yield.as_operation()
            ],
        );
        let inner_else_region = Region::new(db, location, idvec![inner_else_block]);

        // Inner scf.if
        let inner_if = scf::r#if(
            db,
            location,
            inner_cond.result(db),
            i32_ty,
            inner_then_region,
            inner_else_region,
        );

        // Outer then region: inner_cond + inner_if + yield(inner_if result)
        let outer_then_yield = scf::r#yield(db, location, [inner_if.result(db)]);
        let outer_then_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                inner_cond.as_operation(),
                inner_if.as_operation(),
                outer_then_yield.as_operation()
            ],
        );
        let outer_then_region = Region::new(db, location, idvec![outer_then_block]);

        // Outer else: yield 3
        let outer_else_const = arith::Const::i32(db, location, 3);
        let outer_else_yield = scf::r#yield(db, location, [outer_else_const.result(db)]);
        let outer_else_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                outer_else_const.as_operation(),
                outer_else_yield.as_operation()
            ],
        );
        let outer_else_region = Region::new(db, location, idvec![outer_else_block]);

        // Outer scf.if
        let outer_if = scf::r#if(
            db,
            location,
            outer_cond.result(db),
            i32_ty,
            outer_then_region,
            outer_else_region,
        );

        let ret = func::r#return(db, location, [outer_if.result(db)]);

        let entry = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                outer_cond.as_operation(),
                outer_if.as_operation(),
                ret.as_operation()
            ],
        );
        let body = Region::new(db, location, idvec![entry]);
        core::Module::create(db, location, "test_nested_if".into(), body)
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
        let location = test_location(db);

        // Discriminant
        let disc = arith::Const::i32(db, location, 1);

        // Case 0: yield 10
        let case0_const = arith::Const::i32(db, location, 10);
        let case0_yield = scf::r#yield(db, location, [case0_const.result(db)]);
        let case0_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![case0_const.as_operation(), case0_yield.as_operation()],
        );
        let case0_region = Region::new(db, location, idvec![case0_block]);
        let case0 = scf::r#case(db, location, Attribute::IntBits(0), case0_region);

        // Case 1: yield 20
        let case1_const = arith::Const::i32(db, location, 20);
        let case1_yield = scf::r#yield(db, location, [case1_const.result(db)]);
        let case1_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![case1_const.as_operation(), case1_yield.as_operation()],
        );
        let case1_region = Region::new(db, location, idvec![case1_block]);
        let case1 = scf::r#case(db, location, Attribute::IntBits(1), case1_region);

        // Default: yield 99
        let default_const = arith::Const::i32(db, location, 99);
        let default_yield = scf::r#yield(db, location, [default_const.result(db)]);
        let default_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![default_const.as_operation(), default_yield.as_operation()],
        );
        let default_region = Region::new(db, location, idvec![default_block]);
        let default_op = scf::default(db, location, default_region);

        // Switch body: case ops + default
        let switch_body_block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                case0.as_operation(),
                case1.as_operation(),
                default_op.as_operation()
            ],
        );
        let switch_body = Region::new(db, location, idvec![switch_body_block]);

        let switch_op = scf::switch(db, location, disc.result(db), switch_body);

        let ret = func::r#return(db, location, []);

        let entry = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![
                disc.as_operation(),
                switch_op.as_operation(),
                ret.as_operation()
            ],
        );
        let body = Region::new(db, location, idvec![entry]);
        core::Module::create(db, location, "test_switch".into(), body)
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
}
