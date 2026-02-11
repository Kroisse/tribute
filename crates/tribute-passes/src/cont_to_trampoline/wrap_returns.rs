use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use trunk_ir::dialect::func::{self, Func};
use trunk_ir::dialect::trampoline;
use trunk_ir::rewrite::{OpAdaptor, RewritePattern, RewriteResult};
use trunk_ir::{Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Value};

// ============================================================================
// Pattern: Wrap returns in effectful functions with step_done
// ============================================================================

pub(crate) struct WrapReturnsInEffectfulFuncsPattern {
    pub(crate) effectful_funcs: Rc<HashSet<Symbol>>,
}

impl<'db> RewritePattern<'db> for WrapReturnsInEffectfulFuncsPattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match func.func operations
        let Ok(func) = Func::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        // Only process effectful functions
        let func_name = func.sym_name(db);
        if !self.effectful_funcs.contains(&func_name) {
            return RewriteResult::Unchanged;
        }

        tracing::debug!(
            "WrapReturnsInEffectfulFuncsPattern: processing {}",
            func_name
        );

        // Transform the function body - wrap non-Step returns with step_done
        let body = func.body(db);
        let mut block_map = HashMap::new();
        collect_block_map(db, &body, &mut block_map);
        let (new_body, modified) = wrap_returns_in_region(db, body, &block_map);

        tracing::debug!(
            "WrapReturnsInEffectfulFuncsPattern: {} modified={}",
            func_name,
            modified
        );

        if !modified {
            return RewriteResult::Unchanged;
        }

        // Rebuild the function with the transformed body
        let new_op = op.modify(db).regions(IdVec::from(vec![new_body])).build();
        RewriteResult::Replace(new_op)
    }
}

/// Recursively wrap returns in a region with step_done.
/// Returns (new_region, was_modified).
fn wrap_returns_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    block_map: &HashMap<BlockId, Block<'db>>,
) -> (Region<'db>, bool) {
    let mut new_blocks = Vec::new();
    let mut any_modified = false;

    for block in region.blocks(db).iter() {
        let (new_block, modified) = wrap_returns_in_block(db, *block, block_map);
        new_blocks.push(new_block);
        any_modified |= modified;
    }

    if !any_modified {
        return (region, false);
    }

    (
        Region::new(db, region.location(db), IdVec::from(new_blocks)),
        true,
    )
}

/// Wrap returns in a block with step_done.
/// Returns (new_block, was_modified).
fn wrap_returns_in_block<'db>(
    db: &'db dyn salsa::Database,
    block: Block<'db>,
    block_map: &HashMap<BlockId, Block<'db>>,
) -> (Block<'db>, bool) {
    let mut new_ops = Vec::new();
    let mut modified = false;

    for op in block.operations(db).iter() {
        // First, recursively process nested regions
        let mut op_modified = false;
        let op_with_transformed_regions = if !op.regions(db).is_empty() {
            let mut new_regions = Vec::new();
            for r in op.regions(db).iter() {
                let (new_r, r_modified) = wrap_returns_in_region(db, *r, block_map);
                new_regions.push(new_r);
                op_modified |= r_modified;
            }
            if op_modified {
                op.modify(db).regions(IdVec::from(new_regions)).build()
            } else {
                *op
            }
        } else {
            *op
        };

        modified |= op_modified;

        // Check if this is a func.return
        if func::Return::from_operation(db, op_with_transformed_regions).is_ok() {
            let operands = op_with_transformed_regions.operands(db);

            if let Some(&value) = operands.first() {
                // Check if already returning Step
                let is_step = is_step_value(db, value, block_map);
                tracing::debug!(
                    "wrap_returns_in_block: found func.return, value is_step={}",
                    is_step
                );
                if !is_step {
                    let location = op_with_transformed_regions.location(db);
                    let step_ty = trampoline::Step::new(db).as_type();

                    // Create step_done(value)
                    let step_done = trampoline::step_done(db, location, value, step_ty);
                    let step_value = step_done.as_operation().result(db, 0);
                    new_ops.push(step_done.as_operation());

                    // Create new return with step value
                    let new_return = func::r#return(db, location, Some(step_value));
                    new_ops.push(new_return.as_operation());
                    modified = true;
                    tracing::debug!("wrap_returns_in_block: wrapped return with step_done");
                    continue;
                }
            }
        }

        new_ops.push(op_with_transformed_regions);
    }

    if !modified {
        return (block, false);
    }

    (
        Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            IdVec::from(new_ops),
        ),
        true,
    )
}

/// Check if a value is already a Step type (from step_shift, step_done, or check_yield result).
///
/// `block_map` maps `BlockId → Block` for resolving block argument types.
pub(crate) fn is_step_value<'db>(
    db: &'db dyn salsa::Database,
    value: Value<'db>,
    block_map: &HashMap<BlockId, Block<'db>>,
) -> bool {
    use trunk_ir::ValueDef;

    match value.def(db) {
        ValueDef::OpResult(def_op) => {
            // Check if the defining operation produces Step
            trampoline::StepShift::from_operation(db, def_op).is_ok()
                || trampoline::StepDone::from_operation(db, def_op).is_ok()
                // Check if any operation's result type is Step
                // This handles trampoline loop results, unrealized_conversion_cast, etc.
                || def_op
                    .results(db)
                    .first()
                    .is_some_and(|ty| trampoline::Step::from_type(db, *ty).is_some())
        }
        ValueDef::BlockArg(block_id) => {
            let index = value.index(db);
            block_map
                .get(&block_id)
                .and_then(|block| block.args(db).get(index).map(|arg| arg.ty(db)))
                .is_some_and(|ty| trampoline::Step::from_type(db, ty).is_some())
        }
    }
}

/// Collect all blocks in a region (recursively) into a map from BlockId → Block.
fn collect_block_map<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    map: &mut HashMap<BlockId, Block<'db>>,
) {
    for block in region.blocks(db).iter() {
        map.insert(block.id(db), *block);
        for op in block.operations(db).iter() {
            for nested in op.regions(db).iter() {
                collect_block_map(db, nested, map);
            }
        }
    }
}
