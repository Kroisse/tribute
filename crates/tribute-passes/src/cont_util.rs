//! Shared utilities for continuation lowering passes.
//!
//! This module contains functions and types shared between `cont_to_trampoline`
//! and `cont_to_libmprompt` (and potentially other continuation backends).

use std::collections::{HashMap, HashSet};

use trunk_ir::dialect::cont;
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::scf;
use trunk_ir::{Block, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Value};

// ============================================================================
// Region Utilities
// ============================================================================

/// Extract the result value from the last operation of a region.
///
/// Returns the first operand of a trailing `scf.yield`, or
/// the first result of the last operation otherwise.
pub fn get_region_result_value<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Value<'db>> {
    let blocks = region.blocks(db);
    let last_block = blocks.last()?;
    let ops = last_block.operations(db);
    let last_op = ops.last()?;

    // If the last op is scf.yield, return its first operand (the yielded value)
    if let Ok(yield_op) = scf::Yield::from_operation(db, *last_op) {
        return yield_op.values(db).first().copied();
    }

    // Otherwise, return the first result of the last op
    last_op.results(db).first().map(|_| last_op.result(db, 0))
}

// ============================================================================
// Hash-Based Dispatch
// ============================================================================

/// Compute operation index using hash-based dispatch.
///
/// Computes a stable, handler-independent index from ability name and
/// operation name. Both shift sites and handler dispatch use this function,
/// ensuring they always agree on the op index regardless of handler
/// registration order.
///
/// Hashes the interned [`Symbol`] keys directly (no string allocation).
/// Cross-session stability is not required because op indices are never
/// persisted or compared across binaries.
pub fn compute_op_idx(ability_ref: Option<Symbol>, op_name: Option<Symbol>) -> u32 {
    use std::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    ability_ref.hash(&mut hasher);
    op_name.hash(&mut hasher);

    (hasher.finish() % 0x7FFFFFFF) as u32
}

// ============================================================================
// Handler Dispatch Utilities
// ============================================================================

/// Information about a suspend arm for dispatch.
pub struct SuspendArm<'db> {
    /// Expected op_idx for this arm
    pub expected_op_idx: u32,
    /// The body region containing the handler arm code
    pub body: Region<'db>,
}

/// Collect suspend arms from handler_dispatch's body region.
///
/// Uses hash-based dispatch: each arm's op_idx is computed from the ability
/// name and operation name via `compute_op_idx`. This is handler-independent --
/// both shift sites and handler dispatch use the same hash function, so the
/// index matches regardless of handler registration order.
///
/// The body region contains a single block with `cont.done` and `cont.suspend`
/// child operations. This function iterates `cont.suspend` ops.
pub fn collect_suspend_arms<'db>(
    db: &'db dyn salsa::Database,
    body: &Region<'db>,
) -> Vec<SuspendArm<'db>> {
    let mut arms = Vec::new();
    let mut seen_op_indices: HashSet<u32> = HashSet::new();

    let blocks = body.blocks(db);
    let Some(first_block) = blocks.first() else {
        return arms;
    };

    for op in first_block.operations(db).iter() {
        let Ok(suspend_op) = cont::Suspend::from_operation(db, *op) else {
            continue;
        };

        let ability_ref_ty = suspend_op.ability_ref(db);
        let ability_ref = core::AbilityRefType::from_type(db, ability_ref_ty)
            .and_then(|ar| ar.name(db))
            .unwrap_or_else(|| {
                panic!(
                    "collect_suspend_arms: cont.suspend has invalid ability_ref type: {:?}",
                    ability_ref_ty,
                )
            });
        let op_name = suspend_op.op_name(db);

        // Use hash-based dispatch: compute a stable index from ability+op_name.
        let expected_op_idx = compute_op_idx(Some(ability_ref), Some(op_name));
        assert!(
            seen_op_indices.insert(expected_op_idx),
            "compute_op_idx collision in handler dispatch: op_idx {} appears twice \
             (ability={}, op={}). This indicates a hash collision that would \
             cause silent mis-dispatch at runtime.",
            expected_op_idx,
            ability_ref,
            op_name,
        );
        arms.push(SuspendArm {
            expected_op_idx,
            body: suspend_op.body(db),
        });
    }

    arms
}

/// Get the done region from handler_dispatch's body.
///
/// Finds the first `cont.done` child op and returns its body region.
pub fn get_done_region<'db>(
    db: &'db dyn salsa::Database,
    body: &Region<'db>,
) -> Option<Region<'db>> {
    let blocks = body.blocks(db);
    let first_block = blocks.first()?;

    for op in first_block.operations(db).iter() {
        if let Ok(done_op) = cont::Done::from_operation(db, *op) {
            return Some(done_op.body(db));
        }
    }

    None
}

// ============================================================================
// Value Remapping
// ============================================================================

/// Remap a value through a substitution chain.
///
/// Follows the chain in `value_remap` until a fixed point is reached.
/// Panics if a cycle is detected.
pub fn remap_value<'db>(
    v: Value<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Value<'db> {
    let mut current = v;
    let mut seen: HashSet<Value<'db>> = HashSet::new();
    while let Some(&remapped) = value_remap.get(&current) {
        assert!(seen.insert(current), "cycle detected in value_remap");
        current = remapped;
    }
    current
}

/// Rebuild an operation, remapping operands and recursing into regions.
pub fn rebuild_op_with_remap<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Operation<'db> {
    let operands = op.operands(db);
    let remapped_operands: IdVec<Value<'db>> = operands
        .iter()
        .map(|v| remap_value(*v, value_remap))
        .collect();

    let regions = op.regions(db);
    let remapped_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|r| rebuild_region_with_remap(db, r, value_remap))
        .collect();

    if remapped_operands == *operands && remapped_regions == *regions {
        return *op;
    }

    Operation::new(
        db,
        op.location(db),
        op.dialect(db),
        op.name(db),
        remapped_operands,
        op.results(db).clone(),
        op.attributes(db).clone(),
        remapped_regions,
        op.successors(db).clone(),
    )
}

/// Recursively rebuild a block, remapping all operand values in its operations.
pub fn rebuild_block_with_remap<'db>(
    db: &'db dyn salsa::Database,
    block: &Block<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Block<'db> {
    let new_ops: IdVec<Operation<'db>> = block
        .operations(db)
        .iter()
        .map(|op| rebuild_op_with_remap(db, op, value_remap))
        .collect();
    Block::new(
        db,
        block.id(db),
        block.location(db),
        block.args(db).clone(),
        new_ops,
    )
}

/// Recursively rebuild a region, remapping all operand values.
///
/// This is necessary because handler arm bodies may contain nested regions
/// (e.g. `scf.if` branches) that reference block args of the enclosing block.
/// When those block args are replaced, the references inside nested regions
/// must be updated too.
pub fn rebuild_region_with_remap<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    value_remap: &HashMap<Value<'db>, Value<'db>>,
) -> Region<'db> {
    let new_blocks: IdVec<Block<'db>> = region
        .blocks(db)
        .iter()
        .map(|block| rebuild_block_with_remap(db, block, value_remap))
        .collect();
    Region::new(db, region.location(db), new_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::arith;
    use trunk_ir::ir::BlockBuilder;
    use trunk_ir::{Block, BlockId, IdVec, PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> trunk_ir::Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        trunk_ir::Location::new(path, Span::new(0, 0))
    }

    // ====================================================================
    // compute_op_idx
    // ====================================================================

    #[test]
    fn compute_op_idx_is_deterministic() {
        let idx1 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        let idx2 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn compute_op_idx_differs_for_different_ops() {
        let ability = Some(Symbol::new("State"));
        let get_idx = compute_op_idx(ability, Some(Symbol::new("get")));
        let set_idx = compute_op_idx(ability, Some(Symbol::new("set")));
        assert_ne!(get_idx, set_idx);
    }

    #[test]
    fn compute_op_idx_differs_for_different_abilities() {
        let op = Some(Symbol::new("get"));
        let state_idx = compute_op_idx(Some(Symbol::new("State")), op);
        let console_idx = compute_op_idx(Some(Symbol::new("Console")), op);
        assert_ne!(state_idx, console_idx);
    }

    #[test]
    fn compute_op_idx_none_position_matters() {
        // (None, Some(x)) should differ from (Some(x), None)
        // because Option::hash includes the discriminant.
        let sym = Symbol::new("test");
        let idx_none_some = compute_op_idx(None, Some(sym));
        let idx_some_none = compute_op_idx(Some(sym), None);
        assert_ne!(idx_none_some, idx_some_none);
    }

    #[test]
    fn compute_op_idx_within_range() {
        let idx = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        assert!(idx < 0x7FFFFFFF);
    }

    // ====================================================================
    // get_region_result_value
    // ====================================================================

    #[salsa::tracked]
    fn run_get_region_result_value_yield(db: &dyn salsa::Database) -> bool {
        let loc = test_location(db);
        let mut builder = BlockBuilder::new(db, loc);
        let c = builder.op(arith::Const::i32(db, loc, 42));
        builder.op(scf::r#yield(db, loc, vec![c.result(db)]));
        let block = builder.build();

        let region = Region::new(db, loc, IdVec::from(vec![block]));
        get_region_result_value(db, &region).is_some()
    }

    #[salsa_test]
    fn get_region_result_value_returns_yield_operand(db: &salsa::DatabaseImpl) {
        assert!(run_get_region_result_value_yield(db));
    }

    #[salsa::tracked]
    fn run_get_region_result_value_last_op(db: &dyn salsa::Database) -> bool {
        let loc = test_location(db);
        let mut builder = BlockBuilder::new(db, loc);
        builder.op(arith::Const::i32(db, loc, 42));
        let block = builder.build();

        let region = Region::new(db, loc, IdVec::from(vec![block]));
        get_region_result_value(db, &region).is_some()
    }

    #[salsa_test]
    fn get_region_result_value_returns_last_op_result(db: &salsa::DatabaseImpl) {
        assert!(run_get_region_result_value_last_op(db));
    }

    #[salsa::tracked]
    fn run_get_region_result_value_empty_region(db: &dyn salsa::Database) -> bool {
        let loc = test_location(db);
        let region = Region::new(db, loc, IdVec::from(Vec::<Block>::new()));
        get_region_result_value(db, &region).is_none()
    }

    #[salsa_test]
    fn get_region_result_value_empty_region_returns_none(db: &salsa::DatabaseImpl) {
        assert!(run_get_region_result_value_empty_region(db));
    }

    #[salsa::tracked]
    fn run_get_region_result_value_empty_block(db: &dyn salsa::Database) -> bool {
        let loc = test_location(db);
        let block = Block::new(db, BlockId(0), loc, IdVec::default(), IdVec::default());
        let region = Region::new(db, loc, IdVec::from(vec![block]));
        get_region_result_value(db, &region).is_none()
    }

    #[salsa_test]
    fn get_region_result_value_empty_block_returns_none(db: &salsa::DatabaseImpl) {
        assert!(run_get_region_result_value_empty_block(db));
    }

    // ====================================================================
    // collect_suspend_arms
    // ====================================================================

    /// Helper: build a simple body region for child ops.
    fn make_simple_body<'db>(
        db: &'db dyn salsa::Database,
        loc: trunk_ir::Location<'db>,
    ) -> Region<'db> {
        let mut b = BlockBuilder::new(db, loc);
        let c = b.op(arith::Const::i32(db, loc, 0));
        b.op(scf::r#yield(db, loc, vec![c.result(db)]));
        let block = b.build();
        Region::new(db, loc, IdVec::from(vec![block]))
    }

    #[salsa::tracked]
    fn run_collect_suspend_arms_extracts_all(db: &dyn salsa::Database) -> (usize, bool, bool) {
        let loc = test_location(db);

        // Build body region: single block with done + 2 suspend ops
        let done_body = make_simple_body(db, loc);
        let get_body = make_simple_body(db, loc);
        let set_body = make_simple_body(db, loc);

        let state_ref_ty = core::AbilityRefType::simple(db, Symbol::new("State")).as_type();
        let mut builder = BlockBuilder::new(db, loc);
        builder.op(cont::done(db, loc, done_body));
        builder.op(cont::suspend(
            db,
            loc,
            state_ref_ty,
            Symbol::new("get"),
            get_body,
        ));
        builder.op(cont::suspend(
            db,
            loc,
            state_ref_ty,
            Symbol::new("set"),
            set_body,
        ));
        let block = builder.build();
        let body = Region::new(db, loc, IdVec::from(vec![block]));

        let arms = collect_suspend_arms(db, &body);

        let expected_get = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
        let expected_set = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("set")));
        (
            arms.len(),
            arms[0].expected_op_idx == expected_get,
            arms[1].expected_op_idx == expected_set,
        )
    }

    #[salsa_test]
    fn collect_suspend_arms_extracts_all_arms(db: &salsa::DatabaseImpl) {
        let (len, get_ok, set_ok) = run_collect_suspend_arms_extracts_all(db);
        assert_eq!(len, 2);
        assert!(get_ok, "first arm should have State::get op_idx");
        assert!(set_ok, "second arm should have State::set op_idx");
    }

    #[salsa::tracked]
    fn run_collect_suspend_arms_done_only(db: &dyn salsa::Database) -> usize {
        let loc = test_location(db);
        let done_body = make_simple_body(db, loc);
        let mut builder = BlockBuilder::new(db, loc);
        builder.op(cont::done(db, loc, done_body));
        let block = builder.build();
        let body = Region::new(db, loc, IdVec::from(vec![block]));
        collect_suspend_arms(db, &body).len()
    }

    #[salsa_test]
    fn collect_suspend_arms_done_only_returns_empty(db: &salsa::DatabaseImpl) {
        assert_eq!(run_collect_suspend_arms_done_only(db), 0);
    }

    #[salsa::tracked]
    fn run_get_done_region_found(db: &dyn salsa::Database) -> bool {
        let loc = test_location(db);
        let done_body = make_simple_body(db, loc);
        let mut builder = BlockBuilder::new(db, loc);
        builder.op(cont::done(db, loc, done_body));
        let block = builder.build();
        let body = Region::new(db, loc, IdVec::from(vec![block]));
        get_done_region(db, &body).is_some()
    }

    #[salsa_test]
    fn get_done_region_returns_region(db: &salsa::DatabaseImpl) {
        assert!(run_get_done_region_found(db));
    }

    #[salsa::tracked]
    fn run_get_done_region_empty(db: &dyn salsa::Database) -> bool {
        let loc = test_location(db);
        let block = Block::new(db, BlockId(0), loc, IdVec::default(), IdVec::default());
        let body = Region::new(db, loc, IdVec::from(vec![block]));
        get_done_region(db, &body).is_none()
    }

    #[salsa_test]
    fn get_done_region_returns_none_when_no_done(db: &salsa::DatabaseImpl) {
        assert!(run_get_done_region_empty(db));
    }
}
