//! Shared utilities for continuation lowering passes.
//!
//! This module contains functions and types shared between `cont_to_trampoline`
//! and `cont_to_libmprompt` (and potentially other continuation backends).

use std::collections::HashSet;

use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::scf;
use trunk_ir::{Attribute, Block, DialectOp, DialectType, IdVec, Region, Symbol, Value};

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
    /// The block containing the handler arm code
    pub block: Block<'db>,
}

/// Collect suspend arms from handler blocks with their expected op_idx.
///
/// Uses hash-based dispatch: each arm's op_idx is computed from the ability
/// name and operation name via `compute_op_idx`. This is handler-independent --
/// both shift sites and handler dispatch use the same hash function, so the
/// index matches regardless of handler registration order.
///
/// Blocks layout: block 0 is the "done" case; blocks 1+ are suspend arms.
pub fn collect_suspend_arms<'db>(
    db: &'db dyn salsa::Database,
    blocks: &IdVec<Block<'db>>,
) -> Vec<SuspendArm<'db>> {
    let mut arms = Vec::new();
    let mut seen_op_indices: HashSet<u32> = HashSet::new();

    // Skip block 0 (done case), process blocks 1+ (suspend cases)
    for (i, block) in blocks.iter().enumerate().skip(1) {
        // Extract ability_ref and op_name from marker block arg
        let block_args = block.args(db);
        let marker_arg = block_args.first().unwrap_or_else(|| {
            panic!(
                "collect_suspend_arms: suspend block at index {} has no block arguments; \
                 expected a marker arg with ability_ref and op_name attributes. \
                 Block: {:?}",
                i, block,
            )
        });
        let attrs = marker_arg.attrs(db);
        let ability_ref = attrs
            .get(&Symbol::new("ability_ref"))
            .and_then(|a| {
                if let Attribute::Type(ty) = a {
                    core::AbilityRefType::from_type(db, *ty).and_then(|ar| ar.name(db))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "collect_suspend_arms: suspend block at index {} has no valid \
                     'ability_ref' attribute on marker arg. \
                     Attrs: {:?}, Block: {:?}",
                    i, attrs, block,
                )
            });
        let op_name = attrs
            .get(&Symbol::new("op_name"))
            .and_then(|a| {
                if let Attribute::Symbol(s) = a {
                    Some(*s)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "collect_suspend_arms: suspend block at index {} has no valid \
                     'op_name' attribute on marker arg. \
                     Attrs: {:?}, Block: {:?}",
                    i, attrs, block,
                )
            });

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
            block: *block,
        });
    }

    arms
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::arith;
    use trunk_ir::ir::BlockBuilder;
    use trunk_ir::{BlockArg, BlockId, PathId, Span};

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

    /// Helper: build a suspend block with ability_ref and op_name marker.
    fn make_suspend_block<'db>(
        db: &'db dyn salsa::Database,
        loc: trunk_ir::Location<'db>,
        ability: &'static str,
        op: &'static str,
    ) -> Block<'db> {
        let ability_ref_ty = core::AbilityRefType::simple(db, Symbol::new(ability)).as_type();
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(Symbol::new("ability_ref"), Attribute::Type(ability_ref_ty));
        attrs.insert(Symbol::new("op_name"), Attribute::Symbol(Symbol::new(op)));
        let i32_ty = core::I32::new(db).as_type();
        let marker_arg = BlockArg::new(db, i32_ty, attrs);

        let mut b = BlockBuilder::new(db, loc);
        let c = b.op(arith::Const::i32(db, loc, 0));
        b.op(scf::r#yield(db, loc, vec![c.result(db)]));
        let block = b.build();
        Block::new(
            db,
            block.id(db),
            loc,
            IdVec::from(vec![marker_arg]),
            block.operations(db).clone(),
        )
    }

    #[salsa::tracked]
    fn run_collect_suspend_arms_extracts_all(db: &dyn salsa::Database) -> (usize, bool, bool) {
        let loc = test_location(db);
        let done_block = Block::new(db, BlockId(0), loc, IdVec::default(), IdVec::default());
        let get_block = make_suspend_block(db, loc, "State", "get");
        let set_block = make_suspend_block(db, loc, "State", "set");

        let blocks = IdVec::from(vec![done_block, get_block, set_block]);
        let arms = collect_suspend_arms(db, &blocks);

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
        let done_block = Block::new(db, BlockId(0), loc, IdVec::default(), IdVec::default());
        let blocks = IdVec::from(vec![done_block]);
        collect_suspend_arms(db, &blocks).len()
    }

    #[salsa_test]
    fn collect_suspend_arms_done_only_returns_empty(db: &salsa::DatabaseImpl) {
        assert_eq!(run_collect_suspend_arms_done_only(db), 0);
    }
}
