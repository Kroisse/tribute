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
/// NOTE: Uses `FxHasher` from `rustc-hash` for deterministic hashing within
/// a single compilation session. Cross-version stability is not required
/// because op indices are never persisted or compared across binaries.
pub fn compute_op_idx(ability_ref: Option<Symbol>, op_name: Option<Symbol>) -> u32 {
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = FxHasher::default();

    if let Some(ability) = ability_ref {
        ability.to_string().hash(&mut hasher);
    }
    if let Some(name) = op_name {
        name.to_string().hash(&mut hasher);
    }

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
        let ability_ref = attrs.get(&Symbol::new("ability_ref")).and_then(|a| {
            if let Attribute::Type(ty) = a {
                core::AbilityRefType::from_type(db, *ty).and_then(|ar| ar.name(db))
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

        // Use hash-based dispatch: compute a stable index from ability+op_name.
        let expected_op_idx = compute_op_idx(ability_ref, op_name);
        assert!(
            seen_op_indices.insert(expected_op_idx),
            "compute_op_idx collision in handler dispatch: op_idx {} appears twice \
             (ability={:?}, op={:?}). This indicates a hash collision that would \
             cause silent mis-dispatch at runtime.",
            expected_op_idx,
            ability_ref.map(|s| s.to_string()),
            op_name.map(|s| s.to_string()),
        );
        arms.push(SuspendArm {
            expected_op_idx,
            block: *block,
        });
    }

    arms
}
