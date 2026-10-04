//! Cached native managed-liveness views over policy-neutral ownership facts.

use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use trunk_ir::analysis::{Analysis, AnalysisContext, AnalysisError};
use trunk_ir::{BlockRef, OpRef, ValueRef};

use super::facts::{FlowKind, NativeOwnershipFunctionFacts, borrowed_owner};

/// Lazy block-liveness views for a defined `func.func`.
///
/// The policy-neutral facts are the cache prerequisite. Each fixed point is
/// computed only when its view is first requested, so production planning
/// does not compute the unused conservative result.
pub struct NativeManagedLiveness {
    facts: Arc<NativeOwnershipFunctionFacts>,
    conservative: OnceLock<BlockLiveness>,
    owner_extended: OnceLock<BlockLiveness>,
}

/// Block-level liveness sets used by both policy views.
pub struct BlockLiveness {
    pub(super) defs: FxHashMap<BlockRef, HashSet<ValueRef>>,
    pub(super) live_in: FxHashMap<BlockRef, HashSet<ValueRef>>,
    pub(super) live_out: FxHashMap<BlockRef, HashSet<ValueRef>>,
}

impl BlockLiveness {
    /// Managed values defined in this block.
    pub fn defs(&self, block: BlockRef) -> Option<&HashSet<ValueRef>> {
        self.defs.get(&block)
    }

    /// Managed values live on entry to this block.
    pub fn live_in(&self, block: BlockRef) -> Option<&HashSet<ValueRef>> {
        self.live_in.get(&block)
    }

    /// Managed values live on exit from this block.
    pub fn live_out(&self, block: BlockRef) -> Option<&HashSet<ValueRef>> {
        self.live_out.get(&block)
    }
}

impl NativeManagedLiveness {
    /// Select liveness for the field-borrow policy without recomputing it.
    pub fn view(&self, elide_proven_field_borrows: bool) -> &BlockLiveness {
        if elide_proven_field_borrows {
            self.owner_extended
                .get_or_init(|| compute_liveness(&self.facts, self.facts.projection_owners()))
        } else {
            self.conservative
                .get_or_init(|| compute_liveness(&self.facts, &FxHashMap::default()))
        }
    }

    #[cfg(test)]
    pub(super) fn computed_views(&self) -> (bool, bool) {
        (
            self.conservative.get().is_some(),
            self.owner_extended.get().is_some(),
        )
    }
}

impl Analysis for NativeManagedLiveness {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let facts = ctx.get::<NativeOwnershipFunctionFacts>(target)?;
        Ok(Self {
            facts,
            conservative: OnceLock::new(),
            owner_extended: OnceLock::new(),
        })
    }
}

fn compute_liveness(
    facts: &NativeOwnershipFunctionFacts,
    borrowed: &FxHashMap<ValueRef, ValueRef>,
) -> BlockLiveness {
    let cfg = facts.cfg();
    let managed = facts.managed_values();
    let aliases = facts.aliases();
    let blocks = cfg.blocks();
    let mut uses = FxHashMap::default();
    let mut defs = FxHashMap::default();
    for &block in blocks {
        let mut block_uses = HashSet::new();
        let mut block_defs = HashSet::new();
        // Preserve the original scan order: block arguments, then operation
        // operands followed by results.
        for event in facts.block_flow(block).events() {
            match event.kind {
                FlowKind::Def => {
                    if managed.contains(&event.root) {
                        block_defs.insert(event.root);
                    }
                }
                FlowKind::Use => {
                    let root = event.root;
                    if managed.contains(&root) && !block_defs.contains(&root) {
                        block_uses.insert(root);
                    }
                    if let Some(owner) = borrowed_owner(borrowed, aliases, root)
                        && managed.contains(&owner)
                        && !block_defs.contains(&owner)
                    {
                        block_uses.insert(owner);
                    }
                }
            }
        }
        uses.insert(block, block_uses);
        defs.insert(block, block_defs);
    }
    let mut live_in = blocks
        .iter()
        .map(|&b| (b, HashSet::new()))
        .collect::<FxHashMap<_, _>>();
    let mut live_out = live_in.clone();
    loop {
        let mut changed = false;
        for &block in blocks.iter().rev() {
            let mut out = HashSet::new();
            for successor in cfg.successors(block) {
                out.extend(live_in[successor].iter().copied());
            }
            let mut input = uses[&block].clone();
            input.extend(
                out.iter()
                    .filter(|value| !defs[&block].contains(value))
                    .copied(),
            );
            if input != live_in[&block] {
                live_in.insert(block, input);
                changed = true;
            }
            if out != live_out[&block] {
                live_out.insert(block, out);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    BlockLiveness {
        defs,
        live_in,
        live_out,
    }
}
