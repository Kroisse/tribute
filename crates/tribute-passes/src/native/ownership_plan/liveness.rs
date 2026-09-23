//! Cached native managed-liveness views over policy-neutral ownership facts.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use trunk_ir::analysis::{Analysis, AnalysisContext, AnalysisError};
use trunk_ir::{BlockRef, OpRef, ValueRef};

use super::facts::{FlowKind, NativeOwnershipFunctionFacts, borrowed_owner};

mod sealed {
    pub trait Sealed {}
}

/// The two supported native managed-liveness views.
pub trait LivenessPolicy: sealed::Sealed + Send + Sync + 'static {
    /// Whether proven projection borrows also use their managed owner.
    const EXTEND_PROJECTION_OWNERS: bool;
}

/// Conservative managed-value liveness.
pub struct Conservative;

/// Managed liveness extended through proven projection borrows.
pub struct NativeOwnershipExtended;

impl sealed::Sealed for Conservative {}
impl sealed::Sealed for NativeOwnershipExtended {}

impl LivenessPolicy for Conservative {
    const EXTEND_PROJECTION_OWNERS: bool = false;
}

impl LivenessPolicy for NativeOwnershipExtended {
    const EXTEND_PROJECTION_OWNERS: bool = true;
}

/// Cached block liveness for a defined `func.func` under one policy view.
pub struct Liveness<P: LivenessPolicy> {
    blocks: BlockLiveness,
    policy: PhantomData<P>,
}

/// Block-level liveness sets used by both policy views.
pub struct BlockLiveness {
    pub(super) defs: HashMap<BlockRef, HashSet<ValueRef>>,
    pub(super) live_in: HashMap<BlockRef, HashSet<ValueRef>>,
    pub(super) live_out: HashMap<BlockRef, HashSet<ValueRef>>,
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

impl<P: LivenessPolicy> Liveness<P> {
    /// Block-level liveness sets for this policy view.
    pub fn blocks(&self) -> &BlockLiveness {
        &self.blocks
    }
}

impl<P: LivenessPolicy> Analysis for Liveness<P> {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let facts = ctx.get::<NativeOwnershipFunctionFacts>(target)?;
        let empty = HashMap::new();
        let borrowed = if P::EXTEND_PROJECTION_OWNERS {
            facts.projection_owners()
        } else {
            &empty
        };
        Ok(Self {
            blocks: compute_liveness(&facts, borrowed),
            policy: PhantomData,
        })
    }
}

fn compute_liveness(
    facts: &NativeOwnershipFunctionFacts,
    borrowed: &HashMap<ValueRef, ValueRef>,
) -> BlockLiveness {
    let cfg = facts.cfg();
    let managed = facts.managed_values();
    let aliases = facts.aliases();
    let blocks = cfg.blocks();
    let mut uses = HashMap::new();
    let mut defs = HashMap::new();
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
        .collect::<HashMap<_, _>>();
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
