use std::collections::{HashMap, HashSet};

use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::op_interface::BranchOps;
use trunk_ir::ops::DialectOp;
use trunk_ir::{BlockRef, OpRef, RegionRef, ValueRef};

use super::OwnershipPlanError;

/// The flat control-flow facts accepted by typed ownership planning.
///
/// Construction validates registered `Branch` semantics once. Consumers can
/// then use stable block order, successors, and branch argument transfers
/// without decoding operation layouts independently.
pub(super) struct ValidatedFlatCfg {
    blocks: Vec<BlockRef>,
    terminators: HashMap<BlockRef, OpRef>,
    successors: HashMap<BlockRef, Vec<BlockRef>>,
    branches: HashMap<OpRef, Vec<ValueTransfer>>,
}

#[derive(Clone, Copy)]
pub(super) struct ValueTransfer {
    pub(super) source: ValueRef,
    pub(super) destination: ValueRef,
}

impl ValidatedFlatCfg {
    pub(super) fn build(ctx: &IrContext, body: RegionRef) -> Result<Self, OwnershipPlanError> {
        let blocks = ctx.region(body).blocks.to_vec();
        if blocks.is_empty() {
            return Err(OwnershipPlanError::new("defined function has no blocks"));
        }
        let block_set: HashSet<_> = blocks.iter().copied().collect();
        if block_set.len() != blocks.len() {
            return Err(OwnershipPlanError::new(
                "function contains duplicate blocks",
            ));
        }

        let mut terminators = HashMap::new();
        let mut successors = HashMap::new();
        let mut branches = HashMap::new();
        for &block in &blocks {
            let ops = &ctx.block(block).ops;
            let Some((&terminator, preceding)) = ops.split_last() else {
                return Err(OwnershipPlanError::new("function block is empty"));
            };
            if ops.iter().any(|&op| !ctx.op(op).regions.is_empty()) {
                return Err(OwnershipPlanError::new(
                    "unsupported structured or nested control-flow region",
                ));
            }
            if preceding
                .iter()
                .any(|&op| !ctx.op(op).successors.is_empty() || is_native_terminator(ctx, op))
            {
                return Err(OwnershipPlanError::new(
                    "control-flow operation precedes the final block operation",
                ));
            }

            let block_successors = ctx.op(terminator).successors.to_vec();
            if let Some(interface) = BranchOps::get(ctx, terminator) {
                let edges = interface.successors(ctx, terminator).map_err(|error| {
                    OwnershipPlanError::new(format!("Branch interface is incomplete: {error}"))
                })?;
                if edges.as_slice().len() != block_successors.len() {
                    return Err(OwnershipPlanError::new(
                        "Branch successors leave the function or are incomplete",
                    ));
                }
                let mut transfers = Vec::new();
                for (edge, &successor) in edges.as_slice().iter().zip(&block_successors) {
                    if edge.block != successor || !block_set.contains(&successor) {
                        return Err(OwnershipPlanError::new(
                            "Branch successors leave the function or are incomplete",
                        ));
                    }
                    let inputs = ctx.block_args(edge.block);
                    if edge.forwarded.as_slice().len() != inputs.len()
                        || edge.forwarded.as_slice().iter().zip(inputs).any(
                            |(&source, &destination)| {
                                ctx.value_ty(source) != ctx.value_ty(destination)
                            },
                        )
                    {
                        return Err(OwnershipPlanError::new(
                            "branch argument contract is malformed",
                        ));
                    }
                    if edges.as_slice().len() != 1 && !edge.forwarded.is_empty() {
                        return Err(OwnershipPlanError::new(
                            "multi-successor Branch forwarding is unsupported",
                        ));
                    }
                    transfers.extend(
                        edge.forwarded
                            .as_slice()
                            .iter()
                            .copied()
                            .zip(inputs.iter().copied())
                            .map(|(source, destination)| ValueTransfer {
                                source,
                                destination,
                            }),
                    );
                }
                branches.insert(terminator, transfers);
            } else if !is_native_terminator(ctx, terminator) {
                return Err(OwnershipPlanError::new("unsupported native CFG terminator"));
            }

            terminators.insert(block, terminator);
            successors.insert(block, block_successors);
        }

        Ok(Self {
            blocks,
            terminators,
            successors,
            branches,
        })
    }

    pub(super) fn blocks(&self) -> &[BlockRef] {
        &self.blocks
    }

    pub(super) fn entry(&self) -> BlockRef {
        self.blocks[0]
    }

    pub(super) fn terminator(&self, block: BlockRef) -> OpRef {
        self.terminators[&block]
    }

    pub(super) fn is_terminator(&self, op: OpRef) -> bool {
        self.terminators
            .values()
            .any(|&terminator| terminator == op)
    }

    pub(super) fn successors(&self, block: BlockRef) -> &[BlockRef] {
        &self.successors[&block]
    }

    pub(super) fn branch_transfers(
        &self,
        op: OpRef,
    ) -> Option<impl Iterator<Item = ValueTransfer> + '_> {
        self.branches
            .get(&op)
            .map(|transfers| transfers.iter().copied())
    }
}

fn is_native_terminator(ctx: &IrContext, op: OpRef) -> bool {
    BranchOps::get(ctx, op).is_some()
        || func::Return::matches(ctx, op)
        || func::TailCall::matches(ctx, op)
        || func::TailCallIndirect::matches(ctx, op)
        || func::Unreachable::matches(ctx, op)
}
