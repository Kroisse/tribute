use std::collections::{HashMap, HashSet};

use trunk_ir::context::IrContext;
use trunk_ir::dialect::{cf, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::{BlockRef, OpRef, RegionRef, ValueRef};

use super::OwnershipPlanError;

/// The flat control-flow facts accepted by typed ownership planning.
///
/// Construction validates the concrete `cf` representation once. Consumers
/// can then use stable block order, successors, and branch argument transfers
/// without decoding operation layouts independently.
pub(super) struct ValidatedFlatCfg {
    blocks: Vec<BlockRef>,
    terminators: HashMap<BlockRef, OpRef>,
    successors: HashMap<BlockRef, Vec<BlockRef>>,
    branches: HashMap<OpRef, BlockRef>,
}

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
            if block_successors
                .iter()
                .any(|successor| !block_set.contains(successor))
            {
                return Err(OwnershipPlanError::new(
                    "CFG successor leaves function body",
                ));
            }

            if cf::Br::matches(ctx, terminator) {
                let [destination] = block_successors.as_slice() else {
                    return Err(OwnershipPlanError::new(
                        "branch argument contract is malformed",
                    ));
                };
                let operands = ctx.op_operands(terminator);
                let arguments = ctx.block_args(*destination);
                if operands.len() != arguments.len() {
                    return Err(OwnershipPlanError::new(
                        "branch argument contract is malformed",
                    ));
                }
                branches.insert(terminator, *destination);
            } else if cf::CondBr::matches(ctx, terminator) {
                if ctx.op_operands(terminator).len() != 1 || block_successors.len() != 2 {
                    return Err(OwnershipPlanError::new(
                        "conditional branch contract is malformed",
                    ));
                }
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

    pub(super) fn branch_transfers<'a>(
        &'a self,
        ctx: &'a IrContext,
        op: OpRef,
    ) -> Option<impl Iterator<Item = ValueTransfer> + 'a> {
        let destination = *self.branches.get(&op)?;
        Some(
            ctx.op_operands(op)
                .iter()
                .copied()
                .zip(ctx.block_args(destination).iter().copied())
                .map(|(source, destination)| ValueTransfer {
                    source,
                    destination,
                }),
        )
    }
}

fn is_native_terminator(ctx: &IrContext, op: OpRef) -> bool {
    cf::Br::matches(ctx, op)
        || cf::CondBr::matches(ctx, op)
        || func::Return::matches(ctx, op)
        || func::TailCall::matches(ctx, op)
        || func::TailCallIndirect::matches(ctx, op)
        || func::Unreachable::matches(ctx, op)
}
