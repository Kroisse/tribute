//! Dominance analysis for single-region control-flow graphs.

use std::collections::{HashMap, HashSet};

use crate::context::IrContext;
use crate::{BlockRef, RegionRef};

/// Block dominance information for one region.
#[derive(Debug, Clone)]
pub struct DominatorTree {
    region: RegionRef,
    entry: Option<BlockRef>,
    predecessors: HashMap<BlockRef, Vec<BlockRef>>,
    successors: HashMap<BlockRef, Vec<BlockRef>>,
    reachable: HashSet<BlockRef>,
    dominators: HashMap<BlockRef, HashSet<BlockRef>>,
    valid: bool,
}

impl DominatorTree {
    /// Compute dominance from the region's first block.
    pub fn compute(ctx: &IrContext, region: RegionRef) -> Self {
        let blocks = ctx.region(region).blocks.to_vec();
        let block_set: HashSet<_> = blocks.iter().copied().collect();
        let entry = blocks.first().copied();
        let mut predecessors: HashMap<_, Vec<_>> = blocks
            .iter()
            .copied()
            .map(|block| (block, Vec::new()))
            .collect();
        let mut successors = HashMap::new();
        let mut valid = true;

        for &block in &blocks {
            let block_successors = ctx
                .block(block)
                .ops
                .last()
                .map(|&op| ctx.op(op).successors.to_vec())
                .unwrap_or_default();
            for &successor in &block_successors {
                if !block_set.contains(&successor) {
                    valid = false;
                    continue;
                }
                predecessors
                    .get_mut(&successor)
                    .expect("region block has predecessor entry")
                    .push(block);
            }
            successors.insert(block, block_successors);
        }

        let mut reachable = HashSet::new();
        if let Some(entry) = entry {
            let mut pending = vec![entry];
            while let Some(block) = pending.pop() {
                if !reachable.insert(block) {
                    continue;
                }
                if let Some(block_successors) = successors.get(&block) {
                    pending.extend(
                        block_successors
                            .iter()
                            .copied()
                            .filter(|successor| block_set.contains(successor)),
                    );
                }
            }
        }

        let mut dominators = HashMap::new();
        if let Some(entry) = entry {
            for &block in &reachable {
                let initial = if block == entry {
                    HashSet::from([entry])
                } else {
                    reachable.clone()
                };
                dominators.insert(block, initial);
            }

            let mut changed = true;
            while changed {
                changed = false;
                for &block in &blocks {
                    if block == entry || !reachable.contains(&block) {
                        continue;
                    }
                    let reachable_predecessors: Vec<_> = predecessors[&block]
                        .iter()
                        .copied()
                        .filter(|predecessor| reachable.contains(predecessor))
                        .collect();
                    let Some((&first, rest)) = reachable_predecessors.split_first() else {
                        valid = false;
                        continue;
                    };
                    let mut next = dominators[&first].clone();
                    for predecessor in rest {
                        next.retain(|dominator| dominators[predecessor].contains(dominator));
                    }
                    next.insert(block);
                    if next != dominators[&block] {
                        dominators.insert(block, next);
                        changed = true;
                    }
                }
            }
        }

        Self {
            region,
            entry,
            predecessors,
            successors,
            reachable,
            dominators,
            valid,
        }
    }

    pub fn region(&self) -> RegionRef {
        self.region
    }

    pub fn entry(&self) -> Option<BlockRef> {
        self.entry
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn is_reachable(&self, block: BlockRef) -> bool {
        self.reachable.contains(&block)
    }

    pub fn dominates(&self, dominator: BlockRef, block: BlockRef) -> bool {
        self.dominators
            .get(&block)
            .is_some_and(|dominators| dominators.contains(&dominator))
    }

    pub fn predecessors(&self, block: BlockRef) -> &[BlockRef] {
        self.predecessors.get(&block).map_or(&[], Vec::as_slice)
    }

    pub fn successors(&self, block: BlockRef) -> &[BlockRef] {
        self.successors.get(&block).map_or(&[], Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::clif;
    use crate::ops::DialectOp;
    use crate::parser::parse_test_module;

    fn function_cfg(ir: &str) -> (IrContext, RegionRef, Vec<BlockRef>) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let module_block = module.first_block(&ctx).expect("module body");
        let function =
            clif::Func::from_op(&ctx, ctx.block(module_block).ops[0]).expect("clif.func");
        let region = function.body(&ctx);
        let blocks = ctx.region(region).blocks.to_vec();
        (ctx, region, blocks)
    }

    #[test]
    fn diamond_join_is_dominated_only_by_entry() {
        let (ctx, region, blocks) = function_cfg(
            r#"core.module @test {
  clif.func @f(%0: core.i8) -> core.nil {
  ^entry:
    clif.brif %0 [^left, ^right]
  ^left:
    clif.jump [^join]
  ^right:
    clif.jump [^join]
  ^join:
    clif.return
  }
}"#,
        );
        let dominance = DominatorTree::compute(&ctx, region);
        assert!(dominance.is_valid());
        assert!(dominance.dominates(blocks[0], blocks[3]));
        assert!(!dominance.dominates(blocks[1], blocks[3]));
        assert!(!dominance.dominates(blocks[2], blocks[3]));
    }

    #[test]
    fn branch_local_block_dominates_its_descendant() {
        let (ctx, region, blocks) = function_cfg(
            r#"core.module @test {
  clif.func @f(%0: core.i8) -> core.nil {
  ^entry:
    clif.brif %0 [^left, ^exit]
  ^left:
    clif.jump [^child]
  ^child:
    clif.return
  ^exit:
    clif.return
  }
}"#,
        );
        let dominance = DominatorTree::compute(&ctx, region);
        assert!(dominance.dominates(blocks[1], blocks[2]));
        assert!(!dominance.dominates(blocks[1], blocks[3]));
    }

    #[test]
    fn loop_header_dominates_loop_body_and_exit() {
        let (ctx, region, blocks) = function_cfg(
            r#"core.module @test {
  clif.func @f(%0: core.i8) -> core.nil {
  ^entry:
    clif.jump [^header]
  ^header:
    clif.brif %0 [^body, ^exit]
  ^body:
    clif.jump [^header]
  ^exit:
    clif.return
  }
}"#,
        );
        let dominance = DominatorTree::compute(&ctx, region);
        assert!(dominance.dominates(blocks[1], blocks[2]));
        assert!(dominance.dominates(blocks[1], blocks[3]));
        assert!(!dominance.dominates(blocks[2], blocks[1]));
    }

    #[test]
    fn unreachable_blocks_have_no_dominance_relation() {
        let (ctx, region, blocks) = function_cfg(
            r#"core.module @test {
  clif.func @f() -> core.nil {
  ^entry:
    clif.return
  ^dead:
    clif.return
  }
}"#,
        );
        let dominance = DominatorTree::compute(&ctx, region);
        assert!(!dominance.is_reachable(blocks[1]));
        assert!(!dominance.dominates(blocks[1], blocks[1]));
        assert!(!dominance.dominates(blocks[0], blocks[1]));
    }

    #[test]
    fn successor_outside_region_marks_analysis_invalid() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @f() -> core.nil {
  ^entry:
    clif.jump [^exit]
  ^exit:
    clif.return
  }
  clif.func @g() -> core.nil {
  ^entry:
    clif.return
  }
}"#,
        );
        let module_block = module.first_block(&ctx).expect("module body");
        let first = clif::Func::from_op(&ctx, ctx.block(module_block).ops[0]).expect("first func");
        let second =
            clif::Func::from_op(&ctx, ctx.block(module_block).ops[1]).expect("second func");
        let first_region = first.body(&ctx);
        let first_entry = ctx.region(first_region).blocks[0];
        let second_entry = ctx.region(second.body(&ctx)).blocks[0];
        let jump = *ctx.block(first_entry).ops.last().expect("jump");
        ctx.op_mut(jump).successors[0] = second_entry;

        let dominance = DominatorTree::compute(&ctx, first_region);
        assert!(!dominance.is_valid());
    }
}
