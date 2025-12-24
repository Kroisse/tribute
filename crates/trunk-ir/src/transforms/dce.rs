//! Dead Code Elimination (DCE) pass.
//!
//! This pass removes operations whose results are never used and which have no side effects.
//! It uses backward liveness analysis to determine which values are live, then sweeps away
//! dead operations. The pass runs to fixpoint, as removing dead code may expose more dead code.

use std::collections::HashSet;

use crate::op_interface::PureOps;
use crate::{Block, IdVec, Operation, Region, Value};

/// Configuration for dead code elimination.
#[derive(Debug, Clone)]
pub struct DceConfig {
    /// Maximum iterations before giving up.
    /// Default: 100
    pub max_iterations: usize,
    /// Whether to recursively process nested regions.
    /// Default: true
    pub recursive: bool,
}

impl Default for DceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            recursive: true,
        }
    }
}

impl DceConfig {
    fn effective_max_iterations(&self) -> usize {
        if self.max_iterations == 0 {
            100
        } else {
            self.max_iterations
        }
    }
}

/// Result of running dead code elimination.
pub struct DceResult<'db> {
    /// The transformed module with dead code removed.
    pub module: crate::dialect::core::Module<'db>,
    /// Total number of operations removed.
    pub removed_count: usize,
    /// Number of fixpoint iterations performed.
    pub iterations: usize,
    /// Whether fixpoint was reached (no more changes possible).
    pub reached_fixpoint: bool,
}

/// Eliminate dead code from a module.
pub fn eliminate_dead_code<'db>(
    db: &'db dyn salsa::Database,
    module: crate::dialect::core::Module<'db>,
) -> DceResult<'db> {
    eliminate_dead_code_with_config(db, module, DceConfig::default())
}

/// Eliminate dead code with custom configuration.
pub fn eliminate_dead_code_with_config<'db>(
    db: &'db dyn salsa::Database,
    module: crate::dialect::core::Module<'db>,
    config: DceConfig,
) -> DceResult<'db> {
    DcePass::new(db, config).run(module)
}

/// Internal DCE pass implementation.
struct DcePass<'db> {
    db: &'db dyn salsa::Database,
    config: DceConfig,
    live_values: HashSet<Value<'db>>,
    removed_count: usize,
}

impl<'db> DcePass<'db> {
    fn new(db: &'db dyn salsa::Database, config: DceConfig) -> Self {
        Self {
            db,
            config,
            live_values: HashSet::new(),
            removed_count: 0,
        }
    }

    fn run(mut self, module: crate::dialect::core::Module<'db>) -> DceResult<'db> {
        let mut current = module;
        let max_iterations = self.config.effective_max_iterations();

        for iteration in 0..max_iterations {
            self.live_values.clear();
            self.compute_live_values(&current);

            let (new_module, changed) = self.sweep_module(&current);

            if !changed {
                return DceResult {
                    module: new_module,
                    removed_count: self.removed_count,
                    iterations: iteration + 1,
                    reached_fixpoint: true,
                };
            }

            current = new_module;
        }

        DceResult {
            module: current,
            removed_count: self.removed_count,
            iterations: max_iterations,
            reached_fixpoint: false,
        }
    }

    fn compute_live_values(&mut self, module: &crate::dialect::core::Module<'db>) {
        let mut worklist: Vec<Value<'db>> = Vec::new();
        self.collect_root_values(module.body(self.db), &mut worklist);

        while let Some(value) = worklist.pop() {
            if self.live_values.contains(&value) {
                continue;
            }
            self.live_values.insert(value);

            // Find defining operation and mark its operands as live
            if let crate::ValueDef::OpResult(op) = value.def(self.db) {
                for &operand in op.operands(self.db).iter() {
                    worklist.push(operand);
                }
                // Recursively process nested regions
                for &region in op.regions(self.db).iter() {
                    self.collect_root_values(region, &mut worklist);
                }
            }
        }
    }

    fn collect_root_values(&self, region: Region<'db>, worklist: &mut Vec<Value<'db>>) {
        for &block in region.blocks(self.db).iter() {
            for &op in block.operations(self.db).iter() {
                // Any operation that is not pure is a root (must be kept)
                if !PureOps::is_pure(self.db, &op) {
                    // Keep all operands of non-pure operations
                    for &operand in op.operands(self.db).iter() {
                        worklist.push(operand);
                    }
                }
                // Recurse into nested regions
                for &nested in op.regions(self.db).iter() {
                    self.collect_root_values(nested, worklist);
                }
            }
        }
    }

    fn sweep_module(
        &mut self,
        module: &crate::dialect::core::Module<'db>,
    ) -> (crate::dialect::core::Module<'db>, bool) {
        let body = module.body(self.db);
        let (new_body, changed) = self.sweep_region(&body);
        let new_module = crate::dialect::core::Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        );
        (new_module, changed)
    }

    fn sweep_region(&mut self, region: &Region<'db>) -> (Region<'db>, bool) {
        let mut changed = false;
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| {
                let (new_block, block_changed) = self.sweep_block(block);
                changed |= block_changed;
                new_block
            })
            .collect();

        let new_region = Region::new(self.db, region.location(self.db), new_blocks);
        (new_region, changed)
    }

    fn sweep_block(&mut self, block: &Block<'db>) -> (Block<'db>, bool) {
        let mut changed = false;
        let mut new_ops: IdVec<Operation<'db>> = IdVec::new();

        for &op in block.operations(self.db).iter() {
            // First, process nested regions
            let op_with_processed_regions =
                if self.config.recursive && !op.regions(self.db).is_empty() {
                    let mut region_changed = false;
                    let new_regions: IdVec<Region<'db>> = op
                        .regions(self.db)
                        .iter()
                        .map(|region| {
                            let (new_region, rc) = self.sweep_region(region);
                            region_changed |= rc;
                            new_region
                        })
                        .collect();

                    if region_changed {
                        changed = true;
                        op.modify(self.db).regions(new_regions).build()
                    } else {
                        op
                    }
                } else {
                    op
                };

            // Check if operation is dead
            if self.is_dead(&op_with_processed_regions) {
                changed = true;
                self.removed_count += 1;
                continue; // Skip adding to new_ops
            }

            new_ops.push(op_with_processed_regions);
        }

        let new_block = Block::new(
            self.db,
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        );
        (new_block, changed)
    }

    fn is_dead(&self, op: &Operation<'db>) -> bool {
        // Pure operations are removable only if results are unused
        if !PureOps::is_pure(self.db, op) {
            return false; // Keep non-pure operations
        }

        // Operations with no results can be kept (shouldn't happen, but be safe)
        if op.results(self.db).is_empty() {
            return false;
        }

        // Check if any result is live
        for i in 0..op.results(self.db).len() {
            let value = op.result(self.db, i);
            if self.live_values.contains(&value) {
                return false; // At least one result is live
            }
        }

        true // All results are dead
    }
}
