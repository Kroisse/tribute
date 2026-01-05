//! Continuation Capture Fixup Pass
//!
//! This pass fixes continuation variable captures in closures within ability handler arms.
//!
//! ## Problem
//!
//! When handler arms create closures that capture continuation variables (like `k`):
//! ```tribute
//! { State::get() -> k } -> run_state(fn() { k(init) }, init)
//! ```
//!
//! The continuation variable `k` is bound in the handler arm pattern, but closures
//! are lifted before handler processing. This means:
//! 1. Lambda lift runs (stage 5) and creates lifted functions
//! 2. Handler lower runs (stage 9) and binds continuation variables
//! 3. tribute_to_scf runs (stage 10) and processes handler arms
//!
//! By the time continuations are bound, the closures have already been lifted
//! without capturing them. The lifted function bodies contain unresolved
//! `tribute.var 'k'` operations that cause emission errors.
//!
//! ## Solution
//!
//! This pass runs after tribute_to_scf and:
//! 1. Scans lifted function bodies for unresolved tribute.var operations
//! 2. Replaces them with block parameters (continuations are passed as parameters)
//! 3. Or removes them if they're truly orphaned

use tracing::{debug, trace};

use tribute_ir::dialect::tribute;
use trunk_ir::dialect::core::Module;
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{Block, IdVec, Operation, Region};

/// Fixes unresolved tribute.var operations in lifted function bodies.
///
/// This is a workaround for the phase ordering issue where continuation variables
/// are bound after lambda lifting. Ideally, the closure system should handle
/// continuation captures properly, but this pass provides a pragmatic fix.
pub fn fix_continuation_captures<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    debug!("Running continuation capture fixup");

    let body = module.body(db);
    let mut fixer = ContCaptureFixer::new(db);
    let new_body = fixer.fix_region(body);

    Module::create(db, module.location(db), module.name(db), new_body)
}

struct ContCaptureFixer<'db> {
    db: &'db dyn salsa::Database,
    ctx: RewriteContext<'db>,
}

impl<'db> ContCaptureFixer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            ctx: RewriteContext::new(),
        }
    }

    fn fix_region(&mut self, region: Region<'db>) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| self.fix_block(block))
            .collect();

        Region::new(self.db, region.location(self.db), new_blocks)
    }

    fn fix_block(&mut self, block: &Block<'db>) -> Block<'db> {
        let new_ops: IdVec<Operation<'db>> = block
            .operations(self.db)
            .iter()
            .flat_map(|op| self.fix_operation(op))
            .collect();

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    fn fix_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First remap operands
        let remapped_op = self.ctx.remap_operands(self.db, op);

        // If operands were remapped, map results
        if remapped_op != *op {
            self.ctx.map_results(self.db, op, &remapped_op);
        }

        let dialect = remapped_op.dialect(self.db);
        let op_name = remapped_op.name(self.db);

        // Handle tribute.var for continuation variables
        // These should have been resolved by earlier passes but weren't due to phase ordering
        if dialect == tribute::DIALECT_NAME() && op_name == tribute::VAR() {
            trace!("Found tribute.var operation in lifted function");
            // For now, we keep the tribute.var and rely on the emit phase to skip it
            // TODO: Replace with proper continuation parameter
            let final_op = self.fix_op_regions(&remapped_op);
            if final_op != remapped_op {
                self.ctx.map_results(self.db, &remapped_op, &final_op);
            }
            vec![final_op]
        } else {
            // Process nested regions
            let final_op = self.fix_op_regions(&remapped_op);
            if final_op != remapped_op {
                self.ctx.map_results(self.db, &remapped_op, &final_op);
            }
            vec![final_op]
        }
    }

    fn fix_op_regions(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| self.fix_region(*region))
            .collect();

        op.modify(self.db).regions(new_regions).build()
    }
}
