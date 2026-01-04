//! Rewrite context for tracking value mappings.
//!
//! The `RewriteContext` tracks mappings from old values to new values
//! during IR transformation, enabling operand remapping across rewrites.

use std::collections::{HashMap, HashSet};

use crate::{IdVec, Operation, Value};

/// Context for IR rewriting.
///
/// Tracks value mappings when operations are transformed, allowing
/// subsequent operations to reference the new values. Also tracks
/// the number of changes made for fixpoint detection.
pub struct RewriteContext<'db> {
    /// Maps old values to their replacements.
    value_map: HashMap<Value<'db>, Value<'db>>,

    /// Number of changes made in this context.
    changes: usize,
}

impl<'db> RewriteContext<'db> {
    /// Create a new empty rewrite context.
    pub fn new() -> Self {
        Self {
            value_map: HashMap::new(),
            changes: 0,
        }
    }

    /// Look up the mapped value for an old value.
    /// Returns the mapped value if one exists, otherwise the original.
    /// Follows the chain of mappings to get the final value.
    pub fn lookup(&self, old: Value<'db>) -> Value<'db> {
        let mut current = old;
        let mut visited = HashSet::new();
        while let Some(&mapped) = self.value_map.get(&current) {
            if !visited.insert(current) {
                break; // Cycle detected
            }
            current = mapped;
        }
        current
    }

    /// Register a value mapping from old to new.
    pub fn map_value(&mut self, old: Value<'db>, new: Value<'db>) {
        self.value_map.insert(old, new);
    }

    /// Record that a change was made.
    pub fn record_change(&mut self) {
        self.changes += 1;
    }

    /// Get the number of changes made.
    pub fn changes_made(&self) -> usize {
        self.changes
    }

    /// Remap all operands of an operation using the current value map.
    ///
    /// Returns a new operation with remapped operands if any changed,
    /// or the original operation if no operands were mapped.
    pub fn remap_operands(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> Operation<'db> {
        let operands = op.operands(db);
        let mut new_operands: IdVec<Value<'db>> = IdVec::new();
        let mut changed = false;

        for &operand in operands.iter() {
            let mapped = self.lookup(operand);
            new_operands.push(mapped);
            if mapped != operand {
                changed = true;
            }
        }

        if !changed {
            return *op;
        }

        // Rebuild the operation with remapped operands
        op.modify(db).operands(new_operands).build()
    }

    /// Register value mappings from old operation results to new operation results.
    ///
    /// Maps each result of `old_op` to the corresponding result of `new_op`.
    pub fn map_results(
        &mut self,
        db: &'db dyn salsa::Database,
        old_op: &Operation<'db>,
        new_op: &Operation<'db>,
    ) {
        let old_results = old_op.results(db);
        let new_results = new_op.results(db);

        // Map results 1:1 up to the minimum count
        let count = old_results.len().min(new_results.len());
        for i in 0..count {
            let old_val = old_op.result(db, i);
            let new_val = new_op.result(db, i);
            self.map_value(old_val, new_val);
        }
    }

    /// Recursively remap all operands in a region and its nested regions.
    ///
    /// This is useful when a value mapping needs to be applied to an
    /// already-constructed region (e.g., arm bodies in case expressions
    /// that reference a scrutinee value that was later remapped).
    pub fn remap_region_operands(
        &self,
        db: &'db dyn salsa::Database,
        region: &crate::Region<'db>,
    ) -> crate::Region<'db> {
        let new_blocks: IdVec<crate::Block<'db>> = region
            .blocks(db)
            .iter()
            .map(|block| self.remap_block_operands(db, block))
            .collect();
        crate::Region::new(db, region.location(db), new_blocks)
    }

    /// Remap all operands in a block and its operations' nested regions.
    fn remap_block_operands(
        &self,
        db: &'db dyn salsa::Database,
        block: &crate::Block<'db>,
    ) -> crate::Block<'db> {
        let new_ops: IdVec<Operation<'db>> = block
            .operations(db)
            .iter()
            .map(|op| self.remap_operation_deep(db, op))
            .collect();
        crate::Block::new(
            db,
            block.id(db),
            block.location(db),
            block.args(db).clone(),
            new_ops,
        )
    }

    /// Remap operands of an operation and recursively remap its nested regions.
    fn remap_operation_deep(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
    ) -> Operation<'db> {
        // First remap the operands
        let remapped = self.remap_operands(db, op);

        // If no nested regions, we're done
        if op.regions(db).is_empty() {
            return remapped;
        }

        // Recursively remap nested regions
        let new_regions: IdVec<crate::Region<'db>> = op
            .regions(db)
            .iter()
            .map(|r| self.remap_region_operands(db, r))
            .collect();

        // Rebuild operation with new regions
        remapped.modify(db).regions(new_regions).build()
    }
}

impl Default for RewriteContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}
