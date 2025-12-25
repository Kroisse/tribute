//! Constant inlining pass for Tribute.
//!
//! This pass inlines constant values at their use sites:
//! - Finds `src.var` operations marked with `resolved_const=true`
//! - Replaces them with `arith.const` operations containing the inlined value
//!
//! ## Example
//!
//! ```tribute
//! const MAX_SIZE = 1024;
//!
//! fn example() {
//!     let x = MAX_SIZE;
//! }
//! ```
//!
//! After name resolution, `MAX_SIZE` reference is marked as `src.var` with:
//! - `resolved_const = true`
//! - `value = 1024`
//!
//! After this pass, it becomes:
//! - `arith.const(1024)` with type `i64`

use std::collections::HashMap;

use trunk_ir::dialect::arith;
use trunk_ir::dialect::core::Module;
use trunk_ir::{Block, IdVec, Operation, Region, Value};

// =============================================================================
// Attribute Keys
// =============================================================================

trunk_ir::symbols! {
    ATTR_RESOLVED_CONST => "resolved_const",
    ATTR_VALUE => "value",
    ATTR_NAME => "name",
}

// =============================================================================
// Const Inliner
// =============================================================================

/// Constant inliner context.
///
/// Transforms `src.var` operations marked as resolved constants into
/// `arith.const` operations with inlined values.
pub struct ConstInliner<'db> {
    db: &'db dyn salsa::Database,
    /// Maps old values to their replacements during transformation.
    value_map: HashMap<Value<'db>, Value<'db>>,
}

impl<'db> ConstInliner<'db> {
    /// Create a new const inliner.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            value_map: HashMap::new(),
        }
    }

    /// Look up a mapped value, or return the original if not mapped.
    fn lookup_value(&self, old: Value<'db>) -> Value<'db> {
        self.value_map.get(&old).copied().unwrap_or(old)
    }

    /// Map a value from old to new.
    fn map_value(&mut self, old: Value<'db>, new: Value<'db>) {
        self.value_map.insert(old, new);
    }

    /// Inline constants in a module.
    pub fn inline_module(&mut self, module: &Module<'db>) -> Module<'db> {
        let body = module.body(self.db);
        let new_body = self.inline_region(&body);

        Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        )
    }

    /// Inline constants in a region.
    fn inline_region(&mut self, region: &Region<'db>) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| self.inline_block(block))
            .collect();

        Region::new(self.db, region.location(self.db), new_blocks)
    }

    /// Inline constants in a block.
    fn inline_block(&mut self, block: &Block<'db>) -> Block<'db> {
        let new_ops: IdVec<Operation<'db>> = block
            .operations(self.db)
            .iter()
            .flat_map(|op| self.inline_operation(op))
            .collect();

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    /// Inline constants in an operation.
    ///
    /// Returns the transformed operation(s). May return empty vec if erased,
    /// or a single transformed operation.
    fn inline_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First, remap operands from previous transformations
        let remapped_op = self.remap_operands(op);

        // Check if this is a resolved const reference
        if self.is_resolved_const(&remapped_op)
            && let Some(inlined) = self.inline_const_ref(&remapped_op)
        {
            return vec![inlined];
        }

        // Not a const reference - recursively process regions
        vec![self.inline_op_regions(&remapped_op)]
    }

    /// Check if an operation is a resolved const reference.
    fn is_resolved_const(&self, op: &Operation<'db>) -> bool {
        use trunk_ir::Attribute;

        let attrs = op.attributes(self.db);
        matches!(
            attrs.get(&ATTR_RESOLVED_CONST()),
            Some(Attribute::Bool(true)) | Some(Attribute::IntBits(1))
        )
    }

    /// Inline a const reference operation.
    fn inline_const_ref(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        use trunk_ir::DialectOp;

        let attrs = op.attributes(self.db);
        let value_attr = attrs.get(&ATTR_VALUE())?;

        // Get the result type
        let result_ty = op.results(self.db).first().copied()?;
        let location = op.location(self.db);

        // Create arith.const with the inlined value
        let const_op = arith::r#const(self.db, location, result_ty, value_attr.clone());
        let new_operation = const_op.as_operation();

        // Map old result to new result
        let old_result = op.result(self.db, 0);
        let new_result = new_operation.result(self.db, 0);
        self.map_value(old_result, new_result);

        Some(new_operation)
    }

    /// Remap operands using the current value map.
    fn remap_operands(&self, op: &Operation<'db>) -> Operation<'db> {
        let operands = op.operands(self.db);
        let mut new_operands: IdVec<Value<'db>> = IdVec::new();
        let mut changed = false;

        for &operand in operands.iter() {
            let mapped = self.lookup_value(operand);
            new_operands.push(mapped);
            if mapped != operand {
                changed = true;
            }
        }

        if !changed {
            return *op;
        }

        op.modify(self.db).operands(new_operands).build()
    }

    /// Recursively inline constants in regions within an operation.
    fn inline_op_regions(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| self.inline_region(region))
            .collect();

        op.modify(self.db).regions(new_regions).build()
    }
}

// =============================================================================
// Pipeline Integration
// =============================================================================

/// Inline constants in a module (non-tracked version for internal use).
///
/// The tracked version is in pipeline.rs (stage_const_inline).
pub fn inline_module<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) -> Module<'db> {
    let mut inliner = ConstInliner::new(db);
    inliner.inline_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;

    // Import the resolve module tests to reuse their test data
    use crate::resolve::tests::resolve_const_reference_module;

    #[salsa::tracked]
    fn inline_after_resolve(db: &dyn salsa::Database) -> Module<'_> {
        // Get the resolved module (with marked const references)
        let resolved = resolve_const_reference_module(db);
        // Run const inlining
        inline_module(db, &resolved)
    }

    #[salsa_test]
    fn test_const_inlining(db: &salsa::DatabaseImpl) {
        use trunk_ir::{Attribute, Symbol};

        let inlined = inline_after_resolve(db);

        // Collect all operations
        let mut ops = Vec::new();
        for block in inlined.body(db).blocks(db).iter() {
            for op in block.operations(db).iter() {
                ops.push(*op);
                // Also collect from nested regions (function bodies)
                for region in op.regions(db).iter() {
                    for nested_block in region.blocks(db).iter() {
                        for nested_op in nested_block.operations(db).iter() {
                            ops.push(*nested_op);
                        }
                    }
                }
            }
        }

        // Should have no more src.var with resolved_const
        let resolved_consts: Vec<_> = ops
            .iter()
            .filter(|op| {
                op.dialect(db) == Symbol::new("src")
                    && op.name(db) == Symbol::new("var")
                    && matches!(
                        op.attributes(db).get(&Symbol::new("resolved_const")),
                        Some(Attribute::Bool(true))
                    )
            })
            .collect();

        assert!(
            resolved_consts.is_empty(),
            "resolved const references should be inlined"
        );

        // Should have arith.const with value 1024
        let const_ops: Vec<_> = ops
            .iter()
            .filter(|op| {
                op.dialect(db) == Symbol::new("arith") && op.name(db) == Symbol::new("const")
            })
            .collect();

        assert!(!const_ops.is_empty(), "should have arith.const operations");

        // Verify the const has the right value
        let has_correct_value = const_ops.iter().any(|op| {
            matches!(
                op.attributes(db).get(&Symbol::new("value")),
                Some(Attribute::IntBits(1024))
            )
        });

        assert!(has_correct_value, "arith.const should have value 1024");
    }
}
