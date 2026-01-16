//! Live variable analysis for continuation transformations.
//!
//! This module analyzes which local variables are "live" at each shift point,
//! meaning they are defined before the shift and used after it.
//!
//! For Phase 1-2, we support sequential code only (no branches with shifts inside).

use std::collections::HashSet;

use trunk_ir::dialect::cont;
use trunk_ir::{DialectOp, Operation, Region, Value};

/// Information about a single shift point in a function.
#[derive(Debug, Clone)]
pub struct ShiftPoint<'db> {
    /// Index of this shift point (0, 1, 2, ...)
    pub index: usize,
    /// Total number of shift points in the function
    pub total_shifts: usize,
    /// The shift operation itself
    pub shift_op: Operation<'db>,
    /// Values that are live at this shift point (defined before, used after)
    pub live_values: Vec<Value<'db>>,
    /// Operations that come after this shift point (the continuation)
    pub continuation_ops: Vec<Operation<'db>>,
}

/// Result of analyzing a function for shift points.
#[derive(Debug)]
pub struct FunctionAnalysis<'db> {
    /// All shift points in order
    pub shift_points: Vec<ShiftPoint<'db>>,
}

impl<'db> FunctionAnalysis<'db> {
    /// Analyze a function body for shift points.
    ///
    /// For Phase 1-2, we only support single-block functions (no branches).
    /// Returns None if the function has multiple blocks or shifts inside branches.
    pub fn analyze(db: &'db dyn salsa::Database, body: &Region<'db>) -> Option<Self> {
        let blocks = body.blocks(db);

        // Phase 1-2: Only support single-block functions
        if blocks.len() != 1 {
            tracing::debug!("FunctionAnalysis: skipping multi-block function");
            return None;
        }

        let block = blocks.first()?;
        let ops: Vec<Operation<'db>> = block.operations(db).iter().copied().collect();

        // Find all shift operations and their indices
        let mut shift_indices: Vec<(usize, Operation<'db>)> = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            if cont::Shift::from_operation(db, *op).is_ok() {
                // Check if shift is inside a nested region (branch) - not supported in Phase 1-2
                if has_shift_in_nested_region(db, op) {
                    tracing::debug!(
                        "FunctionAnalysis: skipping function with shift in nested region"
                    );
                    return None;
                }
                shift_indices.push((i, *op));
            } else if has_shift_in_nested_region(db, op) {
                // Shift inside scf.if or other nested control flow - not supported
                tracing::debug!("FunctionAnalysis: skipping function with shift in nested region");
                return None;
            }
        }

        if shift_indices.is_empty() {
            return Some(Self {
                shift_points: Vec::new(),
            });
        }

        // Build shift points with continuation ops and live variables
        let mut shift_points = Vec::new();
        let total_shifts = shift_indices.len();
        for (shift_idx, (op_index, shift_op)) in shift_indices.iter().enumerate() {
            // Continuation ops: everything after this shift until the end
            // (includes subsequent shifts - they will be handled in resume functions)
            let continuation_ops: Vec<Operation<'db>> = ops[op_index + 1..].to_vec();

            // Compute live variables: defined before shift, used in continuation
            let defined_before: HashSet<Value<'db>> = collect_defined_values(db, &ops[..*op_index]);
            let used_after: HashSet<Value<'db>> = collect_used_values(db, &continuation_ops);

            let live_values: Vec<Value<'db>> =
                defined_before.intersection(&used_after).copied().collect();

            shift_points.push(ShiftPoint {
                index: shift_idx,
                total_shifts,
                shift_op: *shift_op,
                live_values,
                continuation_ops,
            });
        }

        Some(Self { shift_points })
    }
}

/// Check if an operation has a shift inside any of its nested regions.
fn has_shift_in_nested_region<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
    for region in op.regions(db).iter() {
        if region_contains_shift(db, region) {
            return true;
        }
    }
    false
}

/// Check if a region contains any shift operation.
fn region_contains_shift<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> bool {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if cont::Shift::from_operation(db, *op).is_ok() {
                return true;
            }
            // Recursively check nested regions
            for nested in op.regions(db).iter() {
                if region_contains_shift(db, nested) {
                    return true;
                }
            }
        }
    }
    false
}

/// Collect all values defined by a slice of operations.
fn collect_defined_values<'db>(
    db: &'db dyn salsa::Database,
    ops: &[Operation<'db>],
) -> HashSet<Value<'db>> {
    let mut defined = HashSet::new();
    for op in ops {
        for i in 0..op.results(db).len() {
            defined.insert(op.result(db, i));
        }
    }
    defined
}

/// Collect all values used as operands by a slice of operations.
fn collect_used_values<'db>(
    db: &'db dyn salsa::Database,
    ops: &[Operation<'db>],
) -> HashSet<Value<'db>> {
    let mut used = HashSet::new();
    for op in ops {
        for operand in op.operands(db).iter() {
            used.insert(*operand);
        }
        // Also check operands in nested regions
        for region in op.regions(db).iter() {
            for block in region.blocks(db).iter() {
                for nested_op in block.operations(db).iter() {
                    for operand in nested_op.operands(db).iter() {
                        used.insert(*operand);
                    }
                }
            }
        }
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{BlockId, IdVec, Location, PathId, Span};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn create_empty_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let block =
            trunk_ir::Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        Region::new(db, location, IdVec::from(vec![block]))
    }

    #[salsa::tracked]
    fn create_multi_block_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let block1 =
            trunk_ir::Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        let block2 =
            trunk_ir::Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());
        Region::new(db, location, IdVec::from(vec![block1, block2]))
    }

    #[salsa_test]
    fn test_empty_function(db: &salsa::DatabaseImpl) {
        let region = create_empty_region(db);
        let analysis = FunctionAnalysis::analyze(db, &region).unwrap();
        assert!(analysis.shift_points.is_empty());
    }

    #[salsa_test]
    fn test_multi_block_not_supported(db: &salsa::DatabaseImpl) {
        let region = create_multi_block_region(db);
        let analysis = FunctionAnalysis::analyze(db, &region);
        assert!(
            analysis.is_none(),
            "Multi-block functions not supported in Phase 1-2"
        );
    }
}
