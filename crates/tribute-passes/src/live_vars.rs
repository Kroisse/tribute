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
            // Preserve deterministic order by iterating in program order
            let used_after: HashSet<Value<'db>> = collect_used_values(db, &continuation_ops);
            let mut live_values: Vec<Value<'db>> = Vec::new();

            // 1. Block args (function parameters) are defined before any op
            for i in 0..block.args(db).len() {
                let arg_value = block.arg(db, i);
                if used_after.contains(&arg_value) {
                    live_values.push(arg_value);
                }
            }

            // 2. Op results before the shift, in program order
            for op in &ops[..*op_index] {
                for i in 0..op.results(db).len() {
                    let v = op.result(db, i);
                    if used_after.contains(&v) {
                        live_values.push(v);
                    }
                }
            }

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

/// Collect all values used as operands by a slice of operations.
/// Recursively traverses nested regions to arbitrary depth.
fn collect_used_values<'db>(
    db: &'db dyn salsa::Database,
    ops: &[Operation<'db>],
) -> HashSet<Value<'db>> {
    let mut used = HashSet::new();
    for op in ops {
        collect_used_in_op(db, *op, &mut used);
    }
    used
}

/// Recursively collect all values used as operands in an operation and its nested regions.
fn collect_used_in_op<'db>(
    db: &'db dyn salsa::Database,
    op: Operation<'db>,
    used: &mut HashSet<Value<'db>>,
) {
    // Collect operands of this operation
    for operand in op.operands(db).iter() {
        used.insert(*operand);
    }
    // Recursively check nested regions
    for region in op.regions(db).iter() {
        for block in region.blocks(db).iter() {
            for nested_op in block.operations(db).iter() {
                collect_used_in_op(db, *nested_op, used);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core, func};
    use trunk_ir::ir::BlockBuilder;
    use trunk_ir::{BlockId, DialectType, IdVec, Location, PathId, Span, Symbol};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    fn test_location_at(db: &dyn salsa::Database, offset: usize) -> Location<'_> {
        let path = PathId::new(db, "test.trb".to_owned());
        Location::new(path, Span::new(offset, offset + 1))
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

    /// Create a region with one shift point:
    /// ```
    /// %0 = arith.const 42
    /// %1 = cont.shift() -> Int  // shift point 0
    /// %2 = arith.add %0, %1     // uses %0 (live) and %1 (shift result)
    /// func.return %2
    /// ```
    #[salsa::tracked]
    fn create_single_shift_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        let mut builder = BlockBuilder::new(db, location);

        // %0 = arith.const 42 (defined before shift)
        let const_op = builder.op(arith::Const::i32(db, location, 42));
        let const_val = const_op.result(db);

        // %1 = cont.shift (shift point)
        let handler_region = Region::new(db, location, IdVec::new());
        let shift_op = builder.op(cont::shift(
            db,
            test_location_at(db, 100), // different location to identify
            vec![],
            i32_ty,
            0,
            ability_ref_ty,
            Symbol::new("get"),
            handler_region,
        ));
        let shift_result = shift_op.result(db);

        // %2 = arith.add %0, %1 (uses const_val which is live)
        let add_op = builder.op(arith::add(db, location, const_val, shift_result, i32_ty));
        let add_result = add_op.result(db);

        // func.return %2
        builder.op(func::r#return(db, location, Some(add_result)));

        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
    }

    /// Create a region with two shift points:
    /// ```
    /// %0 = cont.shift() -> Int  // shift point 0
    /// %1 = arith.const 1
    /// %2 = arith.add %0, %1
    /// %3 = cont.shift(%2)       // shift point 1, %0 is live
    /// func.return %0            // uses %0 from before both shifts
    /// ```
    #[salsa::tracked]
    fn create_dual_shift_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        let mut builder = BlockBuilder::new(db, location);

        // %0 = cont.shift() (shift point 0: State::get)
        let handler_region0 = Region::new(db, location, IdVec::new());
        let shift0 = builder.op(cont::shift(
            db,
            test_location_at(db, 100),
            vec![],
            i32_ty,
            0,
            ability_ref_ty,
            Symbol::new("get"),
            handler_region0,
        ));
        let n = shift0.result(db);

        // %1 = arith.const 1
        let one = builder.op(arith::Const::i32(db, location, 1));
        let one_val = one.result(db);

        // %2 = arith.add %0, %1
        let add_op = builder.op(arith::add(db, location, n, one_val, i32_ty));
        let n_plus_1 = add_op.result(db);

        // %3 = cont.shift(%2) (shift point 1: State::set)
        let handler_region1 = Region::new(db, location, IdVec::new());
        builder.op(cont::shift(
            db,
            test_location_at(db, 200),
            vec![n_plus_1],
            nil_ty,
            0,
            ability_ref_ty,
            Symbol::new("set"),
            handler_region1,
        ));

        // func.return %0 (n is live across shift point 1)
        builder.op(func::r#return(db, location, Some(n)));

        let block = builder.build();
        Region::new(db, location, IdVec::from(vec![block]))
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

    #[salsa_test]
    fn test_single_shift_point_detected(db: &salsa::DatabaseImpl) {
        let region = create_single_shift_region(db);
        let analysis = FunctionAnalysis::analyze(db, &region).unwrap();

        assert_eq!(
            analysis.shift_points.len(),
            1,
            "Should detect one shift point"
        );

        let shift = &analysis.shift_points[0];
        assert_eq!(shift.index, 0);
        assert_eq!(shift.total_shifts, 1);
    }

    #[salsa_test]
    fn test_single_shift_live_values(db: &salsa::DatabaseImpl) {
        let region = create_single_shift_region(db);
        let analysis = FunctionAnalysis::analyze(db, &region).unwrap();

        let shift = &analysis.shift_points[0];
        // The const value (42) is defined before shift and used after
        assert_eq!(
            shift.live_values.len(),
            1,
            "One value should be live (the const before shift)"
        );
    }

    #[salsa_test]
    fn test_dual_shift_points_detected(db: &salsa::DatabaseImpl) {
        let region = create_dual_shift_region(db);
        let analysis = FunctionAnalysis::analyze(db, &region).unwrap();

        assert_eq!(
            analysis.shift_points.len(),
            2,
            "Should detect two shift points"
        );

        assert_eq!(analysis.shift_points[0].index, 0);
        assert_eq!(analysis.shift_points[0].total_shifts, 2);
        assert_eq!(analysis.shift_points[1].index, 1);
        assert_eq!(analysis.shift_points[1].total_shifts, 2);
    }

    #[salsa_test]
    fn test_dual_shift_live_values(db: &salsa::DatabaseImpl) {
        let region = create_dual_shift_region(db);
        let analysis = FunctionAnalysis::analyze(db, &region).unwrap();

        // Shift 0: no values defined before it
        assert_eq!(
            analysis.shift_points[0].live_values.len(),
            0,
            "First shift has no live values (nothing defined before)"
        );

        // Shift 1: n (%0) is defined before and used after (in return)
        assert_eq!(
            analysis.shift_points[1].live_values.len(),
            1,
            "Second shift should have n as live value"
        );
    }

    #[salsa_test]
    fn test_continuation_ops_include_subsequent_shifts(db: &salsa::DatabaseImpl) {
        let region = create_dual_shift_region(db);
        let analysis = FunctionAnalysis::analyze(db, &region).unwrap();

        // First shift's continuation_ops should include the second shift
        let first_shift_cont_ops = &analysis.shift_points[0].continuation_ops;
        let has_second_shift = first_shift_cont_ops
            .iter()
            .any(|op| cont::Shift::from_operation(db, *op).is_ok());

        assert!(
            has_second_shift,
            "First shift's continuation_ops should include the second shift"
        );
    }
}
