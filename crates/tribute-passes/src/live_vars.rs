//! Live variable analysis for continuation transformations.
//!
//! This module analyzes which local variables are "live" at each shift point,
//! meaning they are defined before the shift and used after it.
//!
//! For Phase 1-2, we support sequential code only (no branches with shifts inside).

use std::collections::HashSet;
use std::ops::ControlFlow;

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::cont as arena_cont;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::walk;
use trunk_ir::dialect::cont;
use trunk_ir::{DialectOp, Operation, Region, Type, Value};

/// Information about a single shift point in a function.
#[derive(Debug, Clone)]
pub struct ShiftPoint<'db> {
    /// Index of this shift point (0, 1, 2, ...)
    pub index: usize,
    /// Total number of shift points in the function
    pub total_shifts: usize,
    /// The shift operation itself
    pub shift_op: Operation<'db>,
    /// Values that are live at this shift point (defined before, used after) with their types
    pub live_values: Vec<(Value<'db>, Type<'db>)>,
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

            // Phase 1-2: Reject if any continuation op has nested regions
            // (value remapping doesn't traverse into nested regions yet)
            if continuation_ops.iter().any(|op| has_nested_regions(db, op)) {
                tracing::debug!(
                    "FunctionAnalysis: skipping function with nested regions in continuation"
                );
                return None;
            }

            // Compute live variables: defined before shift, used in continuation
            // Preserve deterministic order by iterating in program order
            let used_after: HashSet<Value<'db>> = collect_used_values(db, &continuation_ops);
            let mut live_values: Vec<(Value<'db>, Type<'db>)> = Vec::new();

            // 1. Block args (function parameters) are defined before any op
            for i in 0..block.args(db).len() {
                let arg_value = block.arg(db, i);
                if used_after.contains(&arg_value) {
                    live_values.push((arg_value, block.arg_ty(db, i)));
                }
            }

            // 2. Op results before the shift, in program order
            for op in &ops[..*op_index] {
                for i in 0..op.results(db).len() {
                    let v = op.result(db, i);
                    if used_after.contains(&v) {
                        live_values.push((v, op.results(db)[i]));
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

/// Check if an operation has nested regions that would need value remapping.
/// Phase 1-2: We don't support operations with nested regions in continuation ops
/// because value remapping doesn't traverse into them yet.
/// Note: cont.shift has a handler region, but that's not part of continuation,
/// so we exclude shift operations from this check.
fn has_nested_regions<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
    // Shift has handler region, but that's not continuation code
    if cont::Shift::from_operation(db, *op).is_ok() {
        return false;
    }
    !op.regions(db).is_empty()
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
    use std::ops::ControlFlow;
    use trunk_ir::OperationWalk;

    region
        .walk::<cont::Shift, ()>(db, |_| ControlFlow::Break(()))
        .is_break()
}

/// Collect all values used as operands by a slice of operations.
/// Recursively traverses nested regions to arbitrary depth.
fn collect_used_values<'db>(
    db: &'db dyn salsa::Database,
    ops: &[Operation<'db>],
) -> HashSet<Value<'db>> {
    use std::ops::ControlFlow;
    use trunk_ir::{OperationWalk, WalkAction};

    let mut used = HashSet::new();
    for op in ops {
        let _ = op.walk_all::<()>(db, |nested_op| {
            for operand in nested_op.operands(db).iter() {
                used.insert(*operand);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    used
}

// ============================================================================
// Arena Versions
// ============================================================================

/// Information about a single shift point in a function (arena version).
#[derive(Debug, Clone)]
pub struct ArenaShiftPoint {
    /// Index of this shift point (0, 1, 2, ...)
    pub index: usize,
    /// Total number of shift points in the function
    pub total_shifts: usize,
    /// The shift operation itself
    pub shift_op: OpRef,
    /// Values that are live at this shift point (defined before, used after) with their types
    pub live_values: Vec<(ValueRef, TypeRef)>,
    /// Operations that come after this shift point (the continuation)
    pub continuation_ops: Vec<OpRef>,
}

/// Result of analyzing a function for shift points (arena version).
#[derive(Debug)]
pub struct ArenaFunctionAnalysis {
    /// All shift points in order
    pub shift_points: Vec<ArenaShiftPoint>,
}

impl ArenaFunctionAnalysis {
    /// Analyze a function body for shift points.
    ///
    /// For Phase 1-2, we only support single-block functions (no branches).
    /// Returns None if the function has multiple blocks or shifts inside branches.
    pub fn analyze(ctx: &IrContext, body: RegionRef) -> Option<Self> {
        let blocks = &ctx.region(body).blocks;

        // Phase 1-2: Only support single-block functions
        if blocks.len() != 1 {
            tracing::debug!("ArenaFunctionAnalysis: skipping multi-block function");
            return None;
        }

        let block = blocks[0];
        let ops: Vec<OpRef> = ctx.block(block).ops.iter().copied().collect();

        // Find all shift operations and their indices
        let mut shift_indices: Vec<(usize, OpRef)> = Vec::new();
        for (i, &op) in ops.iter().enumerate() {
            if arena_cont::Shift::matches(ctx, op) {
                if has_shift_in_nested_region_arena(ctx, op) {
                    tracing::debug!(
                        "ArenaFunctionAnalysis: skipping function with shift in nested region"
                    );
                    return None;
                }
                shift_indices.push((i, op));
            } else if has_shift_in_nested_region_arena(ctx, op) {
                tracing::debug!(
                    "ArenaFunctionAnalysis: skipping function with shift in nested region"
                );
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
        for (shift_idx, &(op_index, shift_op)) in shift_indices.iter().enumerate() {
            let continuation_ops: Vec<OpRef> = ops[op_index + 1..].to_vec();

            // Phase 1-2: Reject if any continuation op has nested regions
            if continuation_ops
                .iter()
                .any(|&op| has_nested_regions_arena(ctx, op))
            {
                tracing::debug!(
                    "ArenaFunctionAnalysis: skipping function with nested regions in continuation"
                );
                return None;
            }

            // Compute live variables: defined before shift, used in continuation
            let used_after: HashSet<ValueRef> = collect_used_values_arena(ctx, &continuation_ops);
            let mut live_values: Vec<(ValueRef, TypeRef)> = Vec::new();

            // 1. Block args (function parameters) are defined before any op
            let block_args = ctx.block_args(block);
            for &arg in block_args {
                if used_after.contains(&arg) {
                    live_values.push((arg, ctx.value_ty(arg)));
                }
            }

            // 2. Op results before the shift, in program order
            for &op in &ops[..op_index] {
                let results = ctx.op_results(op);
                for &v in results {
                    if used_after.contains(&v) {
                        live_values.push((v, ctx.value_ty(v)));
                    }
                }
            }

            shift_points.push(ArenaShiftPoint {
                index: shift_idx,
                total_shifts,
                shift_op,
                live_values,
                continuation_ops,
            });
        }

        Some(Self { shift_points })
    }
}

/// Check if an operation has nested regions (arena version).
fn has_nested_regions_arena(ctx: &IrContext, op: OpRef) -> bool {
    // Shift has handler region, but that's not continuation code
    if arena_cont::Shift::matches(ctx, op) {
        return false;
    }
    !ctx.op(op).regions.is_empty()
}

/// Check if an operation has a shift inside any of its nested regions (arena version).
fn has_shift_in_nested_region_arena(ctx: &IrContext, op: OpRef) -> bool {
    for &region in &ctx.op(op).regions {
        if region_contains_shift_arena(ctx, region) {
            return true;
        }
    }
    false
}

/// Check if a region contains any shift operation (arena version).
fn region_contains_shift_arena(ctx: &IrContext, region: RegionRef) -> bool {
    walk::walk_region::<()>(ctx, region, &mut |op| {
        if arena_cont::Shift::matches(ctx, op) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(walk::WalkAction::Advance)
        }
    })
    .is_break()
}

/// Collect all values used as operands by a slice of operations (arena version).
/// Recursively traverses nested regions.
fn collect_used_values_arena(ctx: &IrContext, ops: &[OpRef]) -> HashSet<ValueRef> {
    let mut used = HashSet::new();
    for &op in ops {
        let _ = walk::walk_op::<()>(ctx, op, &mut |nested_op| {
            for &operand in ctx.op_operands(nested_op) {
                used.insert(operand);
            }
            ControlFlow::Continue(walk::WalkAction::Advance)
        });
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{arith, core, func};
    use trunk_ir::ir::BlockBuilder;
    use trunk_ir::{Attribute, BlockId, DialectType, IdVec, Location, PathId, Span, Symbol};

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
    /// %tag = arith.const 0
    /// %1 = cont.shift(%tag) -> Int  // shift point 0
    /// %2 = arith.add %0, %1     // uses %0 (live) and %1 (shift result)
    /// func.return %2
    /// ```
    #[salsa::tracked]
    fn create_single_shift_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let prompt_tag_ty = cont::PromptTag::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        let mut builder = BlockBuilder::new(db, location);

        // %0 = arith.const 42 (defined before shift)
        let const_op = builder.op(arith::Const::i32(db, location, 42));
        let const_val = const_op.result(db);

        // %tag = arith.const 0 (prompt tag)
        let tag_const = builder.op(arith::r#const(
            db,
            location,
            prompt_tag_ty,
            Attribute::IntBits(0),
        ));
        let tag_val = tag_const.result(db);

        // %1 = cont.shift(%tag) (shift point)
        let handler_region = Region::new(db, location, IdVec::new());
        let shift_op = builder.op(cont::shift(
            db,
            test_location_at(db, 100), // different location to identify
            tag_val,
            vec![],
            i32_ty,
            ability_ref_ty,
            Symbol::new("get"),
            None, // op_table_index
            None, // op_offset
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
    /// %tag = arith.const 0
    /// %0 = cont.shift(%tag) -> Int  // shift point 0
    /// %1 = arith.const 1
    /// %2 = arith.add %0, %1
    /// %3 = cont.shift(%tag, %2)     // shift point 1, %0 is live
    /// func.return %0                // uses %0 from before both shifts
    /// ```
    #[salsa::tracked]
    fn create_dual_shift_region(db: &dyn salsa::Database) -> Region<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();
        let nil_ty = core::Nil::new(db).as_type();
        let prompt_tag_ty = cont::PromptTag::new(db).as_type();
        let ability_ref_ty =
            core::AbilityRefType::with_params(db, Symbol::new("State"), IdVec::from(vec![i32_ty]))
                .as_type();

        let mut builder = BlockBuilder::new(db, location);

        // %tag = arith.const 0 (prompt tag)
        let tag_const = builder.op(arith::r#const(
            db,
            location,
            prompt_tag_ty,
            Attribute::IntBits(0),
        ));
        let tag_val = tag_const.result(db);

        // %0 = cont.shift(%tag) (shift point 0: State::get)
        let handler_region0 = Region::new(db, location, IdVec::new());
        let shift0 = builder.op(cont::shift(
            db,
            test_location_at(db, 100),
            tag_val,
            vec![],
            i32_ty,
            ability_ref_ty,
            Symbol::new("get"),
            None, // op_table_index
            None, // op_offset
            handler_region0,
        ));
        let n = shift0.result(db);

        // %1 = arith.const 1
        let one = builder.op(arith::Const::i32(db, location, 1));
        let one_val = one.result(db);

        // %2 = arith.add %0, %1
        let add_op = builder.op(arith::add(db, location, n, one_val, i32_ty));
        let n_plus_1 = add_op.result(db);

        // %3 = cont.shift(%tag, %2) (shift point 1: State::set)
        let handler_region1 = Region::new(db, location, IdVec::new());
        builder.op(cont::shift(
            db,
            test_location_at(db, 200),
            tag_val,
            vec![n_plus_1],
            nil_ty,
            ability_ref_ty,
            Symbol::new("set"),
            None, // op_table_index
            None, // op_offset
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

        // Shift 0: tag_val is defined before and used after (in shift 1)
        assert_eq!(
            analysis.shift_points[0].live_values.len(),
            1,
            "First shift has one live value (tag_val used in second shift)"
        );

        // Shift 1: n (%0) is defined before and used after (in return)
        // Note: tag_val is also defined before shift 1 but not used after it
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

    // ====================================================================
    // Arena tests
    // ====================================================================

    mod arena_tests {
        use super::*;
        use trunk_ir::arena::context::{BlockData, IrContext, RegionData};
        use trunk_ir::arena::dialect::{
            arith as arena_arith, cont as arena_cont, func as arena_func,
        };
        use trunk_ir::arena::ops::ArenaDialectOp;
        use trunk_ir::arena::types::{Attribute as ArenaAttribute, Location, TypeDataBuilder};
        use trunk_ir::location::Span;
        use trunk_ir::smallvec::smallvec;

        fn test_ctx() -> (IrContext, Location) {
            let mut ctx = IrContext::new();
            let path = ctx.paths.intern("test.trb".to_owned());
            let loc = Location::new(path, Span::new(0, 0));
            (ctx, loc)
        }

        fn i32_type(ctx: &mut IrContext) -> TypeRef {
            ctx.types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
        }

        fn nil_type(ctx: &mut IrContext) -> TypeRef {
            ctx.types
                .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
        }

        fn prompt_tag_type(ctx: &mut IrContext) -> TypeRef {
            ctx.types.intern(
                TypeDataBuilder::new(Symbol::new("cont"), Symbol::new("prompt_tag")).build(),
            )
        }

        fn ability_ref_type(ctx: &mut IrContext) -> TypeRef {
            ctx.types.intern(
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                    .attr("name", ArenaAttribute::Symbol(Symbol::new("State")))
                    .build(),
            )
        }

        #[test]
        fn arena_empty_function() {
            let (mut ctx, loc) = test_ctx();
            let block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let region = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![block],
                parent_op: None,
            });

            let analysis = ArenaFunctionAnalysis::analyze(&ctx, region).unwrap();
            assert!(analysis.shift_points.is_empty());
        }

        #[test]
        fn arena_multi_block_not_supported() {
            let (mut ctx, loc) = test_ctx();
            let block1 = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let block2 = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let region = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![block1, block2],
                parent_op: None,
            });

            assert!(
                ArenaFunctionAnalysis::analyze(&ctx, region).is_none(),
                "Multi-block functions not supported in Phase 1-2"
            );
        }

        /// Create a region with one shift point:
        /// %0 = arith.const 42
        /// %tag = arith.const 0
        /// %1 = cont.shift(%tag) -> Int
        /// %2 = arith.add %0, %1
        /// func.return %2
        fn create_single_shift_region_arena(ctx: &mut IrContext, loc: Location) -> RegionRef {
            let i32_ty = i32_type(ctx);
            let prompt_tag_ty = prompt_tag_type(ctx);
            let ability_ref_ty = ability_ref_type(ctx);

            let handler_region = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![],
                parent_op: None,
            });

            let const_op = arena_arith::r#const(ctx, loc, i32_ty, ArenaAttribute::IntBits(42));
            let const_val = const_op.result(ctx);

            let tag_const =
                arena_arith::r#const(ctx, loc, prompt_tag_ty, ArenaAttribute::IntBits(0));
            let tag_val = tag_const.result(ctx);

            let shift_op = arena_cont::shift(
                ctx,
                loc,
                tag_val,
                [],
                i32_ty,
                ability_ref_ty,
                Symbol::new("get"),
                None,
                None,
                handler_region,
            );
            let shift_result = shift_op.result(ctx);

            let add_op = arena_arith::add(ctx, loc, const_val, shift_result, i32_ty);
            let add_result = add_op.result(ctx);

            let ret_op = arena_func::r#return(ctx, loc, [add_result]);

            let block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            ctx.push_op(block, const_op.op_ref());
            ctx.push_op(block, tag_const.op_ref());
            ctx.push_op(block, shift_op.op_ref());
            ctx.push_op(block, add_op.op_ref());
            ctx.push_op(block, ret_op.op_ref());

            ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![block],
                parent_op: None,
            })
        }

        #[test]
        fn arena_single_shift_point_detected() {
            let (mut ctx, loc) = test_ctx();
            let region = create_single_shift_region_arena(&mut ctx, loc);
            let analysis = ArenaFunctionAnalysis::analyze(&ctx, region).unwrap();

            assert_eq!(analysis.shift_points.len(), 1);
            assert_eq!(analysis.shift_points[0].index, 0);
            assert_eq!(analysis.shift_points[0].total_shifts, 1);
        }

        #[test]
        fn arena_single_shift_live_values() {
            let (mut ctx, loc) = test_ctx();
            let region = create_single_shift_region_arena(&mut ctx, loc);
            let analysis = ArenaFunctionAnalysis::analyze(&ctx, region).unwrap();

            let shift = &analysis.shift_points[0];
            assert_eq!(
                shift.live_values.len(),
                1,
                "One value should be live (the const before shift)"
            );
        }

        /// Create a region with two shift points:
        /// %tag = arith.const 0
        /// %0 = cont.shift(%tag) -> Int  (State::get)
        /// %1 = arith.const 1
        /// %2 = arith.add %0, %1
        /// %3 = cont.shift(%tag, %2)     (State::set)
        /// func.return %0
        fn create_dual_shift_region_arena(ctx: &mut IrContext, loc: Location) -> RegionRef {
            let i32_ty = i32_type(ctx);
            let nil_ty = nil_type(ctx);
            let prompt_tag_ty = prompt_tag_type(ctx);
            let ability_ref_ty = ability_ref_type(ctx);

            let handler_region0 = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![],
                parent_op: None,
            });
            let handler_region1 = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![],
                parent_op: None,
            });

            let tag_const =
                arena_arith::r#const(ctx, loc, prompt_tag_ty, ArenaAttribute::IntBits(0));
            let tag_val = tag_const.result(ctx);

            let shift0 = arena_cont::shift(
                ctx,
                loc,
                tag_val,
                [],
                i32_ty,
                ability_ref_ty,
                Symbol::new("get"),
                None,
                None,
                handler_region0,
            );
            let n = shift0.result(ctx);

            let one = arena_arith::r#const(ctx, loc, i32_ty, ArenaAttribute::IntBits(1));
            let one_val = one.result(ctx);

            let add_op = arena_arith::add(ctx, loc, n, one_val, i32_ty);
            let n_plus_1 = add_op.result(ctx);

            let shift1 = arena_cont::shift(
                ctx,
                loc,
                tag_val,
                [n_plus_1],
                nil_ty,
                ability_ref_ty,
                Symbol::new("set"),
                None,
                None,
                handler_region1,
            );

            let ret_op = arena_func::r#return(ctx, loc, [n]);

            let block = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            ctx.push_op(block, tag_const.op_ref());
            ctx.push_op(block, shift0.op_ref());
            ctx.push_op(block, one.op_ref());
            ctx.push_op(block, add_op.op_ref());
            ctx.push_op(block, shift1.op_ref());
            ctx.push_op(block, ret_op.op_ref());

            ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![block],
                parent_op: None,
            })
        }

        #[test]
        fn arena_dual_shift_points_detected() {
            let (mut ctx, loc) = test_ctx();
            let region = create_dual_shift_region_arena(&mut ctx, loc);
            let analysis = ArenaFunctionAnalysis::analyze(&ctx, region).unwrap();

            assert_eq!(analysis.shift_points.len(), 2);
            assert_eq!(analysis.shift_points[0].index, 0);
            assert_eq!(analysis.shift_points[0].total_shifts, 2);
            assert_eq!(analysis.shift_points[1].index, 1);
            assert_eq!(analysis.shift_points[1].total_shifts, 2);
        }

        #[test]
        fn arena_dual_shift_live_values() {
            let (mut ctx, loc) = test_ctx();
            let region = create_dual_shift_region_arena(&mut ctx, loc);
            let analysis = ArenaFunctionAnalysis::analyze(&ctx, region).unwrap();

            // Shift 0: tag_val is defined before and used after (in shift 1)
            assert_eq!(
                analysis.shift_points[0].live_values.len(),
                1,
                "First shift has one live value (tag_val used in second shift)"
            );

            // Shift 1: n (%0) is defined before and used after (in return)
            assert_eq!(
                analysis.shift_points[1].live_values.len(),
                1,
                "Second shift should have n as live value"
            );
        }

        #[test]
        fn arena_continuation_ops_include_subsequent_shifts() {
            let (mut ctx, loc) = test_ctx();
            let region = create_dual_shift_region_arena(&mut ctx, loc);
            let analysis = ArenaFunctionAnalysis::analyze(&ctx, region).unwrap();

            let first_shift_cont_ops = &analysis.shift_points[0].continuation_ops;
            let has_second_shift = first_shift_cont_ops
                .iter()
                .any(|&op| arena_cont::Shift::matches(&ctx, op));

            assert!(
                has_second_shift,
                "First shift's continuation_ops should include the second shift"
            );
        }
    }
}
