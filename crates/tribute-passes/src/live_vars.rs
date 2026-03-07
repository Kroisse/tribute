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
use trunk_ir::arena::ops::DialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::walk;

/// Information about a single shift point in a function.
#[derive(Debug, Clone)]
pub struct ShiftPoint {
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

/// Result of analyzing a function for shift points.
#[derive(Debug)]
pub struct FunctionAnalysis {
    /// All shift points in order
    pub shift_points: Vec<ShiftPoint>,
}

impl FunctionAnalysis {
    /// Analyze a function body for shift points.
    ///
    /// For Phase 1-2, we only support single-block functions (no branches).
    /// Returns None if the function has multiple blocks or shifts inside branches.
    pub fn analyze(ctx: &IrContext, body: RegionRef) -> Option<Self> {
        let blocks = &ctx.region(body).blocks;

        // Phase 1-2: Only support single-block functions
        if blocks.len() != 1 {
            tracing::debug!("FunctionAnalysis: skipping multi-block function");
            return None;
        }

        let block = blocks[0];
        let ops: Vec<OpRef> = ctx.block(block).ops.iter().copied().collect();

        // Find all shift operations and their indices
        let mut shift_indices: Vec<(usize, OpRef)> = Vec::new();
        for (i, &op) in ops.iter().enumerate() {
            if arena_cont::Shift::matches(ctx, op) {
                if has_shift_in_nested_region(ctx, op) {
                    tracing::debug!(
                        "FunctionAnalysis: skipping function with shift in nested region"
                    );
                    return None;
                }
                shift_indices.push((i, op));
            } else if has_shift_in_nested_region(ctx, op) {
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
        for (shift_idx, &(op_index, shift_op)) in shift_indices.iter().enumerate() {
            let continuation_ops: Vec<OpRef> = ops[op_index + 1..].to_vec();

            // Phase 1-2: Reject if any continuation op has nested regions
            if continuation_ops
                .iter()
                .any(|&op| has_nested_regions(ctx, op))
            {
                tracing::debug!(
                    "FunctionAnalysis: skipping function with nested regions in continuation"
                );
                return None;
            }

            // Compute live variables: defined before shift, used in continuation
            let used_after: HashSet<ValueRef> = collect_used_values(ctx, &continuation_ops);
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

            shift_points.push(ShiftPoint {
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

/// Check if an operation has nested regions.
fn has_nested_regions(ctx: &IrContext, op: OpRef) -> bool {
    // Shift has handler region, but that's not continuation code
    if arena_cont::Shift::matches(ctx, op) {
        return false;
    }
    !ctx.op(op).regions.is_empty()
}

/// Check if an operation has a shift inside any of its nested regions.
fn has_shift_in_nested_region(ctx: &IrContext, op: OpRef) -> bool {
    for &region in &ctx.op(op).regions {
        if region_contains_shift(ctx, region) {
            return true;
        }
    }
    false
}

/// Check if a region contains any shift operation.
fn region_contains_shift(ctx: &IrContext, region: RegionRef) -> bool {
    walk::walk_region::<()>(ctx, region, &mut |op| {
        if arena_cont::Shift::matches(ctx, op) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(walk::WalkAction::Advance)
        }
    })
    .is_break()
}

/// Collect all values used as operands by a slice of operations.
/// Recursively traverses nested regions.
fn collect_used_values(ctx: &IrContext, ops: &[OpRef]) -> HashSet<ValueRef> {
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
    use trunk_ir::Symbol;
    use trunk_ir::arena::context::{BlockData, IrContext, RegionData};
    use trunk_ir::arena::dialect::{arith, cont as arena_cont, func as arena_func};
    use trunk_ir::arena::ops::DialectOp;
    use trunk_ir::arena::types::{Attribute, Location, TypeDataBuilder};
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
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("cont"), Symbol::new("prompt_tag")).build())
    }

    fn ability_ref_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
                .attr("name", Attribute::Symbol(Symbol::new("State")))
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

        let analysis = FunctionAnalysis::analyze(&ctx, region).unwrap();
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
            FunctionAnalysis::analyze(&ctx, region).is_none(),
            "Multi-block functions not supported in Phase 1-2"
        );
    }

    /// Create a region with one shift point:
    /// %0 = arith.const 42
    /// %tag = arith.const 0
    /// %1 = cont.shift(%tag) -> Int
    /// %2 = arith.add %0, %1
    /// func.return %2
    fn create_single_shift_region(ctx: &mut IrContext, loc: Location) -> RegionRef {
        let i32_ty = i32_type(ctx);
        let prompt_tag_ty = prompt_tag_type(ctx);
        let ability_ref_ty = ability_ref_type(ctx);

        let handler_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![],
            parent_op: None,
        });

        let const_op = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(42));
        let const_val = const_op.result(ctx);

        let tag_const = arith::r#const(ctx, loc, prompt_tag_ty, Attribute::IntBits(0));
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

        let add_op = arith::add(ctx, loc, const_val, shift_result, i32_ty);
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
        let region = create_single_shift_region(&mut ctx, loc);
        let analysis = FunctionAnalysis::analyze(&ctx, region).unwrap();

        assert_eq!(analysis.shift_points.len(), 1);
        assert_eq!(analysis.shift_points[0].index, 0);
        assert_eq!(analysis.shift_points[0].total_shifts, 1);
    }

    #[test]
    fn arena_single_shift_live_values() {
        let (mut ctx, loc) = test_ctx();
        let region = create_single_shift_region(&mut ctx, loc);
        let analysis = FunctionAnalysis::analyze(&ctx, region).unwrap();

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
    fn create_dual_shift_region(ctx: &mut IrContext, loc: Location) -> RegionRef {
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

        let tag_const = arith::r#const(ctx, loc, prompt_tag_ty, Attribute::IntBits(0));
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

        let one = arith::r#const(ctx, loc, i32_ty, Attribute::IntBits(1));
        let one_val = one.result(ctx);

        let add_op = arith::add(ctx, loc, n, one_val, i32_ty);
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
        let region = create_dual_shift_region(&mut ctx, loc);
        let analysis = FunctionAnalysis::analyze(&ctx, region).unwrap();

        assert_eq!(analysis.shift_points.len(), 2);
        assert_eq!(analysis.shift_points[0].index, 0);
        assert_eq!(analysis.shift_points[0].total_shifts, 2);
        assert_eq!(analysis.shift_points[1].index, 1);
        assert_eq!(analysis.shift_points[1].total_shifts, 2);
    }

    #[test]
    fn arena_dual_shift_live_values() {
        let (mut ctx, loc) = test_ctx();
        let region = create_dual_shift_region(&mut ctx, loc);
        let analysis = FunctionAnalysis::analyze(&ctx, region).unwrap();

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
        let region = create_dual_shift_region(&mut ctx, loc);
        let analysis = FunctionAnalysis::analyze(&ctx, region).unwrap();

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
