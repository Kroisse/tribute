//! Recursive operation traversal utilities for arena IR.
//!
//! Provides `walk_*` functions for traversing nested operations in the arena,
//! analogous to the Salsa-based `walk.rs` but operating on `IrContext` + refs.

use std::ops::ControlFlow;

use super::context::IrContext;
use super::ops::ArenaDialectOp;
use super::refs::{BlockRef, OpRef, RegionRef};

/// Controls whether to descend into children during a walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkAction {
    /// Continue walking and descend into nested regions.
    Advance,
    /// Skip the nested regions of the current operation.
    Skip,
}

/// Walk all operations in a region recursively.
pub fn walk_region<B>(
    ctx: &IrContext,
    region: RegionRef,
    f: &mut dyn FnMut(OpRef) -> ControlFlow<B, WalkAction>,
) -> ControlFlow<B, ()> {
    for &block in &ctx.region(region).blocks {
        walk_block(ctx, block, f)?;
    }
    ControlFlow::Continue(())
}

/// Walk all operations in a block recursively.
pub fn walk_block<B>(
    ctx: &IrContext,
    block: BlockRef,
    f: &mut dyn FnMut(OpRef) -> ControlFlow<B, WalkAction>,
) -> ControlFlow<B, ()> {
    for &op in &ctx.block(block).ops {
        walk_op(ctx, op, f)?;
    }
    ControlFlow::Continue(())
}

/// Walk an operation and its nested regions recursively.
pub fn walk_op<B>(
    ctx: &IrContext,
    op: OpRef,
    f: &mut dyn FnMut(OpRef) -> ControlFlow<B, WalkAction>,
) -> ControlFlow<B, ()> {
    match f(op) {
        ControlFlow::Break(b) => return ControlFlow::Break(b),
        ControlFlow::Continue(WalkAction::Skip) => return ControlFlow::Continue(()),
        ControlFlow::Continue(WalkAction::Advance) => {}
    }
    for &region in &ctx.op(op).regions {
        walk_region(ctx, region, f)?;
    }
    ControlFlow::Continue(())
}

/// Walk operations of a specific dialect type in a region.
pub fn walk_typed<T, B>(
    ctx: &IrContext,
    region: RegionRef,
    f: &mut dyn FnMut(T) -> ControlFlow<B, WalkAction>,
) -> ControlFlow<B, ()>
where
    T: ArenaDialectOp,
{
    walk_region(ctx, region, &mut |op| {
        if let Ok(typed) = T::from_op(ctx, op) {
            f(typed)
        } else {
            ControlFlow::Continue(WalkAction::Advance)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;
    use smallvec::smallvec;

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

    #[test]
    fn walk_region_finds_all_ops() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op1_data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .attr("value", Attribute::IntBits(1))
            .build(&mut ctx);
        let op1 = ctx.create_op(op1_data);
        let op2_data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .attr("value", Attribute::IntBits(2))
            .build(&mut ctx);
        let op2 = ctx.create_op(op2_data);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, op1);
        ctx.push_op(block, op2);

        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        let mut count = 0;
        let _ = walk_region::<()>(&ctx, region, &mut |_op| {
            count += 1;
            ControlFlow::Continue(WalkAction::Advance)
        });
        assert_eq!(count, 2);
    }

    #[test]
    fn walk_with_early_exit() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op1_data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op1 = ctx.create_op(op1_data);
        let op2_data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(i32_ty)
            .build(&mut ctx);
        let op2 = ctx.create_op(op2_data);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, op1);
        ctx.push_op(block, op2);

        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        let mut visited = 0;
        let result = walk_region::<()>(&ctx, region, &mut |_op| {
            visited += 1;
            ControlFlow::Break(())
        });

        assert!(result.is_break());
        assert_eq!(visited, 1);
    }

    #[test]
    fn walk_skip_nested_regions() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        // Inner const
        let inner_op_data =
            OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
                .result(i32_ty)
                .build(&mut ctx);
        let inner_op = ctx.create_op(inner_op_data);
        let inner_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(inner_block, inner_op);
        let inner_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![inner_block],
            parent_op: None,
        });

        // Outer func op containing inner region
        let func_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).build());
        let func_op_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .result(func_ty)
            .region(inner_region)
            .attr("sym_name", Attribute::Symbol(Symbol::new("test_fn")))
            .attr("type", Attribute::Type(func_ty))
            .build(&mut ctx);
        let func_op = ctx.create_op(func_op_data);
        let outer_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(outer_block, func_op);
        let outer_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![outer_block],
            parent_op: None,
        });

        // Walk with Skip on func.func should not visit inner_op
        let mut found_const = false;
        let _ = walk_region::<()>(&ctx, outer_region, &mut |op| {
            let data = ctx.op(op);
            if data.dialect == Symbol::new("func") && data.name == Symbol::new("func") {
                ControlFlow::<(), _>::Continue(WalkAction::Skip)
            } else if data.dialect == Symbol::new("arith") {
                found_const = true;
                ControlFlow::Continue(WalkAction::Advance)
            } else {
                ControlFlow::Continue(WalkAction::Advance)
            }
        });

        assert!(!found_const);
    }
}
