//! Free helper functions for arena IR mutations.
//!
//! These operate on `&mut IrContext` and provide higher-level compound
//! operations built on top of the low-level context methods.

use smallvec::SmallVec;

use crate::arena::context::{BlockData, IrContext};
use crate::arena::refs::{BlockRef, OpRef, RegionRef};

/// Split a block at `before_op`, moving `before_op` and all subsequent
/// operations into a new block.
///
/// The new block inherits no block arguments (caller can add them via
/// `ctx.add_block_arg`). The new block is inserted into the same region
/// immediately after the original block.
///
/// Returns the newly created block.
///
/// # Panics
///
/// Panics if `before_op` is not found in `block`, or if `block` does not
/// belong to a region.
pub fn split_block(ctx: &mut IrContext, block: BlockRef, before_op: OpRef) -> BlockRef {
    let loc = ctx.block(block).location;

    // Find the split point
    let ops = &ctx.block(block).ops;
    let pos = ops
        .iter()
        .position(|&o| o == before_op)
        .expect("split_block: before_op not found in block");

    // Collect ops to move (from split point to end)
    let tail_ops: SmallVec<[OpRef; 4]> = ops[pos..].into();

    // Truncate the original block's ops
    ctx.block_mut(block).ops.truncate(pos);

    // Create the new block (without ops initially, so we can push them)
    let new_block = ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: SmallVec::new(),
        parent_region: None,
    });

    // Move ops to the new block
    for &op in &tail_ops {
        // Update parent_block from old block to new block
        ctx.op_mut(op).parent_block = Some(new_block);
    }
    ctx.block_mut(new_block).ops = tail_ops;

    // Insert new block into the same region after the original block
    let region = ctx
        .block(block)
        .parent_region
        .expect("split_block: block must belong to a region");
    let region_blocks = &ctx.region(region).blocks;
    let block_pos = region_blocks
        .iter()
        .position(|&b| b == block)
        .expect("split_block: block not found in its parent region");
    ctx.region_mut(region)
        .blocks
        .insert(block_pos + 1, new_block);
    ctx.block_mut(new_block).parent_region = Some(region);

    new_block
}

/// Move all blocks from `src_region` into `dest_region`, inserting them
/// before `insert_before` (or at the end if `insert_before` is `None`).
///
/// Returns the list of moved block refs.
///
/// After this call, `src_region` will have an empty block list.
pub fn inline_region_blocks(
    ctx: &mut IrContext,
    src_region: RegionRef,
    dest_region: RegionRef,
    insert_before: Option<BlockRef>,
) -> Vec<BlockRef> {
    if src_region == dest_region {
        return Vec::new();
    }

    // Take blocks from src
    let src_blocks: SmallVec<[BlockRef; 4]> =
        std::mem::take(&mut ctx.region_mut(src_region).blocks);

    let moved: Vec<BlockRef> = src_blocks.to_vec();

    // Update parent_region for each moved block
    for &b in &src_blocks {
        ctx.block_mut(b).parent_region = Some(dest_region);
    }

    // Find insertion point in dest
    let dest_blocks = &mut ctx.region_mut(dest_region).blocks;
    if let Some(before) = insert_before {
        let pos = dest_blocks
            .iter()
            .position(|&b| b == before)
            .expect("inline_region_blocks: insert_before block not found in dest_region");
        // Insert in order at pos
        for (i, &b) in src_blocks.iter().enumerate() {
            dest_blocks.insert(pos + i, b);
        }
    } else {
        dest_blocks.extend(src_blocks);
    }

    moved
}

/// Erase an operation: detach from its parent block and remove it.
///
/// All result values must have no remaining uses, otherwise this panics.
pub fn erase_op(ctx: &mut IrContext, op: OpRef) {
    ctx.detach_op(op);
    ctx.remove_op(op);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;
    use smallvec::smallvec;
    use std::collections::BTreeMap;

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

    fn make_const(ctx: &mut IrContext, loc: Location, ty: TypeRef, val: u64) -> OpRef {
        let data = OperationDataBuilder::new(loc, Symbol::new("arith"), Symbol::new("const"))
            .result(ty)
            .attr("value", Attribute::IntBits(val))
            .build(ctx);
        ctx.create_op(data)
    }

    #[test]
    fn split_block_basic() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op_a = make_const(&mut ctx, loc, i32_ty, 1);
        let op_b = make_const(&mut ctx, loc, i32_ty, 2);
        let op_c = make_const(&mut ctx, loc, i32_ty, 3);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, op_a);
        ctx.push_op(block, op_b);
        ctx.push_op(block, op_c);

        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        // Split at op_b
        let new_block = split_block(&mut ctx, block, op_b);

        // Original block has only op_a
        assert_eq!(ctx.block(block).ops.as_slice(), &[op_a]);

        // New block has op_b and op_c
        assert_eq!(ctx.block(new_block).ops.as_slice(), &[op_b, op_c]);

        // Parent block links are correct
        assert_eq!(ctx.op(op_a).parent_block, Some(block));
        assert_eq!(ctx.op(op_b).parent_block, Some(new_block));
        assert_eq!(ctx.op(op_c).parent_block, Some(new_block));

        // New block is in the region after the original
        assert_eq!(ctx.region(region).blocks.as_slice(), &[block, new_block]);
        assert_eq!(ctx.block(new_block).parent_region, Some(region));
    }

    #[test]
    fn split_block_at_first_op() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op_a = make_const(&mut ctx, loc, i32_ty, 1);
        let op_b = make_const(&mut ctx, loc, i32_ty, 2);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, op_a);
        ctx.push_op(block, op_b);

        let _region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });

        let new_block = split_block(&mut ctx, block, op_a);

        // Original block is empty
        assert!(ctx.block(block).ops.is_empty());

        // New block has both ops
        assert_eq!(ctx.block(new_block).ops.as_slice(), &[op_a, op_b]);
    }

    #[test]
    fn inline_region_blocks_at_end() {
        let (mut ctx, loc) = test_ctx();

        let block_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let block_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let block_c = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let dest_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block_a],
            parent_op: None,
        });
        let src_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block_b, block_c],
            parent_op: None,
        });

        let moved = inline_region_blocks(&mut ctx, src_region, dest_region, None);

        assert_eq!(moved, vec![block_b, block_c]);
        assert_eq!(
            ctx.region(dest_region).blocks.as_slice(),
            &[block_a, block_b, block_c]
        );
        assert!(ctx.region(src_region).blocks.is_empty());
        assert_eq!(ctx.block(block_b).parent_region, Some(dest_region));
        assert_eq!(ctx.block(block_c).parent_region, Some(dest_region));
    }

    #[test]
    fn inline_region_blocks_before() {
        let (mut ctx, loc) = test_ctx();

        let block_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let block_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let block_c = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let dest_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block_a],
            parent_op: None,
        });
        let src_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block_b, block_c],
            parent_op: None,
        });

        inline_region_blocks(&mut ctx, src_region, dest_region, Some(block_a));

        assert_eq!(
            ctx.region(dest_region).blocks.as_slice(),
            &[block_b, block_c, block_a]
        );
    }

    #[test]
    fn inline_region_blocks_same_region_is_noop() {
        let (mut ctx, loc) = test_ctx();

        let block_a = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let block_b = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block_a, block_b],
            parent_op: None,
        });

        // Inlining a region into itself should be a no-op
        let moved = inline_region_blocks(&mut ctx, region, region, None);
        assert!(moved.is_empty());
        assert_eq!(ctx.region(region).blocks.as_slice(), &[block_a, block_b]);

        // Also with insert_before
        let moved = inline_region_blocks(&mut ctx, region, region, Some(block_b));
        assert!(moved.is_empty());
        assert_eq!(ctx.region(region).blocks.as_slice(), &[block_a, block_b]);
    }

    #[test]
    fn erase_op_removes_from_block() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let op = make_const(&mut ctx, loc, i32_ty, 42);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(block, op);

        assert_eq!(ctx.block(block).ops.len(), 1);

        erase_op(&mut ctx, op);

        assert!(ctx.block(block).ops.is_empty());
    }

    #[test]
    fn add_block_arg_works() {
        let (mut ctx, loc) = test_ctx();
        let i32_ty = i32_type(&mut ctx);

        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        assert_eq!(ctx.block_args(block).len(), 0);

        let v0 = ctx.add_block_arg(
            block,
            BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
            },
        );
        let v1 = ctx.add_block_arg(
            block,
            BlockArgData {
                ty: i32_ty,
                attrs: BTreeMap::new(),
            },
        );

        assert_eq!(ctx.block_args(block).len(), 2);
        assert_eq!(ctx.block_args(block)[0], v0);
        assert_eq!(ctx.block_args(block)[1], v1);
        assert_eq!(ctx.value_def(v0), ValueDef::BlockArg(block, 0));
        assert_eq!(ctx.value_def(v1), ValueDef::BlockArg(block, 1));
        assert_eq!(ctx.value_ty(v0), i32_ty);
    }
}
