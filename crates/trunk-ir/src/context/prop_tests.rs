//! Model-based property tests for operation region and successor lists.
//!
//! Each test applies a random sequence of list operations to an `IrContext`
//! and to a plain `Vec` model, and checks after every step that the context
//! reports exactly what the model holds.

use proptest::prelude::*;
use proptest::sample::Index;
use smallvec::SmallVec;

use super::*;
use crate::location::Span;

fn location(ctx: &mut IrContext) -> Location {
    let path = ctx.intern_path("file:///prop.trb");
    Location::new(path, Span::new(0, 0))
}

fn fresh_region(ctx: &mut IrContext, loc: Location) -> RegionRef {
    ctx.create_region(RegionData {
        location: loc,
        blocks: SmallVec::new(),
        parent_op: None,
    })
}

fn create_op(
    ctx: &mut IrContext,
    loc: Location,
    regions: &[RegionRef],
    successors: &[BlockRef],
) -> OpRef {
    let mut builder = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("op"));
    for &region in regions {
        builder = builder.region(region);
    }
    for &block in successors {
        builder = builder.successor(block);
    }
    let data = builder.build(ctx);
    ctx.create_op(data)
}

// ============================================================================
// Regions
// ============================================================================

#[derive(Clone, Debug)]
enum RegionAction {
    /// Create an operation with this many fresh regions.
    Create(usize),
    /// Append a fresh region to an operation.
    PushFresh(Index),
    /// Append a detached region to an operation.
    PushDetached(Index, Index),
    /// Detach any region, attached or not.
    Detach(Index),
    /// Detach every region of an operation.
    Clear(Index),
    /// Deep-clone an operation.
    Clone(Index),
}

fn region_action() -> impl Strategy<Value = RegionAction> {
    prop_oneof![
        (0usize..=4).prop_map(RegionAction::Create),
        any::<Index>().prop_map(RegionAction::PushFresh),
        (any::<Index>(), any::<Index>()).prop_map(|(r, o)| RegionAction::PushDetached(r, o)),
        any::<Index>().prop_map(RegionAction::Detach),
        any::<Index>().prop_map(RegionAction::Clear),
        any::<Index>().prop_map(RegionAction::Clone),
    ]
}

/// The context's view of `op`'s regions must match `expected` through every
/// accessor, and each listed region must name `op` as its parent.
fn check_regions(ctx: &IrContext, op: OpRef, expected: &[RegionRef]) -> Result<(), TestCaseError> {
    let regions: RegionList = ctx.op_regions(op).collect();
    prop_assert_eq!(&regions[..], expected);
    prop_assert_eq!(ctx.op_region_count(op), expected.len());
    prop_assert_eq!(ctx.op_has_regions(op), !expected.is_empty());
    for (index, &region) in expected.iter().enumerate() {
        prop_assert_eq!(ctx.op_region(op, index), Some(region));
        prop_assert_eq!(ctx.region(region).parent_op, Some(op));
    }
    prop_assert_eq!(ctx.op_region(op, expected.len()), None);
    Ok(())
}

proptest! {
    #[test]
    fn region_lists_match_a_vec_model(
        actions in prop::collection::vec(region_action(), 1..48),
    ) {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        // (operation, its regions in order)
        let mut model: Vec<(OpRef, Vec<RegionRef>)> = Vec::new();
        let mut all_regions: Vec<RegionRef> = Vec::new();

        for action in actions {
            match action {
                RegionAction::Create(count) => {
                    let regions: Vec<_> = (0..count).map(|_| fresh_region(&mut ctx, loc)).collect();
                    let op = create_op(&mut ctx, loc, &regions, &[]);
                    all_regions.extend(&regions);
                    model.push((op, regions));
                }
                RegionAction::PushFresh(op) if !model.is_empty() => {
                    let (op, regions) = op.get_mut(&mut model);
                    let region = fresh_region(&mut ctx, loc);
                    ctx.push_op_region(*op, region);
                    regions.push(region);
                    all_regions.push(region);
                }
                RegionAction::PushDetached(region, op) if !model.is_empty() => {
                    let detached: Vec<_> = all_regions
                        .iter()
                        .copied()
                        .filter(|&r| ctx.region(r).parent_op.is_none())
                        .collect();
                    if detached.is_empty() {
                        continue;
                    }
                    let region = *region.get(&detached);
                    let (op, regions) = op.get_mut(&mut model);
                    ctx.push_op_region(*op, region);
                    regions.push(region);
                }
                RegionAction::Detach(region) if !all_regions.is_empty() => {
                    let region = *region.get(&all_regions);
                    ctx.detach_region(region);
                    for (_, regions) in &mut model {
                        regions.retain(|&r| r != region);
                    }
                    prop_assert_eq!(ctx.region(region).parent_op, None);
                }
                RegionAction::Clear(op) if !model.is_empty() => {
                    let (op, regions) = op.get_mut(&mut model);
                    ctx.clear_op_regions(*op);
                    for region in regions.drain(..) {
                        prop_assert_eq!(ctx.region(region).parent_op, None);
                    }
                }
                RegionAction::Clone(op) if !model.is_empty() => {
                    let (op, regions) = op.get(&model).clone();
                    let clone = ctx.clone_op(op, &mut IrMapping::new());
                    let cloned: Vec<_> = ctx.op_regions(clone).collect();
                    prop_assert_eq!(cloned.len(), regions.len());
                    for region in &cloned {
                        prop_assert!(!all_regions.contains(region), "clone reused {region}");
                    }
                    all_regions.extend(&cloned);
                    model.push((clone, cloned));
                }
                _ => {}
            }

            for (op, regions) in &model {
                check_regions(&ctx, *op, regions)?;
            }
            for &region in &all_regions {
                let owner = model
                    .iter()
                    .find(|(_, regions)| regions.contains(&region))
                    .map(|(op, _)| *op);
                prop_assert_eq!(ctx.region(region).parent_op, owner);
            }
        }
    }
}

// ============================================================================
// Successors
// ============================================================================

/// Number of blocks the successor tests draw from; repeats are allowed.
const BLOCKS: usize = 4;

#[derive(Clone, Debug)]
enum SuccessorAction {
    /// Create an operation with these successors (indices into the blocks).
    Create(Vec<usize>),
    /// Replace one successor of an operation.
    Set(Index, Index, usize),
    /// Keep a prefix of an operation's successors.
    Truncate(Index, Index),
    /// Deep-clone an operation.
    Clone(Index),
}

fn successor_action() -> impl Strategy<Value = SuccessorAction> {
    prop_oneof![
        prop::collection::vec(0..BLOCKS, 0..=6).prop_map(SuccessorAction::Create),
        (any::<Index>(), any::<Index>(), 0..BLOCKS)
            .prop_map(|(op, index, block)| SuccessorAction::Set(op, index, block)),
        (any::<Index>(), any::<Index>()).prop_map(|(op, len)| SuccessorAction::Truncate(op, len)),
        any::<Index>().prop_map(SuccessorAction::Clone),
    ]
}

/// The context's view of `op`'s successors must match `expected` through
/// every accessor and in both directions.
fn check_successors(
    ctx: &IrContext,
    op: OpRef,
    expected: &[BlockRef],
) -> Result<(), TestCaseError> {
    let successors: BlockList = ctx.op_successors(op).collect();
    prop_assert_eq!(&successors[..], expected);
    prop_assert_eq!(ctx.op_successors(op).len(), expected.len());
    prop_assert_eq!(ctx.op_successor_count(op), expected.len());
    prop_assert_eq!(ctx.op_has_successors(op), !expected.is_empty());
    prop_assert!(
        ctx.op_successors(op)
            .rev()
            .eq(expected.iter().rev().copied())
    );
    for (index, &block) in expected.iter().enumerate() {
        prop_assert_eq!(ctx.op_successor(op, index), Some(block));
    }
    prop_assert_eq!(ctx.op_successor(op, expected.len()), None);
    Ok(())
}

proptest! {
    #[test]
    fn successor_lists_match_a_vec_model(
        actions in prop::collection::vec(successor_action(), 1..48),
    ) {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let blocks: Vec<BlockRef> = (0..BLOCKS)
            .map(|_| {
                ctx.create_block(BlockData {
                    location: loc,
                    args: vec![],
                    ops: SmallVec::new(),
                    parent_region: None,
                })
            })
            .collect();
        let mut model: Vec<(OpRef, Vec<BlockRef>)> = Vec::new();

        for action in actions {
            match action {
                SuccessorAction::Create(indices) => {
                    let successors: Vec<_> = indices.iter().map(|&i| blocks[i]).collect();
                    let op = create_op(&mut ctx, loc, &[], &successors);
                    model.push((op, successors));
                }
                SuccessorAction::Set(op, index, block) if !model.is_empty() => {
                    let (op, successors) = op.get_mut(&mut model);
                    if successors.is_empty() {
                        continue;
                    }
                    let index = index.index(successors.len());
                    ctx.set_op_successor(*op, index, blocks[block]);
                    successors[index] = blocks[block];
                }
                SuccessorAction::Truncate(op, len) if !model.is_empty() => {
                    let (op, successors) = op.get_mut(&mut model);
                    let len = len.index(successors.len() + 1);
                    ctx.truncate_op_successors(*op, len);
                    successors.truncate(len);
                }
                SuccessorAction::Clone(op) if !model.is_empty() => {
                    let (op, successors) = op.get(&model).clone();
                    let clone = ctx.clone_op(op, &mut IrMapping::new());
                    model.push((clone, successors));
                }
                _ => {}
            }

            for (op, successors) in &model {
                check_successors(&ctx, *op, successors)?;
            }
        }
    }
}
