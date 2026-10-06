//! Model-based property tests for operation region, successor, and operand
//! use-chain lists.
//!
//! Each test applies a random sequence of list operations to an `IrContext`
//! and to a plain `Vec` model, and checks after every step that the context
//! reports exactly what the model holds.

use proptest::prelude::*;
use proptest::sample::Index;
use smallvec::SmallVec;

use super::*;
use crate::location::Span;
use crate::rewrite::Module;

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

// ============================================================================
// Operand use chains
// ============================================================================

#[derive(Clone, Debug)]
enum UseAction {
    /// Create an operation in a live block with operands drawn from the live
    /// values and this many results. A container also owns one region with
    /// one single-argument block that later operations can be created in.
    Create {
        block: Index,
        operands: Vec<Index>,
        results: usize,
        container: bool,
    },
    /// Replace one operand of an operation with a live value.
    Set(Index, Index, Index),
    /// Append a live value to an operation's operands.
    Push(Index, Index),
    /// Remove one operand of an operation.
    RemoveOperand(Index, Index),
    /// Replace every use of any value, live or not, with a live value.
    ReplaceAllUses(Index, Index),
    /// Detach and dispose of an operation whose results are unused.
    Remove(Index),
    /// Deep-clone an operation into its own block.
    Clone(Index),
}

fn use_action() -> impl Strategy<Value = UseAction> {
    prop_oneof![
        3 => (
            any::<Index>(),
            prop::collection::vec(any::<Index>(), 0..=3),
            0usize..=2,
            prop::bool::weighted(0.25),
        )
            .prop_map(|(block, operands, results, container)| UseAction::Create {
                block,
                operands,
                results,
                container,
            }),
        2 => (any::<Index>(), any::<Index>(), any::<Index>())
            .prop_map(|(op, index, value)| UseAction::Set(op, index, value)),
        1 => (any::<Index>(), any::<Index>()).prop_map(|(op, value)| UseAction::Push(op, value)),
        1 => (any::<Index>(), any::<Index>())
            .prop_map(|(op, index)| UseAction::RemoveOperand(op, index)),
        2 => (any::<Index>(), any::<Index>())
            .prop_map(|(old, new)| UseAction::ReplaceAllUses(old, new)),
        2 => any::<Index>().prop_map(UseAction::Remove),
        1 => any::<Index>().prop_map(UseAction::Clone),
    ]
}

/// What the model records for one live operation.
#[derive(Clone, Debug)]
struct ModelOp {
    op: OpRef,
    block: BlockRef,
    operands: Vec<ValueRef>,
    results: Vec<ValueRef>,
    /// The single block of a container's region.
    body: Option<BlockRef>,
    /// The argument of `body`.
    body_arg: Option<ValueRef>,
}

#[derive(Default)]
struct UseModel {
    /// Live operations, in creation order.
    ops: Vec<ModelOp>,
    /// Live blocks: the module body and every live container's body.
    blocks: Vec<BlockRef>,
    /// Operations of each live block, in block order.
    block_ops: Vec<(BlockRef, Vec<OpRef>)>,
    /// Values defined by live operations and blocks.
    live_values: Vec<ValueRef>,
    /// Every value ever created, including those of removed operations.
    all_values: Vec<ValueRef>,
}

impl UseModel {
    fn op(&self, op: OpRef) -> &ModelOp {
        self.ops.iter().find(|m| m.op == op).expect("live model op")
    }

    fn block_ops_mut(&mut self, block: BlockRef) -> &mut Vec<OpRef> {
        &mut self
            .block_ops
            .iter_mut()
            .find(|(b, _)| *b == block)
            .expect("live model block")
            .1
    }

    fn add_block(&mut self, block: BlockRef, args: &[ValueRef]) {
        self.blocks.push(block);
        self.block_ops.push((block, Vec::new()));
        self.live_values.extend(args);
        self.all_values.extend(args);
    }

    /// Record `op`, which the context has just appended to `block`.
    fn add_op(&mut self, ctx: &IrContext, op: OpRef, block: BlockRef, operands: Vec<ValueRef>) {
        let results = ctx.op_results(op).to_vec();
        self.live_values.extend(&results);
        self.all_values.extend(&results);
        let body = ctx
            .op_region(op, 0)
            .map(|region| ctx.region(region).blocks[0]);
        let body_arg = body.map(|body| ctx.block_args(body)[0]);
        if let Some(body) = body {
            self.add_block(body, ctx.block_args(body));
        }
        self.block_ops_mut(block).push(op);
        self.ops.push(ModelOp {
            op,
            block,
            operands,
            results,
            body,
            body_arg,
        });
    }

    /// The uses the model expects for `value`, sorted.
    fn expected_uses(&self, value: ValueRef) -> Vec<(OpRef, u32)> {
        let mut uses: Vec<_> = self
            .ops
            .iter()
            .flat_map(|m| {
                m.operands
                    .iter()
                    .enumerate()
                    .filter(move |&(_, &v)| v == value)
                    .map(move |(index, _)| (m.op, index as u32))
            })
            .collect();
        uses.sort_unstable();
        uses
    }

    /// Mirror `IrContext::clone_op` of `src` into `clone`: operands are
    /// remapped through what has been cloned so far, a body's argument is
    /// mapped before its operations are cloned in order, and the results are
    /// mapped after the body.
    fn record_clone(
        &mut self,
        ctx: &IrContext,
        src: OpRef,
        clone: OpRef,
        block: BlockRef,
        mapping: &mut Vec<(ValueRef, ValueRef)>,
    ) {
        let original = self.op(src).clone();
        let lookup = |mapping: &[(ValueRef, ValueRef)], v: ValueRef| {
            mapping
                .iter()
                .find(|(from, _)| *from == v)
                .map_or(v, |&(_, to)| to)
        };
        let operands = original
            .operands
            .iter()
            .map(|&v| lookup(mapping, v))
            .collect();
        self.add_op(ctx, clone, block, operands);
        let cloned = self.op(clone).clone();
        if let (Some(src_body), Some(clone_body)) = (original.body, cloned.body) {
            mapping.extend(
                ctx.block_args(src_body)
                    .iter()
                    .copied()
                    .zip(ctx.block_args(clone_body).iter().copied()),
            );
            let children = self.block_ops_mut(src_body).clone();
            let cloned_children = ctx.block(clone_body).ops.clone();
            assert_eq!(children.len(), cloned_children.len());
            for (child, cloned_child) in children.into_iter().zip(cloned_children) {
                self.record_clone(ctx, child, cloned_child, clone_body, mapping);
            }
        }
        mapping.extend(original.results.iter().copied().zip(cloned.results));
    }

    /// Forget `op` and every operation nested in it.
    fn remove(&mut self, op: OpRef) {
        let model = self.op(op).clone();
        if let Some(body) = model.body {
            for child in self.block_ops_mut(body).clone() {
                self.remove(child);
            }
            self.blocks.retain(|&b| b != body);
            self.block_ops.retain(|(b, _)| *b != body);
            self.live_values.retain(|&v| Some(v) != model.body_arg);
        }
        self.block_ops_mut(model.block).retain(|&o| o != op);
        self.live_values.retain(|v| !model.results.contains(v));
        self.ops.retain(|m| m.op != op);
    }
}

/// The context's operands and use chains must match the model, and the
/// validator's use-chain check must accept the module.
fn check_uses(ctx: &IrContext, module: Module, model: &UseModel) -> Result<(), TestCaseError> {
    for m in &model.ops {
        prop_assert_eq!(ctx.op_operands(m.op), &m.operands[..]);
    }
    for &value in &model.all_values {
        let mut actual: Vec<_> = ctx
            .uses(value)
            .iter()
            .map(|u| (u.user, u.operand_index))
            .collect();
        actual.sort_unstable();
        let expected = model.expected_uses(value);
        prop_assert_eq!(ctx.has_uses(value), !expected.is_empty());
        prop_assert_eq!(actual, expected, "uses of {}", value);
    }
    let result = crate::validation::validate_use_chains(ctx, module);
    prop_assert!(result.is_ok(), "{}", result);
    Ok(())
}

proptest! {
    #[test]
    fn use_chains_match_an_operand_model(
        actions in prop::collection::vec(use_action(), 1..64),
    ) {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let arg = || BlockArgData { ty, attrs: AttributeMap::new() };
        let new_block = |ctx: &mut IrContext, args: usize| {
            ctx.create_block(BlockData {
                location: loc,
                args: (0..args).map(|_| arg()).collect(),
                ops: SmallVec::new(),
                parent_region: None,
            })
        };

        let module_block = new_block(&mut ctx, 2);
        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec::smallvec![module_block],
            parent_op: None,
        });
        let module_op = crate::dialect::core::Module::operands()
            .sym_name("test")
            .regions(module_region)
            .build(&mut ctx, loc)
            .op_ref();
        let module = Module::new(&ctx, module_op).expect("core.module");

        let mut model = UseModel::default();
        model.add_block(module_block, ctx.block_args(module_block));

        for action in actions {
            match action {
                UseAction::Create { block, operands, results, container } => {
                    let block = *block.get(&model.blocks);
                    let operands: Vec<_> = if model.live_values.is_empty() {
                        Vec::new()
                    } else {
                        operands.iter().map(|i| *i.get(&model.live_values)).collect()
                    };
                    let mut builder =
                        OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("op"))
                            .operands(operands.iter().copied())
                            .results((0..results).map(|_| ty));
                    if container {
                        let body = new_block(&mut ctx, 1);
                        let region = ctx.create_region(RegionData {
                            location: loc,
                            blocks: smallvec::smallvec![body],
                            parent_op: None,
                        });
                        builder = builder.region(region);
                    }
                    let data = builder.build(&mut ctx);
                    let op = ctx.create_op(data);
                    ctx.push_op(block, op);
                    model.add_op(&ctx, op, block, operands);
                }
                UseAction::Set(op, index, value)
                    if !model.ops.is_empty() && !model.live_values.is_empty() =>
                {
                    let value = *value.get(&model.live_values);
                    let m = op.get_mut(&mut model.ops);
                    if m.operands.is_empty() {
                        continue;
                    }
                    let index = index.index(m.operands.len());
                    ctx.set_op_operand(m.op, index as u32, value);
                    m.operands[index] = value;
                }
                UseAction::Push(op, value)
                    if !model.ops.is_empty() && !model.live_values.is_empty() =>
                {
                    let value = *value.get(&model.live_values);
                    let m = op.get_mut(&mut model.ops);
                    ctx.push_op_operand(m.op, value);
                    m.operands.push(value);
                }
                UseAction::RemoveOperand(op, index) if !model.ops.is_empty() => {
                    let m = op.get_mut(&mut model.ops);
                    if m.operands.is_empty() {
                        continue;
                    }
                    let index = index.index(m.operands.len());
                    ctx.remove_op_operand(m.op, index as u32);
                    m.operands.remove(index);
                }
                UseAction::ReplaceAllUses(old, new) if !model.live_values.is_empty() => {
                    let old = *old.get(&model.all_values);
                    let new = *new.get(&model.live_values);
                    ctx.replace_all_uses(old, new);
                    for m in &mut model.ops {
                        for operand in &mut m.operands {
                            if *operand == old {
                                *operand = new;
                            }
                        }
                    }
                    prop_assert!(old == new || !ctx.has_uses(old));
                }
                UseAction::Remove(op) if !model.ops.is_empty() => {
                    let m = op.get(&model.ops).clone();
                    // `remove_op` requires the operation's own results to be
                    // unused; results nested in its body may stay in use.
                    if m.results.iter().any(|&r| !model.expected_uses(r).is_empty()) {
                        continue;
                    }
                    ctx.remove_op_from_block(m.block, m.op);
                    ctx.remove_op(m.op);
                    model.remove(m.op);
                }
                UseAction::Clone(op) if !model.ops.is_empty() => {
                    let m = op.get(&model.ops).clone();
                    let clone = ctx.clone_op(m.op, &mut IrMapping::new());
                    ctx.push_op(m.block, clone);
                    model.record_clone(&ctx, m.op, clone, m.block, &mut Vec::new());
                }
                _ => {}
            }

            check_uses(&ctx, module, &model)?;
        }
    }
}
