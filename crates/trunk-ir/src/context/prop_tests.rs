//! Model-based property tests for operation region, successor, and operand
//! use-chain lists.
//!
//! Each test is a state machine: a reference model built from plain `Vec`s
//! names entities by creation-order ids and generates the next action from
//! its current state, and the system under test applies the same action to
//! an `IrContext`. After every step the context must report exactly what the
//! model holds.

use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::Union;
use proptest_state_machine::{ReferenceStateMachine, StateMachineTest, prop_state_machine};
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

/// Operations and regions are named by creation order.
#[derive(Clone, Debug, Default)]
struct RegionModel {
    /// The regions of each operation, in order.
    ops: Vec<Vec<usize>>,
    /// The number of regions created so far.
    regions: usize,
}

impl RegionModel {
    /// The operation whose list holds `region`, if any.
    fn owner(&self, region: usize) -> Option<usize> {
        self.ops
            .iter()
            .position(|regions| regions.contains(&region))
    }

    fn fresh_regions(&mut self, count: usize) -> Vec<usize> {
        let regions = (self.regions..self.regions + count).collect();
        self.regions += count;
        regions
    }
}

#[derive(Clone, Debug)]
enum RegionAction {
    /// Create an operation with this many fresh regions.
    Create(usize),
    /// Append a fresh region to an operation.
    PushFresh(usize),
    /// Append a detached region to an operation.
    PushDetached { region: usize, op: usize },
    /// Detach any region, attached or not.
    Detach(usize),
    /// Detach every region of an operation.
    Clear(usize),
    /// Deep-clone an operation.
    Clone(usize),
}

struct RegionMachine;

impl ReferenceStateMachine for RegionMachine {
    type State = RegionModel;
    type Transition = RegionAction;

    fn init_state() -> BoxedStrategy<RegionModel> {
        Just(RegionModel::default()).boxed()
    }

    fn transitions(state: &RegionModel) -> BoxedStrategy<RegionAction> {
        let mut arms = vec![(0usize..=4).prop_map(RegionAction::Create).boxed()];
        if !state.ops.is_empty() {
            let ops = 0..state.ops.len();
            arms.push(ops.clone().prop_map(RegionAction::PushFresh).boxed());
            let detached: Vec<_> = (0..state.regions)
                .filter(|&region| state.owner(region).is_none())
                .collect();
            if !detached.is_empty() {
                arms.push(
                    (select(detached), ops.clone())
                        .prop_map(|(region, op)| RegionAction::PushDetached { region, op })
                        .boxed(),
                );
            }
            arms.push(ops.clone().prop_map(RegionAction::Clear).boxed());
            arms.push(ops.prop_map(RegionAction::Clone).boxed());
        }
        if state.regions > 0 {
            arms.push((0..state.regions).prop_map(RegionAction::Detach).boxed());
        }
        Union::new(arms).boxed()
    }

    fn preconditions(state: &RegionModel, action: &RegionAction) -> bool {
        match *action {
            RegionAction::Create(_) => true,
            RegionAction::PushFresh(op) | RegionAction::Clear(op) | RegionAction::Clone(op) => {
                op < state.ops.len()
            }
            RegionAction::PushDetached { region, op } => {
                op < state.ops.len() && region < state.regions && state.owner(region).is_none()
            }
            RegionAction::Detach(region) => region < state.regions,
        }
    }

    fn apply(mut state: RegionModel, action: &RegionAction) -> RegionModel {
        match *action {
            RegionAction::Create(count) => {
                let regions = state.fresh_regions(count);
                state.ops.push(regions);
            }
            RegionAction::PushFresh(op) => {
                let region = state.fresh_regions(1)[0];
                state.ops[op].push(region);
            }
            RegionAction::PushDetached { region, op } => state.ops[op].push(region),
            RegionAction::Detach(region) => {
                for regions in &mut state.ops {
                    regions.retain(|&r| r != region);
                }
            }
            RegionAction::Clear(op) => state.ops[op].clear(),
            RegionAction::Clone(op) => {
                let regions = state.fresh_regions(state.ops[op].len());
                state.ops.push(regions);
            }
        }
        state
    }
}

/// The context, and the references the model's ids stand for.
struct RegionSut {
    ctx: IrContext,
    loc: Location,
    ops: Vec<OpRef>,
    regions: Vec<RegionRef>,
}

impl RegionSut {
    fn fresh_region(&mut self) -> RegionRef {
        let region = fresh_region(&mut self.ctx, self.loc);
        self.regions.push(region);
        region
    }
}

/// The context's view of `op`'s regions must match `expected` through every
/// accessor, and each listed region must name `op` as its parent.
fn check_regions(ctx: &IrContext, op: OpRef, expected: &[RegionRef]) {
    let regions: RegionList = ctx.op_regions(op).collect();
    assert_eq!(&regions[..], expected);
    assert_eq!(ctx.op_region_count(op), expected.len());
    assert_eq!(ctx.op_has_regions(op), !expected.is_empty());
    for (index, &region) in expected.iter().enumerate() {
        assert_eq!(ctx.op_region(op, index), Some(region));
        assert_eq!(ctx.region(region).parent_op, Some(op));
    }
    assert_eq!(ctx.op_region(op, expected.len()), None);
}

struct RegionTest;

impl StateMachineTest for RegionTest {
    type SystemUnderTest = RegionSut;
    type Reference = RegionMachine;

    fn init_test(_: &RegionModel) -> RegionSut {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        RegionSut {
            ctx,
            loc,
            ops: Vec::new(),
            regions: Vec::new(),
        }
    }

    fn apply(mut sut: RegionSut, model: &RegionModel, action: RegionAction) -> RegionSut {
        match action {
            RegionAction::Create(count) => {
                let regions: Vec<_> = (0..count).map(|_| sut.fresh_region()).collect();
                let op = create_op(&mut sut.ctx, sut.loc, &regions, &[]);
                sut.ops.push(op);
            }
            RegionAction::PushFresh(op) => {
                let region = sut.fresh_region();
                sut.ctx.push_op_region(sut.ops[op], region);
            }
            RegionAction::PushDetached { region, op } => {
                sut.ctx.push_op_region(sut.ops[op], sut.regions[region]);
            }
            RegionAction::Detach(region) => {
                let region = sut.regions[region];
                sut.ctx.detach_region(region);
                assert_eq!(sut.ctx.region(region).parent_op, None);
            }
            RegionAction::Clear(op) => {
                let op = sut.ops[op];
                let regions: RegionList = sut.ctx.op_regions(op).collect();
                sut.ctx.clear_op_regions(op);
                for region in regions {
                    assert_eq!(sut.ctx.region(region).parent_op, None);
                }
            }
            RegionAction::Clone(op) => {
                let clone = sut.ctx.clone_op(sut.ops[op], &mut IrMapping::new());
                let cloned: Vec<_> = sut.ctx.op_regions(clone).collect();
                let expected = model.ops.last().expect("the clone's model entry");
                assert_eq!(cloned.len(), expected.len());
                for region in &cloned {
                    assert!(!sut.regions.contains(region), "clone reused {region}");
                }
                sut.regions.extend(cloned);
                sut.ops.push(clone);
            }
        }
        sut
    }

    fn check_invariants(sut: &RegionSut, model: &RegionModel) {
        assert_eq!(sut.ops.len(), model.ops.len());
        assert_eq!(sut.regions.len(), model.regions);
        for (&op, regions) in sut.ops.iter().zip(&model.ops) {
            let expected: Vec<_> = regions.iter().map(|&r| sut.regions[r]).collect();
            check_regions(&sut.ctx, op, &expected);
        }
        for (index, &region) in sut.regions.iter().enumerate() {
            let owner = model.owner(index).map(|op| sut.ops[op]);
            assert_eq!(sut.ctx.region(region).parent_op, owner);
        }
    }
}

prop_state_machine! {
    #[test]
    fn region_lists_match_a_vec_model(sequential 1..48 => RegionTest);
}

// ============================================================================
// Successors
// ============================================================================

/// Number of blocks the successor tests draw from; repeats are allowed.
const BLOCKS: usize = 4;

/// The successors of each operation, as indices into the blocks; operations
/// are named by creation order.
type SuccessorModel = Vec<Vec<usize>>;

#[derive(Clone, Debug)]
enum SuccessorAction {
    /// Create an operation with these successors.
    Create(Vec<usize>),
    /// Replace one successor of an operation.
    Set {
        op: usize,
        index: usize,
        block: usize,
    },
    /// Keep a prefix of an operation's successors.
    Truncate { op: usize, len: usize },
    /// Deep-clone an operation.
    Clone(usize),
}

struct SuccessorMachine;

impl ReferenceStateMachine for SuccessorMachine {
    type State = SuccessorModel;
    type Transition = SuccessorAction;

    fn init_state() -> BoxedStrategy<SuccessorModel> {
        Just(Vec::new()).boxed()
    }

    fn transitions(state: &SuccessorModel) -> BoxedStrategy<SuccessorAction> {
        let mut arms = vec![
            prop::collection::vec(0..BLOCKS, 0..=6)
                .prop_map(SuccessorAction::Create)
                .boxed(),
        ];
        if !state.is_empty() {
            let lens: Vec<usize> = state.iter().map(Vec::len).collect();
            let non_empty: Vec<_> = (0..state.len()).filter(|&op| lens[op] > 0).collect();
            if !non_empty.is_empty() {
                let lens = lens.clone();
                arms.push(
                    select(non_empty)
                        .prop_flat_map(move |op| (Just(op), 0..lens[op], 0..BLOCKS))
                        .prop_map(|(op, index, block)| SuccessorAction::Set { op, index, block })
                        .boxed(),
                );
            }
            arms.push(
                (0..state.len())
                    .prop_flat_map(move |op| (Just(op), 0..=lens[op]))
                    .prop_map(|(op, len)| SuccessorAction::Truncate { op, len })
                    .boxed(),
            );
            arms.push((0..state.len()).prop_map(SuccessorAction::Clone).boxed());
        }
        Union::new(arms).boxed()
    }

    fn preconditions(state: &SuccessorModel, action: &SuccessorAction) -> bool {
        match *action {
            SuccessorAction::Create(_) => true,
            SuccessorAction::Set { op, index, .. } => state
                .get(op)
                .is_some_and(|successors| index < successors.len()),
            SuccessorAction::Truncate { op, len } => state
                .get(op)
                .is_some_and(|successors| len <= successors.len()),
            SuccessorAction::Clone(op) => op < state.len(),
        }
    }

    fn apply(mut state: SuccessorModel, action: &SuccessorAction) -> SuccessorModel {
        match *action {
            SuccessorAction::Create(ref successors) => state.push(successors.clone()),
            SuccessorAction::Set { op, index, block } => state[op][index] = block,
            SuccessorAction::Truncate { op, len } => state[op].truncate(len),
            SuccessorAction::Clone(op) => state.push(state[op].clone()),
        }
        state
    }
}

struct SuccessorSut {
    ctx: IrContext,
    loc: Location,
    blocks: Vec<BlockRef>,
    ops: Vec<OpRef>,
}

/// The context's view of `op`'s successors must match `expected` through
/// every accessor and in both directions.
fn check_successors(ctx: &IrContext, op: OpRef, expected: &[BlockRef]) {
    let successors: BlockList = ctx.op_successors(op).collect();
    assert_eq!(&successors[..], expected);
    assert_eq!(ctx.op_successors(op).len(), expected.len());
    assert_eq!(ctx.op_successor_count(op), expected.len());
    assert_eq!(ctx.op_has_successors(op), !expected.is_empty());
    assert!(
        ctx.op_successors(op)
            .rev()
            .eq(expected.iter().rev().copied())
    );
    for (index, &block) in expected.iter().enumerate() {
        assert_eq!(ctx.op_successor(op, index), Some(block));
    }
    assert_eq!(ctx.op_successor(op, expected.len()), None);
}

struct SuccessorTest;

impl StateMachineTest for SuccessorTest {
    type SystemUnderTest = SuccessorSut;
    type Reference = SuccessorMachine;

    fn init_test(_: &SuccessorModel) -> SuccessorSut {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let blocks = (0..BLOCKS)
            .map(|_| {
                ctx.create_block(BlockData {
                    location: loc,
                    args: vec![],
                    ops: SmallVec::new(),
                    parent_region: None,
                })
            })
            .collect();
        SuccessorSut {
            ctx,
            loc,
            blocks,
            ops: Vec::new(),
        }
    }

    fn apply(mut sut: SuccessorSut, _: &SuccessorModel, action: SuccessorAction) -> SuccessorSut {
        match action {
            SuccessorAction::Create(indices) => {
                let successors: Vec<_> = indices.iter().map(|&i| sut.blocks[i]).collect();
                let op = create_op(&mut sut.ctx, sut.loc, &[], &successors);
                sut.ops.push(op);
            }
            SuccessorAction::Set { op, index, block } => {
                sut.ctx
                    .set_op_successor(sut.ops[op], index, sut.blocks[block]);
            }
            SuccessorAction::Truncate { op, len } => {
                sut.ctx.truncate_op_successors(sut.ops[op], len);
            }
            SuccessorAction::Clone(op) => {
                let clone = sut.ctx.clone_op(sut.ops[op], &mut IrMapping::new());
                sut.ops.push(clone);
            }
        }
        sut
    }

    fn check_invariants(sut: &SuccessorSut, model: &SuccessorModel) {
        assert_eq!(sut.ops.len(), model.len());
        for (&op, successors) in sut.ops.iter().zip(model) {
            let expected: Vec<_> = successors.iter().map(|&b| sut.blocks[b]).collect();
            check_successors(&sut.ctx, op, &expected);
        }
    }
}

prop_state_machine! {
    #[test]
    fn successor_lists_match_a_vec_model(sequential 1..48 => SuccessorTest);
}

// ============================================================================
// Operand use chains
// ============================================================================

/// What the model records for one live operation. Operations, blocks, and
/// values are named by creation order, including those of removed
/// operations.
#[derive(Clone, Debug)]
struct ModelOp {
    op: usize,
    block: usize,
    operands: Vec<usize>,
    results: Vec<usize>,
    /// The single block of a container's region.
    body: Option<usize>,
    /// The argument of `body`.
    body_arg: Option<usize>,
}

#[derive(Clone, Debug)]
struct UseModel {
    /// Live operations, in creation order.
    ops: Vec<ModelOp>,
    /// Live blocks: the module body and every live container's body.
    blocks: Vec<usize>,
    /// Operations of each live block, in block order.
    block_ops: Vec<(usize, Vec<usize>)>,
    /// Values defined by live operations and blocks.
    live_values: Vec<usize>,
    /// Number of operations ever created.
    op_count: usize,
    /// Number of blocks ever created.
    block_count: usize,
    /// Number of values ever created.
    value_count: usize,
}

impl UseModel {
    /// The module body, block 0, with two arguments.
    fn new() -> Self {
        let mut model = UseModel {
            ops: Vec::new(),
            blocks: Vec::new(),
            block_ops: Vec::new(),
            live_values: Vec::new(),
            op_count: 0,
            block_count: 0,
            value_count: 0,
        };
        model.add_block(2);
        model
    }

    fn op(&self, op: usize) -> Option<&ModelOp> {
        self.ops.iter().find(|m| m.op == op)
    }

    fn op_mut(&mut self, op: usize) -> &mut ModelOp {
        self.ops
            .iter_mut()
            .find(|m| m.op == op)
            .expect("live model op")
    }

    fn block_ops_mut(&mut self, block: usize) -> &mut Vec<usize> {
        &mut self
            .block_ops
            .iter_mut()
            .find(|(b, _)| *b == block)
            .expect("live model block")
            .1
    }

    fn fresh_value(&mut self) -> usize {
        self.live_values.push(self.value_count);
        self.value_count += 1;
        self.value_count - 1
    }

    /// Add a live block with `args` arguments; returns the block and its
    /// arguments.
    fn add_block(&mut self, args: usize) -> (usize, Vec<usize>) {
        let block = self.block_count;
        self.block_count += 1;
        self.blocks.push(block);
        self.block_ops.push((block, Vec::new()));
        let args = (0..args).map(|_| self.fresh_value()).collect();
        (block, args)
    }

    /// Append an operation to `block`. Its results are created first, then
    /// a container's body block and its argument.
    fn add_op(
        &mut self,
        block: usize,
        operands: Vec<usize>,
        results: usize,
        container: bool,
    ) -> usize {
        let op = self.op_count;
        self.op_count += 1;
        let results = (0..results).map(|_| self.fresh_value()).collect();
        let (body, body_arg) = if container {
            let (body, args) = self.add_block(1);
            (Some(body), Some(args[0]))
        } else {
            (None, None)
        };
        self.block_ops_mut(block).push(op);
        self.ops.push(ModelOp {
            op,
            block,
            operands,
            results,
            body,
            body_arg,
        });
        op
    }

    /// The uses the model expects for `value`, sorted.
    fn expected_uses(&self, value: usize) -> Vec<(usize, u32)> {
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

    /// Whether `op` is live and its own results are unused; results nested in
    /// its body may stay in use.
    fn removable(&self, op: usize) -> bool {
        self.op(op)
            .is_some_and(|m| m.results.iter().all(|&r| self.expected_uses(r).is_empty()))
    }

    /// Mirror `IrContext::clone_op` of `src` into `block`: operands are
    /// remapped through what has been cloned so far, a body's argument is
    /// mapped before its operations are cloned in order, and the results are
    /// mapped after the body.
    fn record_clone(&mut self, src: usize, block: usize, mapping: &mut Vec<(usize, usize)>) {
        let original = self.op(src).expect("live model op").clone();
        let lookup = |mapping: &[(usize, usize)], v: usize| {
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
        let clone = self.add_op(
            block,
            operands,
            original.results.len(),
            original.body.is_some(),
        );
        let cloned = self.op(clone).expect("cloned model op").clone();
        if let (Some(src_body), Some(clone_body)) = (original.body, cloned.body) {
            mapping.push((
                original.body_arg.expect("body argument"),
                cloned.body_arg.expect("body argument"),
            ));
            for child in self.block_ops_mut(src_body).clone() {
                self.record_clone(child, clone_body, mapping);
            }
        }
        mapping.extend(original.results.iter().copied().zip(cloned.results));
    }

    /// Forget `op` and every operation nested in it.
    fn remove(&mut self, op: usize) {
        let model = self.op(op).expect("live model op").clone();
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

#[derive(Clone, Debug)]
enum UseAction {
    /// Create an operation in a live block with operands drawn from the live
    /// values and this many results. A container also owns one region with
    /// one single-argument block that later operations can be created in.
    Create {
        block: usize,
        operands: Vec<usize>,
        results: usize,
        container: bool,
    },
    /// Replace one operand of an operation with a live value.
    Set {
        op: usize,
        index: usize,
        value: usize,
    },
    /// Append a live value to an operation's operands.
    Push { op: usize, value: usize },
    /// Remove one operand of an operation.
    RemoveOperand { op: usize, index: usize },
    /// Replace every use of any value, live or not, with a live value.
    ReplaceAllUses { old: usize, new: usize },
    /// Detach and dispose of an operation whose results are unused.
    Remove(usize),
    /// Deep-clone an operation into its own block.
    Clone(usize),
}

/// A live operation with operands, and an index into them.
fn operand_slot(model: &UseModel) -> Option<BoxedStrategy<(usize, usize)>> {
    let with_operands: Vec<_> = model
        .ops
        .iter()
        .filter(|m| !m.operands.is_empty())
        .map(|m| (m.op, m.operands.len()))
        .collect();
    (!with_operands.is_empty()).then(|| {
        select(with_operands)
            .prop_flat_map(|(op, len)| (Just(op), 0..len))
            .boxed()
    })
}

struct UseMachine;

impl ReferenceStateMachine for UseMachine {
    type State = UseModel;
    type Transition = UseAction;

    fn init_state() -> BoxedStrategy<UseModel> {
        Just(UseModel::new()).boxed()
    }

    fn transitions(state: &UseModel) -> BoxedStrategy<UseAction> {
        let live_values = state.live_values.clone();
        let operands = if live_values.is_empty() {
            Just(Vec::new()).boxed()
        } else {
            prop::collection::vec(select(live_values.clone()), 0..=3).boxed()
        };
        let mut arms = vec![(
            3,
            (
                select(state.blocks.clone()),
                operands,
                0usize..=2,
                prop::bool::weighted(0.25),
            )
                .prop_map(|(block, operands, results, container)| UseAction::Create {
                    block,
                    operands,
                    results,
                    container,
                })
                .boxed(),
        )];
        let ops: Vec<_> = state.ops.iter().map(|m| m.op).collect();
        if !live_values.is_empty() {
            let value = || select(live_values.clone());
            if let Some(slot) = operand_slot(state) {
                arms.push((
                    2,
                    (slot, value())
                        .prop_map(|((op, index), value)| UseAction::Set { op, index, value })
                        .boxed(),
                ));
            }
            if !ops.is_empty() {
                arms.push((
                    1,
                    (select(ops.clone()), value())
                        .prop_map(|(op, value)| UseAction::Push { op, value })
                        .boxed(),
                ));
            }
            arms.push((
                2,
                (0..state.value_count, value())
                    .prop_map(|(old, new)| UseAction::ReplaceAllUses { old, new })
                    .boxed(),
            ));
        }
        if let Some(slot) = operand_slot(state) {
            arms.push((
                1,
                slot.prop_map(|(op, index)| UseAction::RemoveOperand { op, index })
                    .boxed(),
            ));
        }
        let removable: Vec<_> = ops
            .iter()
            .copied()
            .filter(|&op| state.removable(op))
            .collect();
        if !removable.is_empty() {
            arms.push((2, select(removable).prop_map(UseAction::Remove).boxed()));
        }
        if !ops.is_empty() {
            arms.push((1, select(ops).prop_map(UseAction::Clone).boxed()));
        }
        Union::new_weighted(arms).boxed()
    }

    fn preconditions(state: &UseModel, action: &UseAction) -> bool {
        let live = |value: &usize| state.live_values.contains(value);
        let operand_count = |op: usize| state.op(op).map(|m| m.operands.len());
        match action {
            UseAction::Create {
                block, operands, ..
            } => state.blocks.contains(block) && operands.iter().all(live),
            UseAction::Set { op, index, value } => {
                operand_count(*op).is_some_and(|len| *index < len) && live(value)
            }
            UseAction::Push { op, value } => state.op(*op).is_some() && live(value),
            UseAction::RemoveOperand { op, index } => {
                operand_count(*op).is_some_and(|len| *index < len)
            }
            UseAction::ReplaceAllUses { old, new } => *old < state.value_count && live(new),
            UseAction::Remove(op) => state.removable(*op),
            UseAction::Clone(op) => state.op(*op).is_some(),
        }
    }

    fn apply(mut state: UseModel, action: &UseAction) -> UseModel {
        match *action {
            UseAction::Create {
                block,
                ref operands,
                results,
                container,
            } => {
                state.add_op(block, operands.clone(), results, container);
            }
            UseAction::Set { op, index, value } => state.op_mut(op).operands[index] = value,
            UseAction::Push { op, value } => state.op_mut(op).operands.push(value),
            UseAction::RemoveOperand { op, index } => {
                state.op_mut(op).operands.remove(index);
            }
            UseAction::ReplaceAllUses { old, new } => {
                for m in &mut state.ops {
                    for operand in &mut m.operands {
                        if *operand == old {
                            *operand = new;
                        }
                    }
                }
            }
            UseAction::Remove(op) => state.remove(op),
            UseAction::Clone(op) => {
                let block = state.op(op).expect("live model op").block;
                state.record_clone(op, block, &mut Vec::new());
            }
        }
        state
    }
}

/// A detached block with `args` arguments of type `ty`.
fn new_block(ctx: &mut IrContext, loc: Location, ty: TypeRef, args: usize) -> BlockRef {
    ctx.create_block(BlockData {
        location: loc,
        args: (0..args)
            .map(|_| BlockArgData {
                ty,
                attrs: AttributeMap::new(),
            })
            .collect(),
        ops: SmallVec::new(),
        parent_region: None,
    })
}

/// The context, and the references the model's ids stand for.
struct UseSut {
    ctx: IrContext,
    loc: Location,
    ty: TypeRef,
    module: Module,
    ops: Vec<OpRef>,
    /// The block each operation was created in.
    op_blocks: Vec<BlockRef>,
    blocks: Vec<BlockRef>,
    values: Vec<ValueRef>,
}

impl UseSut {
    fn new_block(&mut self, args: usize) -> BlockRef {
        new_block(&mut self.ctx, self.loc, self.ty, args)
    }

    fn add_block(&mut self, block: BlockRef) {
        self.blocks.push(block);
        self.values.extend(self.ctx.block_args(block));
    }

    /// Name `op`, which the context has just appended to `block`, in the
    /// order `UseModel::add_op` assigns ids.
    fn add_op(&mut self, op: OpRef, block: BlockRef) {
        self.ops.push(op);
        self.op_blocks.push(block);
        self.values.extend(self.ctx.op_results(op));
        if let Some(region) = self.ctx.op_region(op, 0) {
            let body = self.ctx.region(region).blocks[0];
            self.add_block(body);
        }
    }

    /// Name a clone and its nested operations in the order
    /// `UseModel::record_clone` assigns ids.
    fn add_clone(&mut self, op: OpRef, block: BlockRef) {
        self.add_op(op, block);
        if let Some(region) = self.ctx.op_region(op, 0) {
            let body = self.ctx.region(region).blocks[0];
            for child in self.ctx.block(body).ops.clone() {
                self.add_clone(child, body);
            }
        }
    }
}

struct UseTest;

impl StateMachineTest for UseTest {
    type SystemUnderTest = UseSut;
    type Reference = UseMachine;

    fn init_test(_: &UseModel) -> UseSut {
        let mut ctx = IrContext::new();
        let loc = location(&mut ctx);
        let ty = ctx.intern_type(TypeDataBuilder::new("core", "i32").build());
        let module_block = new_block(&mut ctx, loc, ty, 2);
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
        let mut sut = UseSut {
            ctx,
            loc,
            ty,
            module,
            ops: Vec::new(),
            op_blocks: Vec::new(),
            blocks: Vec::new(),
            values: Vec::new(),
        };
        sut.add_block(module_block);
        sut
    }

    fn apply(mut sut: UseSut, _: &UseModel, action: UseAction) -> UseSut {
        match action {
            UseAction::Create {
                block,
                operands,
                results,
                container,
            } => {
                let block = sut.blocks[block];
                let ty = sut.ty;
                let mut builder =
                    OperationDataBuilder::new(sut.loc, Symbol::new("test"), Symbol::new("op"))
                        .operands(operands.iter().map(|&v| sut.values[v]))
                        .results((0..results).map(|_| ty));
                if container {
                    let body = sut.new_block(1);
                    let region = sut.ctx.create_region(RegionData {
                        location: sut.loc,
                        blocks: smallvec::smallvec![body],
                        parent_op: None,
                    });
                    builder = builder.region(region);
                }
                let data = builder.build(&mut sut.ctx);
                let op = sut.ctx.create_op(data);
                sut.ctx.push_op(block, op);
                sut.add_op(op, block);
            }
            UseAction::Set { op, index, value } => {
                sut.ctx
                    .set_op_operand(sut.ops[op], index as u32, sut.values[value]);
            }
            UseAction::Push { op, value } => {
                sut.ctx.push_op_operand(sut.ops[op], sut.values[value]);
            }
            UseAction::RemoveOperand { op, index } => {
                sut.ctx.remove_op_operand(sut.ops[op], index as u32);
            }
            UseAction::ReplaceAllUses { old, new } => {
                let (old, new) = (sut.values[old], sut.values[new]);
                sut.ctx.replace_all_uses(old, new);
                assert!(old == new || !sut.ctx.has_uses(old));
            }
            UseAction::Remove(op) => {
                sut.ctx.remove_op_from_block(sut.op_blocks[op], sut.ops[op]);
                sut.ctx.remove_op(sut.ops[op]);
            }
            UseAction::Clone(op) => {
                let block = sut.op_blocks[op];
                let clone = sut.ctx.clone_op(sut.ops[op], &mut IrMapping::new());
                sut.ctx.push_op(block, clone);
                sut.add_clone(clone, block);
            }
        }
        sut
    }

    /// The context's block contents, operands, and use chains must match the
    /// model, and the validator's use-chain check must accept the module.
    fn check_invariants(sut: &UseSut, model: &UseModel) {
        let ctx = &sut.ctx;
        assert_eq!(sut.ops.len(), model.op_count);
        assert_eq!(sut.blocks.len(), model.block_count);
        assert_eq!(sut.values.len(), model.value_count);
        for (block, ops) in &model.block_ops {
            let expected: Vec<_> = ops.iter().map(|&op| sut.ops[op]).collect();
            assert_eq!(&ctx.block(sut.blocks[*block]).ops[..], &expected[..]);
        }
        for m in &model.ops {
            let expected: Vec<_> = m.operands.iter().map(|&v| sut.values[v]).collect();
            assert_eq!(ctx.op_operands(sut.ops[m.op]), &expected[..]);
        }
        for (index, &value) in sut.values.iter().enumerate() {
            let mut actual: Vec<_> = ctx
                .uses(value)
                .iter()
                .map(|u| (u.user, u.operand_index))
                .collect();
            actual.sort_unstable();
            let mut expected: Vec<_> = model
                .expected_uses(index)
                .into_iter()
                .map(|(op, operand)| (sut.ops[op], operand))
                .collect();
            expected.sort_unstable();
            assert_eq!(ctx.has_uses(value), !expected.is_empty());
            assert_eq!(actual, expected, "uses of {value}");
        }
        let result = crate::validation::validate_use_chains(ctx, sut.module);
        assert!(result.is_ok(), "{result}");
    }
}

prop_state_machine! {
    #[test]
    fn use_chains_match_an_operand_model(sequential 1..64 => UseTest);
}
