//! Recording the IR entities an analysis computation reads, and reducing
//! them to the anchors whose counters cover them.
//!
//! A computation's footprint is the set of outermost anchors over its
//! reads. Each read maps to an anchor independently of the others, and
//! taking the outermost anchors of a union gives the same set whether or
//! not parts of it were reduced first. A computation's pending reads can
//! therefore be folded into its [`FootprintBuilder`] at any point, which
//! bounds the log by the footprint's size rather than the number of reads.
//! Folding early is exact because the IR does not change while a
//! computation runs.

use rustc_hash::FxHashMap as HashMap;
use std::cell::{Cell, RefCell};

use smallvec::SmallVec;

use super::{IrContext, ValueDef};
use crate::refs::{BlockRef, OpRef, RegionRef, ValueRef};

/// One IR entity an analysis computation read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadEntity {
    Op(OpRef),
    Block(BlockRef),
    Region(RegionRef),
    Value(ValueRef),
}

/// The anchors whose counters cover a computation's reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ReadFootprint {
    /// Some read had no anchor chain.
    WholeContext,
    /// The outermost anchors read; none is an ancestor of another.
    Anchors(SmallVec<[OpRef; 2]>),
}

/// The smallest number of pending reads a computation folds into its
/// footprint when the log is full.
const MIN_FOLDED_READS: usize = 1024;

/// Records reads through `&self` accessors during nested analysis
/// computations.
#[derive(Default)]
pub(super) struct ReadRecorder {
    /// The number of active computations, read on every accessor call.
    depth: Cell<u32>,
    state: RefCell<RecorderState>,
}

#[derive(Default)]
struct RecorderState {
    /// Pending reads of every active computation, each computation's after
    /// those of the computations enclosing it.
    log: Vec<ReadEntity>,
    /// The active computations, innermost last.
    computations: Vec<Computation>,
}

struct Computation {
    /// Where this computation's pending reads start in the log.
    start: usize,
    footprint: FootprintBuilder,
}

/// One recorded computation, returned by [`IrContext::begin_reads`].
#[must_use]
pub(crate) struct ReadScope {
    depth: u32,
}

/// The footprint of the reads folded so far.
#[derive(Default)]
struct FootprintBuilder {
    whole_context: bool,
    /// The anchors of the folded reads, possibly with duplicates and
    /// nested anchors until [`FootprintBuilder::reduce`] runs.
    anchors: SmallVec<[OpRef; 2]>,
}

impl FootprintBuilder {
    /// Add the anchors of `reads`.
    fn fold(&mut self, ctx: &IrContext, reads: &[ReadEntity]) {
        if self.whole_context {
            return;
        }
        // Operations in one block share the innermost anchor at or above
        // the block's owner, so the walk to the root runs once per block.
        let mut block_anchors: HashMap<BlockRef, Option<OpRef>> = HashMap::default();
        let mut last_block = None;
        for &entity in reads {
            let (op, block) = match entity {
                ReadEntity::Op(op) => (Some(op), ctx.ops[op].parent_block),
                ReadEntity::Block(block) => (None, Some(block)),
                ReadEntity::Region(region) => match ctx.regions[region].parent_op {
                    Some(op) => (Some(op), ctx.ops[op].parent_block),
                    None => return self.set_whole_context(),
                },
                ReadEntity::Value(value) => match ctx.values[value].def {
                    ValueDef::OpResult(op, _) => (Some(op), ctx.ops[op].parent_block),
                    ValueDef::BlockArg(block, _) => (None, Some(block)),
                },
            };
            let anchor = match (op, block) {
                (Some(op), block) if ctx.is_anchor(op) => {
                    if block.is_some_and(|block| {
                        ctx.anchor_of_block(block, &mut block_anchors).is_none()
                    }) {
                        return self.set_whole_context();
                    }
                    op
                }
                (_, None) => return self.set_whole_context(),
                (_, Some(block)) => {
                    if last_block == Some(block) {
                        continue;
                    }
                    last_block = Some(block);
                    match ctx.anchor_of_block(block, &mut block_anchors) {
                        Some(anchor) => anchor,
                        None => return self.set_whole_context(),
                    }
                }
            };
            if self.anchors.last() != Some(&anchor) {
                self.anchors.push(anchor);
            }
        }
    }

    fn set_whole_context(&mut self) {
        self.whole_context = true;
        self.anchors.clear();
    }

    /// Keep only the outermost anchors, each once.
    fn reduce(&mut self, ctx: &IrContext) {
        self.anchors.sort_unstable();
        self.anchors.dedup();
        let anchors = &self.anchors;
        self.anchors = anchors
            .iter()
            .copied()
            .filter(|&anchor| {
                let mut current = anchor;
                while let Some(parent) = ctx.parent_op(current) {
                    if anchors.binary_search(&parent).is_ok() {
                        return false;
                    }
                    current = parent;
                }
                true
            })
            .collect();
    }

    fn finish(mut self, ctx: &IrContext) -> ReadFootprint {
        if self.whole_context {
            return ReadFootprint::WholeContext;
        }
        self.reduce(ctx);
        ReadFootprint::Anchors(self.anchors)
    }
}

impl IrContext {
    #[inline(always)]
    pub(super) fn record(&self, entity: ReadEntity) {
        if self.reads.depth.get() != 0 {
            self.record_read(entity);
        }
    }

    #[inline(never)]
    fn record_read(&self, entity: ReadEntity) {
        let state = &mut *self.reads.state.borrow_mut();
        let computation = state
            .computations
            .last_mut()
            .expect("reads recorded without an active computation");
        if computation.footprint.whole_context {
            return;
        }
        let pending = &state.log[computation.start..];
        if pending.last() == Some(&entity) {
            return;
        }
        if state.log.len() == state.log.capacity() && pending.len() >= MIN_FOLDED_READS {
            computation.footprint.fold(self, pending);
            computation.footprint.reduce(self);
            state.log.truncate(computation.start);
        }
        state.log.push(entity);
    }

    /// Start recording reads for one computation, nested in any active one.
    pub(crate) fn begin_reads(&self) -> ReadScope {
        let state = &mut *self.reads.state.borrow_mut();
        state.computations.push(Computation {
            start: state.log.len(),
            footprint: FootprintBuilder::default(),
        });
        let depth = self.reads.depth.get() + 1;
        self.reads.depth.set(depth);
        ReadScope { depth }
    }

    /// Stop recording the computation `scope` began and return the
    /// footprint of its reads, excluding those of computations nested in it.
    pub(crate) fn end_reads(&self, scope: ReadScope) -> ReadFootprint {
        let depth = self.reads.depth.get();
        assert_eq!(depth, scope.depth, "read scopes must end innermost first");
        self.reads.depth.set(depth - 1);
        let state = &mut *self.reads.state.borrow_mut();
        let mut computation = state
            .computations
            .pop()
            .expect("depth counts the active computations");
        computation
            .footprint
            .fold(self, &state.log[computation.start..]);
        state.log.truncate(computation.start);
        computation.footprint.finish(self)
    }

    /// The innermost anchor at or above `block`'s owner, or `None` when the
    /// block's root is not an anchor.
    fn anchor_of_block(
        &self,
        block: BlockRef,
        memo: &mut HashMap<BlockRef, Option<OpRef>>,
    ) -> Option<OpRef> {
        if let Some(&anchor) = memo.get(&block) {
            return anchor;
        }
        let anchor = match self.block_owner(block) {
            None => None,
            Some(owner) => {
                let above = self.ops[owner]
                    .parent_block
                    .map(|parent| self.anchor_of_block(parent, memo));
                match above {
                    Some(None) => None,
                    _ if self.is_anchor(owner) => Some(owner),
                    Some(above) => above,
                    None => None,
                }
            }
        };
        memo.insert(block, anchor);
        anchor
    }
}
