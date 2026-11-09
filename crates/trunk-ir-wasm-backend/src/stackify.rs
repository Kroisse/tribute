//! Which values of a `wasm.func` body stay on the Wasm operand stack.
//!
//! `Stackify` plays the part of LLVM's WebAssemblyRegStackify for the
//! [`ExplicitLocals`](crate::passes::explicit_locals::ExplicitLocals) pass,
//! which reads it before it assigns locals: a value it leaves on the stack
//! gets no local, a result nothing uses is dropped, and some local reads move
//! before the computation of the stack values above them. The analysis moves
//! no operation, so side effects keep their order
//! (`design/wasm-backend.md`, "Stack에 남는 값").

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use trunk_ir::analysis::{Analysis, AnalysisContext, AnalysisError, InfallibleAnalysis};
use trunk_ir::callable::{CallableBody, classify_callable_body};
use trunk_ir::dialect::wasm;
use trunk_ir::ops::DialectOp;
use trunk_ir::smallvec::SmallVec;
use trunk_ir::{BlockRef, IrContext, OpRef, RegionRef, ValueDef, ValueRef};

use crate::stack_effect::{
    YieldTransfer, pushed_results, stack_operand_slots, structured_result_type, yield_transfer,
};

/// A local read emitted before another operation than its user: operand
/// `slot` of `user`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HoistedRead {
    pub(crate) user: OpRef,
    pub(crate) slot: u32,
}

/// The stack decisions for one `wasm.func` body.
#[derive(Debug, Default)]
pub(crate) struct Stackify {
    /// Results that stay on the stack from their producer to their one user.
    stacked: HashSet<ValueRef>,
    /// Results left on the stack that nothing uses.
    dropped: HashSet<ValueRef>,
    /// Local reads to emit right before an operation, in order.
    hoisted: HashMap<OpRef, SmallVec<[HoistedRead; 2]>>,
}

impl Stackify {
    pub(crate) fn is_stacked(&self, value: ValueRef) -> bool {
        self.stacked.contains(&value)
    }

    pub(crate) fn is_dropped(&self, value: ValueRef) -> bool {
        self.dropped.contains(&value)
    }

    /// The local reads to emit right before `op`.
    pub(crate) fn hoisted_before(&self, op: OpRef) -> &[HoistedRead] {
        self.hoisted.get(&op).map_or(&[], |reads| reads)
    }
}

impl Analysis for Stackify {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let ir = ctx.ir();
        let mut stackify = Stackify::default();
        if let Ok(CallableBody::Definition { region, .. }) = classify_callable_body(ir, target) {
            stackify.region(ir, region, &mut Vec::new(), false);
        }
        Ok(stackify)
    }
}

impl InfallibleAnalysis for Stackify {}

/// A value on the simulated stack and the position, in its block, of the
/// operation whose instructions first push its computation.
struct Entry {
    value: ValueRef,
    start: usize,
}

impl Stackify {
    /// Decide the values of one region. `loop_args` holds, for each enclosing
    /// structured operation, innermost last, the number of body block
    /// arguments of a loop or `None`; `passes_result` tells whether the
    /// parent takes the region's value.
    fn region(
        &mut self,
        ctx: &IrContext,
        region: RegionRef,
        loop_args: &mut Vec<Option<usize>>,
        passes_result: bool,
    ) {
        // `ExplicitLocals` rejects a region of several blocks.
        let [block] = ctx.region(region).blocks[..] else {
            return;
        };
        let ops = &ctx.block(block).ops;
        let block_ops = BlockOps {
            block,
            ops,
            positions: ops
                .iter()
                .enumerate()
                .map(|(position, &op)| (op, position))
                .collect(),
        };
        // A region whose last operation is not a yield passes that
        // operation's first result, which `ExplicitLocals` reads from its
        // local.
        let implicit_result = ops
            .last()
            .filter(|&&last| passes_result && !wasm::Yield::matches(ctx, last))
            .and_then(|&last| ctx.op_results(last).first().copied());

        let mut pending: Vec<Entry> = Vec::new();
        for (position, &op) in ops.iter().enumerate() {
            let operands = if wasm::Yield::matches(ctx, op) {
                match yield_transfer(ctx, ops, position, loop_args, passes_result) {
                    YieldTransfer::Stack | YieldTransfer::LoopArgument => {
                        stack_operand_slots(ctx, op)
                    }
                    YieldTransfer::LoopArguments(_) | YieldTransfer::None => SmallVec::new(),
                }
            } else {
                stack_operand_slots(ctx, op)
            };
            let start = self.take_operands(ctx, op, position, &block_ops, &operands, &mut pending);

            let pushes_region_result = structured_result_type(ctx, op).is_some();
            let regions: SmallVec<[RegionRef; 4]> = ctx.op_regions(op).collect();
            if !regions.is_empty() {
                loop_args.push(
                    wasm::Loop::from_op(ctx, op)
                        .ok()
                        .and_then(|loop_op| ctx.region(loop_op.body(ctx)).blocks.first().copied())
                        .map(|body| ctx.block_args(body).len()),
                );
                for nested in regions {
                    self.region(ctx, nested, loop_args, pushes_region_result);
                }
                loop_args.pop();
            }

            let pushed = pushed_results(ctx, op);
            for &result in &pushed {
                if Some(result) == implicit_result {
                    continue;
                }
                let uses = ctx.uses(result);
                if uses.is_empty() {
                    self.dropped.insert(result);
                } else if pushed.len() == 1
                    && let [only] = uses
                    && ctx.op(only.user).parent_block == Some(block)
                {
                    pending.push(Entry {
                        value: result,
                        start,
                    });
                }
            }
        }
    }

    /// Match the stack operands of `op` against the simulated stack and
    /// return the position where the computation of `op`'s results starts.
    fn take_operands(
        &mut self,
        ctx: &IrContext,
        op: OpRef,
        position: usize,
        block: &BlockOps<'_>,
        operands: &[(ValueRef, u32)],
        pending: &mut Vec<Entry>,
    ) -> usize {
        let is_pending = |pending: &[Entry], value| pending.iter().any(|e| e.value == value);
        // Operands after the last pending one are read from their locals
        // right before `op`.
        let taken = operands
            .iter()
            .rposition(|&(value, _)| is_pending(pending, value))
            .map_or(0, |last| last + 1);

        let mut cursor = pending.len();
        let mut start = position;
        let mut hoists: SmallVec<[(usize, HoistedRead); 2]> = SmallVec::new();
        let mut matched = true;
        for &(value, slot) in operands[..taken].iter().rev() {
            if cursor > 0 && pending[cursor - 1].value == value {
                cursor -= 1;
                start = pending[cursor].start;
            } else if !is_pending(pending, value) && block.readable_before(ctx, value, start) {
                hoists.push((start, HoistedRead { user: op, slot }));
            } else {
                matched = false;
                break;
            }
        }

        if matched && taken > 0 {
            for entry in pending.drain(cursor..) {
                self.stacked.insert(entry.value);
            }
            // The reads were collected from the top of the stack down. Each
            // goes before the reads already recorded at its position, which
            // belong to computations nested inside the value above it.
            for (at, read) in hoists {
                let before = block.ops[at];
                self.hoisted.entry(before).or_default().insert(0, read);
            }
            start
        } else {
            // What `op` uses but cannot take from the stack stays in locals.
            let used = ctx.op_operands(op);
            pending.retain(|entry| !used.contains(&entry.value));
            position
        }
    }
}

/// The operations of one block and their positions.
struct BlockOps<'a> {
    block: BlockRef,
    ops: &'a [OpRef],
    positions: HashMap<OpRef, usize>,
}

impl BlockOps<'_> {
    /// Whether `value`'s local holds it before the operation at `position`:
    /// a block argument, a value defined outside the block, or a result of
    /// an earlier operation of the block.
    fn readable_before(&self, ctx: &IrContext, value: ValueRef, position: usize) -> bool {
        match ctx.value_def(value) {
            ValueDef::BlockArg(..) => true,
            ValueDef::OpResult(def, _) => {
                ctx.op(def).parent_block != Some(self.block)
                    || self.positions.get(&def).is_some_and(|&at| at < position)
            }
        }
    }
}
