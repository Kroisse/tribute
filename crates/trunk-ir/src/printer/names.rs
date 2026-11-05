//! The naming pre-pass: value names and block labels, numbered before
//! printing in the order the printer writes them.

use rustc_hash::FxHashMap as HashMap;

use crate::asm_format::OpAsmFormatDef;
use crate::context::IrContext;
use crate::dialect::core;
use crate::op_def::OpDef;
use crate::op_flag::{IsolatedFromAbove, OpFlag as _};
use crate::ops::DialectOp;
use crate::refs::{BlockRef, OpRef, RegionRef, ValueRef};

/// The number of each named value (`%N`) and labelled block (`^bbN`).
#[derive(Debug, Default)]
pub(crate) struct AsmNames {
    values: HashMap<ValueRef, u32>,
    blocks: HashMap<BlockRef, u32>,
}

impl AsmNames {
    pub(crate) fn value(&self, value: ValueRef) -> Option<u32> {
        self.values.get(&value).copied()
    }

    pub(crate) fn block(&self, block: BlockRef) -> Option<u32> {
        self.blocks.get(&block).copied()
    }
}

/// Names the values and blocks an operation's format writes, before
/// printing, in the order the format writes them, as MLIR's
/// `SSANameState` does.
///
/// Values and blocks are numbered separately, so a format keeps only the
/// order among the values it names and among the regions whose blocks it
/// labels. An operation with a custom format whose order differs from
/// [`Self::default_op`], or that writes one name for several values,
/// names them in [`OpAsmFormat::assign_names`](crate::asm_format::OpAsmFormat::assign_names).
pub struct AsmNamer<'a> {
    ctx: &'a IrContext,
    names: AsmNames,
    next_value: u32,
    next_block: u32,
}

impl<'a> AsmNamer<'a> {
    pub(crate) fn new(ctx: &'a IrContext) -> Self {
        Self {
            ctx,
            names: AsmNames::default(),
            next_value: 0,
            next_block: 0,
        }
    }

    pub(crate) fn finish(self) -> AsmNames {
        self.names
    }

    /// The IR context.
    pub fn ctx(&self) -> &'a IrContext {
        self.ctx
    }

    /// Name `values` in order, as a format names the values it defines.
    pub fn values(&mut self, values: &[ValueRef]) {
        for &value in values {
            self.names.values.insert(value, self.next_value);
            self.next_value += 1;
        }
    }

    /// Give each of `values` the name of the value at its position in
    /// `like`, for values of sibling regions whose name the format writes
    /// only once, such as a parameter every body of an operation shares.
    pub fn share(&mut self, values: &[ValueRef], like: &[ValueRef]) {
        for (&value, &like) in values.iter().zip(like) {
            if let Some(number) = self.names.value(like) {
                self.names.values.insert(value, number);
            }
        }
    }

    /// Name the results of `op`.
    pub fn results(&mut self, op: OpRef) {
        let ctx = self.ctx;
        self.values(ctx.op_results(op));
    }

    /// Name the entry block arguments of `region`, which a format writes
    /// before the region, as `(%a: A, ..)`.
    pub fn entry_args(&mut self, region: RegionRef) {
        let ctx = self.ctx;
        self.values(entry_args(ctx, region));
    }

    /// Name a region printed in generic form: label its blocks, then name
    /// each block's arguments and operations.
    pub fn region(&mut self, region: RegionRef) {
        let ctx = self.ctx;
        let blocks = &ctx.region(region).blocks;
        self.label(blocks);
        let elide_entry = blocks.first().is_some_and(|&entry| {
            ctx.block_args(entry).is_empty() && super::can_elide_entry_label(ctx, blocks, entry)
        });
        for (index, &block) in blocks.iter().enumerate() {
            if !(index == 0 && elide_entry) {
                self.values(ctx.block_args(block));
            }
            self.ops(block);
        }
    }

    /// Name a region whose entry block arguments the format wrote before
    /// it, printed with its entry block elided: label its blocks, then name
    /// the arguments of the other blocks and each block's operations.
    pub fn region_without_entry(&mut self, region: RegionRef) {
        let ctx = self.ctx;
        let blocks = &ctx.region(region).blocks;
        self.label(blocks);
        for (index, &block) in blocks.iter().enumerate() {
            if index > 0 {
                self.values(ctx.block_args(block));
            }
            self.ops(block);
        }
    }

    /// Run `name` with fresh value and block numbering, as an operation
    /// isolated from above starts, and continue the current numbering
    /// afterwards.
    pub fn fresh_scope(&mut self, name: impl FnOnce(&mut Self)) {
        let saved = (
            std::mem::take(&mut self.next_value),
            std::mem::take(&mut self.next_block),
        );
        name(self);
        (self.next_value, self.next_block) = saved;
    }

    /// Name a nested operation as its format writes it.
    pub fn op(&mut self, op: OpRef) {
        let ctx = self.ctx;
        if core::Module::matches(ctx, op) {
            return self.module(op);
        }
        let def = OpDef::of(ctx, op);
        match def.and_then(|def| Some((def, def.asm_format?))) {
            Some((_, OpAsmFormatDef::Hooks(hooks))) => (hooks.assign_names)(self, op),
            Some((def, OpAsmFormatDef::Declarative(format))) => {
                crate::asm_format::declarative::assign_names(self, op, def, format)
            }
            None => self.generic_op(op),
        }
    }

    /// Name an operation printed in generic form: its results, then each
    /// region.
    pub fn generic_op(&mut self, op: OpRef) {
        let ctx = self.ctx;
        self.results(op);
        for region in ctx.op_regions(op) {
            self.region(region);
        }
    }

    /// Name an operation as a custom format writes it by default: its
    /// results, then each region in order, with fresh numbering when the
    /// operation is isolated from above.
    pub fn default_op(&mut self, op: OpRef) {
        let ctx = self.ctx;
        self.results(op);
        let regions = |namer: &mut Self| {
            for region in ctx.op_regions(op) {
                namer.region(region);
            }
        };
        if IsolatedFromAbove::is_set(ctx, op) {
            self.fresh_scope(regions);
        } else {
            regions(self);
        }
    }

    /// A module numbers each operation of its body afresh.
    pub(crate) fn module(&mut self, op: OpRef) {
        let ctx = self.ctx;
        for region in ctx.op_regions(op) {
            for &block in &ctx.region(region).blocks {
                for &child in &ctx.block(block).ops {
                    self.fresh_scope(|namer| namer.op(child));
                }
            }
        }
    }

    fn label(&mut self, blocks: &[BlockRef]) {
        for &block in blocks {
            self.names.blocks.insert(block, self.next_block);
            self.next_block += 1;
        }
    }

    fn ops(&mut self, block: BlockRef) {
        let ctx = self.ctx;
        for &op in &ctx.block(block).ops {
            self.op(op);
        }
    }
}

/// The entry block arguments of `region`, or none when it has no blocks.
pub fn entry_args(ctx: &IrContext, region: RegionRef) -> &[ValueRef] {
    ctx.region(region)
        .blocks
        .first()
        .map_or(&[][..], |&entry| ctx.block_args(entry))
}
