//! Lower adt.string_const and adt.bytes_const to wasm data segments.
//!
//! This pass uses a two-phase approach:
//! 1. Analysis: Collect all string/bytes constants and allocate data segment offsets
//! 2. Transform: Replace const operations with wasm ops
//!
//! The analysis produces a plain struct (no Salsa tracking).

use std::collections::HashMap;

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::adt as arena_adt;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;
use trunk_ir::arena::refs::{OpRef, RegionRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, ArenaRewritePattern, ArenaTypeConverter, PatternApplicator, PatternRewriter,
};
use trunk_ir::arena::types::Attribute as ArenaAttribute;
use trunk_ir::arena::types::TypeDataBuilder;
use trunk_ir::ir::Symbol;

/// Result of const analysis - maps content to allocated offset.
pub struct ConstAnalysis {
    /// String allocations: (content, offset, length) - for active data segments.
    pub string_allocations: Vec<(Vec<u8>, u32, u32)>,
    /// Bytes allocations: (content, data_idx, length) - for passive data segments.
    /// data_idx is the index into the passive data segment array.
    pub bytes_allocations: Vec<(Vec<u8>, u32, u32)>,
    /// Total size of string data segments (for linear memory).
    pub string_total_size: u32,
}

impl ConstAnalysis {
    /// Legacy accessor for backwards compatibility.
    /// Returns string allocations only.
    pub fn allocations(&self) -> &[(Vec<u8>, u32, u32)] {
        &self.string_allocations
    }

    /// Legacy accessor for backwards compatibility.
    pub fn total_size(&self) -> u32 {
        self.string_total_size
    }

    /// Look up the offset for given string content.
    /// Returns (offset, length).
    pub fn offset_for(&self, content: &[u8]) -> Option<(u32, u32)> {
        self.string_allocations
            .iter()
            .find(|(data, _, _)| data.as_slice() == content)
            .map(|(_, offset, len)| (*offset, *len))
    }

    /// Look up the bytes allocation info for given content.
    /// Returns (data_idx, 0, length) where data_idx is the passive data segment index.
    pub fn bytes_info_for(&self, content: &[u8]) -> Option<(u32, u32, u32)> {
        self.bytes_allocations
            .iter()
            .find(|(data, _, _)| data.as_slice() == content)
            .map(|(_, data_idx, len)| (*data_idx, 0, *len))
    }
}

/// Context for collecting const allocations during analysis.
struct ConstCollector {
    string_allocations: Vec<(Vec<u8>, u32, u32)>,
    bytes_allocations: Vec<(Vec<u8>, u32, u32)>,
    string_seen: HashMap<Vec<u8>, usize>,
    bytes_seen: HashMap<Vec<u8>, usize>,
    next_string_offset: u32,
    next_bytes_idx: u32,
}

impl ConstCollector {
    fn new() -> Self {
        Self {
            string_allocations: Vec::new(),
            bytes_allocations: Vec::new(),
            string_seen: HashMap::new(),
            bytes_seen: HashMap::new(),
            next_string_offset: 0,
            next_bytes_idx: 0,
        }
    }

    fn align_to(value: u32, align: u32) -> u32 {
        if align == 0 {
            return value;
        }
        value.div_ceil(align) * align
    }

    fn visit_op(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);

        if data.dialect == arena_adt::DIALECT_NAME() {
            if data.name == Symbol::new("string_const") {
                if let Some(ArenaAttribute::String(s)) = data.attributes.get(&Symbol::new("value"))
                {
                    let bytes = s.clone().into_bytes();
                    if !self.string_seen.contains_key(&bytes) {
                        let offset = Self::align_to(self.next_string_offset, 4);
                        let len = bytes.len() as u32;
                        self.string_seen
                            .insert(bytes.clone(), self.string_allocations.len());
                        self.string_allocations.push((bytes, offset, len));
                        self.next_string_offset = offset + len;
                    }
                }
            } else if data.name == Symbol::new("bytes_const") {
                if let Some(ArenaAttribute::Bytes(b)) = data.attributes.get(&Symbol::new("value")) {
                    let bytes: Vec<u8> = b.to_vec();
                    if !self.bytes_seen.contains_key(&bytes) {
                        let data_idx = self.next_bytes_idx;
                        let len = bytes.len() as u32;
                        self.bytes_seen
                            .insert(bytes.clone(), self.bytes_allocations.len());
                        self.bytes_allocations.push((bytes, data_idx, len));
                        self.next_bytes_idx += 1;
                    }
                }
            }
        }

        // Recurse into regions
        for &region in data.regions.iter() {
            walk_ops_in_region(ctx, region, &mut |ctx, nested_op| {
                self.visit_op(ctx, nested_op);
            });
        }
    }
}

/// Walk all operations in a region recursively.
fn walk_ops_in_region(
    ctx: &IrContext,
    region: RegionRef,
    callback: &mut impl FnMut(&IrContext, OpRef),
) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            callback(ctx, op);
            for &nested in ctx.op(op).regions.iter() {
                walk_ops_in_region(ctx, nested, callback);
            }
        }
    }
}

/// Analyze a module to collect all string/bytes constants and allocate offsets.
pub fn analyze_consts(ctx: &IrContext, module: ArenaModule) -> ConstAnalysis {
    let mut collector = ConstCollector::new();

    // Walk all operations in module body
    if let Some(body) = module.body(ctx) {
        for &block in ctx.region(body).blocks.iter() {
            for &op in ctx.block(block).ops.iter() {
                collector.visit_op(ctx, op);
            }
        }
    }

    ConstAnalysis {
        string_allocations: collector.string_allocations,
        bytes_allocations: collector.bytes_allocations,
        string_total_size: collector.next_string_offset,
    }
}

/// Lower const operations using pre-computed analysis.
pub fn lower(ctx: &mut IrContext, module: ArenaModule, analysis: &ConstAnalysis) {
    let string_allocations = analysis.string_allocations.clone();
    let bytes_allocations = analysis.bytes_allocations.clone();

    let applicator = PatternApplicator::new(ArenaTypeConverter::new())
        .add_pattern(StringConstPattern::new(string_allocations))
        .add_pattern(BytesConstPattern::new(bytes_allocations));
    applicator.apply_partial(ctx, module);
}

/// Allocation data: (content, offset, length).
type Allocations = Vec<(Vec<u8>, u32, u32)>;

/// Look up offset and length for given content.
fn lookup_offset(allocations: &Allocations, content: &[u8]) -> Option<(u32, u32)> {
    allocations
        .iter()
        .find(|(data, _, _)| data.as_slice() == content)
        .map(|(_, offset, len)| (*offset, *len))
}

/// Pattern for `adt.string_const` -> `wasm.i32_const`
struct StringConstPattern {
    allocations: Allocations,
}

impl StringConstPattern {
    fn new(allocations: Allocations) -> Self {
        Self { allocations }
    }
}

impl ArenaRewritePattern for StringConstPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(string_const) = arena_adt::StringConst::from_op(ctx, op) else {
            return false;
        };

        let value_str = string_const.value(ctx);
        let content = value_str.into_bytes();

        let Some((offset, _len)) = lookup_offset(&self.allocations, &content) else {
            return false;
        };

        let location = ctx.op(op).location;
        let i32_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());

        // Use typed helper to create wasm.i32_const with just the offset.
        // Length information is available in ConstAnalysis and will be used by emit.rs.
        let new_op = arena_wasm::i32_const(ctx, location, i32_ty, offset as i32);

        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "StringConstPattern"
    }
}

/// Bytes allocation data: (content, data_idx, length).
type BytesAllocations = Vec<(Vec<u8>, u32, u32)>;

/// Look up data_idx and length for given bytes content.
fn lookup_bytes_info(allocations: &BytesAllocations, content: &[u8]) -> Option<(u32, u32)> {
    allocations
        .iter()
        .find(|(data, _, _)| data.as_slice() == content)
        .map(|(_, data_idx, len)| (*data_idx, *len))
}

/// Pattern for `adt.bytes_const` -> `wasm.bytes_from_data`
struct BytesConstPattern {
    allocations: BytesAllocations,
}

impl BytesConstPattern {
    fn new(allocations: BytesAllocations) -> Self {
        Self { allocations }
    }
}

impl ArenaRewritePattern for BytesConstPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(bytes_const) = arena_adt::BytesConst::from_op(ctx, op) else {
            return false;
        };

        let value_attr = bytes_const.value(ctx);
        let ArenaAttribute::Bytes(b) = value_attr else {
            return false;
        };
        let content: Vec<u8> = b.to_vec();

        let Some((data_idx, len)) = lookup_bytes_info(&self.allocations, &content) else {
            return false;
        };

        let location = ctx.op(op).location;
        let bytes_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("bytes")).build());

        // Create wasm.bytes_from_data operation
        let new_op = arena_wasm::bytes_from_data(ctx, location, bytes_ty, data_idx, 0, len);

        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesConstPattern"
    }
}
