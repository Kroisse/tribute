//! Lower adt.string_const and adt.bytes_const to wasm data segments.
//!
//! This pass uses a two-phase approach:
//! 1. Analysis: Collect all string/bytes constants and allocate data segment offsets
//! 2. Transform: Replace const operations with wasm.i32_const (pointer to data)
//!
//! The analysis is implemented as a salsa::tracked function for incremental computation.

use std::collections::HashMap;

use tribute_ir::dialect::adt;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol};

use crate::type_converter::wasm_type_converter;

/// Result of const analysis - maps content to allocated offset.
#[salsa::tracked]
pub struct ConstAnalysis<'db> {
    /// String allocations: (content, offset, length) - for active data segments.
    #[returns(ref)]
    pub string_allocations: Vec<(Vec<u8>, u32, u32)>,
    /// Bytes allocations: (content, data_idx, length) - for passive data segments.
    /// data_idx is the index into the passive data segment array.
    #[returns(ref)]
    pub bytes_allocations: Vec<(Vec<u8>, u32, u32)>,
    /// Total size of string data segments (for linear memory).
    pub string_total_size: u32,
}

impl<'db> ConstAnalysis<'db> {
    /// Legacy accessor for backwards compatibility.
    /// Returns string allocations only.
    pub fn allocations(&self, db: &'db dyn salsa::Database) -> &[(Vec<u8>, u32, u32)] {
        self.string_allocations(db)
    }

    /// Legacy accessor for backwards compatibility.
    pub fn total_size(&self, db: &'db dyn salsa::Database) -> u32 {
        self.string_total_size(db)
    }
}

impl<'db> ConstAnalysis<'db> {
    /// Look up the offset for given string content.
    /// Returns (offset, length).
    pub fn offset_for(&self, db: &'db dyn salsa::Database, content: &[u8]) -> Option<(u32, u32)> {
        self.string_allocations(db)
            .iter()
            .find(|(data, _, _)| data.as_slice() == content)
            .map(|(_, offset, len)| (*offset, *len))
    }

    /// Look up the bytes allocation info for given content.
    /// Returns (data_idx, 0, length) where data_idx is the passive data segment index.
    pub fn bytes_info_for(
        &self,
        db: &'db dyn salsa::Database,
        content: &[u8],
    ) -> Option<(u32, u32, u32)> {
        self.bytes_allocations(db)
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

    fn visit_op<'db>(&mut self, db: &'db dyn salsa::Database, op: &Operation<'db>) {
        let dialect = op.dialect(db);
        let name = op.name(db);

        if dialect == adt::DIALECT_NAME() {
            let attrs = op.attributes(db);

            if name == adt::STRING_CONST()
                && let Some(Attribute::String(s)) = attrs.get(&Symbol::new("value"))
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
            } else if name == adt::BYTES_CONST()
                && let Some(Attribute::Bytes(b)) = attrs.get(&Symbol::new("value"))
            {
                let bytes = b.clone();
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

        // Recurse into regions
        for region in op.regions(db).iter() {
            for block in region.blocks(db).iter() {
                for nested_op in block.operations(db).iter() {
                    self.visit_op(db, nested_op);
                }
            }
        }
    }
}

/// Analyze a module to collect all string/bytes constants and allocate offsets.
#[salsa::tracked]
pub fn analyze_consts<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> ConstAnalysis<'db> {
    let mut collector = ConstCollector::new();

    // Walk all operations in module body
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            collector.visit_op(db, op);
        }
    }

    ConstAnalysis::new(
        db,
        collector.string_allocations,
        collector.bytes_allocations,
        collector.next_string_offset,
    )
}

/// Lower const operations using pre-computed analysis.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    analysis: ConstAnalysis<'db>,
) -> Module<'db> {
    // Extract allocations data from salsa-tracked struct for use in 'static patterns
    let string_allocations = analysis.string_allocations(db).clone();
    let bytes_allocations = analysis.bytes_allocations(db).clone();

    // No specific conversion target - const lowering is a dialect transformation
    let target = ConversionTarget::new();
    PatternApplicator::new(wasm_type_converter())
        .add_pattern(StringConstPattern::new(string_allocations))
        .add_pattern(BytesConstPattern::new(bytes_allocations))
        .apply_partial(db, module, target)
        .module
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

impl<'db> RewritePattern<'db> for StringConstPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        let Ok(string_const) = adt::StringConst::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let content = string_const.value(db).clone().into_bytes();

        let Some((offset, _len)) = lookup_offset(&self.allocations, &content) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        // Use typed helper to create wasm.i32_const with just the offset.
        // Length information is available in ConstAnalysis and will be used by emit.rs.
        let new_op = wasm::i32_const(db, location, i32_ty, offset as i32).as_operation();

        RewriteResult::Replace(new_op)
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

impl<'db> RewritePattern<'db> for BytesConstPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
        _adaptor: &OpAdaptor<'a, '_>,
    ) -> RewriteResult<'a> {
        let Ok(bytes_const) = adt::BytesConst::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let Attribute::Bytes(value) = bytes_const.value(db) else {
            return RewriteResult::Unchanged;
        };

        let content = value.clone();

        let Some((data_idx, len)) = lookup_bytes_info(&self.allocations, &content) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let bytes_ty = core::Bytes::new(db).as_type();

        // Create wasm.bytes_from_data operation
        let new_op = wasm::bytes_from_data(db, location, bytes_ty, data_idx, 0, len).as_operation();

        RewriteResult::Replace(new_op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{Block, BlockId, Location, PathId, Region, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_string_const_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let ptr_ty = core::I32::new(db).as_type();

        let string_const =
            adt::string_const(db, location, ptr_ty, "hello".to_string()).as_operation();

        let block = Block::new(
            db,
            BlockId::fresh(),
            location,
            idvec![],
            idvec![string_const],
        );
        let region = Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_string_const_analysis(db: &salsa::DatabaseImpl) {
        let module = make_string_const_module(db);
        let analysis = analyze_consts(db, module);

        assert_eq!(analysis.allocations(db).len(), 1);
        assert_eq!(analysis.allocations(db)[0].0, b"hello".to_vec());
        assert_eq!(analysis.allocations(db)[0].2, 5); // length
    }

    #[salsa::tracked]
    fn lower_and_check(db: &dyn salsa::Database, module: Module<'_>) -> Vec<String> {
        let analysis = analyze_consts(db, module);
        let lowered = lower(db, module, analysis);
        let body = lowered.body(db);
        let ops = body.blocks(db)[0].operations(db);
        ops.iter().map(|op| op.full_name(db)).collect()
    }

    #[salsa_test]
    fn test_string_const_to_wasm(db: &salsa::DatabaseImpl) {
        let module = make_string_const_module(db);
        let op_names = lower_and_check(db, module);

        assert_eq!(op_names.len(), 1);
        assert_eq!(op_names[0], "wasm.i32_const");
    }
}
