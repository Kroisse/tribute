//! Lower adt.string_const and adt.bytes_const to wasm data segments.
//!
//! This pass uses a two-phase approach:
//! 1. Analysis: Collect all string/bytes constants and allocate data segment offsets
//! 2. Transform: Replace const operations with wasm.i32_const (pointer to data)
//!
//! The analysis is implemented as a salsa::tracked function for incremental computation.

use std::collections::HashMap;

use trunk_ir::dialect::adt;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::rewrite::{PatternApplicator, RewritePattern, RewriteResult};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Symbol};

/// Result of const analysis - maps content to allocated offset.
#[salsa::tracked]
pub struct ConstAnalysis<'db> {
    /// Map from content bytes to (offset, length).
    /// Using Vec<(Vec<u8>, u32, u32)> instead of HashMap for salsa compatibility.
    #[returns(ref)]
    pub allocations: Vec<(Vec<u8>, u32, u32)>,
    /// Total size of all data segments.
    pub total_size: u32,
}

impl<'db> ConstAnalysis<'db> {
    /// Look up the offset for given content bytes.
    pub fn offset_for(&self, db: &'db dyn salsa::Database, content: &[u8]) -> Option<(u32, u32)> {
        self.allocations(db)
            .iter()
            .find(|(data, _, _)| data.as_slice() == content)
            .map(|(_, offset, len)| (*offset, *len))
    }
}

/// Analyze a module to collect all string/bytes constants and allocate offsets.
#[salsa::tracked]
pub fn analyze_consts<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> ConstAnalysis<'db> {
    let mut allocations: Vec<(Vec<u8>, u32, u32)> = Vec::new();
    let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut next_offset: u32 = 0;

    // Helper to align offset
    fn align_to(value: u32, align: u32) -> u32 {
        if align == 0 {
            return value;
        }
        value.div_ceil(align) * align
    }

    // Visitor to collect const operations
    fn visit_op<'db>(
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        allocations: &mut Vec<(Vec<u8>, u32, u32)>,
        seen: &mut HashMap<Vec<u8>, usize>,
        next_offset: &mut u32,
    ) {
        let dialect = op.dialect(db);
        let name = op.name(db);

        if dialect == adt::DIALECT_NAME() {
            let attrs = op.attributes(db);

            let content: Option<Vec<u8>> = if name == adt::STRING_CONST() {
                attrs
                    .get(&Symbol::new("value"))
                    .and_then(|attr| match attr {
                        Attribute::String(s) => Some(s.clone().into_bytes()),
                        _ => None,
                    })
            } else if name == adt::BYTES_CONST() {
                attrs
                    .get(&Symbol::new("value"))
                    .and_then(|attr| match attr {
                        Attribute::Bytes(b) => Some(b.clone()),
                        _ => None,
                    })
            } else {
                None
            };

            if let Some(bytes) = content {
                // Deduplicate identical content
                if !seen.contains_key(&bytes) {
                    let offset = align_to(*next_offset, 4);
                    let len = bytes.len() as u32;
                    seen.insert(bytes.clone(), allocations.len());
                    allocations.push((bytes, offset, len));
                    *next_offset = offset + len;
                }
            }
        }

        // Recurse into regions
        for region in op.regions(db).iter() {
            for block in region.blocks(db).iter() {
                for nested_op in block.operations(db).iter() {
                    visit_op(db, nested_op, allocations, seen, next_offset);
                }
            }
        }
    }

    // Walk all operations in module body
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            visit_op(db, op, &mut allocations, &mut seen, &mut next_offset);
        }
    }

    ConstAnalysis::new(db, allocations, next_offset)
}

/// Lower const operations using pre-computed analysis.
pub fn lower<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    analysis: ConstAnalysis<'db>,
) -> Module<'db> {
    // Extract allocations data from salsa-tracked struct for use in 'static patterns
    let allocations = analysis.allocations(db).clone();

    PatternApplicator::new()
        .add_pattern(StringConstPattern::new(allocations.clone()))
        .add_pattern(BytesConstPattern::new(allocations))
        .apply(db, module)
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

impl RewritePattern for StringConstPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
    ) -> RewriteResult<'a> {
        let Ok(string_const) = adt::StringConst::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let content = string_const.value(db).clone().into_bytes();

        let Some((offset, len)) = lookup_offset(&self.allocations, &content) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        let new_op = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(u64::from(offset)))
            .attr("literal_len", Attribute::IntBits(u64::from(len)))
            .results(trunk_ir::idvec![i32_ty])
            .build();

        RewriteResult::Replace(new_op)
    }
}

/// Pattern for `adt.bytes_const` -> `wasm.i32_const`
struct BytesConstPattern {
    allocations: Allocations,
}

impl BytesConstPattern {
    fn new(allocations: Allocations) -> Self {
        Self { allocations }
    }
}

impl RewritePattern for BytesConstPattern {
    fn match_and_rewrite<'a>(
        &self,
        db: &'a dyn salsa::Database,
        op: &Operation<'a>,
    ) -> RewriteResult<'a> {
        let Ok(bytes_const) = adt::BytesConst::from_operation(db, *op) else {
            return RewriteResult::Unchanged;
        };

        let Attribute::Bytes(value) = bytes_const.value(db) else {
            return RewriteResult::Unchanged;
        };

        let content = value.clone();

        let Some((offset, len)) = lookup_offset(&self.allocations, &content) else {
            return RewriteResult::Unchanged;
        };

        let location = op.location(db);
        let i32_ty = core::I32::new(db).as_type();

        let new_op = Operation::of_name(db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(u64::from(offset)))
            .attr("literal_len", Attribute::IntBits(u64::from(len)))
            .results(trunk_ir::idvec![i32_ty])
            .build();

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

        let string_const = Operation::of_name(db, location, "adt.string_const")
            .attr("value", Attribute::String("hello".into()))
            .results(idvec![ptr_ty])
            .build();

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
