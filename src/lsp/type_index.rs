//! Type index for looking up types at source positions.
//!
//! Builds an index from byte positions to types by walking the IR.

use tribute_core::Span;
use tribute_trunk_ir::dialect::core::Module;
use tribute_trunk_ir::{Block, Operation, Region, Type};

/// Entry in the type index.
#[derive(Clone, Debug)]
pub struct TypeEntry<'db> {
    /// Source span of this entry.
    pub span: Span,
    /// Type at this span.
    pub ty: Type<'db>,
    /// Kind of entry (for display purposes).
    #[allow(dead_code)]
    pub kind: EntryKind,
}

/// Kind of type entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum EntryKind {
    /// Function parameter or block argument.
    Parameter,
    /// Expression result.
    Expression,
    /// Function definition.
    Function,
}

/// Index mapping source positions to type information.
pub struct TypeIndex<'db> {
    /// Sorted list of (span, type, kind) entries.
    entries: Vec<TypeEntry<'db>>,
}

impl<'db> TypeIndex<'db> {
    /// Build a type index from a compiled module.
    pub fn build(db: &'db dyn salsa::Database, module: &Module<'db>) -> Self {
        let mut entries = Vec::new();
        Self::collect_from_region(db, &module.body(db), &mut entries);

        // Sort by span start for efficient lookup
        entries.sort_by_key(|e| (e.span.start, e.span.end));

        Self { entries }
    }

    fn collect_from_region(
        db: &'db dyn salsa::Database,
        region: &Region<'db>,
        entries: &mut Vec<TypeEntry<'db>>,
    ) {
        for block in region.blocks(db).iter() {
            Self::collect_from_block(db, block, entries);
        }
    }

    fn collect_from_block(
        db: &'db dyn salsa::Database,
        block: &Block<'db>,
        entries: &mut Vec<TypeEntry<'db>>,
    ) {
        // Note: Block arguments (function parameters) are intentionally not added here
        // because we don't have accurate individual spans for them. Adding them with
        // the block's span would cause the entire function to be highlighted on hover.
        // Individual operations have their own spans and will be found properly.

        // Operations
        for op in block.operations(db).iter() {
            Self::collect_from_op(db, op, entries);
        }
    }

    fn collect_from_op(
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        entries: &mut Vec<TypeEntry<'db>>,
    ) {
        use tribute_trunk_ir::Attribute;

        let span = op.location(db).span;
        let dialect = op.dialect(db).text(db);
        let name = op.name(db).text(db);

        // Determine the kind based on operation type
        let kind = if dialect == "func" && name == "func" {
            EntryKind::Function
        } else {
            EntryKind::Expression
        };

        // Special case for func.func: use the 'type' attribute since it has no results
        if dialect == "func" && name == "func" {
            let type_key = tribute_trunk_ir::Symbol::new(db, "type");
            if let Some(Attribute::Type(func_ty)) = op.attributes(db).get(&type_key) {
                entries.push(TypeEntry {
                    span,
                    ty: *func_ty,
                    kind,
                });
            }
        }

        // Add entries for each result type
        for &result_ty in op.results(db).iter() {
            entries.push(TypeEntry {
                span,
                ty: result_ty,
                kind,
            });
        }

        // Recurse into nested regions
        for region in op.regions(db).iter() {
            Self::collect_from_region(db, region, entries);
        }
    }

    /// Find the type at a given byte offset.
    ///
    /// Returns the innermost (most specific) type entry containing the offset.
    pub fn type_at(&self, offset: usize) -> Option<&TypeEntry<'db>> {
        // Find all entries containing this offset
        let containing: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.span.start <= offset && offset < e.span.end)
            .collect();

        // Return the innermost (smallest span)
        containing
            .into_iter()
            .min_by_key(|e| e.span.end - e.span.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa::prelude::*;
    use tribute_core::{SourceFile, TributeDatabaseImpl};
    use tribute_passes::compile;

    #[test]
    fn test_type_index_basic() {
        TributeDatabaseImpl::default().attach(|db| {
            let source = SourceFile::new(
                db,
                std::path::PathBuf::from("test.tr"),
                "fn add(x: Int, y: Int) -> Int { x + y }".to_string(),
            );

            let module = compile(db, source);
            let index = TypeIndex::build(db, &module);

            // Should find the function type
            assert!(index.type_at(0).is_some());
        });
    }
}
