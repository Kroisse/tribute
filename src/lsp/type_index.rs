//! Type index for looking up types at source positions.
//!
//! Builds an index from byte positions to types by walking the IR.

use trunk_ir::Span;
use trunk_ir::dialect::core::Module;
use trunk_ir::{Block, Operation, Region, Type};

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
        use trunk_ir::Attribute;

        let span = op.location(db).span;
        let dialect = op.dialect(db);
        let name = op.name(db);

        // Determine the kind based on operation type
        let kind = if dialect == "func" && name == "func" {
            EntryKind::Function
        } else {
            EntryKind::Expression
        };

        // Special case for func.func: use the 'type' attribute since it has no results
        // and use 'name_span' if available for precise hover targeting
        if dialect == "func" && name == "func" {
            let type_key = trunk_ir::Symbol::new("type");
            let name_span_key = trunk_ir::Symbol::new("name_span");

            if let Some(Attribute::Type(func_ty)) = op.attributes(db).get(&type_key) {
                // Use name_span if available, otherwise fall back to operation span
                let hover_span = match op.attributes(db).get(&name_span_key) {
                    Some(Attribute::Span(s)) => *s,
                    _ => span,
                };
                entries.push(TypeEntry {
                    span: hover_span,
                    ty: *func_ty,
                    kind,
                });
            }
        }

        // Only enable hover for source-level reference operations
        let is_hoverable = dialect == "src" && (name == "var" || name == "path");
        if is_hoverable {
            for &result_ty in op.results(db).iter() {
                entries.push(TypeEntry {
                    span,
                    ty: result_ty,
                    kind: EntryKind::Expression,
                });
            }
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
    use tree_sitter::Parser;
    use tribute::compile;
    use tribute::{SourceCst, TributeDatabaseImpl};

    #[test]
    fn test_type_index_basic() {
        TributeDatabaseImpl::default().attach(|db| {
            //                    0         1         2         3
            //                    0123456789012345678901234567890123456789
            let source_text = "fn add(x: Int, y: Int) -> Int { x + y }";
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_tribute::LANGUAGE.into())
                .expect("Failed to set language");
            let tree = parser.parse(source_text, None).expect("tree");
            let source = SourceCst::from_path(db, "test.trb", source_text.into(), Some(tree));

            let module = compile(db, source);
            let index = TypeIndex::build(db, &module);

            // Should find the function type at the function name "add" (position 3-6)
            assert!(index.type_at(3).is_some());
            // Should NOT find function type at "fn" keyword (position 0)
            assert!(index.type_at(0).is_none());
        });
    }

    #[test]
    fn test_type_index_local_var() {
        use crate::lsp::pretty::print_type;

        TributeDatabaseImpl::default().attach(|db| {
            //                    0         1         2         3
            //                    0123456789012345678901234567890123456
            let source_text = "fn foo(a: Int) -> Int { a }";
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_tribute::LANGUAGE.into())
                .expect("Failed to set language");
            let tree = parser.parse(source_text, None).expect("tree");
            let source = SourceCst::from_path(db, "test.trb", source_text.into(), Some(tree));

            let module = compile(db, source);
            let index = TypeIndex::build(db, &module);

            // Position of 'a' in body "{ a }" should show type Int
            let a_pos = source_text.find("{ a }").unwrap() + 2;

            let entry = index
                .type_at(a_pos)
                .expect("Should find type for local variable 'a'");
            let ty_str = print_type(db, entry.ty);
            assert_eq!(
                ty_str, "Int",
                "Expected Int for variable 'a', got {}",
                ty_str
            );

            // Function name hover should also work (position 3 = "foo")
            assert!(
                index.type_at(3).is_some(),
                "Function name hover should work"
            );
        });
    }
}
