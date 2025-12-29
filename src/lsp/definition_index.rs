//! Definition index for Go to Definition functionality.
//!
//! Builds a mapping from:
//! 1. Definition names to their declaration locations (function name → function def location)
//! 2. Usage locations to definition names (reference span → what it refers to)

use trunk_ir::Span;
use trunk_ir::dialect::core::Module;
use trunk_ir::{Attribute, Block, Operation, Region, Symbol};

/// Entry representing a definition location.
#[derive(Clone, Debug)]
pub struct DefinitionEntry {
    /// Name of the defined symbol.
    pub name: Symbol,
    /// Span of the definition (typically the name span).
    pub span: Span,
    /// Kind of definition.
    pub kind: DefinitionKind,
}

/// Kind of definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefinitionKind {
    /// A function definition.
    Function,
    /// A type definition (struct/enum).
    Type,
    /// An ability definition.
    Ability,
}

/// Entry representing a reference (usage) location.
#[derive(Clone, Debug)]
pub struct ReferenceEntry {
    /// Span of the reference in source.
    pub span: Span,
    /// Name being referenced.
    pub target: Symbol,
}

/// Index for Go to Definition lookups.
pub struct DefinitionIndex {
    /// All definitions in the module.
    definitions: Vec<DefinitionEntry>,
    /// All references (usages) that can be jumped from.
    references: Vec<ReferenceEntry>,
}

impl DefinitionIndex {
    /// Build a definition index from a compiled module.
    pub fn build(db: &dyn salsa::Database, module: &Module<'_>) -> Self {
        let mut definitions = Vec::new();
        let mut references = Vec::new();

        Self::collect_from_region(db, &module.body(db), &mut definitions, &mut references);

        // Sort for efficient lookup
        definitions.sort_by_key(|e| (e.span.start, e.span.end));
        references.sort_by_key(|e| (e.span.start, e.span.end));

        Self {
            definitions,
            references,
        }
    }

    fn collect_from_region(
        db: &dyn salsa::Database,
        region: &Region<'_>,
        definitions: &mut Vec<DefinitionEntry>,
        references: &mut Vec<ReferenceEntry>,
    ) {
        for block in region.blocks(db).iter() {
            Self::collect_from_block(db, block, definitions, references);
        }
    }

    fn collect_from_block(
        db: &dyn salsa::Database,
        block: &Block<'_>,
        definitions: &mut Vec<DefinitionEntry>,
        references: &mut Vec<ReferenceEntry>,
    ) {
        for op in block.operations(db).iter() {
            Self::collect_from_op(db, op, definitions, references);
        }
    }

    fn collect_from_op(
        db: &dyn salsa::Database,
        op: &Operation<'_>,
        definitions: &mut Vec<DefinitionEntry>,
        references: &mut Vec<ReferenceEntry>,
    ) {
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::{func, src, ty};

        let attrs = op.attributes(db);
        let op_span = op.location(db).span;

        // Helper to get name_span attribute from attrs
        let name_span = attrs.get(&Symbol::new("name_span")).and_then(|a| match a {
            Attribute::Span(s) => Some(*s),
            _ => None,
        });

        // Collect definitions using typed wrappers
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            // func.func - function definition
            let qname = func_op.sym_name(db);
            let name = qname.name();
            let span = name_span.unwrap_or(op_span);

            definitions.push(DefinitionEntry {
                name,
                span,
                kind: DefinitionKind::Function,
            });
        } else if let Ok(struct_op) = ty::Struct::from_operation(db, *op) {
            // ty.struct - struct type definition
            let name = struct_op.name(db);
            let span = name_span.unwrap_or(op_span);

            definitions.push(DefinitionEntry {
                name,
                span,
                kind: DefinitionKind::Type,
            });
        } else if let Ok(enum_op) = ty::Enum::from_operation(db, *op) {
            // ty.enum - enum type definition
            let name = enum_op.name(db);
            let span = name_span.unwrap_or(op_span);

            definitions.push(DefinitionEntry {
                name,
                span,
                kind: DefinitionKind::Type,
            });
        } else if let Ok(ability_op) = ty::Ability::from_operation(db, *op) {
            // ty.ability - ability definition
            let name = ability_op.name(db);
            let span = name_span.unwrap_or(op_span);

            definitions.push(DefinitionEntry {
                name,
                span,
                kind: DefinitionKind::Ability,
            });
        }

        // Collect references using typed wrappers
        if let Ok(call_op) = func::Call::from_operation(db, *op) {
            // func.call - function call (resolved reference)
            let callee = call_op.callee(db);
            references.push(ReferenceEntry {
                span: op_span,
                target: callee.name(),
            });
        } else if let Ok(var_op) = src::Var::from_operation(db, *op) {
            // src.var - variable reference (before resolution)
            let name = var_op.name(db);
            references.push(ReferenceEntry {
                span: op_span,
                target: name,
            });
        } else if let Ok(path_op) = src::Path::from_operation(db, *op) {
            // src.path - qualified path reference (before resolution)
            let path = path_op.path(db);
            references.push(ReferenceEntry {
                span: op_span,
                target: path.name(),
            });
        }

        // Recurse into nested regions
        for region in op.regions(db).iter() {
            Self::collect_from_region(db, region, definitions, references);
        }
    }

    /// Find a reference at the given byte offset.
    pub fn reference_at(&self, offset: usize) -> Option<&ReferenceEntry> {
        self.references
            .iter()
            .filter(|e| e.span.start <= offset && offset < e.span.end)
            .min_by_key(|e| e.span.end - e.span.start)
    }

    /// Find the definition of a symbol by name.
    pub fn definition_of(&self, name: Symbol) -> Option<&DefinitionEntry> {
        self.definitions.iter().find(|e| e.name == name)
    }

    /// Get the definition location for a reference at the given offset.
    pub fn definition_at(&self, offset: usize) -> Option<&DefinitionEntry> {
        let reference = self.reference_at(offset)?;
        self.definition_of(reference.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tree_sitter::Parser;
    use tribute::SourceCst;
    use tribute::stage_lower_case;

    fn make_source(path: &str, text: &str) -> SourceCst {
        salsa::with_attached_database(|db| {
            let mut parser = Parser::new();
            parser
                .set_language(&tree_sitter_tribute::LANGUAGE.into())
                .expect("Failed to set language");
            let tree = parser.parse(text, None).expect("tree");
            SourceCst::from_path(db, path, text.into(), Some(tree))
        })
        .expect("attached db")
    }

    #[salsa_test]
    fn test_definition_index_basic(db: &salsa::DatabaseImpl) {
        //                    0         1         2         3
        //                    0123456789012345678901234567890123456789
        let source_text = "fn add(x: Int, y: Int) -> Int { x + y }";
        let source = make_source("test.trb", source_text);

        let module = stage_lower_case(db, source);
        let index = DefinitionIndex::build(db, &module);

        // Should find the function definition "add"
        let def = index.definition_of(Symbol::new("add"));
        assert!(def.is_some(), "Should find function 'add'");
        assert_eq!(def.unwrap().kind, DefinitionKind::Function);
    }

    #[salsa_test]
    fn test_goto_definition(db: &salsa::DatabaseImpl) {
        //                    0         1         2         3         4         5
        //                    0123456789012345678901234567890123456789012345678901234567890
        let source_text = "fn foo(x: Int) -> Int { x }\nfn bar() -> Int { foo(1) }";
        let source = make_source("test.trb", source_text);

        let module = stage_lower_case(db, source);
        let index = DefinitionIndex::build(db, &module);

        // Find position of "foo" call in bar (around position 46)
        let foo_call_pos = source_text.rfind("foo").unwrap();

        // Should be able to go from the call to the definition
        let def = index.definition_at(foo_call_pos);
        assert!(
            def.is_some(),
            "Should find definition from 'foo' call at position {}",
            foo_call_pos
        );
        if let Some(d) = def {
            assert_eq!(d.name, Symbol::new("foo"));
            assert_eq!(d.kind, DefinitionKind::Function);
        }
    }
}
