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
    /// A local variable (let binding).
    Variable,
    /// A function parameter (for future use).
    #[allow(dead_code)]
    Parameter,
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

        // Sort for deterministic iteration order (by span position)
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
        use tribute_ir::dialect::{tribute, tribute_pat};
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::func;

        let attrs = op.attributes(db);
        let op_span = op.location(db).span;

        // Helper to get name_location attribute from attrs
        let name_span = attrs
            .get(&Symbol::new("name_location"))
            .and_then(|a| match a {
                Attribute::Location(loc) => Some(loc.span),
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
        } else if let Ok(struct_op) = tribute::StructDef::from_operation(db, *op) {
            // tribute.struct_def - struct type definition
            let name = struct_op.sym_name(db);
            let span = name_span.unwrap_or(op_span);
            definitions.push(DefinitionEntry {
                name,
                span,
                kind: DefinitionKind::Type,
            });
        } else if let Ok(enum_op) = tribute::EnumDef::from_operation(db, *op) {
            // tribute.enum_def - enum type definition
            let name = enum_op.sym_name(db);
            let span = name_span.unwrap_or(op_span);
            definitions.push(DefinitionEntry {
                name,
                span,
                kind: DefinitionKind::Type,
            });
        } else if let Ok(ability_op) = tribute::AbilityDef::from_operation(db, *op) {
            // tribute.ability_def - ability definition
            let name = ability_op.sym_name(db);
            let span = name_span.unwrap_or(op_span);
            definitions.push(DefinitionEntry {
                name,
                span,
                kind: DefinitionKind::Ability,
            });
        } else if let Ok(bind_op) = tribute_pat::Bind::from_operation(db, *op) {
            // tribute_pat.bind - pattern binding (variable definition)
            let name = bind_op.name(db);
            definitions.push(DefinitionEntry {
                name,
                span: op_span,
                kind: DefinitionKind::Variable,
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
        } else if let Ok(var_op) = tribute::Var::from_operation(db, *op) {
            // tribute.var - variable reference (before resolution)
            let name = var_op.name(db);
            references.push(ReferenceEntry {
                span: op_span,
                target: name,
            });
        } else if let Ok(path_op) = tribute::Path::from_operation(db, *op) {
            // tribute.path - qualified path reference (before resolution)
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

    /// Find a definition at the given byte offset.
    pub fn definition_at_position(&self, offset: usize) -> Option<&DefinitionEntry> {
        self.definitions
            .iter()
            .filter(|e| e.span.start <= offset && offset < e.span.end)
            .min_by_key(|e| e.span.end - e.span.start)
    }

    /// Find all references to a given symbol.
    pub fn references_of(&self, name: Symbol) -> Vec<&ReferenceEntry> {
        self.references
            .iter()
            .filter(|e| e.target == name)
            .collect()
    }

    /// Find all references from a position.
    ///
    /// If the position is on a definition, returns all references to that symbol.
    /// If the position is on a reference, finds the target symbol and returns all references to it.
    ///
    /// Returns the symbol name and a list of all references to it.
    pub fn references_at(&self, offset: usize) -> Option<(Symbol, Vec<&ReferenceEntry>)> {
        // First check if we're on a definition
        if let Some(def) = self.definition_at_position(offset) {
            let refs = self.references_of(def.name);
            return Some((def.name, refs));
        }

        // Otherwise check if we're on a reference
        if let Some(reference) = self.reference_at(offset) {
            let refs = self.references_of(reference.target);
            return Some((reference.target, refs));
        }

        None
    }

    /// Check if the symbol at the given offset can be renamed.
    /// Returns the definition entry and the span to highlight if renameable.
    pub fn can_rename(&self, offset: usize) -> Option<(&DefinitionEntry, Span)> {
        // First check if we're on a definition
        if let Some(def) = self.definition_at_position(offset) {
            return Some((def, def.span));
        }

        // Otherwise check if we're on a reference
        if let Some(reference) = self.reference_at(offset)
            && let Some(def) = self.definition_of(reference.target)
        {
            return Some((def, reference.span));
        }

        None
    }

    /// Get all definitions in the module.
    pub fn definitions(&self) -> &[DefinitionEntry] {
        &self.definitions
    }
}

/// Reserved keywords in Tribute.
/// Based on `contrib/zed/grammars/tribute/grammar.js`.
pub const KEYWORDS: &[&str] = &[
    "fn", "let", "case", "struct", "enum", "ability", "const", "pub", "use", "mod", "if", "handle",
    "as", "True", "False", "Nil",
];

/// Check if a name is a reserved keyword.
pub fn is_keyword(name: &str) -> bool {
    KEYWORDS.contains(&name)
}

/// Error type for rename validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenameError {
    /// The new name is empty.
    EmptyName,
    /// The new name is not a valid identifier.
    InvalidIdentifier,
    /// The new name is not a valid type identifier (must start with uppercase).
    InvalidTypeIdentifier,
    /// The new name contains invalid characters.
    InvalidCharacter,
    /// The new name is a reserved keyword.
    ReservedKeyword,
}

impl std::fmt::Display for RenameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenameError::EmptyName => write!(f, "Name cannot be empty"),
            RenameError::InvalidIdentifier => {
                write!(
                    f,
                    "Identifier must start with lowercase letter or underscore"
                )
            }
            RenameError::InvalidTypeIdentifier => {
                write!(f, "Type identifier must start with uppercase letter")
            }
            RenameError::InvalidCharacter => {
                write!(f, "Name contains invalid characters")
            }
            RenameError::ReservedKeyword => write!(f, "Name is a reserved keyword"),
        }
    }
}

impl std::error::Error for RenameError {}

/// Validates if a string is a valid Tribute identifier.
///
/// Rules:
/// - Regular identifiers: `[a-z_][a-zA-Z0-9_]*`
/// - Type identifiers: `[A-Z][a-zA-Z0-9_]*`
/// - Cannot be a reserved keyword
pub fn validate_identifier(name: &str, kind: DefinitionKind) -> Result<(), RenameError> {
    // Check empty
    if name.is_empty() {
        return Err(RenameError::EmptyName);
    }

    let first = name.chars().next().unwrap();

    // Check pattern based on kind
    match kind {
        DefinitionKind::Type | DefinitionKind::Ability => {
            // Type identifiers start with uppercase
            if !first.is_ascii_uppercase() {
                return Err(RenameError::InvalidTypeIdentifier);
            }
        }
        _ => {
            // Regular identifiers start with lowercase or underscore
            if !first.is_ascii_lowercase() && first != '_' {
                return Err(RenameError::InvalidIdentifier);
            }
        }
    }

    // Check all characters are alphanumeric or underscore
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(RenameError::InvalidCharacter);
    }

    // Check for reserved keywords
    if is_keyword(name) {
        return Err(RenameError::ReservedKeyword);
    }

    Ok(())
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

    #[salsa_test]
    fn test_find_references(db: &salsa::DatabaseImpl) {
        //                    0         1         2         3         4         5         6
        //                    0123456789012345678901234567890123456789012345678901234567890123456789
        let source_text = "fn foo(x: Int) -> Int { x }\nfn bar() -> Int { foo(1) + foo(2) }";
        let source = make_source("test.trb", source_text);

        let module = stage_lower_case(db, source);
        let index = DefinitionIndex::build(db, &module);

        // Find position of "foo" definition
        let foo_def_pos = source_text.find("foo").unwrap();

        // Should find all references from the definition
        let result = index.references_at(foo_def_pos);
        assert!(result.is_some(), "Should find references from definition");

        let (symbol, refs) = result.unwrap();
        assert_eq!(symbol, Symbol::new("foo"));
        assert_eq!(refs.len(), 2, "Should find 2 call sites");
    }

    #[salsa_test]
    fn test_find_references_from_call_site(db: &salsa::DatabaseImpl) {
        //                    0         1         2         3         4         5         6
        //                    0123456789012345678901234567890123456789012345678901234567890123456789
        let source_text = "fn foo(x: Int) -> Int { x }\nfn bar() -> Int { foo(1) + foo(2) }";
        let source = make_source("test.trb", source_text);

        let module = stage_lower_case(db, source);
        let index = DefinitionIndex::build(db, &module);

        // Find position of second "foo" call
        let foo_call_pos = source_text.rfind("foo").unwrap();

        // Should find all references from the call site too
        let result = index.references_at(foo_call_pos);
        assert!(result.is_some(), "Should find references from call site");

        let (symbol, refs) = result.unwrap();
        assert_eq!(symbol, Symbol::new("foo"));
        assert_eq!(refs.len(), 2, "Should find 2 call sites");
    }

    #[salsa_test]
    fn test_can_rename_function(db: &salsa::DatabaseImpl) {
        let source_text = "fn foo() -> Int { 42 }";
        let source = make_source("test.trb", source_text);

        let module = stage_lower_case(db, source);
        let index = DefinitionIndex::build(db, &module);

        let foo_pos = source_text.find("foo").unwrap();
        let result = index.can_rename(foo_pos);

        assert!(result.is_some(), "Function should be renameable");
        let (def, _) = result.unwrap();
        assert_eq!(def.kind, DefinitionKind::Function);
        assert_eq!(def.name, Symbol::new("foo"));
    }

    #[salsa_test]
    fn test_can_rename_from_reference(db: &salsa::DatabaseImpl) {
        let source_text = "fn foo() -> Int { 1 }\nfn bar() -> Int { foo() }";
        let source = make_source("test.trb", source_text);

        let module = stage_lower_case(db, source);
        let index = DefinitionIndex::build(db, &module);

        // Position of "foo" call in bar
        let foo_call_pos = source_text.rfind("foo").unwrap();
        let result = index.can_rename(foo_call_pos);

        assert!(result.is_some(), "Should be able to rename from reference");
        let (def, _) = result.unwrap();
        assert_eq!(def.name, Symbol::new("foo"));
    }

    #[test]
    fn test_validate_identifier_valid() {
        // Valid function/variable identifiers
        assert!(validate_identifier("foo", DefinitionKind::Function).is_ok());
        assert!(validate_identifier("_bar", DefinitionKind::Variable).is_ok());
        assert!(validate_identifier("foo_bar123", DefinitionKind::Function).is_ok());

        // Valid type identifiers
        assert!(validate_identifier("Foo", DefinitionKind::Type).is_ok());
        assert!(validate_identifier("FooBar", DefinitionKind::Ability).is_ok());
        assert!(validate_identifier("MyType123", DefinitionKind::Type).is_ok());
    }

    #[test]
    fn test_validate_identifier_invalid() {
        // Empty name
        assert_eq!(
            validate_identifier("", DefinitionKind::Function),
            Err(RenameError::EmptyName)
        );

        // Wrong case for type
        assert_eq!(
            validate_identifier("foo", DefinitionKind::Type),
            Err(RenameError::InvalidTypeIdentifier)
        );

        // Wrong case for function (starts with uppercase)
        assert_eq!(
            validate_identifier("Foo", DefinitionKind::Function),
            Err(RenameError::InvalidIdentifier)
        );

        // Invalid characters
        assert_eq!(
            validate_identifier("foo bar", DefinitionKind::Function),
            Err(RenameError::InvalidCharacter)
        );

        // Reserved keyword
        assert_eq!(
            validate_identifier("fn", DefinitionKind::Function),
            Err(RenameError::ReservedKeyword)
        );
        assert_eq!(
            validate_identifier("let", DefinitionKind::Variable),
            Err(RenameError::ReservedKeyword)
        );
    }
}
