//! Completion index for auto-completion functionality.
//!
//! Builds completion candidates from the compiled module.

use tribute::SourceCst;
use trunk_ir::Symbol;

use super::definition_index::{DefinitionKind, KEYWORDS};

/// Entry representing a completion candidate.
#[derive(Clone, Debug, Eq, Hash, PartialEq, salsa::Update)]
pub struct CompletionEntry {
    pub name: Symbol,
    pub kind: CompletionKind,
    pub detail: Option<String>,
}

/// Kind of completion item.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompletionKind {
    Function,
    Constructor,
    Keyword,
    Ability,
}

/// Index for completion lookups.
#[salsa::tracked]
pub struct CompletionIndex<'db> {
    /// All completion entries from module definitions.
    #[returns(deref)]
    entries: Vec<CompletionEntry>,
}

/// Build a completion index from a source file.
#[salsa::tracked]
pub fn build<'db>(db: &'db dyn salsa::Database, source_cst: SourceCst) -> CompletionIndex<'db> {
    let def_index = super::definition_index::build(db, source_cst);
    let mut entries = Vec::new();

    // Add all definitions as completion candidates
    for def in def_index.definitions(db) {
        let kind = match def.kind {
            DefinitionKind::Function => CompletionKind::Function,
            DefinitionKind::Type => CompletionKind::Constructor,
            DefinitionKind::Ability => CompletionKind::Ability,
            // Skip local variables and parameters in global completion
            DefinitionKind::Variable | DefinitionKind::Parameter => continue,
        };
        entries.push(CompletionEntry {
            name: def.name,
            kind,
            detail: None,
        });
    }

    CompletionIndex::new(db, entries)
}

/// Get keyword completions.
pub fn complete_keywords(prefix: &str) -> Vec<CompletionEntry> {
    KEYWORDS
        .iter()
        .filter(|kw| kw.starts_with(prefix))
        .map(|kw| CompletionEntry {
            name: Symbol::new(kw),
            kind: CompletionKind::Keyword,
            detail: None,
        })
        .collect()
}

impl<'db> CompletionIndex<'db> {
    /// Get completions for expression context with a prefix filter.
    pub fn complete_expression(
        &self,
        db: &'db dyn salsa::Database,
        prefix: &str,
    ) -> Vec<&CompletionEntry> {
        self.entries(db)
            .iter()
            .filter(|e| e.name.with_str(|s| s.starts_with(prefix)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tree_sitter::Parser;
    use tribute::SourceCst;

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
    fn test_complete_functions(db: &salsa::DatabaseImpl) {
        let source_text = "fn foo() { }\nfn bar() { }\nfn baz() { }";
        let source = make_source("test.trb", source_text);

        let index = build(db, source);

        // Complete with "ba" prefix
        let completions = index.complete_expression(db, "ba");
        assert_eq!(completions.len(), 2, "Should find bar and baz");

        // Complete with "f" prefix
        let completions = index.complete_expression(db, "f");
        assert_eq!(completions.len(), 1, "Should find foo");
    }

    #[test]
    fn test_complete_keywords() {
        let completions = complete_keywords("fn");
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].name, "fn");

        let completions = complete_keywords("s");
        assert_eq!(completions.len(), 1); // struct
    }
}
