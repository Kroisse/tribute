//! Completion index for auto-completion functionality.
//!
//! Builds completion candidates from the compiled module.

use trunk_ir::dialect::core::Module;

use super::definition_index::{DefinitionIndex, DefinitionKind, KEYWORDS};

/// Entry representing a completion candidate.
#[derive(Clone, Debug)]
pub struct CompletionEntry {
    pub name: String,
    pub kind: CompletionKind,
    pub detail: Option<String>,
}

/// Kind of completion item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionKind {
    Function,
    Constructor,
    Keyword,
    Ability,
}

/// Index for completion lookups.
pub struct CompletionIndex {
    /// All completion entries from module definitions.
    entries: Vec<CompletionEntry>,
}

impl CompletionIndex {
    /// Build a completion index from pre-resolve and post-resolve modules.
    ///
    /// - `pre_resolve`: Module before name resolution (from `parse_and_lower`).
    /// - `post_resolve`: Module after full compilation (from `compile_for_lsp`).
    pub fn build(
        db: &dyn salsa::Database,
        pre_resolve: &Module<'_>,
        post_resolve: &Module<'_>,
    ) -> Self {
        let def_index = DefinitionIndex::build(db, pre_resolve, post_resolve);
        let mut entries = Vec::new();

        // Add all definitions as completion candidates
        for def in def_index.definitions() {
            let kind = match def.kind {
                DefinitionKind::Function => CompletionKind::Function,
                DefinitionKind::Type => CompletionKind::Constructor,
                DefinitionKind::Ability => CompletionKind::Ability,
                // Skip local variables and parameters in global completion
                DefinitionKind::Variable | DefinitionKind::Parameter => continue,
            };
            entries.push(CompletionEntry {
                name: def.name.to_string(),
                kind,
                detail: None,
            });
        }

        Self { entries }
    }

    /// Get completions for expression context with a prefix filter.
    pub fn complete_expression(&self, prefix: &str) -> Vec<&CompletionEntry> {
        self.entries
            .iter()
            .filter(|e| e.name.starts_with(prefix))
            .collect()
    }

    /// Get keyword completions.
    pub fn complete_keywords(prefix: &str) -> Vec<CompletionEntry> {
        KEYWORDS
            .iter()
            .filter(|kw| kw.starts_with(prefix))
            .map(|kw| CompletionEntry {
                name: (*kw).to_string(),
                kind: CompletionKind::Keyword,
                detail: None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use tree_sitter::Parser;
    use tribute::SourceCst;
    use tribute::{compile_for_lsp, parse_and_lower};

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

        let pre_resolve = parse_and_lower(db, source);
        let post_resolve = compile_for_lsp(db, source);
        let index = CompletionIndex::build(db, &pre_resolve, &post_resolve);

        // Complete with "ba" prefix
        let completions = index.complete_expression("ba");
        assert_eq!(completions.len(), 2, "Should find bar and baz");

        // Complete with "f" prefix
        let completions = index.complete_expression("f");
        assert_eq!(completions.len(), 1, "Should find foo");
    }

    #[salsa_test]
    fn test_complete_keywords(_db: &salsa::DatabaseImpl) {
        let completions = CompletionIndex::complete_keywords("fn");
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].name, "fn");

        let completions = CompletionIndex::complete_keywords("s");
        assert_eq!(completions.len(), 1); // struct
    }
}
