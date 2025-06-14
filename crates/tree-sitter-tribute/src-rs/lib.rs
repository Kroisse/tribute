unsafe extern "C" {
    fn tree_sitter_tribute() -> tree_sitter::Language;
}

/// The tree-sitter Language for the Tribute programming language.
pub fn language() -> tree_sitter::Language {
    unsafe { tree_sitter_tribute() }
}

/// The content of the [`node-types.json`][] file for this grammar.
///
/// [`node-types.json`]: https://tree-sitter.github.io/tree-sitter/using-parsers#static-node-types
pub const NODE_TYPES: &str = include_str!("../src/node-types.json");

/// The syntax highlighting query for this language.
pub const HIGHLIGHTS_QUERY: &str = "";

/// The syntax injection query for this language.
pub const INJECTIONS_QUERY: &str = "";

/// The symbol tagging query for this language.
pub const LOCALS_QUERY: &str = "";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_load_grammar() {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&language())
            .expect("Error loading Tribute language");
    }
}
