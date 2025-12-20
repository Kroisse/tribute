//! CST navigation helpers and utility types.

use std::hash::{Hash, Hasher};

use tree_sitter::{Node, Tree};
use trunk_ir::Span;
use trunk_ir::{Symbol, SymbolVec, idvec};

// =============================================================================
// Parsed CST (Salsa-cacheable)
// =============================================================================

/// A parsed CST tree, wrapped for Salsa caching.
///
/// Tree-sitter's `Tree` is internally reference-counted (`ts_tree_copy` is O(1)),
/// so cloning is cheap and we can use it directly without additional wrapping.
#[derive(Clone, Debug)]
pub struct ParsedCst(Tree);

impl ParsedCst {
    /// Create a new ParsedCst from a tree-sitter Tree.
    pub fn new(tree: Tree) -> Self {
        Self(tree)
    }

    /// Get a reference to the underlying tree.
    pub fn tree(&self) -> &Tree {
        &self.0
    }

    /// Get the root node of the CST.
    pub fn root_node(&self) -> Node<'_> {
        self.0.root_node()
    }
}

impl PartialEq for ParsedCst {
    fn eq(&self, other: &Self) -> bool {
        // Trees from the same parse are equal if they have the same root node id
        self.0.root_node().id() == other.0.root_node().id()
    }
}

impl Eq for ParsedCst {}

impl Hash for ParsedCst {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by root node id
        self.0.root_node().id().hash(state);
    }
}

// =============================================================================
// Symbol Helpers
// =============================================================================

/// Create a symbol from a string.
pub fn sym(name: &str) -> Symbol {
    Symbol::new(name)
}

/// Create a symbol reference (path) from a single name.
pub fn sym_ref(name: &str) -> SymbolVec {
    idvec![Symbol::new(name)]
}

// =============================================================================
// CST Navigation Helpers
// =============================================================================

/// Check if a node is a comment that should be skipped.
pub fn is_comment(kind: &str) -> bool {
    matches!(
        kind,
        "line_comment" | "block_comment" | "line_doc_comment" | "block_doc_comment"
    )
}

/// Get text from a node.
pub fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    node.utf8_text(source.as_bytes()).unwrap_or("")
}

/// Create a Span from a tree-sitter Node.
pub fn span_from_node(node: &Node) -> Span {
    Span::new(node.start_byte(), node.end_byte())
}
