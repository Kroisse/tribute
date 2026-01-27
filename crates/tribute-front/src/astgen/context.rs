//! Lowering context for CST to AST conversion.

use std::borrow::Cow;

use ropey::Rope;
use tree_sitter::Node;
use trunk_ir::{Span, Symbol};

use crate::ast::{NodeId, SpanMapBuilder};

/// Context for lowering CST to AST.
pub struct AstLoweringCtx {
    pub source: Rope,
    /// Builder for span map.
    span_builder: SpanMapBuilder,
}

impl AstLoweringCtx {
    /// Create a new lowering context.
    pub fn new(source: Rope) -> Self {
        Self {
            source,
            span_builder: SpanMapBuilder::new(),
        }
    }

    /// Generate a NodeId from a CST node and record its span.
    ///
    /// Uses the CST node's `id()` directly, enabling reverse lookup:
    /// given a byte offset, Tree-sitter can find the CST node, and if
    /// that node's ID is in the SpanMap, it corresponds to an AST node.
    pub fn fresh_id_with_span(&mut self, node: &Node) -> NodeId {
        let id = NodeId::from_cst(node);
        let span = Span::new(node.start_byte(), node.end_byte());
        self.span_builder.insert(id, span);
        id
    }

    /// Consume the context and return the span builder.
    pub fn into_span_builder(self) -> SpanMapBuilder {
        self.span_builder
    }

    /// Get the text content of a node.
    ///
    /// Returns a `Cow<str>` that borrows if the text is in a single chunk,
    /// or allocates if it spans multiple chunks.
    pub fn node_text(&self, node: &Node) -> Cow<'_, str> {
        let start = node.start_byte();
        let end = node.end_byte();
        let slice = self.source.byte_slice(start..end);
        slice
            .as_str()
            .map(Cow::Borrowed)
            .unwrap_or_else(|| Cow::Owned(slice.to_string()))
    }

    /// Get the text content of a node as an owned String.
    pub fn node_text_owned(&self, node: &Node) -> String {
        let start = node.start_byte();
        let end = node.end_byte();
        self.source.byte_slice(start..end).to_string()
    }

    /// Get a Symbol from a node's text.
    pub fn node_symbol(&self, node: &Node) -> Symbol {
        Symbol::from_dynamic(&self.node_text(node))
    }
}
