//! Lowering context for CST to AST conversion.

use ropey::Rope;
use tree_sitter::Node;
use trunk_ir::Symbol;

use crate::ast::{NodeId, NodeIdGen};

/// Context for lowering CST to AST.
pub struct AstLoweringCtx {
    pub source: Rope,
    /// Generator for unique node IDs.
    node_id_gen: NodeIdGen,
}

impl AstLoweringCtx {
    /// Create a new lowering context.
    pub fn new(source: Rope) -> Self {
        Self {
            source,
            node_id_gen: NodeIdGen::new(),
        }
    }

    /// Generate a fresh NodeId.
    pub fn fresh_id(&mut self) -> NodeId {
        self.node_id_gen.fresh()
    }

    /// Get the text content of a node.
    pub fn node_text(&self, node: &Node) -> &str {
        let start = node.start_byte();
        let end = node.end_byte();
        // Get the slice from the rope
        let slice = self.source.byte_slice(start..end);
        // Convert to contiguous string (may allocate if spans multiple chunks)
        // For short identifiers this is usually a single chunk
        // Fallback: shouldn't happen for well-formed source
        slice.as_str().unwrap_or("")
    }

    /// Get the text content of a node as an owned String.
    pub fn node_text_owned(&self, node: &Node) -> String {
        let start = node.start_byte();
        let end = node.end_byte();
        self.source.byte_slice(start..end).to_string()
    }

    /// Get a Symbol from a node's text.
    pub fn node_symbol(&self, node: &Node) -> Symbol {
        Symbol::from_dynamic(self.node_text(node))
    }
}
