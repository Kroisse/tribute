//! Node ID for AST nodes.
//!
//! Each AST node has a unique NodeId that can be used to look up
//! additional information (spans, types, etc.) in separate tables.
//! This follows the rust-analyzer pattern of separating structure from metadata.

use tree_sitter::Node;

/// Unique identifier for an AST node within a module.
///
/// This is the CST node's `id()` value from tree-sitter, ensuring a direct
/// correspondence between CST nodes and AST nodes. This enables reverse lookup:
/// given a byte offset, Tree-sitter can find the CST node, and the SpanMap
/// can confirm if it corresponds to an AST node.
///
/// NodeIds are local to a compilation unit and should not be used
/// for cross-module references.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct NodeId(usize);

impl NodeId {
    /// Create a NodeId from a CST node.
    ///
    /// This ensures the NodeId corresponds to an actual CST node,
    /// enabling reverse lookup from byte offsets.
    #[inline]
    pub fn from_cst(node: &Node) -> Self {
        Self(node.id())
    }

    /// Get the raw value of this NodeId.
    #[inline]
    pub const fn raw(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

#[cfg(test)]
impl NodeId {
    /// Create a NodeId from a raw value (for testing only).
    pub const fn from_raw(id: usize) -> Self {
        Self(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_from_raw() {
        let id = NodeId::from_raw(12345);
        assert_eq!(id.raw(), 12345);
    }

    #[test]
    fn test_node_id_equality() {
        let id1 = NodeId::from_raw(42);
        let id2 = NodeId::from_raw(42);
        let id3 = NodeId::from_raw(99);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_node_id_display() {
        let id = NodeId::from_raw(123);
        assert_eq!(format!("{}", id), "#123");
    }
}
