//! Node ID for AST nodes.
//!
//! Each AST node has a unique NodeId that can be used to look up
//! additional information (spans, types, etc.) in separate tables.
//! This follows the rust-analyzer pattern of separating structure from metadata.

use std::hash::{Hash, Hasher};

use tree_sitter::Node;

/// Unique identifier for an AST node within a compilation unit.
///
/// This combines a source hash with the CST node's `id()` value from tree-sitter.
/// The source hash is derived from the source file URI, so nodes from different
/// files (e.g., prelude vs user code) are naturally distinguished without manual
/// tag assignment. This prevents key-space collisions when merging SpanMaps or
/// node_types HashMaps.
///
/// NodeIds are local to a compilation unit and should not be used
/// for cross-module references.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, salsa::Update)]
pub struct NodeId {
    source: u64,
    raw: usize,
}

impl NodeId {
    /// Create a NodeId from a CST node with a source hash.
    ///
    /// The source hash distinguishes nodes from different parse sessions.
    /// Use [`source_hash`] to compute the hash from a URI.
    #[inline]
    pub fn from_cst(node: &Node, source: u64) -> Self {
        Self {
            source,
            raw: node.id(),
        }
    }

    /// Get the raw CST node id.
    #[inline]
    pub const fn raw(self) -> usize {
        self.raw
    }

    /// Get the source hash.
    #[inline]
    pub const fn source(self) -> u64 {
        self.source
    }
}

/// Compute a source hash from a URI string.
///
/// This produces a stable `u64` hash used to tag NodeIds so that nodes
/// from different source files never collide.
pub fn source_hash(uri: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    uri.hash(&mut hasher);
    hasher.finish()
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.source == 0 {
            write!(f, "#{}", self.raw)
        } else {
            write!(f, "#{:x}:{}", self.source, self.raw)
        }
    }
}

#[cfg(test)]
impl NodeId {
    /// Create a NodeId from a raw value (for testing only).
    /// Uses source hash 0 by default.
    pub const fn from_raw(id: usize) -> Self {
        Self { source: 0, raw: id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_from_raw() {
        let id = NodeId::from_raw(12345);
        assert_eq!(id.raw(), 12345);
        assert_eq!(id.source(), 0);
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
    fn test_node_id_different_source_not_equal() {
        let id1 = NodeId {
            source: source_hash("file:///a.trb"),
            raw: 42,
        };
        let id2 = NodeId {
            source: source_hash("prelude:///std/prelude"),
            raw: 42,
        };
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_node_id_display() {
        let id = NodeId::from_raw(123);
        assert_eq!(format!("{}", id), "#123");
    }

    #[test]
    fn test_node_id_display_with_source() {
        let id = NodeId {
            source: 0xff,
            raw: 123,
        };
        assert_eq!(format!("{}", id), "#ff:123");
    }

    #[test]
    fn test_source_hash_deterministic() {
        let h1 = source_hash("file:///foo.trb");
        let h2 = source_hash("file:///foo.trb");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_source_hash_different_uris() {
        let h1 = source_hash("file:///a.trb");
        let h2 = source_hash("file:///b.trb");
        assert_ne!(h1, h2);
    }
}
