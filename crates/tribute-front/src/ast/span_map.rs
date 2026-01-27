//! SpanMap for tracking AST node source locations.
//!
//! Following the rust-analyzer pattern, AST nodes don't store spans directly.
//! Instead, each node has a `NodeId` that can be used to look up its span
//! in a separate `SpanMap`. This has several benefits:
//!
//! - AST is purely structural (easier to work with)
//! - Span changes don't invalidate Salsa caches
//! - Additional metadata can be added using the same pattern

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use trunk_ir::Span;

use super::NodeId;

/// Builder for constructing a SpanMap during AST lowering.
///
/// This is used during CST → AST conversion to collect spans for each node.
/// After lowering is complete, call `finish()` to create a `SpanMap`.
#[derive(Debug, Default)]
pub struct SpanMapBuilder {
    spans: HashMap<NodeId, Span>,
}

impl SpanMapBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a span for a node.
    pub fn insert(&mut self, id: NodeId, span: Span) {
        self.spans.insert(id, span);
    }

    /// Create a SpanMap from this builder.
    pub fn finish(self) -> SpanMap {
        SpanMap(Arc::new(self.spans))
    }
}

/// NodeId → Span mapping.
///
/// Uses `Arc<HashMap>` internally for efficient sharing and cheap cloning.
/// The SpanMap is immutable once created.
///
/// This is embedded in `ParsedAst` (a Salsa tracked struct) which handles
/// the incremental caching. SpanMap itself is not a Salsa type.
///
/// Note: `Hash` is implemented using Arc pointer identity for Salsa compatibility.
/// Two SpanMaps hash the same only if they point to the same underlying data.
#[derive(Clone, Debug, salsa::Update)]
pub struct SpanMap(Arc<HashMap<NodeId, Span>>);

impl PartialEq for SpanMap {
    fn eq(&self, other: &Self) -> bool {
        // Compare by Arc pointer identity for efficiency
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for SpanMap {}

impl Hash for SpanMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by Arc pointer address for Salsa compatibility
        // This means two SpanMaps with the same content but different Arc instances
        // will have different hashes, but Salsa will fall back to equality check
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

impl SpanMap {
    /// Look up a span by NodeId.
    pub fn get(&self, id: NodeId) -> Option<Span> {
        self.0.get(&id).copied()
    }

    /// Look up a span by NodeId, returning a default span if not found.
    pub fn get_or_default(&self, id: NodeId) -> Span {
        self.0.get(&id).copied().unwrap_or(Span::new(0, 0))
    }

    /// Check if this NodeId has a recorded span.
    ///
    /// This is useful for reverse lookup: given a CST node, check if it
    /// corresponds to an AST node by checking if its ID is in the SpanMap.
    pub fn contains(&self, id: NodeId) -> bool {
        self.0.contains_key(&id)
    }
}

impl Default for SpanMap {
    fn default() -> Self {
        Self(Arc::new(HashMap::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_map_builder() {
        let mut builder = SpanMapBuilder::new();
        let id1 = NodeId::from_raw(0);
        let id2 = NodeId::from_raw(1);

        builder.insert(id1, Span::new(10, 20));
        builder.insert(id2, Span::new(30, 40));

        let span_map = builder.finish();

        assert_eq!(span_map.get(id1), Some(Span::new(10, 20)));
        assert_eq!(span_map.get(id2), Some(Span::new(30, 40)));
        assert_eq!(span_map.get(NodeId::from_raw(999)), None);
    }

    #[test]
    fn test_span_map_get_or_default() {
        let mut builder = SpanMapBuilder::new();
        let id1 = NodeId::from_raw(0);
        builder.insert(id1, Span::new(10, 20));

        let span_map = builder.finish();

        assert_eq!(span_map.get_or_default(id1), Span::new(10, 20));
        assert_eq!(
            span_map.get_or_default(NodeId::from_raw(999)),
            Span::new(0, 0)
        );
    }

    #[test]
    fn test_span_map_clone() {
        let mut builder = SpanMapBuilder::new();
        builder.insert(NodeId::from_raw(0), Span::new(10, 20));

        let span_map1 = builder.finish();
        let span_map2 = span_map1.clone();

        // Both should refer to the same data (Arc)
        assert_eq!(
            span_map1.get(NodeId::from_raw(0)),
            span_map2.get(NodeId::from_raw(0))
        );
    }

    #[test]
    fn test_span_map_contains() {
        let mut builder = SpanMapBuilder::new();
        let id = NodeId::from_raw(100);
        builder.insert(id, Span::new(0, 50));

        let span_map = builder.finish();

        assert!(span_map.contains(id));
        assert!(!span_map.contains(NodeId::from_raw(999)));
    }
}
