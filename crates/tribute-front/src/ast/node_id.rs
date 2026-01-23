//! Node ID for AST nodes.
//!
//! Each AST node has a unique NodeId that can be used to look up
//! additional information (spans, types, etc.) in separate tables.
//! This follows the rust-analyzer pattern of separating structure from metadata.

/// Unique identifier for an AST node within a module.
///
/// NodeIds are local to a compilation unit and should not be used
/// for cross-module references.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct NodeId(u32);

impl NodeId {
    /// Create a new NodeId from a raw value.
    ///
    /// This should only be used by `NodeIdGen`.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw value of this NodeId.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Generator for unique NodeIds within a compilation unit.
#[derive(Debug, Default)]
pub struct NodeIdGen(u32);

impl NodeIdGen {
    /// Create a new generator starting at 0.
    pub fn new() -> Self {
        Self(0)
    }

    /// Generate a fresh unique NodeId.
    pub fn fresh(&mut self) -> NodeId {
        let id = NodeId(self.0);
        self.0 += 1;
        id
    }

    /// Get the number of IDs generated so far.
    pub fn count(&self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_gen_produces_sequential_ids() {
        let mut id_gen = NodeIdGen::new();
        assert_eq!(id_gen.fresh(), NodeId::new(0));
        assert_eq!(id_gen.fresh(), NodeId::new(1));
        assert_eq!(id_gen.fresh(), NodeId::new(2));
        assert_eq!(id_gen.count(), 3);
    }
}
