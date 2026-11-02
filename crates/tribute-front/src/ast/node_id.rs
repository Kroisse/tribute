//! Node ID for AST nodes.
//!
//! Each AST node has a unique NodeId that can be used to look up
//! additional information (spans, types, etc.) in separate tables.
//! This follows the rust-analyzer pattern of separating structure from metadata.

use std::hash::{Hash, Hasher};
use std::num::NonZero;

use siphasher::sip::SipHasher13;

/// Unique identifier for an AST node within a compilation unit.
///
/// A NodeId names a node by the item that owns it and the node's position
/// among that item's nodes, as rust-analyzer names an expression by its body
/// and an index into it. The owner is a hash of the item's module path, kind,
/// name, and how many items of that kind and name precede it in its module;
/// module-level nodes (the module itself, `mod` and `use` declarations) are
/// owned by their module. An edit therefore changes the NodeIds of the items
/// it touches only: an unchanged item keeps its NodeIds wherever it moves,
/// unless it moves past an item of the same kind and name in its module,
/// which changes how many such items precede it.
///
/// The source hash is derived from the source file URI, so nodes from
/// different files (e.g., prelude vs user code) are distinguished without
/// manual tag assignment. This prevents key-space collisions when merging
/// SpanMaps or node_types HashMaps.
///
/// For monomorphized copies of generic functions, the `variant` field
/// distinguishes nodes from different specializations. Original AST nodes
/// have `variant: None`, while specialized copies carry a hash of the
/// concrete type arguments. Use [`origin()`](NodeId::origin) to recover
/// the original NodeId for SpanMap lookups.
///
/// NodeIds are local to a compilation unit and should not be used
/// for cross-module references.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, salsa::SalsaValue)]
pub struct NodeId {
    source: u64,
    owner: u64,
    index: u32,
    variant: Option<NonZero<u64>>,
}

impl NodeId {
    /// The node at `index` among the nodes of the item `owner` in the source
    /// file whose hash is `source`.
    ///
    /// Use [`source_hash`] to compute the source hash from a URI and
    /// [`IdentityHasher`] to compute the owner from an item's key.
    #[inline]
    pub const fn new(source: u64, owner: u64, index: u32) -> Self {
        Self {
            source,
            owner,
            index,
            variant: None,
        }
    }

    /// Create a specialized copy of this NodeId for monomorphization.
    ///
    /// The `variant` hash distinguishes different specializations of the
    /// same generic function (e.g., `identity$Int` vs `identity$Float`).
    /// A copy of a copy combines both variants; [`origin()`](NodeId::origin)
    /// still recovers the original node.
    #[inline]
    pub fn with_variant(self, variant: NonZero<u64>) -> Self {
        // A copy of a copy: two nodes that differ only in their variant must
        // stay distinct, so hash the outer variant together with the new one.
        let variant = self.variant.map_or(variant, |outer| {
            IdentityHasher::new()
                .u64(outer.get())
                .u64(variant.get())
                .finish_nonzero()
        });
        Self {
            variant: Some(variant),
            ..self
        }
    }

    /// Get the original NodeId, stripping any specialization variant.
    ///
    /// Use this to look up spans in SpanMap, which only contains entries
    /// for original (non-specialized) nodes.
    #[inline]
    pub const fn origin(self) -> Self {
        Self {
            source: self.source,
            owner: self.owner,
            index: self.index,
            variant: None,
        }
    }

    /// Get the hash of the item that owns the node.
    #[inline]
    pub const fn owner(self) -> u64 {
        self.owner
    }

    /// Get the node's position among its owner's nodes.
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Get the source hash.
    #[inline]
    pub const fn source(self) -> u64 {
        self.source
    }

    /// Get the specialization variant, if any.
    #[inline]
    pub const fn variant(self) -> Option<NonZero<u64>> {
        self.variant
    }
}

/// The hasher of the identity hashes in a [`NodeId`]: its owner and its
/// variant.
///
/// A collision merges two items' or two specializations' nodes, so it is
/// SipHash-1-3, a well-distributed hash. It is a specified algorithm with
/// fixed keys, so an identity does not change with the toolchain, as the
/// standard library's `DefaultHasher` may. Each value is written in an
/// explicit encoding rather than through `Hash`: integers as little-endian
/// bytes, and text and sequences prefixed with their length.
#[derive(Clone, Debug)]
pub struct IdentityHasher(SipHasher13);

impl IdentityHasher {
    pub fn new() -> Self {
        Self(SipHasher13::new_with_keys(0, 0))
    }

    /// Write an integer.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.0.write(&value.to_le_bytes());
        self
    }

    /// Write a string.
    pub fn str(&mut self, text: &str) -> &mut Self {
        self.u64(text.len() as u64);
        self.0.write(text.as_bytes());
        self
    }

    /// Write a sequence of strings.
    pub fn strs<'a>(&mut self, texts: impl ExactSizeIterator<Item = &'a str>) -> &mut Self {
        self.u64(texts.len() as u64);
        for text in texts {
            self.str(text);
        }
        self
    }

    pub fn finish(&self) -> u64 {
        self.0.finish()
    }

    /// The hash, with zero replaced by one.
    pub fn finish_nonzero(&self) -> NonZero<u64> {
        NonZero::new(self.finish()).unwrap_or(NonZero::<u64>::MIN)
    }
}

impl Default for IdentityHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a source hash from a URI string.
///
/// This produces a stable `u64` hash used to tag NodeIds so that nodes
/// from different source files never collide.
pub fn source_hash(uri: &str) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    uri.hash(&mut hasher);
    hasher.finish()
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.source == 0 && self.owner == 0 {
            write!(f, "#{}", self.index)
        } else {
            write!(f, "#{:x}:{:x}:{}", self.source, self.owner, self.index)
        }
    }
}

#[cfg(test)]
impl NodeId {
    /// Create a NodeId from a raw value (for testing only).
    /// Uses source hash and owner 0.
    pub const fn from_raw(id: usize) -> Self {
        Self::new(0, 0, id as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_from_raw() {
        let id = NodeId::from_raw(12345);
        assert_eq!(id.index(), 12345);
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
        let id1 = NodeId::new(source_hash("file:///a.trb"), 0, 42);
        let id2 = NodeId::new(source_hash("prelude:///std/prelude"), 0, 42);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_node_id_different_owner_not_equal() {
        let id1 = NodeId::new(0, IdentityHasher::new().str("fn").str("f").finish(), 3);
        let id2 = NodeId::new(0, IdentityHasher::new().str("fn").str("g").finish(), 3);
        assert_ne!(id1, id2);
    }

    const PINNED_EMPTY: u64 = 15130871412783076140;
    const PINNED_ITEM: u64 = 793205033555831425;
    const PINNED_VARIANT: u64 = 1141703569963311043;

    /// Identity hashes are pinned: a change in these values changes every
    /// NodeId, so it must be deliberate.
    #[test]
    fn identity_hashes_are_pinned() {
        assert_eq!(IdentityHasher::new().finish(), PINNED_EMPTY);
        assert_eq!(
            IdentityHasher::new()
                .u64(1)
                .strs(["std", "io"].into_iter())
                .str("fn")
                .str("print_line")
                .u64(0)
                .finish(),
            PINNED_ITEM
        );
        let variant = NonZero::new(7).unwrap();
        let copy = NodeId::new(0, 0, 0)
            .with_variant(variant)
            .with_variant(variant);
        assert_eq!(copy.variant().map(NonZero::get), Some(PINNED_VARIANT));
    }

    #[test]
    fn identity_hasher_separates_strings() {
        let joined = IdentityHasher::new().str("ab").str("c").finish();
        let split = IdentityHasher::new().str("a").str("bc").finish();
        assert_ne!(joined, split);
    }

    #[test]
    fn test_node_id_display() {
        let id = NodeId::from_raw(123);
        assert_eq!(format!("{}", id), "#123");
    }

    #[test]
    fn test_node_id_display_with_source() {
        let id = NodeId::new(0xff, 0xa, 123);
        assert_eq!(format!("{}", id), "#ff:a:123");
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
