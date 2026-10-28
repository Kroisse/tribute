//! SpanMap for tracking AST node source locations.
//!
//! Following the rust-analyzer pattern, AST nodes don't store spans directly.
//! Instead, each node has a `NodeId` that can be used to look up its span
//! in a separate `SpanMap`. This has several benefits:
//!
//! - AST is purely structural (easier to work with)
//! - Span changes don't invalidate Salsa caches
//! - Additional metadata can be added using the same pattern

use rustc_hash::{FxBuildHasher, FxHashMap as HashMap};

use trunk_ir::Span;

use super::NodeId;

/// Builder for constructing a SpanMap during AST lowering.
///
/// This is used during CST → AST conversion to collect spans for each node.
/// Set an item's start before inserting its nodes, and take its spans once
/// its nodes are lowered; after lowering is complete, call `finish()` for the
/// layout and any spans not taken.
#[derive(Debug, Default)]
pub struct SpanMapBuilder {
    spans: HashMap<u64, ItemSpans>,
    layout: ItemLayout,
}

impl SpanMapBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that the item `owner` starts at byte `start`.
    pub fn set_start(&mut self, owner: u64, start: usize) {
        self.layout.starts.insert(owner, start);
    }

    /// Record the span of the root `Module` node, which no item owns.
    pub fn set_module(&mut self, id: NodeId, span: Span) {
        self.layout.module = Some((id, span));
    }

    /// Insert the source span of a node relative to its item's start. A node
    /// no item owns keeps its span as is.
    pub fn insert(&mut self, id: NodeId, span: Span) {
        let start = self.layout.starts.get(&id.owner()).copied().unwrap_or(0);
        debug_assert!(start <= span.start, "a node starts within its item");
        let relative = Span::new(span.start - start, span.end - start);
        self.spans
            .entry(id.owner())
            .or_default()
            .0
            .insert(id, relative);
    }

    /// Take the relative spans of the nodes `owner` owns.
    pub fn take(&mut self, owner: u64) -> ItemSpans {
        self.spans.remove(&owner).unwrap_or_default()
    }

    /// The relative spans not taken and the layout of the items.
    pub fn finish(self) -> (ItemSpans, ItemLayout) {
        (self.spans.values().collect(), self.layout)
    }
}

/// Each item node's span relative to the start of its item, in one parsed
/// source.
///
/// Moving an item without changing it leaves its nodes' entries equal.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ItemSpans(HashMap<NodeId, Span>);

impl<'a> FromIterator<&'a ItemSpans> for ItemSpans {
    /// The spans of several items' nodes in one map.
    fn from_iter<I: IntoIterator<Item = &'a ItemSpans>>(parts: I) -> Self {
        Self(
            parts
                .into_iter()
                .flat_map(|part| part.0.iter())
                .map(|(&id, &span)| (id, span))
                .collect(),
        )
    }
}

/// Where each item of one parsed source starts, and the span of its root
/// `Module`.
///
/// Moving items changes only this part of a source's spans. The root spans
/// the whole source, so any edit that changes the source's length changes it
/// too.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ItemLayout {
    starts: HashMap<u64, usize>,
    module: Option<(NodeId, Span)>,
}

/// NodeId → Span mapping of one parsed source.
///
/// The parts belong to the parse that built them (`ParsedAst`, a Salsa
/// tracked struct); the map borrows them and adds each item node's relative
/// span to its item's start. They compare by content, so a reparse that
/// moves no item leaves both unchanged, and one that only moves items leaves
/// the relative spans unchanged.
#[derive(Clone, Copy, Debug)]
pub struct SpanMap<'a> {
    spans: &'a ItemSpans,
    layout: &'a ItemLayout,
}

impl<'a> SpanMap<'a> {
    /// The map that positions `spans` by `layout`.
    pub fn new(spans: &'a ItemSpans, layout: &'a ItemLayout) -> Self {
        Self { spans, layout }
    }

    /// The map of no nodes, for checking code that has no source.
    pub fn empty() -> SpanMap<'static> {
        static SPANS: ItemSpans = ItemSpans(HashMap::with_hasher(FxBuildHasher));
        static LAYOUT: ItemLayout = ItemLayout {
            starts: HashMap::with_hasher(FxBuildHasher),
            module: None,
        };
        SpanMap::new(&SPANS, &LAYOUT)
    }

    /// Look up a span by NodeId.
    ///
    /// For specialized (monomorphized) nodes, falls back to the origin NodeId
    /// so that spans are inherited from the original generic function.
    pub fn get(&self, id: NodeId) -> Option<Span> {
        self.get_exact(id).or_else(|| {
            id.variant()
                .is_some()
                .then(|| self.get_exact(id.origin()))
                .flatten()
        })
    }

    fn get_exact(&self, id: NodeId) -> Option<Span> {
        if let Some(relative) = self.spans.0.get(&id) {
            let start = self.layout.starts.get(&id.owner()).copied().unwrap_or(0);
            return Some(Span::new(relative.start + start, relative.end + start));
        }
        self.layout
            .module
            .and_then(|(module, span)| (module == id).then_some(span))
    }

    /// Look up a span by NodeId, returning a default span if not found.
    pub fn get_or_default(&self, id: NodeId) -> Span {
        self.get(id).unwrap_or(Span::new(0, 0))
    }

    /// Check if this NodeId has a recorded span.
    pub fn contains(&self, id: NodeId) -> bool {
        self.get_exact(id).is_some()
    }
}

/// The span maps of several parsed sources, looked up as one, such as a
/// module's and those of the packages it depends on.
///
/// NodeIds carry their source's hash, so a node is in at most one of them.
#[derive(Clone, Debug, Default)]
pub struct SpanMaps<'a>(Vec<SpanMap<'a>>);

impl<'a> SpanMaps<'a> {
    /// Look up spans in `maps`.
    pub fn new(maps: impl IntoIterator<Item = SpanMap<'a>>) -> Self {
        Self(maps.into_iter().collect())
    }

    /// Look up a span by NodeId in each map in turn.
    pub fn get(&self, id: NodeId) -> Option<Span> {
        self.0.iter().find_map(|map| map.get(id))
    }

    /// Look up a span by NodeId, returning a default span if not found.
    pub fn get_or_default(&self, id: NodeId) -> Span {
        self.get(id).unwrap_or(Span::new(0, 0))
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

        let (spans, layout) = builder.finish();
        let span_map = SpanMap::new(&spans, &layout);

        assert_eq!(span_map.get(id1), Some(Span::new(10, 20)));
        assert_eq!(span_map.get(id2), Some(Span::new(30, 40)));
        assert_eq!(span_map.get(NodeId::from_raw(999)), None);
    }

    #[test]
    fn test_span_map_get_or_default() {
        let mut builder = SpanMapBuilder::new();
        let id1 = NodeId::from_raw(0);
        builder.insert(id1, Span::new(10, 20));

        let (spans, layout) = builder.finish();
        let span_map = SpanMap::new(&spans, &layout);

        assert_eq!(span_map.get_or_default(id1), Span::new(10, 20));
        assert_eq!(
            span_map.get_or_default(NodeId::from_raw(999)),
            Span::new(0, 0)
        );
    }

    #[test]
    fn test_span_maps_look_up_each_map() {
        let mut user = SpanMapBuilder::new();
        user.insert(NodeId::from_raw(0), Span::new(10, 20));
        let (user_spans, user_layout) = user.finish();
        let mut dependency = SpanMapBuilder::new();
        dependency.insert(NodeId::new(1, 0, 0), Span::new(30, 40));
        let (dependency_spans, dependency_layout) = dependency.finish();

        let spans = SpanMaps::new([
            SpanMap::new(&user_spans, &user_layout),
            SpanMap::new(&dependency_spans, &dependency_layout),
        ]);
        assert_eq!(spans.get(NodeId::from_raw(0)), Some(Span::new(10, 20)));
        assert_eq!(spans.get(NodeId::new(1, 0, 0)), Some(Span::new(30, 40)));
        assert_eq!(spans.get_or_default(NodeId::from_raw(9)), Span::new(0, 0));
    }

    #[test]
    fn test_span_map_contains() {
        let mut builder = SpanMapBuilder::new();
        let id = NodeId::from_raw(100);
        builder.insert(id, Span::new(0, 50));

        let (spans, layout) = builder.finish();
        let span_map = SpanMap::new(&spans, &layout);

        assert!(span_map.contains(id));
        assert!(!span_map.contains(NodeId::from_raw(999)));
    }

    #[test]
    fn test_span_map_variant_fallback() {
        let mut builder = SpanMapBuilder::new();
        let origin = NodeId::from_raw(42);
        let span = Span::new(10, 20);
        builder.insert(origin, span);

        let (spans, layout) = builder.finish();
        let span_map = SpanMap::new(&spans, &layout);

        // Specialized NodeId should fall back to origin's span
        let variant = std::num::NonZero::new(12345u64).unwrap();
        let specialized = origin.with_variant(variant);

        assert_ne!(origin, specialized);
        assert_eq!(span_map.get(specialized), Some(span));
        assert_eq!(span_map.get_or_default(specialized), span);
    }

    #[test]
    fn test_span_map_positions_nodes_from_their_owner() {
        let build = |start: usize| {
            let mut builder = SpanMapBuilder::new();
            builder.set_start(7, start);
            builder.insert(NodeId::new(0, 7, 0), Span::new(start, start + 10));
            builder.insert(NodeId::new(0, 7, 1), Span::new(start + 4, start + 6));
            builder.insert(NodeId::new(0, 3, 0), Span::new(0, 2));
            builder.set_module(NodeId::new(0, 3, 1), Span::new(0, start + 10));
            builder.finish()
        };
        let (spans, layout) = build(20);
        let (moved_spans, moved_layout) = build(35);

        assert_eq!(
            spans, moved_spans,
            "moving an owner keeps its relative spans"
        );
        assert_ne!(layout, moved_layout);
        let span_map = SpanMap::new(&moved_spans, &moved_layout);
        assert_eq!(span_map.get(NodeId::new(0, 7, 1)), Some(Span::new(39, 41)));
        assert_eq!(span_map.get(NodeId::new(0, 3, 0)), Some(Span::new(0, 2)));
        assert_eq!(span_map.get(NodeId::new(0, 3, 1)), Some(Span::new(0, 45)));
    }
}
