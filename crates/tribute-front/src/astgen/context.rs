//! Lowering context for CST to AST conversion.

use std::borrow::Cow;

use rustc_hash::FxHashMap as HashMap;

use ropey::Rope;
use salsa::Accumulator;
use tree_sitter::Node;
use trunk_ir::{Span, Symbol, SymbolPath};

use crate::ast::{ItemSpans, ModuleItem, NodeId, SpanMapBuilder, UnresolvedName, owner_hash};
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};

/// Context for lowering CST to AST.
pub struct AstLoweringCtx<'db> {
    pub source: Rope,
    /// Source hash for distinguishing parse sessions (derived from source URI).
    source_hash: u64,
    /// Builder for span map.
    span_builder: SpanMapBuilder,
    /// Salsa database for accumulating diagnostics directly.
    /// `None` in test-only paths where diagnostics are discarded.
    db: Option<&'db dyn salsa::Database>,
    /// The inline modules around the items being lowered.
    module_path: Vec<Symbol>,
    /// The item that owns the nodes being lowered.
    owner: u64,
    /// How many items of the same kind and name precede that item in its
    /// module.
    occurrence: u32,
    /// How many nodes each owner has numbered so far.
    next_index: HashMap<u64, u32>,
    /// How many items of each kind and name each module has entered so far.
    item_counts: HashMap<(u64, &'static str, String), u32>,
}

impl<'db> AstLoweringCtx<'db> {
    /// Create a new lowering context without a database.
    ///
    /// Diagnostics emitted through this context will be silently discarded.
    /// **Intended for tests and debugging only.** Production code should use
    /// [`AstLoweringCtx::with_db`].
    pub fn new(source: Rope, source_hash: u64) -> Self {
        Self {
            source,
            source_hash,
            span_builder: SpanMapBuilder::new(),
            db: None,
            module_path: Vec::new(),
            owner: module_owner(&[]),
            occurrence: 0,
            next_index: HashMap::default(),
            item_counts: HashMap::default(),
        }
    }

    /// Create a new lowering context with a Salsa database.
    ///
    /// Diagnostics are accumulated directly into Salsa via the database.
    pub fn with_db(db: &'db dyn salsa::Database, source: Rope, source_hash: u64) -> Self {
        Self {
            db: Some(db),
            ..Self::new(source, source_hash)
        }
    }

    /// Generate the NodeId of the next node of the current owner and record
    /// the CST node's span for it.
    pub fn fresh_id_with_span(&mut self, node: &Node) -> NodeId {
        self.fresh_id_with_range(node.start_byte(), node.end_byte())
    }

    /// Generate the NodeId of the next node of the current owner and record
    /// the bytes `start..end` as its span.
    pub fn fresh_id_with_range(&mut self, start: usize, end: usize) -> NodeId {
        let id = self.next_id();
        self.span_builder.insert(id, Span::new(start, end));
        id
    }

    /// Generate the NodeId of the root `Module` node and record its span in
    /// the item layout.
    pub fn module_id_with_span(&mut self, node: &Node) -> NodeId {
        let id = self.next_id();
        let span = Span::new(node.start_byte(), node.end_byte());
        self.span_builder.set_module(id, span);
        id
    }

    fn next_id(&mut self) -> NodeId {
        let next = self.next_index.entry(self.owner).or_default();
        let id = NodeId::new(self.source_hash, self.owner, *next);
        *next += 1;
        id
    }

    /// Lower the body of the inline module `name` with its module-level nodes
    /// owned by that module.
    pub fn in_module<T>(&mut self, name: Symbol, lower: impl FnOnce(&mut Self) -> T) -> T {
        self.module_path.push(name);
        let outer = std::mem::replace(&mut self.owner, module_owner(&self.module_path));
        let lowered = lower(self);
        self.owner = outer;
        self.module_path.pop();
        lowered
    }

    /// Lower the item of `kind` named `name`, which starts at byte `start`,
    /// with its nodes owned by that item and positioned from its start.
    /// Items of one kind and name in one module are told apart by how many
    /// of them precede the item. `lower` returns the item's declarations and
    /// the items an inline module's body declares.
    pub fn in_item(
        &mut self,
        kind: &'static str,
        name: &str,
        start: usize,
        lower: impl FnOnce(&mut Self) -> (Vec<ModuleItem<UnresolvedName>>, Vec<LoweredItem>),
    ) -> LoweredItem {
        let module = module_owner(&self.module_path);
        let count = self
            .item_counts
            .entry((module, kind, name.to_owned()))
            .or_default();
        let occurrence = *count;
        *count += 1;
        let item = owner_hash((path_text(&self.module_path), kind, name, occurrence));
        self.span_builder.set_start(item, start);
        let outer = std::mem::replace(&mut self.owner, item);
        let outer_occurrence = std::mem::replace(&mut self.occurrence, occurrence);
        let (decls, children) = lower(self);
        self.owner = outer;
        self.occurrence = outer_occurrence;
        LoweredItem {
            owner: item,
            decls,
            children,
            spans: self.span_builder.take(item),
        }
    }

    /// How many items of the same kind and name precede the item being
    /// lowered in its module.
    pub fn occurrence(&self) -> u32 {
        self.occurrence
    }

    /// Emit a diagnostic error.
    pub fn error(&mut self, span: Span, message: impl Into<String>) {
        if let Some(db) = self.db {
            Diagnostic::new(
                message,
                span,
                DiagnosticSeverity::Error,
                CompilationPhase::AstGeneration,
            )
            .accumulate(db);
        }
    }

    /// Emit a parse error diagnostic for syntax errors.
    pub fn parse_error(&mut self, span: Span, message: impl Into<String>) {
        if let Some(db) = self.db {
            Diagnostic::new(
                message,
                span,
                DiagnosticSeverity::Error,
                CompilationPhase::Parsing,
            )
            .accumulate(db);
        }
    }

    /// Consume the context and return the span builder.
    pub fn finish(self) -> SpanMapBuilder {
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

    /// The path a name node spells, one component per segment.
    pub fn node_path(&self, node: &Node) -> SymbolPath {
        let mut cursor = node.walk();
        let segments: Vec<Symbol> = node
            .named_children(&mut cursor)
            .filter(|child| {
                matches!(
                    child.kind(),
                    "path_root"
                        | "path_segment"
                        | "path_keyword"
                        | "identifier"
                        | "type_identifier"
                )
            })
            .map(|child| self.node_symbol(&child))
            .collect();
        if !segments.is_empty() {
            return SymbolPath::new(segments);
        }
        // A qualified operator such as `Int::+` is a single token.
        if node.kind() == "qualified_operator"
            && let Some((owner, operator)) = self.node_text(node).split_once("::")
        {
            return SymbolPath::new([owner, operator]);
        }
        SymbolPath::new([self.node_symbol(node)])
    }

    /// The constructor a `type_path` or an `implicit_constructor` node names.
    pub fn constructor_name(&mut self, node: &Node) -> UnresolvedName {
        let path = self.node_path(node);
        let id = self.fresh_id_with_span(node);
        if node.kind() == "implicit_constructor" {
            let name = path.leaf().expect("a constructor has a name").clone();
            UnresolvedName::implicit(name, id)
        } else {
            UnresolvedName::new(path, id)
        }
    }

    /// Get a Symbol from a node's text, without the `r#` of raw identifiers.
    pub fn node_symbol(&self, node: &Node) -> Symbol {
        if node.kind() == "path_root" {
            return Symbol::new(crate::keywords::PATH_ROOT);
        }
        Symbol::new(&crate::keywords::unraw(&self.node_text(node)))
    }
}

/// One lowered item: its declarations, the items of an inline module's body,
/// and its nodes' spans relative to its start.
#[derive(Debug)]
pub struct LoweredItem {
    pub owner: u64,
    /// An inline module's declaration has an empty body; its items are
    /// `children`.
    pub decls: Vec<ModuleItem<UnresolvedName>>,
    pub children: Vec<LoweredItem>,
    pub spans: ItemSpans,
}

/// The owner of the module-level nodes of the module at `path`.
fn module_owner(path: &[Symbol]) -> u64 {
    owner_hash(("mod", path_text(path)))
}

/// The text of a module path, which owner hashes are computed from.
fn path_text(path: &[Symbol]) -> Vec<&str> {
    path.iter().map(Symbol::as_str).collect()
}
