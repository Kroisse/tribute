//! Lowering context for CST to AST conversion.

use std::borrow::Cow;

use rustc_hash::FxHashMap as HashMap;

use ropey::Rope;
use salsa::Accumulator;
use tree_sitter::Node;
use trunk_ir::{Span, Symbol, SymbolPath};

use crate::ast::{NodeId, SpanMapBuilder, UnresolvedName, owner_hash};
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};

/// Context for lowering CST to AST.
pub struct AstLoweringCtx<'db> {
    pub source: Rope,
    /// Source hash for distinguishing parse sessions (derived from source URI).
    source_hash: u64,
    /// The source id stamped on spans: `0` for the file an analysis starts
    /// from, the source hash for a file module.
    span_source: u64,
    /// Builder for span map.
    span_builder: SpanMapBuilder,
    /// Salsa database for accumulating diagnostics directly.
    /// `None` in test-only paths where diagnostics are discarded.
    db: Option<&'db dyn salsa::Database>,
    /// The inline modules around the items being lowered.
    module_path: Vec<Symbol>,
    /// The item that owns the nodes being lowered.
    owner: u64,
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
            span_source: 0,
            span_builder: SpanMapBuilder::new(),
            db: None,
            module_path: Vec::new(),
            owner: module_owner(&[]),
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

    /// Stamp spans and diagnostics with this source's hash, as a file
    /// module's are.
    pub fn in_file_module(self) -> Self {
        Self {
            span_source: self.source_hash,
            ..self
        }
    }

    /// Generate the NodeId of the next node of the current owner and record
    /// the CST node's span for it.
    pub fn fresh_id_with_span(&mut self, node: &Node) -> NodeId {
        let id = self.next_id();
        let span = Span::new(node.start_byte(), node.end_byte()).in_source(self.span_source);
        self.span_builder.insert(id, span);
        id
    }

    /// Generate the NodeId of the root `Module` node and record its span.
    pub fn module_id_with_span(&mut self, node: &Node) -> NodeId {
        let id = self.next_id();
        let span = Span::new(node.start_byte(), node.end_byte()).in_source(self.span_source);
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
    /// of them precede the item.
    pub fn in_item<T>(
        &mut self,
        kind: &'static str,
        name: &str,
        start: usize,
        lower: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let module = module_owner(&self.module_path);
        let count = self
            .item_counts
            .entry((module, kind, name.to_owned()))
            .or_default();
        let occurrence = *count;
        *count += 1;
        let item = owner_hash((path_text(&self.module_path), kind, name, occurrence));
        self.span_builder.set_start(self.source_hash, item, start);
        let outer = std::mem::replace(&mut self.owner, item);
        let lowered = lower(self);
        self.owner = outer;
        lowered
    }

    /// Emit a diagnostic error.
    pub fn error(&mut self, span: Span, message: impl Into<String>) {
        if let Some(db) = self.db {
            Diagnostic::new(
                message,
                span.in_source(self.span_source),
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
                span.in_source(self.span_source),
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

/// The owner of the module-level nodes of the module at `path`.
fn module_owner(path: &[Symbol]) -> u64 {
    owner_hash(("mod", path_text(path)))
}

/// The text of a module path, which owner hashes are computed from.
fn path_text(path: &[Symbol]) -> Vec<&str> {
    path.iter().map(Symbol::as_str).collect()
}
