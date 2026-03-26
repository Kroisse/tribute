//! Lowering context for CST to AST conversion.

use std::borrow::Cow;

use ropey::Rope;
use salsa::Accumulator;
use tree_sitter::Node;
use trunk_ir::{Span, Symbol};

use crate::ast::{NodeId, SpanMapBuilder};
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
        }
    }

    /// Create a new lowering context with a Salsa database.
    ///
    /// Diagnostics are accumulated directly into Salsa via the database.
    pub fn with_db(db: &'db dyn salsa::Database, source: Rope, source_hash: u64) -> Self {
        Self {
            source,
            source_hash,
            span_builder: SpanMapBuilder::new(),
            db: Some(db),
        }
    }

    /// Generate a NodeId from a CST node and record its span.
    ///
    /// Uses the CST node's `id()` directly, enabling reverse lookup:
    /// given a byte offset, Tree-sitter can find the CST node, and if
    /// that node's ID is in the SpanMap, it corresponds to an AST node.
    pub fn fresh_id_with_span(&mut self, node: &Node) -> NodeId {
        let id = NodeId::from_cst(node, self.source_hash);
        let span = Span::new(node.start_byte(), node.end_byte());
        self.span_builder.insert(id, span);
        id
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

    /// Get a Symbol from a node's text.
    pub fn node_symbol(&self, node: &Node) -> Symbol {
        Symbol::from_dynamic(&self.node_text(node))
    }
}
