//! LSP server implementation using lsp-server (synchronous).
//!
//! This is a simple synchronous LSP server that handles requests one at a time.

use std::error::Error;
use std::io;

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _,
    PublishDiagnostics,
};
use lsp_types::request::HoverRequest;
use lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, Hover, HoverContents, HoverParams, HoverProviderCapability,
    InitializeParams, MarkupContent, MarkupKind, PublishDiagnosticsParams, ServerCapabilities,
    TextDocumentContentChangeEvent, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, Uri,
};
use ropey::Rope;
use salsa::Database;
use tree_sitter::{InputEdit, Parser, Point, Tree};

use tribute::compile;
use tribute::{SourceFile, TributeDatabaseImpl};

use super::pretty::print_type;
use super::type_index::TypeIndex;
/// Document state stored per-file.
struct Document {
    rope: Rope,
    parser: Parser,
    tree: Option<Tree>,
}

/// Main LSP server state.
struct LspServer {
    connection: Connection,
    documents: std::collections::HashMap<Uri, Document>,
}

impl LspServer {
    fn new(connection: Connection) -> Self {
        Self {
            connection,
            documents: std::collections::HashMap::new(),
        }
    }

    fn run(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        loop {
            let msg = self.connection.receiver.recv()?;
            match msg {
                Message::Request(req) => {
                    if self.connection.handle_shutdown(&req)? {
                        return Ok(());
                    }
                    self.handle_request(req)?;
                }
                Message::Response(_) => {
                    // We don't send requests, so we shouldn't get responses
                }
                Message::Notification(notif) => {
                    self.handle_notification(notif)?;
                }
            }
        }
    }

    fn handle_request(&mut self, req: Request) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some((id, params)) = cast_request::<HoverRequest>(req.clone()) {
            let result = self.hover(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        }
        Ok(())
    }

    fn handle_notification(
        &mut self,
        notif: Notification,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(params) = cast_notification::<DidOpenTextDocument>(notif.clone()) {
            self.did_open(params)?;
        } else if let Some(params) = cast_notification::<DidChangeTextDocument>(notif.clone()) {
            self.did_change(params)?;
        } else if let Some(params) = cast_notification::<DidCloseTextDocument>(notif) {
            self.did_close(params)?;
        }
        Ok(())
    }

    fn did_open(
        &mut self,
        params: DidOpenTextDocumentParams,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let uri = params.text_document.uri;
        let text = params.text_document.text;

        let rope = Rope::from_str(&text);
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parse_with_rope(&mut parser, &rope, None);
        self.documents
            .insert(uri.clone(), Document { rope, parser, tree });

        self.publish_diagnostics(&uri, &text)?;
        Ok(())
    }

    fn did_change(
        &mut self,
        params: DidChangeTextDocumentParams,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let uri = params.text_document.uri;
        let text = {
            let Some(doc) = self.documents.get_mut(&uri) else {
                return Ok(());
            };

            for change in params.content_changes {
                Self::apply_change(doc, change)?;
            }

            doc.rope.to_string()
        };

        self.publish_diagnostics(&uri, &text)?;
        Ok(())
    }

    fn did_close(
        &mut self,
        params: DidCloseTextDocumentParams,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let uri = params.text_document.uri;
        self.documents.remove(&uri);

        // Clear diagnostics
        let params = PublishDiagnosticsParams {
            uri,
            diagnostics: vec![],
            version: None,
        };
        let notif = Notification::new(PublishDiagnostics::METHOD.to_string(), params);
        self.connection.sender.send(Message::Notification(notif))?;
        Ok(())
    }

    fn hover(&self, params: HoverParams) -> Option<Hover> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = self.documents.get(uri)?;
        let offset = offset_from_position(&doc.rope, position.line, position.character)?;

        let text = doc.rope.to_string();

        // Run Salsa compilation
        let db = TributeDatabaseImpl::default();
        let (type_str, span) = db.attach(|db| {
            let source_file = SourceFile::new(db, (**uri).clone(), text.to_string());
            let module = compile(db, source_file);
            let type_index = TypeIndex::build(db, &module);

            type_index.type_at(offset).map(|entry| {
                let type_str = print_type(db, entry.ty);
                (type_str, entry.span)
            })
        })?;

        let contents = HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!("```tribute\n{}\n```", type_str),
        });
        let range = span_to_range(&doc.rope, span);

        Some(Hover {
            contents,
            range: Some(range),
        })
    }

    fn publish_diagnostics(
        &self,
        uri: &Uri,
        text: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Some(doc) = self.documents.get(uri) else {
            return Ok(());
        };

        // Run Salsa compilation
        let db = TributeDatabaseImpl::default();
        let diags = db.attach(|db| {
            let source_file = SourceFile::new(db, (**uri).clone(), text.to_string());
            let result = tribute::compile_with_diagnostics(db, source_file);
            result.diagnostics
        });

        // Convert to LSP diagnostics
        let diagnostics: Vec<Diagnostic> = diags
            .iter()
            .map(|d| {
                let range = span_to_range(&doc.rope, d.span);
                Diagnostic {
                    range,
                    severity: Some(match d.severity {
                        tribute_passes::DiagnosticSeverity::Error => DiagnosticSeverity::ERROR,
                        tribute_passes::DiagnosticSeverity::Warning => DiagnosticSeverity::WARNING,
                        tribute_passes::DiagnosticSeverity::Info => DiagnosticSeverity::INFORMATION,
                    }),
                    message: d.message.clone(),
                    source: Some("tribute".to_string()),
                    ..Default::default()
                }
            })
            .collect();

        let params = PublishDiagnosticsParams {
            uri: uri.clone(),
            diagnostics,
            version: None,
        };
        let notif = Notification::new(PublishDiagnostics::METHOD.to_string(), params);
        self.connection.sender.send(Message::Notification(notif))?;
        Ok(())
    }

    fn apply_change(
        doc: &mut Document,
        change: TextDocumentContentChangeEvent,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        match change.range {
            Some(range) => {
                let start = range.start;
                let end = range.end;
                let start_byte =
                    offset_from_position(&doc.rope, start.line, start.character).ok_or(
                        io::Error::new(io::ErrorKind::InvalidInput, "invalid start position"),
                    )?;
                let old_end_byte = offset_from_position(&doc.rope, end.line, end.character).ok_or(
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid end position"),
                )?;

                let (start_row, start_col) = byte_line_col(&doc.rope, start_byte);
                let (old_end_row, old_end_col) = byte_line_col(&doc.rope, old_end_byte);
                let start_point = Point {
                    row: start_row as usize,
                    column: start_col as usize,
                };
                let old_end_point = Point {
                    row: old_end_row as usize,
                    column: old_end_col as usize,
                };
                let new_end_point = point_after_text(start_point, &change.text);
                let new_end_byte = start_byte + change.text.len();

                let start_char = doc.rope.byte_to_char(start_byte);
                let old_end_char = doc.rope.byte_to_char(old_end_byte);
                doc.rope.remove(start_char..old_end_char);
                doc.rope.insert(start_char, &change.text);

                if let Some(mut tree) = doc.tree.take() {
                    tree.edit(&InputEdit {
                        start_byte,
                        old_end_byte,
                        new_end_byte,
                        start_position: start_point,
                        old_end_position: old_end_point,
                        new_end_position: new_end_point,
                    });
                    doc.tree = parse_with_rope(&mut doc.parser, &doc.rope, Some(&tree));
                } else {
                    doc.tree = parse_with_rope(&mut doc.parser, &doc.rope, None);
                }
            }
            None => {
                doc.rope = Rope::from_str(&change.text);
                doc.tree = parse_with_rope(&mut doc.parser, &doc.rope, None);
            }
        }
        Ok(())
    }
}

/// Start the LSP server.
pub fn serve() -> Result<(), Box<dyn Error + Send + Sync>> {
    let (connection, io_threads) = Connection::stdio();

    // Server capabilities
    let capabilities = ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Options(
            TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(TextDocumentSyncKind::INCREMENTAL),
                ..Default::default()
            },
        )),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        ..Default::default()
    };

    let server_capabilities = serde_json::to_value(capabilities)?;
    let init_params = connection.initialize(server_capabilities)?;
    let _params: InitializeParams = serde_json::from_value(init_params)?;

    let mut server = LspServer::new(connection);
    server.run()?;

    io_threads.join()?;
    Ok(())
}

fn point_after_text(start: Point, text: &str) -> Point {
    let mut row = start.row;
    let mut column = start.column;
    let mut lines = text.split('\n');
    if let Some(first) = lines.next() {
        column += first.len();
    }
    for line in lines {
        row += 1;
        column = line.len();
    }
    Point { row, column }
}

fn parse_with_rope(parser: &mut Parser, rope: &Rope, old_tree: Option<&Tree>) -> Option<Tree> {
    let mut callback = |byte: usize, _: Point| chunk_from_byte(rope, byte);
    parser.parse_with_options(&mut callback, old_tree, None)
}

fn span_to_range(rope: &Rope, span: trunk_ir::Span) -> lsp_types::Range {
    let start = position_from_offset(rope, span.start);
    let end = position_from_offset(rope, span.end);
    lsp_types::Range {
        start: lsp_types::Position {
            line: start.0,
            character: start.1,
        },
        end: lsp_types::Position {
            line: end.0,
            character: end.1,
        },
    }
}

fn position_from_offset(rope: &Rope, offset: usize) -> (u32, u32) {
    let offset = offset.min(rope.len_bytes());
    let char_index = rope.byte_to_char(offset);
    let line = rope.char_to_line(char_index);
    let line_start_char = rope.line_to_char(line);
    let line_slice = rope.line(line);
    let mut end = line_slice.len_chars();
    if end > 0 {
        let last = line_slice.slice(end - 1..end).chars().next().unwrap();
        if last == '\n' {
            end -= 1;
        }
    }
    let slice = line_slice.slice(..end);
    let char_in_line = char_index.saturating_sub(line_start_char);
    let utf16 = slice.char_to_utf16_cu(char_in_line);
    (line as u32, utf16 as u32)
}

fn offset_from_position(rope: &Rope, line: u32, character: u32) -> Option<usize> {
    let line = line as usize;
    if line >= rope.len_lines() {
        return None;
    }
    let line_start_char = rope.line_to_char(line);
    let line_slice = rope.line(line);
    let mut end = line_slice.len_chars();
    if end > 0 {
        let last = line_slice.slice(end - 1..end).chars().next().unwrap();
        if last == '\n' {
            end -= 1;
        }
    }
    let slice = line_slice.slice(..end);
    let utf16_offset = (character as usize).min(slice.len_utf16_cu());
    let char_offset = slice.utf16_cu_to_char(utf16_offset);
    let char_index = line_start_char + char_offset;
    Some(rope.char_to_byte(char_index))
}

fn byte_line_col(rope: &Rope, offset: usize) -> (u32, u32) {
    let offset = offset.min(rope.len_bytes());
    let char_index = rope.byte_to_char(offset);
    let line = rope.char_to_line(char_index);
    let line_start_char = rope.line_to_char(line);
    let line_start_byte = rope.char_to_byte(line_start_char);
    let column = offset - line_start_byte;
    (line as u32, column as u32)
}

fn chunk_from_byte(rope: &Rope, byte: usize) -> &str {
    if byte >= rope.len_bytes() {
        return "";
    }
    let (chunk, chunk_start, _, _) = rope.chunk_at_byte(byte);
    let start = byte - chunk_start;
    &chunk[start..]
}

/// Cast a request to a specific type.
fn cast_request<R: lsp_types::request::Request>(req: Request) -> Option<(RequestId, R::Params)> {
    if req.method == R::METHOD {
        let params = serde_json::from_value(req.params).ok()?;
        Some((req.id, params))
    } else {
        None
    }
}

/// Cast a notification to a specific type.
fn cast_notification<N: lsp_types::notification::Notification>(
    notif: Notification,
) -> Option<N::Params> {
    if notif.method == N::METHOD {
        serde_json::from_value(notif.params).ok()
    } else {
        None
    }
}
