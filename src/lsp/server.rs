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
use salsa::Database;
use tree_sitter::{InputEdit, Parser, Point, Tree};

use tribute::compile;
use tribute::{SourceFile, TributeDatabaseImpl};

use super::pretty::print_type;
use super::type_index::TypeIndex;
use tribute_front::LineIndex;

/// Document state stored per-file.
struct Document {
    line_index: LineIndex,
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

        let line_index = LineIndex::new(&text);
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(&text, None);
        self.documents.insert(
            uri.clone(),
            Document {
                line_index,
                parser,
                tree,
            },
        );

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

            doc.line_index.text().to_string()
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
        let offset = doc.line_index.offset(position.line, position.character)?;

        let text = doc.line_index.text();

        // Run Salsa compilation
        let db = TributeDatabaseImpl::default();
        let (type_str, span) = db.attach(|db| {
            let uri =
                fluent_uri::Uri::parse_from(uri.as_str().to_owned()).expect("valid URI from LSP");
            let source_file = SourceFile::new(db, uri, text.to_string());
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
        let range = doc.line_index.span_to_range(span);

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
            let uri =
                fluent_uri::Uri::parse_from(uri.as_str().to_owned()).expect("valid URI from LSP");
            let source_file = SourceFile::new(db, uri, text.to_string());
            let result = tribute::compile_with_diagnostics(db, source_file);
            result.diagnostics
        });

        // Convert to LSP diagnostics
        let diagnostics: Vec<Diagnostic> = diags
            .iter()
            .map(|d| {
                let range = doc.line_index.span_to_range(d.span);
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
                    doc.line_index
                        .offset(start.line, start.character)
                        .ok_or(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "invalid start position",
                        ))?;
                let old_end_byte =
                    doc.line_index
                        .offset(end.line, end.character)
                        .ok_or(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "invalid end position",
                        ))?;

                let (start_row, start_col) = doc.line_index.byte_line_col(start_byte);
                let (old_end_row, old_end_col) = doc.line_index.byte_line_col(old_end_byte);
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

                doc.line_index.apply_edit(
                    start.line,
                    end.line,
                    start_byte,
                    old_end_byte,
                    &change.text,
                );

                if let Some(mut tree) = doc.tree.take() {
                    tree.edit(&InputEdit {
                        start_byte,
                        old_end_byte,
                        new_end_byte,
                        start_position: start_point,
                        old_end_position: old_end_point,
                        new_end_position: new_end_point,
                    });
                    doc.tree = doc.parser.parse(doc.line_index.text(), Some(&tree));
                } else {
                    doc.tree = doc.parser.parse(doc.line_index.text(), None);
                }
            }
            None => {
                doc.line_index = LineIndex::new(&change.text);
                doc.tree = doc.parser.parse(doc.line_index.text(), None);
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
