//! LSP server implementation using lsp-server (synchronous).
//!
//! This is a simple synchronous LSP server that handles requests one at a time.

use std::error::Error;
use std::io;

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse, Hover,
    HoverContents, HoverParams, HoverProviderCapability, InitializeParams, MarkupContent,
    MarkupKind, PublishDiagnosticsParams, ServerCapabilities, SymbolKind,
    TextDocumentContentChangeEvent, TextDocumentSyncCapability, TextDocumentSyncKind,
    TextDocumentSyncOptions, Uri,
    notification::{
        DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _,
        PublishDiagnostics,
    },
    request::{DocumentSymbolRequest, HoverRequest},
};
use ropey::Rope;
use salsa::{Database, Setter};
use tree_sitter::{InputEdit, Point};

use super::pretty::print_type;
use super::type_index::TypeIndex;
use tribute::{TributeDatabaseImpl, compile, database::parse_with_thread_local};

/// Main LSP server state.
struct LspServer {
    connection: Connection,
    db: TributeDatabaseImpl,
}

impl LspServer {
    fn new(connection: Connection) -> Self {
        Self {
            connection,
            db: TributeDatabaseImpl::default(),
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
        } else if let Some((id, params)) = cast_request::<DocumentSymbolRequest>(req) {
            let result = self.document_symbols(params);
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
        self.db.open_document(&uri, rope);

        self.publish_diagnostics(&uri)?;
        Ok(())
    }

    fn did_change(
        &mut self,
        params: DidChangeTextDocumentParams,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let uri = params.text_document.uri;
        for change in params.content_changes {
            self.apply_change(&uri, change)?;
        }

        self.publish_diagnostics(&uri)?;
        Ok(())
    }

    fn did_close(
        &mut self,
        params: DidCloseTextDocumentParams,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let uri = params.text_document.uri;
        self.db.close_document(&uri);

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

        let rope = self.db.source_cst(uri)?.text(&self.db).clone();
        let offset = offset_from_position(&rope, position.line, position.character)?;
        let source_cst = self.db.source_cst(uri)?;

        // Run Salsa compilation
        let (type_str, span) = self.db.attach(|db| {
            let module = compile(db, source_cst);
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
        let range = span_to_range(&rope, span);

        Some(Hover {
            contents,
            range: Some(range),
        })
    }

    fn document_symbols(&self, params: DocumentSymbolParams) -> Option<DocumentSymbolResponse> {
        use trunk_ir::DialectOp;
        use trunk_ir::dialect::{core, func, ty};

        let uri = &params.text_document.uri;
        let source_cst = self.db.source_cst(uri)?;
        let rope = source_cst.text(&self.db);

        // Run Salsa compilation
        let symbols = self.db.attach(|db| {
            let module = compile(db, source_cst);

            // Extract module name and body
            let core_module = core::Module::from_operation(db, module.as_operation()).ok()?;
            let body = core_module.body(db);

            let mut symbols = Vec::new();

            // Iterate through top-level operations
            for block in body.blocks(db).iter() {
                for op in block.operations(db).iter() {
                    // Try to extract as function definition
                    if let Ok(func_op) = func::Func::from_operation(db, *op) {
                        let name = func_op.sym_name(db).to_string();
                        let location = op.location(db);
                        let span = location.span;

                        let range = span_to_range(rope, span);
                        let selection_range = range; // TODO: Use name span when available

                        symbols.push(DocumentSymbol {
                            name,
                            detail: None,
                            kind: SymbolKind::FUNCTION,
                            tags: None,
                            range,
                            selection_range,
                            children: None,
                            #[allow(deprecated)]
                            deprecated: Some(false),
                        });
                    }

                    // Try to extract as struct definition
                    if let Ok(struct_op) = ty::Struct::from_operation(db, *op) {
                        if let trunk_ir::Attribute::Symbol(sym) = struct_op.sym_name(db) {
                            let name = sym.to_string();
                            let location = op.location(db);
                            let span = location.span;

                            let range = span_to_range(rope, span);
                            let selection_range = range;

                            symbols.push(DocumentSymbol {
                                name,
                                detail: None,
                                kind: SymbolKind::STRUCT,
                                tags: None,
                                range,
                                selection_range,
                                children: None,
                                #[allow(deprecated)]
                                deprecated: Some(false),
                            });
                        }
                    }

                    // Try to extract as enum definition
                    if let Ok(enum_op) = ty::Enum::from_operation(db, *op) {
                        if let trunk_ir::Attribute::Symbol(sym) = enum_op.sym_name(db) {
                            let name = sym.to_string();
                            let location = op.location(db);
                            let span = location.span;

                            let range = span_to_range(rope, span);
                            let selection_range = range;

                            symbols.push(DocumentSymbol {
                                name,
                                detail: None,
                                kind: SymbolKind::ENUM,
                                tags: None,
                                range,
                                selection_range,
                                children: None,
                                #[allow(deprecated)]
                                deprecated: Some(false),
                            });
                        }
                    }

                    // Try to extract as ability definition
                    if let Ok(ability_op) = ty::Ability::from_operation(db, *op) {
                        if let trunk_ir::Attribute::Symbol(sym) = ability_op.sym_name(db) {
                            let name = sym.to_string();
                            let location = op.location(db);
                            let span = location.span;

                            let range = span_to_range(rope, span);
                            let selection_range = range;

                            symbols.push(DocumentSymbol {
                                name,
                                detail: None,
                                kind: SymbolKind::INTERFACE,
                                tags: None,
                                range,
                                selection_range,
                                children: None,
                                #[allow(deprecated)]
                                deprecated: Some(false),
                            });
                        }
                    }
                }
            }

            Some(symbols)
        })?;

        Some(DocumentSymbolResponse::Nested(symbols))
    }

    fn publish_diagnostics(&self, uri: &Uri) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Some(source_cst) = self.db.source_cst(uri) else {
            return Ok(());
        };
        let rope = source_cst.text(&self.db);

        // Run Salsa compilation
        let diags = self
            .db
            .attach(|db| tribute::compile_with_diagnostics(db, source_cst).diagnostics);

        // Convert to LSP diagnostics
        let diagnostics: Vec<Diagnostic> = diags
            .iter()
            .map(|d| {
                let range = span_to_range(rope, d.span);
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
        &mut self,
        uri: &Uri,
        change: TextDocumentContentChangeEvent,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Some(doc) = self.db.source_cst(uri) else {
            return Ok(());
        };

        match change.range {
            Some(range) => {
                let start = range.start;
                let end = range.end;
                let start_byte =
                    offset_from_position(doc.text(&self.db), start.line, start.character).ok_or(
                        io::Error::new(io::ErrorKind::InvalidInput, "invalid start position"),
                    )?;
                let old_end_byte =
                    offset_from_position(doc.text(&self.db), end.line, end.character).ok_or(
                        io::Error::new(io::ErrorKind::InvalidInput, "invalid end position"),
                    )?;

                let (start_row, start_col) = byte_line_col(doc.text(&self.db), start_byte);
                let (old_end_row, old_end_col) = byte_line_col(doc.text(&self.db), old_end_byte);
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

                let mut rope = doc.text(&self.db).clone();
                let start_char = rope.byte_to_char(start_byte);
                let old_end_char = rope.byte_to_char(old_end_byte);
                rope.remove(start_char..old_end_char);
                rope.insert(start_char, &change.text);

                let current_tree = doc.tree(&self.db).clone();
                let updated_tree = if let Some(mut tree) = current_tree {
                    tree.edit(&InputEdit {
                        start_byte,
                        old_end_byte,
                        new_end_byte,
                        start_position: start_point,
                        old_end_position: old_end_point,
                        new_end_position: new_end_point,
                    });
                    parse_with_thread_local(&rope, Some(&tree))
                } else {
                    parse_with_thread_local(&rope, None)
                };
                doc.set_text(&mut self.db).to(rope.clone());
                doc.set_tree(&mut self.db).to(updated_tree);
            }
            None => {
                let rope = Rope::from_str(&change.text);
                doc.set_text(&mut self.db).to(rope.clone());
                doc.set_tree(&mut self.db)
                    .to(parse_with_thread_local(&rope, None));
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
        document_symbol_provider: Some(lsp_types::OneOf::Left(true)),
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
