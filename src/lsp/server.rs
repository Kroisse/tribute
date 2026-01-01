//! LSP server implementation using lsp-server (synchronous).
//!
//! This is a simple synchronous LSP server that handles requests one at a time.

use std::error::Error;
use std::io;

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionList, CompletionOptions, CompletionParams,
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse,
    GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, Location, MarkupContent, MarkupKind,
    PublishDiagnosticsParams, ReferenceParams, ServerCapabilities, SignatureHelp,
    SignatureHelpOptions, SignatureHelpParams, SymbolKind, TextDocumentContentChangeEvent,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions, Uri,
    notification::{
        DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _,
        PublishDiagnostics,
    },
    request::{
        Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, References,
        SignatureHelpRequest,
    },
};
use ropey::Rope;
use salsa::{Database, Setter};
use tree_sitter::{InputEdit, Point};

use super::call_index::{CallIndex, get_param_names};
use super::completion_index::{CompletionIndex, CompletionKind};
use super::definition_index::DefinitionIndex;
use super::pretty::{format_signature, print_type};
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
        tracing::debug!(method = %req.method, "Received request");

        if let Some((id, params)) = cast_request::<HoverRequest>(req.clone()) {
            let result = self.hover(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<GotoDefinition>(req.clone()) {
            let result = self.goto_definition(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<DocumentSymbolRequest>(req.clone()) {
            let result = self.document_symbols(params);
            tracing::debug!(symbols = ?result.as_ref().map(|r| match r {
                DocumentSymbolResponse::Flat(v) => v.len(),
                DocumentSymbolResponse::Nested(v) => v.len(),
            }), "Document symbols response");
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<SignatureHelpRequest>(req.clone()) {
            let result = self.signature_help(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<References>(req.clone()) {
            let result = self.find_references(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<Completion>(req) {
            let result = self.completion(params);
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

        tracing::info!(uri = ?uri, "Document opened");

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

        tracing::debug!(
            line = position.line,
            character = position.character,
            "Hover request"
        );

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

        tracing::debug!(type_str = %type_str, "Found type information");

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

    fn goto_definition(&self, params: GotoDefinitionParams) -> Option<GotoDefinitionResponse> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        tracing::debug!(
            line = position.line,
            character = position.character,
            "Go to Definition request"
        );

        let rope = self.db.source_cst(uri)?.text(&self.db).clone();
        let offset = offset_from_position(&rope, position.line, position.character)?;
        let source_cst = self.db.source_cst(uri)?;

        // Run Salsa compilation and build definition index
        let definition = self.db.attach(|db| {
            let module = compile(db, source_cst);
            let index = DefinitionIndex::build(db, &module);
            index.definition_at(offset).cloned()
        })?;

        tracing::debug!(
            name = %definition.name,
            kind = ?definition.kind,
            "Found definition"
        );

        let range = span_to_range(&rope, definition.span);
        let location = lsp_types::Location {
            uri: uri.clone(),
            range,
        };

        Some(GotoDefinitionResponse::Scalar(location))
    }

    fn find_references(&self, params: ReferenceParams) -> Option<Vec<Location>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let include_declaration = params.context.include_declaration;

        tracing::debug!(
            line = position.line,
            character = position.character,
            include_declaration,
            "Find References request"
        );

        let rope = self.db.source_cst(uri)?.text(&self.db).clone();
        let offset = offset_from_position(&rope, position.line, position.character)?;
        let source_cst = self.db.source_cst(uri)?;

        let locations = self.db.attach(|db| {
            let module = compile(db, source_cst);
            let index = DefinitionIndex::build(db, &module);

            let (symbol, refs) = index.references_at(offset)?;

            let mut locations = Vec::new();

            // Optionally include the definition itself
            if include_declaration && let Some(def) = index.definition_of(symbol) {
                locations.push(Location {
                    uri: uri.clone(),
                    range: span_to_range(&rope, def.span),
                });
            }

            // Add all references
            for reference in refs {
                locations.push(Location {
                    uri: uri.clone(),
                    range: span_to_range(&rope, reference.span),
                });
            }

            Some(locations)
        })?;

        tracing::debug!(count = locations.len(), "Found references");

        Some(locations)
    }

    fn completion(&self, params: CompletionParams) -> Option<CompletionList> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        tracing::debug!(
            line = position.line,
            character = position.character,
            "Completion request"
        );

        let rope = self.db.source_cst(uri)?.text(&self.db).clone();
        let offset = offset_from_position(&rope, position.line, position.character)?;
        let source_cst = self.db.source_cst(uri)?;

        // Extract prefix from current position (for filtering)
        let prefix = extract_completion_prefix(&rope, offset);

        let items = self.db.attach(|db| {
            let module = compile(db, source_cst);
            let index = CompletionIndex::build(db, &module);

            // Get expression completions filtered by prefix
            let mut completions: Vec<_> = index
                .complete_expression(&prefix)
                .into_iter()
                .cloned()
                .collect();

            // Add keyword completions
            completions.extend(CompletionIndex::complete_keywords(&prefix));

            Some(completions)
        })?;

        // Convert to LSP CompletionItems
        let completion_items: Vec<CompletionItem> = items
            .into_iter()
            .map(|entry| CompletionItem {
                label: entry.name,
                kind: Some(match entry.kind {
                    CompletionKind::Function => CompletionItemKind::FUNCTION,
                    CompletionKind::Constructor => CompletionItemKind::CONSTRUCTOR,
                    CompletionKind::Keyword => CompletionItemKind::KEYWORD,
                    CompletionKind::Ability => CompletionItemKind::INTERFACE,
                }),
                detail: entry.detail,
                ..Default::default()
            })
            .collect();

        tracing::debug!(count = completion_items.len(), "Completion items");

        Some(CompletionList {
            is_incomplete: false,
            items: completion_items,
        })
    }

    fn document_symbols(&self, params: DocumentSymbolParams) -> Option<DocumentSymbolResponse> {
        let uri = &params.text_document.uri;
        let source_cst = self.db.source_cst(uri)?;
        let rope = source_cst.text(&self.db);

        tracing::debug!(uri = ?uri, "Document symbols request");

        let result = self
            .db
            .attach(|db| {
                let module = compile(db, source_cst);
                Some(extract_symbols_from_module(db, &module, rope))
            })
            .map(DocumentSymbolResponse::Nested);

        if let Some(DocumentSymbolResponse::Nested(ref symbols)) = result {
            tracing::debug!(count = symbols.len(), "Found document symbols");
        }

        result
    }

    fn signature_help(&self, params: SignatureHelpParams) -> Option<SignatureHelp> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        tracing::debug!(
            line = position.line,
            character = position.character,
            "Signature help request"
        );

        let source_cst = self.db.source_cst(uri)?;
        let rope = source_cst.text(&self.db).clone();
        let offset = offset_from_position(&rope, position.line, position.character)?;

        // Use tree-sitter to find the enclosing call expression
        let tree = source_cst.tree(&self.db).as_ref()?;
        let call_info = find_enclosing_call(tree, &rope, offset)?;

        tracing::debug!(
            callee = %call_info.callee_name,
            active_param = call_info.active_param,
            "Found call expression"
        );

        // Extract doc comment from CST (before entering Salsa context)
        let doc_comment = find_doc_comment(tree, &rope, &call_info.callee_name);

        // Compile and look up the function type
        let callee_name = call_info.callee_name.clone();
        let active_param = call_info.active_param;

        let result = self.db.attach(|db| {
            let module = compile(db, source_cst);

            // Find the function definition by name
            let callee_sym = trunk_ir::Symbol::from_dynamic(&callee_name);
            let callee_qname = trunk_ir::QualifiedName::simple(callee_sym);
            let func_ty = CallIndex::find_function_type(db, &module, &callee_qname)?;

            // Get parameter names from the function definition
            let param_names = get_param_names(db, &module, &callee_qname);

            // Format the signature
            Some(format_signature(
                db,
                func_ty,
                &callee_name,
                &param_names,
                doc_comment.as_deref(),
                active_param,
            ))
        })?;

        Some(result)
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
pub fn serve(_log_level: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
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
        definition_provider: Some(lsp_types::OneOf::Left(true)),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
            retrigger_characters: Some(vec![")".to_string()]),
            work_done_progress_options: Default::default(),
        }),
        references_provider: Some(lsp_types::OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
            resolve_provider: Some(false),
            ..Default::default()
        }),
        ..Default::default()
    };

    let server_capabilities = serde_json::to_value(&capabilities)?;
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

/// Extract the identifier prefix at the cursor position for completion filtering.
fn extract_completion_prefix(rope: &Rope, offset: usize) -> String {
    if offset == 0 {
        return String::new();
    }

    let text = rope.slice(..);
    let mut start = offset;

    // Walk backwards to find start of identifier
    while start > 0 {
        let char_idx = rope.byte_to_char(start.saturating_sub(1));
        if char_idx >= text.len_chars() {
            break;
        }
        let c = text.char(char_idx);
        if c.is_alphanumeric() || c == '_' {
            start = start.saturating_sub(c.len_utf8());
        } else {
            break;
        }
    }

    if start >= offset {
        return String::new();
    }

    // Extract the prefix string
    rope.slice(rope.byte_to_char(start)..rope.byte_to_char(offset))
        .to_string()
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

/// Extract all symbols from a TrunkIR module.
fn extract_symbols_from_module<'db>(
    db: &'db dyn salsa::Database,
    module: &trunk_ir::dialect::core::Module<'db>,
    rope: &Rope,
) -> Vec<DocumentSymbol> {
    module
        .body(db)
        .blocks(db)
        .iter()
        .flat_map(|block| block.operations(db))
        .filter_map(|op| extract_symbol_from_operation(db, op, rope))
        .collect()
}

/// Extract a DocumentSymbol from a single operation, if applicable.
#[allow(clippy::collapsible_if)]
fn extract_symbol_from_operation<'db>(
    db: &'db dyn salsa::Database,
    op: &trunk_ir::Operation<'db>,
    rope: &Rope,
) -> Option<DocumentSymbol> {
    use tribute_ir::dialect::tribute;
    use trunk_ir::DialectOp;
    use trunk_ir::dialect::{core, func};

    if let Ok(module_op) = core::Module::from_operation(db, *op) {
        return Some(create_symbol(
            module_op.sym_name(db),
            SymbolKind::MODULE,
            op.location(db).span,
            rope,
            extract_symbols_from_module(db, &module_op, rope),
        ));
    }

    // Try function
    if let Ok(func_op) = func::Func::from_operation(db, *op) {
        return Some(create_symbol(
            func_op.name(db),
            SymbolKind::FUNCTION,
            op.location(db).span,
            rope,
            vec![],
        ));
    }

    // Try struct
    if let Ok(struct_op) = tribute::StructDef::from_operation(db, *op) {
        let name = struct_op.sym_name(db);
        return Some(create_symbol(
            name,
            SymbolKind::STRUCT,
            op.location(db).span,
            rope,
            vec![],
        ));
    }

    // Try enum
    if let Ok(enum_op) = tribute::EnumDef::from_operation(db, *op) {
        let name = enum_op.sym_name(db);
        return Some(create_symbol(
            name,
            SymbolKind::ENUM,
            op.location(db).span,
            rope,
            vec![],
        ));
    }

    // Try ability
    if let Ok(ability_op) = tribute::AbilityDef::from_operation(db, *op) {
        let name = ability_op.sym_name(db);
        return Some(create_symbol(
            name,
            SymbolKind::INTERFACE,
            op.location(db).span,
            rope,
            vec![],
        ));
    }

    None
}

/// Create a DocumentSymbol with the given parameters.
fn create_symbol(
    name: trunk_ir::Symbol,
    kind: SymbolKind,
    span: trunk_ir::Span,
    rope: &Rope,
    children: Vec<DocumentSymbol>,
) -> DocumentSymbol {
    let range = span_to_range(rope, span);
    let selection_range = range; // TODO: Use name span when available
    let children = if children.is_empty() {
        None
    } else {
        Some(children)
    };

    DocumentSymbol {
        name: name.to_string(),
        detail: None,
        kind,
        tags: None,
        range,
        selection_range,
        children,
        #[allow(deprecated)]
        deprecated: None,
    }
}

// =============================================================================
// Signature Help Helpers
// =============================================================================

/// Information about a function call at a cursor position.
struct CallInfo {
    /// Name of the function being called.
    callee_name: String,
    /// Index of the currently active parameter (0-based).
    active_param: u32,
}

/// Find the enclosing call expression at the given offset using tree-sitter.
fn find_enclosing_call(tree: &tree_sitter::Tree, rope: &Rope, offset: usize) -> Option<CallInfo> {
    let root = tree.root_node();

    // Find the smallest node containing the offset
    let mut node = root.descendant_for_byte_range(offset, offset)?;

    // Walk up the tree to find a call_expression
    loop {
        if node.kind() == "call_expression" {
            return extract_call_info(node, rope, offset);
        }

        node = node.parent()?;
    }
}

/// Extract call information from a call_expression node.
fn extract_call_info(
    call_node: tree_sitter::Node,
    rope: &Rope,
    cursor_offset: usize,
) -> Option<CallInfo> {
    // Get the callee (first child, typically an identifier or member expression)
    let callee_node = call_node.child_by_field_name("function")?;
    let callee_start = callee_node.start_byte();
    let callee_end = callee_node.end_byte();
    let callee_name = rope
        .get_byte_slice(callee_start..callee_end)
        .map(|s| s.to_string())?;

    // Find the arguments node
    let args_node = call_node.child_by_field_name("arguments")?;

    // Count commas before the cursor to determine active parameter
    let active_param = count_commas_before(args_node, cursor_offset);

    Some(CallInfo {
        callee_name,
        active_param,
    })
}

/// Count the number of commas before the cursor position within an argument list.
fn count_commas_before(args_node: tree_sitter::Node, cursor_offset: usize) -> u32 {
    let mut count = 0;
    let mut cursor = args_node.walk();

    // Skip the opening parenthesis
    if !cursor.goto_first_child() {
        return 0;
    }

    loop {
        let node = cursor.node();
        let node_end = node.end_byte();

        // Stop if we've passed the cursor
        if node.start_byte() >= cursor_offset {
            break;
        }

        // Count commas
        if node.kind() == "," && node_end <= cursor_offset {
            count += 1;
        }

        if !cursor.goto_next_sibling() {
            break;
        }
    }

    count
}

/// Find the doc comment for a function definition by name.
fn find_doc_comment(tree: &tree_sitter::Tree, rope: &Rope, func_name: &str) -> Option<String> {
    let root = tree.root_node();

    // Find the function definition with the given name
    find_function_node(&root, rope, func_name)
        .and_then(|func_node| extract_doc_comment(func_node, rope))
}

/// Find a function definition node by name.
fn find_function_node<'tree>(
    node: &tree_sitter::Node<'tree>,
    rope: &Rope,
    func_name: &str,
) -> Option<tree_sitter::Node<'tree>> {
    if node.kind() == "function_definition" {
        // Get the function name
        if let Some(name_node) = node.child_by_field_name("name") {
            let start = name_node.start_byte();
            let end = name_node.end_byte();
            if let Some(name) = rope.get_byte_slice(start..end)
                && name == func_name
            {
                return Some(*node);
            }
        }
    }

    // Recurse into children
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            if let Some(found) = find_function_node(&cursor.node(), rope, func_name) {
                return Some(found);
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    None
}

/// Extract doc comments from preceding siblings of a node.
fn extract_doc_comment(node: tree_sitter::Node, rope: &Rope) -> Option<String> {
    let mut comments = Vec::new();
    let mut current = node;

    // Walk backwards through siblings to collect doc comments
    while let Some(prev) = current.prev_sibling() {
        current = prev;
        let kind = current.kind();

        if kind == "line_doc_comment" || kind == "block_doc_comment" {
            let start = current.start_byte();
            let end = current.end_byte();
            if let Some(text) = rope.get_byte_slice(start..end) {
                let text = text.to_string();
                // Strip the comment markers
                let cleaned = if kind == "line_doc_comment" {
                    text.strip_prefix("///").unwrap_or(&text).trim()
                } else {
                    // Block doc comment: /** ... */
                    text.strip_prefix("/**")
                        .and_then(|s| s.strip_suffix("*/"))
                        .map(|s| s.trim())
                        .unwrap_or(&text)
                };
                comments.push(cleaned.to_string());
            }
        } else if kind != "line_comment" && kind != "block_comment" {
            // Stop at non-comment nodes
            break;
        }
    }

    if comments.is_empty() {
        None
    } else {
        // Reverse because we collected in reverse order
        comments.reverse();
        Some(comments.join("\n"))
    }
}
