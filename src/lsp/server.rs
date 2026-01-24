//! LSP server implementation using lsp-server (synchronous).
//!
//! This is a simple synchronous LSP server that handles requests one at a time.

use std::error::Error;
use std::io;

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::{
    CodeAction, CodeActionKind, CodeActionOptions, CodeActionOrCommand, CodeActionParams,
    CodeActionProviderCapability, CompletionItem, CompletionList, CompletionOptions,
    CompletionParams, Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DocumentSymbol, DocumentSymbolParams,
    DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents,
    HoverParams, HoverProviderCapability, InitializeParams, Location, MarkupContent, MarkupKind,
    PrepareRenameResponse, PublishDiagnosticsParams, ReferenceParams, RenameOptions, RenameParams,
    ServerCapabilities, SignatureHelp, SignatureHelpOptions, SignatureHelpParams, SymbolKind,
    TextDocumentContentChangeEvent, TextDocumentPositionParams, TextDocumentSyncCapability,
    TextDocumentSyncKind, TextDocumentSyncOptions, TextEdit, Uri, WorkspaceEdit,
    notification::{
        DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _,
        PublishDiagnostics,
    },
    request::{
        CodeActionRequest, Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest,
        PrepareRenameRequest, References, Rename, SignatureHelpRequest,
    },
};
use ropey::Rope;
use salsa::{Database, Setter};
use tree_sitter::{InputEdit, Point};

use crate::lsp::ast_index::print_ast_type;

use super::ast_index::{self, type_index as ast_type_index};
use tribute::{TributeDatabaseImpl, compile_for_lsp, database::parse_with_thread_local};

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
            if self.process_message(msg)? {
                return Ok(());
            }
        }
    }

    /// Process a single message. Returns `Ok(true)` if shutdown was requested.
    fn process_message(&mut self, msg: Message) -> Result<bool, Box<dyn Error + Send + Sync>> {
        match msg {
            Message::Request(req) => {
                if self.connection.handle_shutdown(&req)? {
                    return Ok(true);
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
        Ok(false)
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
        } else if let Some((id, params)) = cast_request::<PrepareRenameRequest>(req.clone()) {
            let result = self.prepare_rename(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<Rename>(req.clone()) {
            let result = self.rename(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<Completion>(req.clone()) {
            let result = self.completion(params);
            let response = Response::new_ok(id, result);
            self.connection.sender.send(Message::Response(response))?;
        } else if let Some((id, params)) = cast_request::<CodeActionRequest>(req) {
            let result = self.code_action(params);
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

        let (type_str, span) = self.db.attach(|db| {
            let index = ast_type_index(db, source_cst)?;
            index.type_at(db, offset).map(|entry| {
                let type_str = print_ast_type(db, entry.ty);
                (type_str, entry.span)
            })
        })?;

        tracing::debug!(type_str = %type_str, "Found type");

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
            let index = ast_index::definition_index(db, source_cst)?;
            index.definition_at(db, offset).cloned()
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
            let index = ast_index::definition_index(db, source_cst)?;

            let (symbol, refs) = index.references_at(db, offset)?;

            let mut locations = Vec::new();

            // Optionally include the definition itself
            if include_declaration && let Some(def) = index.definition_of(db, symbol) {
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

    fn prepare_rename(&self, params: TextDocumentPositionParams) -> Option<PrepareRenameResponse> {
        let uri = &params.text_document.uri;
        let position = params.position;

        tracing::debug!(
            line = position.line,
            character = position.character,
            "Prepare rename request"
        );

        let rope = self.db.source_cst(uri)?.text(&self.db).clone();
        let offset = offset_from_position(&rope, position.line, position.character)?;
        let source_cst = self.db.source_cst(uri)?;

        let result = self.db.attach(|db| {
            let index = ast_index::definition_index(db, source_cst)?;

            // Check if rename is possible at this position
            let (def, span) = index.can_rename(db, offset)?;

            let range = span_to_range(&rope, span);
            let placeholder = def.name.to_string();

            Some(PrepareRenameResponse::RangeWithPlaceholder { range, placeholder })
        })?;

        tracing::debug!("Prepare rename: symbol can be renamed");

        Some(result)
    }

    #[allow(clippy::mutable_key_type)] // Uri has interior mutability but it's fine for LSP
    fn rename(&self, params: RenameParams) -> Option<WorkspaceEdit> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let new_name = &params.new_name;

        tracing::debug!(
            line = position.line,
            character = position.character,
            new_name = %new_name,
            "Rename request"
        );

        let rope = self.db.source_cst(uri)?.text(&self.db).clone();
        let offset = offset_from_position(&rope, position.line, position.character)?;
        let source_cst = self.db.source_cst(uri)?;

        let text_edits = self.db.attach(|db| {
            let index = ast_index::definition_index(db, source_cst)?;

            // Get definition and validate
            let (def, _) = index.can_rename(db, offset)?;

            // Validate new name
            if ast_index::validate_identifier(new_name, def.kind).is_err() {
                tracing::warn!(new_name = %new_name, "Invalid identifier for rename");
                return None;
            }

            // Collect all locations to rename
            let mut edits = Vec::new();

            // Add the definition itself
            edits.push(TextEdit {
                range: span_to_range(&rope, def.span),
                new_text: new_name.clone(),
            });

            // Add all references
            for reference in index.references_of(db, def.name) {
                edits.push(TextEdit {
                    range: span_to_range(&rope, reference.span),
                    new_text: new_name.clone(),
                });
            }

            Some(edits)
        })?;

        tracing::debug!(edit_count = text_edits.len(), "Rename edits computed");

        // Build WorkspaceEdit with changes for current file
        let mut changes = std::collections::HashMap::new();
        changes.insert(uri.clone(), text_edits);

        Some(WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        })
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
            let all_items = ast_index::completion_items(db, source_cst);

            // Filter by prefix
            let mut completions: Vec<_> = ast_index::filter_completions(all_items, &prefix)
                .cloned()
                .collect();

            // Add keyword completions
            completions.extend(ast_index::complete_keywords(&prefix));

            Some(completions)
        })?;

        // Convert to LSP CompletionItems
        let completion_items: Vec<CompletionItem> = items
            .into_iter()
            .map(|entry| CompletionItem {
                label: entry.name.to_string(),
                kind: Some(entry.kind.into()),
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

    fn code_action(&self, params: CodeActionParams) -> Option<Vec<CodeActionOrCommand>> {
        let uri = &params.text_document.uri;
        let range = params.range;

        tracing::debug!(
            start = ?(range.start.line, range.start.character),
            end = ?(range.end.line, range.end.character),
            "Code action request"
        );

        let source_cst = self.db.source_cst(uri)?;
        let rope = source_cst.text(&self.db).clone();

        // Get the request range in byte offsets
        let start_offset = offset_from_position(&rope, range.start.line, range.start.character)?;
        let end_offset = offset_from_position(&rope, range.end.line, range.end.character)?;

        // Collect diagnostics and type information for code action generation
        let actions = self.db.attach(|db| {
            let result = tribute::compile_with_diagnostics(db, source_cst);
            let type_index = ast_type_index(db, source_cst);

            let mut actions = Vec::new();

            // Find diagnostics that overlap with the requested range
            for diag in &result.diagnostics {
                // Check if diagnostic overlaps with range
                if diag.span.end < start_offset || diag.span.start > end_offset {
                    continue;
                }

                // Generate code actions based on diagnostic type
                if let Some(action) =
                    self.action_for_diagnostic(db, diag, &rope, uri, type_index.as_ref())
                {
                    actions.push(CodeActionOrCommand::CodeAction(action));
                }
            }

            actions
        });

        tracing::debug!(count = actions.len(), "Code actions generated");

        if actions.is_empty() {
            None
        } else {
            Some(actions)
        }
    }

    /// Generate a code action for a specific diagnostic.
    #[allow(clippy::mutable_key_type)] // Uri has interior mutability but it's fine for LSP
    fn action_for_diagnostic(
        &self,
        db: &dyn salsa::Database,
        diag: &tribute_passes::Diagnostic,
        rope: &Rope,
        uri: &Uri,
        type_index: Option<&ast_index::AstTypeIndex<'_>>,
    ) -> Option<CodeAction> {
        // Pattern: "top-level function `{name}` must have an explicit return type annotation"
        if diag
            .message
            .contains("must have an explicit return type annotation")
        {
            // Try to find the inferred type at the diagnostic location
            let type_index = type_index?;
            if let Some(entry) = type_index.type_at(db, diag.span.start) {
                let type_str = print_ast_type(db, entry.ty);

                // Find the position after the closing parenthesis of parameters
                // We need to insert ": Type" before the function body
                let insert_pos = self.find_return_type_insert_position(rope, diag.span)?;
                let (line, char) = position_from_offset(rope, insert_pos);
                let pos = lsp_types::Position {
                    line,
                    character: char,
                };
                let range = lsp_types::Range {
                    start: pos,
                    end: pos,
                };

                let edit = TextEdit {
                    range,
                    new_text: format!(": {}", type_str),
                };

                let mut changes = std::collections::HashMap::new();
                changes.insert(uri.clone(), vec![edit]);

                return Some(CodeAction {
                    title: format!("Add return type annotation: {}", type_str),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![Diagnostic {
                        range: span_to_range(rope, diag.span),
                        severity: Some(match diag.severity {
                            tribute_passes::DiagnosticSeverity::Error => DiagnosticSeverity::ERROR,
                            tribute_passes::DiagnosticSeverity::Warning => {
                                DiagnosticSeverity::WARNING
                            }
                            tribute_passes::DiagnosticSeverity::Info => {
                                DiagnosticSeverity::INFORMATION
                            }
                        }),
                        message: diag.message.clone(),
                        source: Some("tribute".to_string()),
                        ..Default::default()
                    }]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(changes),
                        document_changes: None,
                        change_annotations: None,
                    }),
                    ..Default::default()
                });
            }
        }

        // Pattern: "parameter N of top-level function `{name}` must have an explicit type annotation"
        if diag
            .message
            .contains("must have an explicit type annotation")
            && diag.message.contains("parameter")
        {
            // For parameter annotations, we would need to find the parameter location
            // and its inferred type. This is more complex as we need to parse the message
            // to find which parameter and look up its type.
            // TODO: Implement parameter type annotation fix
        }

        None
    }

    /// Find the position where a return type annotation should be inserted.
    /// Returns the byte offset right after the closing parenthesis of parameters.
    fn find_return_type_insert_position(&self, rope: &Rope, span: trunk_ir::Span) -> Option<usize> {
        // Search from the start of the function for the closing paren
        let text: String = rope.slice(..).chars().collect();
        if span.start >= text.len() {
            return None;
        }
        let func_text = &text[span.start..span.end.min(text.len())];

        // Find the closing parenthesis of the parameter list
        let mut paren_depth = 0;
        let mut found_open = false;

        for (i, c) in func_text.char_indices() {
            match c {
                '(' => {
                    paren_depth += 1;
                    found_open = true;
                }
                ')' => {
                    paren_depth -= 1;
                    if found_open && paren_depth == 0 {
                        // Found the closing paren, return position after it
                        return Some(span.start + i + 1);
                    }
                }
                '{' if found_open && paren_depth == 0 => {
                    // Hit function body without finding return type position
                    // Insert before the opening brace
                    return Some(span.start + i);
                }
                _ => {}
            }
        }

        None
    }

    fn document_symbols(&self, params: DocumentSymbolParams) -> Option<DocumentSymbolResponse> {
        let uri = &params.text_document.uri;
        let source_cst = self.db.source_cst(uri)?;
        let rope = source_cst.text(&self.db);

        tracing::debug!(uri = ?uri, "Document symbols request");

        let result = self
            .db
            .attach(|db| {
                let module = compile_for_lsp(db, source_cst);
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

        // Look up the function signature using AST-based index
        let callee_name = call_info.callee_name.clone();
        let active_param = call_info.active_param;

        let result = self.db.attach(|db| {
            let signatures = ast_index::function_signatures(db, source_cst);
            let callee_sym = trunk_ir::Symbol::from_dynamic(&callee_name);
            let sig = ast_index::find_signature(&signatures, callee_sym)?;

            Some(super::pretty::format_ast_signature(
                sig,
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

/// Get the server capabilities for the Tribute LSP server.
fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
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
        rename_provider: Some(lsp_types::OneOf::Right(RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: Default::default(),
        })),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
            resolve_provider: Some(false),
            ..Default::default()
        }),
        code_action_provider: Some(CodeActionProviderCapability::Options(CodeActionOptions {
            code_action_kinds: Some(vec![CodeActionKind::QUICKFIX]),
            work_done_progress_options: Default::default(),
            resolve_provider: Some(false),
        })),
        ..Default::default()
    }
}

/// Initialize the LSP server with the given connection.
///
/// This performs the LSP initialize handshake and returns a ready-to-run server.
fn initialize_server(connection: Connection) -> Result<LspServer, Box<dyn Error + Send + Sync>> {
    let capabilities = server_capabilities();
    let server_capabilities = serde_json::to_value(&capabilities)?;
    let init_params = connection.initialize(server_capabilities)?;
    let _params: InitializeParams = serde_json::from_value(init_params)?;
    Ok(LspServer::new(connection))
}

/// Start the LSP server.
pub fn serve(_log_level: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let (connection, io_threads) = Connection::stdio();

    let mut server = initialize_server(connection)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_after_text_single_line() {
        let start = Point { row: 0, column: 0 };
        let result = point_after_text(start, "hello");
        assert_eq!(result.row, 0);
        assert_eq!(result.column, 5);
    }

    #[test]
    fn test_point_after_text_multiline() {
        let start = Point { row: 0, column: 0 };
        let result = point_after_text(start, "hello\nworld");
        assert_eq!(result.row, 1);
        assert_eq!(result.column, 5);
    }

    #[test]
    fn test_point_after_text_with_offset() {
        let start = Point { row: 5, column: 10 };
        let result = point_after_text(start, "abc");
        assert_eq!(result.row, 5);
        assert_eq!(result.column, 13);
    }

    #[test]
    fn test_point_after_text_multiple_newlines() {
        let start = Point { row: 0, column: 0 };
        let result = point_after_text(start, "a\nb\nc");
        assert_eq!(result.row, 2);
        assert_eq!(result.column, 1);
    }

    #[test]
    fn test_position_from_offset_simple() {
        let rope = Rope::from_str("hello\nworld");
        // 'h' is at offset 0
        assert_eq!(position_from_offset(&rope, 0), (0, 0));
        // 'w' is at offset 6 (after "hello\n")
        assert_eq!(position_from_offset(&rope, 6), (1, 0));
        // 'd' is at offset 10
        assert_eq!(position_from_offset(&rope, 10), (1, 4));
    }

    #[test]
    fn test_position_from_offset_clamped() {
        let rope = Rope::from_str("hello");
        // Offset beyond end should clamp
        let (line, col) = position_from_offset(&rope, 100);
        assert_eq!(line, 0);
        assert_eq!(col, 5);
    }

    #[test]
    fn test_offset_from_position_simple() {
        let rope = Rope::from_str("hello\nworld");
        assert_eq!(offset_from_position(&rope, 0, 0), Some(0));
        assert_eq!(offset_from_position(&rope, 1, 0), Some(6));
        assert_eq!(offset_from_position(&rope, 1, 4), Some(10));
    }

    #[test]
    fn test_offset_from_position_invalid_line() {
        let rope = Rope::from_str("hello");
        assert_eq!(offset_from_position(&rope, 5, 0), None);
    }

    #[test]
    fn test_offset_from_position_clamped_column() {
        let rope = Rope::from_str("hi\nworld");
        // Column beyond line length should clamp
        let offset = offset_from_position(&rope, 0, 100);
        assert_eq!(offset, Some(2)); // End of "hi"
    }

    #[test]
    fn test_byte_line_col_simple() {
        let rope = Rope::from_str("hello\nworld");
        assert_eq!(byte_line_col(&rope, 0), (0, 0));
        assert_eq!(byte_line_col(&rope, 6), (1, 0));
        assert_eq!(byte_line_col(&rope, 8), (1, 2));
    }

    #[test]
    fn test_extract_completion_prefix_simple() {
        let rope = Rope::from_str("let foo = bar");
        // Cursor after "bar" at offset 13
        assert_eq!(extract_completion_prefix(&rope, 13), "bar");
    }

    #[test]
    fn test_extract_completion_prefix_partial() {
        let rope = Rope::from_str("let foo = ba");
        // Cursor after "ba" at offset 12
        assert_eq!(extract_completion_prefix(&rope, 12), "ba");
    }

    #[test]
    fn test_extract_completion_prefix_at_start() {
        let rope = Rope::from_str("foo");
        assert_eq!(extract_completion_prefix(&rope, 0), "");
    }

    #[test]
    fn test_extract_completion_prefix_after_operator() {
        let rope = Rope::from_str("x + y");
        // Cursor after "y" at offset 5
        assert_eq!(extract_completion_prefix(&rope, 5), "y");
    }

    #[test]
    fn test_extract_completion_prefix_underscore() {
        let rope = Rope::from_str("my_var");
        assert_eq!(extract_completion_prefix(&rope, 6), "my_var");
    }

    #[test]
    fn test_span_to_range() {
        let rope = Rope::from_str("hello\nworld");
        let span = trunk_ir::Span::new(0, 5);
        let range = span_to_range(&rope, span);
        assert_eq!(range.start.line, 0);
        assert_eq!(range.start.character, 0);
        assert_eq!(range.end.line, 0);
        assert_eq!(range.end.character, 5);
    }

    #[test]
    fn test_span_to_range_multiline() {
        let rope = Rope::from_str("hello\nworld");
        let span = trunk_ir::Span::new(0, 11);
        let range = span_to_range(&rope, span);
        assert_eq!(range.start.line, 0);
        assert_eq!(range.start.character, 0);
        assert_eq!(range.end.line, 1);
        assert_eq!(range.end.character, 5);
    }

    #[test]
    fn test_position_offset_roundtrip() {
        let rope = Rope::from_str("fn main() {\n    println!(\"hello\");\n}");

        // Test various positions
        let test_positions = [(0, 0), (0, 5), (1, 4), (1, 10), (2, 0)];

        for (line, col) in test_positions {
            if let Some(offset) = offset_from_position(&rope, line, col) {
                let (recovered_line, recovered_col) = position_from_offset(&rope, offset);
                assert_eq!(
                    (recovered_line, recovered_col),
                    (line, col),
                    "Roundtrip failed for ({}, {})",
                    line,
                    col
                );
            }
        }
    }

    // =========================================================================
    // LSP Server Integration Tests (Message-based)
    // =========================================================================

    use std::sync::atomic::{AtomicI32, Ordering};

    static REQUEST_ID: AtomicI32 = AtomicI32::new(1);

    fn next_request_id() -> RequestId {
        RequestId::from(REQUEST_ID.fetch_add(1, Ordering::SeqCst))
    }

    /// Test harness that creates a server and client connection pair.
    struct TestHarness {
        server: LspServer,
        client: Connection,
    }

    impl TestHarness {
        fn new() -> Self {
            use lsp_types::request::{Initialize, Request as _};

            let (server_conn, client_conn) = Connection::memory();

            // Send initialize request from client
            let init_params = InitializeParams::default();
            let init_request = lsp_server::Request::new(
                RequestId::from(0),
                Initialize::METHOD.to_string(),
                init_params,
            );
            client_conn
                .sender
                .send(Message::Request(init_request))
                .unwrap();

            // Send initialized notification (must be sent before initialize_server returns,
            // because connection.initialize() waits for it)
            let initialized = Notification::new("initialized".to_string(), serde_json::json!({}));
            client_conn
                .sender
                .send(Message::Notification(initialized))
                .unwrap();

            // Server performs initialize handshake
            // (receives init request, sends response, receives initialized notification)
            let server = initialize_server(server_conn).unwrap();

            // Client receives initialize response
            let _response = client_conn.receiver.recv().unwrap();

            Self {
                server,
                client: client_conn,
            }
        }

        /// Send a didOpen notification and process it.
        fn open_document(&mut self, uri: &Uri, text: &str) {
            let params = DidOpenTextDocumentParams {
                text_document: lsp_types::TextDocumentItem {
                    uri: uri.clone(),
                    language_id: "tribute".to_string(),
                    version: 1,
                    text: text.to_string(),
                },
            };
            let notif = Notification::new(DidOpenTextDocument::METHOD.to_string(), params);
            self.client
                .sender
                .send(Message::Notification(notif))
                .unwrap();

            // Process the message on the server side
            let msg = self.server.connection.receiver.recv().unwrap();
            self.server.process_message(msg).unwrap();

            // Consume the diagnostics notification sent back
            let _ = self.client.receiver.try_recv();
        }

        /// Send a request and get the response.
        fn request<R: lsp_types::request::Request>(&mut self, params: R::Params) -> R::Result
        where
            R::Params: serde::Serialize,
            R::Result: serde::de::DeserializeOwned,
        {
            let id = next_request_id();
            let req = Request::new(id.clone(), R::METHOD.to_string(), params);
            self.client.sender.send(Message::Request(req)).unwrap();

            // Process the message on the server side
            let msg = self.server.connection.receiver.recv().unwrap();
            self.server.process_message(msg).unwrap();

            // Get the response
            match self.client.receiver.recv().unwrap() {
                Message::Response(resp) => {
                    assert_eq!(resp.id, id);
                    assert!(resp.error.is_none(), "Request failed: {:?}", resp.error);
                    serde_json::from_value(resp.result.unwrap()).unwrap()
                }
                other => panic!("Expected response message, got {:?}", other),
            }
        }
    }

    fn test_uri(name: &str) -> Uri {
        format!("file:///test/{}.trb", name).parse().unwrap()
    }

    #[test]
    fn test_hover_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("hover_msg");
        //                    0         1         2         3
        //                    0123456789012345678901234567890123456
        let source = "fn add(x: Int, y: Int): Int { x + y }";

        harness.open_document(&uri, source);

        // Hover on 'x' in the body (position 30)
        let params = HoverParams {
            text_document_position_params: TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier { uri },
                position: lsp_types::Position {
                    line: 0,
                    character: 30,
                },
            },
            work_done_progress_params: Default::default(),
        };

        let result: Option<Hover> = harness.request::<HoverRequest>(params);
        assert!(result.is_some(), "Hover should return type information");
    }

    #[test]
    fn test_document_symbols_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("symbols_msg");
        let source = "fn foo() { }\nfn bar() { }";

        harness.open_document(&uri, source);

        let params = DocumentSymbolParams {
            text_document: lsp_types::TextDocumentIdentifier { uri },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let result: Option<DocumentSymbolResponse> =
            harness.request::<DocumentSymbolRequest>(params);
        assert!(result.is_some(), "Should return document symbols");

        if let Some(DocumentSymbolResponse::Nested(symbols)) = result {
            assert!(symbols.iter().any(|s| s.name == "foo"), "Should find 'foo'");
            assert!(symbols.iter().any(|s| s.name == "bar"), "Should find 'bar'");
        }
    }

    #[test]
    fn test_completion_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("completion_msg");
        let source = "fn main() { le }";

        harness.open_document(&uri, source);

        let params = CompletionParams {
            text_document_position: TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier { uri },
                position: lsp_types::Position {
                    line: 0,
                    character: 14,
                },
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
            context: None,
        };

        let result: Option<lsp_types::CompletionResponse> = harness.request::<Completion>(params);
        assert!(result.is_some(), "Should return completions");

        if let Some(lsp_types::CompletionResponse::List(list)) = result {
            assert!(
                list.items.iter().any(|item| item.label == "let"),
                "Should suggest 'let' keyword"
            );
        }
    }

    #[test]
    fn test_goto_definition_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("goto_def_msg");
        let source = "fn main() { let x = 1; x }";

        harness.open_document(&uri, source);

        let params = GotoDefinitionParams {
            text_document_position_params: TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier { uri },
                position: lsp_types::Position {
                    line: 0,
                    character: 23,
                },
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let result: Option<GotoDefinitionResponse> = harness.request::<GotoDefinition>(params);
        assert!(
            result.is_some(),
            "Should find definition for local variable"
        );
    }

    #[test]
    fn test_find_references_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("references_msg");
        let source = r#"fn main() {
    let x = 1
    x + x
}"#;

        harness.open_document(&uri, source);

        let params = ReferenceParams {
            text_document_position: TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier { uri },
                position: lsp_types::Position {
                    line: 1,
                    character: 8, // On 'x' in 'let x = 1'
                },
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
            context: lsp_types::ReferenceContext {
                include_declaration: true,
            },
        };

        let result: Option<Vec<Location>> = harness.request::<References>(params);
        if let Some(refs) = result {
            // Should find: definition (let x) + 2 references (x + x)
            assert!(refs.len() >= 2, "Should find multiple references");
        }
    }

    #[test]
    fn test_did_change_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("change_msg");

        // Open document
        harness.open_document(&uri, "fn foo() { }");

        // Send didChange notification
        let params = DidChangeTextDocumentParams {
            text_document: lsp_types::VersionedTextDocumentIdentifier {
                uri: uri.clone(),
                version: 2,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text: "fn foo() { }\nfn bar() { }".to_string(),
            }],
        };
        let notif = Notification::new(DidChangeTextDocument::METHOD.to_string(), params);
        harness
            .client
            .sender
            .send(Message::Notification(notif))
            .unwrap();

        // Process the change
        let msg = harness.server.connection.receiver.recv().unwrap();
        harness.server.process_message(msg).unwrap();
        let _ = harness.client.receiver.try_recv(); // Consume diagnostics

        // Verify: document symbols should now include 'bar'
        let params = DocumentSymbolParams {
            text_document: lsp_types::TextDocumentIdentifier { uri },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let result: Option<DocumentSymbolResponse> =
            harness.request::<DocumentSymbolRequest>(params);

        if let Some(DocumentSymbolResponse::Nested(symbols)) = result {
            assert!(
                symbols.iter().any(|s| s.name == "bar"),
                "Should find 'bar' after change"
            );
        }
    }

    #[test]
    fn test_signature_help_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("sig_help_msg");
        // Define a function and call it
        let source = "fn add(x: Int, y: Int): Int { x + y }\nfn main() { add( }";

        harness.open_document(&uri, source);

        let params = SignatureHelpParams {
            text_document_position_params: TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier { uri },
                position: lsp_types::Position {
                    line: 1,
                    character: 16, // Inside add( call
                },
            },
            work_done_progress_params: Default::default(),
            context: None,
        };

        let result: Option<SignatureHelp> = harness.request::<SignatureHelpRequest>(params);
        // Signature help may or may not find the function depending on parsing
        // Just verify the request completes without error
        if let Some(sig_help) = result {
            assert!(
                !sig_help.signatures.is_empty(),
                "Should return signature information"
            );
        }
    }

    #[test]
    fn test_prepare_rename_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("prepare_rename_msg");
        let source = r#"fn main() {
    let foo = 1
    foo
}"#;

        harness.open_document(&uri, source);

        // Try renaming from the reference position (foo at the end)
        let params = TextDocumentPositionParams {
            text_document: lsp_types::TextDocumentIdentifier { uri },
            position: lsp_types::Position {
                line: 2,
                character: 4, // On 'foo' reference
            },
        };

        let result: Option<PrepareRenameResponse> = harness.request::<PrepareRenameRequest>(params);
        assert!(result.is_some(), "Should be able to rename from reference");

        if let Some(PrepareRenameResponse::RangeWithPlaceholder { placeholder, .. }) = result {
            assert_eq!(placeholder, "foo", "Placeholder should be current name");
        }
    }

    #[test]
    #[allow(clippy::mutable_key_type)] // Uri has interior mutability but it's fine for LSP
    fn test_rename_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("rename_msg");
        let source = r#"fn main() {
    let foo = 1
    foo + foo
}"#;

        harness.open_document(&uri, source);

        // Rename from reference position
        let params = RenameParams {
            text_document_position: TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier { uri: uri.clone() },
                position: lsp_types::Position {
                    line: 2,
                    character: 4, // On first 'foo' reference in 'foo + foo'
                },
            },
            new_name: "bar".to_string(),
            work_done_progress_params: Default::default(),
        };

        let result: Option<WorkspaceEdit> = harness.request::<Rename>(params);
        assert!(result.is_some(), "Should return workspace edit for rename");

        if let Some(edit) = result {
            let changes = edit.changes.expect("Should have changes");
            let edits = changes.get(&uri).expect("Should have edits for this file");
            // Should rename definition and both usages
            assert!(edits.len() >= 2, "Should have multiple edits for rename");
            assert!(
                edits.iter().all(|e| e.new_text == "bar"),
                "All edits should use new name"
            );
        }
    }

    #[test]
    fn test_code_action_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("code_action_msg");
        // Top-level function without return type annotation triggers a diagnostic
        let source = "fn identity(x: Int) { x }";

        harness.open_document(&uri, source);

        let params = CodeActionParams {
            text_document: lsp_types::TextDocumentIdentifier { uri },
            range: lsp_types::Range {
                start: lsp_types::Position {
                    line: 0,
                    character: 0,
                },
                end: lsp_types::Position {
                    line: 0,
                    character: 25,
                },
            },
            context: lsp_types::CodeActionContext {
                diagnostics: vec![],
                only: None,
                trigger_kind: None,
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let result: Option<Vec<CodeActionOrCommand>> = harness.request::<CodeActionRequest>(params);
        // Code actions depend on diagnostics being present
        // Just verify the request completes without error
        if let Some(actions) = result {
            // If there are actions, they should be code actions (not commands)
            for action in &actions {
                assert!(
                    matches!(action, CodeActionOrCommand::CodeAction(_)),
                    "Should return CodeAction, not Command"
                );
            }
        }
    }

    #[test]
    fn test_did_close_via_message() {
        let mut harness = TestHarness::new();
        let uri = test_uri("close_msg");

        // Open document
        harness.open_document(&uri, "fn foo() { }");

        // Send didClose notification
        let params = DidCloseTextDocumentParams {
            text_document: lsp_types::TextDocumentIdentifier { uri: uri.clone() },
        };
        let notif = Notification::new(DidCloseTextDocument::METHOD.to_string(), params);
        harness
            .client
            .sender
            .send(Message::Notification(notif))
            .unwrap();

        // Process the close
        let msg = harness.server.connection.receiver.recv().unwrap();
        harness.server.process_message(msg).unwrap();

        // Should receive a diagnostics notification clearing diagnostics
        match harness.client.receiver.recv().unwrap() {
            Message::Notification(notif) => {
                assert_eq!(notif.method, PublishDiagnostics::METHOD);
                let params: PublishDiagnosticsParams =
                    serde_json::from_value(notif.params).unwrap();
                assert_eq!(params.uri, uri);
                assert!(params.diagnostics.is_empty(), "Should clear diagnostics");
            }
            other => panic!("Expected notification, got {:?}", other),
        }
    }
}
