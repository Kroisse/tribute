//! Completion, document symbols, and signature help.
//!
//! Provides completion candidates, document outline, and function
//! signature information for LSP features.

use trunk_ir::{Span, Symbol};

use tribute_front::SourceCst;
use tribute_front::ast::{Decl, TypeAnnotation, TypeAnnotationKind};
use tribute_front::query as ast_query;

// =============================================================================
// Completion
// =============================================================================

/// Completion item kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompletionKind {
    Function,
    Struct,
    Enum,
    Ability,
    Const,
    Variable,
    Keyword,
    Constructor,
}

impl From<CompletionKind> for lsp_types::CompletionItemKind {
    fn from(kind: CompletionKind) -> Self {
        match kind {
            CompletionKind::Function => lsp_types::CompletionItemKind::FUNCTION,
            CompletionKind::Struct => lsp_types::CompletionItemKind::STRUCT,
            CompletionKind::Enum => lsp_types::CompletionItemKind::ENUM,
            CompletionKind::Ability => lsp_types::CompletionItemKind::CLASS,
            CompletionKind::Const => lsp_types::CompletionItemKind::CONSTANT,
            CompletionKind::Variable => lsp_types::CompletionItemKind::VARIABLE,
            CompletionKind::Keyword => lsp_types::CompletionItemKind::KEYWORD,
            CompletionKind::Constructor => lsp_types::CompletionItemKind::CONSTRUCTOR,
        }
    }
}

/// Completion item.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct AstCompletionItem {
    /// Name of the item.
    pub name: Symbol,
    /// Kind of completion.
    pub kind: CompletionKind,
    /// Brief documentation or type hint.
    pub detail: Option<String>,
}

/// Reserved keywords in Tribute.
pub const KEYWORDS: &[&str] = &[
    "fn", "let", "case", "struct", "enum", "ability", "const", "pub", "use", "mod", "if", "handle",
    "as", "True", "False", "Nil",
];

/// Get keyword completions filtered by prefix.
pub fn complete_keywords(prefix: &str) -> Vec<AstCompletionItem> {
    KEYWORDS
        .iter()
        .filter(|kw| kw.starts_with(prefix))
        .map(|kw| AstCompletionItem {
            name: Symbol::new(kw),
            kind: CompletionKind::Keyword,
            detail: None,
        })
        .collect()
}

/// Build completion items from a typed module.
#[salsa::tracked(returns(deref))]
pub fn completion_items(db: &dyn salsa::Database, source: SourceCst) -> Vec<AstCompletionItem> {
    let Some(module) = ast_query::tdnr_module(db, source) else {
        return Vec::new();
    };

    let mut items = Vec::new();

    for decl in &module.decls {
        match decl {
            Decl::Function(func) => {
                items.push(AstCompletionItem {
                    name: func.name,
                    kind: CompletionKind::Function,
                    detail: None,
                });
            }
            Decl::Struct(s) => {
                items.push(AstCompletionItem {
                    name: s.name,
                    kind: CompletionKind::Struct,
                    detail: None,
                });
            }
            Decl::Enum(e) => {
                items.push(AstCompletionItem {
                    name: e.name,
                    kind: CompletionKind::Enum,
                    detail: None,
                });
                // Add variant constructors
                for variant in &e.variants {
                    items.push(AstCompletionItem {
                        name: variant.name,
                        kind: CompletionKind::Constructor,
                        detail: Some(format!("{}::{}", e.name, variant.name)),
                    });
                }
            }
            Decl::ExternFunction(func) => {
                items.push(AstCompletionItem {
                    name: func.name,
                    kind: CompletionKind::Function,
                    detail: None,
                });
            }
            Decl::Ability(a) => {
                items.push(AstCompletionItem {
                    name: a.name,
                    kind: CompletionKind::Ability,
                    detail: None,
                });
            }
            Decl::Use(_) | Decl::Module(_) => {}
        }
    }

    items
}

/// Filter completion items by prefix.
pub fn filter_completions<'a>(
    items: &'a [AstCompletionItem],
    prefix: &'a str,
) -> impl Iterator<Item = &'a AstCompletionItem> {
    items
        .iter()
        .filter(move |item| item.name.with_str(|s| s.starts_with(prefix)))
}

// =============================================================================
// Document Symbols
// =============================================================================

/// Symbol kind for document outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Ability,
    Field,
    Variant,
}

/// Document symbol information.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct DocumentSymbolInfo {
    /// Symbol name.
    pub name: Symbol,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Full span of the symbol definition.
    pub span: Span,
    /// Children symbols (e.g., struct fields, enum variants).
    pub children: Vec<DocumentSymbolInfo>,
}

/// Build document symbols from a parsed module.
///
/// Uses the parsed module (before type checking) for faster response.
#[salsa::tracked]
pub fn document_symbols(db: &dyn salsa::Database, source: SourceCst) -> Vec<DocumentSymbolInfo> {
    let Some(module) = ast_query::parsed_module(db, source) else {
        return Vec::new();
    };
    let Some(span_map) = ast_query::span_map(db, source) else {
        return Vec::new();
    };

    let mut symbols = Vec::new();

    for decl in &module.decls {
        match decl {
            Decl::Function(func) => {
                symbols.push(DocumentSymbolInfo {
                    name: func.name,
                    kind: SymbolKind::Function,
                    span: span_map.get_or_default(func.id),
                    children: vec![],
                });
            }
            Decl::Struct(s) => {
                let children: Vec<_> = s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        f.name.map(|name| DocumentSymbolInfo {
                            name,
                            kind: SymbolKind::Field,
                            span: span_map.get_or_default(f.id),
                            children: vec![],
                        })
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: s.name,
                    kind: SymbolKind::Struct,
                    span: span_map.get_or_default(s.id),
                    children,
                });
            }
            Decl::Enum(e) => {
                let children: Vec<_> = e
                    .variants
                    .iter()
                    .map(|v| DocumentSymbolInfo {
                        name: v.name,
                        kind: SymbolKind::Variant,
                        span: span_map.get_or_default(v.id),
                        children: vec![],
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: e.name,
                    kind: SymbolKind::Enum,
                    span: span_map.get_or_default(e.id),
                    children,
                });
            }
            Decl::Ability(a) => {
                let children: Vec<_> = a
                    .operations
                    .iter()
                    .map(|op| DocumentSymbolInfo {
                        name: op.name,
                        kind: SymbolKind::Function,
                        span: span_map.get_or_default(op.id),
                        children: vec![],
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: a.name,
                    kind: SymbolKind::Ability,
                    span: span_map.get_or_default(a.id),
                    children,
                });
            }
            Decl::ExternFunction(func) => {
                symbols.push(DocumentSymbolInfo {
                    name: func.name,
                    kind: SymbolKind::Function,
                    span: span_map.get_or_default(func.id),
                    children: vec![],
                });
            }
            Decl::Use(_) | Decl::Module(_) => {}
        }
    }

    symbols
}

// =============================================================================
// Function Signatures
// =============================================================================

/// Function signature information for signature help.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionSignature {
    /// Function name.
    pub name: Symbol,
    /// Parameter names and type strings.
    pub params: Vec<(Symbol, Option<String>)>,
    /// Return type string (if specified).
    pub return_ty: Option<String>,
    /// Comma-separated list of effect names, without surrounding braces.
    /// For example, a function declared `fn foo() ->{IO, State} Int` stores
    /// `Some("IO, State")` here. `None` means no effect annotation was present.
    /// Callers formatting this value should split on `,` and trim whitespace;
    /// the `{}` delimiters are *not* included in the stored string.
    pub effects: Option<String>,
    /// Span of the function definition.
    pub span: Span,
}

/// Pretty-print a type annotation to a string.
fn print_type_annotation(ty: &TypeAnnotation) -> String {
    match &ty.kind {
        TypeAnnotationKind::Named(name) => name.to_string(),
        TypeAnnotationKind::Path(parts) => parts
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        TypeAnnotationKind::App { ctor, args } => {
            let ctor_str = print_type_annotation(ctor);
            let args_str: Vec<_> = args.iter().map(print_type_annotation).collect();
            format!("{}({})", ctor_str, args_str.join(", "))
        }
        TypeAnnotationKind::Func { params, result, .. } => {
            let params_str: Vec<_> = params.iter().map(print_type_annotation).collect();
            let result_str = print_type_annotation(result);
            format!("({}) -> {}", params_str.join(", "), result_str)
        }
        TypeAnnotationKind::Tuple(elems) => {
            let elems_str: Vec<_> = elems.iter().map(print_type_annotation).collect();
            format!("({})", elems_str.join(", "))
        }
        TypeAnnotationKind::Infer => "_".to_string(),
        TypeAnnotationKind::Error => "?".to_string(),
    }
}

/// Build function signatures from a typed module.
pub fn function_signatures(db: &dyn salsa::Database, source: SourceCst) -> Vec<FunctionSignature> {
    let Some(module) = ast_query::tdnr_module(db, source) else {
        return Vec::new();
    };
    let Some(span_map) = ast_query::span_map(db, source) else {
        return Vec::new();
    };

    let mut signatures = Vec::new();

    for decl in &module.decls {
        if let Decl::Function(func) = decl {
            let params: Vec<_> = func
                .params
                .iter()
                .map(|p| {
                    let ty_str = p.ty.as_ref().map(print_type_annotation);
                    (p.name, ty_str)
                })
                .collect();

            let return_ty = func.return_ty.as_ref().map(print_type_annotation);

            let effects = func.effects.as_ref().and_then(|effs| {
                if effs.is_empty() {
                    None
                } else {
                    let effect_strs: Vec<_> = effs.iter().map(print_type_annotation).collect();
                    Some(effect_strs.join(", "))
                }
            });

            signatures.push(FunctionSignature {
                name: func.name,
                params,
                return_ty,
                effects,
                span: span_map.get_or_default(func.id),
            });
        }
    }

    signatures
}

/// Find a function signature by name.
pub fn find_signature(
    signatures: &[FunctionSignature],
    name: Symbol,
) -> Option<&FunctionSignature> {
    signatures.iter().find(|s| s.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ropey::Rope;
    use tree_sitter::Parser;
    use tribute_front::path_to_uri;

    fn make_source(db: &dyn salsa::Database, text: &str) -> SourceCst {
        let uri = path_to_uri(std::path::Path::new("test.trb"));
        let rope = Rope::from_str(text);

        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(text, None);

        SourceCst::new(db, uri, rope, tree)
    }

    // Completion tests

    #[test]
    fn test_complete_keywords_fn() {
        let completions = complete_keywords("fn");
        assert_eq!(completions.len(), 1);
        assert!(completions[0].name == trunk_ir::Symbol::new("fn"));
        assert_eq!(completions[0].kind, CompletionKind::Keyword);
    }

    #[test]
    fn test_complete_keywords_empty_prefix() {
        let completions = complete_keywords("");
        assert_eq!(completions.len(), KEYWORDS.len());
    }

    #[test]
    fn test_complete_keywords_no_match() {
        let completions = complete_keywords("xyz");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_keywords_partial() {
        let completions = complete_keywords("st");
        assert_eq!(completions.len(), 1);
        assert!(completions[0].name == trunk_ir::Symbol::new("struct"));
    }

    #[test]
    fn test_filter_completions() {
        let items = vec![
            AstCompletionItem {
                name: trunk_ir::Symbol::new("foo"),
                kind: CompletionKind::Function,
                detail: None,
            },
            AstCompletionItem {
                name: trunk_ir::Symbol::new("bar"),
                kind: CompletionKind::Function,
                detail: None,
            },
            AstCompletionItem {
                name: trunk_ir::Symbol::new("foobar"),
                kind: CompletionKind::Function,
                detail: None,
            },
        ];

        let filtered: Vec<_> = filter_completions(&items, "foo").collect();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_completion_items_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn hello() { 1 }");

        let items = completion_items(&db, source);
        assert!(!items.is_empty());

        let hello_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("hello"));
        assert!(hello_item.is_some());
        assert_eq!(hello_item.unwrap().kind, CompletionKind::Function);
    }

    #[test]
    fn test_completion_items_struct() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int, y: Int }");

        let items = completion_items(&db, source);
        let point_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("Point"));
        assert!(point_item.is_some());
        assert_eq!(point_item.unwrap().kind, CompletionKind::Struct);
    }

    #[test]
    fn test_completion_items_enum_with_variants() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Color { Red, Green, Blue }");

        let items = completion_items(&db, source);

        // Should have the enum
        let color_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("Color"));
        assert!(color_item.is_some());
        assert_eq!(color_item.unwrap().kind, CompletionKind::Enum);

        // Should have the variants
        let red_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("Red"));
        assert!(red_item.is_some());
        assert_eq!(red_item.unwrap().kind, CompletionKind::Constructor);
    }

    // Document symbols tests

    #[test]
    fn test_document_symbols_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 1 }");

        let symbols = document_symbols(&db, source);
        assert!(!symbols.is_empty());

        let main_sym = symbols
            .iter()
            .find(|s| s.name == trunk_ir::Symbol::new("main"));
        assert!(main_sym.is_some());
        assert_eq!(main_sym.unwrap().kind, SymbolKind::Function);
    }

    #[test]
    fn test_document_symbols_struct_with_fields() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int, y: Int }");

        let symbols = document_symbols(&db, source);

        let point_sym = symbols
            .iter()
            .find(|s| s.name == trunk_ir::Symbol::new("Point"));
        assert!(point_sym.is_some());
        assert_eq!(point_sym.unwrap().kind, SymbolKind::Struct);

        // Should have field children
        let point = point_sym.unwrap();
        assert_eq!(point.children.len(), 2);
        assert!(
            point
                .children
                .iter()
                .any(|c| c.name == trunk_ir::Symbol::new("x"))
        );
        assert!(
            point
                .children
                .iter()
                .any(|c| c.name == trunk_ir::Symbol::new("y"))
        );
    }

    #[test]
    fn test_document_symbols_enum_with_variants() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Option { Some(a), None }");

        let symbols = document_symbols(&db, source);

        let option_sym = symbols
            .iter()
            .find(|s| s.name == trunk_ir::Symbol::new("Option"));
        assert!(option_sym.is_some());
        assert_eq!(option_sym.unwrap().kind, SymbolKind::Enum);

        // Should have variant children
        let option = option_sym.unwrap();
        assert_eq!(option.children.len(), 2);
    }

    // Function signatures tests

    #[test]
    fn test_function_signatures_simple() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn add(a: Int, b: Int) -> Int { a + b }");

        let signatures = function_signatures(&db, source);
        assert_eq!(signatures.len(), 1);

        let sig = &signatures[0];
        assert!(sig.name == trunk_ir::Symbol::new("add"));
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].1, Some("Int".to_string()));
        assert_eq!(sig.params[1].1, Some("Int".to_string()));
        assert_eq!(sig.return_ty, Some("Int".to_string()));
    }

    #[test]
    fn test_function_signatures_no_annotations() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn identity(x) { x }");

        let signatures = function_signatures(&db, source);
        assert_eq!(signatures.len(), 1);

        let sig = &signatures[0];
        assert!(sig.params[0].1.is_none());
        assert!(sig.return_ty.is_none());
    }

    #[test]
    fn test_function_signatures_no_effects() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn add(a: Int, b: Int) -> Int { a + b }");

        let signatures = function_signatures(&db, source);
        assert_eq!(signatures.len(), 1);
        assert!(signatures[0].effects.is_none());
    }

    #[test]
    fn test_find_signature() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { 1 }\nfn bar() { 2 }");

        let signatures = function_signatures(&db, source);
        assert_eq!(signatures.len(), 2);

        let foo = find_signature(&signatures, trunk_ir::Symbol::new("foo"));
        assert!(foo.is_some());

        let baz = find_signature(&signatures, trunk_ir::Symbol::new("baz"));
        assert!(baz.is_none());
    }
}
