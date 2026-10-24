//! Completion, document symbols, and signature help.
//!
//! Provides completion candidates, document outline, and function
//! signature information for LSP features.

use trunk_ir::{Span, Symbol, SymbolPath};

use tribute_front::SourceCst;
use tribute_front::ast::{
    AbilityId, Decl, DeclTable, ModuleItem, SpanMap, TypeAnnotation, TypeAnnotationKind, TypedRef,
    UnresolvedName,
};
use tribute_front::query as ast_query;
use tribute_front::resolve::Binding;

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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AstCompletionItem {
    /// Name of the item.
    pub name: Symbol,
    /// Kind of completion.
    pub kind: CompletionKind,
    /// Brief documentation or type hint.
    pub detail: Option<String>,
    /// The module whose body names the item, or `None` for an item every
    /// module names.
    pub module: Option<SymbolPath>,
}

/// Keywords offered as completions.
fn keywords() -> impl Iterator<Item = &'static &'static str> {
    use tribute_front::keywords::{KEYWORDS, LITERAL_KEYWORDS};
    KEYWORDS.iter().chain(LITERAL_KEYWORDS)
}

/// Get keyword completions filtered by prefix.
pub fn complete_keywords(prefix: &str) -> Vec<AstCompletionItem> {
    keywords()
        .filter(|kw| kw.starts_with(prefix))
        .map(|kw| AstCompletionItem {
            name: Symbol::new(kw),
            kind: CompletionKind::Keyword,
            detail: None,
            module: None,
        })
        .collect()
}

/// The declarations of the root module of `module`.
fn root_decls<'a, 'db>(
    module: &'a DeclTable<TypedRef<'db>>,
) -> impl Iterator<Item = &'a Decl<TypedRef<'db>>> {
    module
        .entries
        .iter()
        .filter(|entry| entry.module.is_root())
        .map(|entry| &entry.decl)
}

/// Build completion items from a typed module: the declarations of each of
/// its modules, which that module's body names, then the prelude's public
/// root items, which every module of the package names.
#[salsa::tracked(returns(deref))]
pub fn completion_items(db: &dyn salsa::Database, source: SourceCst) -> Vec<AstCompletionItem> {
    let Some(output) = tribute::parse_and_lower_ast(db, source) else {
        return Vec::new();
    };
    let module = output.module(db);

    let mut items = Vec::new();

    for entry in &module.entries {
        let item = |name: &Symbol, kind, detail| AstCompletionItem {
            name: name.clone(),
            kind,
            detail,
            module: Some(entry.module.clone()),
        };
        match &entry.decl {
            Decl::Function(func) => items.push(item(&func.name, CompletionKind::Function, None)),
            Decl::Struct(s) => items.push(item(&s.name, CompletionKind::Struct, None)),
            // Its variants are only in its namespace, not names of the module.
            Decl::Enum(e) => items.push(item(&e.name, CompletionKind::Enum, None)),
            Decl::ExternFunction(func) => {
                items.push(item(&func.name, CompletionKind::Function, None));
            }
            Decl::Ability(a) => items.push(item(&a.name, CompletionKind::Ability, None)),
            Decl::Use(use_decl) => {
                if let Some(ability) = AbilityId::builtin_from_path(db, &use_decl.path) {
                    let name = use_decl.alias.clone().unwrap_or_else(|| ability.name(db));
                    if !items.iter().any(|item| {
                        item.name == name && item.module.as_ref() == Some(&entry.module)
                    }) {
                        items.push(item(
                            &name,
                            CompletionKind::Ability,
                            Some(format!(
                                "compiler-owned ambient ability ({})",
                                ability.qualified(db)
                            )),
                        ));
                    }
                }
            }
        }
    }

    let mut library = prelude_names(db);
    library.sort_by(|(left, _), (right, _)| left.cmp(right));
    for (name, binding) in library {
        let kind = match binding {
            Binding::Function { .. } => CompletionKind::Function,
            Binding::Constructor { tag: Some(_), .. } => CompletionKind::Constructor,
            Binding::Constructor { tag: None, .. } => CompletionKind::Struct,
            Binding::TypeDef { .. } => CompletionKind::Enum,
            Binding::Ability { .. } => CompletionKind::Ability,
            Binding::Module { .. } | Binding::Import { .. } | Binding::AbilityOp { .. } => continue,
        };
        items.push(AstCompletionItem {
            detail: Some(format!(
                "{}::{name}",
                tribute_front::resolve::LIBRARY_PACKAGE
            )),
            name,
            kind,
            module: None,
        });
    }

    items
}

/// The constructors an implicit constructor `.Name` may name: the variants
/// and structs of the types each module declares or imports, which that
/// module's body names, then those of the prelude's types, which every
/// module names. Each item's detail is the constructor's path through its
/// type as the module names the type.
#[salsa::tracked(returns(deref))]
pub fn constructor_items(db: &dyn salsa::Database, source: SourceCst) -> Vec<AstCompletionItem> {
    use rustc_hash::FxHashMap as HashMap;

    let Some(output) = tribute::parse_and_lower_ast(db, source) else {
        return Vec::new();
    };
    let entries = &output.module(db).entries;
    // Each type's constructors, by the type's package-root path.
    let types: HashMap<SymbolPath, TypeConstructors<'_>> = entries
        .iter()
        .filter_map(|entry| {
            let constructors = match &entry.decl {
                Decl::Enum(e) => TypeConstructors::Variants(
                    &e.name,
                    e.variants.iter().map(|v| &v.name).collect(),
                ),
                Decl::Struct(s) => TypeConstructors::Struct(&s.name),
                _ => return None,
            };
            Some((
                entry.module.child(constructors.name().clone()),
                constructors,
            ))
        })
        .collect();
    let mut items = Vec::new();
    let mut add = |module: &SymbolPath, ty: &Symbol, constructors: &TypeConstructors<'_>| {
        let mut push = |name: &Symbol, detail: String| {
            items.push(AstCompletionItem {
                name: name.clone(),
                kind: CompletionKind::Constructor,
                detail: Some(detail),
                module: Some(module.clone()),
            });
        };
        match constructors {
            TypeConstructors::Variants(_, variants) => {
                for &variant in variants {
                    push(variant, format!("{ty}::{variant}"));
                }
            }
            // A struct's constructor has the struct's own name, which an
            // alias does not rename.
            TypeConstructors::Struct(own) if ty == *own => push(own, ty.to_string()),
            TypeConstructors::Struct(_) => {}
        }
    };
    for entry in entries {
        match &entry.decl {
            Decl::Enum(e) => add(
                &entry.module,
                &e.name,
                &types[&entry.module.child(e.name.clone())],
            ),
            Decl::Struct(s) => add(
                &entry.module,
                &s.name,
                &types[&entry.module.child(s.name.clone())],
            ),
            Decl::Use(u) => {
                if let Some(target) = &u.target
                    && let Some(ty) = types.get(&target[..])
                {
                    add(&entry.module, u.alias.as_ref().unwrap_or(ty.name()), ty);
                }
            }
            _ => {}
        }
    }
    for (ty, name) in prelude_constructors(db) {
        items.push(AstCompletionItem {
            detail: Some(format!(
                "{}::{ty}::{name}",
                tribute_front::resolve::LIBRARY_PACKAGE
            )),
            name,
            kind: CompletionKind::Constructor,
            module: None,
        });
    }
    items
}

/// The names the prelude gives a user package: the public ones of those
/// its modules see by name.
fn prelude_names(db: &dyn salsa::Database) -> Vec<(Symbol, Binding<'_>)> {
    let dependencies = tribute::prelude_dependencies(db);
    let own = tribute_front::resolve::PackageEnv::new();
    let scope = tribute_front::resolve::Scope::new(&own, &dependencies);
    scope
        .iter_library_names()
        .into_iter()
        .filter_map(|name| {
            let binding = scope.lookup_library(&name)?;
            scope
                .restriction(binding)
                .is_none()
                .then(|| (name, binding.clone()))
        })
        .collect()
}

/// The constructors of a type the completion index knows.
enum TypeConstructors<'a> {
    /// An enum, by its name, and its variants.
    Variants(&'a Symbol, Vec<&'a Symbol>),
    /// A struct, its own constructor, by its name.
    Struct(&'a Symbol),
}

impl TypeConstructors<'_> {
    fn name(&self) -> &Symbol {
        match self {
            Self::Variants(name, _) | Self::Struct(name) => name,
        }
    }
}

/// The constructors of the prelude's types, as `(type, variant)`: the
/// public variants of its enums, the only prelude types with constructors.
fn prelude_constructors(db: &dyn salsa::Database) -> Vec<(Symbol, Symbol)> {
    let dependencies = tribute::prelude_dependencies(db);
    let own = tribute_front::resolve::PackageEnv::new();
    let scope = tribute_front::resolve::Scope::new(&own, &dependencies);
    let mut constructors: Vec<_> = prelude_names(db)
        .into_iter()
        .filter(|(_, binding)| matches!(binding, Binding::TypeDef { .. }))
        .filter_map(|(name, _)| Some((scope.library_namespace(&name)?, name)))
        .flat_map(|(namespace, name)| {
            scope
                .iter_namespace(&namespace)
                .filter(|(_, binding)| {
                    matches!(binding, Binding::Constructor { tag: Some(_), .. })
                        && scope.restriction(binding).is_none()
                })
                .map(move |(variant, _)| (name.clone(), variant))
                .collect::<Vec<_>>()
        })
        .collect();
    constructors.sort();
    constructors
}

/// The identity path of the module whose body contains the byte offset
/// `offset` of `source`, as the module's declarations record it.
pub fn module_at(db: &dyn salsa::Database, source: SourceCst, offset: usize) -> SymbolPath {
    fn descend(
        items: &[ModuleItem<UnresolvedName>],
        span_map: &SpanMap,
        offset: usize,
        path: &mut SymbolPath,
    ) {
        for item in items {
            let ModuleItem::Module(module) = item else {
                continue;
            };
            let Some(body) = &module.body else { continue };
            let span = span_map.get_or_default(module.id);
            if span.start <= offset && offset < span.end {
                path.push(module.name.clone());
                return descend(body, span_map, offset, path);
            }
        }
    }

    let mut path = SymbolPath::root();
    if let Some(parsed) = ast_query::parsed_ast(db, source) {
        descend(
            &parsed.module(db).decls,
            &parsed.span_map(db),
            offset,
            &mut path,
        );
    }
    tribute_front::resolve::PackageName::compiled().identity_path(&path)
}

/// The items of `items` the body of `module` names. A declaration of the
/// module hides a library item of its name.
pub fn completions_in<'a>(
    items: &'a [AstCompletionItem],
    module: &'a SymbolPath,
) -> impl Iterator<Item = &'a AstCompletionItem> {
    let declares = move |name: &Symbol| {
        items
            .iter()
            .any(|item| item.module.as_ref() == Some(module) && item.name == *name)
    };
    items.iter().filter(move |item| match &item.module {
        Some(own) => own == module,
        None => !declares(&item.name),
    })
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
#[salsa::tracked(returns(deref))]
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
            ModuleItem::Decl(Decl::Function(func)) => {
                symbols.push(DocumentSymbolInfo {
                    name: func.name.clone(),
                    kind: SymbolKind::Function,
                    span: span_map.get_or_default(func.id),
                    children: vec![],
                });
            }
            ModuleItem::Decl(Decl::Struct(s)) => {
                let children: Vec<_> = s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        f.name.clone().map(|name| DocumentSymbolInfo {
                            name,
                            kind: SymbolKind::Field,
                            span: span_map.get_or_default(f.id),
                            children: vec![],
                        })
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: s.name.clone(),
                    kind: SymbolKind::Struct,
                    span: span_map.get_or_default(s.id),
                    children,
                });
            }
            ModuleItem::Decl(Decl::Enum(e)) => {
                let children: Vec<_> = e
                    .variants
                    .iter()
                    .map(|v| DocumentSymbolInfo {
                        name: v.name.clone(),
                        kind: SymbolKind::Variant,
                        span: span_map.get_or_default(v.id),
                        children: vec![],
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: e.name.clone(),
                    kind: SymbolKind::Enum,
                    span: span_map.get_or_default(e.id),
                    children,
                });
            }
            ModuleItem::Decl(Decl::Ability(a)) => {
                let children: Vec<_> = a
                    .operations
                    .iter()
                    .map(|op| DocumentSymbolInfo {
                        name: op.name.clone(),
                        kind: SymbolKind::Function,
                        span: span_map.get_or_default(op.id),
                        children: vec![],
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: a.name.clone(),
                    kind: SymbolKind::Ability,
                    span: span_map.get_or_default(a.id),
                    children,
                });
            }
            ModuleItem::Decl(Decl::ExternFunction(func)) => {
                symbols.push(DocumentSymbolInfo {
                    name: func.name.clone(),
                    kind: SymbolKind::Function,
                    span: span_map.get_or_default(func.id),
                    children: vec![],
                });
            }
            ModuleItem::Decl(Decl::Use(_)) | ModuleItem::Module(_) => {}
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
    /// `Some("IO, State")` here. `None` means no ability annotation was present.
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
    let Some(output) = tribute::parse_and_lower_ast(db, source) else {
        return Vec::new();
    };
    let module = output.module(db);
    let Some(span_map) = ast_query::span_map(db, source) else {
        return Vec::new();
    };

    let mut signatures = Vec::new();

    for decl in root_decls(module) {
        if let Decl::Function(func) = decl {
            let params: Vec<_> = func
                .params
                .iter()
                .map(|p| {
                    let ty_str = p.ty.as_ref().map(print_type_annotation);
                    (p.name.clone(), ty_str)
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
                name: func.name.clone(),
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
pub fn find_signature<'a>(
    signatures: &'a [FunctionSignature],
    name: &Symbol,
) -> Option<&'a FunctionSignature> {
    signatures.iter().find(|s| s.name == name)
}

/// Find the signature of a callee as written in source, e.g. `r#case`.
pub fn find_callee_signature<'a>(
    signatures: &'a [FunctionSignature],
    callee: &str,
) -> Option<&'a FunctionSignature> {
    let name = Symbol::new(&tribute_front::keywords::unraw(callee));
    find_signature(signatures, &name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(db: &dyn salsa::Database, text: &str) -> SourceCst {
        SourceCst::from_source_str(db, "test.trb", text)
    }

    // Completion tests

    #[test]
    fn test_constructor_items_include_imported_types() {
        let db = salsa::DatabaseImpl::default();
        let text = "mod shapes {\n    pub enum Shape { Dot, Line(Nat) }\n    pub struct Point { x: Nat }\n    pub enum Pair { Pair(Nat, Nat), Empty }\n}\nuse shapes::Shape\nuse shapes::Point\nmod inner {\n    use super::shapes::Shape as S\n    use super::shapes::Point as P\n    use super::shapes::Pair as Q\n}\n";
        let source = make_source(&db, text);
        let items = constructor_items(&db, source);
        let at = |module: &[&str]| -> Vec<(String, String)> {
            let module = SymbolPath::new(module.iter().copied());
            let mut found: Vec<_> = items
                .iter()
                .filter(|item| item.module.as_ref() == Some(&module))
                .map(|item| {
                    (
                        item.name.to_string(),
                        item.detail.clone().unwrap_or_default(),
                    )
                })
                .collect();
            found.sort();
            found
        };
        let pair = |name: &str, detail: &str| (name.to_owned(), detail.to_owned());
        assert_eq!(
            at(&[]),
            [
                pair("Dot", "Shape::Dot"),
                pair("Line", "Shape::Line"),
                pair("Point", "Point")
            ]
        );
        // An alias names the type, but not a struct's own constructor; a
        // variant named like its enum is still a variant.
        assert_eq!(
            at(&["inner"]),
            [
                pair("Dot", "S::Dot"),
                pair("Empty", "Q::Empty"),
                pair("Line", "S::Line"),
                pair("Pair", "Q::Pair"),
            ]
        );
    }

    #[test]
    fn test_constructor_items_are_variants_structs_and_the_preludes() {
        let db = salsa::DatabaseImpl::default();
        let text = "enum Color { Red, Green }\nstruct Point { x: Nat }\nmod inner {\n    enum Mode { Quiet }\n}\n";
        let source = make_source(&db, text);
        let items = constructor_items(&db, source);
        let at = |module: &SymbolPath| -> Vec<(String, Option<String>)> {
            items
                .iter()
                .filter(|item| item.module.as_ref().is_none_or(|own| own == module))
                .map(|item| (item.name.to_string(), item.detail.clone()))
                .collect()
        };
        let root = at(&SymbolPath::root());
        for (name, detail) in [
            ("Red", "Color::Red"),
            ("Green", "Color::Green"),
            ("Point", "Point"),
            ("Some", "std::Option::Some"),
            ("Error", "std::Result::Error"),
        ] {
            assert!(
                root.iter()
                    .any(|(n, d)| n == name && d.as_deref() == Some(detail)),
                "{name} in {root:?}"
            );
        }
        assert!(!root.iter().any(|(name, _)| name == "Quiet"));
        assert!(
            items
                .iter()
                .all(|item| item.kind == CompletionKind::Constructor)
        );
    }

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
        assert_eq!(completions.len(), keywords().count());
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
                module: None,
            },
            AstCompletionItem {
                name: trunk_ir::Symbol::new("bar"),
                kind: CompletionKind::Function,
                detail: None,
                module: None,
            },
            AstCompletionItem {
                name: trunk_ir::Symbol::new("foobar"),
                kind: CompletionKind::Function,
                detail: None,
                module: None,
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

        let hello_item = items.iter().find(|i| i.name == "hello");
        assert!(hello_item.is_some());
        assert_eq!(hello_item.unwrap().kind, CompletionKind::Function);
    }

    #[test]
    fn test_completions_offer_what_the_module_sees() {
        let db = salsa::DatabaseImpl::default();
        let text = "fn top() -> Nat { 1 }\nmod inner {\n    fn local() -> Nat { 2 }\n}\n";
        let source = make_source(&db, text);
        let items = completion_items(&db, source);
        let names_at = |offset| -> Vec<Symbol> {
            let module = module_at(&db, source, offset);
            completions_in(items, &module)
                .map(|item| item.name.clone())
                .collect()
        };

        let root = names_at(0);
        assert!(root.iter().any(|name| name == "top"));
        assert!(!root.iter().any(|name| name == "local"));
        let inner = names_at(text.find("fn local").unwrap());
        assert!(inner.iter().any(|name| name == "local"));
        assert!(!inner.iter().any(|name| name == "top"));

        // The prelude's public items, and not its private ones. Variants
        // are only in their enum's namespace.
        for names in [&root, &inner] {
            assert!(!names.iter().any(|name| name == "Some"));
            assert!(names.iter().any(|name| name == "Option"));
            assert!(names.iter().any(|name| name == "List"));
            assert!(!names.iter().any(|name| name == "__tribute_bytes_len"));
        }
    }

    #[test]
    fn test_completions_hide_shadowed_library_names() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Option { Some, Empty }\nfn main() -> Nil {}\n");
        let items = completion_items(&db, source);
        let root = SymbolPath::root();
        let named: Vec<_> = completions_in(items, &root)
            .filter(|item| item.name == "Option" || item.name == "Some")
            .collect();
        assert_eq!(named.len(), 1, "{named:#?}");
        assert!(named.iter().all(|item| item.module.is_some()));
    }

    #[test]
    fn test_completions_inside_a_user_std_module() {
        let db = salsa::DatabaseImpl::default();
        let text = "mod std {\n    fn local() -> Nat { 1 }\n}\nfn main() -> Nil {}\n";
        let source = make_source(&db, text);
        let items = completion_items(&db, source);
        let module = module_at(&db, source, text.find("fn local").unwrap());
        assert!(completions_in(items, &module).any(|item| item.name == "local"));
    }

    #[test]
    fn test_completion_items_struct() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int, y: Int }");

        let items = completion_items(&db, source);
        let point_item = items.iter().find(|i| i.name == "Point");
        assert!(point_item.is_some());
        assert_eq!(point_item.unwrap().kind, CompletionKind::Struct);
    }

    #[test]
    fn test_completion_items_imported_builtin_io() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "use std::io::Io\nfn main() -> Nil { Nil }");

        let items = completion_items(&db, source);
        let io = items
            .iter()
            .find(|item| item.name == "Io")
            .expect("imported builtin Io should appear in completions");
        assert_eq!(io.kind, CompletionKind::Ability);
        assert_eq!(
            io.detail.as_deref(),
            Some("compiler-owned ambient ability (std::io::Io)")
        );
    }

    #[test]
    fn test_completion_items_enum_without_its_variants() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Color { Red, Green, Blue }");

        let items = completion_items(&db, source);

        // Should have the enum
        let color_item = items.iter().find(|i| i.name == "Color");
        assert!(color_item.is_some());
        assert_eq!(color_item.unwrap().kind, CompletionKind::Enum);

        // The variants are only in the enum's namespace
        assert!(!items.iter().any(|i| i.name == "Red"));
    }

    // Document symbols tests

    #[test]
    fn test_document_symbols_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() -> Nil { 1 }");

        let symbols = document_symbols(&db, source);
        assert!(!symbols.is_empty());

        let main_sym = symbols.iter().find(|s| s.name == "main");
        assert!(main_sym.is_some());
        assert_eq!(main_sym.unwrap().kind, SymbolKind::Function);
    }

    #[test]
    fn test_document_symbols_struct_with_fields() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int, y: Int }");

        let symbols = document_symbols(&db, source);

        let point_sym = symbols.iter().find(|s| s.name == "Point");
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

        let option_sym = symbols.iter().find(|s| s.name == "Option");
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

        let foo = find_signature(&signatures, &trunk_ir::Symbol::new("foo"));
        assert!(foo.is_some());

        let baz = find_signature(&signatures, &trunk_ir::Symbol::new("baz"));
        assert!(baz.is_none());
    }

    #[test]
    fn test_find_callee_signature_normalizes_raw_names() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn r#case(x: Nat) -> Nat { x }\nfn plain() { 1 }");

        let signatures = function_signatures(&db, source);
        for callee in ["r#case", "case"] {
            let sig = find_callee_signature(&signatures, callee);
            assert_eq!(
                sig.map(|sig| sig.name.clone()),
                Some(trunk_ir::Symbol::new("case")),
                "{callee}"
            );
        }
        assert!(find_callee_signature(&signatures, "r#plain").is_some());
    }

    #[test]
    fn test_completion_items_extern_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, r#"extern "intrinsic" fn __add(a: Int, b: Int) -> Int"#);

        let items = completion_items(&db, source);
        let add_item = items.iter().find(|i| i.name == "__add");
        assert!(
            add_item.is_some(),
            "Extern function should appear in completions"
        );
        assert_eq!(add_item.unwrap().kind, CompletionKind::Function);
    }

    #[test]
    fn test_completion_items_extern_and_regular_functions() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"extern "intrinsic" fn __add(a: Int, b: Int) -> Int
fn main() -> Nil { 1 }"#,
        );

        let items = completion_items(&db, source);

        let add_item = items.iter().find(|i| i.name == "__add");
        assert!(add_item.is_some(), "Extern function should appear");
        assert_eq!(add_item.unwrap().kind, CompletionKind::Function);

        let main_item = items.iter().find(|i| i.name == "main");
        assert!(main_item.is_some(), "Regular function should appear");
        assert_eq!(main_item.unwrap().kind, CompletionKind::Function);
    }
}
