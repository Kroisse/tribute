//! DSL parser for `#[dialect]`.
//!
//! Parses the module body into structured types for code generation.

use proc_macro2::{Delimiter, Ident, TokenTree};
use rustc_hash::FxHashSet as HashSet;
use unsynn::{Parser, ToTokenIter, TokenIter};

mod constraint;
pub use constraint::{BoundPath, KindType, ListExpr, Projection, TypeExpr, TypeVar, ValueExpr};

// ============================================================================
// Parsed types
// ============================================================================

pub struct DialectModule {
    pub name: String,
    pub items: Vec<DialectItem>,
}

pub enum DialectItem {
    Operation(OperationDef),
    TypeDef(TypeDefData),
}

pub struct OperationDef {
    /// Clean name without `r#` prefix (e.g., "return")
    pub name: String,
    pub attrs: Vec<AttrDef>,
    pub operands: Vec<Operand>,
    pub results: ResultDef,
    pub regions: Vec<RegionOrSuccessor>,
    pub type_vars: Vec<TypeVar>,
    pub result_constraint: ValueExpr,
    /// `#[verify]`: call the wrapper's `Verify` impl after the schema checks. Holds the attribute's span for diagnostics.
    pub verify: Option<proc_macro2::Span>,
}

pub struct TypeDefData {
    /// Clean name without `r#` prefix (e.g., "nil")
    pub name: String,
    /// Original ident for use in generated code
    #[allow(dead_code)]
    pub raw_ident: Ident,
    pub params: Vec<TypeParam>,
    pub attrs: Vec<AttrDef>,
}

pub struct TypeParam {
    #[allow(dead_code)]
    pub name: String,
    pub raw_ident: Ident,
    pub variadic: bool,
}

pub struct AttrDef {
    /// Clean name without `r#` prefix
    pub name: String,
    /// Original ident
    pub raw_ident: Ident,
    pub kind: AttrKind,
    /// `Attr<[K]>`: a list whose every element has kind `kind`.
    pub list: bool,
    pub optional: bool,
    pub binds: Option<usize>,
}

/// The kind of a declared attribute, or of a list attribute's elements.
#[derive(Clone)]
pub enum AttrKind {
    /// `_`: any attribute value.
    Any,
    /// `V::Type`: a type attribute bound to a type variable.
    BoundType,
    /// A Rust type implementing `trunk_ir::attr_kind::AttrKind`.
    Path(KindType),
}

impl AttrKind {
    /// Whether this is the string kind, the one kind recognized by name: its
    /// accessors come with a `<name>_ref` handle accessor, and its setters
    /// take anything convertible to a `StringArg`.
    pub fn is_string(&self) -> bool {
        matches!(self, AttrKind::Path(path) if path.is_ident("String"))
    }
}

pub struct Operand {
    /// Clean name without `r#` prefix (used in tests/diagnostics)
    #[allow(dead_code)]
    pub name: String,
    /// Original ident
    pub raw_ident: Ident,
    pub variadic: bool,
    pub constraint: ValueExpr,
}

pub enum ResultDef {
    None,
    Single(String),
    Variadic(String),
    /// Zero or one result, declared as `-> Option<result>`.
    Optional(String),
}

pub enum RegionOrSuccessor {
    /// A region; `optional` regions are declared as `#[region(name?)]`.
    Region { name: String, optional: bool },
    /// A successor; a `variadic` one, declared as `#[successors(name)]`, is
    /// the rest of the successor list.
    Successor { name: String, variadic: bool },
}

// ============================================================================
// Top-level input parsing
// ============================================================================

/// Parse attribute macro input: `attr` must be empty, `item` contains
/// `mod name { ... }`.
pub fn parse_input(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> Result<DialectModule, String> {
    if !attr.is_empty() {
        return Err("`#[dialect]` does not accept arguments".to_string());
    }
    parse_module(item)
}

/// Parse `mod name { ... }` from the item token stream.
fn parse_module(stream: proc_macro2::TokenStream) -> Result<DialectModule, String> {
    let mut iter = stream.to_token_iter();
    let module = parse_module_inner(&mut iter)?;
    expect_consumed(&iter, "module")?;
    Ok(module)
}

// ============================================================================
// Module parsing
// ============================================================================

/// Parse `mod dialect_name { items... }`.
fn parse_module_inner(iter: &mut TokenIter) -> Result<DialectModule, String> {
    // Expect "mod"
    let kw: Ident = Ident::parser(iter).map_err(|e| format!("expected `mod`: {e}"))?;
    if kw != "mod" {
        return Err(format!("expected `mod`, got `{kw}`"));
    }

    // Parse dialect name
    let name_ident: Ident =
        Ident::parser(iter).map_err(|e| format!("expected dialect name: {e}"))?;
    let name = ident_str(&name_ident);

    // Parse brace body
    let body = expect_group(iter, Delimiter::Brace)?;
    let mut body_iter = body.stream().to_token_iter();

    let mut items = Vec::new();
    let mut seen_names = HashSet::default();
    while has_remaining(&body_iter) {
        let item = parse_item(&mut body_iter)?;
        let item_name = match &item {
            DialectItem::Operation(op) => Some(op.name.clone()),
            DialectItem::TypeDef(td) => Some(td.name.clone()),
        };
        if let Some(name) = item_name
            && !seen_names.insert(name.clone())
        {
            return Err(format!("duplicate item name: `{name}`"));
        }
        items.push(item);
    }

    Ok(DialectModule { name, items })
}

// ============================================================================
// Item parsing
// ============================================================================

fn parse_item(iter: &mut TokenIter) -> Result<DialectItem, String> {
    let mut op_attrs = Vec::new();
    let mut rest_results = false;
    let mut verify = None;

    // Collect outer attributes: #[doc = "..."] (skip), #[attr(...)],
    // #[rest_results], #[verify]
    while peek_punct(iter, '#') {
        let attr = parse_outer_attr(iter)?;
        match attr {
            OuterAttr::Doc => { /* skip doc comments */ }
            OuterAttr::OpAttrs(attrs) => {
                if !op_attrs.is_empty() {
                    return Err(
                        "duplicate #[attr(...)] on the same operation; merge into a single #[attr(...)]".into(),
                    );
                }
                op_attrs = attrs;
            }
            OuterAttr::RestResults => {
                if rest_results {
                    return Err("duplicate #[rest_results] on the same operation".into());
                }
                rest_results = true;
            }
            OuterAttr::Verify(span) => {
                if verify.is_some() {
                    return Err("duplicate #[verify] on the same operation".into());
                }
                verify = Some(span);
            }
        }
    }

    // Parse keyword: fn or struct
    let kw: Ident = Ident::parser(iter).map_err(|e| format!("expected `fn` or `struct`: {e}"))?;

    match kw.to_string().as_str() {
        "fn" => {
            if !op_attrs.is_empty() {
                return Err(
                    "`#[attr(..)]` is not supported on operations; declare `Attr<..>` parameters"
                        .into(),
                );
            }
            if rest_results {
                return Err("`#[rest_results]` is not supported; declare `-> Variadic<_>`".into());
            }
            let mut op = parse_operation(iter)?;
            if verify.is_some() && entity_names(&op).any(|name| name == "verify") {
                return Err("#[verify] reserves the name `verify` for `Verify::verify`".into());
            }
            for name in entity_names(&op) {
                if RESERVED_WRAPPER_NAMES.contains(&name) {
                    return Err(format!(
                        "`{name}` is reserved for the operation wrapper's own method"
                    ));
                }
            }
            for attr in &op.attrs {
                if !attr.kind.is_string() {
                    continue;
                }
                let handle = format!("{}_ref", attr.name);
                if RESERVED_WRAPPER_NAMES.contains(&handle.as_str()) {
                    return Err(format!(
                        "string attribute `{}` would generate `{handle}`, which is reserved for the operation wrapper's own method; rename the attribute",
                        attr.name
                    ));
                }
                if entity_names(&op).any(|name| name == handle) {
                    return Err(format!(
                        "string attribute `{}` reserves the name `{handle}` for its handle accessor",
                        attr.name
                    ));
                }
            }
            op.verify = verify;
            Ok(DialectItem::Operation(op))
        }
        "struct" => {
            if rest_results {
                return Err("#[rest_results] is not allowed on struct items".into());
            }
            if verify.is_some() {
                return Err("#[verify] is not allowed on struct items".into());
            }
            let td = parse_struct_def(iter, op_attrs)?;
            Ok(DialectItem::TypeDef(td))
        }
        other => Err(format!("expected `fn` or `struct`, got `{other}`")),
    }
}

enum OuterAttr {
    Doc,
    OpAttrs(Vec<AttrDef>),
    RestResults,
    Verify(proc_macro2::Span),
}

/// Methods every operation wrapper defines itself; an entity accessor must not
/// take one of these names.
const RESERVED_WRAPPER_NAMES: &[&str] = &["op_ref", "from_op", "matches", "operands"];

/// Names of an operation's entities, each of which becomes an accessor.
fn entity_names(op: &OperationDef) -> impl Iterator<Item = &str> {
    let results: Vec<&str> = match &op.results {
        ResultDef::None => Vec::new(),
        ResultDef::Single(name) | ResultDef::Variadic(name) | ResultDef::Optional(name) => {
            vec![name]
        }
    };
    op.operands
        .iter()
        .map(|operand| operand.name.as_str())
        .chain(op.attrs.iter().map(|attr| attr.name.as_str()))
        .chain(op.regions.iter().map(|item| match item {
            RegionOrSuccessor::Region { name, .. } | RegionOrSuccessor::Successor { name, .. } => {
                name.as_str()
            }
        }))
        .chain(results)
}

/// Parse `#[...]` — `#[doc = "..."]`, `#[attr(...)]`, `#[rest_results]`, or
/// `#[verify]`.
fn parse_outer_attr(iter: &mut TokenIter) -> Result<OuterAttr, String> {
    expect_punct(iter, '#')?;
    let bracket = expect_group(iter, Delimiter::Bracket)?;
    let mut inner = bracket.stream().to_token_iter();

    let ident: Ident =
        Ident::parser(&mut inner).map_err(|e| format!("expected attribute name: {e}"))?;

    match ident.to_string().as_str() {
        "doc" => {
            // doc attributes may have `= "..."` content; skip remaining tokens
            Ok(OuterAttr::Doc)
        }
        "attr" => {
            let paren = expect_group(&mut inner, Delimiter::Parenthesis)?;
            expect_consumed(&inner, "#[attr(...)]")?;
            let attrs = parse_attr_list(paren.stream())?;
            Ok(OuterAttr::OpAttrs(attrs))
        }
        "rest_results" => {
            expect_consumed(&inner, "#[rest_results]")?;
            Ok(OuterAttr::RestResults)
        }
        "verify" => {
            expect_consumed(&inner, "#[verify]")?;
            Ok(OuterAttr::Verify(ident.span()))
        }
        other => Err(format!(
            "unexpected attribute `{other}`, expected `doc`, `attr`, `rest_results`, or `verify`"
        )),
    }
}

/// Parse comma-separated attribute definitions: `name: Type, opt?: Type`.
fn parse_attr_list(stream: proc_macro2::TokenStream) -> Result<Vec<AttrDef>, String> {
    let mut iter = stream.to_token_iter();
    let mut attrs = Vec::new();
    let mut seen_names = HashSet::default();

    while has_remaining(&iter) {
        let name_ident: Ident =
            Ident::parser(&mut iter).map_err(|e| format!("expected attribute name: {e}"))?;

        let name = ident_str(&name_ident);
        if !seen_names.insert(name.clone()) {
            return Err(format!("duplicate attribute name: `{name}`"));
        }

        // Check for optional marker '?'
        let optional = peek_punct(&iter, '?');
        if optional {
            consume_punct(&mut iter)?;
        }

        expect_punct(&mut iter, ':')?;

        let ty_ident: Ident =
            Ident::parser(&mut iter).map_err(|e| format!("expected attribute type: {e}"))?;

        attrs.push(AttrDef {
            name,
            raw_ident: name_ident,
            kind: AttrKind::Path(KindType::from_ident(ty_ident)),
            list: false,
            optional,
            binds: None,
        });

        // Require comma between elements (trailing comma is allowed)
        if has_remaining(&iter) {
            if !peek_punct(&iter, ',') {
                return Err("expected `,` between attributes".into());
            }
            consume_punct(&mut iter)?;
        }
    }

    Ok(attrs)
}

// ============================================================================
// Operation parsing
// ============================================================================

fn parse_operation(iter: &mut TokenIter) -> Result<OperationDef, String> {
    let name_ident: Ident =
        Ident::parser(iter).map_err(|e| format!("expected operation name: {e}"))?;
    constraint::parse_typed_operation(iter, name_ident)
}

/// Parse body content: `#[region(name)] {}`, `#[successor(name)] {}`, and
/// `#[successors(name)] {}`.
fn parse_regions(stream: proc_macro2::TokenStream) -> Result<Vec<RegionOrSuccessor>, String> {
    let mut iter = stream.to_token_iter();
    let mut items = Vec::new();
    let mut seen_names = HashSet::default();

    while has_remaining(&iter) {
        expect_punct(&mut iter, '#')?;
        let bracket = expect_group(&mut iter, Delimiter::Bracket)?;
        let mut inner = bracket.stream().to_token_iter();

        let kw: Ident = Ident::parser(&mut inner)
            .map_err(|e| format!("expected `region` or `successor`: {e}"))?;

        let paren = expect_group(&mut inner, Delimiter::Parenthesis)?;
        expect_consumed(&inner, "#[region/successor(...)]")?;
        let mut name_iter = paren.stream().to_token_iter();
        let name_ident: Ident = Ident::parser(&mut name_iter)
            .map_err(|e| format!("expected region/successor name: {e}"))?;
        let name = ident_str(&name_ident);
        let optional = if peek_punct(&name_iter, '?') {
            consume_punct(&mut name_iter)?;
            true
        } else {
            false
        };
        expect_consumed(&name_iter, "region/successor name")?;

        if !seen_names.insert(name.clone()) {
            return Err(format!("duplicate region/successor name: `{name}`"));
        }

        match kw.to_string().as_str() {
            "region" => {
                let _body = expect_group(&mut iter, Delimiter::Brace)?;
                if items
                    .iter()
                    .any(|item| matches!(item, RegionOrSuccessor::Region { optional: true, .. }))
                {
                    return Err("an optional region must be the last region".into());
                }
                items.push(RegionOrSuccessor::Region { name, optional });
            }
            keyword @ ("successor" | "successors") => {
                if optional {
                    return Err("successors cannot be optional".into());
                }
                if items
                    .iter()
                    .any(|item| matches!(item, RegionOrSuccessor::Successor { variadic: true, .. }))
                {
                    return Err("variadic successors must be the last successors".into());
                }
                // Consume `{}` after the attribute (required for valid Rust syntax)
                let _body = expect_group(&mut iter, Delimiter::Brace)?;
                items.push(RegionOrSuccessor::Successor {
                    name,
                    variadic: keyword == "successors",
                });
            }
            other => {
                return Err(format!(
                    "expected `region`, `successor`, or `successors`, got `{other}`"
                ));
            }
        }
    }

    Ok(items)
}

// ============================================================================
// Struct (type) definition parsing
// ============================================================================

/// Parse `struct Name;` or `struct Name<Param, ...>;`
fn parse_struct_def(iter: &mut TokenIter, attrs: Vec<AttrDef>) -> Result<TypeDefData, String> {
    // Parse type name
    let name_ident: Ident =
        Ident::parser(iter).map_err(|e| format!("expected struct name: {e}"))?;
    let name = ident_str(&name_ident);

    // Optional generic params: <Param1, Param2>
    let params = if peek_punct(iter, '<') {
        parse_angle_params(iter)?
    } else {
        Vec::new()
    };

    expect_punct(iter, ';')?;

    Ok(TypeDefData {
        name,
        raw_ident: name_ident,
        params,
        attrs,
    })
}

/// Parse `<Ident, #[rest] Ident, ...>` angle-bracket generic parameters.
///
/// Supports `#[rest]` on the last type parameter for variadic params.
fn parse_angle_params(iter: &mut TokenIter) -> Result<Vec<TypeParam>, String> {
    expect_punct(iter, '<')?;
    let mut params = Vec::new();
    let mut seen_names = HashSet::default();
    let mut seen_variadic = false;

    while !peek_punct(iter, '>') {
        if !has_remaining(iter) {
            return Err("expected `>` to close generic parameters".into());
        }

        // Check for #[rest] marker
        let variadic = if peek_punct(iter, '#') {
            consume_punct(iter)?;
            let bracket = expect_group(iter, Delimiter::Bracket)?;
            let mut inner = bracket.stream().to_token_iter();
            let kw: Ident =
                Ident::parser(&mut inner).map_err(|e| format!("expected `rest`: {e}"))?;
            if kw != "rest" {
                return Err(format!("expected `rest`, got `{kw}`"));
            }
            expect_consumed(&inner, "#[rest]")?;
            if seen_variadic {
                return Err("at most one #[rest] type parameter is allowed".into());
            }
            seen_variadic = true;
            true
        } else {
            if seen_variadic {
                return Err("#[rest] type parameter must be the last parameter".into());
            }
            false
        };

        let param_ident: Ident =
            Ident::parser(iter).map_err(|e| format!("expected type parameter name: {e}"))?;
        let param_name = ident_str(&param_ident);
        if param_name == "Type" {
            return Err("type parameter `Type` is reserved for the Type projection".into());
        }
        if !seen_names.insert(param_name.clone()) {
            return Err(format!("duplicate type parameter: `{param_name}`"));
        }
        params.push(TypeParam {
            name: param_name,
            raw_ident: param_ident,
            variadic,
        });

        // Require comma between params (trailing comma allowed)
        if !peek_punct(iter, '>') {
            if !peek_punct(iter, ',') {
                return Err("expected `,` between type parameters".into());
            }
            consume_punct(iter)?;
        }
    }

    expect_punct(iter, '>')?;
    Ok(params)
}

// ============================================================================
// Helper functions
// ============================================================================

/// Strip `r#` prefix from an ident.
fn ident_str(ident: &Ident) -> String {
    let s = ident.to_string();
    s.strip_prefix("r#").unwrap_or(&s).to_string()
}

fn peek_punct(iter: &TokenIter, ch: char) -> bool {
    matches!(iter.clone().next(), Some(TokenTree::Punct(p)) if p.as_char() == ch)
}

fn has_remaining(iter: &TokenIter) -> bool {
    iter.clone().next().is_some()
}

fn expect_punct(iter: &mut TokenIter, ch: char) -> Result<(), String> {
    let tt: TokenTree = TokenTree::parser(iter).map_err(|e| format!("expected `{ch}`: {e}"))?;
    match tt {
        TokenTree::Punct(p) if p.as_char() == ch => Ok(()),
        other => Err(format!("expected `{ch}`, got `{other}`")),
    }
}

/// Consume any single punct token.
fn consume_punct(iter: &mut TokenIter) -> Result<(), String> {
    let tt: TokenTree = TokenTree::parser(iter).map_err(|e| format!("expected punct: {e}"))?;
    match tt {
        TokenTree::Punct(_) => Ok(()),
        other => Err(format!("expected punct, got `{other}`")),
    }
}

/// Ensure a token iterator is fully consumed. Returns an error if any tokens remain.
fn expect_consumed(iter: &TokenIter, context: &str) -> Result<(), String> {
    if let Some(tt) = iter.clone().next() {
        Err(format!("unexpected trailing token `{tt}` in {context}"))
    } else {
        Ok(())
    }
}

fn expect_group(iter: &mut TokenIter, delim: Delimiter) -> Result<proc_macro2::Group, String> {
    let tt: TokenTree =
        TokenTree::parser(iter).map_err(|e| format!("expected {delim:?} group: {e}"))?;
    match tt {
        TokenTree::Group(g) if g.delimiter() == delim => Ok(g),
        other => Err(format!("expected {delim:?} group, got `{other}`")),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn parse_test_module(item: proc_macro2::TokenStream) -> Result<DialectModule, String> {
        parse_input(quote! {}, item)
    }

    #[test]
    fn test_parse_empty_attr() {
        let module = parse_input(
            quote! {},
            quote! {
                mod arith {
                    fn add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
                }
            },
        )
        .unwrap();

        assert_eq!(module.name, "arith");
    }

    #[test]
    fn test_parse_rejects_non_empty_attr() {
        // The macro emits absolute `::trunk_ir::...` paths, so the
        // legacy `crate = crate` knob has no purpose. Reject any
        // attribute argument outright.
        let result = parse_input(
            quote! { crate = crate },
            quote! {
                mod arith {
                    fn add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
                }
            },
        );

        let Err(err) = result else {
            panic!("expected error for non-empty attr");
        };
        assert!(err.contains("does not accept arguments"), "got: {err}");
    }

    #[test]
    fn test_parse_simple_module() {
        let module = parse_test_module(quote! {
            mod arith {
                fn add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
            }
        })
        .unwrap();

        assert_eq!(module.name, "arith");
        assert_eq!(module.items.len(), 1);
        match &module.items[0] {
            DialectItem::Operation(op) => {
                assert_eq!(op.name, "add");
                assert_eq!(op.operands.len(), 2);
                assert!(!op.operands[0].variadic);
                assert_eq!(op.operands[0].name, "lhs");
                assert!(matches!(&op.results, ResultDef::Single(s) if s == "result"));
            }
            _ => panic!("expected operation"),
        }
    }

    #[test]
    fn test_parse_variadic_operands() {
        let module = parse_test_module(quote! {
            mod func {
                fn call(args: Variadic<_>) -> Value<_> {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.operands.len(), 1);
        assert!(op.operands[0].variadic);
        assert_eq!(op.operands[0].name, "args");
    }

    #[test]
    fn test_parse_mixed_operands() {
        let module = parse_test_module(quote! {
            mod func {
                fn call_indirect(callee: Value<_>, args: Variadic<_>) -> Value<_> {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.operands.len(), 2);
        assert!(!op.operands[0].variadic);
        assert_eq!(op.operands[0].name, "callee");
        assert!(op.operands[1].variadic);
        assert_eq!(op.operands[1].name, "args");
    }

    #[test]
    fn test_parse_attributes() {
        let module = parse_test_module(quote! {
            mod adt {
                fn struct_get(r#type: Attr<Type>, field: Attr<u32>, r#ref: Value<_>) -> Value<_> {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.attrs.len(), 2);
        assert_eq!(op.attrs[0].name, "type");
        assert!(!op.attrs[0].optional);
        assert!(matches!(&op.attrs[0].kind, AttrKind::Path(path) if path.is_ident("Type")));
        assert_eq!(op.attrs[1].name, "field");
        assert!(matches!(&op.attrs[1].kind, AttrKind::Path(path) if path.is_ident("u32")));
    }

    #[test]
    fn test_parse_list_attributes() {
        let module = parse_test_module(quote! {
            mod rtti {
                fn layout(fields: Attr<[String]>, sizes: Option<Attr<[u32]>>) {}
            }
        })
        .unwrap();

        let DialectItem::Operation(op) = &module.items[0] else {
            panic!("expected operation")
        };
        assert!(op.attrs[0].list && op.attrs[0].kind.is_string());
        assert!(op.attrs[1].list && op.attrs[1].optional);
        assert!(matches!(&op.attrs[1].kind, AttrKind::Path(path) if path.is_ident("u32")));

        for (kind, expected) in [
            (quote!([_]), "a list attribute needs a named element kind"),
            (quote!([[u32]]), "invalid attribute kind"),
            (quote!([u32, u32]), "expected one element type"),
        ] {
            let error = parse_test_module(quote! {
                mod rtti {
                    fn layout(fields: Attr<#kind>) {}
                }
            })
            .err()
            .expect(expected);
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn test_parse_optional_attributes() {
        let module = parse_test_module(quote! {
            mod wasm {
                fn table(reftype: Attr<SymbolRef>, min: Attr<u32>, max: Option<Attr<u32>>) {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.attrs.len(), 3);
        assert!(!op.attrs[0].optional);
        assert!(!op.attrs[1].optional);
        assert!(op.attrs[2].optional);
        assert_eq!(op.attrs[2].name, "max");
    }

    #[test]
    fn test_parse_regions() {
        let module = parse_test_module(quote! {
            mod func {
                fn func(sym_name: Attr<String>) {
                    #[region(body)] {}
                }
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.regions.len(), 1);
        assert!(matches!(
            &op.regions[0],
            RegionOrSuccessor::Region { name, optional: false } if name == "body"
        ));
    }

    #[test]
    fn test_parse_successors() {
        let module = parse_test_module(quote! {
            mod cf {
                fn cond_br(cond: Value<_>) {
                    #[successor(then_dest)] {}
                    #[successor(else_dest)] {}
                }
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.regions.len(), 2);
        assert!(matches!(
            &op.regions[0],
            RegionOrSuccessor::Successor { name, variadic: false } if name == "then_dest"
        ));
        assert!(matches!(
            &op.regions[1],
            RegionOrSuccessor::Successor { name, variadic: false } if name == "else_dest"
        ));
    }

    #[test]
    fn test_parse_variadic_successors() {
        let module = parse_test_module(quote! {
            mod test {
                fn switch(index: Value<_>) {
                    #[successor(default)] {}
                    #[successors(targets)] {}
                }
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert!(matches!(
            &op.regions[0],
            RegionOrSuccessor::Successor { name, variadic: false } if name == "default"
        ));
        assert!(matches!(
            &op.regions[1],
            RegionOrSuccessor::Successor { name, variadic: true } if name == "targets"
        ));

        let not_last = parse_test_module(quote! {
            mod test {
                fn switch(index: Value<_>) {
                    #[successors(targets)] {}
                    #[successor(default)] {}
                }
            }
        });
        assert!(
            not_last
                .err()
                .unwrap()
                .contains("variadic successors must be the last successors")
        );
    }

    #[test]
    fn test_parse_raw_identifiers() {
        let module = parse_test_module(quote! {
            mod scf {
                fn r#return(values: Variadic<_>) {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.name, "return");
    }

    #[test]
    fn test_parse_no_result_no_operands() {
        let module = parse_test_module(quote! {
            mod func {
                fn unreachable() {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.name, "unreachable");
        assert!(op.operands.is_empty());
        assert!(matches!(op.results, ResultDef::None));
        assert!(op.regions.is_empty());
    }

    #[test]
    fn test_parse_variadic_results() {
        let module = parse_test_module(quote! {
            mod wasm {
                fn call(args: Variadic<_>) -> Variadic<_> {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert!(matches!(&op.results, ResultDef::Variadic(s) if s == "results"));
    }

    #[test]
    fn test_parse_optional_result_and_region() {
        let module = parse_test_module(quote! {
            mod test {
                fn maybe(cond: Value<_>) -> Option<Value<_>> {
                    #[region(first)]
                    {}
                    #[region(body?)]
                    {}
                }
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert!(matches!(&op.results, ResultDef::Optional(s) if s == "result"));
        assert!(matches!(
            &op.regions[0],
            RegionOrSuccessor::Region { name, optional: false } if name == "first"
        ));
        assert!(matches!(
            &op.regions[1],
            RegionOrSuccessor::Region { name, optional: true } if name == "body"
        ));
    }

    #[test]
    fn test_parse_rejects_malformed_modules() {
        // (case, module, fragments the error must contain)
        let cases: [(&str, proc_macro2::TokenStream, &[&str]); 11] = [
            (
                "optional region not last",
                quote! {
                    mod test {
                        fn f() {
                            #[region(body?)]
                            {}
                            #[region(tail)]
                            {}
                        }
                    }
                },
                &["optional region must be the last region"],
            ),
            (
                "optional successor",
                quote! {
                    mod test {
                        fn f() {
                            #[successor(dest?)]
                            {}
                        }
                    }
                },
                &["successors cannot be optional"],
            ),
            (
                "duplicate entity name",
                quote! {
                    mod test {
                        fn op(a: Value<_>, a: Value<_>) {}
                    }
                },
                &["duplicate entity name `a`"],
            ),
            (
                "duplicate region name",
                quote! {
                    mod test {
                        fn op() { #[region(body)] {} #[region(body)] {} };
                    }
                },
                &["duplicate region/successor name"],
            ),
            (
                "#[rest_results] on struct",
                quote! {
                    mod test {
                        #[rest_results]
                        struct Nil;
                    }
                },
                &["#[rest_results] is not allowed on struct items"],
            ),
            (
                "duplicate type name",
                quote! {
                    mod test {
                        struct Nil;
                        struct Nil;
                    }
                },
                &["duplicate item name"],
            ),
            (
                "operation and type name collision",
                quote! {
                    mod test {
                        fn foo() {}
                        struct foo;
                    }
                },
                &["duplicate item name"],
            ),
            (
                "duplicate operation name",
                quote! {
                    mod test {
                        fn add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
                        fn add(a: Value<_>) -> Value<_> {}
                    }
                },
                &["duplicate item name"],
            ),
            (
                "#[rest] not the last struct parameter",
                quote! {
                    mod test {
                        struct Bad<#[rest] A, B>;
                    }
                },
                &["must be the last parameter"],
            ),
            (
                "multiple #[rest] struct parameters",
                quote! {
                    mod test {
                        struct Bad<#[rest] A, #[rest] B>;
                    }
                },
                &["rest", "one"],
            ),
            (
                "trailing tokens after the module",
                quote! {
                    mod test {
                        fn op() {}
                    }
                    garbage
                },
                &["trailing"],
            ),
        ];

        for (case, item, fragments) in cases {
            let err = parse_test_module(item)
                .err()
                .unwrap_or_else(|| panic!("{case}: expected a parse error"));
            for fragment in fragments {
                assert!(
                    err.contains(fragment),
                    "{case}: missing {fragment:?} in: {err}"
                );
            }
        }
    }

    #[test]
    fn test_parse_unit_struct() {
        let module = parse_test_module(quote! {
            mod core {
                struct Nil;
            }
        })
        .unwrap();

        assert_eq!(module.items.len(), 1);
        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.name, "Nil");
                assert!(td.params.is_empty());
                assert!(td.attrs.is_empty());
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn test_parse_struct_with_param() {
        let module = parse_test_module(quote! {
            mod core {
                struct Array<Element>;
            }
        })
        .unwrap();

        assert_eq!(module.items.len(), 1);
        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.name, "Array");
                assert_eq!(td.params.len(), 1);
                assert_eq!(td.params[0].raw_ident.to_string(), "Element");
                assert!(td.attrs.is_empty());
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn test_parse_struct_with_multiple_params() {
        let module = parse_test_module(quote! {
            mod core {
                struct Pair<First, Second>;
            }
        })
        .unwrap();

        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.params.len(), 2);
                assert_eq!(td.params[0].raw_ident.to_string(), "First");
                assert_eq!(td.params[1].raw_ident.to_string(), "Second");
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn test_parse_struct_with_attrs() {
        let module = parse_test_module(quote! {
            mod core {
                #[attr(nullable: bool)]
                struct Ref<Pointee>;
            }
        })
        .unwrap();

        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.name, "Ref");
                assert_eq!(td.params.len(), 1);
                assert_eq!(td.attrs.len(), 1);
                assert_eq!(td.attrs[0].name, "nullable");
                assert!(matches!(&td.attrs[0].kind, AttrKind::Path(path) if path.is_ident("bool")));
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn test_parse_struct_and_fn_together() {
        let module = parse_test_module(quote! {
            mod core {
                struct Nil;
                fn add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
            }
        })
        .unwrap();

        assert_eq!(module.items.len(), 2);
        assert!(matches!(module.items[0], DialectItem::TypeDef(_)));
        assert!(matches!(module.items[1], DialectItem::Operation(_)));
    }

    #[test]
    fn test_parse_multiple_operations() {
        let module = parse_test_module(quote! {
            mod arith {
                fn add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
                fn sub(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
                fn neg(operand: Value<_>) -> Value<_> {}
            }
        })
        .unwrap();

        assert_eq!(module.items.len(), 3);
    }

    #[test]
    fn test_verify_marks_operations() {
        let module = parse_test_module(quote! {
            mod test {
                #[verify]
                fn checked(x: Value<_>) {}
                fn plain(xs: Variadic<_>) {}
            }
        })
        .unwrap();
        let flags: Vec<bool> = module
            .items
            .iter()
            .map(|item| match item {
                DialectItem::Operation(op) => op.verify.is_some(),
                DialectItem::TypeDef(_) => unreachable!(),
            })
            .collect();
        assert_eq!(flags, [true, false]);

        for (item, expected) in [
            (
                quote!(
                    mod test {
                        #[verify]
                        #[verify]
                        fn op() {}
                    }
                ),
                "duplicate #[verify]",
            ),
            (
                quote!(
                    mod test {
                        #[verify]
                        struct Ty;
                    }
                ),
                "#[verify] is not allowed on struct items",
            ),
            (
                quote!(
                    mod test {
                        #[verify(x)]
                        fn op() {}
                    }
                ),
                "#[verify]",
            ),
            (
                quote!(
                    mod test {
                        #[verify]
                        fn op(verify: Value<_>) {}
                    }
                ),
                "reserves the name `verify`",
            ),
        ] {
            let err = parse_test_module(item).err().expect("should fail");
            assert!(err.contains(expected), "unexpected error: {err}");
        }
    }

    #[test]
    fn test_parse_region_with_result() {
        let module = parse_test_module(quote! {
            mod scf {
                fn r#if(cond: Value<_>) -> Value<_> {
                    #[region(then_region)] {}
                    #[region(else_region)] {}
                }
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.name, "if");
        assert!(matches!(&op.results, ResultDef::Single(s) if s == "result"));
        assert_eq!(op.regions.len(), 2);
    }

    #[test]
    fn test_trailing_comma_accepted_in_attr_params() {
        let module = parse_test_module(quote! {
            mod test {
                fn op(a: Attr<u32>, b: Attr<u32>,) {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.attrs.len(), 2);
    }

    #[test]
    fn test_trailing_comma_accepted_in_operands() {
        let module = parse_test_module(quote! {
            mod test {
                fn op(a: Value<_>, b: Value<_>,) {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert_eq!(op.operands.len(), 2);
    }

    // ================================================================
    // Variadic type params (#[rest])
    // ================================================================

    #[test]
    fn test_parse_struct_variadic_only() {
        let module = parse_test_module(quote! {
            mod core {
                struct Tuple<#[rest] Elements>;
            }
        })
        .unwrap();

        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.name, "Tuple");
                assert_eq!(td.params.len(), 1);
                assert!(td.params[0].variadic);
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn test_parse_struct_mixed_fixed_and_variadic() {
        let module = parse_test_module(quote! {
            mod core {
                struct Func<Return, #[rest] Params>;
            }
        })
        .unwrap();

        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.params.len(), 2);
                assert!(!td.params[0].variadic);
                assert_eq!(td.params[0].raw_ident.to_string(), "Return");
                assert!(td.params[1].variadic);
                assert_eq!(td.params[1].raw_ident.to_string(), "Params");
            }
            _ => panic!("expected TypeDef"),
        }
    }
}
