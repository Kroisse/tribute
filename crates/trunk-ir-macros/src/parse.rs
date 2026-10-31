//! DSL parser for `#[dialect]`.
//!
//! Parses the module body into structured types for code generation.

use heck::ToSnakeCase;
use proc_macro2::{Delimiter, Ident, TokenTree};
use rustc_hash::FxHashSet as HashSet;
use unsynn::{Parser, ToTokenIter, TokenIter};

mod constraint;
use constraint::parse_type_fields;
pub use constraint::{
    BoundPath, KindType, ListExpr, Projection, RelationDef, RelationExpr, TypeExpr, TypeVar,
    ValueExpr,
};

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
    /// Equalities from the `where` clause.
    pub relations: Vec<RelationDef>,
    /// `#[verify]`: call the wrapper's `Verify` impl after the schema checks. Holds the attribute's span for diagnostics.
    pub verify: Option<proc_macro2::Span>,
    /// `#[asm_format]`: print and parse through the wrapper's `OpAsmFormat` impl. Holds the attribute's span for diagnostics.
    pub asm_format: Option<proc_macro2::Span>,
    /// `#[asm_format(..)]`: the format elements the printer and parser
    /// interpret instead of an `OpAsmFormat` impl.
    pub format: Option<proc_macro2::TokenStream>,
    /// `#[implements(..)]`: the operation interfaces the wrapper implements.
    pub implements: Vec<PathEntry>,
    /// `#[flags(..)]`: the operation flags the definition declares.
    pub flags: Vec<PathEntry>,
    /// Doc comments, forwarded to the wrapper struct.
    pub docs: Vec<proc_macro2::Group>,
}

/// One entry of `#[implements(..)]` or `#[flags(..)]`: an interface trait's
/// or an operation flag's path.
pub struct PathEntry {
    pub path: proc_macro2::TokenStream,
    /// The entry's first token, where a wrong entry is reported.
    pub span: proc_macro2::Span,
}

pub struct TypeDefData {
    /// Clean name without `r#` prefix (e.g., "nil")
    pub name: String,
    /// Original ident for use in generated code
    #[allow(dead_code)]
    pub raw_ident: Ident,
    pub params: Vec<TypeParam>,
    pub attrs: Vec<AttrDef>,
    /// `#[verify]`: run the wrapper's `VerifyType` impl in IR validation. Holds the attribute's span for diagnostics.
    pub verify: Option<proc_macro2::Span>,
    /// `#[asm_format]`: print and parse through the wrapper's `TypeAsmFormat` impl. Holds the attribute's span for diagnostics.
    pub asm_format: Option<proc_macro2::Span>,
    /// `#[alias_hint]`: suggest printer alias names through the wrapper's `TypeAliasHint` impl. Holds the attribute's span for diagnostics.
    pub alias_hint: Option<proc_macro2::Span>,
    /// `#[implements(..)]`: the type interfaces the wrapper implements.
    pub implements: Vec<PathEntry>,
    /// Doc comments, forwarded to the wrapper struct.
    pub docs: Vec<proc_macro2::Group>,
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
    /// Doc comments, forwarded to the accessor.
    pub docs: Vec<proc_macro2::Group>,
}

/// The kind of a declared attribute, or of a list attribute's elements.
#[derive(Clone)]
pub enum AttrKind {
    /// `_`: any attribute value.
    Any,
    /// `TypeOf<V>`: a type attribute bound to a type variable.
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
    /// A region; `optional` regions are declared as `#[region(name?)]`, and
    /// a `variadic` one, declared as `#[regions(name)]`, is the rest of the
    /// region list.
    Region {
        name: String,
        optional: bool,
        variadic: bool,
    },
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
    let mut legacy_attrs = false;
    let mut rest_results = false;
    let mut verify = None;
    let mut asm_format = None;
    let mut alias_hint = None;
    let mut implements: Option<Vec<PathEntry>> = None;
    let mut flags: Option<Vec<PathEntry>> = None;
    let mut format: Option<proc_macro2::TokenStream> = None;
    let mut docs = Vec::new();

    // Collect outer attributes: #[doc = "..."], #[verify], #[asm_format],
    // #[alias_hint], #[implements(..)], #[flags(..)],
    // and the legacy #[attr(...)] and #[rest_results], rejected below
    while peek_punct(iter, '#') {
        let attr = parse_outer_attr(iter)?;
        match attr {
            OuterAttr::Doc(group) => docs.push(group),
            OuterAttr::LegacyAttrs => legacy_attrs = true,
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
            OuterAttr::AsmFormat(span, elements) => {
                if asm_format.is_some() {
                    return Err("duplicate #[asm_format] on the same operation".into());
                }
                asm_format = Some(span);
                format = elements;
            }
            OuterAttr::AliasHint(span) => {
                if alias_hint.is_some() {
                    return Err("duplicate #[alias_hint] on the same type".into());
                }
                alias_hint = Some(span);
            }
            OuterAttr::Implements(entries) => {
                if implements.is_some() {
                    return Err("duplicate #[implements] on the same operation".into());
                }
                implements = Some(entries);
            }
            OuterAttr::Flags(entries) => {
                if flags.is_some() {
                    return Err("duplicate #[flags] on the same operation".into());
                }
                flags = Some(entries);
            }
        }
    }

    // Parse keyword: fn or struct
    let kw: Ident = Ident::parser(iter).map_err(|e| format!("expected `fn` or `struct`: {e}"))?;

    match kw.to_string().as_str() {
        "fn" => {
            if legacy_attrs {
                return Err(
                    "`#[attr(..)]` is not supported on operations; declare `Attr<..>` parameters"
                        .into(),
                );
            }
            if rest_results {
                return Err("`#[rest_results]` is not supported; declare `-> Variadic<_>`".into());
            }
            if alias_hint.is_some() {
                return Err(
                    "#[alias_hint] suggests type alias names and is not allowed on operations"
                        .into(),
                );
            }
            let mut op = parse_operation(iter)?;
            if verify.is_some() && entity_names(&op).any(|name| name == "verify") {
                return Err("#[verify] reserves the name `verify` for `Verify::verify`".into());
            }
            if asm_format.is_some()
                && format.is_none()
                && let Some(name) = entity_names(&op).find(|name| ["print", "parse"].contains(name))
            {
                return Err(format!(
                    "#[asm_format] reserves the name `{name}` for `OpAsmFormat::{name}`"
                ));
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
            op.asm_format = asm_format;
            op.format = format;
            op.implements = implements.unwrap_or_default();
            op.flags = flags.unwrap_or_default();
            op.docs = docs;
            Ok(DialectItem::Operation(op))
        }
        "struct" => {
            if legacy_attrs {
                return Err(
                    "`#[attr(..)]` is not supported on types; declare `Attr<..>` fields".into(),
                );
            }
            if rest_results {
                return Err("#[rest_results] is not allowed on struct items".into());
            }
            if flags.is_some() {
                return Err("#[flags] declares operation flags and is not allowed on types".into());
            }
            if format.is_some() {
                return Err(
                    "a type's custom syntax is a hand-written `TypeAsmFormat`; write `#[asm_format]` without format elements".into(),
                );
            }
            let mut td = parse_struct_def(iter)?;
            // Each parameter, attribute, and string attribute handle has an
            // accessor on the wrapper.
            let accessors: Vec<String> = td
                .params
                .iter()
                .map(|param| param.raw_ident.to_string().to_snake_case())
                .chain(td.attrs.iter().flat_map(|attr| {
                    let handle = attr.kind.is_string().then(|| format!("{}_ref", attr.name));
                    std::iter::once(attr.name.clone()).chain(handle)
                }))
                .collect();
            let mut seen = HashSet::default();
            for name in &accessors {
                if RESERVED_TYPE_NAMES.contains(&name.as_str()) {
                    return Err(format!(
                        "`{name}` is reserved for the type wrapper's or builder's own method"
                    ));
                }
                if !seen.insert(name) {
                    return Err(format!("two accessors of the type are named `{name}`"));
                }
            }
            for (span, attr, names) in [
                (verify, "#[verify]", &["verify"][..]),
                (asm_format, "#[asm_format]", &["print", "parse"][..]),
                (alias_hint, "#[alias_hint]", &["alias_hint"][..]),
            ] {
                if span.is_some()
                    && let Some(name) = accessors.iter().find(|name| names.contains(&name.as_str()))
                {
                    return Err(format!(
                        "{attr} reserves the name `{name}` for the trait method"
                    ));
                }
            }
            td.verify = verify;
            td.asm_format = asm_format;
            td.alias_hint = alias_hint;
            td.implements = implements.unwrap_or_default();
            td.docs = docs;
            Ok(DialectItem::TypeDef(td))
        }
        other => Err(format!("expected `fn` or `struct`, got `{other}`")),
    }
}

enum OuterAttr {
    /// The bracketed `doc = ".."` attribute.
    Doc(proc_macro2::Group),
    /// The legacy `#[attr(..)]`, superseded by typed parameters and fields.
    LegacyAttrs,
    RestResults,
    Verify(proc_macro2::Span),
    /// `#[asm_format]`, with the format elements of `#[asm_format(..)]`.
    AsmFormat(proc_macro2::Span, Option<proc_macro2::TokenStream>),
    AliasHint(proc_macro2::Span),
    Implements(Vec<PathEntry>),
    Flags(Vec<PathEntry>),
}

/// Methods every operation wrapper defines itself; an entity accessor must not
/// take one of these names.
const RESERVED_WRAPPER_NAMES: &[&str] = &["op_ref", "from_op", "matches", "operands"];

/// Methods every type wrapper and its builder define themselves; a parameter
/// or attribute accessor must not take one of these names.
const RESERVED_TYPE_NAMES: &[&str] = &[
    "as_type_ref",
    "from_type_ref",
    "matches",
    "params",
    "param_attrs",
    "intern",
];

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

/// Parse `#[...]` — `#[doc = "..."]`, `#[attr(...)]`, `#[rest_results]`,
/// `#[verify]`, `#[asm_format]`, `#[alias_hint]`, `#[implements(..)]`, or
/// `#[flags(..)]`.
fn parse_outer_attr(iter: &mut TokenIter) -> Result<OuterAttr, String> {
    expect_punct(iter, '#')?;
    let bracket = expect_group(iter, Delimiter::Bracket)?;
    let mut inner = bracket.stream().to_token_iter();

    let ident: Ident =
        Ident::parser(&mut inner).map_err(|e| format!("expected attribute name: {e}"))?;

    match ident.to_string().as_str() {
        "doc" => Ok(OuterAttr::Doc(bracket)),
        "attr" => Ok(OuterAttr::LegacyAttrs),
        "rest_results" => {
            expect_consumed(&inner, "#[rest_results]")?;
            Ok(OuterAttr::RestResults)
        }
        "verify" => {
            expect_consumed(&inner, "#[verify]")?;
            Ok(OuterAttr::Verify(ident.span()))
        }
        "asm_format" => {
            let elements = if has_remaining(&inner) {
                let group = expect_group(&mut inner, Delimiter::Parenthesis).map_err(|_| {
                    "expected `#[asm_format]` or `#[asm_format(elements..)]`".to_string()
                })?;
                expect_consumed(&inner, "#[asm_format(..)]")?;
                Some(group.stream())
            } else {
                None
            };
            Ok(OuterAttr::AsmFormat(ident.span(), elements))
        }
        "alias_hint" => {
            expect_consumed(&inner, "#[alias_hint]")?;
            Ok(OuterAttr::AliasHint(ident.span()))
        }
        "implements" => Ok(OuterAttr::Implements(parse_path_list(
            &mut inner,
            "implements",
            "Interface",
        )?)),
        "flags" => Ok(OuterAttr::Flags(parse_path_list(
            &mut inner, "flags", "Flag",
        )?)),
        other => Err(format!(
            "unexpected attribute `{other}`, expected `doc`, `attr`, `rest_results`, `verify`, `asm_format`, `alias_hint`, `implements`, or `flags`"
        )),
    }
}

/// Parse the parenthesized, comma-separated paths of `#[attr(..)]`.
fn parse_path_list(
    inner: &mut TokenIter,
    attr: &str,
    example: &str,
) -> Result<Vec<PathEntry>, String> {
    let list = expect_group(inner, Delimiter::Parenthesis)
        .map_err(|_| format!("expected `#[{attr}({example}, ..)]`"))?;
    expect_consumed(inner, &format!("#[{attr}(..)]"))?;
    let mut entries: Vec<PathEntry> = Vec::new();
    let mut current = Vec::new();
    let mut tokens = list.stream().into_iter().peekable();
    while tokens.peek().is_some() || !current.is_empty() {
        match tokens.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ',' => {}
            Some(tt) => {
                current.push(tt);
                continue;
            }
            None => {}
        }
        let Some(first) = current.first() else {
            return Err(format!("empty entry in #[{attr}(..)]"));
        };
        let span = first.span();
        let path: proc_macro2::TokenStream = current.drain(..).collect();
        let text = path.to_string();
        if entries.iter().any(|entry| entry.path.to_string() == text) {
            return Err(format!("`{text}` is listed twice in #[{attr}(..)]"));
        }
        entries.push(PathEntry { path, span });
    }
    if entries.is_empty() {
        return Err(format!("#[{attr}(..)] lists nothing"));
    }
    Ok(entries)
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
            .map_err(|e| format!("expected `region`, `regions`, or `successor`: {e}"))?;

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
            keyword @ ("region" | "regions") => {
                let _body = expect_group(&mut iter, Delimiter::Brace)?;
                let variadic = keyword == "regions";
                if variadic && optional {
                    return Err("variadic regions cannot be optional".into());
                }
                if items
                    .iter()
                    .any(|item| matches!(item, RegionOrSuccessor::Region { optional: true, .. }))
                {
                    return Err("an optional region must be the last region".into());
                }
                if items
                    .iter()
                    .any(|item| matches!(item, RegionOrSuccessor::Region { variadic: true, .. }))
                {
                    return Err("variadic regions must be the last regions".into());
                }
                items.push(RegionOrSuccessor::Region {
                    name,
                    optional,
                    variadic,
                });
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
                    "expected `region`, `regions`, `successor`, or `successors`, got `{other}`"
                ));
            }
        }
    }

    Ok(items)
}

// ============================================================================
// Struct (type) definition parsing
// ============================================================================

/// Parse `struct Name;`, `struct Name<Param, ...>;`, or either with
/// attribute fields in place of the `;`: `struct Name<Param> { attr: Attr<K> }`.
fn parse_struct_def(iter: &mut TokenIter) -> Result<TypeDefData, String> {
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

    let attrs = match iter.clone().next() {
        Some(TokenTree::Group(body)) if body.delimiter() == Delimiter::Brace => {
            iter.next();
            parse_type_fields(body.stream())?
        }
        Some(TokenTree::Group(body)) if body.delimiter() == Delimiter::Parenthesis => {
            return Err("a type declares its attributes as named fields in `{ .. }`".into());
        }
        _ => {
            expect_punct(iter, ';')?;
            Vec::new()
        }
    };

    Ok(TypeDefData {
        name,
        raw_ident: name_ident,
        params,
        attrs,
        verify: None,
        asm_format: None,
        alias_hint: None,
        implements: Vec::new(),
        docs: Vec::new(),
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
            RegionOrSuccessor::Region { name, optional: false, variadic: false } if name == "body"
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
    fn test_parse_variadic_regions() {
        let module = parse_test_module(quote! {
            mod test {
                fn lambda() {
                    #[region(header)] {}
                    #[regions(bodies)] {}
                }
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        assert!(matches!(
            &op.regions[1],
            RegionOrSuccessor::Region { name, optional: false, variadic: true } if name == "bodies"
        ));

        let not_last = parse_test_module(quote! {
            mod test {
                fn lambda() {
                    #[regions(bodies)] {}
                    #[region(tail)] {}
                }
            }
        });
        assert!(
            not_last
                .err()
                .unwrap()
                .contains("variadic regions must be the last regions")
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
            RegionOrSuccessor::Region { name, optional: false, variadic: false } if name == "first"
        ));
        assert!(matches!(
            &op.regions[1],
            RegionOrSuccessor::Region { name, optional: true, variadic: false } if name == "body"
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
                struct Ref<Pointee> {
                    /// Whether the reference may be null.
                    nullable: Attr<bool>,
                    tags: Option<Attr<[String]>>,
                }
            }
        })
        .unwrap();

        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.name, "Ref");
                assert_eq!(td.params.len(), 1);
                assert_eq!(td.attrs.len(), 2);
                assert_eq!(td.attrs[0].name, "nullable");
                assert!(matches!(&td.attrs[0].kind, AttrKind::Path(path) if path.is_ident("bool")));
                assert!(!td.attrs[0].optional && !td.attrs[0].list);
                assert_eq!(td.attrs[0].docs.len(), 1);
                assert_eq!(td.attrs[1].name, "tags");
                assert!(td.attrs[1].optional && td.attrs[1].list);
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn test_parse_struct_rejects_malformed_fields() {
        for (item, expected) in [
            (
                quote!(
                    mod t {
                        #[attr(nullable: bool)]
                        struct Ref<Pointee>;
                    }
                ),
                "`#[attr(..)]` is not supported on types; declare `Attr<..>` fields",
            ),
            (
                quote!(
                    mod t {
                        struct Ref<Pointee> {
                            pub nullable: Attr<bool>,
                        }
                    }
                ),
                "take no visibility",
            ),
            (
                quote!(
                    mod t {
                        struct Ref<Pointee> {
                            nullable: bool,
                        }
                    }
                ),
                "must be `Attr<..>` or `Option<Attr<..>>`",
            ),
            (
                quote!(
                    mod t {
                        struct Ref<Pointee> {
                            nullable: Value<_>,
                        }
                    }
                ),
                "must be `Attr<..>` or `Option<Attr<..>>`",
            ),
            (
                quote!(
                    mod t {
                        struct Ref<Pointee> {
                            a: Attr<bool>,
                            a: Attr<bool>,
                        }
                    }
                ),
                "duplicate attribute name: `a`",
            ),
            (
                quote!(
                    mod t {
                        struct Ref<Pointee> {
                            r#type: Attr<TypeOf<Pointee>>,
                        }
                    }
                ),
                "declares no type variables",
            ),
            (
                quote!(
                    mod t {
                        struct Ref(bool);
                    }
                ),
                "named fields",
            ),
            (
                quote!(
                    mod t {
                        struct Ref<Pointee> {
                            pointee: Attr<bool>,
                        }
                    }
                ),
                "two accessors of the type are named `pointee`",
            ),
            (
                quote!(
                    mod t {
                        struct Ref {
                            name: Attr<String>,
                            name_ref: Attr<bool>,
                        }
                    }
                ),
                "two accessors of the type are named `name_ref`",
            ),
            (
                quote!(
                    mod t {
                        struct Ref<Params>;
                    }
                ),
                "`params` is reserved",
            ),
            (
                quote!(
                    mod t {
                        struct Ref {
                            intern: Attr<bool>,
                        }
                    }
                ),
                "`intern` is reserved",
            ),
            (
                quote!(
                    mod t {
                        struct Ref {
                            #[allow(unused)]
                            a: Attr<bool>,
                        }
                    }
                ),
                "only doc comments",
            ),
        ] {
            let Err(err) = parse_test_module(item) else {
                panic!("expected `{expected}`");
            };
            assert!(err.contains(expected), "expected `{expected}`, got: {err}");
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
                        struct Ty<Verify>;
                    }
                ),
                "#[verify] reserves the name `verify`",
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
    fn test_implements_and_flags_list_paths() {
        let module = parse_test_module(quote! {
            mod test {
                #[implements(Branch, crate::op_interface::CallLike,)]
                #[flags(Pure)]
                fn call() {}
                fn plain() {}
            }
        })
        .unwrap();
        let paths = |entries: &[PathEntry]| -> Vec<String> {
            entries.iter().map(|entry| entry.path.to_string()).collect()
        };
        let lists: Vec<(Vec<String>, Vec<String>)> = module
            .items
            .iter()
            .map(|item| match item {
                DialectItem::Operation(op) => (paths(&op.implements), paths(&op.flags)),
                DialectItem::TypeDef(_) => unreachable!(),
            })
            .collect();
        assert_eq!(
            lists,
            [
                (
                    vec![
                        "Branch".to_string(),
                        "crate :: op_interface :: CallLike".into()
                    ],
                    vec!["Pure".to_string()],
                ),
                (vec![], vec![]),
            ]
        );

        let out = crate::codegen::generate(&quote!(::trunk_ir), &module).to_string();
        for path in ["Branch", "crate :: op_interface :: CallLike"] {
            let submit = format!(
                ":: trunk_ir :: __private :: inventory :: submit ! {{ < Call as {path} > :: __TRUNK_IR_INTERFACE }}"
            );
            assert!(out.contains(&submit), "missing `{submit}` in: {out}");
        }
        assert_eq!(out.matches("__TRUNK_IR_INTERFACE").count(), 2, "got: {out}");
        assert!(
            out.contains("flags : & [< Pure as :: trunk_ir :: op_flag :: OpFlag > :: DEF]"),
            "got: {out}"
        );
        assert!(out.contains("flags : & []"), "got: {out}");

        for (item, expected) in [
            (
                quote!(
                    mod test {
                        #[implements(Branch)]
                        #[implements(CallLike)]
                        fn op() {}
                    }
                ),
                "duplicate #[implements]",
            ),
            (
                quote!(
                    mod test {
                        #[flags(Pure)]
                        #[flags(IsolatedFromAbove)]
                        fn op() {}
                    }
                ),
                "duplicate #[flags]",
            ),
            (
                quote!(
                    mod test {
                        #[flags(Pure, Pure)]
                        fn op() {}
                    }
                ),
                "listed twice in #[flags(..)]",
            ),
            (
                quote!(
                    mod test {
                        #[implements()]
                        fn op() {}
                    }
                ),
                "#[implements(..)] lists nothing",
            ),
            (
                quote!(
                    mod test {
                        #[implements(Branch,, CallLike)]
                        fn op() {}
                    }
                ),
                "empty entry",
            ),
            (
                quote!(
                    mod test {
                        #[implements]
                        fn op() {}
                    }
                ),
                "expected `#[implements(Interface, ..)]`",
            ),
            (
                quote!(
                    mod test {
                        #[flags]
                        fn op() {}
                    }
                ),
                "expected `#[flags(Flag, ..)]`",
            ),
            (
                quote!(
                    mod test {
                        #[alias_hint]
                        #[alias_hint]
                        struct Ty;
                    }
                ),
                "duplicate #[alias_hint]",
            ),
            (
                quote!(
                    mod test {
                        #[alias_hint(x)]
                        struct Ty;
                    }
                ),
                "#[alias_hint]",
            ),
            (
                quote!(
                    mod test {
                        #[alias_hint]
                        fn op() {}
                    }
                ),
                "#[alias_hint] suggests type alias names and is not allowed on operations",
            ),
            (
                quote!(
                    mod test {
                        #[alias_hint]
                        struct Ty {
                            alias_hint: Attr<String>,
                        }
                    }
                ),
                "#[alias_hint] reserves the name `alias_hint`",
            ),
            (
                quote!(
                    mod test {
                        #[flags(Pure)]
                        struct Ty;
                    }
                ),
                "#[flags] declares operation flags and is not allowed on types",
            ),
        ] {
            let err = parse_test_module(item).err().expect("should fail");
            assert!(err.contains(expected), "unexpected error: {err}");
        }
    }

    #[test]
    fn test_hooks_mark_types() {
        let module = parse_test_module(quote! {
            mod test {
                #[verify]
                struct Checked;
                #[asm_format]
                struct Custom<Elem>;
                #[alias_hint]
                #[implements(Managed, crate::Sized)]
                struct Named;
                struct Plain;
            }
        })
        .unwrap();
        let flags: Vec<(bool, bool, bool, usize)> = module
            .items
            .iter()
            .map(|item| match item {
                DialectItem::TypeDef(td) => (
                    td.verify.is_some(),
                    td.asm_format.is_some(),
                    td.alias_hint.is_some(),
                    td.implements.len(),
                ),
                DialectItem::Operation(_) => unreachable!(),
            })
            .collect();
        assert_eq!(
            flags,
            [
                (true, false, false, 0),
                (false, true, false, 0),
                (false, false, true, 2),
                (false, false, false, 0),
            ]
        );

        let out = crate::codegen::generate(&quote!(::trunk_ir), &module).to_string();
        for path in ["Managed", "crate :: Sized"] {
            let submit = format!(
                ":: trunk_ir :: __private :: inventory :: submit ! {{ < Named as {path} > :: __TRUNK_IR_INTERFACE }}"
            );
            assert!(out.contains(&submit), "missing `{submit}` in: {out}");
        }
        assert!(
            out.contains("alias_hint : Some (< Named as :: trunk_ir :: ops :: __private :: DeclaredTypeAliasHint > :: ALIAS_HINT)"),
            "got: {out}"
        );
    }

    #[test]
    fn test_asm_format_marks_operations() {
        let module = parse_test_module(quote! {
            mod test {
                #[asm_format]
                #[verify]
                fn custom(x: Value<_>) {}
                fn plain(xs: Variadic<_>) {}
            }
        })
        .unwrap();
        let flags: Vec<(bool, bool)> = module
            .items
            .iter()
            .map(|item| match item {
                DialectItem::Operation(op) => (op.asm_format.is_some(), op.verify.is_some()),
                DialectItem::TypeDef(_) => unreachable!(),
            })
            .collect();
        assert_eq!(flags, [(true, true), (false, false)]);

        for (item, expected) in [
            (
                quote!(
                    mod test {
                        #[asm_format]
                        #[asm_format]
                        fn op() {}
                    }
                ),
                "duplicate #[asm_format]",
            ),
            (
                quote!(
                    mod test {
                        #[asm_format]
                        struct Ty {
                            parse: Attr<String>,
                        }
                    }
                ),
                "#[asm_format] reserves the name `parse`",
            ),
            (
                quote!(
                    mod test {
                        #[asm_format("custom")]
                        struct Ty;
                    }
                ),
                "hand-written `TypeAsmFormat`",
            ),
            (
                quote!(
                    mod test {
                        #[asm_format]
                        fn op(print: Value<_>) {}
                    }
                ),
                "reserves the name `print`",
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
                struct Func<Return, #[rest] Inputs>;
            }
        })
        .unwrap();

        match &module.items[0] {
            DialectItem::TypeDef(td) => {
                assert_eq!(td.params.len(), 2);
                assert!(!td.params[0].variadic);
                assert_eq!(td.params[0].raw_ident.to_string(), "Return");
                assert!(td.params[1].variadic);
                assert_eq!(td.params[1].raw_ident.to_string(), "Inputs");
            }
            _ => panic!("expected TypeDef"),
        }
    }
}
