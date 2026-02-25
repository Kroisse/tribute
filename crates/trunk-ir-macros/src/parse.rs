//! DSL parser for `#[arena_dialect]`.
//!
//! Parses the module body into structured types for code generation.

use proc_macro2::{Delimiter, Ident, TokenTree};
use unsynn::{Parser, ToTokenIter, TokenIter};

// ============================================================================
// Parsed types
// ============================================================================

pub struct DialectModule {
    pub name: String,
    pub items: Vec<DialectItem>,
}

pub enum DialectItem {
    Operation(OperationDef),
    /// Type definitions are parsed but ignored for codegen (matching current behavior).
    TypeDef,
}

pub struct OperationDef {
    /// Clean name without `r#` prefix (e.g., "return")
    pub name: String,
    /// Original ident for use in generated code (e.g., `r#return`)
    pub raw_ident: Ident,
    pub attrs: Vec<AttrDef>,
    pub operands: Vec<Operand>,
    pub results: ResultDef,
    pub regions: Vec<RegionOrSuccessor>,
}

pub struct AttrDef {
    /// Clean name without `r#` prefix
    pub name: String,
    /// Original ident
    pub raw_ident: Ident,
    pub ty: AttrType,
    pub optional: bool,
}

#[derive(Clone, Copy)]
pub enum AttrType {
    Any,
    Bool,
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
    Type,
    String,
    Symbol,
    QualifiedName,
}

pub struct Operand {
    /// Clean name without `r#` prefix (used in tests/diagnostics)
    #[allow(dead_code)]
    pub name: String,
    /// Original ident
    pub raw_ident: Ident,
    pub variadic: bool,
}

pub enum ResultDef {
    None,
    Single(String),
    Multi(Vec<String>),
    Variadic(String),
}

pub enum RegionOrSuccessor {
    Region(String),
    Successor(String),
}

// ============================================================================
// Top-level input parsing
// ============================================================================

/// Parse attribute macro input: `attr` contains optional `crate = path`,
/// `item` contains `mod name { ... }`.
pub fn parse_input(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> Result<(proc_macro2::TokenStream, DialectModule), String> {
    let crate_path = parse_crate_attr(attr)?;
    let module = parse_module(item)?;
    Ok((crate_path, module))
}

/// Parse optional `crate = path` from the attribute arguments.
fn parse_crate_attr(stream: proc_macro2::TokenStream) -> Result<proc_macro2::TokenStream, String> {
    let mut iter = stream.to_token_iter();

    if !has_remaining(&iter) {
        return Ok(quote::quote!(trunk_ir));
    }

    let kw: Ident =
        Ident::parser(&mut iter).map_err(|e| format!("expected `crate` keyword: {e}"))?;
    if kw != "crate" {
        return Err(format!("expected `crate`, got `{kw}`"));
    }

    expect_punct(&mut iter, '=')?;

    // Collect the remaining tokens as the crate path
    let mut path_tokens = proc_macro2::TokenStream::new();
    while has_remaining(&iter) {
        let tt: TokenTree =
            TokenTree::parser(&mut iter).map_err(|e| format!("error parsing crate path: {e}"))?;
        path_tokens.extend(std::iter::once(tt));
    }

    if path_tokens.is_empty() {
        return Err("expected crate path after `=`".into());
    }

    Ok(path_tokens)
}

/// Parse `mod name { ... }` from the item token stream.
fn parse_module(stream: proc_macro2::TokenStream) -> Result<DialectModule, String> {
    let mut iter = stream.to_token_iter();
    parse_module_inner(&mut iter)
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
    while has_remaining(&body_iter) {
        items.push(parse_item(&mut body_iter)?);
    }

    Ok(DialectModule { name, items })
}

// ============================================================================
// Item parsing
// ============================================================================

fn parse_item(iter: &mut TokenIter) -> Result<DialectItem, String> {
    let mut op_attrs = Vec::new();
    let mut rest_results = false;

    // Collect outer attributes: #[doc = "..."] (skip), #[attr(...)], #[rest_results]
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
            OuterAttr::RestResults => rest_results = true,
        }
    }

    // Parse keyword: fn or type
    let kw: Ident = Ident::parser(iter).map_err(|e| format!("expected `fn` or `type`: {e}"))?;

    match kw.to_string().as_str() {
        "fn" => {
            let op = parse_operation(iter, op_attrs, rest_results)?;
            Ok(DialectItem::Operation(op))
        }
        "type" => {
            parse_type_def(iter)?;
            Ok(DialectItem::TypeDef)
        }
        other => Err(format!("expected `fn` or `type`, got `{other}`")),
    }
}

enum OuterAttr {
    Doc,
    OpAttrs(Vec<AttrDef>),
    RestResults,
}

/// Parse `#[...]` — either `#[doc = "..."]`, `#[attr(...)]`, or `#[rest_results]`.
fn parse_outer_attr(iter: &mut TokenIter) -> Result<OuterAttr, String> {
    expect_punct(iter, '#')?;
    let bracket = expect_group(iter, Delimiter::Bracket)?;
    let mut inner = bracket.stream().to_token_iter();

    let ident: Ident =
        Ident::parser(&mut inner).map_err(|e| format!("expected attribute name: {e}"))?;

    match ident.to_string().as_str() {
        "doc" => Ok(OuterAttr::Doc),
        "attr" => {
            let paren = expect_group(&mut inner, Delimiter::Parenthesis)?;
            let attrs = parse_attr_list(paren.stream())?;
            Ok(OuterAttr::OpAttrs(attrs))
        }
        "rest_results" => Ok(OuterAttr::RestResults),
        other => Err(format!(
            "unexpected attribute `{other}`, expected `doc`, `attr`, or `rest_results`"
        )),
    }
}

/// Parse comma-separated attribute definitions: `name: Type, opt?: Type`.
fn parse_attr_list(stream: proc_macro2::TokenStream) -> Result<Vec<AttrDef>, String> {
    let mut iter = stream.to_token_iter();
    let mut attrs = Vec::new();

    while has_remaining(&iter) {
        let name_ident: Ident =
            Ident::parser(&mut iter).map_err(|e| format!("expected attribute name: {e}"))?;

        // Check for optional marker '?'
        let optional = peek_punct(&iter, '?');
        if optional {
            consume_punct(&mut iter)?;
        }

        expect_punct(&mut iter, ':')?;

        let ty_ident: Ident =
            Ident::parser(&mut iter).map_err(|e| format!("expected attribute type: {e}"))?;
        let ty = parse_attr_type(&ty_ident)?;

        attrs.push(AttrDef {
            name: ident_str(&name_ident),
            raw_ident: name_ident,
            ty,
            optional,
        });

        // Consume optional comma
        if peek_punct(&iter, ',') {
            consume_punct(&mut iter)?;
        }
    }

    Ok(attrs)
}

fn parse_attr_type(ident: &Ident) -> Result<AttrType, String> {
    match ident.to_string().as_str() {
        "any" => Ok(AttrType::Any),
        "bool" => Ok(AttrType::Bool),
        "i32" => Ok(AttrType::I32),
        "i64" => Ok(AttrType::I64),
        "u32" => Ok(AttrType::U32),
        "u64" => Ok(AttrType::U64),
        "f32" => Ok(AttrType::F32),
        "f64" => Ok(AttrType::F64),
        "Type" => Ok(AttrType::Type),
        "String" => Ok(AttrType::String),
        "Symbol" => Ok(AttrType::Symbol),
        "QualifiedName" => Ok(AttrType::QualifiedName),
        other => Err(format!("unknown attribute type `{other}`")),
    }
}

// ============================================================================
// Operation parsing
// ============================================================================

fn parse_operation(
    iter: &mut TokenIter,
    attrs: Vec<AttrDef>,
    rest_results: bool,
) -> Result<OperationDef, String> {
    // Parse operation name
    let name_ident: Ident =
        Ident::parser(iter).map_err(|e| format!("expected operation name: {e}"))?;

    // Parse operands (with `: type` annotations)
    let paren = expect_group(iter, Delimiter::Parenthesis)?;
    let operands = parse_operands(paren.stream())?;

    // Parse optional result: `-> result` or `-> (a, b)`
    let results = if peek_punct(iter, '-') {
        let r = parse_results(iter)?;
        // Convert single result to variadic if #[rest_results] was present
        if rest_results {
            match r {
                ResultDef::Single(name) => ResultDef::Variadic(name),
                other => other,
            }
        } else {
            r
        }
    } else {
        ResultDef::None
    };

    // Parse body `{ ... }` — always present, may contain regions/successors
    let body = expect_group(iter, Delimiter::Brace)?;
    let regions = parse_regions(body.stream())?;

    Ok(OperationDef {
        name: ident_str(&name_ident),
        raw_ident: name_ident,
        attrs,
        operands,
        results,
        regions,
    })
}

/// Parse operand list: `a: (), b: (), #[rest] c: ()`.
fn parse_operands(stream: proc_macro2::TokenStream) -> Result<Vec<Operand>, String> {
    let mut iter = stream.to_token_iter();
    let mut operands = Vec::new();

    let mut seen_variadic = false;

    while has_remaining(&iter) {
        // Check for #[rest] marker
        let variadic = if peek_punct(&iter, '#') {
            consume_punct(&mut iter)?;
            let bracket = expect_group(&mut iter, Delimiter::Bracket)?;
            let mut inner = bracket.stream().to_token_iter();
            let kw: Ident =
                Ident::parser(&mut inner).map_err(|e| format!("expected `rest`: {e}"))?;
            if kw != "rest" {
                return Err(format!("expected `rest`, got `{kw}`"));
            }
            if seen_variadic {
                return Err("at most one #[rest] operand is allowed".into());
            }
            seen_variadic = true;
            true
        } else {
            if seen_variadic {
                return Err("#[rest] operand must be the last operand".into());
            }
            false
        };

        let name_ident: Ident =
            Ident::parser(&mut iter).map_err(|e| format!("expected operand name: {e}"))?;

        // Consume `: type` annotation (type is ignored, only name matters)
        expect_punct(&mut iter, ':')?;
        skip_type(&mut iter)?;

        operands.push(Operand {
            name: ident_str(&name_ident),
            raw_ident: name_ident,
            variadic,
        });

        // Consume optional comma
        if peek_punct(&iter, ',') {
            consume_punct(&mut iter)?;
        }
    }

    Ok(operands)
}

/// Parse result definition after `->`: single `result`, multi `(a, b)`.
///
/// Variadic results are indicated by the `#[rest_results]` outer attribute
/// on the function, not by `-> #[rest] results` (which is not valid Rust).
fn parse_results(iter: &mut TokenIter) -> Result<ResultDef, String> {
    // Consume `->`
    expect_punct(iter, '-')?;
    expect_punct(iter, '>')?;

    // Check for `(a, b)` (multi)
    if peek_group(iter, Delimiter::Parenthesis) {
        let paren = expect_group(iter, Delimiter::Parenthesis)?;
        let mut inner = paren.stream().to_token_iter();
        let mut names = Vec::new();
        while has_remaining(&inner) {
            let ident: Ident =
                Ident::parser(&mut inner).map_err(|e| format!("expected result name: {e}"))?;
            names.push(ident_str(&ident));
            if peek_punct(&inner, ',') {
                consume_punct(&mut inner)?;
            }
        }
        return Ok(ResultDef::Multi(names));
    }

    // Single result
    let name_ident: Ident =
        Ident::parser(iter).map_err(|e| format!("expected result name: {e}"))?;
    Ok(ResultDef::Single(ident_str(&name_ident)))
}

/// Parse body content: `#[region(name)] {}` and `#[successor(name)] {}`.
fn parse_regions(stream: proc_macro2::TokenStream) -> Result<Vec<RegionOrSuccessor>, String> {
    let mut iter = stream.to_token_iter();
    let mut items = Vec::new();

    while has_remaining(&iter) {
        expect_punct(&mut iter, '#')?;
        let bracket = expect_group(&mut iter, Delimiter::Bracket)?;
        let mut inner = bracket.stream().to_token_iter();

        let kw: Ident = Ident::parser(&mut inner)
            .map_err(|e| format!("expected `region` or `successor`: {e}"))?;

        let paren = expect_group(&mut inner, Delimiter::Parenthesis)?;
        let mut name_iter = paren.stream().to_token_iter();
        let name_ident: Ident = Ident::parser(&mut name_iter)
            .map_err(|e| format!("expected region/successor name: {e}"))?;
        let name = ident_str(&name_ident);

        match kw.to_string().as_str() {
            "region" => {
                let _body = expect_group(&mut iter, Delimiter::Brace)?;
                items.push(RegionOrSuccessor::Region(name));
            }
            "successor" => {
                // Consume `{}` after the attribute (required for valid Rust syntax)
                let _body = expect_group(&mut iter, Delimiter::Brace)?;
                items.push(RegionOrSuccessor::Successor(name));
            }
            other => return Err(format!("expected `region` or `successor`, got `{other}`")),
        }
    }

    Ok(items)
}

// ============================================================================
// Type definition parsing (parsed but no codegen)
// ============================================================================

fn parse_type_def(iter: &mut TokenIter) -> Result<(), String> {
    // Parse type name
    let _name: Ident = Ident::parser(iter).map_err(|e| format!("expected type name: {e}"))?;

    // Optional params: (param1, param2)
    if peek_group(iter, Delimiter::Parenthesis) {
        let _paren = expect_group(iter, Delimiter::Parenthesis)?;
    }

    // Optional `with #[attr(...)]` - consume until semicolon
    // For simplicity, skip any remaining tokens until `;`
    while !peek_punct(iter, ';') {
        if !has_remaining(iter) {
            return Err("expected `;` after type definition".into());
        }
        let _: TokenTree =
            TokenTree::parser(iter).map_err(|e| format!("error in type definition: {e}"))?;
    }

    expect_punct(iter, ';')?;
    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

/// Skip a single type token tree (e.g., `()`, `Ident`, or path).
fn skip_type(iter: &mut TokenIter) -> Result<(), String> {
    let _: TokenTree =
        TokenTree::parser(iter).map_err(|e| format!("expected type annotation: {e}"))?;
    Ok(())
}

/// Strip `r#` prefix from an ident.
fn ident_str(ident: &Ident) -> String {
    let s = ident.to_string();
    s.strip_prefix("r#").unwrap_or(&s).to_string()
}

fn peek_punct(iter: &TokenIter, ch: char) -> bool {
    matches!(iter.clone().next(), Some(TokenTree::Punct(p)) if p.as_char() == ch)
}

fn peek_group(iter: &TokenIter, delim: Delimiter) -> bool {
    matches!(iter.clone().next(), Some(TokenTree::Group(g)) if g.delimiter() == delim)
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
        let (_, module) = parse_input(quote! {}, item)?;
        Ok(module)
    }

    #[test]
    fn test_parse_default_crate_path() {
        let (path, module) = parse_input(
            quote! {},
            quote! {
                mod arith {
                    fn add(lhs: (), rhs: ()) -> result {}
                }
            },
        )
        .unwrap();

        assert_eq!(path.to_string(), "trunk_ir");
        assert_eq!(module.name, "arith");
    }

    #[test]
    fn test_parse_crate_path_crate() {
        let (path, _) = parse_input(
            quote! { crate = crate },
            quote! {
                mod arith {
                    fn add(lhs: (), rhs: ()) -> result {}
                }
            },
        )
        .unwrap();

        assert_eq!(path.to_string(), "crate");
    }

    #[test]
    fn test_parse_crate_path_custom() {
        let (path, _) = parse_input(
            quote! { crate = my_crate::ir },
            quote! {
                mod arith {
                    fn add(lhs: (), rhs: ()) -> result {}
                }
            },
        )
        .unwrap();

        assert_eq!(path.to_string(), "my_crate :: ir");
    }

    #[test]
    fn test_parse_simple_module() {
        let module = parse_test_module(quote! {
            mod arith {
                fn add(lhs: (), rhs: ()) -> result {}
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
                fn call(#[rest] args: ()) -> result {}
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
                fn call_indirect(callee: (), #[rest] args: ()) -> result {}
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
                #[attr(r#type: Type, field: u32)]
                fn struct_get(r#ref: ()) -> result {}
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
        assert!(matches!(op.attrs[0].ty, AttrType::Type));
        assert_eq!(op.attrs[1].name, "field");
        assert!(matches!(op.attrs[1].ty, AttrType::U32));
    }

    #[test]
    fn test_parse_optional_attributes() {
        let module = parse_test_module(quote! {
            mod wasm {
                #[attr(reftype: Symbol, min: u32, max?: u32)]
                fn table() {}
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
                #[attr(sym_name: Symbol)]
                fn func() {
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
        assert!(matches!(&op.regions[0], RegionOrSuccessor::Region(s) if s == "body"));
    }

    #[test]
    fn test_parse_successors() {
        let module = parse_test_module(quote! {
            mod cf {
                fn cond_br(cond: ()) {
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
        assert!(matches!(&op.regions[0], RegionOrSuccessor::Successor(s) if s == "then_dest"));
        assert!(matches!(&op.regions[1], RegionOrSuccessor::Successor(s) if s == "else_dest"));
    }

    #[test]
    fn test_parse_raw_identifiers() {
        let module = parse_test_module(quote! {
            mod scf {
                fn r#return(#[rest] values: ()) {}
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
                #[rest_results]
                fn call(#[rest] args: ()) -> results {}
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
    fn test_parse_multi_results() {
        let module = parse_test_module(quote! {
            mod test {
                fn multi() -> (a, b) {}
            }
        })
        .unwrap();

        let op = match &module.items[0] {
            DialectItem::Operation(op) => op,
            _ => panic!("expected operation"),
        };
        match &op.results {
            ResultDef::Multi(names) => {
                assert_eq!(names, &["a", "b"]);
            }
            _ => panic!("expected multi result"),
        }
    }

    #[test]
    fn test_parse_type_def_skipped() {
        let module = parse_test_module(quote! {
            mod core {
                type i32 = ();
                fn add(lhs: (), rhs: ()) -> result {}
            }
        })
        .unwrap();

        assert_eq!(module.items.len(), 2);
        assert!(matches!(module.items[0], DialectItem::TypeDef));
        assert!(matches!(module.items[1], DialectItem::Operation(_)));
    }

    #[test]
    fn test_parse_multiple_operations() {
        let module = parse_test_module(quote! {
            mod arith {
                fn add(lhs: (), rhs: ()) -> result {}
                fn sub(lhs: (), rhs: ()) -> result {}
                fn neg(operand: ()) -> result {}
            }
        })
        .unwrap();

        assert_eq!(module.items.len(), 3);
    }

    #[test]
    fn test_duplicate_attr_rejected() {
        let result = parse_test_module(quote! {
            mod test {
                #[attr(a: u32)]
                #[attr(b: u32)]
                fn op() {}
            }
        });
        let err = result.err().expect("should fail");
        assert!(err.contains("duplicate #[attr("), "unexpected error: {err}");
    }

    #[test]
    fn test_rest_must_be_last_operand() {
        let result = parse_test_module(quote! {
            mod test {
                fn op(#[rest] a: (), b: ()) {}
            }
        });
        let err = result.err().expect("should fail");
        assert!(
            err.contains("must be the last operand"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_multiple_rest_rejected() {
        let result = parse_test_module(quote! {
            mod test {
                fn op(#[rest] a: (), #[rest] b: ()) {}
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_region_with_result() {
        let module = parse_test_module(quote! {
            mod scf {
                fn r#if(cond: ()) -> result {
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
}
