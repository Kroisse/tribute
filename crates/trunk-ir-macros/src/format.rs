//! `#[asm_format(..)]`: parse an operation's format elements, check them
//! against its declaration, and emit the `OpFormat` table the printer and
//! parser interpret.
//!
//! The rules are those of `new-plans/ir.md`, "선언적 어셈블리 형식". A rule
//! the format breaks is reported at the element that breaks it.

use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use quote::{quote, quote_spanned};

use crate::codegen::attr_kind_trait;
use crate::parse::{AttrKind, OperationDef, RegionOrSuccessor, ResultDef};

type Error = (Span, String);

/// The `OpAsmFormatDef` expression for `op`'s format elements, or a
/// `compile_error!` at the element that breaks a rule.
pub fn gen_format(
    crate_path: &TokenStream,
    op: &OperationDef,
    attr_span: Span,
    tokens: &TokenStream,
) -> TokenStream {
    match check(crate_path, op, attr_span, tokens) {
        Ok(expr) => expr,
        Err((span, message)) => quote_spanned!(span=> {
            ::core::compile_error!(#message);
            None
        }),
    }
}

// ============================================================================
// Syntax
// ============================================================================

/// One element as written, before it is resolved against the declaration.
enum Item {
    Literal {
        text: String,
        span: Span,
    },
    /// An entity, or a directive `name(args..)`; `anchor` when marked `^`.
    Named {
        name: String,
        args: Option<Vec<(String, Span)>>,
        span: Span,
        anchor: bool,
    },
    Optional {
        items: Vec<Item>,
        span: Span,
    },
}

fn strip_raw(name: String) -> String {
    match name.strip_prefix("r#") {
        Some(name) => name.to_owned(),
        None => name,
    }
}

fn parse_items(tokens: TokenStream) -> Result<Vec<Item>, Error> {
    let tokens: Vec<TokenTree> = tokens.into_iter().collect();
    let mut tokens = tokens.into_iter().peekable();
    let mut items = Vec::new();
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Literal(literal) => {
                let raw = literal.to_string();
                let text = raw
                    .strip_prefix('"')
                    .and_then(|rest| rest.strip_suffix('"'))
                    .filter(|text| !text.contains(['\\', '"']))
                    .ok_or_else(|| {
                        (
                            literal.span(),
                            "a format literal is a plain string literal, such as `\"(\"`"
                                .to_owned(),
                        )
                    })?;
                if text.chars().any(char::is_whitespace) {
                    return Err((
                        literal.span(),
                        "a format literal is one token; write `\"a\" \"b\"` for two".to_owned(),
                    ));
                }
                items.push(Item::Literal {
                    text: text.to_owned(),
                    span: literal.span(),
                });
            }
            TokenTree::Ident(ident) => {
                let span = ident.span();
                let name = strip_raw(ident.to_string());
                // `name(..)` is a directive; `name (..)?` is an entity
                // followed by an optional group.
                let followed_by_group = {
                    let mut ahead = tokens.clone();
                    matches!(ahead.next(), Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Parenthesis)
                        && !matches!(ahead.next(), Some(TokenTree::Punct(p)) if p.as_char() == '?')
                };
                let args = match tokens.peek() {
                    Some(TokenTree::Group(group))
                        if followed_by_group && group.delimiter() == Delimiter::Parenthesis =>
                    {
                        let stream = group.stream();
                        tokens.next();
                        let mut args = Vec::new();
                        for token in stream {
                            match token {
                                TokenTree::Ident(arg) => {
                                    args.push((strip_raw(arg.to_string()), arg.span()))
                                }
                                TokenTree::Punct(p) if p.as_char() == ',' => {}
                                other => {
                                    return Err((
                                        other.span(),
                                        format!("expected an entity name in `{name}(..)`"),
                                    ));
                                }
                            }
                        }
                        Some(args)
                    }
                    _ => None,
                };
                let anchor =
                    matches!(tokens.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '^');
                if anchor {
                    tokens.next();
                }
                items.push(Item::Named {
                    name,
                    args,
                    span,
                    anchor,
                });
            }
            TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis => {
                match tokens.next() {
                    Some(TokenTree::Punct(p)) if p.as_char() == '?' => {}
                    _ => {
                        return Err((
                            group.span(),
                            "a parenthesized group is an optional group `( .. )?`; write punctuation as a string literal".to_owned(),
                        ));
                    }
                }
                items.push(Item::Optional {
                    items: parse_items(group.stream())?,
                    span: group.span(),
                });
            }
            other => {
                return Err((
                    other.span(),
                    "write punctuation as a string literal, such as `\":\"`".to_owned(),
                ));
            }
        }
    }
    Ok(items)
}

// ============================================================================
// Resolution
// ============================================================================

/// One element resolved against the declaration.
enum Element {
    Literal(String),
    Operand(usize),
    Attribute(usize),
    Symbol(usize),
    ResultType(usize),
    ResultTypes,
    AttrDict {
        keyword: bool,
    },
    Region(usize),
    Successor(usize),
    Optional {
        anchor: Anchor,
        elements: Vec<(Element, Span)>,
    },
}

#[derive(Clone, Copy)]
enum Anchor {
    Attribute(usize),
    Flag(usize),
    Operand(usize),
    Results,
    Region(usize),
}

/// The regions and successors of an operation, each by its position in its
/// own list.
struct Bodies<'a> {
    regions: Vec<(&'a str, bool, bool)>,
    successors: Vec<(&'a str, bool)>,
}

impl<'a> Bodies<'a> {
    fn of(op: &'a OperationDef) -> Self {
        let mut regions = Vec::new();
        let mut successors = Vec::new();
        for item in &op.regions {
            match item {
                RegionOrSuccessor::Region {
                    name,
                    optional,
                    variadic,
                } => regions.push((name.as_str(), *optional, *variadic)),
                RegionOrSuccessor::Successor { name, variadic } => {
                    successors.push((name.as_str(), *variadic))
                }
            }
        }
        Self {
            regions,
            successors,
        }
    }
}

/// What the format has written so far, to report what is written twice or
/// left out.
#[derive(Default)]
struct Written {
    operands: Vec<bool>,
    attributes: Vec<bool>,
    results: Vec<bool>,
    all_results: bool,
    regions: Vec<bool>,
    successors: Vec<bool>,
    attr_dict: bool,
}

struct Resolver<'a> {
    op: &'a OperationDef,
    bodies: Bodies<'a>,
    written: Written,
}

fn mark(slot: &mut bool, span: Span, what: &str) -> Result<(), Error> {
    if std::mem::replace(slot, true) {
        return Err((span, format!("the format writes {what} twice")));
    }
    Ok(())
}

impl Resolver<'_> {
    fn result_name(&self) -> Option<&str> {
        match &self.op.results {
            ResultDef::None => None,
            ResultDef::Single(name) | ResultDef::Variadic(name) | ResultDef::Optional(name) => {
                Some(name)
            }
        }
    }

    fn one_arg<'s>(
        &self,
        name: &str,
        args: &'s [(String, Span)],
        span: Span,
    ) -> Result<&'s (String, Span), Error> {
        match args {
            [arg] => Ok(arg),
            _ => Err((span, format!("`{name}(..)` takes one entity"))),
        }
    }

    fn attribute(&self, name: &str) -> Option<usize> {
        self.op.attrs.iter().position(|attr| attr.name == name)
    }

    fn items(&mut self, items: &[Item], in_group: bool) -> Result<Vec<(Element, Span)>, Error> {
        items.iter().map(|item| self.item(item, in_group)).collect()
    }

    fn item(&mut self, item: &Item, in_group: bool) -> Result<(Element, Span), Error> {
        match item {
            Item::Literal { text, span } => Ok((Element::Literal(text.clone()), *span)),
            Item::Optional { items, span } => {
                if in_group {
                    return Err((*span, "optional groups do not nest".to_owned()));
                }
                self.group(items, *span)
            }
            Item::Named {
                name,
                args,
                span,
                anchor,
            } => {
                if *anchor && !in_group {
                    return Err((
                        *span,
                        "`^` marks the anchor of an optional group".to_owned(),
                    ));
                }
                let element = match args {
                    None => self.entity(name, *span)?,
                    Some(args) => self.directive(name, args, *span)?,
                };
                Ok((element, *span))
            }
        }
    }

    fn entity(&mut self, name: &str, span: Span) -> Result<Element, Error> {
        if let Some(index) = self.op.operands.iter().position(|o| o.name == name) {
            mark(
                &mut self.written.operands[index],
                span,
                &format!("operand `{name}`"),
            )?;
            return Ok(Element::Operand(index));
        }
        if let Some(index) = self.attribute(name) {
            mark(
                &mut self.written.attributes[index],
                span,
                &format!("attribute `{name}`"),
            )?;
            return Ok(Element::Attribute(index));
        }
        if name == "attr_dict" {
            mark(&mut self.written.attr_dict, span, "`attr_dict`")?;
            return Ok(Element::AttrDict { keyword: false });
        }
        let hint = if self.result_name() == Some(name) {
            "a result's type is written with `type(..)` or `types(..)`"
        } else if self.bodies.regions.iter().any(|r| r.0 == name) {
            "a region is written with `region(..)` or `regions(..)`"
        } else if self.bodies.successors.iter().any(|s| s.0 == name) {
            "a successor is written with `successor(..)` or `successors(..)`"
        } else {
            return Err((span, format!("the operation declares no `{name}`")));
        };
        Err((span, hint.to_owned()))
    }

    fn directive(
        &mut self,
        name: &str,
        args: &[(String, Span)],
        span: Span,
    ) -> Result<Element, Error> {
        match name {
            "attr_dict" => {
                let keyword = match args {
                    [] => false,
                    [(arg, _)] if arg == "keyword" => true,
                    _ => {
                        return Err((
                            span,
                            "expected `attr_dict` or `attr_dict(keyword)`".to_owned(),
                        ));
                    }
                };
                mark(&mut self.written.attr_dict, span, "`attr_dict`")?;
                Ok(Element::AttrDict { keyword })
            }
            "symbol" => {
                let (arg, arg_span) = self.one_arg(name, args, span)?;
                let index = self.attribute(arg).ok_or_else(|| {
                    (
                        *arg_span,
                        format!("the operation declares no attribute `{arg}`"),
                    )
                })?;
                let attr = &self.op.attrs[index];
                if !attr.kind.is_string() || attr.list || attr.optional {
                    return Err((
                        *arg_span,
                        "`symbol(..)` writes a required `Attr<String>`".to_owned(),
                    ));
                }
                mark(
                    &mut self.written.attributes[index],
                    *arg_span,
                    &format!("attribute `{arg}`"),
                )?;
                Ok(Element::Symbol(index))
            }
            "type" | "types" => {
                let (arg, arg_span) = self.one_arg(name, args, span)?;
                if self.result_name() != Some(arg.as_str()) {
                    return Err((
                        *arg_span,
                        format!("`{arg}` is not a result of the operation"),
                    ));
                }
                match (&self.op.results, name) {
                    (ResultDef::Single(_), "type") => {
                        mark(
                            &mut self.written.results[0],
                            *arg_span,
                            &format!("the type of `{arg}`"),
                        )?;
                        Ok(Element::ResultType(0))
                    }
                    (ResultDef::Optional(_), "type") | (ResultDef::Variadic(_), "types") => {
                        mark(
                            &mut self.written.all_results,
                            *arg_span,
                            &format!("the type of `{arg}`"),
                        )?;
                        Ok(Element::ResultTypes)
                    }
                    (ResultDef::Variadic(_), _) => {
                        Err((span, format!("`{arg}` is variadic; write `types({arg})`")))
                    }
                    _ => Err((span, format!("`{arg}` is one result; write `type({arg})`"))),
                }
            }
            "region" | "regions" => {
                let (arg, arg_span) = self.one_arg(name, args, span)?;
                let index = self
                    .bodies
                    .regions
                    .iter()
                    .position(|r| r.0 == arg)
                    .ok_or_else(|| {
                        (
                            *arg_span,
                            format!("the operation declares no region `{arg}`"),
                        )
                    })?;
                let variadic = self.bodies.regions[index].2;
                if variadic != (name == "regions") {
                    let expected = if variadic { "regions" } else { "region" };
                    return Err((span, format!("write `{expected}({arg})`")));
                }
                mark(
                    &mut self.written.regions[index],
                    *arg_span,
                    &format!("region `{arg}`"),
                )?;
                Ok(Element::Region(index))
            }
            "successor" | "successors" => {
                let (arg, arg_span) = self.one_arg(name, args, span)?;
                let index = self
                    .bodies
                    .successors
                    .iter()
                    .position(|s| s.0 == arg)
                    .ok_or_else(|| {
                        (
                            *arg_span,
                            format!("the operation declares no successor `{arg}`"),
                        )
                    })?;
                let variadic = self.bodies.successors[index].1;
                if variadic != (name == "successors") {
                    let expected = if variadic { "successors" } else { "successor" };
                    return Err((span, format!("write `{expected}({arg})`")));
                }
                mark(
                    &mut self.written.successors[index],
                    *arg_span,
                    &format!("successor `{arg}`"),
                )?;
                Ok(Element::Successor(index))
            }
            other => Err((
                span,
                format!(
                    "unknown format directive `{other}`; expected `symbol`, `type`, `types`, `attr_dict`, `region`, `regions`, `successor`, or `successors`"
                ),
            )),
        }
    }

    /// An optional group: literals and one entity, its anchor, which may be
    /// absent.
    fn group(&mut self, items: &[Item], span: Span) -> Result<(Element, Span), Error> {
        let elements = self.items(items, true)?;
        let mut entities = items
            .iter()
            .zip(&elements)
            .filter(|(item, _)| !matches!(item, Item::Literal { .. }));
        let (Some((anchor_item, (anchor_element, anchor_span))), None) =
            (entities.next(), entities.next())
        else {
            return Err((
                span,
                "an optional group holds literals and exactly one entity, its anchor".to_owned(),
            ));
        };
        if !matches!(anchor_item, Item::Named { anchor: true, .. }) {
            return Err((
                *anchor_span,
                "mark the group's entity as its anchor with `^`".to_owned(),
            ));
        }
        let first_is_literal_or_anchor = matches!(items.first(), Some(Item::Literal { .. }))
            || std::ptr::eq(items.first().unwrap(), anchor_item);
        if !first_is_literal_or_anchor {
            return Err((
                span,
                "an optional group begins with a literal or its anchor".to_owned(),
            ));
        }
        let anchor = match *anchor_element {
            Element::Attribute(index) => {
                let attr = &self.op.attrs[index];
                if !attr.optional {
                    return Err((
                        *anchor_span,
                        "an attribute anchor is an optional attribute".to_owned(),
                    ));
                }
                if !attr.list && matches!(&attr.kind, AttrKind::Path(path) if path.is_ident("bool"))
                {
                    Anchor::Flag(index)
                } else {
                    Anchor::Attribute(index)
                }
            }
            Element::Operand(index) if self.op.operands[index].variadic => Anchor::Operand(index),
            Element::ResultTypes => Anchor::Results,
            Element::Region(index)
                if self.bodies.regions[index].1 || self.bodies.regions[index].2 =>
            {
                Anchor::Region(index)
            }
            _ => {
                return Err((
                    *anchor_span,
                    "an anchor is an optional attribute, a variadic operand or result, or an optional or variadic region".to_owned(),
                ));
            }
        };
        // A flag stands for the group and writes no value.
        let elements: Vec<_> = elements
            .into_iter()
            .filter(|(element, _)| {
                !matches!((anchor, element), (Anchor::Flag(_), Element::Attribute(_)))
            })
            .collect();
        if elements.is_empty() {
            return Err((
                span,
                "a flag's group writes a literal that stands for it".to_owned(),
            ));
        }
        Ok((Element::Optional { anchor, elements }, span))
    }
}

/// The first token an element may begin with, when it is known.
fn first_token(element: &Element) -> Option<String> {
    match element {
        Element::Literal(text) => Some(text.clone()),
        Element::Operand(_) => Some("%".into()),
        Element::Symbol(_) => Some("@".into()),
        Element::Successor(_) => Some("^".into()),
        Element::Region(_) | Element::AttrDict { keyword: false } => Some("{".into()),
        Element::AttrDict { keyword: true } => Some("attributes".into()),
        Element::Optional { elements, .. } => elements.first().and_then(|(e, _)| first_token(e)),
        Element::Attribute(_) | Element::ResultType(_) | Element::ResultTypes => None,
    }
}

/// Whether a parser that tries `group` first cannot tell it from `next`.
fn ambiguous(group: Option<&str>, next: Option<&str>) -> bool {
    let value_like = |token: &str| {
        token.starts_with(['@', '{', '[', '"', '-'])
            || token.starts_with(|c: char| c.is_ascii_digit())
            || ["true", "false", "unit"].contains(&token)
    };
    match (group, next) {
        (None, None) => true,
        (None, Some(token)) | (Some(token), None) => value_like(token),
        (Some(a), Some(b)) => a == b,
    }
}

fn check_layout(elements: &[(Element, Span)], first_attrs: &mut Vec<usize>) -> Result<(), Error> {
    // The first element after the symbol decides custom against generic
    // syntax, which begins with `{`, `%`, or `:`.
    for (element, span) in elements {
        let (first, first_span) = match element {
            Element::Symbol(_) => continue,
            Element::Optional { elements, .. } => (&elements[0].0, elements[0].1),
            _ => (element, *span),
        };
        match first {
            Element::Literal(text) if text.is_empty() || text.starts_with(['{', '%', ':']) => {
                return Err((
                    first_span,
                    "a format cannot begin like generic syntax, with `{`, `%`, or `:`".to_owned(),
                ));
            }
            Element::Literal(_) => {}
            Element::Attribute(index) => first_attrs.push(*index),
            Element::Operand(_)
            | Element::ResultType(_)
            | Element::ResultTypes
            | Element::AttrDict { .. }
            | Element::Region(_)
            | Element::Successor(_) => {
                return Err((first_span, "a format begins with a literal or an attribute, not with what generic syntax begins with".to_owned()));
            }
            Element::Symbol(_) | Element::Optional { .. } => {}
        }
        if !matches!(element, Element::Optional { .. }) {
            break;
        }
    }

    for (position, (element, span)) in elements.iter().enumerate() {
        let next = elements.get(position + 1);
        if let (Element::AttrDict { keyword: false }, Some((next, _))) = (element, next)
            && first_token(next).as_deref() == Some("{")
        {
            return Err((*span, "a bare `attr_dict` before a region cannot be told apart from it; write `attr_dict(keyword)`".to_owned()));
        }
        if let Element::Optional { .. } = element
            && let Some((next, _)) = next
            && ambiguous(
                first_token(element).as_deref(),
                first_token(next).as_deref(),
            )
        {
            return Err((*span, "the parser cannot tell this optional group from the element after it by its first token".to_owned()));
        }
    }
    Ok(())
}

fn check(
    crate_path: &TokenStream,
    op: &OperationDef,
    attr_span: Span,
    tokens: &TokenStream,
) -> Result<TokenStream, Error> {
    if op
        .operands
        .iter()
        .filter(|operand| operand.variadic)
        .count()
        > 1
    {
        return Err((
            attr_span,
            "a declarative format supports at most one variadic operand".to_owned(),
        ));
    }
    let items = parse_items(tokens.clone())?;
    let bodies = Bodies::of(op);
    let mut resolver = Resolver {
        written: Written {
            operands: vec![false; op.operands.len()],
            attributes: vec![false; op.attrs.len()],
            results: vec![false; usize::from(matches!(op.results, ResultDef::Single(_)))],
            regions: vec![false; bodies.regions.len()],
            successors: vec![false; bodies.successors.len()],
            ..Written::default()
        },
        bodies,
        op,
    };
    let elements = resolver.items(&items, false)?;
    let written = &resolver.written;

    let missing = |what: String| Err((attr_span, format!("the format does not write {what}")));
    if let Some(operand) = op
        .operands
        .iter()
        .zip(&written.operands)
        .find(|(_, w)| !**w)
    {
        return missing(format!("operand `{}`", operand.0.name));
    }
    if let Some((region, _)) = resolver
        .bodies
        .regions
        .iter()
        .zip(&written.regions)
        .find(|(_, w)| !**w)
    {
        return missing(format!("region `{}`", region.0));
    }
    if let Some((successor, _)) = resolver
        .bodies
        .successors
        .iter()
        .zip(&written.successors)
        .find(|(_, w)| !**w)
    {
        return missing(format!("successor `{}`", successor.0));
    }
    let results_written = match op.results {
        ResultDef::None => true,
        ResultDef::Single(_) => written.results[0],
        ResultDef::Variadic(_) | ResultDef::Optional(_) => written.all_results,
    };
    if !results_written {
        return missing(format!(
            "the type of `{}`",
            resolver.result_name().unwrap_or_default()
        ));
    }
    if !written.attr_dict
        && let Some((attr, _)) = op.attrs.iter().zip(&written.attributes).find(|(_, w)| !**w)
    {
        return Err((
            attr_span,
            format!(
                "the format does not write attribute `{}`; write it or add `attr_dict`",
                attr.name
            ),
        ));
    }
    // An optional attribute is written only as the anchor of a group, and
    // results that may be absent likewise, so their punctuation goes with
    // them.
    for (element, span) in &elements {
        if let Element::ResultTypes = element {
            return Err((
                *span,
                "the operation may have no results; write their types as the anchor of an optional group, such as `(\":\" types(results)^)?`".to_owned(),
            ));
        }
        if let Element::Attribute(index) = element
            && op.attrs[*index].optional
        {
            return Err((*span, "an optional attribute is written as the anchor of an optional group `( .. attr^ .. )?`".to_owned()));
        }
    }

    let mut first_attrs = Vec::new();
    check_layout(&elements, &mut first_attrs)?;

    let fmt = quote!(#crate_path::asm_format::declarative);
    let elements = gen_elements(&fmt, &elements);
    let assertions = first_attrs.into_iter().map(|index| {
        let attr = &op.attrs[index];
        let kind = attr_kind_trait(crate_path, attr);
        let message = format!(
            "attribute `{}` begins the format, so its kind cannot be `_` or a dictionary, which generic syntax could begin with",
            attr.name
        );
        let span = attr.raw_ident.span();
        quote_spanned!(span=> const _: () = ::core::assert!(!#fmt::ambiguous_first_kind(#kind::KIND), #message);)
    });
    Ok(quote!({
        #(#assertions)*
        Some(#crate_path::asm_format::OpAsmFormatDef::Declarative(
            &#fmt::OpFormat { elements: &[#(#elements),*] }
        ))
    }))
}

fn gen_elements(fmt: &TokenStream, elements: &[(Element, Span)]) -> Vec<TokenStream> {
    elements
        .iter()
        .map(|(element, _)| match element {
            Element::Literal(text) => quote!(#fmt::FormatElement::Literal(#text)),
            Element::Operand(i) => quote!(#fmt::FormatElement::Operand(#i)),
            Element::Attribute(i) => quote!(#fmt::FormatElement::Attribute(#i)),
            Element::Symbol(i) => quote!(#fmt::FormatElement::Symbol(#i)),
            Element::ResultType(i) => quote!(#fmt::FormatElement::ResultType(#i)),
            Element::ResultTypes => quote!(#fmt::FormatElement::ResultTypes),
            Element::AttrDict { keyword } => quote!(#fmt::FormatElement::AttrDict { keyword: #keyword }),
            Element::Region(i) => quote!(#fmt::FormatElement::Region(#i)),
            Element::Successor(i) => quote!(#fmt::FormatElement::Successor(#i)),
            Element::Optional { anchor, elements } => {
                let anchor = match *anchor {
                    Anchor::Attribute(i) => quote!(#fmt::FormatAnchor::Attribute(#i)),
                    Anchor::Flag(i) => quote!(#fmt::FormatAnchor::Flag(#i)),
                    Anchor::Operand(i) => quote!(#fmt::FormatAnchor::Operand(#i)),
                    Anchor::Results => quote!(#fmt::FormatAnchor::Results),
                    Anchor::Region(i) => quote!(#fmt::FormatAnchor::Region(#i)),
                };
                let elements = gen_elements(fmt, elements);
                quote!(#fmt::FormatElement::Optional { anchor: #anchor, elements: &[#(#elements),*] })
            }
        })
        .collect()
}
