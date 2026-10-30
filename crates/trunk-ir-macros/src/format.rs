//! `#[asm_format(..)]`: check an operation's format elements against its
//! declaration and emit the `OpFormat` table the printer and parser
//! interpret, following `new-plans/ir.md`, "선언적 어셈블리 형식". A rule the
//! format breaks is reported at the element that breaks it.

use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use quote::{quote, quote_spanned};

use crate::codegen::attr_kind_trait;
use crate::parse::{AttrKind, OperationDef, RegionOrSuccessor, ResultDef};

/// A format rule broken by the element at `span`.
struct FormatError {
    span: Span,
    message: String,
}

impl FormatError {
    /// A `compile_error!` reporting the rule at the element.
    fn to_compile_error(&self) -> TokenStream {
        let message = &self.message;
        quote_spanned!(self.span=> ::core::compile_error!(#message);)
    }
}

fn err<T>(span: Span, message: impl Into<String>) -> Result<T, FormatError> {
    Err(FormatError {
        span,
        message: message.into(),
    })
}

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
        Err(error) => {
            let error = error.to_compile_error();
            quote!({ #error None })
        }
    }
}

/// A declared entity a format element writes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Entity {
    Operand(usize),
    Attribute(usize),
    Results,
    Region(usize),
    Successor(usize),
    AttrDict,
}

enum Element {
    Literal(String),
    Operand(usize),
    Attribute(usize),
    Symbol(usize),
    ResultType,
    ResultTypes,
    AttrDict,
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

struct Format<'a> {
    op: &'a OperationDef,
    /// Regions as (name, optional, variadic), by position among regions.
    regions: Vec<(&'a str, bool, bool)>,
    /// Successors as (name, variadic), by position among successors.
    successors: Vec<(&'a str, bool)>,
    written: Vec<Entity>,
}

impl<'a> Format<'a> {
    fn new(op: &'a OperationDef) -> Self {
        let (mut regions, mut successors) = (Vec::new(), Vec::new());
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
            op,
            regions,
            successors,
            written: Vec::new(),
        }
    }

    fn result_name(&self) -> Option<&'a str> {
        match &self.op.results {
            ResultDef::None => None,
            ResultDef::Single(name) | ResultDef::Variadic(name) | ResultDef::Optional(name) => {
                Some(name)
            }
        }
    }

    fn lookup(&self, name: &str) -> Option<Entity> {
        if let Some(i) = self.op.operands.iter().position(|o| o.name == name) {
            Some(Entity::Operand(i))
        } else if let Some(i) = self.op.attrs.iter().position(|a| a.name == name) {
            Some(Entity::Attribute(i))
        } else if self.result_name() == Some(name) {
            Some(Entity::Results)
        } else if let Some(i) = self.regions.iter().position(|r| r.0 == name) {
            Some(Entity::Region(i))
        } else {
            self.successors
                .iter()
                .position(|s| s.0 == name)
                .map(Entity::Successor)
        }
    }

    fn describe(&self, entity: Entity) -> String {
        match entity {
            Entity::Operand(i) => format!("operand `{}`", self.op.operands[i].name),
            Entity::Attribute(i) => format!("attribute `{}`", self.op.attrs[i].name),
            Entity::Results => format!("the type of `{}`", self.result_name().unwrap_or_default()),
            Entity::Region(i) => format!("region `{}`", self.regions[i].0),
            Entity::Successor(i) => format!("successor `{}`", self.successors[i].0),
            Entity::AttrDict => "`attr_dict`".to_owned(),
        }
    }

    fn write(&mut self, entity: Entity, span: Span) -> Result<(), FormatError> {
        if self.written.contains(&entity) {
            return err(
                span,
                format!("the format writes {} twice", self.describe(entity)),
            );
        }
        self.written.push(entity);
        Ok(())
    }

    /// Parse format tokens into elements, each with whether it is marked
    /// as an anchor with `^`.
    fn elements(
        &mut self,
        tokens: TokenStream,
        in_group: bool,
    ) -> Result<Vec<(Element, Span, bool)>, FormatError> {
        let tokens: Vec<TokenTree> = tokens.into_iter().collect();
        let mut elements = Vec::new();
        let mut index = 0;
        while let Some(token) = tokens.get(index) {
            index += 1;
            let is_punct = |token: Option<&TokenTree>, c: char| matches!(token, Some(TokenTree::Punct(p)) if p.as_char() == c);
            let span = token.span();
            let element = match token {
                TokenTree::Literal(literal) => {
                    let raw = literal.to_string();
                    let Some(text) = raw
                        .strip_prefix('"')
                        .and_then(|rest| rest.strip_suffix('"'))
                        .filter(|text| {
                            !text.contains(['\\', '"']) && !text.contains(char::is_whitespace)
                        })
                    else {
                        return err(
                            span,
                            "a format literal is one token in a plain string literal, such as `\"(\"`",
                        );
                    };
                    Element::Literal(text.to_owned())
                }
                TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis => {
                    if !is_punct(tokens.get(index), '?') {
                        return err(
                            span,
                            "a parenthesized group is an optional group `( .. )?`; write punctuation as a string literal",
                        );
                    }
                    index += 1;
                    if in_group {
                        return err(span, "optional groups do not nest");
                    }
                    let inner = self.elements(group.stream(), true)?;
                    self.group(inner, span)?
                }
                TokenTree::Ident(ident) => {
                    let name = ident.to_string();
                    let name = name.strip_prefix("r#").unwrap_or(&name);
                    // `name(..)` is a directive; `name (..)?` is an entity
                    // followed by an optional group.
                    match tokens.get(index) {
                        Some(TokenTree::Group(group))
                            if group.delimiter() == Delimiter::Parenthesis
                                && !is_punct(tokens.get(index + 1), '?') =>
                        {
                            index += 1;
                            self.directive(name, group.stream(), span)?
                        }
                        _ => self.entity(name, span)?,
                    }
                }
                _ => {
                    return err(
                        span,
                        "write punctuation as a string literal, such as `\":\"`",
                    );
                }
            };
            let anchor = is_punct(tokens.get(index), '^');
            if anchor {
                if !in_group {
                    return err(span, "`^` marks the anchor of an optional group");
                }
                index += 1;
            }
            elements.push((element, span, anchor));
        }
        Ok(elements)
    }

    fn entity(&mut self, name: &str, span: Span) -> Result<Element, FormatError> {
        if name == "attr_dict" {
            self.write(Entity::AttrDict, span)?;
            return Ok(Element::AttrDict);
        }
        let Some(entity) = self.lookup(name) else {
            return err(span, format!("the operation declares no `{name}`"));
        };
        let element = match entity {
            Entity::Operand(i) => Element::Operand(i),
            Entity::Attribute(i) => Element::Attribute(i),
            Entity::Results => {
                return err(
                    span,
                    "a result's type is written with `type(..)` or `types(..)`",
                );
            }
            Entity::Region(_) => {
                return err(
                    span,
                    "a region is written with `region(..)` or `regions(..)`",
                );
            }
            Entity::Successor(_) => {
                return err(
                    span,
                    "a successor is written with `successor(..)` or `successors(..)`",
                );
            }
            Entity::AttrDict => unreachable!("`lookup` finds declared entities"),
        };
        self.write(entity, span)?;
        Ok(element)
    }

    fn directive(
        &mut self,
        name: &str,
        args: TokenStream,
        span: Span,
    ) -> Result<Element, FormatError> {
        if name == "attr_dict" {
            return err(span, "`attr_dict` takes no arguments");
        }
        let args: Vec<TokenTree> = args.into_iter().collect();
        let [TokenTree::Ident(arg)] = &args[..] else {
            return err(span, format!("`{name}(..)` takes one name"));
        };
        let arg_span = arg.span();
        let arg = arg.to_string();
        let arg = arg.strip_prefix("r#").unwrap_or(&arg);
        let entity = self.lookup(arg);
        let plural = name.ends_with('s');
        let element = match (name, entity) {
            ("symbol", Some(Entity::Attribute(i))) => {
                let attr = &self.op.attrs[i];
                if !attr.kind.is_string() || attr.list || attr.optional {
                    return err(arg_span, "`symbol(..)` writes a required `Attr<String>`");
                }
                Element::Symbol(i)
            }
            ("type" | "types", Some(Entity::Results)) => match (&self.op.results, plural) {
                (ResultDef::Single(_), false) => Element::ResultType,
                (ResultDef::Optional(_), false) | (ResultDef::Variadic(_), true) => {
                    Element::ResultTypes
                }
                (ResultDef::Variadic(_), false) => {
                    return err(span, format!("`{arg}` is variadic; write `types({arg})`"));
                }
                _ => {
                    return err(
                        span,
                        format!("`{arg}` is at most one result; write `type({arg})`"),
                    );
                }
            },
            ("region" | "regions", Some(Entity::Region(i))) => {
                if self.regions[i].2 != plural {
                    return err(
                        span,
                        if plural {
                            "write `region(..)` for one region"
                        } else {
                            "write `regions(..)` for a variadic region list"
                        },
                    );
                }
                Element::Region(i)
            }
            ("successor" | "successors", Some(Entity::Successor(i))) => {
                if self.successors[i].1 != plural {
                    return err(
                        span,
                        if plural {
                            "write `successor(..)` for one successor"
                        } else {
                            "write `successors(..)` for a variadic successor list"
                        },
                    );
                }
                Element::Successor(i)
            }
            (
                "symbol" | "type" | "types" | "region" | "regions" | "successor" | "successors",
                _,
            ) => {
                return err(arg_span, format!("`{name}(..)` cannot write `{arg}`"));
            }
            _ => {
                return err(
                    span,
                    format!(
                        "unknown format directive `{name}`; expected `symbol`, `type`, `types`, `region`, `regions`, `successor`, or `successors`"
                    ),
                );
            }
        };
        self.write(entity.expect("matched above"), arg_span)?;
        Ok(element)
    }

    /// An optional group: literals and one entity, its anchor, which may be
    /// absent.
    fn group(
        &self,
        elements: Vec<(Element, Span, bool)>,
        span: Span,
    ) -> Result<Element, FormatError> {
        let mut entities = elements
            .iter()
            .filter(|e| !matches!(e.0, Element::Literal(_)));
        let (Some((anchor, anchor_span, marked)), None) = (entities.next(), entities.next()) else {
            return err(
                span,
                "an optional group holds literals and exactly one entity, its anchor",
            );
        };
        if !marked {
            return err(
                *anchor_span,
                "mark the group's entity as its anchor with `^`",
            );
        }
        let anchor = match *anchor {
            Element::Attribute(i) if self.op.attrs[i].optional => {
                let attr = &self.op.attrs[i];
                match &attr.kind {
                    AttrKind::Path(path) if !attr.list && path.is_ident("bool") => Anchor::Flag(i),
                    _ => Anchor::Attribute(i),
                }
            }
            Element::Operand(i) if self.op.operands[i].variadic => Anchor::Operand(i),
            Element::ResultTypes => Anchor::Results,
            Element::Region(i) if self.regions[i].1 || self.regions[i].2 => Anchor::Region(i),
            _ => {
                return err(
                    *anchor_span,
                    "an anchor is an optional attribute, a variadic operand or result, or an optional or variadic region",
                );
            }
        };
        // A flag stands for the group and writes no value.
        let is_flag = matches!(anchor, Anchor::Flag(_));
        let elements: Vec<_> = elements
            .into_iter()
            .filter(|(element, ..)| !(is_flag && matches!(element, Element::Attribute(_))))
            .map(|(element, span, _)| (element, span))
            .collect();
        if elements.is_empty() {
            return err(span, "a flag's group writes a literal that stands for it");
        }
        Ok(Element::Optional { anchor, elements })
    }
}

/// What an element's text can begin with: fixed text, a value, or the end
/// of the operation, after which the next line can begin with `%`, `^`, `}`,
/// or another operation's name.
#[derive(Clone, Copy)]
enum Token<'a> {
    Text(&'a str),
    Value,
    End,
}

fn first_token(element: &Element) -> Token<'_> {
    match element {
        Element::Literal(text) => Token::Text(text),
        Element::Operand(_) => Token::Text("%"),
        Element::Symbol(_) => Token::Text("@"),
        Element::Successor(_) => Token::Text("^"),
        Element::Region(_) => Token::Text("{"),
        Element::AttrDict => Token::Text("attributes"),
        Element::Optional { elements, .. } => first_token(&elements[0].0),
        Element::Attribute(_) | Element::ResultType | Element::ResultTypes => Token::Value,
    }
}

/// Whether a parser reading one of the two cannot tell it from the other by
/// its first token. A keyword is read as a whole identifier, but punctuation
/// is read by prefix.
fn conflict(a: Token<'_>, b: Token<'_>) -> bool {
    let value_like = |text: &str| {
        text.starts_with(['@', '{', '[', '"', '-'])
            || text.starts_with(|c: char| c.is_ascii_digit())
            || ["true", "false", "unit"].contains(&text)
    };
    let word = |text: &str| text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
    match (a, b) {
        (Token::Value, Token::Value | Token::End) | (Token::End, Token::Value) => true,
        (Token::Value, Token::Text(text)) | (Token::Text(text), Token::Value) => value_like(text),
        (Token::Text(text), Token::End) | (Token::End, Token::Text(text)) => {
            text.starts_with(['%', '^', '}'])
        }
        (Token::Text(a), Token::Text(b)) => {
            a == b || (!word(a) && !word(b) && (a.starts_with(b) || b.starts_with(a)))
        }
        (Token::End, Token::End) => false,
    }
}

impl Format<'_> {
    /// Whether an element may write nothing.
    fn may_be_empty(&self, element: &Element) -> bool {
        match *element {
            Element::Operand(i) => self.op.operands[i].variadic,
            Element::Successor(i) => self.successors[i].1,
            Element::Region(i) => self.regions[i].1 || self.regions[i].2,
            Element::AttrDict | Element::Optional { .. } | Element::ResultTypes => true,
            _ => false,
        }
    }

    /// The first tokens that can follow the end of `elements[..position]`,
    /// with `tail` following the end of `elements`.
    fn follow<'e>(
        &self,
        elements: &'e [(Element, Span)],
        position: usize,
        tail: &[Token<'e>],
    ) -> Vec<Token<'e>> {
        let mut tokens = Vec::new();
        for (element, _) in &elements[position..] {
            tokens.push(first_token(element));
            if !self.may_be_empty(element) {
                return tokens;
            }
        }
        tokens.extend_from_slice(tail);
        tokens
    }

    /// Whether an element writes a comma-separated list, which would read a
    /// `,` after it as its own separator.
    fn is_list(&self, element: &Element) -> bool {
        match *element {
            Element::Operand(i) => self.op.operands[i].variadic,
            Element::Successor(i) => self.successors[i].1,
            Element::ResultTypes => true,
            _ => false,
        }
    }

    /// Reject a list that a `,` can follow, at the top level or inside a
    /// group.
    fn check_lists(&self, elements: &[(Element, Span)]) -> Result<(), FormatError> {
        let comma = |tokens: Vec<Token<'_>>| tokens.iter().any(|t| matches!(t, Token::Text(",")));
        for (position, (element, span)) in elements.iter().enumerate() {
            let after = self.follow(elements, position + 1, &[Token::End]);
            let inner = match element {
                Element::Optional { elements, .. } => elements,
                _ => {
                    if self.is_list(element) && comma(after) {
                        return err(
                            *span,
                            "a comma-separated list cannot be followed by `\",\"`",
                        );
                    }
                    continue;
                }
            };
            for (j, (element, span)) in inner.iter().enumerate() {
                if self.is_list(element) && comma(self.follow(inner, j + 1, &after)) {
                    return err(
                        *span,
                        "a comma-separated list cannot be followed by `\",\"`",
                    );
                }
            }
        }
        Ok(())
    }
}

fn check(
    crate_path: &TokenStream,
    op: &OperationDef,
    attr_span: Span,
    tokens: &TokenStream,
) -> Result<TokenStream, FormatError> {
    let mut format = Format::new(op);
    let elements: Vec<(Element, Span)> = format
        .elements(tokens.clone(), false)?
        .into_iter()
        .map(|(element, span, _)| (element, span))
        .collect();

    let has_dict = format.written.contains(&Entity::AttrDict);
    let required = (0..op.operands.len())
        .map(Entity::Operand)
        .chain((0..format.regions.len()).map(Entity::Region))
        .chain((0..format.successors.len()).map(Entity::Successor))
        .chain(format.result_name().map(|_| Entity::Results))
        .chain(
            (0..op.attrs.len())
                .filter(|_| !has_dict)
                .map(Entity::Attribute),
        );
    for entity in required {
        if !format.written.contains(&entity) {
            let hint = if matches!(entity, Entity::Attribute(_)) {
                "; write it or add `attr_dict`"
            } else {
                ""
            };
            return err(
                attr_span,
                format!(
                    "the format does not write {}{hint}",
                    format.describe(entity)
                ),
            );
        }
    }

    for (position, (element, span)) in elements.iter().enumerate() {
        match element {
            Element::ResultTypes => {
                return err(
                    *span,
                    "the operation may have no results; write their types as the anchor of an optional group, such as `(\":\" types(results)^)?`",
                );
            }
            Element::Attribute(i) if op.attrs[*i].optional => {
                return err(
                    *span,
                    "an optional attribute is written as the anchor of an optional group `( .. attr^ .. )?`",
                );
            }
            _ => {}
        }
        // An element that may write nothing must not begin like anything
        // that can follow it, or the parser reads that as this element.
        if format.may_be_empty(element) {
            let first = first_token(element);
            if format
                .follow(&elements, position + 1, &[Token::End])
                .into_iter()
                .any(|next| conflict(first, next))
            {
                return err(
                    *span,
                    "this element may write nothing, and it begins like what can follow it; separate them with a literal",
                );
            }
        }
    }

    format.check_lists(&elements)?;

    // The elements up to the first that is neither a symbol nor optional
    // decide custom against generic syntax, which begins with `{`, `%`, `:`,
    // or `[`, or ends the operation before a line beginning with `%`, `^`,
    // or `}`. An attribute there must not be of a kind generic syntax could
    // begin with, which only the kind's `KIND` tells.
    let fmt = quote!(#crate_path::asm_format::declarative);
    let mut assertions = Vec::new();
    for (element, span) in &elements {
        let (first, first_span) = match element {
            Element::Symbol(_) => continue,
            Element::Optional { elements, .. } => (&elements[0].0, elements[0].1),
            _ => (element, *span),
        };
        match first {
            Element::Literal(text)
                if text.is_empty() || text.starts_with(['{', '%', ':', '[', '^', '}']) =>
            {
                return err(
                    first_span,
                    "a format cannot begin like generic syntax, with `{`, `%`, `:`, `[`, `^`, or `}`",
                );
            }
            Element::Literal(_) => {}
            Element::Attribute(i) => {
                let attr = &op.attrs[*i];
                let kind = attr_kind_trait(crate_path, attr);
                let message = format!(
                    "attribute `{}` begins the format, so its kind cannot be `_`, a dictionary, or a type, which generic syntax or the next operation could begin with",
                    attr.name
                );
                assertions.push(quote_spanned!(attr.raw_ident.span()=>
                    const _: () = ::core::assert!(!#fmt::ambiguous_first_kind(#kind::KIND), #message);
                ));
            }
            _ => {
                return err(
                    first_span,
                    "a format begins with a literal or an attribute, not with what generic syntax begins with",
                );
            }
        }
        if !matches!(element, Element::Optional { .. }) {
            break;
        }
    }

    let elements = gen_elements(&fmt, &elements);
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
            Element::ResultType => quote!(#fmt::FormatElement::ResultType(0)),
            Element::ResultTypes => quote!(#fmt::FormatElement::ResultTypes),
            Element::AttrDict => quote!(#fmt::FormatElement::AttrDict),
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

#[cfg(test)]
mod tests {
    use quote::quote;

    use super::*;
    use crate::parse::{DialectItem, parse_input};

    /// Check the format of the one operation `item` declares.
    fn check_op(item: TokenStream) -> Result<TokenStream, String> {
        let module = parse_input(quote!(), item).expect("the declaration parses");
        let Some(DialectItem::Operation(op)) = module.items.first() else {
            panic!("expected an operation");
        };
        let tokens = op.format.as_ref().expect("a format");
        check(&quote!(::trunk_ir), op, Span::call_site(), tokens).map_err(|error| error.message)
    }

    #[test]
    fn accepted_formats_emit_their_tables() {
        for item in [
            quote!(
                mod t {
                    #[asm_format(symbol(sym_name) "kind" ("on" on^)? attr_dict region(body))]
                    fn op(sym_name: Attr<String>, on: Option<Attr<bool>>) {
                        #[region(body?)]
                        {}
                    }
                }
            ),
            quote!(
                mod t {
                    #[asm_format("to" successors(dests) ";" "(" args ")" (":" types(results)^)? regions(bodies))]
                    fn op(args: Variadic<_>) -> Variadic<_> {
                        #[successors(dests)]
                        {}
                        #[regions(bodies)]
                        {}
                    }
                }
            ),
            quote!(
                mod t {
                    #[asm_format(tag (":" type(result)^)? "on" successor(dest) region(body))]
                    fn op(tag: Attr<SymbolRef>) -> Option<Value<_>> {
                        #[region(body)]
                        {}
                        #[successor(dest)]
                        {}
                    }
                }
            ),
        ] {
            let tokens = check_op(item).expect("the format is accepted").to_string();
            assert!(tokens.contains("OpFormat"), "{tokens}");
        }
    }

    #[test]
    fn broken_rules_are_reported() {
        let cases = [
            (
                quote!(
                    fn op() {}
                ),
                quote!("x" "a b"),
                "one token",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("x" 'c'),
                "plain string literal",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("x"("y")),
                "optional group `( .. )?`",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("x" ;),
                "punctuation",
            ),
            (
                quote!(
                    fn op(a: Option<Attr<bool>>) {}
                ),
                quote!("x" (("y" a^)?)?),
                "do not nest",
            ),
            (
                quote!(
                    fn op(a: Value<_>) {}
                ),
                quote!("x" a^),
                "anchor of an optional group",
            ),
            (
                quote!(
                    fn op(a: Value<_>) {}
                ),
                quote!("x" a a),
                "twice",
            ),
            (
                quote!(
                    fn op() -> Value<_> {}
                ),
                quote!("x" result),
                "`type(..)`",
            ),
            (
                quote!(
                    fn op() {
                        #[region(r)]
                        {}
                    }
                ),
                quote!("x" r),
                "`region(..)`",
            ),
            (
                quote!(
                    fn op() {
                        #[successor(s)]
                        {}
                    }
                ),
                quote!("x" s),
                "`successor(..)`",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("x" missing),
                "declares no `missing`",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("x" attr_dict(keyword)),
                "takes no arguments",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("x" type(a, b)),
                "takes one name",
            ),
            (
                quote!(
                    fn op(a: Attr<u32>) {}
                ),
                quote!("x" symbol(a)),
                "`Attr<String>`",
            ),
            (
                quote!(
                    fn op() -> Variadic<_> {}
                ),
                quote!("x" (":" type(results)^)?),
                "write `types",
            ),
            (
                quote!(
                    fn op() -> Value<_> {}
                ),
                quote!("x" ":" types(result)),
                "write `type",
            ),
            (
                quote!(
                    fn op() {
                        #[region(r)]
                        {}
                    }
                ),
                quote!("x" regions(r)),
                "write `region(..)`",
            ),
            (
                quote!(
                    fn op() {
                        #[successor(s)]
                        {}
                    }
                ),
                quote!("x" successors(s)),
                "write `successor(..)`",
            ),
            (
                quote!(
                    fn op(a: Value<_>) {}
                ),
                quote!("x" region(a)),
                "cannot write `a`",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("x" unknown(a)),
                "unknown format directive",
            ),
            (
                quote!(
                    fn op(a: Option<Attr<bool>>, b: Option<Attr<bool>>) {}
                ),
                quote!("x" (a^ b^)?),
                "exactly one entity",
            ),
            (
                quote!(
                    fn op(a: Option<Attr<bool>>) {}
                ),
                quote!("x" ("y" a)?),
                "mark the group's entity",
            ),
            (
                quote!(
                    fn op(a: Attr<String>) {}
                ),
                quote!("x" ("y" a^)?),
                "an anchor is",
            ),
            (
                quote!(
                    fn op(a: Option<Attr<bool>>) {}
                ),
                quote!("x" (a^)?),
                "flag's group",
            ),
            (
                quote!(
                    fn op(a: Value<_>) {}
                ),
                quote!("x"),
                "does not write operand `a`",
            ),
            (
                quote!(
                    fn op(a: Attr<u32>) {}
                ),
                quote!("x"),
                "add `attr_dict`",
            ),
            (
                quote!(
                    fn op() -> Variadic<_> {}
                ),
                quote!("x" ":" types(results)),
                "may have no results",
            ),
            (
                quote!(
                    fn op(a: Option<Attr<u32>>) {}
                ),
                quote!("x" a),
                "anchor of an optional group `(",
            ),
            (
                quote!(
                    fn op(a: Variadic<_>) {}
                ),
                quote!("x" a),
                "may write nothing",
            ),
            (
                quote!(
                    fn op(a: Value<_>, b: Variadic<_>) {}
                ),
                quote!("x" b "," a),
                "comma-separated list",
            ),
            (
                quote!(
                    fn op() {}
                ),
                quote!("[" "x"),
                "cannot begin like generic syntax",
            ),
            (
                quote!(
                    fn op(a: Value<_>) {}
                ),
                quote!(a "x"),
                "begins with a literal or an attribute",
            ),
        ];
        let failures: Vec<String> = cases
            .into_iter()
            .filter_map(|(op, format, expected)| {
                let item = quote!(mod t { #[asm_format(#format)] #op });
                match check_op(item) {
                    Err(message) if message.contains(expected) => None,
                    Err(message) => Some(format!("{format}: `{message}` lacks `{expected}`")),
                    Ok(_) => Some(format!("{format}: accepted, expected `{expected}`")),
                }
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
