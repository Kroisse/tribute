//! `#[asm_format(..)]`: check an operation's format elements against its
//! declaration and emit the `OpFormat` table the printer and parser
//! interpret, following `new-plans/ir.md`, "선언적 어셈블리 형식". A rule the
//! format breaks is reported at the element that breaks it.

use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use quote::{quote, quote_spanned};

use crate::codegen::attr_kind_trait;
use crate::parse::{AttrKind, OperationDef, RegionOrSuccessor, ResultDef};

type Error = (Span, String);

fn err<T>(span: Span, message: impl Into<String>) -> Result<T, Error> {
    Err((span, message.into()))
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
        Err((span, message)) => quote_spanned!(span=> {
            ::core::compile_error!(#message);
            None
        }),
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

    fn write(&mut self, entity: Entity, span: Span) -> Result<(), Error> {
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
    ) -> Result<Vec<(Element, Span, bool)>, Error> {
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

    fn entity(&mut self, name: &str, span: Span) -> Result<Element, Error> {
        if name == "attr_dict" {
            self.write(Entity::AttrDict, span)?;
            return Ok(Element::AttrDict { keyword: false });
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

    fn directive(&mut self, name: &str, args: TokenStream, span: Span) -> Result<Element, Error> {
        let args: Vec<TokenTree> = args.into_iter().collect();
        let [TokenTree::Ident(arg)] = &args[..] else {
            return err(span, format!("`{name}(..)` takes one name"));
        };
        let arg_span = arg.span();
        let arg = arg.to_string();
        let arg = arg.strip_prefix("r#").unwrap_or(&arg);
        if name == "attr_dict" {
            if arg != "keyword" {
                return err(arg_span, "expected `attr_dict` or `attr_dict(keyword)`");
            }
            self.write(Entity::AttrDict, span)?;
            return Ok(Element::AttrDict { keyword: true });
        }
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
                        "unknown format directive `{name}`; expected `symbol`, `type`, `types`, `attr_dict`, `region`, `regions`, `successor`, or `successors`"
                    ),
                );
            }
        };
        self.write(entity.expect("matched above"), arg_span)?;
        Ok(element)
    }

    /// An optional group: literals and one entity, its anchor, which may be
    /// absent.
    fn group(&self, elements: Vec<(Element, Span, bool)>, span: Span) -> Result<Element, Error> {
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

/// The first token an element begins with, when it is fixed.
fn first_token(element: &Element) -> Option<&str> {
    match element {
        Element::Literal(text) => Some(text),
        Element::Operand(_) => Some("%"),
        Element::Symbol(_) => Some("@"),
        Element::Successor(_) => Some("^"),
        Element::Region(_) | Element::AttrDict { keyword: false } => Some("{"),
        Element::AttrDict { keyword: true } => Some("attributes"),
        Element::Optional { elements, .. } => first_token(&elements[0].0),
        Element::Attribute(_) | Element::ResultType | Element::ResultTypes => None,
    }
}

/// Whether a parser that tries an optional group first cannot tell it from
/// the element after it. An element without a fixed first token is a value,
/// which a value-like token can begin.
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
        return err(
            attr_span,
            "a declarative format supports at most one variadic operand",
        );
    }
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
        let next = elements
            .get(position + 1)
            .map(|(next, _)| first_token(next));
        if matches!(element, Element::AttrDict { keyword: false }) && next == Some(Some("{")) {
            return err(
                *span,
                "a bare `attr_dict` before a region cannot be told apart from it; write `attr_dict(keyword)`",
            );
        }
        if matches!(element, Element::Optional { .. })
            && let Some(next) = next
            && ambiguous(first_token(element), next)
        {
            return err(
                *span,
                "the parser cannot tell this optional group from the element after it by its first token",
            );
        }
    }

    // The elements up to the first that is neither a symbol nor optional
    // decide custom against generic syntax, which begins with `{`, `%`, or
    // `:`. An attribute there must not be of a kind generic syntax could
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
            Element::Literal(text) if text.is_empty() || text.starts_with(['{', '%', ':']) => {
                return err(
                    first_span,
                    "a format cannot begin like generic syntax, with `{`, `%`, or `:`",
                );
            }
            Element::Literal(_) => {}
            Element::Attribute(i) => {
                let attr = &op.attrs[*i];
                let kind = attr_kind_trait(crate_path, attr);
                let message = format!(
                    "attribute `{}` begins the format, so its kind cannot be `_` or a dictionary, which generic syntax could begin with",
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
