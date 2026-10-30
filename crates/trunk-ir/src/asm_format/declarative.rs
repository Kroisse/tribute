//! Declarative operation assembly formats.
//!
//! `#[asm_format(..)]` with format elements stores an [`OpFormat`] in the
//! operation's definition. The printer and the parser interpret it against
//! the definition's [`OpSchema`], so one declaration gives both directions
//! (`new-plans/ir.md`, "선언적 어셈블리 형식").

use std::borrow::Cow;
use std::fmt::{self, Write};
use std::ops::Range;

use winnow::combinator::{opt, separated};
use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
use winnow::prelude::*;

use crate::op_def::OpDef;
use crate::op_schema::{Arity, AttributeKind, OpSchema, ResultSchema};
use crate::parser::raw::{
    RawAttribute, RawOperation, RawRegion, RawType, block_label, ident, raw_attr_dict,
    raw_attr_value, raw_region, raw_type, symbol_ref, value_ref, ws,
};
use crate::printer::OpPrintHelper;
use crate::types::Attribute;
use crate::{IrContext, OpRef};

/// The format elements of an operation, in order.
#[derive(Debug)]
pub struct OpFormat {
    pub elements: &'static [FormatElement],
}

/// One element of an operation format. Indices refer to the entities of the
/// operation's [`OpSchema`].
#[derive(Debug)]
pub enum FormatElement {
    /// A keyword or punctuation written as is. `""` drops the space before
    /// the next element.
    Literal(&'static str),
    /// An operand: `%name`, or a comma-separated list for a variadic one.
    Operand(usize),
    /// An attribute in generic attribute syntax.
    Attribute(usize),
    /// A string attribute written as a definition name, `@name`.
    Symbol(usize),
    /// The type of one fixed result.
    ResultType(usize),
    /// The comma-separated types of every result, for a variadic or optional
    /// result.
    ResultTypes,
    /// The attributes no other element writes, as `attributes {..}`.
    AttrDict,
    /// A region in generic form, or each region of a variadic list.
    Region(usize),
    /// A successor's label, or a comma-separated list for a variadic one.
    Successor(usize),
    /// Elements written only when their anchor is present.
    Optional {
        anchor: FormatAnchor,
        elements: &'static [FormatElement],
    },
}

/// What decides whether an optional group is written.
#[derive(Clone, Copy, Debug)]
pub enum FormatAnchor {
    /// An optional attribute, written in the group.
    Attribute(usize),
    /// An optional `bool` attribute the group stands for: present and `true`
    /// exactly when the group is written.
    Flag(usize),
    /// A variadic operand that is not empty.
    Operand(usize),
    /// Results, when there is at least one.
    Results,
    /// An optional region, or a variadic region list that is not empty.
    Region(usize),
}

// =============================================================================
// Entity positions
// =============================================================================

/// The positions of entity `index` in a list of `count` values whose
/// entities have the given arities, of which at most one is variadic.
fn segment(
    arities: impl ExactSizeIterator<Item = bool>,
    index: usize,
    count: usize,
) -> Range<usize> {
    let len = arities.len();
    let variadic: Vec<bool> = arities.collect();
    match variadic.iter().position(|&v| v) {
        None => index..index + 1,
        Some(v) => {
            let rest = count.saturating_sub(len - 1);
            if index < v {
                index..index + 1
            } else if index == v {
                v..v + rest
            } else {
                let start = index - 1 + rest;
                start..start + 1
            }
        }
    }
}

fn operand_range(schema: &OpSchema, index: usize, count: usize) -> Range<usize> {
    let arities = schema
        .operands
        .iter()
        .map(|operand| operand.arity == Arity::Variadic);
    segment(arities, index, count)
}

/// Regions after the required ones are optional or variadic; either way an
/// index reaches to the end of the list.
fn region_range(schema: &OpSchema, index: usize, count: usize) -> Range<usize> {
    let region = &schema.regions[index];
    if region.variadic {
        index.min(count)..count
    } else {
        index.min(count)..(index + 1).min(count)
    }
}

fn successor_range(schema: &OpSchema, index: usize, count: usize) -> Range<usize> {
    let arities = schema.successors.iter().map(|successor| successor.variadic);
    segment(arities, index, count)
}

fn fixed_count(arities: impl Iterator<Item = bool>) -> (usize, bool) {
    let mut fixed = 0;
    let mut variadic = false;
    for v in arities {
        if v {
            variadic = true;
        } else {
            fixed += 1;
        }
    }
    (fixed, variadic)
}

fn count_fits((fixed, variadic): (usize, bool), count: usize) -> bool {
    if variadic {
        count >= fixed
    } else {
        count == fixed
    }
}

// =============================================================================
// Printing
// =============================================================================

/// Print `op` with `format`, or generically when the format cannot spell it
/// without loss.
pub(crate) fn print(
    h: &mut OpPrintHelper<'_, '_>,
    op: OpRef,
    def: &'static OpDef,
    format: &OpFormat,
) -> fmt::Result {
    if !printable(h.ctx(), op, &def.schema, format) {
        return h.print_generic(op);
    }
    let ctx = h.ctx();
    let results = ctx.op_results(op);
    if !results.is_empty() {
        for (index, &result) in results.iter().enumerate() {
            if index > 0 {
                h.write_str(", ")?;
            }
            let name = h.assign_value_name(result);
            h.write_str(&name)?;
        }
        h.write_str(" = ")?;
    }
    write!(h, "{}", ctx.op(op).name.display(ctx))?;
    let mut printer = Printer {
        h,
        op,
        schema: &def.schema,
        format,
        spacing: Spacing::Space,
    };
    printer.elements(format.elements)
}

/// Whether the space before the next token is written.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Spacing {
    Space,
    Glue,
}

struct Printer<'h, 'a, 'ctx> {
    h: &'h mut OpPrintHelper<'a, 'ctx>,
    op: OpRef,
    schema: &'static OpSchema,
    format: &'h OpFormat,
    spacing: Spacing,
}

impl Printer<'_, '_, '_> {
    /// Start a token whose text begins with `first`, writing the space
    /// before it unless the spacing rules drop it.
    fn token(&mut self, first: char) -> fmt::Result {
        let glued = matches!(first, ')' | ']' | ',' | '(');
        if self.spacing == Spacing::Space && !glued {
            self.h.write_char(' ')?;
        }
        self.spacing = Spacing::Space;
        Ok(())
    }

    fn elements(&mut self, elements: &[FormatElement]) -> fmt::Result {
        elements
            .iter()
            .try_for_each(|element| self.element(element))
    }

    fn element(&mut self, element: &FormatElement) -> fmt::Result {
        let ctx = self.h.ctx();
        let op = self.op;
        match *element {
            FormatElement::Literal("") => {
                self.spacing = Spacing::Glue;
                Ok(())
            }
            FormatElement::Literal(text) => {
                self.token(text.chars().next().unwrap_or(' '))?;
                self.h.write_str(text)?;
                if text.ends_with(['(', '[']) {
                    self.spacing = Spacing::Glue;
                }
                Ok(())
            }
            FormatElement::Operand(index) => {
                let operands = ctx.op_operands(op);
                let range = operand_range(self.schema, index, operands.len());
                if range.is_empty() {
                    return Ok(());
                }
                self.token('%')?;
                for (position, &value) in operands[range].iter().enumerate() {
                    if position > 0 {
                        self.h.write_str(", ")?;
                    }
                    let name = self.h.get_value_name(value).to_owned();
                    self.h.write_str(&name)?;
                }
                Ok(())
            }
            FormatElement::Attribute(index) => {
                let name = self.schema.attributes[index].name;
                match ctx.op_attrs(op).get(name) {
                    Some(value) => {
                        self.token('a')?;
                        self.h.write_attribute(value)
                    }
                    None => Ok(()),
                }
            }
            FormatElement::Symbol(index) => {
                let name = self.schema.attributes[index].name;
                let text = ctx
                    .op_attrs(op)
                    .get_str(name)
                    .expect("a printable symbol attribute is a string");
                self.token('@')?;
                self.h.write_symbol_text(text)
            }
            FormatElement::ResultType(index) => {
                let ty = ctx.op_result_types(op)[index];
                self.token('t')?;
                self.h.write_type(ty)
            }
            FormatElement::ResultTypes => {
                let types = ctx.op_result_types(op);
                if types.is_empty() {
                    return Ok(());
                }
                self.token('t')?;
                for (position, &ty) in types.iter().enumerate() {
                    if position > 0 {
                        self.h.write_str(", ")?;
                    }
                    self.h.write_type(ty)?;
                }
                Ok(())
            }
            FormatElement::AttrDict => {
                let written = written_attributes(self.schema, self.format);
                let mut rest = ctx
                    .op(op)
                    .attributes
                    .iter()
                    .filter(|&(key, _)| !written.contains(&ctx.str(key)))
                    .peekable();
                if rest.peek().is_none() {
                    return Ok(());
                }
                self.token('a')?;
                self.h.write_str("attributes ")?;
                self.h.write_attr_dict(rest)
            }
            FormatElement::Region(index) => {
                let count = ctx.op_region_count(op);
                for position in region_range(self.schema, index, count) {
                    let region = ctx
                        .op_region(op, position)
                        .expect("a region within the counted list");
                    self.token('{')?;
                    self.h.print_region(region)?;
                }
                Ok(())
            }
            FormatElement::Successor(index) => {
                let count = ctx.op_successor_count(op);
                let range = successor_range(self.schema, index, count);
                if range.is_empty() {
                    return Ok(());
                }
                self.token('^')?;
                for (position, successor) in range.enumerate() {
                    if position > 0 {
                        self.h.write_str(", ")?;
                    }
                    let block = ctx
                        .op_successor(op, successor)
                        .expect("a successor within the counted list");
                    let label = self.h.block_label(block).to_owned();
                    self.h.write_str(&label)?;
                }
                Ok(())
            }
            FormatElement::Optional { anchor, elements } => {
                if anchor_present(ctx, op, self.schema, anchor) {
                    self.elements(elements)?;
                }
                Ok(())
            }
        }
    }
}

/// The names of the attributes a format writes outside its attribute
/// dictionary.
fn written_attributes(schema: &OpSchema, format: &OpFormat) -> Vec<&'static str> {
    fn collect(schema: &OpSchema, elements: &[FormatElement], names: &mut Vec<&'static str>) {
        for element in elements {
            match *element {
                FormatElement::Attribute(index) | FormatElement::Symbol(index) => {
                    names.push(schema.attributes[index].name)
                }
                FormatElement::Optional { anchor, elements } => {
                    if let FormatAnchor::Flag(index) = anchor {
                        names.push(schema.attributes[index].name);
                    }
                    collect(schema, elements, names);
                }
                _ => {}
            }
        }
    }
    let mut names = Vec::new();
    collect(schema, format.elements, &mut names);
    names
}

fn anchor_present(ctx: &IrContext, op: OpRef, schema: &OpSchema, anchor: FormatAnchor) -> bool {
    match anchor {
        FormatAnchor::Attribute(index) => {
            ctx.op_attrs(op).contains_key(schema.attributes[index].name)
        }
        FormatAnchor::Flag(index) => {
            matches!(
                ctx.op_attrs(op).get(schema.attributes[index].name),
                Some(Attribute::Bool(true))
            )
        }
        FormatAnchor::Operand(index) => {
            !operand_range(schema, index, ctx.op_operands(op).len()).is_empty()
        }
        FormatAnchor::Results => !ctx.op_results(op).is_empty(),
        FormatAnchor::Region(index) => {
            !region_range(schema, index, ctx.op_region_count(op)).is_empty()
        }
    }
}

/// Whether `format` spells `op` without loss, so that parsing the text gives
/// the same operation back.
fn printable(ctx: &IrContext, op: OpRef, schema: &OpSchema, format: &OpFormat) -> bool {
    let results_fit = match schema.results {
        ResultSchema::Fixed(names) => ctx.op_results(op).len() == names.len(),
        ResultSchema::Optional(_) => ctx.op_results(op).len() <= 1,
        ResultSchema::Variadic(_) => true,
    };
    let operands = fixed_count(schema.operands.iter().map(|o| o.arity == Arity::Variadic));
    let successors = fixed_count(schema.successors.iter().map(|s| s.variadic));
    let required_regions = schema
        .regions
        .iter()
        .filter(|region| !region.optional && !region.variadic)
        .count();
    let region_count = ctx.op_region_count(op);
    let regions_fit = region_count >= required_regions
        && (schema.regions.last().is_some_and(|region| region.variadic)
            || region_count <= schema.regions.len());
    if !results_fit
        || !count_fits(operands, ctx.op_operands(op).len())
        || !count_fits(successors, ctx.op_successor_count(op))
        || !regions_fit
    {
        return false;
    }

    let attrs = ctx.op_attrs(op);
    let mut written_ok = true;
    visit(format.elements, &mut |element| match *element {
        FormatElement::Attribute(index) => {
            let attribute = &schema.attributes[index];
            written_ok &= match attrs.get(attribute.name) {
                Some(value) => attribute.kind.accepts(value),
                None => attribute.optional,
            };
        }
        FormatElement::Symbol(index) => {
            let attribute = &schema.attributes[index];
            written_ok &= matches!(attrs.get(attribute.name), Some(Attribute::String(_)));
        }
        FormatElement::Optional {
            anchor: FormatAnchor::Flag(index),
            ..
        } => {
            written_ok &= matches!(
                attrs.get(schema.attributes[index].name),
                None | Some(Attribute::Bool(true))
            );
        }
        _ => {}
    });
    if !written_ok {
        return false;
    }
    let has_dict = {
        let mut found = false;
        visit(format.elements, &mut |element| {
            found |= matches!(element, FormatElement::AttrDict)
        });
        found
    };
    has_dict || {
        let written = written_attributes(schema, format);
        ctx.op(op)
            .attributes
            .iter()
            .all(|(key, _)| written.contains(&ctx.str(key)))
    }
}

fn visit(elements: &[FormatElement], f: &mut impl FnMut(&FormatElement)) {
    for element in elements {
        f(element);
        if let FormatElement::Optional { elements, .. } = element {
            visit(elements, f);
        }
    }
}

// =============================================================================
// Parsing
// =============================================================================

/// The entities parsed so far, by schema position.
struct Parsed<'a> {
    operands: Vec<Vec<&'a str>>,
    attributes: Vec<(Cow<'a, str>, RawAttribute<'a>)>,
    result_types: Vec<Vec<RawType<'a>>>,
    regions: Vec<Vec<RawRegion<'a>>>,
    successors: Vec<Vec<&'a str>>,
}

/// Parse an operation written with `format` after its `dialect.name`.
///
/// Until the first element that is not a symbol has read its token, a
/// mismatch backtracks so that the caller reads generic syntax instead;
/// after that, it is an error.
pub(crate) fn parse<'a>(
    input: &mut &'a str,
    name: &'a str,
    results: Vec<&'a str>,
    def: &'static OpDef,
    format: &'static OpFormat,
) -> ModalResult<RawOperation<'a>> {
    let schema = &def.schema;
    let result_slots = match schema.results {
        ResultSchema::Fixed(names) => names.len(),
        ResultSchema::Optional(_) | ResultSchema::Variadic(_) => 1,
    };
    let mut parser = FormatParser {
        schema,
        format,
        committed: false,
        matched: false,
        parsed: Parsed {
            operands: vec![Vec::new(); schema.operands.len()],
            attributes: Vec::new(),
            result_types: vec![Vec::new(); result_slots],
            regions: vec![Vec::new(); schema.regions.len()],
            successors: vec![Vec::new(); schema.successors.len()],
        },
    };
    for element in format.elements {
        parser.committed_element(input, element)?;
    }
    // Every element was absent, so the input may as well be generic syntax,
    // which reads an operation without any of them the same way.
    if !parser.matched {
        return Err(ErrMode::Backtrack(expected("the operation's format")));
    }
    let Parsed {
        operands,
        attributes,
        result_types,
        regions,
        successors,
    } = parser.parsed;
    Ok(RawOperation {
        results,
        name,
        sym_name: None,
        has_func_signature: false,
        func_params: Vec::new(),
        return_type: None,
        operands: operands.into_iter().flatten().collect(),
        attributes,
        result_types: result_types.into_iter().flatten().collect(),
        regions: regions.into_iter().flatten().collect(),
        successors: successors.into_iter().flatten().collect(),
    })
}

fn expected(description: &'static str) -> ContextError {
    let mut error = ContextError::new();
    error.push(StrContext::Expected(StrContextValue::Description(
        description,
    )));
    error
}

struct FormatParser<'a> {
    schema: &'static OpSchema,
    format: &'static OpFormat,
    committed: bool,
    /// Whether any element, a symbol included, has read a token.
    matched: bool,
    parsed: Parsed<'a>,
}

impl<'a> FormatParser<'a> {
    /// Parse one element, committing once it reads a token.
    fn committed_element(
        &mut self,
        input: &mut &'a str,
        element: &FormatElement,
    ) -> ModalResult<()> {
        ws.parse_next(input)?;
        let before = input.len();
        match self.element(input, element) {
            Ok(()) => {
                if input.len() != before {
                    self.matched = true;
                    self.committed |= !matches!(element, FormatElement::Symbol(_));
                }
                Ok(())
            }
            Err(ErrMode::Backtrack(error)) if self.committed => Err(ErrMode::Cut(error)),
            Err(error) => Err(error),
        }
    }

    fn element(&mut self, input: &mut &'a str, element: &FormatElement) -> ModalResult<()> {
        ws.parse_next(input)?;
        match *element {
            FormatElement::Literal("") => Ok(()),
            FormatElement::Literal(text) => literal(input, text),
            FormatElement::Operand(index) => {
                let values = if self.schema.operands[index].arity == Arity::Variadic {
                    separated(0.., (ws, value_ref, ws).map(|(_, v, _)| v), ',').parse_next(input)?
                } else {
                    vec![
                        value_ref
                            .parse_next(input)
                            .map_err(|e| e.map(|_| expected("operand")))?,
                    ]
                };
                self.parsed.operands[index] = values;
                Ok(())
            }
            FormatElement::Attribute(index) => {
                let checkpoint = *input;
                let value = raw_attr_value
                    .parse_next(input)
                    .map_err(|e| e.map(|_| expected("attribute value")))?;
                // `true`, `false`, and `unit` are read by prefix; one that
                // continues as a name is part of the next operation's name.
                let keyword_prefix = matches!(value, RawAttribute::Bool(_) | RawAttribute::Unit)
                    && input
                        .starts_with(|c: char| c == '.' || c == '_' || c.is_ascii_alphanumeric());
                if keyword_prefix || !raw_accepts(self.schema.attributes[index].kind, &value) {
                    *input = checkpoint;
                    return Err(ErrMode::Backtrack(expected(
                        "an attribute value of the declared kind",
                    )));
                }
                self.attribute(index, value);
                Ok(())
            }
            FormatElement::Symbol(index) => {
                let name = symbol_ref
                    .parse_next(input)
                    .map_err(|e| e.map(|_| expected("`@name`")))?;
                self.attribute(index, RawAttribute::String(name));
                Ok(())
            }
            FormatElement::ResultType(index) => {
                let ty = raw_type
                    .parse_next(input)
                    .map_err(|e| e.map(|_| expected("result type")))?;
                self.parsed.result_types[index] = vec![ty];
                Ok(())
            }
            FormatElement::ResultTypes => {
                let types: Vec<RawType<'a>> =
                    separated(0.., (ws, raw_type, ws).map(|(_, t, _)| t), ',').parse_next(input)?;
                self.parsed.result_types[0] = types;
                Ok(())
            }
            FormatElement::AttrDict => {
                let dict = if literal(input, "attributes").is_ok() {
                    ws.parse_next(input)?;
                    Some(raw_attr_dict.parse_next(input)?)
                } else {
                    None
                };
                let written = written_attributes(self.schema, self.format);
                for (key, value) in dict.into_iter().flatten() {
                    if written.contains(&key.as_ref()) {
                        return Err(ErrMode::Cut(expected(
                            "an attribute dictionary without the attributes the format writes",
                        )));
                    }
                    self.parsed.attributes.push((key, value));
                }
                Ok(())
            }
            FormatElement::Region(index) => {
                let region = &self.schema.regions[index];
                let mut regions = Vec::new();
                while input.starts_with('{') {
                    regions.push(raw_region.parse_next(input)?);
                    if !region.variadic {
                        break;
                    }
                    ws.parse_next(input)?;
                }
                if regions.is_empty() && !region.optional && !region.variadic {
                    return Err(ErrMode::Backtrack(expected("region")));
                }
                self.parsed.regions[index] = regions;
                Ok(())
            }
            FormatElement::Successor(index) => {
                let labels = if self.schema.successors[index].variadic {
                    separated(0.., (ws, block_label, ws).map(|(_, l, _)| l), ',')
                        .parse_next(input)?
                } else {
                    vec![
                        block_label
                            .parse_next(input)
                            .map_err(|e| e.map(|_| expected("successor")))?,
                    ]
                };
                self.parsed.successors[index] = labels;
                Ok(())
            }
            FormatElement::Optional { anchor, elements } => {
                // `""` spacing literals read nothing; the first element after
                // them decides the group.
                let start = elements
                    .iter()
                    .position(|element| !matches!(element, FormatElement::Literal("")))
                    .unwrap_or(elements.len());
                let Some((first, rest)) = elements[start..].split_first() else {
                    return Ok(());
                };
                // The group is present when its first element reads a token.
                let checkpoint = *input;
                match self.element(input, first) {
                    Ok(()) if input.len() != checkpoint.len() => {}
                    Ok(()) | Err(ErrMode::Backtrack(_)) => {
                        *input = checkpoint;
                        return Ok(());
                    }
                    Err(error) => return Err(error),
                }
                self.committed = true;
                self.matched = true;
                if let FormatAnchor::Flag(index) = anchor {
                    self.attribute(index, RawAttribute::Bool(true));
                }
                rest.iter()
                    .try_for_each(|element| self.committed_element(input, element))?;
                // The printer writes a group only for a present anchor.
                let present = match anchor {
                    FormatAnchor::Attribute(_) | FormatAnchor::Flag(_) => true,
                    FormatAnchor::Operand(index) => !self.parsed.operands[index].is_empty(),
                    FormatAnchor::Results => !self.parsed.result_types[0].is_empty(),
                    FormatAnchor::Region(index) => !self.parsed.regions[index].is_empty(),
                };
                if !present {
                    return Err(ErrMode::Cut(expected("the optional group's anchor")));
                }
                Ok(())
            }
        }
    }

    fn attribute(&mut self, index: usize, value: RawAttribute<'a>) {
        let name = self.schema.attributes[index].name;
        self.parsed.attributes.push((name.into(), value));
    }
}

/// Whether a parsed attribute value can have `kind`, as far as its syntax
/// shows; the builder checks integer ranges and types.
fn raw_accepts(kind: AttributeKind, value: &RawAttribute<'_>) -> bool {
    match (kind, value) {
        (AttributeKind::Any, _) => true,
        (AttributeKind::Bool, RawAttribute::Bool(_)) => true,
        (
            AttributeKind::I32 | AttributeKind::I64 | AttributeKind::U32 | AttributeKind::U64,
            RawAttribute::Int(_),
        ) => true,
        (AttributeKind::F32 | AttributeKind::F64, RawAttribute::Float(_)) => true,
        (AttributeKind::Type, RawAttribute::Type(_)) => true,
        (AttributeKind::String, RawAttribute::String(_)) => true,
        (AttributeKind::SymbolRef, RawAttribute::SymbolRef(_)) => true,
        (AttributeKind::Bytes, RawAttribute::Bytes(_)) => true,
        (AttributeKind::List(element), RawAttribute::List(items)) => {
            items.iter().all(|item| raw_accepts(*element, item))
        }
        (AttributeKind::Dict(value), RawAttribute::Dict(entries)) => {
            entries.iter().all(|(_, entry)| raw_accepts(*value, entry))
        }
        _ => false,
    }
}

/// Read `text`; a keyword must not continue as a longer identifier or as an
/// operation name.
fn literal(input: &mut &str, text: &'static str) -> ModalResult<()> {
    let error = || {
        let mut error = ContextError::new();
        error.push(StrContext::Expected(StrContextValue::StringLiteral(text)));
        ErrMode::Backtrack(error)
    };
    let is_word = text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
    if is_word {
        let checkpoint = *input;
        return match opt(ident).parse_next(input)? {
            Some(word) if word == text && !input.starts_with('.') => Ok(()),
            _ => {
                *input = checkpoint;
                Err(error())
            }
        };
    }
    match input.strip_prefix(text) {
        Some(rest) => {
            *input = rest;
            Ok(())
        }
        None => Err(error()),
    }
}

/// Whether `kind` cannot begin a declarative format: its syntax may begin
/// with `{`, which also begins generic operation syntax, or read the name of
/// the next operation as a type.
pub const fn ambiguous_first_kind(kind: AttributeKind) -> bool {
    matches!(
        kind,
        AttributeKind::Any | AttributeKind::Dict(_) | AttributeKind::Type
    )
}

#[cfg(test)]
mod tests {
    use crate::IrContext;
    use crate::attr_kind::SymbolRef;
    use crate::parser::{parse_module, parse_test_module};
    use crate::printer::print_module;

    #[crate::dialect]
    mod asm_format_test {
        #[asm_format("call" callee "(" args ")" attr_dict (":" types(results)^)?)]
        fn invoke(callee: Attr<SymbolRef>, args: Variadic<_>) -> Variadic<_> {}

        #[asm_format(
            symbol(sym_name) "kind" kind ("inferred" inferred^)? attr_dict region(body)
        )]
        fn def(sym_name: Attr<String>, kind: Attr<String>, inferred: Option<Attr<bool>>) {
            #[region(body?)]
            {}
        }

        #[asm_format(tag ("note" note^)? ":" type(result))]
        fn tagged(tag: Attr<SymbolRef>, note: Option<Attr<String>>) -> Value<_> {}

        #[asm_format("to" successor(dest) "(" args ")")]
        fn branch(args: Variadic<_>) {
            #[successor(dest)]
            {}
        }

        #[asm_format(("on" on^)?)]
        fn toggle(on: Option<Attr<bool>>) {}

        #[asm_format("go" "(" ("" "on" on^)? ")")]
        fn run(on: Option<Attr<bool>>) {}

        #[asm_format(symbol(callee))]
        fn named(callee: Attr<String>) {}

        #[asm_format(flag "set")]
        fn bit(flag: Attr<bool>) {}

        #[asm_format(("quick" quick^)? "now" ("with" args^)? attr_dict)]
        fn mark(quick: Option<Attr<bool>>, args: Variadic<_>) {}

        #[asm_format("cases" "[" successors(dests) "]" (":" type(result)^)? regions(bodies))]
        fn switch() -> Option<Value<_>> {
            #[successors(dests)]
            {}
            #[regions(bodies)]
            {}
        }
    }

    /// Parse `text`, check that it prints as itself, and that the printed
    /// text reads back to the same text.
    fn assert_round_trip(text: &str) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        let printed = print_module(&ctx, module.op());
        assert_eq!(printed, text);
        let mut reparsed_ctx = IrContext::new();
        let reparsed = parse_test_module(&mut reparsed_ctx, &printed);
        assert_eq!(print_module(&reparsed_ctx, reparsed.op()), printed);
    }

    fn print(text: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, text);
        print_module(&ctx, module.op())
    }

    fn parse_error(text: &str) -> String {
        let mut ctx = IrContext::new();
        parse_module(&mut ctx, text)
            .map(|_| ())
            .expect_err("the text should not parse")
            .to_string()
    }

    #[test]
    fn formats_round_trip() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.def @f kind "pure" inferred attributes {extra = 1} {
      %0 = asm_format_test.tagged @f note "n" : core.i32
      %1 = asm_format_test.tagged @g : core.i32
      %2 = asm_format_test.invoke call @f(%0, %1) : core.i32
      asm_format_test.invoke call @g()
  }
  asm_format_test.def @decl kind "x"
}
"#,
        );
    }

    #[test]
    fn successors_round_trip() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.def @cfg kind "c" {
    ^bb0:
      asm_format_test.branch to ^bb1()

    ^bb1:
      asm_format_test.invoke call @g()
  }
}
"#,
        );
    }

    #[test]
    fn optional_groups_and_lists_round_trip() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.def @cfg kind "c" {
    ^bb0:
      asm_format_test.mark quick now
      asm_format_test.mark now
      %0 = asm_format_test.tagged @f : core.i32
      asm_format_test.mark now with %0
      asm_format_test.mark now attributes {extra = 1}
      asm_format_test.switch cases [^bb1, ^bb0] {
          asm_format_test.mark now
      } {
          asm_format_test.mark quick now
      }

    ^bb1:
      %1 = asm_format_test.switch cases [] : core.i32
  }
}
"#,
        );
    }

    #[test]
    fn a_format_beginning_with_an_absent_group_reads_generic_syntax() {
        let text = r#"core.module @m {
  asm_format_test.mark {quick = false}
  asm_format_test.toggle {on = false}
  asm_format_test.bit
  true.probe
}
"#;
        assert_eq!(print(text), text);
    }

    #[test]
    fn absent_elements_do_not_read_the_next_operation() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.def @f kind "k" {
      asm_format_test.toggle on
      asm_format_test.toggle
      asm_format_test.run go(on)
      asm_format_test.run go()
      asm_format_test.named @target
      asm_format_test.mark now
      attributes.probe
  }
}
"#,
        );
    }

    #[test]
    fn generic_syntax_reads_as_the_same_operation() {
        let generic = r#"core.module @m {
  %0 = asm_format_test.tagged {note = "n", tag = @f} : core.i32
  asm_format_test.def {kind = "k", sym_name = "d"}
}
"#;
        assert_eq!(
            print(generic),
            r#"core.module @m {
  %0 = asm_format_test.tagged @f note "n" : core.i32
  asm_format_test.def @d kind "k"
}
"#
        );
    }

    #[test]
    fn operations_the_format_cannot_spell_print_generically() {
        let text = r#"core.module @m {
  %0 = asm_format_test.tagged {extra = 1, tag = @f} : core.i32
  asm_format_test.def {inferred = false, kind = "k", sym_name = "d"}
  asm_format_test.def {sym_name = "e"}
}
"#;
        assert_eq!(print(text), text);
    }

    #[test]
    fn written_attributes_are_rejected_in_the_dictionary() {
        let error = parse_error(
            r#"core.module @m {
  asm_format_test.def @f kind "k" attributes {kind = "j"}
}"#,
        );
        assert!(
            error.contains("without the attributes the format writes"),
            "{error}"
        );
    }

    #[test]
    fn a_group_without_its_anchor_is_an_error() {
        let error = parse_error(
            r#"core.module @m {
  asm_format_test.mark now with
}"#,
        );
        assert!(error.contains("anchor"), "{error}");
    }

    #[test]
    fn a_mismatch_after_the_first_token_is_an_error() {
        let error = parse_error(
            r#"core.module @m {
  asm_format_test.invoke call @f(%0
}"#,
        );
        assert!(error.contains(r#"StringLiteral(")")"#), "{error}");
    }
}
