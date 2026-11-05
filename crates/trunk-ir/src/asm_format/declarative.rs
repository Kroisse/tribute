//! Declarative operation assembly formats.
//!
//! `#[asm_format(..)]` with format elements stores an [`OpFormat`] in the
//! operation's definition. The printer and the parser interpret it against
//! the definition's [`OpSchema`], so one declaration gives both directions
//! (`new-plans/ir.md`, "선언적 어셈블리 형식").

use std::borrow::Cow;
use std::fmt::{self, Write};
use std::ops::Range;

use itertools::Itertools as _;
use winnow::combinator::{delimited, opt, preceded, repeat, separated};
use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
use winnow::prelude::*;

use crate::asm_format::{
    CustomEntity, OpAsmDirectiveHooks, RawCustomEntity, SignatureCodecHooks, SignatureParts,
    expected, parse_signature_results, write_signature_param, write_signature_results,
};
use crate::op_def::OpDef;
use crate::op_flag::IsolatedFromAbove;
use crate::op_schema::{Arity, OpSchema, ResultSchema};
use crate::parser::raw::{
    RawAttribute, RawOperation, RawParam, RawRegion, RawType, block_label, ident, raw_attr_dict,
    raw_param, raw_param_list, raw_region, raw_type, symbol_ref, value_ref, ws,
};
use crate::printer::{AsmNamer, OpAsmPrinter, display_symbol_text};
use crate::types::Attribute;
use crate::{BlockRef, IrContext, OpRef, TypeRef};

/// The format elements of an operation, in order.
#[derive(Debug)]
pub struct OpFormat {
    pub elements: &'static [FormatElement],
}

/// One element of an operation format. Indices refer to the entities of the
/// operation's [`OpSchema`].
#[derive(Debug)]
pub enum FormatElement {
    /// A keyword or punctuation written as is.
    Literal(&'static str),
    /// An operand: `%name`, or a comma-separated list for a variadic one.
    Operand(usize),
    /// An attribute: the keyword its kind's codec spells, if the kind has
    /// one, or generic attribute syntax.
    Attribute(usize),
    /// A string attribute written as a definition name, `@name`.
    Symbol(usize),
    /// The type of a single operand, which the parser checks against the
    /// operand's definition.
    OperandType(usize),
    /// The type of one fixed result.
    ResultType(usize),
    /// The comma-separated types of every result, for a variadic or optional
    /// result.
    ResultTypes,
    /// The entry block arguments of a fixed region, as `(%a: A, ..)`; the
    /// region elides its entry block.
    RegionArgs(usize),
    /// The types of an operand and of the results as a function type,
    /// `(A, B) -> R`, `(A) -> (R, S)`, or `(A) -> ()`.
    FunctionalType(usize),
    /// The attributes no other element writes, as `attributes {..}`, or as
    /// `{..}` when `bare`.
    AttrDict { bare: bool },
    /// A region in generic form, or each region of a variadic list. A region
    /// a [`Signature`](Self::Signature) lists elides its entry block.
    Region(usize),
    /// A successor's label, or a comma-separated list for a variadic one.
    Successor(usize),
    /// A signature type in function form, `(%a: A) -> R`, whose inputs are
    /// the entry block arguments of the listed regions, or `(A) -> R` when
    /// it lists none. With `results`, the signature's results are the
    /// operation's result types.
    Signature {
        source: SignatureSource,
        regions: &'static [usize],
        codec: SignatureCodecHooks,
        results: bool,
    },
    /// The entities a [`OpAsmDirective`](crate::asm_format::OpAsmDirective)
    /// writes by hand, in order.
    Custom {
        hooks: OpAsmDirectiveHooks,
        entities: &'static [CustomSlot],
    },
    /// Elements written only when their anchor is present.
    Optional {
        anchor: FormatAnchor,
        elements: &'static [FormatElement],
    },
}

/// An entity a [`FormatElement::Custom`] writes.
#[derive(Clone, Copy, Debug)]
pub enum CustomSlot {
    Operand(usize),
    Attribute(usize),
    Successor(usize),
    Region(usize),
}

/// The entities a custom element names, as its directive prints them.
fn custom_entities<'c>(
    ctx: &'c IrContext,
    op: OpRef,
    schema: &OpSchema,
    slots: &[CustomSlot],
) -> Vec<CustomEntity<'c>> {
    slots
        .iter()
        .map(|&slot| match slot {
            CustomSlot::Operand(index) => {
                let operands = ctx.op_operands(op);
                CustomEntity::Operands(&operands[operand_range(schema, index, operands.len())])
            }
            CustomSlot::Attribute(index) => {
                CustomEntity::Attribute(ctx.op_attrs(op).get(schema.attributes[index].name))
            }
            CustomSlot::Successor(index) => {
                let range = successor_range(schema, index, ctx.op_successor_count(op));
                CustomEntity::Successors(
                    range
                        .filter_map(|position| ctx.op_successor(op, position))
                        .collect(),
                )
            }
            CustomSlot::Region(index) => {
                let range = region_range(schema, index, ctx.op_region_count(op));
                CustomEntity::Regions(
                    range
                        .filter_map(|position| ctx.op_region(op, position))
                        .collect(),
                )
            }
        })
        .collect()
}

/// What a [`FormatElement::Signature`] writes the type of.
#[derive(Clone, Copy, Debug)]
pub enum SignatureSource {
    /// A type attribute, which the signature writes.
    Attribute(usize),
    /// A fixed operand, which another element writes; the signature writes
    /// only its type.
    Operand(usize),
    /// The single result, whose type the signature writes.
    Result,
}

/// The type a signature element writes, if `op` has it.
fn signature_type(
    ctx: &IrContext,
    op: OpRef,
    schema: &OpSchema,
    source: SignatureSource,
) -> Option<crate::TypeRef> {
    match source {
        SignatureSource::Attribute(index) => {
            ctx.op_attrs(op).get_type(schema.attributes[index].name)
        }
        SignatureSource::Operand(index) => {
            let operands = ctx.op_operands(op);
            let range = operand_range(schema, index, operands.len());
            (range.len() == 1).then(|| ctx.value_ty(operands[range.start]))
        }
        SignatureSource::Result => match ctx.op_result_types(op) {
            [ty] => Some(*ty),
            _ => None,
        },
    }
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
    p: &mut OpAsmPrinter<'_>,
    op: OpRef,
    def: &'static OpDef,
    format: &OpFormat,
) -> fmt::Result {
    let ctx = p.ctx();
    if !printable(ctx, op, &def.schema, format) {
        return p.print_generic(op);
    }
    let results = ctx.op_results(op);
    if !results.is_empty() {
        write!(p, "{} = ", p.state().values(results))?;
    }
    write!(p, "{}", ctx.op(op).name.display(ctx))?;
    Printer {
        p,
        op,
        schema: &def.schema,
        format,
        spacing: Spacing::Space,
    }
    .elements(format.elements)
}

/// Name the values and blocks [`print`] writes, in the order it writes them.
pub(crate) fn assign_names(
    namer: &mut AsmNamer<'_>,
    op: OpRef,
    def: &'static OpDef,
    format: &OpFormat,
) {
    if !printable(namer.ctx(), op, &def.schema, format) {
        return namer.generic_op(op);
    }
    namer.results(op);
    let mut naming = Naming {
        op,
        schema: &def.schema,
        format,
    };
    if def.has_flag::<IsolatedFromAbove>() {
        namer.fresh_scope(|namer| naming.elements(namer, format.elements));
    } else {
        naming.elements(namer, format.elements);
    }
}

/// The naming counterpart of [`Printer`].
struct Naming<'f> {
    op: OpRef,
    schema: &'static OpSchema,
    format: &'f OpFormat,
}

impl Naming<'_> {
    fn elements(&mut self, namer: &mut AsmNamer<'_>, elements: &[FormatElement]) {
        for element in elements {
            self.element(namer, element);
        }
    }

    fn element(&mut self, namer: &mut AsmNamer<'_>, element: &FormatElement) {
        let ctx = namer.ctx();
        let op = self.op;
        match *element {
            FormatElement::Region(index) => {
                let count = ctx.op_region_count(op);
                for position in region_range(self.schema, index, count) {
                    let region = ctx
                        .op_region(op, position)
                        .expect("a region within the counted list");
                    if signature_lists(self.format.elements, index) {
                        namer.region_without_entry(region);
                    } else {
                        namer.region(region);
                    }
                }
            }
            FormatElement::Signature {
                source,
                regions,
                codec,
                ..
            } if !regions.is_empty() => {
                let ty = signature_type(ctx, op, self.schema, source)
                    .expect("a printable signature is a type");
                let parts = (codec.decompose)(ctx, ty).expect("a printable signature decomposes");
                let entries = signature_entries(ctx, op, self.schema, regions);
                if let Some((&first, rest)) = entries.split_first() {
                    let args = &ctx.block_args(first)[..parts.inputs.len()];
                    namer.values(args);
                    for &other in rest {
                        namer.share(&ctx.block_args(other)[..parts.inputs.len()], args);
                    }
                }
            }
            FormatElement::RegionArgs(index) => {
                let region = ctx
                    .op_region(op, index)
                    .expect("a printable region with entry arguments is present");
                namer.entry_args(region);
            }
            FormatElement::Custom { hooks, entities } => {
                let entities = custom_entities(ctx, op, self.schema, entities);
                (hooks.assign_names)(namer, &entities);
            }
            FormatElement::Optional { anchor, elements }
                if anchor_present(ctx, op, self.schema, anchor) =>
            {
                self.elements(namer, elements);
            }
            _ => {}
        }
    }
}

/// Whether the space before the next token is written.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Spacing {
    Space,
    Glue,
    /// After punctuation, which `(` does not glue to.
    Punctuation,
}

struct Printer<'p, 'a> {
    p: &'p mut OpAsmPrinter<'a>,
    op: OpRef,
    schema: &'static OpSchema,
    format: &'p OpFormat,
    spacing: Spacing,
}

impl Printer<'_, '_> {
    /// Start a token whose text begins with `first`, writing the space
    /// before it unless the spacing rules drop it.
    fn token(&mut self, first: char) -> fmt::Result {
        let glued = match first {
            ')' | ']' | ',' => true,
            '(' => self.spacing != Spacing::Punctuation,
            _ => false,
        };
        if self.spacing != Spacing::Glue && !glued {
            self.p.write_char(' ')?;
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
        let state = self.p.state();
        let ctx = state.ctx();
        let op = self.op;
        match *element {
            FormatElement::Literal(text) => {
                self.token(text.chars().next().unwrap_or(' '))?;
                self.p.write_str(text)?;
                if text.ends_with(['(', '[']) {
                    self.spacing = Spacing::Glue;
                } else if !is_keyword(text) {
                    self.spacing = Spacing::Punctuation;
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
                write!(self.p, "{}", state.values(&operands[range]))
            }
            FormatElement::Attribute(index) => {
                let attribute = &self.schema.attributes[index];
                let Some(value) = ctx.op_attrs(op).get(attribute.name) else {
                    return Ok(());
                };
                self.token('a')?;
                (attribute.print_text)(self.p, value)
            }
            FormatElement::Symbol(index) => {
                let name = self.schema.attributes[index].name;
                let text = ctx
                    .op_attrs(op)
                    .get_str(name)
                    .expect("a printable symbol attribute is a string");
                self.token('@')?;
                write!(self.p, "{}", display_symbol_text(text))
            }
            FormatElement::OperandType(index) => {
                let operands = ctx.op_operands(op);
                let range = operand_range(self.schema, index, operands.len());
                self.token('t')?;
                write!(self.p, "{}", state.ty(ctx.value_ty(operands[range.start])))
            }
            FormatElement::ResultType(index) => {
                let ty = ctx.op_result_types(op)[index];
                self.token('t')?;
                write!(self.p, "{}", state.ty(ty))
            }
            FormatElement::ResultTypes => {
                let types = ctx.op_result_types(op);
                if types.is_empty() {
                    return Ok(());
                }
                self.token('t')?;
                write!(
                    self.p,
                    "{}",
                    types.iter().map(|&ty| state.ty(ty)).format(", ")
                )
            }
            FormatElement::AttrDict { bare } => {
                let written = written_attributes(self.schema, self.format);
                let schema = self.schema;
                let mut rest = ctx
                    .op(op)
                    .attributes
                    .iter()
                    .filter(move |&(key, value)| {
                        let key = ctx.str(key);
                        !written.contains(&key) && !schema.holds_default(ctx, key, value)
                    })
                    .peekable();
                if rest.peek().is_none() {
                    return Ok(());
                }
                if bare {
                    self.token('{')?;
                } else {
                    self.token('a')?;
                    self.p.write_str("attributes ")?;
                }
                write!(self.p, "{}", state.attr_dict(rest))
            }
            FormatElement::Region(index) => {
                let count = ctx.op_region_count(op);
                for position in region_range(self.schema, index, count) {
                    let region = ctx
                        .op_region(op, position)
                        .expect("a region within the counted list");
                    self.token('{')?;
                    if signature_lists(self.format.elements, index) {
                        self.p.print_region_without_entry(region)?;
                    } else {
                        self.p.print_region(region)?;
                    }
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
                let labels = range.map(|successor| {
                    let block = ctx
                        .op_successor(op, successor)
                        .expect("a successor within the counted list");
                    state.block(block)
                });
                write!(self.p, "{}", labels.format(", "))
            }
            FormatElement::Custom { hooks, entities } => {
                let entities = custom_entities(ctx, op, self.schema, entities);
                if hooks.spaced {
                    self.token('c')?;
                } else {
                    self.spacing = Spacing::Space;
                }
                (hooks.print)(self.p, &entities)
            }
            FormatElement::Signature {
                source,
                regions,
                codec,
                ..
            } => {
                let ty = signature_type(ctx, op, self.schema, source)
                    .expect("a printable signature is a type");
                let parts = (codec.decompose)(ctx, ty).expect("a printable signature decomposes");
                let entries = signature_entries(ctx, op, self.schema, regions);
                self.token('(')?;
                self.p.write_char('(')?;
                for (position, &(ty, attrs)) in parts.inputs.iter().enumerate() {
                    if position > 0 {
                        self.p.write_str(", ")?;
                    }
                    // A signature without regions writes its input types alone.
                    if !regions.is_empty() {
                        match entries.first() {
                            Some(&first) => {
                                let arg = ctx.block_args(first)[position];
                                write!(self.p, "{}", state.value(arg))?;
                            }
                            None => write!(self.p, "%arg{position}")?,
                        }
                        self.p.write_str(": ")?;
                    }
                    write_signature_param(self.p, ty, attrs)?;
                }
                self.p.write_char(')')?;
                // A function's header drops `-> ()`; a function type keeps it.
                if !regions.is_empty() && parts.results.is_empty() && parts.tail.is_empty() {
                    return Ok(());
                }
                self.p.write_str(" -> ")?;
                write_signature_results(self.p, &parts.results)?;
                (codec.write_tail)(self.p, &parts.tail)
            }
            FormatElement::FunctionalType(index) => {
                let operands = ctx.op_operands(op);
                let range = operand_range(self.schema, index, operands.len());
                self.token('(')?;
                let inputs = operands[range]
                    .iter()
                    .map(|&value| state.ty(ctx.value_ty(value)))
                    .format(", ");
                write!(self.p, "({inputs}) -> ")?;
                let results = ctx.op_result_types(op);
                let types = results.iter().map(|&ty| state.ty(ty)).format(", ");
                if results.len() == 1 {
                    write!(self.p, "{types}")?;
                } else {
                    write!(self.p, "({types})")?;
                }
                self.spacing = Spacing::Space;
                Ok(())
            }
            FormatElement::RegionArgs(index) => {
                let region = ctx
                    .op_region(op, index)
                    .expect("a printable region with entry arguments is present");
                self.token('(')?;
                write!(self.p, "{}", state.entry_args(region))
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

/// Whether an element of the format writes the entry block arguments of
/// region `index`, which then elides its entry block.
fn signature_lists(elements: &[FormatElement], index: usize) -> bool {
    let mut listed = false;
    visit(elements, &mut |element| match element {
        FormatElement::Signature { regions, .. } => listed |= regions.contains(&index),
        FormatElement::RegionArgs(region) => listed |= *region == index,
        _ => {}
    });
    listed
}

/// The attributes the `signature` elements of an operation's declarative
/// format write in the function form.
pub(crate) fn signature_attributes(def: &OpDef) -> Vec<&'static str> {
    let Some(super::OpAsmFormatDef::Declarative(format)) = def.asm_format else {
        return Vec::new();
    };
    let mut names = Vec::new();
    visit(format.elements, &mut |element| {
        if let FormatElement::Signature {
            source: SignatureSource::Attribute(attribute),
            ..
        } = *element
        {
            names.push(def.schema.attributes[attribute].name);
        }
    });
    names
}

/// The entry blocks of the present regions a `signature` element lists, or
/// `None` for a present region without blocks.
fn signature_entry_blocks(
    ctx: &IrContext,
    op: OpRef,
    schema: &OpSchema,
    regions: &[usize],
) -> Option<Vec<BlockRef>> {
    let count = ctx.op_region_count(op);
    regions
        .iter()
        .flat_map(|&index| region_range(schema, index, count))
        .map(|position| {
            let region = ctx.op_region(op, position)?;
            ctx.region(region).blocks.first().copied()
        })
        .collect()
}

/// The entry blocks of a printable signature's regions.
fn signature_entries(
    ctx: &IrContext,
    op: OpRef,
    schema: &OpSchema,
    regions: &[usize],
) -> Vec<BlockRef> {
    signature_entry_blocks(ctx, op, schema, regions)
        .expect("a printable signature's regions have entry blocks")
}

/// The names of the attributes a format writes outside its attribute
/// dictionary.
fn written_attributes(schema: &OpSchema, format: &OpFormat) -> Vec<&'static str> {
    fn collect(schema: &OpSchema, elements: &[FormatElement], names: &mut Vec<&'static str>) {
        for element in elements {
            match *element {
                FormatElement::Attribute(index)
                | FormatElement::Symbol(index)
                | FormatElement::Signature {
                    source: SignatureSource::Attribute(index),
                    ..
                } => names.push(schema.attributes[index].name),
                FormatElement::Custom { entities, .. } => {
                    names.extend(entities.iter().filter_map(|&slot| match slot {
                        CustomSlot::Attribute(index) => Some(schema.attributes[index].name),
                        _ => None,
                    }));
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
            let attribute = &schema.attributes[index];
            ctx.op_attrs(op)
                .get(attribute.name)
                .is_some_and(|value| !attribute.holds_default(ctx, value))
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
        || !fixed_results_fit(ctx, op, schema, format)
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
                Some(value) => (attribute.accepts)(ctx, value),
                None => attribute.optional,
            };
        }
        FormatElement::Symbol(index) => {
            let attribute = &schema.attributes[index];
            written_ok &= matches!(attrs.get(attribute.name), Some(Attribute::String(_)));
        }
        FormatElement::Custom { hooks, entities } => {
            written_ok &= (hooks.fits)(ctx, &custom_entities(ctx, op, schema, entities));
        }
        FormatElement::Signature {
            source,
            regions,
            codec,
            results,
        } => {
            written_ok &= signature_type(ctx, op, schema, source)
                .and_then(|ty| (codec.decompose)(ctx, ty))
                .is_some_and(|parts| {
                    signature_fits(ctx, op, schema, regions, &parts)
                        && (!results
                            || ctx
                                .op_result_types(op)
                                .iter()
                                .eq(parts.results.iter().map(|(ty, _)| ty)))
                });
        }
        FormatElement::RegionArgs(index) => {
            written_ok &= ctx
                .op_region(op, index)
                .is_some_and(|region| !ctx.region(region).blocks.is_empty());
        }
        FormatElement::Optional {
            anchor: FormatAnchor::Flag(index),
            ..
        } => {
            let attribute = &schema.attributes[index];
            written_ok &= match attrs.get(attribute.name) {
                None | Some(Attribute::Bool(true)) => true,
                Some(value) => attribute.holds_default(ctx, value),
            };
        }
        _ => {}
    });
    if !written_ok {
        return false;
    }
    let has_dict = {
        let mut found = false;
        visit(format.elements, &mut |element| {
            found |= matches!(element, FormatElement::AttrDict { .. })
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

/// Whether every present region a signature lists receives its inputs as
/// entry block arguments.
fn signature_fits(
    ctx: &IrContext,
    op: OpRef,
    schema: &OpSchema,
    regions: &[usize],
    parts: &SignatureParts<'_>,
) -> bool {
    signature_entry_blocks(ctx, op, schema, regions).is_some_and(|entries| {
        entries.iter().all(|&entry| {
            let args = ctx.block_args(entry);
            args.len() == parts.inputs.len()
                && args
                    .iter()
                    .zip(&parts.inputs)
                    .all(|(&arg, &(ty, _))| ctx.value_ty(arg) == ty)
        })
    })
}

/// Whether the result types a format does not write are the ones the schema
/// fixes, so that the parser restores them.
fn fixed_results_fit(ctx: &IrContext, op: OpRef, schema: &OpSchema, format: &OpFormat) -> bool {
    if !leaves_results_unwritten(schema, format) {
        return true;
    }
    let results = ctx.op_result_types(op);
    // A fixed type is the bare type of its name, so a result with type
    // parameters or attributes would not survive the round trip.
    let bare = |ty: TypeRef| {
        let data = ctx.get_type(ty);
        data.params.is_empty() && data.attrs.is_empty()
    };
    schema.fixed_results().is_some_and(|bounds| {
        bounds.len() == results.len()
            && bounds
                .iter()
                .zip(results.iter())
                .all(|(bound, &ty)| (bound.matches)(ctx, ty) && bare(ty))
    })
}

/// Whether `format` leaves the operation's result types unwritten, which the
/// macro accepts only when the schema fixes them.
fn leaves_results_unwritten(schema: &OpSchema, format: &OpFormat) -> bool {
    let has_results = match schema.results {
        ResultSchema::Fixed(names) => !names.is_empty(),
        ResultSchema::Optional(_) | ResultSchema::Variadic(_) => true,
    };
    let mut written = false;
    visit(format.elements, &mut |element| {
        written |= matches!(
            element,
            FormatElement::ResultType(_)
                | FormatElement::ResultTypes
                | FormatElement::FunctionalType(_)
                | FormatElement::Signature { results: true, .. }
                | FormatElement::Signature {
                    source: SignatureSource::Result,
                    ..
                }
        );
    });
    has_results && !written
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
    /// The types written for an operand entity, by its schema position.
    operand_types: Vec<(usize, Vec<RawType<'a>>)>,
    attributes: Vec<(Cow<'a, str>, RawAttribute<'a>)>,
    result_types: Vec<Vec<RawType<'a>>>,
    regions: Vec<Vec<RawRegion<'a>>>,
    successors: Vec<Vec<&'a str>>,
}

/// Parse an operation written with `format` after its `dialect.name`.
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
        signature_regions: Vec::new(),
        entry_args: Vec::new(),
        parsed: Parsed {
            operands: vec![Vec::new(); schema.operands.len()],
            operand_types: Vec::new(),
            attributes: Vec::new(),
            result_types: vec![Vec::new(); result_slots],
            regions: vec![Vec::new(); schema.regions.len()],
            successors: vec![Vec::new(); schema.successors.len()],
        },
    };
    for element in format.elements {
        parser.element(input, element)?;
    }
    let Parsed {
        operands,
        operand_types: written,
        attributes,
        result_types,
        regions,
        successors,
    } = parser.parsed;
    // A signature's inputs are the entry block arguments of its regions.
    let mut func_param_regions = Vec::new();
    let mut position = 0;
    for (index, parsed) in regions.iter().enumerate() {
        if parser.signature_regions.contains(&index) {
            func_param_regions.extend(position..position + parsed.len());
        }
        position += parsed.len();
    }
    let mut operand_types = Vec::new();
    for (index, types) in written {
        if types.len() != operands[index].len() {
            return Err(ErrMode::Cut(expected("one type for each operand")));
        }
        operand_types.extend(operands[index].iter().copied().zip(types));
    }
    Ok(RawOperation {
        results,
        name,
        sym_name: None,
        func_params: parser.entry_args,
        func_param_regions,
        operands: operands.into_iter().flatten().collect(),
        operand_types,
        attributes,
        result_types: result_types.into_iter().flatten().collect(),
        fixed_results: leaves_results_unwritten(schema, format),
        regions: regions.into_iter().flatten().collect(),
        successors: successors.into_iter().flatten().collect(),
    })
}

struct FormatParser<'a> {
    schema: &'static OpSchema,
    format: &'static OpFormat,
    /// The regions a signature lists, which take `entry_args` as their
    /// entry block arguments.
    signature_regions: Vec<usize>,
    entry_args: Vec<(&'a str, RawType<'a>)>,
    parsed: Parsed<'a>,
}

impl<'a> FormatParser<'a> {
    fn element(&mut self, input: &mut &'a str, element: &FormatElement) -> ModalResult<()> {
        let before = *input;
        ws.parse_next(input)?;
        let new_line = before[..before.len() - input.len()].contains('\n');
        match *element {
            FormatElement::Literal(text) => literal(input, text),
            FormatElement::Operand(index) => {
                let values = if self.schema.operands[index].arity == Arity::Variadic {
                    // A list is written on one line, so an operand after a
                    // line break begins the next operation.
                    if new_line {
                        Vec::new()
                    } else {
                        let first = opt(value_ref).parse_next(input)?;
                        let mut values: Vec<_> = first.into_iter().collect();
                        if !values.is_empty() {
                            let rest: Vec<_> = repeat(0.., preceded((ws, ',', ws), value_ref))
                                .parse_next(input)?;
                            values.extend(rest);
                        }
                        values
                    }
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
                let value = (self.schema.attributes[index].parse_text)(input)?;
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
            FormatElement::Custom { hooks, entities } => {
                let parsed = (hooks.parse)(input)?;
                if parsed.len() != entities.len() {
                    return Err(ErrMode::Cut(expected(
                        "one entity for each the custom element names",
                    )));
                }
                for (&slot, entity) in entities.iter().zip(parsed) {
                    match (slot, entity) {
                        (CustomSlot::Operand(index), RawCustomEntity::Operands(values)) => {
                            self.parsed.operands[index] = values;
                        }
                        (CustomSlot::Attribute(index), RawCustomEntity::Attribute(value)) => {
                            if let Some(value) = value {
                                self.attribute(index, value);
                            }
                        }
                        (CustomSlot::Successor(index), RawCustomEntity::Successors(labels)) => {
                            self.parsed.successors[index] = labels;
                        }
                        (CustomSlot::Region(index), RawCustomEntity::Regions(regions)) => {
                            self.parsed.regions[index] = regions;
                        }
                        _ => {
                            return Err(ErrMode::Cut(expected(
                                "the kind of entity the custom element names",
                            )));
                        }
                    }
                }
                Ok(())
            }
            FormatElement::OperandType(index) => {
                let ty = raw_type
                    .parse_next(input)
                    .map_err(|e| e.map(|_| expected("operand type")))?;
                self.parsed.operand_types.push((index, vec![ty]));
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
            FormatElement::AttrDict { bare } => {
                let dict = if bare {
                    if input.starts_with('{') {
                        Some(raw_attr_dict.parse_next(input)?)
                    } else {
                        None
                    }
                } else if literal(input, "attributes").is_ok() {
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
            FormatElement::FunctionalType(index) => {
                let mut list = |input: &mut &'a str| {
                    delimited(
                        ('(', ws),
                        separated(0.., (ws, raw_type, ws).map(|(_, t, _)| t), ','),
                        (ws, ')'),
                    )
                    .parse_next(input)
                };
                let types: Vec<RawType<'a>> = list
                    .parse_next(input)
                    .map_err(|e| e.map(|_| expected("operand types `(A, ..)`")))?;
                self.parsed.operand_types.push((index, types));
                (ws, "->", ws).parse_next(input)?;
                let results: Vec<RawType<'a>> = if input.starts_with('(') {
                    list.parse_next(input)?
                } else {
                    vec![
                        raw_type
                            .parse_next(input)
                            .map_err(|e| e.map(|_| expected("result type")))?,
                    ]
                };
                match self.parsed.result_types.first_mut() {
                    Some(slot) => *slot = results,
                    None if results.is_empty() => {}
                    None => return Err(ErrMode::Cut(expected("`-> ()` for no results"))),
                }
                Ok(())
            }
            FormatElement::RegionArgs(index) => {
                let args = super::parse_entry_args(input)
                    .map_err(|e| e.map(|_| expected("entry arguments `(%a: A, ..)`")))?;
                self.signature_regions = vec![index];
                self.entry_args = args;
                Ok(())
            }
            FormatElement::Signature {
                source,
                regions,
                codec,
                results: writes_results,
            } => {
                if !input.starts_with('(') {
                    return Err(ErrMode::Backtrack(expected("a signature `(..)`")));
                }
                let params: Vec<(Option<&'a str>, RawParam<'a>)> = if regions.is_empty() {
                    raw_param_list
                        .parse_next(input)?
                        .into_iter()
                        .map(|param| (None, param))
                        .collect()
                } else {
                    delimited(
                        ('(', ws),
                        separated(
                            0..,
                            (ws, value_ref, ws, ':', ws, raw_param, ws)
                                .map(|(_, name, _, _, _, param, _)| (Some(name), param)),
                            ',',
                        ),
                        (ws, ')'),
                    )
                    .parse_next(input)?
                };
                let (results, tail) = if opt((ws, "->", ws)).parse_next(input)?.is_some() {
                    let results = parse_signature_results(input)?;
                    (results, (codec.parse_tail)(input)?)
                } else {
                    (Vec::new(), Vec::new())
                };
                if writes_results {
                    self.parsed.result_types[0] =
                        results.iter().map(|result| result.ty.clone()).collect();
                }
                let args = params
                    .iter()
                    .filter_map(|(name, param)| Some(((*name)?, param.ty.clone())))
                    .collect();
                let inputs = params.into_iter().map(|(_, param)| param).collect();
                let ty = (codec.build)(inputs, results, tail)?;
                match source {
                    SignatureSource::Attribute(attribute) => {
                        self.attribute(attribute, RawAttribute::Type(ty));
                    }
                    // The operand's definition gives its type.
                    SignatureSource::Operand(_) => {}
                    SignatureSource::Result => self.parsed.result_types[0] = vec![ty],
                }
                self.signature_regions = regions.to_vec();
                self.entry_args = args;
                Ok(())
            }
            FormatElement::Optional { anchor, elements } => {
                // A group begins with its keyword, which tells whether it is
                // present.
                let Some((first, rest)) = elements.split_first() else {
                    return Ok(());
                };
                match self.element(input, first) {
                    Ok(()) => {}
                    Err(ErrMode::Backtrack(_)) => return Ok(()),
                    Err(error) => return Err(error),
                }
                if let FormatAnchor::Flag(index) = anchor {
                    self.attribute(index, RawAttribute::Bool(true));
                }
                rest.iter().try_for_each(|element| {
                    self.element(input, element).map_err(|error| error.cut())
                })?;
                // The printer writes a group only for a present anchor.
                let present = match anchor {
                    FormatAnchor::Attribute(_) | FormatAnchor::Flag(_) => true,
                    FormatAnchor::Operand(index) => !self.parsed.operands[index].is_empty(),
                    FormatAnchor::Results => !self.parsed.result_types[0].is_empty(),
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

/// Whether a literal is a keyword, read as a whole identifier.
fn is_keyword(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
}

/// Read `text`; a keyword must not continue as a longer identifier or as an
/// operation name.
fn literal(input: &mut &str, text: &'static str) -> ModalResult<()> {
    let error = || {
        let mut error = ContextError::new();
        error.push(StrContext::Expected(StrContextValue::StringLiteral(text)));
        ErrMode::Backtrack(error)
    };
    let is_word = is_keyword(text);
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

#[cfg(test)]
mod tests {
    use crate::IrContext;
    use crate::attr_kind::SymbolRef;
    use crate::dialect::core::I1;
    use crate::dialect::func::FuncSig;
    use crate::op_flag::IsolatedFromAbove;
    use crate::ops::DialectOp;
    use crate::parser::{parse_module, parse_test_module};
    use crate::printer::print_module;
    use crate::types::Attribute;

    /// A speed stored as a flag, `true` for `fast`, written as its keyword.
    struct Speed;

    impl crate::attr_kind::AttrKind for Speed {
        const KIND: crate::op_schema::AttributeKind = crate::op_schema::AttributeKind::Bool;
        type Out<'ctx> = bool;
        type In = bool;
        fn read<'ctx>(_: &'ctx IrContext, attr: &'ctx Attribute) -> bool {
            matches!(attr, Attribute::Bool(true))
        }

        fn write(_: &mut IrContext, fast: bool) -> Attribute {
            Attribute::Bool(fast)
        }

        fn print_text(
            p: &mut crate::printer::AsmPrinter<'_>,
            attr: &Attribute,
        ) -> std::fmt::Result {
            std::fmt::Write::write_str(
                p,
                if matches!(attr, Attribute::Bool(true)) {
                    "fast"
                } else {
                    "slow"
                },
            )
        }

        fn parse_text<'a>(
            input: &mut &'a str,
        ) -> winnow::ModalResult<crate::parser::raw::RawAttribute<'a>> {
            crate::asm_format::parse_keyword(input, |word| match word {
                "fast" => Some(crate::parser::raw::RawAttribute::Bool(true)),
                "slow" => Some(crate::parser::raw::RawAttribute::Bool(false)),
                _ => None,
            })
        }
    }

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

        #[asm_format("ref" tag ("note" note^)? ":" type(result))]
        fn tagged(tag: Attr<SymbolRef>, note: Option<Attr<String>>) -> Value<_> {}

        #[asm_format("to" successor(dest) "(" args ")")]
        fn branch(args: Variadic<_>) {
            #[successor(dest)]
            {}
        }

        #[asm_format("turn" ("on" on^)?)]
        fn toggle(on: Option<Attr<bool>>) {}

        #[asm_format("go" "(" ("on" on^)? ")")]
        fn run(on: Option<Attr<bool>>) {}

        #[asm_format(symbol(callee))]
        fn named(callee: Attr<String>) {}

        #[asm_format(target ("note" note^)? ":" type(result))]
        fn reference(target: Attr<SymbolRef>, note: Option<Attr<String>>) -> Value<_> {}

        #[asm_format("now" ("quick" quick^)? ("with" "(" args^ ")")? attr_dict)]
        fn mark(quick: Option<Attr<bool>>, args: Variadic<_>) {}

        #[asm_format("cases" "[" successors(dests) "]" regions(bodies) (":" type(result)^)?)]
        fn switch() -> Option<Value<_>> {
            #[successors(dests)]
            {}
            #[regions(bodies)]
            {}
        }

        #[asm_format(symbol(sym_name) signature(r#type, body) attr_dict region(body))]
        #[flags(IsolatedFromAbove)]
        fn fun<S: FuncSig>(sym_name: Attr<String>, r#type: Attr<TypeOf<S>>) {
            #[region(body?)]
            {}
        }

        #[asm_format(
            symbol(sym_name) signature(r#type, first, second) attr_dict region(first)
            region(second)
        )]
        #[flags(IsolatedFromAbove)]
        fn both<S: FuncSig>(sym_name: Attr<String>, r#type: Attr<TypeOf<S>>) {
            #[region(first)]
            {}
            #[region(second)]
            {}
        }

        #[asm_format("values" ("n" n^)? ("label" label^)? attr_dict)]
        fn pick(#[default(3)] n: Attr<u32>, #[default("none")] label: Attr<String>) {}

        fn scale(#[default(0.0)] factor: Attr<f64>, #[default(0.0)] ratio: Attr<f32>) {}

        #[asm_format("go" "(" speed ")" attr_dict)]
        fn drive(speed: Attr<Speed>) {}

        #[asm_format(lhs "," rhs attr_dict ":" type(result))]
        fn add(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}

        #[asm_format(value ":" type(value) attr_dict)]
        fn consume(value: Value<_>) {}

        #[asm_format(lhs "," rhs attr_dict_bare ":" type(lhs))]
        fn compare<T>(lhs: Value<T>, rhs: Value<T>) -> Value<I1> {}

        struct Token;

        #[asm_format(attr_dict)]
        fn make_token() -> Value<Token> {}

        #[asm_format(args attr_dict_bare (":" types(results)^)?)]
        fn emit(args: Variadic<_>) -> Variadic<_> {}

        #[asm_format(callee "(" args ")" attr_dict ":" functional_type(args, results))]
        fn call_fn(callee: Attr<SymbolRef>, args: Variadic<_>) -> Variadic<_> {}

        #[asm_format(callee "(" args ")" attr_dict ":" functional_type(args))]
        fn tail_fn(callee: Attr<SymbolRef>, args: Variadic<_>) {}

        #[asm_format("try" region(body) "catch" args(handler) region(handler) attr_dict)]
        fn guard() {
            #[region(body)]
            {}
            #[region(handler)]
            {}
        }

        #[asm_format(callee "(" args ")" attr_dict ":" signature(signature))]
        fn apply<S: FuncSig>(
            signature: Attr<TypeOf<S>>,
            callee: Value<_>,
            args: Variadic<_>,
        ) -> Variadic<_> {
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
    fn signatures_round_trip() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.fun @add(%0: core.i32, %1: core.i32) -> core.i32 {
      %2 = asm_format_test.invoke call @add(%0, %1) : core.i32
      asm_format_test.invoke call @ret(%2)
  }
  asm_format_test.fun @decl(%arg0: core.i32, %arg1: core.i1)
  asm_format_test.fun @none()
  asm_format_test.fun @param_attrs(%0: core.i32 {k = 1}) -> (core.i32 {r = 2}) {
      asm_format_test.invoke call @ret(%0)
  }
  asm_format_test.fun @tail(%0: core.i32) -> (), {call_conv = "tail"} attributes {extra = 1} {
      asm_format_test.invoke call @ret()
  }
  asm_format_test.fun @blocks(%0: core.i32) {
      asm_format_test.branch to ^bb1(%0)
    ^bb1(%1: core.i32):
      asm_format_test.invoke call @ret(%1)
  }
}
"#,
        );
    }

    #[test]
    fn every_region_a_signature_lists_takes_its_parameters() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.both @b(%0: core.i32) -> core.i32 {
      asm_format_test.invoke call @ret(%0)
  } {
      asm_format_test.invoke call @ret(%0)
  }
}
"#,
        );
    }

    #[test]
    fn calls_write_the_callee_and_the_signature_as_a_function_type() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.ptr, %1: core.i32) -> core.i32 {
      %2 = asm_format_test.apply %0(%1) : (core.i32) -> core.i32
      asm_format_test.apply %0() : () -> ()
      asm_format_test.apply %0(%2) attributes {extra = 1} : (core.i32) -> (), {call_conv = "tail"}
      asm_format_test.invoke call @ret(%2)
  }
}
"#,
        );
    }

    #[test]
    fn calls_read_generic_syntax_and_print_other_results_generically() {
        let generic = r#"core.module @m {
  asm_format_test.fun @f(%0: core.ptr, %1: core.i32) {
      %2 = "asm_format_test.apply"(%0, %1) {signature = func.func_sig<(core.i32) -> core.i32>} : core.i32
      %3 = "asm_format_test.apply"(%0, %1) {signature = func.func_sig<(core.i32) -> core.i32>} : core.i1
  }
}
"#;
        assert_eq!(
            print(generic),
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.ptr, %1: core.i32) {
      %2 = asm_format_test.apply %0(%1) : (core.i32) -> core.i32
      %3 = "asm_format_test.apply"(%0, %1) {signature = func.func_sig<(core.i32) -> core.i32>} : core.i1
  }
}
"#
        );
    }

    #[test]
    fn defaulted_attributes_are_absent_in_their_canonical_form() {
        let canonical = r#"core.module @m {
  asm_format_test.pick values
  asm_format_test.pick values n 7
}
"#;
        assert_round_trip(canonical);
        // An explicit default, in the format or generically, reads as absent.
        assert_eq!(
            print(
                r#"core.module @m {
  asm_format_test.pick values n 3 label "none"
  "asm_format_test.pick"() {label = "none", n = 7}
}
"#
            ),
            canonical
        );

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, canonical);
        let picks: Vec<_> = module
            .ops(&ctx)
            .iter()
            .map(|&op| Pick::from_op(&ctx, op).unwrap())
            .collect();
        let read = |pick: &Pick| (pick.n(&ctx), pick.label(&ctx).to_owned());
        assert_eq!(read(&picks[0]), (3, "none".to_owned()));
        assert_eq!(read(&picks[1]), (7, "none".to_owned()));
        assert_eq!(picks[0].label_ref(&ctx), None);

        // The printer leaves out a stored default, however it was stored.
        let first = picks[0].op_ref();
        ctx.op_attrs_mut(first).insert("n", Attribute::Int(3));
        assert_eq!(print_module(&ctx, module.op()), canonical);

        let loc = ctx.op(first).location;
        let default = Pick::operands().n(3).label("none").build(&mut ctx, loc);
        assert!(ctx.op(default.op_ref()).attributes.is_empty());
        let other = Pick::operands().n(7).build(&mut ctx, loc);
        assert_eq!(
            ctx.op_attrs(other.op_ref()).get("n"),
            Some(&Attribute::Int(7))
        );
    }

    #[test]
    fn a_float_default_is_its_bits() {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("test.trb");
        let loc = crate::Location::new(path, crate::Span::new(0, 0));
        let zero = Scale::operands()
            .factor(0.0)
            .ratio(0.0)
            .build(&mut ctx, loc);
        assert!(ctx.op(zero.op_ref()).attributes.is_empty());
        let negative = Scale::operands()
            .factor(-0.0)
            .ratio(-0.0)
            .build(&mut ctx, loc);
        let negative_zero = Some(&Attribute::FloatBits((-0.0f64).to_bits()));
        assert_eq!(ctx.op_attrs(negative.op_ref()).get("factor"), negative_zero);
        assert_eq!(ctx.op_attrs(negative.op_ref()).get("ratio"), negative_zero);
        assert!(negative.factor(&ctx).is_sign_negative());
        assert!(negative.ratio(&ctx).is_sign_negative());
    }

    #[test]
    fn a_format_may_begin_with_an_operand() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32) {
      %1 = asm_format_test.add %0, %0 : core.i32
      %2 = asm_format_test.add %1, %0 attributes {extra = 1} : core.i32
  }
}
"#,
        );
    }

    #[test]
    fn a_bare_dictionary_follows_an_operand_list_on_its_line() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32) {
      asm_format_test.emit
      %1 = asm_format_test.emit %0, %0 {tag = 1} : core.i32
      %2, %3 = asm_format_test.emit %1 : core.i32, core.i32
      asm_format_test.emit {tag = 2}
  }
}
"#,
        );
    }

    #[test]
    fn entry_arguments_are_written_in_the_header() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.guard try {
      "test.op"()
  } catch(%0: core.i32, %1: core.i64) {
      "test.use"(%0, %1)
  }
}
"#,
        );
        // A region without an entry block prints generically.
        let empty = "core.module @m {\n  \"asm_format_test.guard\"() ({\n  }, {\n  })\n}\n";
        assert_eq!(print(empty), empty);
    }

    #[test]
    fn calls_write_their_operand_and_result_types_as_a_function_type() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32, %1: core.i64) {
      %2 = asm_format_test.call_fn @g(%0, %1) : (core.i32, core.i64) -> core.i32
      %3, %4 = asm_format_test.call_fn @g(%0) attributes {extra = 1} : (core.i32) -> (core.i32, core.i64)
      asm_format_test.call_fn @g() : () -> ()
      asm_format_test.tail_fn @g(%0) : (core.i32) -> ()
  }
}
"#,
        );
    }

    #[test]
    fn operand_types_are_written_and_checked() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32) {
      asm_format_test.consume %0 : core.i32
      asm_format_test.consume %0 : core.i32 attributes {extra = 1}
  }
}
"#,
        );
        let error = parse_error(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32) {
      asm_format_test.consume %0 : core.i64
  }
}
"#,
        );
        assert!(
            error.contains("operand '%0' of 'asm_format_test.consume' is written as core.i64"),
            "{error}"
        );
    }

    #[test]
    fn fixed_result_types_a_format_does_not_write_come_from_the_schema() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32, %1: core.i32) {
      %2 = asm_format_test.compare %0, %1 : core.i32
      asm_format_test.consume %2 : core.i1
      %3 = asm_format_test.make_token
  }
}
"#,
        );
        // A result of another type, or with attributes the fixed type does
        // not have, prints generically.
        let text = r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32, %1: core.i32) {
      %2 = "asm_format_test.compare"(%0, %1) : core.i32
      %3 = "asm_format_test.make_token"() : asm_format_test.token<{note = 1}>
  }
}
"#;
        assert_eq!(print(text), text);
    }

    #[test]
    fn written_operand_types_must_match_the_operands() {
        let error = parse_error(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32) {
      asm_format_test.tail_fn @g(%0) : (core.i64) -> ()
  }
}
"#,
        );
        assert!(
            error.contains("operand '%0' of 'asm_format_test.tail_fn' is written as core.i64, but its type is core.i32"),
            "{error}"
        );
        let error = parse_error(
            r#"core.module @m {
  asm_format_test.fun @f(%0: core.i32) {
      asm_format_test.tail_fn @g(%0) : (core.i32, core.i32) -> ()
  }
}
"#,
        );
        assert!(error.contains("one type for each operand"), "{error}");
    }

    #[test]
    fn keywords_write_what_their_codec_spells() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.drive go(fast)
  asm_format_test.drive go(slow) attributes {extra = 1}
}
"#,
        );
        assert_eq!(
            print("core.module @m {\n  \"asm_format_test.drive\"() {speed = 1}\n}\n"),
            "core.module @m {\n  \"asm_format_test.drive\"() {speed = 1}\n}\n"
        );
        let error = parse_error("core.module @m {\n  asm_format_test.drive go(medium)\n}\n");
        assert!(error.contains("keyword"), "{error}");
    }

    #[test]
    fn generic_signatures_read_back() {
        let generic = r#"core.module @m {
  "asm_format_test.fun"() ({
    ^bb0(%0: core.i32):
      asm_format_test.invoke call @ret(%0)
  }) {sym_name = "g", type = func.func_sig<(core.i32) -> core.i32, {call_conv = "tail"}>}
}
"#;
        assert_eq!(
            print(generic),
            r#"core.module @m {
  asm_format_test.fun @g(%0: core.i32) -> core.i32, {call_conv = "tail"} {
      asm_format_test.invoke call @ret(%0)
  }
}
"#
        );
    }

    #[test]
    fn signatures_that_do_not_match_their_body_print_generically() {
        let text = r#"core.module @m {
  "asm_format_test.fun"() ({
    ^bb0(%0: core.i64):
      asm_format_test.invoke call @ret(%0)
  }) {sym_name = "g", type = func.func_sig<(core.i32) -> ()>}
}
"#;
        assert_eq!(print(text), text);
    }

    #[test]
    fn signatures_reject_more_results_than_their_type_allows() {
        let error = parse_error(
            r#"core.module @m {
  asm_format_test.fun @f() -> (core.i32, core.i32)
}"#,
        );
        assert!(error.contains("no more results"), "{error}");
    }

    #[test]
    fn formats_round_trip() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.def @f kind "pure" inferred attributes {extra = 1} {
      %0 = asm_format_test.tagged ref @f note "n" : core.i32
      %1 = asm_format_test.tagged ref @g : core.i32
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
      asm_format_test.mark now quick
      asm_format_test.mark now
      %0 = asm_format_test.tagged ref @f : core.i32
      asm_format_test.mark now with(%0)
      asm_format_test.mark now attributes {extra = 1}
      asm_format_test.switch cases [^bb1, ^bb0] {
          asm_format_test.mark now
      } {
          asm_format_test.mark now quick
      }

    ^bb1:
      %1 = asm_format_test.switch cases [] : core.i32
  }
}
"#,
        );
    }

    #[test]
    fn false_flags_print_generically_and_read_back() {
        let text = r#"core.module @m {
  "asm_format_test.mark"() {quick = false}
  "asm_format_test.toggle"() {on = false}
}
"#;
        assert_eq!(print(text), text);
    }

    #[test]
    fn absent_elements_do_not_read_the_next_operation() {
        assert_round_trip(
            r#"core.module @m {
  asm_format_test.def @f kind "k" {
      asm_format_test.toggle turn on
      asm_format_test.toggle turn
      asm_format_test.run go(on)
      asm_format_test.run go()
      asm_format_test.named @target
      %0 = asm_format_test.reference @f note "n" : core.i32
      %1 = asm_format_test.reference @g : core.i32
      asm_format_test.mark now
      "attributes.probe"()
  }
}
"#,
        );
    }

    #[test]
    fn generic_syntax_reads_as_the_same_operation() {
        let generic = r#"core.module @m {
  %0 = "asm_format_test.tagged"() {note = "n", tag = @f} : core.i32
  "asm_format_test.def"() {kind = "k", sym_name = "d"}
  %1 = "asm_format_test.reference"() {target = @f} : core.i32
  "asm_format_test.named"() {extra = 1, sym_name = "t"}
}
"#;
        assert_eq!(
            print(generic),
            r#"core.module @m {
  %0 = asm_format_test.tagged ref @f note "n" : core.i32
  asm_format_test.def @d kind "k"
  %0 = asm_format_test.reference @f : core.i32
  "asm_format_test.named"() {extra = 1, sym_name = "t"}
}
"#
        );
    }

    #[test]
    fn operations_the_format_cannot_spell_print_generically() {
        let text = r#"core.module @m {
  %0 = "asm_format_test.tagged"() {extra = 1, tag = @f} : core.i32
  "asm_format_test.def"() {inferred = false, kind = "k", sym_name = "d"}
  "asm_format_test.def"() {sym_name = "e"}
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
  asm_format_test.mark now with()
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
