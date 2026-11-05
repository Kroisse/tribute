//! Textual assembly format hooks.
//!
//! Operations and types declare these hooks with `#[asm_format]`, and their
//! definitions hold them, to control how they are spelled in textual IR. They
//! affect only the printer and the parser: unlike [operation interfaces](mod@crate::op_interface), they carry no meaning
//! that passes or analyses consult. An operation declares its format either
//! by hand through [`OpAsmFormat`], or with format elements in
//! `#[asm_format(..)]`, which [`declarative`] interprets.

pub mod declarative;

use std::fmt::{self, Write as _};

use crate::ops::{DialectOp, DialectType};
use crate::parser::raw::{RawAttrDict, RawAttribute, RawOperation, RawParam, RawType};
use crate::printer::{AsmNamer, AsmPrinter, OpAsmPrinter};
use crate::types::Attribute;
use crate::{AttributeMap, IrContext, OpRef, StringRef, TypeRef};

// =============================================================================
// TypeAliasHint — alias name suggestions for the printer
// =============================================================================

/// Alias name a type suggests when the printer auto-generates type aliases.
///
/// `#[dialect]` opts in with `#[alias_hint]`, and a hand-written definition
/// adds it with [`TypeDef::with_alias_hint`](crate::TypeDef::with_alias_hint).
/// A type without a suggestion is aliased only when it is complex enough, under
/// a numbered name.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is declared with `#[alias_hint]` but does not implement `TypeAliasHint`",
    label = "`#[alias_hint]` requires this implementation",
    note = "add `impl TypeAliasHint for {Self} {{ fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {{ .. }} }}`"
)]
pub trait TypeAliasHint: DialectType {
    /// Suggest an alias name for this type. A type the wrapper does not
    /// accept, such as a malformed one, suggests none.
    fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str>;
}

/// A type's alias name hint, calling the wrapper's [`TypeAliasHint`] impl.
pub type TypeAliasHook = for<'a> fn(&'a IrContext, TypeRef) -> Option<&'a str>;

/// `T`'s alias name hint as a [`TypeDef`](crate::TypeDef) stores it.
pub(crate) fn erased_alias_hint<T: TypeAliasHint>(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    T::from_type_ref(ctx, ty)?.alias_hint(ctx)
}

/// The alias name `ty`'s definition suggests, if any.
pub fn suggest_type_alias_name(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    let hint = crate::TypeDef::of(ctx, ty)?.alias_hint?;
    hint(ctx, ty)
}

// =============================================================================
// OpAsmFormat — custom assembly format for operations (print + parse)
// =============================================================================

/// Custom assembly format for an operation — bundles print + parse.
///
/// Modeled after MLIR's `hasCustomAssemblyFormat`. Required for operations
/// declared with `#[asm_format]`; the operation's [`OpDef`](crate::OpDef)
/// holds the hooks, so the printer and the parser reach them through its
/// name.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is declared with `#[asm_format]` but does not implement `OpAsmFormat`",
    label = "`#[asm_format]` requires this implementation",
    note = "add `impl OpAsmFormat for {Self} {{ fn print(..) {{ .. }} fn parse(..) {{ .. }} }}`"
)]
pub trait OpAsmFormat: DialectOp {
    /// Print the operation. Called instead of generic printing, after the
    /// line's indentation and before its newline. The names it writes come
    /// from [`Self::assign_names`], through the printer's state.
    fn print(&self, p: &mut OpAsmPrinter<'_>) -> fmt::Result;

    /// Name the values and blocks [`Self::print`] writes, in the order it
    /// writes them, before printing. By default the results, then each
    /// region, with fresh numbering when the operation is isolated from
    /// above ([`AsmNamer::default_op`]).
    fn assign_names(&self, namer: &mut AsmNamer<'_>) {
        namer.default_op(self.op_ref());
    }

    /// Parse the operation. Called after `dialect.op` and optional
    /// `@sym_name` are consumed; `results` and `sym_name` are already parsed
    /// by the caller.
    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<RawOperation<'a>>;
}

/// The custom assembly format an [`OpDef`](crate::OpDef) stores for an
/// `#[asm_format]` operation.
#[derive(Clone, Copy, Debug)]
pub enum OpAsmFormatDef {
    /// Hand-written hooks, from the wrapper's [`OpAsmFormat`] impl.
    Hooks(OpAsmHooks),
    /// Format elements declared in `#[asm_format(..)]`.
    Declarative(&'static declarative::OpFormat),
}

/// The hooks an [`OpDef`](crate::OpDef) stores for an `#[asm_format]`
/// operation.
#[derive(Clone, Copy, Debug)]
pub struct OpAsmHooks {
    /// Custom printer. Called instead of generic printing.
    pub print: fn(&mut OpAsmPrinter<'_>, OpRef) -> fmt::Result,
    /// Names the values and blocks the printer writes, before printing.
    pub assign_names: fn(&mut AsmNamer<'_>, OpRef),
    /// Custom parser. Called after `dialect.op` and optional `@sym_name` are consumed.
    pub parse: for<'a> fn(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<RawOperation<'a>>,
}

// =============================================================================
// SignatureCodec — the function form of a signature type
// =============================================================================

/// The parameters a [`SignatureCodec`] decomposes a signature type into.
#[derive(Debug)]
pub struct SignatureParts<'a> {
    /// Each input with its own attributes.
    pub inputs: Vec<(TypeRef, &'a AttributeMap)>,
    /// Each result with its own attributes.
    pub results: Vec<(TypeRef, &'a AttributeMap)>,
    /// The type attributes the codec writes after the results.
    pub tail: Vec<(StringRef, &'a Attribute)>,
}

/// Conversion between a signature type and the function form
/// `(%a: A, %b: B) -> R` that the `signature` format element writes
/// (`new-plans/ir.md`, "Signature codec").
///
/// The dialect that defines the signature type implements it on the type's
/// wrapper. A format names the codec through the bound of the type variable
/// its `signature` element writes, so trunk-ir knows no signature type.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is the bound of a `signature` format element but does not implement `SignatureCodec`",
    label = "`signature(..)` requires this implementation"
)]
pub trait SignatureCodec: DialectType {
    /// The inputs, results, and tail of `ty`, or `None` when the function
    /// form cannot spell it without loss.
    fn decompose(ctx: &IrContext, ty: TypeRef) -> Option<SignatureParts<'_>>;

    /// Write the tail after the results; nothing when it is empty. By
    /// default it is a dictionary after a comma, [`write_dict_tail`].
    fn write_tail(p: &mut AsmPrinter<'_>, tail: &[(StringRef, &Attribute)]) -> fmt::Result {
        write_dict_tail(p, tail)
    }

    /// Parse the tail [`write_tail`](Self::write_tail) writes, or an empty
    /// one when its introducer is absent.
    fn parse_tail<'a>(input: &mut &'a str) -> winnow::ModalResult<RawAttrDict<'a>> {
        parse_dict_tail(input)
    }

    /// Build the type's generic raw form from the parsed inputs, results,
    /// and tail. A result count or a tail attribute the type does not allow
    /// is an error.
    fn build<'a>(
        inputs: Vec<RawParam<'a>>,
        results: Vec<RawParam<'a>>,
        tail: RawAttrDict<'a>,
    ) -> winnow::ModalResult<RawType<'a>>;
}

/// A [`SignatureCodec`]'s hooks, as a `signature` format element stores them.
#[derive(Clone, Copy, Debug)]
pub struct SignatureCodecHooks {
    pub decompose: for<'c> fn(&'c IrContext, TypeRef) -> Option<SignatureParts<'c>>,
    pub write_tail: fn(&mut AsmPrinter<'_>, &[(StringRef, &Attribute)]) -> fmt::Result,
    pub parse_tail: for<'a> fn(&mut &'a str) -> winnow::ModalResult<RawAttrDict<'a>>,
    pub build: for<'a> fn(
        Vec<RawParam<'a>>,
        Vec<RawParam<'a>>,
        RawAttrDict<'a>,
    ) -> winnow::ModalResult<RawType<'a>>,
}

impl SignatureCodecHooks {
    /// The hooks of `T`'s codec.
    pub const fn of<T: SignatureCodec>() -> Self {
        Self {
            decompose: T::decompose,
            write_tail: T::write_tail,
            parse_tail: T::parse_tail,
            build: T::build,
        }
    }
}

// =============================================================================
// Keywords — attribute kinds spelled as bare identifiers
// =============================================================================

/// Read a bare identifier as the attribute `attribute` makes of it, for an
/// attribute kind whose values are keywords
/// ([`AttrKind::parse_text`](crate::attr_kind::AttrKind::parse_text)).
/// A word `attribute` rejects leaves the input where it was.
pub fn parse_keyword<'a>(
    input: &mut &'a str,
    attribute: impl FnOnce(&str) -> Option<RawAttribute<'a>>,
) -> winnow::ModalResult<RawAttribute<'a>> {
    use winnow::Parser as _;

    let checkpoint = *input;
    match winnow::combinator::opt(crate::parser::raw::ident)
        .parse_next(input)?
        .and_then(attribute)
    {
        Some(value) => Ok(value),
        None => {
            *input = checkpoint;
            Err(winnow::error::ErrMode::Backtrack(expected("keyword")))
        }
    }
}

/// A parse error that names what the input should have held.
pub(crate) fn expected(description: &'static str) -> winnow::error::ContextError {
    use winnow::error::{ContextError, StrContext, StrContextValue};

    let mut error = ContextError::new();
    error.push(StrContext::Expected(StrContextValue::Description(
        description,
    )));
    error
}

// =============================================================================
// OpAsmDirective — a part of a declarative format written by hand
// =============================================================================

/// The syntax of the entities a `custom(D(a, b, ..))` format element names,
/// written by hand inside an otherwise declarative format, as MLIR's
/// `custom<D>` directive is (`new-plans/ir.md`, "선언적 어셈블리 형식").
///
/// The element passes the entities in the order it names them: operands,
/// attributes, successors, and regions, each as a [`CustomEntity`] when
/// printing and a [`RawCustomEntity`] when parsing.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is named by a `custom(..)` format element but does not implement `OpAsmDirective`",
    label = "`custom(..)` requires this implementation"
)]
pub trait OpAsmDirective {
    /// What [`Self::decompose`] reads from the entities for [`Self::print`].
    type Parts<'a>;

    /// Whether the printer writes a space before the directive, as before
    /// other elements. A directive that starts on its own line with
    /// [`OpAsmPrinter::newline`] writes none.
    const SPACED: bool = true;

    /// The parts [`Self::print`] writes so that [`Self::parse`] reads the
    /// entities back, or `None` when it cannot; the operation then prints
    /// generically.
    fn decompose<'a>(
        ctx: &'a IrContext,
        entities: &'a [CustomEntity<'a>],
    ) -> Option<Self::Parts<'a>>;

    /// Write the parts [`Self::decompose`] read.
    fn print(p: &mut OpAsmPrinter<'_>, parts: Self::Parts<'_>) -> fmt::Result;

    /// Name the values and blocks [`Self::print`] writes, in the order it
    /// writes them, before printing: by default each region the entities
    /// hold, in order, as [`AsmNamer::region`] names it.
    fn assign_names(namer: &mut AsmNamer<'_>, entities: &[CustomEntity<'_>]) {
        for entity in entities {
            if let CustomEntity::Regions(regions) = entity {
                for &region in regions {
                    namer.region(region);
                }
            }
        }
    }

    /// Read the entities back, one for each the element names, in order.
    fn parse<'a>(input: &mut &'a str) -> winnow::ModalResult<Vec<RawCustomEntity<'a>>>;
}

/// An entity a [`OpAsmDirective`] prints.
#[derive(Clone, Debug)]
pub enum CustomEntity<'a> {
    /// An operand, or the operands of a variadic one.
    Operands(&'a [crate::ValueRef]),
    /// An attribute, if the operation has it.
    Attribute(Option<&'a Attribute>),
    /// A successor, or the successors of a variadic one.
    Successors(crate::BlockList),
    /// A region, or the regions of a variadic or optional one.
    Regions(crate::RegionList),
}

/// An entity a [`OpAsmDirective`] parses.
#[derive(Clone, Debug)]
pub enum RawCustomEntity<'a> {
    /// Operand names, without `%`.
    Operands(Vec<&'a str>),
    /// An attribute, or `None` when it is absent.
    Attribute(Option<RawAttribute<'a>>),
    /// Successor labels, without `^`.
    Successors(Vec<&'a str>),
    /// Regions, in order.
    Regions(Vec<crate::parser::raw::RawRegion<'a>>),
}

/// A [`OpAsmDirective`]'s hooks, as a `custom` format element stores them.
#[derive(Clone, Copy, Debug)]
pub struct OpAsmDirectiveHooks {
    pub fits: fn(&IrContext, &[CustomEntity<'_>]) -> bool,
    pub print: fn(&mut OpAsmPrinter<'_>, &[CustomEntity<'_>]) -> fmt::Result,
    pub assign_names: fn(&mut AsmNamer<'_>, &[CustomEntity<'_>]),
    pub parse: for<'a> fn(&mut &'a str) -> winnow::ModalResult<Vec<RawCustomEntity<'a>>>,
    pub spaced: bool,
}

impl OpAsmDirectiveHooks {
    /// The hooks of `T`'s directive.
    pub const fn of<T: OpAsmDirective>() -> Self {
        Self {
            fits: directive_fits::<T>,
            print: directive_print::<T>,
            assign_names: T::assign_names,
            parse: T::parse,
            spaced: T::SPACED,
        }
    }
}

/// Parse the entry block arguments
/// [`AsmState::entry_args`](crate::printer::AsmState::entry_args) writes.
pub fn parse_entry_args<'a>(
    input: &mut &'a str,
) -> winnow::ModalResult<Vec<(&'a str, RawType<'a>)>> {
    crate::parser::raw::func_params(input)
}

/// Give `region`, parsed with its entry block elided, the entry block
/// arguments [`parse_entry_args`] read. Fails when the region's entry block
/// already writes arguments of its own.
pub fn attach_entry_args<'a>(
    region: &mut crate::parser::raw::RawRegion<'a>,
    args: Vec<(&'a str, RawType<'a>)>,
) -> winnow::ModalResult<()> {
    match region.blocks.first_mut() {
        None => region.blocks.push(crate::parser::raw::RawBlock {
            label: "",
            args,
            ops: Vec::new(),
        }),
        Some(entry) if entry.args.is_empty() => entry.args = args,
        Some(_) if args.is_empty() => {}
        Some(_) => {
            return Err(winnow::error::ErrMode::Cut(expected(
                "a region whose entry block writes no arguments",
            )));
        }
    }
    Ok(())
}

fn directive_fits<T: OpAsmDirective>(ctx: &IrContext, entities: &[CustomEntity<'_>]) -> bool {
    T::decompose(ctx, entities).is_some()
}

fn directive_print<T: OpAsmDirective>(
    p: &mut OpAsmPrinter<'_>,
    entities: &[CustomEntity<'_>],
) -> fmt::Result {
    let parts =
        T::decompose(p.ctx(), entities).expect("the printer checked that the directive fits");
    T::print(p, parts)
}

/// Write a function form's parameter type and its attributes, `T {k = v}`.
pub fn write_signature_param(
    p: &mut AsmPrinter<'_>,
    ty: TypeRef,
    attrs: &AttributeMap,
) -> fmt::Result {
    let s = p.state();
    write!(p, "{}", s.ty(ty))?;
    if !attrs.is_empty() {
        write!(p, " {}", s.attr_dict(attrs.iter()))?;
    }
    Ok(())
}

/// Write a function form's results after `->`: one result without
/// attributes bare, and otherwise a parenthesized list, `(R {k = v})`.
pub fn write_signature_results(
    p: &mut AsmPrinter<'_>,
    results: &[(TypeRef, &AttributeMap)],
) -> fmt::Result {
    if let [(ty, attrs)] = results
        && attrs.is_empty()
    {
        return p.print_type(*ty);
    }
    p.write_char('(')?;
    for (index, &(ty, attrs)) in results.iter().enumerate() {
        if index > 0 {
            p.write_str(", ")?;
        }
        write_signature_param(p, ty, attrs)?;
    }
    p.write_char(')')
}

/// Parse a function form's results after `->` as
/// [`write_signature_results`] writes them.
pub fn parse_signature_results<'a>(input: &mut &'a str) -> winnow::ModalResult<Vec<RawParam<'a>>> {
    use crate::parser::raw::{raw_param_list, raw_type};
    use winnow::prelude::*;

    if input.starts_with('(') {
        raw_param_list.parse_next(input)
    } else {
        Ok(vec![raw_type.parse_next(input)?.into()])
    }
}

/// Write a tail of type attributes as a dictionary after a comma,
/// `, {k = v}`, as the `func.func_sig` family does.
pub fn write_dict_tail(p: &mut AsmPrinter<'_>, tail: &[(StringRef, &Attribute)]) -> fmt::Result {
    if tail.is_empty() {
        return Ok(());
    }
    write!(p, ", {}", p.state().attr_dict(tail.iter().copied()))
}

/// Parse the tail [`write_dict_tail`] writes.
pub fn parse_dict_tail<'a>(input: &mut &'a str) -> winnow::ModalResult<RawAttrDict<'a>> {
    use crate::parser::raw::{raw_attr_dict, ws};
    use winnow::prelude::*;

    let checkpoint = *input;
    ws.parse_next(input)?;
    if !input.starts_with(',') {
        *input = checkpoint;
        return Ok(Vec::new());
    }
    ','.parse_next(input)?;
    ws.parse_next(input)?;
    raw_attr_dict.parse_next(input)
}

// =============================================================================
// TypeAsmFormat — custom assembly format for types (print + parse)
// =============================================================================

/// Custom assembly format for a type — bundles print + parse.
///
/// The type's [`TypeDef`](crate::TypeDef) holds the hooks: a type declared in
/// `#[dialect]` opts in with `#[asm_format]`, and a hand-written definition
/// adds it with [`TypeDef::with_asm_format`](crate::TypeDef::with_asm_format). Like every type, the custom form is
/// self-delimiting: it is written inside `dialect.name<...>`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is declared with `#[asm_format]` but does not implement `TypeAsmFormat`",
    label = "`#[asm_format]` requires this implementation",
    note = "add `impl TypeAsmFormat for {Self} {{ fn print(..) {{ .. }} fn parse(..) {{ .. }} }}`"
)]
pub trait TypeAsmFormat: DialectType {
    /// Print the whole type, including `dialect.name<` and the closing `>`.
    /// A type the wrapper does not accept, such as a malformed one, is left
    /// to generic printing.
    fn print(&self, p: &mut AsmPrinter<'_>) -> fmt::Result;

    /// Parse the type after `dialect.name<` is consumed, including the
    /// closing `>`, into its generic raw form, so the parser builds it like
    /// any other type. A backtrack falls back to the generic form.
    fn parse<'a>(input: &mut &'a str, name: &'a str) -> winnow::ModalResult<RawType<'a>>;
}

/// The hooks a [`TypeDef`](crate::TypeDef) stores for a type with a custom
/// assembly format.
#[derive(Clone, Copy, Debug)]
pub struct TypeAsmHooks {
    /// Custom printer, or `None` to leave the type to generic printing.
    pub print: fn(&mut AsmPrinter<'_>, TypeRef) -> Option<fmt::Result>,
    /// Custom parser. Called after `dialect.name<` is consumed.
    pub parse: for<'a> fn(input: &mut &'a str, name: &'a str) -> winnow::ModalResult<RawType<'a>>,
}

impl TypeAsmHooks {
    /// The hooks calling `T`'s [`TypeAsmFormat`] impl.
    pub(crate) const fn of<T: TypeAsmFormat>() -> Self {
        Self {
            print: print_type::<T>,
            parse: T::parse,
        }
    }
}

fn print_type<T: TypeAsmFormat>(p: &mut AsmPrinter<'_>, ty: TypeRef) -> Option<fmt::Result> {
    let ty = T::from_type_ref(p.ctx(), ty)?;
    Some(ty.print(p))
}
