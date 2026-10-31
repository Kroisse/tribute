//! Textual assembly format hooks.
//!
//! Operations and types declare these hooks with `#[asm_format]`, and their
//! definitions hold them, to control how they are spelled in textual IR. They
//! affect only the printer and the parser: unlike [operation interfaces](mod@crate::op_interface), they carry no meaning
//! that passes or analyses consult. An operation declares its format either
//! by hand through [`OpAsmFormat`], or with format elements in
//! `#[asm_format(..)]`, which [`declarative`] interprets.

pub mod declarative;

use std::fmt;

use crate::ops::{DialectOp, DialectType};
use crate::parser::raw::{RawOperation, RawParam, RawType};
use crate::printer::{OpPrintHelper, TypePrintHelper};
use crate::{AttributeMap, IrContext, OpRef, TypeRef};

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
    /// line's indentation and before its newline.
    fn print(&self, h: &mut OpPrintHelper<'_, '_>) -> fmt::Result;

    /// Parse the operation. Called after `dialect.op` and optional
    /// `@sym_name` are consumed; `results` and `sym_name` are already parsed
    /// by the generic parser.
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
    pub print: fn(&mut OpPrintHelper<'_, '_>, OpRef) -> fmt::Result,
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
    /// Whether the codec writes a tail after the results.
    pub tail: bool,
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
    /// The inputs and results of `ty`, or `None` when the function form
    /// cannot spell it without loss.
    fn decompose(ctx: &IrContext, ty: TypeRef) -> Option<SignatureParts<'_>>;

    /// Print the tail after the results. Called only when
    /// [`decompose`](Self::decompose) reports one.
    fn print_tail(h: &mut OpPrintHelper<'_, '_>, ty: TypeRef) -> fmt::Result;

    /// Parse the tail after the results, if any, and build the type's generic
    /// raw form from the parsed inputs and results. A result count the type
    /// does not allow is an error.
    fn parse<'a>(
        input: &mut &'a str,
        inputs: Vec<RawParam<'a>>,
        results: Vec<RawParam<'a>>,
    ) -> winnow::ModalResult<RawType<'a>>;
}

/// A [`SignatureCodec`]'s hooks, as a `signature` format element stores them.
#[derive(Clone, Copy, Debug)]
pub struct SignatureCodecHooks {
    pub decompose: for<'c> fn(&'c IrContext, TypeRef) -> Option<SignatureParts<'c>>,
    pub print_tail: fn(&mut OpPrintHelper<'_, '_>, TypeRef) -> fmt::Result,
    pub parse: for<'a> fn(
        &mut &'a str,
        Vec<RawParam<'a>>,
        Vec<RawParam<'a>>,
    ) -> winnow::ModalResult<RawType<'a>>,
}

impl SignatureCodecHooks {
    /// The hooks of `T`'s codec.
    pub const fn of<T: SignatureCodec>() -> Self {
        Self {
            decompose: T::decompose,
            print_tail: T::print_tail,
            parse: T::parse,
        }
    }
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
    fn print(&self, h: &mut TypePrintHelper<'_, '_>) -> fmt::Result;

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
    pub print: fn(&mut TypePrintHelper<'_, '_>, TypeRef) -> Option<fmt::Result>,
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

fn print_type<T: TypeAsmFormat>(
    h: &mut TypePrintHelper<'_, '_>,
    ty: TypeRef,
) -> Option<fmt::Result> {
    let ty = T::from_type_ref(h.ctx(), ty)?;
    Some(ty.print(h))
}
