//! Textual assembly format hooks.
//!
//! Operations declare these hooks with `#[asm_format]`, and types register
//! them, to control how they are spelled in textual IR. They affect only the printer and the parser:
//! unlike [operation interfaces](crate::op_interface), they carry no meaning
//! that passes or analyses consult.

use std::fmt;
use std::sync::LazyLock;

use crate::ops::{DialectOp, DialectType};
use crate::parser::raw::RawOperation;
use crate::printer::OpPrintHelper;
use crate::registry::{Registered, Registry};
use crate::{IrContext, OpRef, TypeRef};

// =============================================================================
// TypeAliasHint — dialect-provided alias name suggestions for printer
// =============================================================================

/// Dialect-provided hint for suggesting type alias names during printing.
///
/// Each dialect can register a hint that maps its types to suggested alias names.
/// The printer uses these hints when auto-generating type aliases.
pub struct TypeAliasHint {
    /// Dialect name this hint applies to (e.g., "adt").
    pub dialect: &'static str,
    /// Given a type belonging to this dialect, suggest an alias name.
    /// Returns `None` if no name can be suggested.
    pub suggest: for<'a> fn(&'a IrContext, TypeRef) -> Option<&'a str>,
}

inventory::collect!(TypeAliasHint);

/// Query all registered `TypeAliasHint`s to find a suggested name for the given type.
pub fn suggest_type_alias_name(ctx: &IrContext, ty: TypeRef) -> Option<&str> {
    let dialect = ctx.get_type(ty).name.dialect(ctx);
    for hint in inventory::iter::<TypeAliasHint> {
        if dialect == hint.dialect
            && let Some(name) = (hint.suggest)(ctx, ty)
        {
            return Some(name);
        }
    }
    None
}

// =============================================================================
// AsmFormat — custom assembly format for operations (print + parse)
// =============================================================================

/// Custom assembly format for an operation — bundles print + parse.
///
/// Modeled after MLIR's `hasCustomAssemblyFormat`. Required for operations
/// declared with `#[asm_format]`; the operation's [`OpDef`](crate::OpDef)
/// holds the hooks, so the printer and the parser reach them through its
/// name.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is declared with `#[asm_format]` but does not implement `AsmFormat`",
    label = "`#[asm_format]` requires this implementation",
    note = "add `impl AsmFormat for {Self} {{ fn print(..) {{ .. }} fn parse(..) {{ .. }} }}`"
)]
pub trait AsmFormat: DialectOp {
    /// Print the operation. Called instead of generic printing, after the
    /// line's indentation and before its newline.
    fn print(self, h: &mut OpPrintHelper<'_, '_>) -> fmt::Result;

    /// Parse the operation. Called after `dialect.op` and optional
    /// `@sym_name` are consumed; `results` and `sym_name` are already parsed
    /// by the generic parser.
    fn parse<'a>(
        input: &mut &'a str,
        results: Vec<&'a str>,
        sym_name: Option<String>,
    ) -> winnow::ModalResult<RawOperation<'a>>;
}

/// The hooks an [`OpDef`](crate::OpDef) stores for an `#[asm_format]`
/// operation.
#[derive(Clone, Copy, Debug)]
pub struct OpAsmHooks {
    /// Custom printer. Called instead of generic printing.
    pub print: OpPrintFn,
    /// Custom parser. Called after `dialect.op` and optional `@sym_name` are consumed.
    pub parse: OpParseFn,
}

/// Custom printer of an [`OpAsmHooks`].
pub type OpPrintFn = fn(&mut OpPrintHelper<'_, '_>, OpRef) -> fmt::Result;

/// Custom parser of an [`OpAsmHooks`].
pub type OpParseFn = for<'a> fn(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<RawOperation<'a>>;

// =============================================================================
// TypeAsmFormat — custom assembly format for types (print + parse)
// =============================================================================

/// Custom assembly format for a type — bundles print + parse.
///
/// A dialect registers [`TypeAsmFormat::new`] via `inventory::submit!` to own
/// its type's textual syntax. Like every type, the custom form is
/// self-delimiting: it is written inside `dialect.name<...>`.
pub struct TypeAsmFormat {
    def: &'static crate::TypeDef,
    /// Custom printer. Writes the whole type, including `dialect.name<` and
    /// the closing `>`, or returns `None` to leave a type it cannot represent,
    /// such as a malformed one, to generic printing.
    pub print_fn: TypePrintFn,
    /// Custom parser. Called after `dialect.name<` is consumed; it consumes the
    /// closing `>`. A backtrack falls back to the generic form.
    pub parse_fn: TypeParseFn,
}

/// Custom printer of a [`TypeAsmFormat`].
pub type TypePrintFn =
    fn(&mut crate::printer::TypePrintHelper<'_, '_>, TypeRef) -> Option<fmt::Result>;

/// Custom parser of a [`TypeAsmFormat`].
///
/// It returns the type in generic raw form, so the parser builds it like any
/// other type.
pub type TypeParseFn = for<'a> fn(
    input: &mut &'a str,
    name: &'a str,
) -> winnow::ModalResult<crate::parser::raw::RawType<'a>>;

impl TypeAsmFormat {
    /// Custom assembly format for the type wrapped by `T`.
    pub const fn new<T: DialectType>(print_fn: TypePrintFn, parse_fn: TypeParseFn) -> Self {
        Self {
            def: T::DEF,
            print_fn,
            parse_fn,
        }
    }
}

inventory::collect!(TypeAsmFormat);

impl Registered for TypeAsmFormat {
    type Def = crate::TypeDef;
    const KIND: &'static str = "TypeAsmFormat";

    fn def(&self) -> &'static crate::TypeDef {
        self.def
    }
}

static TYPE_ASM_FORMATS: LazyLock<Registry<TypeAsmFormat>> = LazyLock::new(Registry::collect);

/// Look up a registered custom assembly format for the given type.
pub fn lookup_type_asm_format(name: crate::TypeName) -> Option<&'static TypeAsmFormat> {
    TYPE_ASM_FORMATS.get(name)
}
