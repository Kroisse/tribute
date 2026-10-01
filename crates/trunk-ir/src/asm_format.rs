//! Textual assembly format hooks.
//!
//! Dialects register these hooks to control how their operations and types
//! are spelled in textual IR. They affect only the printer and the parser:
//! unlike [operation interfaces](crate::op_interface), they carry no meaning
//! that passes or analyses consult.

use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;

use crate::Symbol;
use crate::ops::{DialectOp, DialectType};
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
    pub suggest: fn(&IrContext, TypeRef) -> Option<Symbol>,
}

inventory::collect!(TypeAliasHint);

/// Query all registered `TypeAliasHint`s to find a suggested name for the given type.
pub fn suggest_type_alias_name(ctx: &IrContext, ty: TypeRef) -> Option<Symbol> {
    let data = ctx.get_type(ty);
    let dialect = data.dialect;
    for hint in inventory::iter::<TypeAliasHint> {
        if dialect.with_str(|s| s == hint.dialect)
            && let Some(name) = (hint.suggest)(ctx, ty)
        {
            return Some(name);
        }
    }
    None
}

// =============================================================================
// OpAsmFormat — custom assembly format for operations (print + parse)
// =============================================================================

/// Custom assembly format for an operation — bundles print + parse.
///
/// Modeled after MLIR's `hasCustomAssemblyFormat`. Register
/// [`OpAsmFormat::new`] via `inventory::submit!` at dialect definition sites.
/// The printer/parser dispatch automatically routes to the registered format.
pub struct OpAsmFormat {
    dialect: &'static str,
    op_name: &'static str,
    /// Custom printer. Called instead of generic printing.
    pub print_fn: OpPrintFn,
    /// Custom parser. Called after `dialect.op` and optional `@sym_name` are consumed.
    /// `results` and `sym_name` are already parsed by the generic parser.
    pub parse_fn: OpParseFn,
}

/// Custom printer of an [`OpAsmFormat`].
pub type OpPrintFn = fn(&mut crate::printer::OpPrintHelper<'_, '_>, OpRef, usize) -> fmt::Result;

/// Custom parser of an [`OpAsmFormat`].
pub type OpParseFn = for<'a> fn(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<crate::parser::raw::RawOperation<'a>>;

impl OpAsmFormat {
    /// Custom assembly format for the operation wrapped by `T`.
    pub const fn new<T: DialectOp>(print_fn: OpPrintFn, parse_fn: OpParseFn) -> Self {
        Self {
            dialect: T::DIALECT_NAME,
            op_name: T::OP_NAME,
            print_fn,
            parse_fn,
        }
    }
}

inventory::collect!(OpAsmFormat);

impl Registered for OpAsmFormat {
    const KIND: &'static str = "OpAsmFormat";

    fn key(&self) -> (&'static str, &'static str) {
        (self.dialect, self.op_name)
    }
}

static OP_ASM_FORMATS: LazyLock<Registry<OpAsmFormat>> = LazyLock::new(Registry::collect);

/// Look up a registered custom assembly format for the given operation.
pub fn lookup_asm_format(dialect: Symbol, op_name: Symbol) -> Option<&'static OpAsmFormat> {
    OP_ASM_FORMATS.get(dialect, op_name)
}

// =============================================================================
// TypeAsmFormat — custom assembly format for types (print + parse)
// =============================================================================

/// Custom assembly format for a type — bundles print + parse.
///
/// A dialect registers [`TypeAsmFormat::new`] via `inventory::submit!` to own
/// its type's textual syntax. Like every type, the custom form is
/// self-delimiting: it is written inside `dialect.name<...>`.
pub struct TypeAsmFormat {
    dialect: &'static str,
    type_name: &'static str,
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
    dialect: &'a str,
    name: &'a str,
) -> winnow::ModalResult<crate::parser::raw::RawType<'a>>;

impl TypeAsmFormat {
    /// Custom assembly format for the type wrapped by `T`.
    pub const fn new<T: DialectType>(print_fn: TypePrintFn, parse_fn: TypeParseFn) -> Self {
        Self {
            dialect: T::DIALECT_NAME,
            type_name: T::TYPE_NAME,
            print_fn,
            parse_fn,
        }
    }
}

inventory::collect!(TypeAsmFormat);

impl Registered for TypeAsmFormat {
    const KIND: &'static str = "TypeAsmFormat";

    fn key(&self) -> (&'static str, &'static str) {
        (self.dialect, self.type_name)
    }
}

static TYPE_ASM_FORMATS: LazyLock<Registry<TypeAsmFormat>> = LazyLock::new(Registry::collect);

/// Look up a registered custom assembly format for the given type.
pub fn lookup_type_asm_format(dialect: Symbol, name: Symbol) -> Option<&'static TypeAsmFormat> {
    TYPE_ASM_FORMATS.get(dialect, name)
}

// =============================================================================
// Registry shared by the format kinds
// =============================================================================

/// A format registered for one `(dialect, name)`.
trait Registered: inventory::Collect {
    /// The format kind, for duplicate-registration diagnostics.
    const KIND: &'static str;

    fn key(&self) -> (&'static str, &'static str);
}

/// Formats of one kind by `(dialect, name)`, built once from `inventory`.
struct Registry<F: 'static>(HashMap<(Symbol, Symbol), &'static F>);

impl<F: Registered> Registry<F> {
    fn collect() -> Self {
        let mut map = HashMap::new();
        for format in inventory::iter::<F> {
            let (dialect, name) = format.key();
            let key = (Symbol::from_dynamic(dialect), Symbol::from_dynamic(name));
            if map.insert(key, format).is_some() {
                panic!("duplicate {} registration for '{dialect}.{name}'", F::KIND);
            }
        }
        Self(map)
    }

    fn get(&self, dialect: Symbol, name: Symbol) -> Option<&'static F> {
        self.0.get(&(dialect, name)).copied()
    }
}
