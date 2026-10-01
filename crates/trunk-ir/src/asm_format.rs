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
use crate::ops::DialectOp;
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

/// Global registry mapping (dialect, op_name) → OpAsmFormat, lazily built from inventory.
static ASM_FORMAT_REGISTRY: LazyLock<HashMap<(Symbol, Symbol), &'static OpAsmFormat>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();
        for fmt in inventory::iter::<OpAsmFormat> {
            let dialect = Symbol::from_dynamic(fmt.dialect);
            let op_name = Symbol::from_dynamic(fmt.op_name);
            if map.contains_key(&(dialect, op_name)) {
                panic!(
                    "duplicate OpAsmFormat registration for '{}.{}'",
                    fmt.dialect, fmt.op_name
                );
            }
            map.insert((dialect, op_name), fmt);
        }
        map
    });

/// Look up a registered custom assembly format for the given operation.
pub fn lookup_asm_format(dialect: Symbol, op_name: Symbol) -> Option<&'static OpAsmFormat> {
    ASM_FORMAT_REGISTRY.get(&(dialect, op_name)).copied()
}
