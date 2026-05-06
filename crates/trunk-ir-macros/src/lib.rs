//! Proc macros for trunk-ir dialect definitions.
//!
//! Provides `#[dialect]` for defining dialect operations with type-safe
//! wrappers/accessors/constructors, plus `#[canonicalize_fold]` and
//! `#[canonicalize_pattern]` for registering canonicalize-pass entries
//! next to the function definition.

use proc_macro::TokenStream as ProcTokenStream;

mod canonicalize;
mod codegen;
mod parse;

/// Define dialect operations and types.
///
/// ```ignore
/// #[dialect]
/// mod func {
///     #[attr(sym_name: Symbol, r#type: Type)]
///     fn func() {
///         #[region(body)] {}
///     }
///
///     #[attr(callee: Symbol)]
///     fn call(#[rest] args: ()) -> result {}
///
///     fn r#return(#[rest] values: ()) {}
/// }
/// ```
///
/// The macro emits absolute `::trunk_ir::...` paths. In-crate uses rely
/// on `extern crate self as trunk_ir;` (declared at the trunk-ir crate
/// root); external crates resolve `trunk_ir` through the normal
/// dependency.
///
/// ## Generated code
///
/// For each operation `fn op_name(...)`:
/// - `DIALECT_NAME()` / `OP_NAME()` — Symbol helper functions
/// - `struct OpName(OpRef)` — wrapper struct
/// - `impl DialectOp for OpName` — type-safe matching
/// - Operand, result, attribute, region/successor accessors
/// - Constructor function `op_name(ctx, location, ...)`
#[proc_macro_attribute]
pub fn dialect(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match dialect_impl(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}

/// Deprecated alias for [`dialect`]. Use `#[dialect]` instead.
#[proc_macro_attribute]
pub fn arena_dialect(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    dialect(attr, item)
}

fn dialect_impl(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> Result<proc_macro2::TokenStream, String> {
    let module = parse::parse_input(attr, item)?;
    let crate_path = quote::quote!(::trunk_ir);
    Ok(codegen::generate(&crate_path, &module))
}

/// Register a per-op fold for the canonicalize pass.
///
/// ```ignore
/// #[trunk_ir::canonicalize_fold(arith.addi)]
/// pub(crate) fn fold_addi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { ... }
/// ```
///
/// The attribute payload is `<dialect_ident>.<op_ident>`. Raw
/// identifiers (`r#const`, `r#return`) are stripped so the registered
/// op name matches the printed form. The original function item is
/// preserved unchanged; the macro only emits an adjacent
/// `inventory::submit!` block.
#[proc_macro_attribute]
pub fn canonicalize_fold(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match canonicalize::gen_fold(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}

/// Register a full `RewritePattern` for the canonicalize pass — used
/// when a rewrite needs more than the per-op fold shape allows
/// (e.g. multi-op region splicing).
///
/// ```ignore
/// #[trunk_ir::canonicalize_pattern]
/// fn make_if_const_fold() -> Box<dyn RewritePattern> { Box::new(IfConstFold) }
/// ```
///
/// The attribute takes no arguments. The original function item is
/// preserved unchanged; the macro only emits an adjacent
/// `inventory::submit!` block.
#[proc_macro_attribute]
pub fn canonicalize_pattern(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match canonicalize::gen_pattern(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}
