//! Proc macros for trunk-ir dialect definitions.
//!
//! Provides `#[dialect]` for defining dialect operations with type-safe
//! wrappers, accessors, and builders, plus `#[canonicalize_fold]` for
//! registering canonicalize-pass folds next to the function definition.

use proc_macro::TokenStream as ProcTokenStream;

mod canonicalize;
mod codegen;
mod parse;

/// Define dialect operations and types.
///
/// ```ignore
/// #[dialect]
/// mod func {
///     fn func(sym_name: Attr<Symbol>, r#type: Attr<Type>) {
///         #[region(body?)] {}
///     }
///
///     fn call(callee: Attr<Symbol>, args: Variadic<_>) -> Variadic<_> {}
///
///     fn r#return(values: Variadic<_>) {}
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
/// - A builder started by `OpName::operands(..)` (empty for operations
///   without operands) and finished by `.build(ctx, location)`
/// - `OpSchema` registration; `#[verify]` on the operation adds a call to
///   the wrapper's `trunk_ir::ops::Verify` impl after the generated checks
#[proc_macro_attribute]
pub fn dialect(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match dialect_impl(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
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
