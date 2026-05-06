//! Proc macros for trunk-ir dialect definitions.
//!
//! Provides `#[dialect]` attribute macro for defining
//! dialect operations with type-safe wrappers, accessors, and constructors.

use proc_macro::TokenStream as ProcTokenStream;

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
