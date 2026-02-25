//! Proc macros for trunk-ir dialect definitions.
//!
//! Provides `#[arena_dialect]` attribute macro for defining arena-based
//! dialect operations with type-safe wrappers, accessors, and constructors.

use proc_macro::TokenStream as ProcTokenStream;

mod codegen;
mod parse;

/// Define arena-based dialect operations.
///
/// ```ignore
/// #[arena_dialect]
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
/// ## Crate path
///
/// - `#[arena_dialect] mod ...` — defaults to `trunk_ir` (for external crates)
/// - `#[arena_dialect(crate = crate)] mod ...` — for use within `trunk-ir` itself
///
/// ## Generated code
///
/// For each operation `fn op_name(...)`:
/// - `DIALECT_NAME()` / `OP_NAME()` — Symbol helper functions
/// - `struct OpName(OpRef)` — wrapper struct
/// - `impl ArenaDialectOp for OpName` — type-safe matching
/// - Operand, result, attribute, region/successor accessors
/// - Constructor function `op_name(ctx, location, ...)`
#[proc_macro_attribute]
pub fn arena_dialect(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match arena_dialect_impl(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}

fn arena_dialect_impl(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> Result<proc_macro2::TokenStream, String> {
    let (crate_path, module) = parse::parse_input(attr, item)?;
    Ok(codegen::generate(&crate_path, &module))
}
