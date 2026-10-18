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
///     fn func(sym_name: Attr<String>, r#type: Attr<Type>) {
///         #[region(body?)] {}
///     }
///
///     fn call(callee: Attr<SymbolRef>, args: Variadic<_>) -> Variadic<_> {}
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
/// For the dialect, a `DIALECT_NAME` constant. For each operation
/// `fn op_name(...)`:
/// - `struct Wrapper(OpRef)` — wrapper struct, named in `UpperCamelCase`
/// - A `static` `OpDef` whose address identifies the operation, and
///   `impl DialectOp for Wrapper` whose `NAME` points to it — matching
///   compares name handles
/// - Operand, result, attribute, region/successor accessors
/// - A builder started by `Wrapper::operands(..)` (empty for operations
///   without operands) and finished by `.build(ctx, location)`
/// - `OpDef` registration: the declarative `OpSchema`, plus a call to the
///   wrapper's `trunk_ir::ops::Verify` impl for `#[verify]` operations and
///   the wrapper's `trunk_ir::asm_format::OpAsmFormat` print/parse hooks for
///   `#[asm_format]` operations
///
/// For each type `struct`, a `static` `TypeDef` holding the wrapper's
/// `trunk_ir::type_verifier::VerifyType` impl for `#[verify]` and its
/// `trunk_ir::asm_format::TypeAsmFormat` impl for `#[asm_format]`.
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
/// #[trunk_ir::canonicalize_fold(Addi)]
/// pub(crate) fn fold_addi(ctx: &IrContext, op: OpRef) -> Option<FoldResult> { ... }
/// ```
///
/// The attribute payload is the operation's typed wrapper, so a misspelled
/// operation fails to compile. The original function item is preserved
/// unchanged; the macro only emits an adjacent `inventory::submit!` block.
#[proc_macro_attribute]
pub fn canonicalize_fold(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match canonicalize::gen_fold(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}
