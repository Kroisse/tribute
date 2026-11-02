//! Proc macros for trunk-ir dialect definitions.
//!
//! Provides `#[dialect]` for defining dialect operations with type-safe
//! wrappers, accessors, and builders, plus `#[op_interface]` and `#[op_flag]`
//! for declaring operation interfaces and flags.

use proc_macro::TokenStream as ProcTokenStream;

mod codegen;
mod flag;
mod format;
mod interface;
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
///   `#[asm_format]` operations, or the checked format elements of
///   `#[asm_format(..)]`
/// - For `#[implements(A, path::B)]`, a registration of the wrapper's impl of
///   each listed operation interface; a missing impl is reported at its entry
/// - For `#[flags(Pure, ..)]`, the listed operation flags in the `OpDef`
///
/// For each type `struct`, a `static` `TypeDef` holding the wrapper's
/// `trunk_ir::type_verifier::VerifyType` impl for `#[verify]`, its
/// `trunk_ir::asm_format::TypeAsmFormat` impl for `#[asm_format]`, and its
/// `trunk_ir::asm_format::TypeAliasHint` impl for `#[alias_hint]`, plus a
/// registration of each type interface listed in `#[implements(..)]`.
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

/// Declare an operation interface: a trait that operation wrappers
/// implement and that analyses query for any operation by its name.
///
/// ```ignore
/// #[trunk_ir::op_interface]
/// pub trait Branch: DialectOp {
///     fn successors(&self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError>;
/// }
///
/// let successors = DynBranch::get(ctx, op)?.successors(ctx);
/// ```
///
/// Every method takes `&self` followed by the IR context (`&IrContext` or
/// `&mut IrContext`); methods may be generic over lifetimes only. A result
/// borrowed from the context names the context's lifetime, since elision
/// would tie it to `&self`:
/// `fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath`.
/// Default bodies are kept and dispatched like any other method.
///
/// ## Generated code
///
/// For `trait Name`:
/// - A hidden registration constant on the trait. An operation is
///   registered by submitting `<Op as Name>::__TRUNK_IR_INTERFACE`, which
///   requires `impl Name for Op`.
/// - A hidden `__NameVTable` with one function pointer per method, and its
///   registry keyed by operation name. Registering one operation twice
///   panics when the registry is first used.
/// - `DynName`, a copyable handle to an operation and its implementation:
///   `DynName::get(ctx, op)` finds it, and its methods mirror the trait's
///   without the receiver.
#[proc_macro_attribute]
pub fn op_interface(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match interface::gen_interface(interface::Kind::Op, attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}

/// Declare a type interface: a trait that type wrappers implement and that
/// analyses query for any type by its name.
///
/// ```ignore
/// #[trunk_ir::type_interface]
/// pub trait ManagedRefType: DialectType {}
///
/// let managed = DynManagedRefType::is_implemented(ctx, ty);
/// ```
///
/// The trait and its generated code follow [`op_interface`], with types in
/// place of operations: the registry is keyed by type name, and
/// `DynName::get(ctx, ty)` finds a type's implementation. A method is called
/// on the type's wrapper, so it may be queried only on a type that passes its
/// definition's shape and verifier.
#[proc_macro_attribute]
pub fn type_interface(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match interface::gen_interface(interface::Kind::Type, attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}

/// Declare an operation flag: a property without behavior that
/// `#[flags(..)]` lists on an operation.
///
/// ```ignore
/// #[trunk_ir::op_flag]
/// pub struct Pure;
///
/// let removable = Pure::is_set(ctx, op);
/// ```
///
/// The flag must be a unit struct. The macro adds the `static`
/// `trunk_ir::op_flag::OpFlagDef` that identifies it by address and the
/// `trunk_ir::op_flag::OpFlag` impl that points to it.
#[proc_macro_attribute]
pub fn op_flag(attr: ProcTokenStream, item: ProcTokenStream) -> ProcTokenStream {
    match flag::gen_flag(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(msg) => quote::quote!(compile_error!(#msg);).into(),
    }
}
