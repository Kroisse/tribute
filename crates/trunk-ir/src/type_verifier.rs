//! Type verifiers that dialects declare for their types.
//!
//! A verifier checks the rules a type's data must satisfy beyond its generic
//! shape, the way an operation's `#[verify]` hook does for operations.
//! IR validation runs the verifier of every interned type whose
//! [`TypeDef`](crate::TypeDef) has one.

use crate::ops::DialectType;
use crate::{IrContext, TypeRef};

/// Rules a type's data must satisfy beyond its generic shape.
///
/// A type declared in `#[dialect]` opts in with `#[verify]`; a hand-written
/// definition sets `verifier: Some(<T as VerifyType>::verify)`. It takes the
/// raw type rather than the wrapper, because a wrapper may accept only types
/// that pass this check.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is declared with `#[verify]` but does not implement `VerifyType`",
    label = "`#[verify]` requires this implementation",
    note = "add `impl VerifyType for {Self} {{ fn verify(ctx: &IrContext, ty: TypeRef) -> Result<(), String> {{ .. }} }}`"
)]
pub trait VerifyType: DialectType {
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<(), String>;
}

/// The verifier a [`TypeDef`](crate::TypeDef) stores.
pub type TypeVerifierFn = fn(&IrContext, TypeRef) -> Result<(), String>;
