//! Type verifiers that dialects declare for their types.
//!
//! A verifier checks the rules a type's data must satisfy beyond its generic
//! shape, the way an operation's `#[verify]` hook does for operations.
//! IR validation runs the verifier of every interned type whose
//! [`TypeDef`](crate::TypeDef) has one.

use crate::ops::{DialectType, VerificationError};
use crate::{IrContext, TypeRef};

/// Rules a type's data must satisfy beyond its generic shape.
///
/// A type declared in `#[dialect]` opts in with `#[verify]`; a hand-written
/// definition adds it with [`TypeDef::with_verifier`](crate::TypeDef::with_verifier).
/// Unlike an operation's [`Verify`](crate::ops::Verify), it takes the raw
/// type and returns the wrapper, because a wrapper accepts only types that
/// pass it.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is declared with `#[verify]` but does not implement `VerifyType`",
    label = "`#[verify]` requires this implementation",
    note = "add `impl VerifyType for {Self} {{ fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, VerificationError> {{ .. }} }}`"
)]
pub trait VerifyType: DialectType {
    /// Check a type of this kind, given by name, and wrap it.
    fn verify(ctx: &IrContext, ty: TypeRef) -> Result<Self, VerificationError>;
}

/// `T`'s verifier as a [`TypeDef`](crate::TypeDef) stores it.
pub(crate) fn erased_verifier<T: VerifyType>(
    ctx: &IrContext,
    ty: TypeRef,
) -> Result<(), VerificationError> {
    T::verify(ctx, ty).map(|_| ())
}
