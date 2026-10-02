//! Type verifiers that dialects register for their types.
//!
//! A verifier checks the rules a type's data must satisfy beyond its generic
//! shape, the way an operation's `#[verify]` hook does for operations.
//! IR validation runs the verifier of every interned type that has one.

use std::sync::LazyLock;

use crate::Symbol;
use crate::ops::DialectType;
use crate::registry::{Registered, Registry};
use crate::{IrContext, TypeRef};

/// The verifier of one type, registered via `inventory::submit!` at the
/// dialect definition site.
pub struct TypeVerifier {
    dialect: &'static str,
    type_name: &'static str,
    /// Checks one interned type of this kind.
    pub verify_fn: fn(&IrContext, TypeRef) -> Result<(), String>,
}

impl TypeVerifier {
    /// Verifier for the type wrapped by `T`.
    pub const fn new<T: DialectType>(
        verify_fn: fn(&IrContext, TypeRef) -> Result<(), String>,
    ) -> Self {
        Self {
            dialect: T::DIALECT_NAME,
            type_name: T::TYPE_NAME,
            verify_fn,
        }
    }
}

inventory::collect!(TypeVerifier);

impl Registered for TypeVerifier {
    const KIND: &'static str = "TypeVerifier";

    fn key(&self) -> (&'static str, &'static str) {
        (self.dialect, self.type_name)
    }
}

static TYPE_VERIFIERS: LazyLock<Registry<TypeVerifier>> = LazyLock::new(Registry::collect);

/// Look up the registered verifier for the given type kind.
pub fn lookup_type_verifier(dialect: Symbol, name: Symbol) -> Option<&'static TypeVerifier> {
    TYPE_VERIFIERS.get(dialect, name)
}
