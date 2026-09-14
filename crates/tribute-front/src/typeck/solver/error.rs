//! Structured solver failures and stable effect-row formatting.

use super::format_effect_row;
use crate::ast::{EffectRow, Type, UniVarId};
use crate::typeck::constraint::ConstraintOrigin;

/// Error during constraint solving.
#[derive(Clone, Debug)]
pub enum SolveError<'db> {
    /// Type mismatch: expected one type, got another.
    TypeMismatch {
        expected: Type<'db>,
        actual: Type<'db>,
    },
    /// Occurs check failed (infinite type).
    OccursCheck { var: UniVarId<'db>, ty: Type<'db> },
    /// Effect row mismatch.
    RowMismatch {
        expected: EffectRow<'db>,
        actual: EffectRow<'db>,
    },
    /// Several exact-identity candidates remain possible until more types solve.
    AmbiguousEffect {
        expected: EffectRow<'db>,
        actual: EffectRow<'db>,
    },
    /// Effect type argument arity mismatch.
    EffectArgArityMismatch {
        effect_name: trunk_ir::Symbol,
        expected: usize,
        found: usize,
    },
}

/// A solver error together with the source construct that introduced the
/// failing constraint, when one was recorded by the checker.
#[derive(Clone, Debug)]
pub struct LocatedSolveError<'db> {
    pub error: SolveError<'db>,
    pub origin: Option<ConstraintOrigin>,
}

impl std::fmt::Display for SolveError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch { expected, actual } => {
                write!(f, "expected `{}`, found `{}`", expected, actual)
            }
            Self::OccursCheck { ty, .. } => {
                write!(
                    f,
                    "infinite type: cannot construct the infinite type `{}`",
                    ty
                )
            }
            Self::RowMismatch { expected, actual } | Self::AmbiguousEffect { expected, actual } => {
                salsa::with_attached_database(|db| {
                    write!(
                        f,
                        "effect mismatch: expected `{}`, found `{}`",
                        format_effect_row(db, *expected),
                        format_effect_row(db, *actual)
                    )
                })
                .unwrap_or(Err(std::fmt::Error))
            }
            Self::EffectArgArityMismatch {
                effect_name,
                expected,
                found,
            } => {
                use tribute_core::fmt::PluralExt;
                effect_name.with_str(|name| {
                    write!(
                        f,
                        "ability `{}` expects {} type argument{}, but {} {} given",
                        name,
                        expected,
                        expected.plural(),
                        found,
                        found.verb("was", "were"),
                    )
                })
            }
        }
    }
}
