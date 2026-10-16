//! Structured solver failures and their diagnostic messages.

use crate::ast::{AbilityRow, Type, UniVarId};
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
    /// Ability row mismatch.
    RowMismatch {
        expected: AbilityRow<'db>,
        actual: AbilityRow<'db>,
    },
    /// Several exact-identity candidates remain possible until more types solve.
    AmbiguousAbility {
        expected: AbilityRow<'db>,
        actual: AbilityRow<'db>,
    },
    /// Effect type argument arity mismatch.
    AbilityArgArityMismatch {
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
            Self::RowMismatch { expected, actual } => {
                write!(
                    f,
                    "ability row mismatch: expected `{expected}`, found `{actual}`"
                )
            }
            Self::AmbiguousAbility { expected, actual } => {
                write!(
                    f,
                    "ambiguous ability: cannot match `{actual}` with `{expected}`, more than one instance could correspond; annotate the ability's type arguments"
                )
            }
            Self::AbilityArgArityMismatch {
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
