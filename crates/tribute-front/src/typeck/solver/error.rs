//! Structured solver failures and their diagnostic messages.

use crate::ast::{AbilityRow, Type, UniVarId};
use crate::resolve::NameDisplay;
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

impl<'db> SolveError<'db> {
    /// The message of this failure, spelling declarations as `names` does.
    pub fn display<'a>(
        &'a self,
        db: &'db dyn salsa::Database,
        names: &'a NameDisplay,
    ) -> SolveErrorDisplay<'a, 'db> {
        SolveErrorDisplay {
            error: self,
            db,
            names,
        }
    }
}

/// A [`SolveError`] formatted for a diagnostic.
pub struct SolveErrorDisplay<'a, 'db> {
    error: &'a SolveError<'db>,
    db: &'db dyn salsa::Database,
    names: &'a NameDisplay,
}

impl std::fmt::Display for SolveErrorDisplay<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (db, names) = (self.db, self.names);
        match self.error {
            SolveError::TypeMismatch { expected, actual } => {
                write!(
                    f,
                    "expected `{}`, found `{}`",
                    expected.display(db, names),
                    actual.display(db, names)
                )
            }
            SolveError::OccursCheck { ty, .. } => {
                write!(
                    f,
                    "infinite type: cannot construct the infinite type `{}`",
                    ty.display(db, names)
                )
            }
            SolveError::RowMismatch { expected, actual } => {
                write!(
                    f,
                    "ability row mismatch: expected `{}`, found `{}`",
                    expected.display(db, names),
                    actual.display(db, names)
                )
            }
            SolveError::AmbiguousAbility { expected, actual } => {
                write!(
                    f,
                    "ambiguous ability: cannot match `{}` with `{}`, more than one instance could correspond; annotate the ability's type arguments",
                    actual.display(db, names),
                    expected.display(db, names)
                )
            }
            SolveError::AbilityArgArityMismatch {
                effect_name,
                expected,
                found,
            } => {
                use tribute_core::fmt::PluralExt;
                effect_name.with_str(|name| {
                    write!(
                        f,
                        "ability `{}` expects {} type argument{}, but {} {} given",
                        names.text(name),
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
