//! Source-oriented type and effect formatting.

use std::fmt;

use itertools::Itertools;

use super::{AbilityInstance, AbilityRow, Type, TypeKind, TypeParam};

impl fmt::Display for Type<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| write!(f, "{}", self.kind(db)))
            .unwrap_or(Err(fmt::Error))
    }
}

impl fmt::Display for TypeKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| write_kind(db, self, &[], f)).unwrap_or(Err(fmt::Error))
    }
}

/// A type whose `BoundVar`s refer to the type parameters of one scheme,
/// formatted with those parameters' names.
pub(crate) struct WithTypeParams<'a, 'db> {
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    params: &'a [TypeParam],
}

impl<'db> Type<'db> {
    /// Format this type, spelling each `BoundVar` by the name of the
    /// parameter in `params` it refers to.
    pub(crate) fn display_with_params<'a>(
        self,
        db: &'db dyn salsa::Database,
        params: &'a [TypeParam],
    ) -> WithTypeParams<'a, 'db> {
        WithTypeParams {
            db,
            ty: self,
            params,
        }
    }
}

impl fmt::Display for WithTypeParams<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_kind(self.db, self.ty.kind(self.db), self.params, f)
    }
}

fn write_kind<'db>(
    db: &'db dyn salsa::Database,
    kind: &TypeKind<'db>,
    params: &[TypeParam],
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    match kind {
        TypeKind::Int => f.write_str("Int"),
        TypeKind::Nat => f.write_str("Nat"),
        TypeKind::Float => f.write_str("Float"),
        TypeKind::Bool => f.write_str("Bool"),
        TypeKind::Bytes => f.write_str("Bytes"),
        TypeKind::Rune => f.write_str("Rune"),
        TypeKind::Nil => f.write_str("Nil"),
        TypeKind::Never => f.write_str("Never"),
        TypeKind::Named { name, args, .. } => {
            name.with_str(|s| f.write_str(s))?;
            if !args.is_empty() {
                write!(f, "({})", list(db, args, params))
            } else {
                Ok(())
            }
        }
        TypeKind::Func {
            params: inputs,
            result,
            ..
        } => write!(
            f,
            "fn({}) -> {}",
            list(db, inputs, params),
            result.display_with_params(db, params)
        ),
        TypeKind::Tuple(elems) => write!(f, "#({})", list(db, elems, params)),
        TypeKind::BoundVar { index } => {
            match params
                .get(*index as usize)
                .and_then(|param| param.name.as_ref())
            {
                Some(name) => name.with_str(|s| f.write_str(s)),
                None => write!(f, "_{}", index),
            }
        }
        TypeKind::LocalBoundVar { index, .. } => write!(f, "_local{}", index),
        TypeKind::UniVar { .. } => f.write_str("_"),
        TypeKind::App { ctor, args } => {
            write!(
                f,
                "{}({})",
                ctor.display_with_params(db, params),
                list(db, args, params)
            )
        }
        TypeKind::Continuation { arg, result, .. } => write!(
            f,
            "Continuation({} -> {})",
            arg.display_with_params(db, params),
            result.display_with_params(db, params)
        ),
        TypeKind::Error => f.write_str("<error>"),
    }
}

fn list<'a, 'db: 'a>(
    db: &'db dyn salsa::Database,
    types: &'a [Type<'db>],
    params: &'a [TypeParam],
) -> impl fmt::Display + 'a {
    types
        .iter()
        .map(move |ty| ty.display_with_params(db, params))
        .format(", ")
}

impl fmt::Display for AbilityInstance<'_> {
    /// Formats this effect for diagnostic messages.
    ///
    /// Produces strings like `"State(Int)"` or `"Console"`.
    ///
    /// Requires an attached salsa database on the current thread
    /// (i.e., must be called inside `db.attach()`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| {
            let name = self.ability_id.name(db);
            if self.args.is_empty() {
                name.with_str(|s| f.write_str(s))
            } else {
                let args = self
                    .args
                    .iter()
                    .format_with(", ", |ty, f| f(&format_args!("{}", ty.kind(db))));
                name.with_str(|s| write!(f, "{}({args})", s))
            }
        })
        .unwrap_or(Err(fmt::Error))
    }
}

impl fmt::Display for AbilityRow<'_> {
    /// Format a canonical diagnostic row, hiding internal row-variable IDs.
    ///
    /// Open tails use the name `e`. Without an attached Salsa database,
    /// displays `<ability row>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| {
            let effects = self.instances(db);
            // The displayed strings are both the sort keys and the output.
            let sorted = effects.iter().map(ToString::to_string).sorted();
            write!(f, "{{{}", sorted.format(", "))?;
            if self.rest(db).is_some() {
                if !effects.is_empty() {
                    f.write_str(", ")?;
                }
                f.write_str("e")?;
            }
            f.write_str("}")
        })
        .unwrap_or_else(|| f.write_str("<ability row>"))
    }
}

#[cfg(test)]
mod tests;
