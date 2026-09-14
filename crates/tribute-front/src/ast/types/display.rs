//! Source-oriented type and effect formatting.

use std::fmt;

use itertools::Itertools;

use super::{Effect, EffectRow, Type, TypeKind};

impl fmt::Display for Type<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| write!(f, "{}", self.kind(db)))
            .unwrap_or(Err(fmt::Error))
    }
}

impl fmt::Display for TypeKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| match self {
            Self::Int => f.write_str("Int"),
            Self::Nat => f.write_str("Nat"),
            Self::Float => f.write_str("Float"),
            Self::Bool => f.write_str("Bool"),
            Self::Bytes => f.write_str("Bytes"),
            Self::Rune => f.write_str("Rune"),
            Self::Nil => f.write_str("Nil"),
            Self::Never => f.write_str("Never"),
            Self::Named { name, args, .. } => {
                name.with_str(|s| f.write_str(s))?;
                if !args.is_empty() {
                    let args = args
                        .iter()
                        .format_with(", ", |ty, f| f(&format_args!("{}", ty.kind(db))));
                    write!(f, "({args})")
                } else {
                    Ok(())
                }
            }
            Self::Func { params, result, .. } => {
                let params = params
                    .iter()
                    .format_with(", ", |ty, f| f(&format_args!("{}", ty.kind(db))));
                write!(f, "fn({params}) -> {}", result.kind(db))
            }
            Self::Tuple(elems) => {
                let elems = elems
                    .iter()
                    .format_with(", ", |ty, f| f(&format_args!("{}", ty.kind(db))));
                write!(f, "#({elems})")
            }
            Self::BoundVar { index } => write!(f, "_{}", index),
            Self::LocalBoundVar { index, .. } => write!(f, "_local{}", index),
            Self::UniVar { .. } => f.write_str("_"),
            Self::App { ctor, args } => {
                let args = args
                    .iter()
                    .format_with(", ", |ty, f| f(&format_args!("{}", ty.kind(db))));
                write!(f, "{}({args})", ctor.kind(db))
            }
            Self::Continuation { arg, result, .. } => {
                write!(f, "Continuation({} -> {})", arg.kind(db), result.kind(db))
            }
            Self::Error => f.write_str("<error>"),
        })
        .unwrap_or(Err(fmt::Error))
    }
}

impl fmt::Display for Effect<'_> {
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

impl fmt::Display for EffectRow<'_> {
    /// Format a canonical diagnostic row, hiding internal row-variable IDs.
    ///
    /// Requires an attached Salsa database. Open tails use the name `e`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| {
            let effects = self.effects(db);
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
        .expect("EffectRow formatting requires an attached Salsa database")
    }
}

#[cfg(test)]
mod tests;
