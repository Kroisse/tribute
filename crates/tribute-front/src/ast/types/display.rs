//! Source-oriented type and effect formatting.
//!
//! Every formatter spells declaration names relative to the package being
//! compiled (`new-plans/modules.md`, "진단의 이름 표기"), so each takes the
//! [`NameDisplay`] of that package.

use std::fmt;

use itertools::Itertools;

use super::{AbilityInstance, AbilityRow, Type, TypeKind, TypeParam};
use crate::resolve::NameDisplay;

/// A type formatted for a diagnostic, its `BoundVar`s spelled by the names
/// of the type parameters of one scheme.
pub struct TypeDisplay<'a, 'db> {
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    params: &'a [TypeParam],
    names: &'a NameDisplay,
}

impl<'db> Type<'db> {
    /// Format this type for a diagnostic of the package `names` spells.
    pub fn display<'a>(
        self,
        db: &'db dyn salsa::Database,
        names: &'a NameDisplay,
    ) -> TypeDisplay<'a, 'db> {
        self.display_with_params(db, &[], names)
    }

    /// Format this type, spelling each `BoundVar` by the name of the
    /// parameter in `params` it refers to.
    pub fn display_with_params<'a>(
        self,
        db: &'db dyn salsa::Database,
        params: &'a [TypeParam],
        names: &'a NameDisplay,
    ) -> TypeDisplay<'a, 'db> {
        TypeDisplay {
            db,
            ty: self,
            params,
            names,
        }
    }
}

impl fmt::Display for TypeDisplay<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_kind(self.db, self.ty.kind(self.db), self.params, self.names, f)
    }
}

fn write_kind<'db>(
    db: &'db dyn salsa::Database,
    kind: &TypeKind<'db>,
    params: &[TypeParam],
    names: &NameDisplay,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let display = |ty: &Type<'db>| ty.display_with_params(db, params, names);
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
            name.with_str(|s| f.write_str(names.text(s)))?;
            if !args.is_empty() {
                write!(f, "({})", list(db, args, params, names))
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
            list(db, inputs, params, names),
            display(result)
        ),
        TypeKind::Tuple(elems) => write!(f, "#({})", list(db, elems, params, names)),
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
        TypeKind::Continuation { arg, result, .. } => {
            write!(f, "Continuation({} -> {})", display(arg), display(result))
        }
        TypeKind::Error => f.write_str("<error>"),
    }
}

fn list<'a, 'db: 'a>(
    db: &'db dyn salsa::Database,
    types: &'a [Type<'db>],
    params: &'a [TypeParam],
    names: &'a NameDisplay,
) -> impl fmt::Display + 'a {
    types
        .iter()
        .map(move |ty| ty.display_with_params(db, params, names))
        .format(", ")
}

/// An effect formatted for a diagnostic, such as `State(Int)` or `Console`.
pub struct AbilityInstanceDisplay<'a, 'db> {
    db: &'db dyn salsa::Database,
    instance: &'a AbilityInstance<'db>,
    names: &'a NameDisplay,
}

impl<'db> AbilityInstance<'db> {
    /// Format this effect for a diagnostic of the package `names` spells.
    pub fn display<'a>(
        &'a self,
        db: &'db dyn salsa::Database,
        names: &'a NameDisplay,
    ) -> AbilityInstanceDisplay<'a, 'db> {
        AbilityInstanceDisplay {
            db,
            instance: self,
            names,
        }
    }
}

impl fmt::Display for AbilityInstanceDisplay<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.instance.ability_id.name(self.db);
        if self.instance.args.is_empty() {
            name.with_str(|s| f.write_str(s))
        } else {
            let args = list(self.db, &self.instance.args, &[], self.names);
            name.with_str(|s| write!(f, "{}({args})", s))
        }
    }
}

/// A canonical ability row formatted for a diagnostic, hiding internal
/// row-variable IDs. An open tail is named `e`.
pub struct AbilityRowDisplay<'a, 'db> {
    db: &'db dyn salsa::Database,
    row: AbilityRow<'db>,
    names: &'a NameDisplay,
}

impl<'db> AbilityRow<'db> {
    /// Format this row for a diagnostic of the package `names` spells.
    pub fn display<'a>(
        self,
        db: &'db dyn salsa::Database,
        names: &'a NameDisplay,
    ) -> AbilityRowDisplay<'a, 'db> {
        AbilityRowDisplay {
            db,
            row: self,
            names,
        }
    }
}

impl fmt::Display for AbilityRowDisplay<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let effects = self.row.instances(self.db);
        // The displayed strings are both the sort keys and the output.
        let sorted = effects
            .iter()
            .map(|effect| effect.display(self.db, self.names).to_string())
            .sorted();
        write!(f, "{{{}", sorted.format(", "))?;
        if self.row.rest(self.db).is_some() {
            if !effects.is_empty() {
                f.write_str(", ")?;
            }
            f.write_str("e")?;
        }
        f.write_str("}")
    }
}

#[cfg(test)]
mod tests;
