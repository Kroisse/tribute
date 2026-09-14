//! Source-oriented type and effect formatting, including diagnostic ordering.

use std::{cmp::Ordering, fmt};

use itertools::Itertools;
use smallvec::{SmallVec, smallvec};
use trunk_ir::Symbol;

use super::{Effect, EffectRow, Type, TypeKind};

impl fmt::Display for Type<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| write!(f, "{}", self.kind(db)))
            .unwrap_or(Err(fmt::Error))
    }
}

impl fmt::Display for TypeKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| {
            write!(f, "{}", DisplayParts::new(db, Part::Type(self)).format(""))
        })
        .unwrap_or(Err(fmt::Error))
    }
}

impl fmt::Display for Effect<'_> {
    /// Requires an attached Salsa database, as does type formatting.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| {
            write!(f, "{}", DisplayParts::effect(db, self).format(""))
        })
        .unwrap_or(Err(fmt::Error))
    }
}

impl Effect<'_> {
    /// Compare diagnostic spellings without constructing strings.
    ///
    /// This is lexicographic UTF-8 ordering, including punctuation and decimal
    /// indices, exactly as when comparing the formatted strings. Distinct
    /// identities may have equal spellings, so this is deliberately not `Ord`.
    /// The supplied database suffices; no attached database is required.
    pub(crate) fn cmp_for_display(&self, db: &dyn salsa::Database, other: &Self) -> Ordering {
        compare_parts(
            DisplayParts::effect(db, self),
            DisplayParts::effect(db, other),
        )
    }
}

impl fmt::Display for EffectRow<'_> {
    /// Format a canonical diagnostic row, hiding internal row-variable IDs.
    ///
    /// Requires an attached Salsa database. Open tails use the name `e`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        salsa::with_attached_database(|db| {
            let mut effects: SmallVec<[_; 4]> = self.effects(db).iter().collect();
            effects.sort_by(|a, b| a.cmp_for_display(db, b));
            write!(f, "{{{}", effects.iter().format(", "))?;
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

/// Formatting and comparison share this expansion so their spelling cannot
/// diverge. Pending slices are expanded lazily, keeping the stack proportional
/// to nesting depth rather than the number of type arguments.
struct DisplayParts<'a> {
    db: &'a dyn salsa::Database,
    pending: SmallVec<[Part<'a>; 8]>,
}

enum Part<'a> {
    Fragment(Fragment),
    Type(&'a TypeKind<'a>),
    Types(&'a [Type<'a>]),
}

enum Fragment {
    Text(&'static str),
    Symbol(Symbol),
    Index(u32),
}

impl Fragment {
    fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        match self {
            Self::Text(text) => f(text),
            Self::Symbol(symbol) => symbol.with_str(f),
            Self::Index(index) => f(itoa::Buffer::new().format(*index)),
        }
    }
}

impl fmt::Display for Fragment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with_str(|text| f.write_str(text))
    }
}

impl<'a> DisplayParts<'a> {
    fn new(db: &'a dyn salsa::Database, part: Part<'a>) -> Self {
        Self {
            db,
            pending: smallvec![part],
        }
    }

    fn effect(db: &'a dyn salsa::Database, effect: &'a Effect<'a>) -> Self {
        let mut parts = Self {
            db,
            pending: SmallVec::new(),
        };
        if !effect.args.is_empty() {
            parts.parenthesized(&effect.args);
        }
        parts
            .pending
            .push(Part::Fragment(Fragment::Symbol(effect.ability_id.name(db))));
        parts
    }

    fn text(&mut self, text: &'static str) {
        self.pending.push(Part::Fragment(Fragment::Text(text)));
    }

    fn parenthesized(&mut self, types: &'a [Type<'a>]) {
        self.text(")");
        self.pending.push(Part::Types(types));
        self.text("(");
    }
}

impl Iterator for DisplayParts<'_> {
    type Item = Fragment;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(part) = self.pending.pop() {
            match part {
                Part::Fragment(fragment) => {
                    if fragment.with_str(|text| !text.is_empty()) {
                        return Some(fragment);
                    }
                }
                Part::Types(types) => {
                    if let Some((first, rest)) = types.split_first() {
                        if !rest.is_empty() {
                            self.pending.push(Part::Types(rest));
                            self.text(", ");
                        }
                        self.pending.push(Part::Type(first.kind(self.db)));
                    }
                }
                Part::Type(kind) => match kind {
                    TypeKind::Int => self.text("Int"),
                    TypeKind::Nat => self.text("Nat"),
                    TypeKind::Float => self.text("Float"),
                    TypeKind::Bool => self.text("Bool"),
                    TypeKind::Bytes => self.text("Bytes"),
                    TypeKind::Rune => self.text("Rune"),
                    TypeKind::Nil => self.text("Nil"),
                    TypeKind::Never => self.text("Never"),
                    TypeKind::Named { name, args, .. } => {
                        if !args.is_empty() {
                            self.parenthesized(args);
                        }
                        self.pending.push(Part::Fragment(Fragment::Symbol(*name)));
                    }
                    TypeKind::Func { params, result, .. } => {
                        self.pending.push(Part::Type(result.kind(self.db)));
                        self.text(" -> ");
                        self.parenthesized(params);
                        self.text("fn");
                    }
                    TypeKind::Tuple(elems) => {
                        self.parenthesized(elems);
                        self.text("#");
                    }
                    TypeKind::BoundVar { index } => {
                        self.pending.push(Part::Fragment(Fragment::Index(*index)));
                        self.text("_");
                    }
                    TypeKind::LocalBoundVar { index, .. } => {
                        self.pending.push(Part::Fragment(Fragment::Index(*index)));
                        self.text("_local");
                    }
                    TypeKind::UniVar { .. } => self.text("_"),
                    TypeKind::App { ctor, args } => {
                        self.parenthesized(args);
                        self.pending.push(Part::Type(ctor.kind(self.db)));
                    }
                    TypeKind::Continuation { arg, result, .. } => {
                        self.text(")");
                        self.pending.push(Part::Type(result.kind(self.db)));
                        self.text(" -> ");
                        self.pending.push(Part::Type(arg.kind(self.db)));
                        self.text("Continuation(");
                    }
                    TypeKind::Error => self.text("<error>"),
                },
            }
        }
        None
    }
}

/// Compare bytes across fragment boundaries: e.g. `Int` + `(` versus `IntBox`.
fn compare_parts(mut left: DisplayParts<'_>, mut right: DisplayParts<'_>) -> Ordering {
    let (mut a, mut b) = (left.next(), right.next());
    let (mut a_offset, mut b_offset) = (0, 0);
    loop {
        let (Some(a_part), Some(b_part)) = (&a, &b) else {
            return a.is_some().cmp(&b.is_some());
        };
        let (order, a_done, b_done) = a_part.with_str(|a| {
            b_part.with_str(|b| {
                let a = &a.as_bytes()[a_offset..];
                let b = &b.as_bytes()[b_offset..];
                let len = a.len().min(b.len());
                let order = a[..len].cmp(&b[..len]);
                a_offset += len;
                b_offset += len;
                (order, len == a.len(), len == b.len())
            })
        });
        if order != Ordering::Equal {
            return order;
        }
        if a_done {
            a = left.next();
            a_offset = 0;
        }
        if b_done {
            b = right.next();
            b_offset = 0;
        }
    }
}

#[cfg(test)]
mod tests;
