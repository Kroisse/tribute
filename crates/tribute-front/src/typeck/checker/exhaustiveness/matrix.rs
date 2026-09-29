//! Pattern-matrix usefulness (Maranget, "Warnings for pattern matching").
//!
//! This module knows nothing about the database: callers lower source
//! patterns into [`Pat`] and describe each nominal type as a [`Family`].
//! The kind of a column is decided by the constructors that appear in it, so
//! no field or scrutinee type is needed.

use std::fmt;

use itertools::Itertools;
use trunk_ir::Symbol;

use crate::ast::LiteralPattern;

/// Upper bound on recursive steps before the analysis gives up.
const STEP_BUDGET: usize = 100_000;

/// Index of a [`Family`] in the slice given to [`Analyzer::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FamilyId(pub(super) usize);

/// The constructors of one nominal type: an enum, or a struct with one.
#[derive(Clone, Debug)]
pub(super) struct Family {
    pub(super) variants: Vec<VariantInfo>,
}

#[derive(Clone, Debug)]
pub(super) struct VariantInfo {
    pub(super) name: Symbol,
    pub(super) arity: usize,
}

#[derive(Clone, Debug)]
pub(super) enum Pat {
    Wild,
    Ctor(Ctor, Vec<Pat>),
    /// `[a, b]` when `rest` is false, `[a, b, ..]` when it is true.
    List {
        prefix: Vec<Pat>,
        rest: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Ctor {
    Bool(bool),
    Unit,
    Tuple(usize),
    Variant {
        family: FamilyId,
        index: usize,
    },
    /// A number or string literal; its domain is never covered.
    Literal(LiteralPattern),
    /// Lists of exactly this length.
    ListLen(usize),
    /// Lists of at least this length.
    ListAtLeast(usize),
}

/// A value shape no row matches.
#[derive(Clone, Debug)]
pub(super) enum Witness {
    Wild,
    Ctor(Ctor, Vec<Witness>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GiveUp {
    /// Patterns in one column disagree about their type; typing reports it.
    Conflict,
    /// Literals and enum variants share a column.
    Unsupported,
    /// The step budget ran out.
    Budget,
}

/// What the constructors in one column say about its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    /// Only wildcards.
    Wild,
    Bool,
    Unit,
    Tuple(usize),
    Family(FamilyId),
    Literal,
    /// Lists, split at this length.
    List(usize),
}

pub(super) struct Analyzer<'a> {
    families: &'a [Family],
    steps: usize,
}

impl<'a> Analyzer<'a> {
    pub(super) fn new(families: &'a [Family]) -> Self {
        Self { families, steps: 0 }
    }

    /// Whether some value matches `q` but none of `rows`.
    pub(super) fn is_useful(&mut self, rows: &[Vec<Pat>], q: &[Pat]) -> Result<bool, GiveUp> {
        self.tick()?;
        let Some(head) = q.first() else {
            return Ok(rows.is_empty());
        };
        let column = self.column(rows.iter().map(|row| &row[0]).chain([head]))?;
        match head {
            Pat::Wild => match self.signature(column) {
                Some(signature) => {
                    for ctor in signature {
                        if self.is_useful_specialized(rows, q, &ctor)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }
                None => self.is_useful(&default_rows(rows), &q[1..]),
            },
            Pat::Ctor(ctor, _) => self.is_useful_specialized(rows, q, ctor),
            Pat::List { .. } => {
                for ctor in self.signature(column).unwrap_or_default() {
                    if self.is_useful_specialized(rows, q, &ctor)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    fn is_useful_specialized(
        &mut self,
        rows: &[Vec<Pat>],
        q: &[Pat],
        ctor: &Ctor,
    ) -> Result<bool, GiveUp> {
        match self.specialize_row(q, ctor)? {
            Some(q) => {
                let rows = self.specialize(rows, ctor)?;
                self.is_useful(&rows, &q)
            }
            None => Ok(false),
        }
    }

    /// Up to `limit` rows of `width` witnesses that no row of `rows` matches.
    pub(super) fn missing(
        &mut self,
        rows: &[Vec<Pat>],
        width: usize,
        limit: usize,
    ) -> Result<Vec<Vec<Witness>>, GiveUp> {
        self.tick()?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        if width == 0 {
            return Ok(if rows.is_empty() {
                vec![Vec::new()]
            } else {
                Vec::new()
            });
        }
        let column = self.column(rows.iter().map(|row| &row[0]))?;
        let Some(signature) = self.signature(column) else {
            let mut found = self.missing(&default_rows(rows), width - 1, limit)?;
            for witness in &mut found {
                witness.insert(0, Witness::Wild);
            }
            return Ok(found);
        };
        let mut found = Vec::new();
        for ctor in signature {
            let arity = self.arity(&ctor);
            let specialized = self.specialize(rows, &ctor)?;
            for mut fields in self.missing(&specialized, arity + width - 1, limit - found.len())? {
                let rest = fields.split_off(arity);
                let mut witness = vec![Witness::Ctor(ctor.clone(), fields)];
                witness.extend(rest);
                found.push(witness);
            }
            if found.len() >= limit {
                break;
            }
        }
        Ok(found)
    }

    pub(super) fn render<'w>(&'w self, witness: &'w Witness) -> impl fmt::Display + 'w {
        Rendered {
            families: self.families,
            witness,
        }
    }

    fn tick(&mut self) -> Result<(), GiveUp> {
        self.steps += 1;
        if self.steps > STEP_BUDGET {
            Err(GiveUp::Budget)
        } else {
            Ok(())
        }
    }

    fn column<'p>(&self, heads: impl Iterator<Item = &'p Pat>) -> Result<Column, GiveUp> {
        let mut column = Column::Wild;
        let mut exact_max = None;
        let mut rest_max = 0;
        for head in heads {
            let kind = match head {
                Pat::Wild => continue,
                Pat::Ctor(Ctor::Bool(_), _) => Column::Bool,
                Pat::Ctor(Ctor::Unit, _) => Column::Unit,
                Pat::Ctor(Ctor::Tuple(n), _) => Column::Tuple(*n),
                Pat::Ctor(Ctor::Variant { family, .. }, _) => Column::Family(*family),
                Pat::Ctor(Ctor::Literal(_), _) => Column::Literal,
                Pat::Ctor(Ctor::ListLen(_) | Ctor::ListAtLeast(_), _) => {
                    return Err(GiveUp::Conflict);
                }
                Pat::List { prefix, rest } => {
                    if *rest {
                        rest_max = rest_max.max(prefix.len());
                    } else {
                        exact_max = exact_max.max(Some(prefix.len()));
                    }
                    Column::List(0)
                }
            };
            column = match (column, kind) {
                (Column::Wild, kind) => kind,
                (column, kind) if column == kind => column,
                (Column::Literal, Column::Family(_)) | (Column::Family(_), Column::Literal) => {
                    return Err(GiveUp::Unsupported);
                }
                _ => return Err(GiveUp::Conflict),
            };
        }
        if column == Column::List(0) {
            // Every length at or past the split matches the same rows.
            let split = exact_max.map_or(0, |max| max + 1).max(rest_max);
            column = Column::List(split);
        }
        Ok(column)
    }

    /// All constructors of a column, or `None` if they cannot be listed.
    fn signature(&self, column: Column) -> Option<Vec<Ctor>> {
        Some(match column {
            Column::Wild | Column::Literal => return None,
            Column::Bool => vec![Ctor::Bool(true), Ctor::Bool(false)],
            Column::Unit => vec![Ctor::Unit],
            Column::Tuple(n) => vec![Ctor::Tuple(n)],
            Column::Family(family) => (0..self.families[family.0].variants.len())
                .map(|index| Ctor::Variant { family, index })
                .collect(),
            Column::List(split) => (0..split)
                .map(Ctor::ListLen)
                .chain([Ctor::ListAtLeast(split)])
                .collect(),
        })
    }

    fn arity(&self, ctor: &Ctor) -> usize {
        match ctor {
            Ctor::Bool(_) | Ctor::Unit | Ctor::Literal(_) => 0,
            Ctor::Tuple(n) | Ctor::ListLen(n) | Ctor::ListAtLeast(n) => *n,
            Ctor::Variant { family, index } => self.families[family.0].variants[*index].arity,
        }
    }

    fn specialize(&self, rows: &[Vec<Pat>], ctor: &Ctor) -> Result<Vec<Vec<Pat>>, GiveUp> {
        let mut specialized = Vec::new();
        for row in rows {
            if let Some(row) = self.specialize_row(row, ctor)? {
                specialized.push(row);
            }
        }
        Ok(specialized)
    }

    /// The row's fields under `ctor` followed by its tail, or `None` if the
    /// row's head excludes `ctor`.
    fn specialize_row(&self, row: &[Pat], ctor: &Ctor) -> Result<Option<Vec<Pat>>, GiveUp> {
        let arity = self.arity(ctor);
        let (head, tail) = row.split_first().ok_or(GiveUp::Conflict)?;
        let fields = match head {
            Pat::Wild => vec![Pat::Wild; arity],
            Pat::Ctor(head, fields) if head == ctor => {
                if fields.len() != arity {
                    return Err(GiveUp::Conflict);
                }
                fields.clone()
            }
            Pat::Ctor(..) => return Ok(None),
            Pat::List { prefix, rest } => {
                let matches = match ctor {
                    Ctor::ListLen(len) if *rest => prefix.len() <= *len,
                    Ctor::ListLen(len) => prefix.len() == *len,
                    Ctor::ListAtLeast(len) => *rest && prefix.len() <= *len,
                    _ => return Err(GiveUp::Conflict),
                };
                if !matches {
                    return Ok(None);
                }
                let mut fields = prefix.clone();
                fields.resize(arity, Pat::Wild);
                fields
            }
        };
        let mut specialized = fields;
        specialized.extend_from_slice(tail);
        Ok(Some(specialized))
    }
}

/// Tails of the rows whose head is a wildcard.
fn default_rows(rows: &[Vec<Pat>]) -> Vec<Vec<Pat>> {
    rows.iter()
        .filter(|row| matches!(row[0], Pat::Wild))
        .map(|row| row[1..].to_vec())
        .collect()
}

struct Rendered<'a> {
    families: &'a [Family],
    witness: &'a Witness,
}

impl Rendered<'_> {
    fn nested<'w>(&'w self, witness: &'w Witness) -> Rendered<'w> {
        Rendered {
            families: self.families,
            witness,
        }
    }
}

impl fmt::Display for Rendered<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Witness::Ctor(ctor, fields) = self.witness else {
            return f.write_str("_");
        };
        let has_fields = !fields.is_empty();
        let fields = || fields.iter().map(|field| self.nested(field)).format(", ");
        match ctor {
            Ctor::Bool(true) => f.write_str("True"),
            Ctor::Bool(false) => f.write_str("False"),
            Ctor::Unit => f.write_str("()"),
            Ctor::Tuple(_) => write!(f, "#({})", fields()),
            Ctor::Variant { family, index } => {
                let name = self.families[family.0].variants[*index].name;
                if !has_fields {
                    write!(f, "{name}")
                } else {
                    write!(f, "{name}({})", fields())
                }
            }
            Ctor::Literal(_) | Ctor::ListAtLeast(0) => f.write_str("_"),
            Ctor::ListLen(_) => write!(f, "[{}]", fields()),
            Ctor::ListAtLeast(_) => write!(f, "[{}, ..]", fields()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wild() -> Pat {
        Pat::Wild
    }

    fn bool_(value: bool) -> Pat {
        Pat::Ctor(Ctor::Bool(value), Vec::new())
    }

    fn tuple(fields: Vec<Pat>) -> Pat {
        Pat::Ctor(Ctor::Tuple(fields.len()), fields)
    }

    fn nat(value: u64) -> Pat {
        Pat::Ctor(Ctor::Literal(LiteralPattern::Nat(value)), Vec::new())
    }

    fn list(prefix: Vec<Pat>, rest: bool) -> Pat {
        Pat::List { prefix, rest }
    }

    /// `Option` as family 0: `None`, `Some(_)`.
    fn option() -> Vec<Family> {
        vec![Family {
            variants: vec![
                VariantInfo {
                    name: Symbol::new("None"),
                    arity: 0,
                },
                VariantInfo {
                    name: Symbol::new("Some"),
                    arity: 1,
                },
            ],
        }]
    }

    fn none() -> Pat {
        Pat::Ctor(
            Ctor::Variant {
                family: FamilyId(0),
                index: 0,
            },
            Vec::new(),
        )
    }

    fn some(field: Pat) -> Pat {
        Pat::Ctor(
            Ctor::Variant {
                family: FamilyId(0),
                index: 1,
            },
            vec![field],
        )
    }

    /// Rendered witnesses for a one-column case over `arms`.
    fn missing(families: &[Family], arms: Vec<Pat>) -> Result<Vec<String>, GiveUp> {
        let mut analyzer = Analyzer::new(families);
        let rows: Vec<Vec<Pat>> = arms.into_iter().map(|arm| vec![arm]).collect();
        let found = analyzer.missing(&rows, 1, 4)?;
        Ok(found
            .iter()
            .map(|witness| analyzer.render(&witness[0]).to_string())
            .collect())
    }

    /// Indices of arms that earlier arms already cover.
    fn unreachable(families: &[Family], arms: Vec<Pat>) -> Vec<usize> {
        let mut analyzer = Analyzer::new(families);
        let mut rows = Vec::new();
        let mut found = Vec::new();
        for (index, arm) in arms.into_iter().enumerate() {
            let row = vec![arm];
            if !analyzer.is_useful(&rows, &row).unwrap() {
                found.push(index);
            }
            rows.push(row);
        }
        found
    }

    #[test]
    fn bool_coverage() {
        assert!(
            missing(&[], vec![bool_(true), bool_(false)])
                .unwrap()
                .is_empty()
        );
        assert_eq!(missing(&[], vec![bool_(true)]).unwrap(), ["False"]);
        assert_eq!(missing(&[], vec![]).unwrap(), ["_"]);
    }

    #[test]
    fn tuple_coverage() {
        let arms = vec![
            tuple(vec![bool_(true), wild()]),
            tuple(vec![wild(), bool_(true)]),
        ];
        assert_eq!(missing(&[], arms).unwrap(), ["#(False, False)"]);
    }

    #[test]
    fn nested_variant_is_not_covered_by_one_field_value() {
        let families = option();
        let arms = vec![none(), some(bool_(true))];
        assert_eq!(missing(&families, arms).unwrap(), ["Some(False)"]);
        let arms = vec![none(), some(bool_(true)), some(bool_(false))];
        assert!(missing(&families, arms).unwrap().is_empty());
        assert_eq!(missing(&families, vec![some(wild())]).unwrap(), ["None"]);
    }

    #[test]
    fn list_lengths() {
        let arms = vec![list(vec![wild()], true)];
        assert_eq!(missing(&[], arms).unwrap(), ["[]"]);
        let arms = vec![list(vec![nat(0)], false)];
        assert_eq!(missing(&[], arms).unwrap(), ["[]", "[_]", "[_, _, ..]"]);
        let arms = vec![list(vec![], false), list(vec![wild()], true)];
        assert!(missing(&[], arms).unwrap().is_empty());
        let arms = vec![
            list(vec![], false),
            list(vec![wild()], false),
            list(vec![wild(), wild(), wild()], true),
        ];
        assert_eq!(missing(&[], arms).unwrap(), ["[_, _]"]);
    }

    #[test]
    fn literals_need_a_wildcard() {
        assert_eq!(missing(&[], vec![nat(0), nat(1)]).unwrap(), ["_"]);
        assert!(missing(&[], vec![nat(0), wild()]).unwrap().is_empty());
        let string = Pat::Ctor(
            Ctor::Literal(LiteralPattern::String("a".to_owned())),
            Vec::new(),
        );
        assert_eq!(missing(&[], vec![string]).unwrap(), ["_"]);
        let unit = Pat::Ctor(Ctor::Unit, Vec::new());
        assert!(missing(&[], vec![unit]).unwrap().is_empty());
    }

    #[test]
    fn witnesses_are_limited() {
        // Five witnesses exist: `#(False, _, _, _, _)`, `#(True, False, _, _, _)`, ...
        let arms = vec![tuple((0..5).map(|_| bool_(true)).collect())];
        assert_eq!(missing(&[], arms).unwrap().len(), 4);
    }

    #[test]
    fn unreachable_arms() {
        assert_eq!(unreachable(&[], vec![wild(), bool_(true)]), [1]);
        let families = option();
        assert_eq!(
            unreachable(&families, vec![some(wild()), some(wild())]),
            [1]
        );
        assert_eq!(
            unreachable(
                &families,
                vec![some(bool_(true)), some(bool_(false)), some(wild())]
            ),
            [2]
        );
        assert!(unreachable(&[], vec![nat(1), nat(2), wild()]).is_empty());
        let arms = vec![
            list(vec![], false),
            list(vec![wild()], true),
            list(vec![], false),
        ];
        assert_eq!(unreachable(&[], arms), [2]);
    }

    #[test]
    fn mixed_columns_give_up() {
        let families = option();
        assert_eq!(
            missing(&families, vec![none(), nat(0)]),
            Err(GiveUp::Unsupported)
        );
        let arms = vec![tuple(vec![wild()]), tuple(vec![wild(), wild()])];
        assert_eq!(missing(&[], arms), Err(GiveUp::Conflict));
        assert_eq!(
            missing(&[], vec![bool_(true), nat(0)]),
            Err(GiveUp::Conflict)
        );
    }

    #[test]
    fn budget_gives_up() {
        // `True` in one column and `_` elsewhere, then all `False`: complete,
        // but every column splits the search in two.
        let mut arms: Vec<Pat> = (0..40)
            .map(|index| {
                tuple(
                    (0..40)
                        .map(|column| if index == column { bool_(true) } else { wild() })
                        .collect(),
                )
            })
            .collect();
        arms.push(tuple((0..40).map(|_| bool_(false)).collect()));
        assert_eq!(missing(&[], arms), Err(GiveUp::Budget));
    }
}
