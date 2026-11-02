//! Structural traversal of types, ability rows, and type schemes.
//!
//! A [`TypeVisitor`] reads a [`TypeVisitable`] value and a [`TypeFolder`]
//! rebuilds a [`TypeFoldable`] one. Their default methods recurse
//! structurally; an implementation overrides the method for the nodes its
//! policy concerns and calls `super_visit_with` or `try_super_fold_with` to
//! continue into the children. An override that does not call them skips the
//! subtree.
//!
//! Traversal order is part of the contract (see `new-plans/generics.md`,
//! "타입 순회와 binder별 치환"):
//!
//! - `Named` and `Tuple`: arguments in order.
//! - `Func`: parameters, result, then the ability row.
//! - `Continuation`: argument, result, then the ability row.
//! - An ability row: each instance's arguments, instances in order. The row
//!   tail is not a type; a folder or visitor that concerns it reads it in its
//!   own `fold_row`/`visit_row`.
//! - A [`TypeScheme`]: the body, each retained union's sources then result,
//!   then each removal's source, removed, and result rows. The scheme's binder
//!   lists are not changed.
//!
//! A fold returns its input handle when no child changed, so an unchanged
//! type or row is never interned again.

use std::borrow::Cow;
use std::convert::Infallible;
use std::ops::ControlFlow;

use super::{AbilityInstance, AbilityRow, RowRemoval, RowUnion, Type, TypeKind, TypeScheme};

/// Rebuilds types bottom-up. Infallible folders use [`Infallible`] as their
/// error and call [`TypeFoldable::fold_with`].
pub trait TypeFolder<'db> {
    type Error;

    fn db(&self) -> &'db dyn salsa::Database;

    fn fold_type(&mut self, ty: Type<'db>) -> Result<Type<'db>, Self::Error> {
        ty.try_super_fold_with(self)
    }

    fn fold_row(&mut self, row: AbilityRow<'db>) -> Result<AbilityRow<'db>, Self::Error> {
        row.try_super_fold_with(self)
    }
}

/// Reads types top-down, stopping at the first `Break`.
pub trait TypeVisitor<'db> {
    type Break;

    fn db(&self) -> &'db dyn salsa::Database;

    fn visit_type(&mut self, ty: Type<'db>) -> ControlFlow<Self::Break> {
        ty.super_visit_with(self)
    }

    fn visit_row(&mut self, row: AbilityRow<'db>) -> ControlFlow<Self::Break> {
        row.super_visit_with(self)
    }
}

/// A value that holds types a [`TypeVisitor`] reaches.
pub trait TypeVisitable<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break>;
}

/// A value that a [`TypeFolder`] rebuilds. It folds the same types, in the
/// same order, as its [`TypeVisitable`] implementation visits.
pub trait TypeFoldable<'db>: TypeVisitable<'db> + Sized {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error>;

    fn fold_with<F: TypeFolder<'db, Error = Infallible> + ?Sized>(self, folder: &mut F) -> Self {
        let Ok(folded) = self.try_fold_with(folder);
        folded
    }
}

impl<'db> TypeVisitable<'db> for Type<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        visitor.visit_type(*self)
    }
}

impl<'db> TypeFoldable<'db> for Type<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.fold_type(self)
    }
}

impl<'db> TypeVisitable<'db> for AbilityRow<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        visitor.visit_row(*self)
    }
}

impl<'db> TypeFoldable<'db> for AbilityRow<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.fold_row(self)
    }
}

impl<'db> TypeVisitable<'db> for AbilityInstance<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        self.args.visit_with(visitor)
    }
}

impl<'db> TypeFoldable<'db> for AbilityInstance<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(Self {
            args: self.args.try_fold_with(folder)?,
            ..self
        })
    }
}

impl<'db> TypeVisitable<'db> for RowUnion<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        self.sources.visit_with(visitor)?;
        visitor.visit_row(self.result)
    }
}

impl<'db> TypeFoldable<'db> for RowUnion<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(Self {
            sources: self.sources.try_fold_with(folder)?,
            result: folder.fold_row(self.result)?,
        })
    }
}

impl<'db> TypeVisitable<'db> for RowRemoval<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        self.rows()
            .into_iter()
            .try_for_each(|row| visitor.visit_row(row))
    }
}

impl<'db> TypeFoldable<'db> for RowRemoval<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(Self {
            source: folder.fold_row(self.source)?,
            removed: folder.fold_row(self.removed)?,
            result: folder.fold_row(self.result)?,
        })
    }
}

/// An owned vector folds its items in place, without copying them.
impl<'db, T: TypeVisitable<'db>> TypeVisitable<'db> for Vec<T> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        self.iter().try_for_each(|item| item.visit_with(visitor))
    }
}

impl<'db, T: TypeFoldable<'db>> TypeFoldable<'db> for Vec<T> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.into_iter()
            .map(|item| item.try_fold_with(folder))
            .collect()
    }
}

impl<'db> TypeVisitable<'db> for TypeScheme<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        let db = visitor.db();
        visitor.visit_type(self.body(db))?;
        self.row_unions(db)
            .iter()
            .try_for_each(|union| union.visit_with(visitor))?;
        self.row_removals(db)
            .iter()
            .try_for_each(|removal| removal.visit_with(visitor))
    }
}

impl<'db> TypeFoldable<'db> for TypeScheme<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        let db = folder.db();
        let body = folder.fold_type(self.body(db))?;
        let row_unions = fold_slice(self.row_unions(db), folder)?;
        let row_removals = fold_slice(self.row_removals(db), folder)?;
        if body == self.body(db) && row_unions.is_none() && row_removals.is_none() {
            return Ok(self);
        }
        let mut builder = self.to_builder(db);
        builder.body = body;
        if let Some(unions) = row_unions {
            builder.row_unions = Cow::Owned(unions);
        }
        if let Some(removals) = row_removals {
            builder.row_removals = Cow::Owned(removals);
        }
        Ok(builder.build(db))
    }
}

impl<'db> super::TypeSchemeBuilder<'db> {
    /// Fold the body and the retained constraints, in scheme order. Borrowed
    /// constraint lists are copied only when a row changes.
    pub fn try_fold_types<F: TypeFolder<'db> + ?Sized>(
        mut self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        self.body = folder.fold_type(self.body)?;
        if let Some(unions) = fold_slice(&self.row_unions, folder)? {
            self.row_unions = Cow::Owned(unions);
        }
        if let Some(removals) = fold_slice(&self.row_removals, folder)? {
            self.row_removals = Cow::Owned(removals);
        }
        Ok(self)
    }
}

impl<'db> Type<'db> {
    /// Fold the children of this type, rebuilding it only if one changed.
    pub fn try_super_fold_with<F: TypeFolder<'db> + ?Sized>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let db = folder.db();
        let kind = match self.kind(db) {
            TypeKind::Named { id, name, args } => {
                let Some(args) = fold_slice(args, folder)? else {
                    return Ok(self);
                };
                TypeKind::Named {
                    id: *id,
                    name: name.clone(),
                    args,
                }
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let new_params = fold_slice(params, folder)?;
                let new_result = folder.fold_type(*result)?;
                let new_effect = folder.fold_row(*effect)?;
                if new_params.is_none() && new_result == *result && new_effect == *effect {
                    return Ok(self);
                }
                TypeKind::Func {
                    params: new_params.unwrap_or_else(|| params.clone()),
                    result: new_result,
                    effect: new_effect,
                }
            }
            TypeKind::Tuple(elements) => {
                let Some(elements) = fold_slice(elements, folder)? else {
                    return Ok(self);
                };
                TypeKind::Tuple(elements)
            }
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                let new_arg = folder.fold_type(*arg)?;
                let new_result = folder.fold_type(*result)?;
                let new_effect = folder.fold_row(*effect)?;
                if new_arg == *arg && new_result == *result && new_effect == *effect {
                    return Ok(self);
                }
                TypeKind::Continuation {
                    arg: new_arg,
                    result: new_result,
                    effect: new_effect,
                }
            }
            TypeKind::Int
            | TypeKind::Nat
            | TypeKind::Float
            | TypeKind::Bool
            | TypeKind::Bytes
            | TypeKind::Rune
            | TypeKind::Nil
            | TypeKind::Never
            | TypeKind::BoundVar { .. }
            | TypeKind::LocalBoundVar { .. }
            | TypeKind::UniVar { .. }
            | TypeKind::Error => return Ok(self),
        };
        Ok(Type::new(db, kind))
    }

    /// Visit the children of this type.
    pub fn super_visit_with<V: TypeVisitor<'db> + ?Sized>(
        self,
        visitor: &mut V,
    ) -> ControlFlow<V::Break> {
        match self.kind(visitor.db()) {
            TypeKind::Named { args, .. } | TypeKind::Tuple(args) => args.visit_with(visitor),
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                params.visit_with(visitor)?;
                visitor.visit_type(*result)?;
                visitor.visit_row(*effect)
            }
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                visitor.visit_type(*arg)?;
                visitor.visit_type(*result)?;
                visitor.visit_row(*effect)
            }
            TypeKind::Int
            | TypeKind::Nat
            | TypeKind::Float
            | TypeKind::Bool
            | TypeKind::Bytes
            | TypeKind::Rune
            | TypeKind::Nil
            | TypeKind::Never
            | TypeKind::BoundVar { .. }
            | TypeKind::LocalBoundVar { .. }
            | TypeKind::UniVar { .. }
            | TypeKind::Error => ControlFlow::Continue(()),
        }
    }
}

impl<'db> AbilityRow<'db> {
    /// Fold the arguments of each instance, keeping the tail. The row is
    /// interned again only if an argument changed.
    pub fn try_super_fold_with<F: TypeFolder<'db> + ?Sized>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let db = folder.db();
        Ok(match fold_slice(self.instances(db), folder)? {
            Some(instances) => AbilityRow::new(db, instances, self.rest(db)),
            None => self,
        })
    }

    /// Visit the arguments of each instance.
    pub fn super_visit_with<V: TypeVisitor<'db> + ?Sized>(
        self,
        visitor: &mut V,
    ) -> ControlFlow<V::Break> {
        self.instances(visitor.db())
            .iter()
            .try_for_each(|instance| instance.visit_with(visitor))
    }
}

/// Fold each item in order. `None` means no item changed, so the caller keeps
/// its existing value; otherwise the result holds every folded item.
fn fold_slice<'db, T, F>(items: &[T], folder: &mut F) -> Result<Option<Vec<T>>, F::Error>
where
    T: TypeFoldable<'db> + Clone + PartialEq,
    F: TypeFolder<'db> + ?Sized,
{
    for (index, item) in items.iter().enumerate() {
        let folded = item.clone().try_fold_with(folder)?;
        if folded != *item {
            let mut out = Vec::with_capacity(items.len());
            out.extend_from_slice(&items[..index]);
            out.push(folded);
            for item in &items[index + 1..] {
                out.push(item.clone().try_fold_with(folder)?);
            }
            return Ok(Some(out));
        }
    }
    Ok(None)
}

/// Apply `map` to each type the traversal reaches at the top level, without
/// recursing into the types it returns.
pub(super) struct MapTypes<'db, M> {
    pub(super) db: &'db dyn salsa::Database,
    pub(super) map: M,
}

impl<'db, M: FnMut(Type<'db>) -> Type<'db>> TypeFolder<'db> for MapTypes<'db, M> {
    type Error = Infallible;

    fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    fn fold_type(&mut self, ty: Type<'db>) -> Result<Type<'db>, Infallible> {
        Ok((self.map)(ty))
    }
}

/// Call `visit` on each type the traversal reaches at the top level.
pub(crate) struct ForEachType<'db, V> {
    pub(crate) db: &'db dyn salsa::Database,
    pub(crate) visit: V,
}

impl<'db, V: FnMut(Type<'db>)> TypeVisitor<'db> for ForEachType<'db, V> {
    type Break = Infallible;

    fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    fn visit_type(&mut self, ty: Type<'db>) -> ControlFlow<Infallible> {
        (self.visit)(ty);
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityId, RowVar, TypeParam};
    use salsa_test_macros::salsa_test;

    /// Replaces `Int` with `Nat` and counts the types it is asked to fold.
    struct IntToNat<'db> {
        db: &'db dyn salsa::Database,
        folded: usize,
    }

    impl<'db> TypeFolder<'db> for IntToNat<'db> {
        type Error = Infallible;

        fn db(&self) -> &'db dyn salsa::Database {
            self.db
        }

        fn fold_type(&mut self, ty: Type<'db>) -> Result<Type<'db>, Infallible> {
            self.folded += 1;
            if *ty.kind(self.db) == TypeKind::Int {
                return Ok(Type::new(self.db, TypeKind::Nat));
            }
            ty.try_super_fold_with(self)
        }
    }

    fn scheme_with_rows<'db>(db: &'db dyn salsa::Database, arg: Type<'db>) -> TypeScheme<'db> {
        let ability = AbilityId::source(db, trunk_ir::SymbolPath::new(["State"]));
        let row = |rest| {
            AbilityRow::new(
                db,
                vec![AbilityInstance {
                    ability_id: ability,
                    args: vec![arg],
                }],
                rest,
            )
        };
        let tail = RowVar { id: 1 };
        let body = Type::new(
            db,
            TypeKind::Func {
                params: vec![Type::new(db, TypeKind::Bool)],
                result: Type::new(db, TypeKind::Nil),
                effect: row(Some(tail)),
            },
        );
        TypeScheme::builder(vec![TypeParam::anonymous()], vec![tail], body)
            .row_unions(vec![RowUnion {
                sources: vec![row(None), row(Some(tail))],
                result: row(Some(tail)),
            }])
            .row_removals(vec![RowRemoval {
                source: row(Some(tail)),
                removed: row(None),
                result: AbilityRow::open(db, tail),
            }])
            .build(db)
    }

    #[salsa_test]
    fn unchanged_fold_keeps_interned_handles(db: &dyn salsa::Database) {
        let scheme = scheme_with_rows(db, Type::new(db, TypeKind::Rune));
        let mut folder = IntToNat { db, folded: 0 };
        assert_eq!(scheme.fold_with(&mut folder), scheme);
        assert!(folder.folded > 0);
    }

    #[salsa_test]
    fn fold_reaches_effect_arguments_in_retained_rows(db: &dyn salsa::Database) {
        let int = Type::new(db, TypeKind::Int);
        let nat = Type::new(db, TypeKind::Nat);
        let folded = scheme_with_rows(db, int).fold_with(&mut IntToNat { db, folded: 0 });
        assert_eq!(folded, scheme_with_rows(db, nat));
    }

    /// Records the order in which a visitor reaches the types and rows.
    struct Order<'db> {
        db: &'db dyn salsa::Database,
        seen: Vec<String>,
    }

    impl<'db> TypeVisitor<'db> for Order<'db> {
        type Break = ();

        fn db(&self) -> &'db dyn salsa::Database {
            self.db
        }

        fn visit_type(&mut self, ty: Type<'db>) -> ControlFlow<()> {
            if let Some(name) = ty.kind(self.db).primitive_name() {
                self.seen.push(name.to_owned());
            }
            ty.super_visit_with(self)
        }

        fn visit_row(&mut self, row: AbilityRow<'db>) -> ControlFlow<()> {
            self.seen.push("row".to_owned());
            row.super_visit_with(self)
        }
    }

    #[salsa_test]
    fn visitor_follows_the_scheme_order(db: &dyn salsa::Database) {
        let scheme = scheme_with_rows(db, Type::new(db, TypeKind::Int));
        let mut order = Order {
            db,
            seen: Vec::new(),
        };
        assert!(scheme.visit_with(&mut order).is_continue());
        let row = ["row", "Int"];
        let expected: Vec<_> = ["Bool", "Nil"]
            .into_iter()
            .chain(row)
            .chain(row.into_iter().cycle().take(6))
            .chain(row.into_iter().cycle().take(4))
            .chain(["row"])
            .collect();
        assert_eq!(order.seen, expected);
    }
}
