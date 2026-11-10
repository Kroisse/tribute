//! Structural type equality and occurs checks.

use super::{AbilityRow, SolveError, Type, TypeKind, TypeSolver, UniVarId};

impl<'db> TypeSolver<'db> {
    /// Unify two types.
    pub(super) fn unify_types(
        &mut self,
        t1: Type<'db>,
        t2: Type<'db>,
    ) -> Result<(), SolveError<'db>> {
        // Apply current substitution first
        let t1 = self.type_subst.apply(self.db, t1);
        let t2 = self.type_subst.apply(self.db, t2);

        // Same type: done
        if t1 == t2 {
            return Ok(());
        }

        match (t1.kind(self.db), t2.kind(self.db)) {
            // Unify type variables
            (&TypeKind::UniVar { id: id1 }, _) => {
                self.bind_type_var(id1, t2)?;
                Ok(())
            }
            (_, &TypeKind::UniVar { id: id2 }) => {
                self.bind_type_var(id2, t1)?;
                Ok(())
            }

            // BoundVar and LocalBoundVar should never reach the solver — they must be instantiated first.
            (TypeKind::BoundVar { .. }, _)
            | (_, TypeKind::BoundVar { .. })
            | (TypeKind::LocalBoundVar { .. }, _)
            | (_, TypeKind::LocalBoundVar { .. }) => {
                debug_assert!(
                    false,
                    "quantified type variable (BoundVar or LocalBoundVar) reached solver — should have been instantiated"
                );
                Err(SolveError::TypeMismatch {
                    expected: t1,
                    actual: t2,
                })
            }

            // Error types unify with anything
            (&TypeKind::Error, _) | (_, &TypeKind::Error) => Ok(()),

            // Structural unification for compound types
            (
                &TypeKind::Named {
                    id: i1,
                    row_args: ref r1,
                    args: ref a1,
                    ..
                },
                &TypeKind::Named {
                    id: i2,
                    row_args: ref r2,
                    args: ref a2,
                    ..
                },
            ) => {
                if i1 != i2 || r1.len() != r2.len() || a1.len() != a2.len() {
                    return Err(SolveError::TypeMismatch {
                        expected: t1,
                        actual: t2,
                    });
                }
                for (r1, r2) in r1.iter().zip(r2.iter()) {
                    self.unify_rows(*r1, *r2)?;
                }
                for (a1, a2) in a1.iter().zip(a2.iter()) {
                    self.unify_types(*a1, *a2)?;
                }
                Ok(())
            }

            (
                &TypeKind::Func {
                    params: ref p1,
                    result: r1,
                    effect: e1,
                    ..
                },
                &TypeKind::Func {
                    params: ref p2,
                    result: r2,
                    effect: e2,
                    ..
                },
            ) => {
                if p1.len() != p2.len() {
                    return Err(SolveError::TypeMismatch {
                        expected: t1,
                        actual: t2,
                    });
                }
                for (p1, p2) in p1.iter().zip(p2.iter()) {
                    self.unify_types(*p1, *p2)?;
                }
                self.unify_types(r1, r2)?;
                self.unify_rows(e1, e2)?;
                Ok(())
            }

            (TypeKind::Tuple(e1), TypeKind::Tuple(e2)) => {
                if e1.len() != e2.len() {
                    return Err(SolveError::TypeMismatch {
                        expected: t1,
                        actual: t2,
                    });
                }
                for (e1, e2) in e1.iter().zip(e2.iter()) {
                    self.unify_types(*e1, *e2)?;
                }
                Ok(())
            }

            (
                &TypeKind::Continuation {
                    arg: a1,
                    result: r1,
                    effect: e1,
                },
                &TypeKind::Continuation {
                    arg: a2,
                    result: r2,
                    effect: e2,
                },
            ) => {
                self.unify_types(a1, a2)?;
                self.unify_types(r1, r2)?;
                self.unify_rows(e1, e2)?;
                Ok(())
            }

            // Primitive types must match exactly
            _ => Err(SolveError::TypeMismatch {
                expected: t1,
                actual: t2,
            }),
        }
    }

    /// Bind a type variable to a type.
    pub(super) fn bind_type_var(
        &mut self,
        var: UniVarId<'db>,
        ty: Type<'db>,
    ) -> Result<(), SolveError<'db>> {
        // Occurs check: prevent infinite types
        if self.occurs_in(var, ty) {
            return Err(SolveError::OccursCheck { var, ty });
        }
        self.type_subst.insert(var, ty);
        Ok(())
    }

    /// Check if a type variable occurs in a type (for occurs check).
    ///
    /// Bound variables are followed, since a type argument reached through
    /// a row substitution has not had the type substitution applied.
    pub(super) fn occurs_in(&self, var: UniVarId<'db>, ty: Type<'db>) -> bool {
        match ty.kind(self.db) {
            TypeKind::UniVar { id } => {
                *id == var
                    || self
                        .type_subst
                        .get(*id)
                        .is_some_and(|bound| self.occurs_in(var, bound))
            }
            TypeKind::Named { row_args, args, .. } => {
                row_args
                    .iter()
                    .any(|row| self.occurs_in_ability_row(var, *row))
                    || args.iter().any(|a| self.occurs_in(var, *a))
            }
            TypeKind::Func {
                params,
                result,
                effect,
                ..
            } => {
                params.iter().any(|p| self.occurs_in(var, *p))
                    || self.occurs_in(var, *result)
                    || self.occurs_in_ability_row(var, *effect)
            }
            TypeKind::Tuple(elements) => elements.iter().any(|e| self.occurs_in(var, *e)),
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                self.occurs_in(var, *arg)
                    || self.occurs_in(var, *result)
                    || self.occurs_in_ability_row(var, *effect)
            }
            _ => false,
        }
    }

    /// Check a substituted ability row's type arguments for a type variable.
    pub(super) fn occurs_in_ability_row(
        &self,
        var: UniVarId<'db>,
        effect: AbilityRow<'db>,
    ) -> bool {
        self.row_subst
            .apply(self.db, effect)
            .instances(self.db)
            .iter()
            .any(|effect| effect.args.iter().any(|arg| self.occurs_in(var, *arg)))
    }

    /// Whether some substitution of the effects' type variables makes them
    /// the same instance. Unlike [`Self::types_unifiable`], the arguments
    /// share one substitution, so `S(a, a)` and `S(Int, Bool)` do not
    /// coincide.
    pub(crate) fn effects_may_coincide(
        db: &'db dyn salsa::Database,
        a: &crate::ast::AbilityInstance<'db>,
        b: &crate::ast::AbilityInstance<'db>,
    ) -> bool {
        if a.ability_id != b.ability_id || a.args.len() != b.args.len() {
            return false;
        }
        let mut solver = Self::new(db);
        a.args
            .iter()
            .zip(&b.args)
            .all(|(x, y)| solver.unify_types(*x, *y).is_ok())
    }

    /// Check if two types can be unified without modifying substitution.
    ///
    /// This is a quick check that doesn't perform actual unification.
    pub(super) fn types_unifiable(&self, t1: Type<'db>, t2: Type<'db>) -> bool {
        let t1 = self.type_subst.apply(self.db, t1);
        let t2 = self.type_subst.apply(self.db, t2);

        if t1 == t2 {
            return true;
        }

        match (t1.kind(self.db), t2.kind(self.db)) {
            // Type variables can unify with anything
            (TypeKind::UniVar { .. }, _) | (_, TypeKind::UniVar { .. }) => true,
            // Error type unifies with anything
            (TypeKind::Error, _) | (_, TypeKind::Error) => true,
            // Same kind, check recursively
            (TypeKind::Int, TypeKind::Int)
            | (TypeKind::Nat, TypeKind::Nat)
            | (TypeKind::Float, TypeKind::Float)
            | (TypeKind::Bool, TypeKind::Bool)
            | (TypeKind::Rune, TypeKind::Rune)
            | (TypeKind::Nil, TypeKind::Nil) => true,
            (
                TypeKind::Named {
                    id: i1,
                    row_args: r1,
                    args: a1,
                    ..
                },
                TypeKind::Named {
                    id: i2,
                    row_args: r2,
                    args: a2,
                    ..
                },
            ) => {
                i1 == i2
                    && r1.len() == r2.len()
                    && a1.len() == a2.len()
                    && r1
                        .iter()
                        .zip(r2.iter())
                        .all(|(x, y)| self.rows_unifiable(*x, *y))
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(x, y)| self.types_unifiable(*x, *y))
            }
            (TypeKind::Tuple(elems1), TypeKind::Tuple(elems2)) => {
                elems1.len() == elems2.len()
                    && elems1
                        .iter()
                        .zip(elems2.iter())
                        .all(|(x, y)| self.types_unifiable(*x, *y))
            }
            (
                TypeKind::Func {
                    params: p1,
                    result: r1,
                    effect: e1,
                },
                TypeKind::Func {
                    params: p2,
                    result: r2,
                    effect: e2,
                },
            ) => {
                p1.len() == p2.len()
                    && p1
                        .iter()
                        .zip(p2.iter())
                        .all(|(x, y)| self.types_unifiable(*x, *y))
                    && self.types_unifiable(*r1, *r2)
                    && self.rows_unifiable(*e1, *e2)
            }
            (
                TypeKind::Continuation {
                    arg: a1,
                    result: r1,
                    effect: e1,
                },
                TypeKind::Continuation {
                    arg: a2,
                    result: r2,
                    effect: e2,
                },
            ) => {
                self.types_unifiable(*a1, *a2)
                    && self.types_unifiable(*r1, *r2)
                    && self.rows_unifiable(*e1, *e2)
            }
            _ => false,
        }
    }

    /// Check whether two rows may unify, without modifying substitution.
    ///
    /// Like [`Self::types_unifiable`], this never rejects a pair that
    /// unifies. Its only rejection is a label of one side with no candidate
    /// instance on the other when that other side is closed; two open rows,
    /// whether their tails are shared or distinct, rigid or not, are never
    /// rejected. So two
    /// instances of one ability whose function arguments have rows that can
    /// never be equal, such as `{Ask}` and `{Tell}`, stay distinct.
    pub(super) fn rows_unifiable(&self, r1: AbilityRow<'db>, r2: AbilityRow<'db>) -> bool {
        let r1 = self.normalize_row(r1);
        let r2 = self.normalize_row(r2);
        if r1 == r2 {
            return true;
        }
        let covered = |labels: AbilityRow<'db>, by: AbilityRow<'db>| {
            labels.instances(self.db).iter().all(|effect| {
                by.instances(self.db).iter().any(|candidate| {
                    candidate.ability_id == effect.ability_id
                        && candidate.args.len() == effect.args.len()
                        && candidate
                            .args
                            .iter()
                            .zip(&effect.args)
                            .all(|(a, b)| self.types_unifiable(*a, *b))
                })
            })
        };
        match (r1.rest(self.db), r2.rest(self.db)) {
            (None, None) => covered(r1, r2) && covered(r2, r1),
            (Some(_), None) => covered(r1, r2),
            (None, Some(_)) => covered(r2, r1),
            // Another position may bind an open tail to the labels the other
            // side names, so two open rows are never rejected.
            (Some(_), Some(_)) => true,
        }
    }
}
