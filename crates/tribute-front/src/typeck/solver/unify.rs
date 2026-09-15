//! Structural type equality and occurs checks.

use super::{EffectRow, SolveError, Type, TypeKind, TypeSolver, UniVarId};

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
                    args: ref a1,
                    ..
                },
                &TypeKind::Named {
                    id: i2,
                    args: ref a2,
                    ..
                },
            ) => {
                if i1 != i2 || a1.len() != a2.len() {
                    return Err(SolveError::TypeMismatch {
                        expected: t1,
                        actual: t2,
                    });
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
                &TypeKind::App {
                    ctor: c1,
                    args: ref a1,
                },
                &TypeKind::App {
                    ctor: c2,
                    args: ref a2,
                },
            ) => {
                self.unify_types(c1, c2)?;
                if a1.len() != a2.len() {
                    return Err(SolveError::TypeMismatch {
                        expected: t1,
                        actual: t2,
                    });
                }
                for (a1, a2) in a1.iter().zip(a2.iter()) {
                    self.unify_types(*a1, *a2)?;
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
    pub(super) fn occurs_in(&self, var: UniVarId<'db>, ty: Type<'db>) -> bool {
        match ty.kind(self.db) {
            TypeKind::UniVar { id } => *id == var,
            TypeKind::Named { args, .. } => args.iter().any(|a| self.occurs_in(var, *a)),
            TypeKind::Func {
                params,
                result,
                effect,
                ..
            } => {
                params.iter().any(|p| self.occurs_in(var, *p))
                    || self.occurs_in(var, *result)
                    || self.occurs_in_effect_row(var, *effect)
            }
            TypeKind::Tuple(elements) => elements.iter().any(|e| self.occurs_in(var, *e)),
            TypeKind::App { ctor, args } => {
                self.occurs_in(var, *ctor) || args.iter().any(|a| self.occurs_in(var, *a))
            }
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                self.occurs_in(var, *arg)
                    || self.occurs_in(var, *result)
                    || self.occurs_in_effect_row(var, *effect)
            }
            _ => false,
        }
    }

    /// Check a substituted effect row's type arguments for a type variable.
    pub(super) fn occurs_in_effect_row(&self, var: UniVarId<'db>, effect: EffectRow<'db>) -> bool {
        self.row_subst
            .apply(self.db, effect)
            .effects(self.db)
            .iter()
            .any(|effect| effect.args.iter().any(|arg| self.occurs_in(var, *arg)))
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
            | (TypeKind::Bytes, TypeKind::Bytes)
            | (TypeKind::Rune, TypeKind::Rune)
            | (TypeKind::Nil, TypeKind::Nil) => true,
            (
                TypeKind::Named {
                    id: i1, args: a1, ..
                },
                TypeKind::Named {
                    id: i2, args: a2, ..
                },
            ) => {
                i1 == i2
                    && a1.len() == a2.len()
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
                    ..
                },
                TypeKind::Func {
                    params: p2,
                    result: r2,
                    ..
                },
            ) => {
                p1.len() == p2.len()
                    && p1
                        .iter()
                        .zip(p2.iter())
                        .all(|(x, y)| self.types_unifiable(*x, *y))
                    && self.types_unifiable(*r1, *r2)
            }
            (TypeKind::App { ctor: c1, args: a1 }, TypeKind::App { ctor: c2, args: a2 }) => {
                self.types_unifiable(*c1, *c2)
                    && a1.len() == a2.len()
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(x, y)| self.types_unifiable(*x, *y))
            }
            (
                TypeKind::Continuation {
                    arg: a1,
                    result: r1,
                    ..
                },
                TypeKind::Continuation {
                    arg: a2,
                    result: r2,
                    ..
                },
            ) => self.types_unifiable(*a1, *a2) && self.types_unifiable(*r1, *r2),
            _ => false,
        }
    }
}
