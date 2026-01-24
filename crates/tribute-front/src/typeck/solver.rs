//! Type constraint solver.
//!
//! Solves type constraints using union-find based unification.

use std::collections::HashMap;

use crate::ast::{EffectRow, EffectVar, Type, TypeKind};

use super::constraint::{Constraint, ConstraintSet};

/// Error during constraint solving.
#[derive(Clone, Debug)]
pub enum SolveError<'db> {
    /// Type mismatch: expected one type, got another.
    TypeMismatch {
        expected: Type<'db>,
        actual: Type<'db>,
    },
    /// Occurs check failed (infinite type).
    OccursCheck { var: u64, ty: Type<'db> },
    /// Effect row mismatch.
    RowMismatch {
        expected: EffectRow<'db>,
        actual: EffectRow<'db>,
    },
}

/// Type substitution: maps type variable IDs to types.
#[derive(Clone, Debug, Default)]
pub struct TypeSubst<'db> {
    map: HashMap<u64, Type<'db>>,
}

impl<'db> TypeSubst<'db> {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert a mapping.
    pub fn insert(&mut self, var: u64, ty: Type<'db>) {
        self.map.insert(var, ty);
    }

    /// Look up a type variable.
    pub fn get(&self, var: u64) -> Option<Type<'db>> {
        self.map.get(&var).copied()
    }

    /// Apply the substitution to a type.
    pub fn apply(&self, db: &'db dyn salsa::Database, ty: Type<'db>) -> Type<'db> {
        match ty.kind(db) {
            TypeKind::UniVar { id } => {
                if let Some(subst_ty) = self.get(*id) {
                    // Recursively apply to handle chains
                    self.apply(db, subst_ty)
                } else {
                    ty
                }
            }
            TypeKind::Named { name, args } => {
                let args = args.iter().map(|a| self.apply(db, *a)).collect();
                Type::new(db, TypeKind::Named { name: *name, args })
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let params = params.iter().map(|p| self.apply(db, *p)).collect();
                let result = self.apply(db, *result);
                // Note: Effect row substitution requires RowSubst, use apply_with_rows for full substitution
                Type::new(
                    db,
                    TypeKind::Func {
                        params,
                        result,
                        effect: *effect,
                    },
                )
            }
            TypeKind::Tuple(elements) => {
                let elements = elements.iter().map(|e| self.apply(db, *e)).collect();
                Type::new(db, TypeKind::Tuple(elements))
            }
            TypeKind::App { ctor, args } => {
                let ctor = self.apply(db, *ctor);
                let args = args.iter().map(|a| self.apply(db, *a)).collect();
                Type::new(db, TypeKind::App { ctor, args })
            }
            // Primitive types and bound variables are unchanged
            _ => ty,
        }
    }

    /// Apply substitution to a type, including effect row substitution.
    pub fn apply_with_rows(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
    ) -> Type<'db> {
        match ty.kind(db) {
            TypeKind::UniVar { id } => {
                if let Some(subst_ty) = self.get(*id) {
                    self.apply_with_rows(db, subst_ty, row_subst)
                } else {
                    ty
                }
            }
            TypeKind::Named { name, args } => {
                let args = args
                    .iter()
                    .map(|a| self.apply_with_rows(db, *a, row_subst))
                    .collect();
                Type::new(db, TypeKind::Named { name: *name, args })
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let params = params
                    .iter()
                    .map(|p| self.apply_with_rows(db, *p, row_subst))
                    .collect();
                let result = self.apply_with_rows(db, *result, row_subst);
                let effect = row_subst.apply(db, *effect);
                Type::new(
                    db,
                    TypeKind::Func {
                        params,
                        result,
                        effect,
                    },
                )
            }
            TypeKind::Tuple(elements) => {
                let elements = elements
                    .iter()
                    .map(|e| self.apply_with_rows(db, *e, row_subst))
                    .collect();
                Type::new(db, TypeKind::Tuple(elements))
            }
            TypeKind::App { ctor, args } => {
                let ctor = self.apply_with_rows(db, *ctor, row_subst);
                let args = args
                    .iter()
                    .map(|a| self.apply_with_rows(db, *a, row_subst))
                    .collect();
                Type::new(db, TypeKind::App { ctor, args })
            }
            _ => ty,
        }
    }
}

/// Row substitution: maps row variable IDs to effect rows.
#[derive(Clone, Debug, Default)]
pub struct RowSubst<'db> {
    map: HashMap<u64, EffectRow<'db>>,
}

impl<'db> RowSubst<'db> {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert a mapping.
    pub fn insert(&mut self, var: u64, row: EffectRow<'db>) {
        self.map.insert(var, row);
    }

    /// Look up a row variable.
    pub fn get(&self, var: u64) -> Option<EffectRow<'db>> {
        self.map.get(&var).copied()
    }

    /// Apply substitution to an effect row.
    pub fn apply(&self, db: &'db dyn salsa::Database, row: EffectRow<'db>) -> EffectRow<'db> {
        // Check if the row has a rest variable that needs substitution
        if let Some(var) = row.rest(db)
            && let Some(subst_row) = self.get(var.id)
        {
            // Combine the concrete effects with the substituted row
            let mut effects = row.effects(db).clone();
            for effect in subst_row.effects(db) {
                if !effects.contains(effect) {
                    effects.push(effect.clone());
                }
            }
            // Use the substituted row's rest
            let rest = subst_row.rest(db);
            return EffectRow::new(db, effects, rest);
        }
        row
    }
}

/// Type constraint solver.
#[allow(dead_code)]
pub struct TypeSolver<'db> {
    db: &'db dyn salsa::Database,
    /// Type variable substitution.
    type_subst: TypeSubst<'db>,
    /// Row variable substitution.
    row_subst: RowSubst<'db>,
    /// Counter for fresh row variables.
    next_row_var: u64,
}

impl<'db> TypeSolver<'db> {
    /// Create a new solver.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            type_subst: TypeSubst::new(),
            row_subst: RowSubst::new(),
            next_row_var: 1000, // Start high to avoid collisions
        }
    }

    /// Get the type substitution.
    pub fn type_subst(&self) -> &TypeSubst<'db> {
        &self.type_subst
    }

    /// Get the row substitution.
    pub fn row_subst(&self) -> &RowSubst<'db> {
        &self.row_subst
    }

    /// Generate a fresh row variable.
    #[allow(dead_code)]
    fn fresh_row_var(&mut self) -> EffectVar {
        let id = self.next_row_var;
        self.next_row_var += 1;
        EffectVar { id }
    }

    /// Solve a set of constraints.
    pub fn solve(&mut self, constraints: ConstraintSet<'db>) -> Result<(), SolveError<'db>> {
        for constraint in constraints.into_constraints() {
            self.solve_constraint(constraint)?;
        }
        Ok(())
    }

    /// Solve a single constraint.
    fn solve_constraint(&mut self, constraint: Constraint<'db>) -> Result<(), SolveError<'db>> {
        match constraint {
            Constraint::TypeEq(t1, t2) => self.unify_types(t1, t2),
            Constraint::RowEq(r1, r2) => self.unify_rows(r1, r2),
            Constraint::And(cs) => {
                for c in cs {
                    self.solve_constraint(c)?;
                }
                Ok(())
            }
        }
    }

    /// Unify two types.
    fn unify_types(&mut self, t1: Type<'db>, t2: Type<'db>) -> Result<(), SolveError<'db>> {
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

            // Error types unify with anything
            (&TypeKind::Error, _) | (_, &TypeKind::Error) => Ok(()),

            // Structural unification for compound types
            (
                &TypeKind::Named {
                    name: n1,
                    args: ref a1,
                },
                &TypeKind::Named {
                    name: n2,
                    args: ref a2,
                },
            ) => {
                if n1 != n2 || a1.len() != a2.len() {
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
                },
                &TypeKind::Func {
                    params: ref p2,
                    result: r2,
                    effect: e2,
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

            // Primitive types must match exactly
            _ => Err(SolveError::TypeMismatch {
                expected: t1,
                actual: t2,
            }),
        }
    }

    /// Bind a type variable to a type.
    fn bind_type_var(&mut self, var: u64, ty: Type<'db>) -> Result<(), SolveError<'db>> {
        // Occurs check: prevent infinite types
        if self.occurs_in(var, ty) {
            return Err(SolveError::OccursCheck { var, ty });
        }
        self.type_subst.insert(var, ty);
        Ok(())
    }

    /// Check if a type variable occurs in a type (for occurs check).
    fn occurs_in(&self, var: u64, ty: Type<'db>) -> bool {
        match ty.kind(self.db) {
            TypeKind::UniVar { id } => *id == var,
            TypeKind::Named { args, .. } => args.iter().any(|a| self.occurs_in(var, *a)),
            TypeKind::Func { params, result, .. } => {
                params.iter().any(|p| self.occurs_in(var, *p)) || self.occurs_in(var, *result)
            }
            TypeKind::Tuple(elements) => elements.iter().any(|e| self.occurs_in(var, *e)),
            TypeKind::App { ctor, args } => {
                self.occurs_in(var, *ctor) || args.iter().any(|a| self.occurs_in(var, *a))
            }
            _ => false,
        }
    }

    /// Unify two effect rows.
    fn unify_rows(
        &mut self,
        r1: EffectRow<'db>,
        r2: EffectRow<'db>,
    ) -> Result<(), SolveError<'db>> {
        // Simple implementation: just check equality for now
        // TODO: Implement proper row unification with row variables
        if r1 == r2 {
            return Ok(());
        }

        // If both are pure, they're equal
        if r1.is_pure(self.db) && r2.is_pure(self.db) {
            return Ok(());
        }

        // If one is an open row, bind it to the other
        if let Some(var) = r1.rest(self.db) {
            self.row_subst.insert(var.id, r2);
            return Ok(());
        }
        if let Some(var) = r2.rest(self.db) {
            self.row_subst.insert(var.id, r1);
            return Ok(());
        }

        // Otherwise, check if effects match
        // TODO: More sophisticated row unification
        Err(SolveError::RowMismatch {
            expected: r1,
            actual: r2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    #[test]
    fn test_unify_same_type() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        solver.unify_types(int_ty, int_ty).unwrap();
    }

    #[test]
    fn test_unify_type_var() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = Type::new(&db, TypeKind::UniVar { id: 0 });
        let int_ty = Type::new(&db, TypeKind::Int);

        solver.unify_types(var_ty, int_ty).unwrap();

        // Check that the substitution was recorded
        let result = solver.type_subst.apply(&db, var_ty);
        assert_eq!(result, int_ty);
    }

    #[test]
    fn test_unify_type_mismatch() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        let result = solver.unify_types(int_ty, bool_ty);
        assert!(result.is_err());
    }

    #[test]
    fn test_occurs_check() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = Type::new(&db, TypeKind::UniVar { id: 0 });
        // Try to unify x with List(x) - should fail occurs check
        let list_ty = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![var_ty],
            },
        );

        let result = solver.unify_types(var_ty, list_ty);
        assert!(matches!(result, Err(SolveError::OccursCheck { .. })));
    }

    #[test]
    fn test_unify_tuple() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var1 = Type::new(&db, TypeKind::UniVar { id: 0 });
        let var2 = Type::new(&db, TypeKind::UniVar { id: 1 });
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        let tuple1 = Type::new(&db, TypeKind::Tuple(vec![var1, var2]));
        let tuple2 = Type::new(&db, TypeKind::Tuple(vec![int_ty, bool_ty]));

        solver.unify_types(tuple1, tuple2).unwrap();

        assert_eq!(solver.type_subst.apply(&db, var1), int_ty);
        assert_eq!(solver.type_subst.apply(&db, var2), bool_ty);
    }
}
