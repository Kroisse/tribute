//! Type constraint solver.
//!
//! Solves type constraints using union-find based unification.

use std::collections::HashMap;

use crate::ast::{EffectRow, EffectVar, Type, TypeKind, UniVarId};

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
    OccursCheck { var: UniVarId<'db>, ty: Type<'db> },
    /// Effect row mismatch.
    RowMismatch {
        expected: EffectRow<'db>,
        actual: EffectRow<'db>,
    },
}

/// Type substitution: maps type variable IDs to types.
#[derive(Clone, Debug, Default)]
pub struct TypeSubst<'db> {
    map: HashMap<UniVarId<'db>, Type<'db>>,
}

impl<'db> TypeSubst<'db> {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert a mapping.
    pub fn insert(&mut self, var: UniVarId<'db>, ty: Type<'db>) {
        self.map.insert(var, ty);
    }

    /// Look up a type variable.
    pub fn get(&self, var: UniVarId<'db>) -> Option<Type<'db>> {
        self.map.get(&var).copied()
    }

    /// Apply the substitution to a type.
    ///
    /// Note: This delegates to `apply_with_rows` with an empty row substitution.
    /// If you need to substitute effect row variables, use `apply_with_rows` directly
    /// with the appropriate `RowSubst`.
    pub fn apply(&self, db: &'db dyn salsa::Database, ty: Type<'db>) -> Type<'db> {
        self.apply_with_rows(db, ty, &RowSubst::new())
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
    fn bind_type_var(&mut self, var: UniVarId<'db>, ty: Type<'db>) -> Result<(), SolveError<'db>> {
        // Occurs check: prevent infinite types
        if self.occurs_in(var, ty) {
            return Err(SolveError::OccursCheck { var, ty });
        }
        self.type_subst.insert(var, ty);
        Ok(())
    }

    /// Check if a type variable occurs in a type (for occurs check).
    fn occurs_in(&self, var: UniVarId<'db>, ty: Type<'db>) -> bool {
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
    use crate::ast::UniVarSource;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    /// Create a fresh type variable for testing.
    fn fresh_var(db: &dyn salsa::Database, n: u64) -> Type<'_> {
        let source = UniVarSource::Anonymous(n);
        let id = UniVarId::new(db, source, 0);
        Type::new(db, TypeKind::UniVar { id })
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

        let var_ty = fresh_var(&db, 0);
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

        let var_ty = fresh_var(&db, 0);
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

        let var1 = fresh_var(&db, 0);
        let var2 = fresh_var(&db, 1);
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        let tuple1 = Type::new(&db, TypeKind::Tuple(vec![var1, var2]));
        let tuple2 = Type::new(&db, TypeKind::Tuple(vec![int_ty, bool_ty]));

        solver.unify_types(tuple1, tuple2).unwrap();

        assert_eq!(solver.type_subst.apply(&db, var1), int_ty);
        assert_eq!(solver.type_subst.apply(&db, var2), bool_ty);
    }

    // =========================================================================
    // Row unification tests (adapted from tribute-passes)
    // =========================================================================

    #[test]
    fn test_empty_row_unification() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let r1 = EffectRow::new(&db, vec![], None);
        let r2 = EffectRow::new(&db, vec![], None);

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_row_var_unification() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        // Create an open row with a variable
        let row_var = EffectVar { id: 42 };
        let r1 = EffectRow::new(&db, vec![], Some(row_var));
        let r2 = EffectRow::new(&db, vec![], None); // empty/pure row

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());

        // The row variable should now be bound to the empty row
        let resolved = solver.row_subst.get(row_var.id);
        assert!(resolved.is_some());
        assert!(resolved.unwrap().is_pure(&db));
    }

    #[test]
    fn test_function_effect_unification() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        // Create two function types with the same (empty) effect row
        let empty_effect = EffectRow::new(&db, vec![], None);
        let int_ty = Type::new(&db, TypeKind::Int);

        let func1 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect: empty_effect,
            },
        );
        let func2 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect: empty_effect,
            },
        );

        let result = solver.unify_types(func1, func2);
        assert!(result.is_ok(), "Same function types should unify");
    }

    #[test]
    fn test_function_effect_unification_with_row_var() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);

        // Create a function with empty effect (pure)
        let empty_effect = EffectRow::new(&db, vec![], None);
        let func_pure = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect: empty_effect,
            },
        );

        // Create a function with a row variable effect (polymorphic)
        let row_var = EffectVar { id: 99 };
        let poly_effect = EffectRow::new(&db, vec![], Some(row_var));
        let func_poly = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect: poly_effect,
            },
        );

        // Unifying should bind the row variable to empty
        let result = solver.unify_types(func_pure, func_poly);
        assert!(
            result.is_ok(),
            "Pure function should unify with polymorphic function"
        );

        // Check that the row variable was bound to empty
        let resolved = solver.row_subst.get(row_var.id);
        assert!(resolved.is_some(), "Row variable should be bound");
        assert!(
            resolved.unwrap().is_pure(&db),
            "Row variable should be bound to empty"
        );
    }

    #[test]
    fn test_unify_named_types_with_args() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = fresh_var(&db, 0);
        let int_ty = Type::new(&db, TypeKind::Int);

        // List(var) and List(Int)
        let list_var = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![var_ty],
            },
        );
        let list_int = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![int_ty],
            },
        );

        solver.unify_types(list_var, list_int).unwrap();

        // var should be bound to Int
        assert_eq!(solver.type_subst.apply(&db, var_ty), int_ty);
    }

    #[test]
    fn test_unify_named_types_mismatch() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);

        // List(Int) and Option(Int) should not unify
        let list_int = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int_ty],
            },
        );

        let result = solver.unify_types(list_int, option_int);
        assert!(matches!(result, Err(SolveError::TypeMismatch { .. })));
    }

    #[test]
    fn test_unify_app_types() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = fresh_var(&db, 0);
        let int_ty = Type::new(&db, TypeKind::Int);
        let ctor_ty = fresh_var(&db, 1);
        let list_ctor = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![],
            },
        );

        // App(ctor, [var]) and App(List, [Int])
        let app1 = Type::new(
            &db,
            TypeKind::App {
                ctor: ctor_ty,
                args: vec![var_ty],
            },
        );
        let app2 = Type::new(
            &db,
            TypeKind::App {
                ctor: list_ctor,
                args: vec![int_ty],
            },
        );

        solver.unify_types(app1, app2).unwrap();

        assert_eq!(solver.type_subst.apply(&db, var_ty), int_ty);
        assert_eq!(solver.type_subst.apply(&db, ctor_ty), list_ctor);
    }

    #[test]
    fn test_error_type_unifies_with_anything() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let error_ty = Type::new(&db, TypeKind::Error);
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        // Error should unify with any type
        assert!(solver.unify_types(error_ty, int_ty).is_ok());
        assert!(solver.unify_types(bool_ty, error_ty).is_ok());
    }

    #[test]
    fn test_transitive_unification() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var1 = fresh_var(&db, 0);
        let var2 = fresh_var(&db, 1);
        let int_ty = Type::new(&db, TypeKind::Int);

        // var1 = var2, var2 = Int => var1 = Int
        solver.unify_types(var1, var2).unwrap();
        solver.unify_types(var2, int_ty).unwrap();

        assert_eq!(solver.type_subst.apply(&db, var1), int_ty);
        assert_eq!(solver.type_subst.apply(&db, var2), int_ty);
    }

    #[test]
    fn test_row_subst_apply() {
        let db = test_db();
        let mut row_subst = RowSubst::new();

        // Create a row variable and bind it to an empty row
        let row_var = EffectVar { id: 10 };
        let empty_row = EffectRow::new(&db, vec![], None);
        row_subst.insert(row_var.id, empty_row);

        // Apply substitution to an open row
        let open_row = EffectRow::new(&db, vec![], Some(row_var));
        let result = row_subst.apply(&db, open_row);

        assert!(result.is_pure(&db));
    }

    #[test]
    fn test_type_subst_apply_with_rows() {
        let db = test_db();
        let mut type_subst = TypeSubst::new();
        let mut row_subst = RowSubst::new();

        let int_ty = Type::new(&db, TypeKind::Int);
        let var_ty = fresh_var(&db, 0);
        let var_id = match var_ty.kind(&db) {
            TypeKind::UniVar { id } => *id,
            _ => unreachable!(),
        };
        type_subst.insert(var_id, int_ty);

        // Create a function type with a row variable
        let row_var = EffectVar { id: 20 };
        let empty_row = EffectRow::new(&db, vec![], None);
        row_subst.insert(row_var.id, empty_row);

        let poly_effect = EffectRow::new(&db, vec![], Some(row_var));
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![var_ty],
                result: var_ty,
                effect: poly_effect,
            },
        );

        // Apply both substitutions
        let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

        // Check params and result are substituted
        if let TypeKind::Func {
            params,
            result,
            effect,
        } = result.kind(&db)
        {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], int_ty);
            assert_eq!(*result, int_ty);
            assert!(effect.is_pure(&db));
        } else {
            panic!("Expected Func type");
        }
    }
}
