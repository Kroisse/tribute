//! Type constraint solver using union-find.
//!
//! The solver handles:
//! - Type unification (including type variables)
//! - Row unification for effect types
//! - Occurs check to prevent infinite types

use std::collections::HashMap;

use tracing::{debug, trace};
use tribute_ir::dialect::ty;
use trunk_ir::{Attribute, IdVec, Symbol, Type};

use super::constraint::{Constraint, ConstraintSet, TypeVar};
use super::effect_row::{AbilityRef, EffectRow, RowVar};

/// Error during type solving.
#[derive(Clone, Debug)]
pub enum SolveError<'db> {
    /// Type mismatch: expected `expected`, found `found`.
    TypeMismatch {
        expected: Type<'db>,
        found: Type<'db>,
    },
    /// Occurs check failed: type variable appears in its own definition.
    OccursCheck { var: TypeVar, ty: Type<'db> },
    /// Row occurs check: row variable appears in its own definition.
    RowOccursCheck { var: RowVar, row: EffectRow<'db> },
    /// Duplicate ability in effect row.
    DuplicateAbility { ability: AbilityRef<'db> },
    /// Missing ability in effect row.
    MissingAbility {
        ability: AbilityRef<'db>,
        row: EffectRow<'db>,
    },
    /// Arity mismatch in type application.
    ArityMismatch { expected: usize, found: usize },
}

impl<'db> std::fmt::Display for SolveError<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::TypeMismatch { expected, found } => {
                write!(
                    f,
                    "type mismatch: expected {:?}, found {:?}",
                    expected, found
                )
            }
            SolveError::OccursCheck { var, ty } => {
                write!(f, "infinite type: {:?} occurs in {:?}", var, ty)
            }
            SolveError::RowOccursCheck { var, row } => {
                write!(f, "infinite effect row: {:?} occurs in {:?}", var, row)
            }
            SolveError::DuplicateAbility { ability } => {
                write!(f, "duplicate ability: {:?}", ability)
            }
            SolveError::MissingAbility { ability, row } => {
                write!(f, "missing ability {:?} in effect row {:?}", ability, row)
            }
            SolveError::ArityMismatch { expected, found } => {
                write!(
                    f,
                    "arity mismatch: expected {} type arguments, found {}",
                    expected, found
                )
            }
        }
    }
}

impl<'db> std::error::Error for SolveError<'db> {}

/// Result of solving constraints.
pub type SolveResult<'db, T> = Result<T, SolveError<'db>>;

/// Type substitution: maps type variable IDs to types.
#[derive(Clone, Debug, Default)]
pub struct TypeSubst<'db> {
    /// Map from type variable ID to its substitution.
    types: HashMap<u64, Type<'db>>,
}

impl<'db> TypeSubst<'db> {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
        }
    }

    /// Insert a type variable binding.
    pub fn insert(&mut self, var_id: u64, ty: Type<'db>) {
        trace!(var_id, ?ty, "inserting type substitution");
        self.types.insert(var_id, ty);
    }

    /// Look up a type variable.
    pub fn get(&self, var_id: u64) -> Option<Type<'db>> {
        self.types.get(&var_id).copied()
    }

    /// Check if a type variable is bound.
    pub fn contains(&self, var_id: u64) -> bool {
        self.types.contains_key(&var_id)
    }

    /// Get the number of substitutions.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Apply this substitution to a type, resolving type variables.
    pub fn apply(&self, db: &'db dyn salsa::Database, ty: Type<'db>) -> Type<'db> {
        // If it's a type variable, look it up
        if ty::is_var(db, ty) {
            if let Some(Attribute::IntBits(id)) = ty.get_attr(db, Symbol::new("id")) {
                if let Some(resolved) = self.get(*id) {
                    trace!(id, ?resolved, "applying type substitution");
                    // Recursively apply in case the resolved type also has variables
                    return self.apply(db, resolved);
                } else {
                    trace!(id, "type variable not in substitution map");
                }
            }
            return ty;
        }

        // If it has type parameters, apply substitution to them
        let params = ty.params(db);
        if params.is_empty() {
            return ty;
        }

        let new_params: IdVec<Type<'db>> = params.iter().map(|p| self.apply(db, *p)).collect();
        Type::new(
            db,
            ty.dialect(db),
            ty.name(db),
            new_params,
            ty.attrs(db).clone(),
        )
    }
}

/// Row substitution: maps row variable IDs to effect rows.
#[derive(Clone, Debug, Default)]
pub struct RowSubst<'db> {
    /// Map from row variable ID to its substitution.
    rows: HashMap<u64, EffectRow<'db>>,
}

impl<'db> RowSubst<'db> {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            rows: HashMap::new(),
        }
    }

    /// Insert a row variable binding.
    pub fn insert(&mut self, var: RowVar, row: EffectRow<'db>) {
        self.rows.insert(var.0, row);
    }

    /// Look up a row variable.
    pub fn get(&self, var: RowVar) -> Option<&EffectRow<'db>> {
        self.rows.get(&var.0)
    }

    /// Apply this substitution to an effect row, resolving row variables.
    pub fn apply(&self, row: &EffectRow<'db>) -> EffectRow<'db> {
        match row.tail() {
            None => row.clone(),
            Some(tail_var) => {
                if let Some(tail_row) = self.get(tail_var) {
                    // Substitute the tail with its resolved row
                    let mut result = row.clone();
                    // Add abilities from the substituted tail
                    for ability in tail_row.abilities() {
                        result.add_ability(ability.clone());
                    }
                    // Set the new tail (from the substituted row)
                    result.set_tail(tail_row.tail());
                    // Recursively apply
                    self.apply(&result)
                } else {
                    row.clone()
                }
            }
        }
    }
}

/// Type solver using union-find style substitution.
pub struct TypeSolver<'db> {
    db: &'db dyn salsa::Database,
    /// Type variable substitutions.
    type_subst: TypeSubst<'db>,
    /// Row variable substitutions.
    row_subst: RowSubst<'db>,
    /// Counter for generating fresh row variables.
    next_row_var: u64,
}

impl<'db> TypeSolver<'db> {
    /// Create a new type solver.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            type_subst: TypeSubst::new(),
            row_subst: RowSubst::new(),
            next_row_var: 0,
        }
    }

    /// Generate a fresh row variable.
    pub fn fresh_row_var(&mut self) -> RowVar {
        let id = self.next_row_var;
        self.next_row_var += 1;
        RowVar::new(id)
    }

    /// Solve a set of constraints.
    pub fn solve(&mut self, constraints: ConstraintSet<'db>) -> SolveResult<'db, ()> {
        let constraint_list = constraints.into_constraints();
        debug!(
            num_constraints = constraint_list.len(),
            "solving constraints"
        );
        for constraint in constraint_list {
            self.solve_one(constraint)?;
        }
        debug!(
            num_substitutions = self.type_subst.len(),
            "solved constraints"
        );
        Ok(())
    }

    /// Solve a single constraint.
    fn solve_one(&mut self, constraint: Constraint<'db>) -> SolveResult<'db, ()> {
        match constraint {
            Constraint::TypeEq(t1, t2) => self.unify_types(t1, t2),
            Constraint::RowEq(r1, r2) => self.unify_rows(r1, r2),
            Constraint::RowSub(sub, sup) => self.check_row_sub(&sub, &sup),
            Constraint::AbilityIn(ability, row_var) => self.check_ability_in(&ability, row_var),
            Constraint::And(cs) => {
                for c in cs {
                    self.solve_one(c)?;
                }
                Ok(())
            }
        }
    }

    /// Unify two types.
    pub fn unify_types(&mut self, t1: Type<'db>, t2: Type<'db>) -> SolveResult<'db, ()> {
        let t1 = self.type_subst.apply(self.db, t1);
        let t2 = self.type_subst.apply(self.db, t2);

        // Same type (interned equality)
        if t1 == t2 {
            return Ok(());
        }

        // Check for type variables
        let t1_is_var = ty::is_var(self.db, t1);
        let t2_is_var = ty::is_var(self.db, t2);

        match (t1_is_var, t2_is_var) {
            (true, _) => {
                // Bind t1 to t2
                if let Some(Attribute::IntBits(id)) = t1.get_attr(self.db, Symbol::new("id")) {
                    // Occurs check
                    if self.occurs_in(*id, t2) {
                        return Err(SolveError::OccursCheck {
                            var: TypeVar::new(*id),
                            ty: t2,
                        });
                    }
                    self.type_subst.insert(*id, t2);
                }
                Ok(())
            }
            (false, true) => {
                // Bind t2 to t1
                if let Some(Attribute::IntBits(id)) = t2.get_attr(self.db, Symbol::new("id")) {
                    // Occurs check
                    if self.occurs_in(*id, t1) {
                        return Err(SolveError::OccursCheck {
                            var: TypeVar::new(*id),
                            ty: t1,
                        });
                    }
                    self.type_subst.insert(*id, t1);
                }
                Ok(())
            }
            (false, false) => {
                // Both are concrete types - check structure
                self.unify_concrete(t1, t2)
            }
        }
    }

    /// Unify two concrete (non-variable) types.
    fn unify_concrete(&mut self, t1: Type<'db>, t2: Type<'db>) -> SolveResult<'db, ()> {
        // Check dialect and name match
        if t1.dialect(self.db) != t2.dialect(self.db) || t1.name(self.db) != t2.name(self.db) {
            return Err(SolveError::TypeMismatch {
                expected: t1,
                found: t2,
            });
        }

        // Check params match
        let params1 = t1.params(self.db);
        let params2 = t2.params(self.db);

        if params1.len() != params2.len() {
            return Err(SolveError::ArityMismatch {
                expected: params1.len(),
                found: params2.len(),
            });
        }

        // Unify each parameter
        for (p1, p2) in params1.iter().zip(params2.iter()) {
            self.unify_types(*p1, *p2)?;
        }

        // For function types, also unify effect rows
        if t1.is_function(self.db) && t2.is_function(self.db) {
            let eff1 = t1.function_effect(self.db);
            let eff2 = t2.function_effect(self.db);
            // Convert effect types to EffectRow and use row unification
            let row1 = eff1
                .and_then(|e| EffectRow::from_type(self.db, e))
                .unwrap_or_else(EffectRow::empty);
            let row2 = eff2
                .and_then(|e| EffectRow::from_type(self.db, e))
                .unwrap_or_else(EffectRow::empty);
            self.unify_rows(row1, row2)?;
        }

        Ok(())
    }

    /// Check if a type variable occurs in a type (for occurs check).
    fn occurs_in(&self, var_id: u64, ty: Type<'db>) -> bool {
        let ty = self.type_subst.apply(self.db, ty);

        if ty::is_var(self.db, ty)
            && let Some(Attribute::IntBits(id)) = ty.get_attr(self.db, Symbol::new("id"))
        {
            return *id == var_id;
        }

        // Check in type parameters
        ty.params(self.db)
            .iter()
            .any(|p| self.occurs_in(var_id, *p))
    }

    /// Unify two effect rows.
    pub fn unify_rows(&mut self, r1: EffectRow<'db>, r2: EffectRow<'db>) -> SolveResult<'db, ()> {
        let r1 = self.row_subst.apply(&r1);
        let r2 = self.row_subst.apply(&r2);

        // Both empty
        if r1.is_empty() && r2.is_empty() {
            return Ok(());
        }

        // One is just a variable
        if r1.is_var()
            && let Some(v) = r1.tail()
        {
            // Check occurs
            if self.row_occurs_in(v, &r2) {
                return Err(SolveError::RowOccursCheck { var: v, row: r2 });
            }
            self.row_subst.insert(v, r2);
            return Ok(());
        }

        if r2.is_var()
            && let Some(v) = r2.tail()
        {
            if self.row_occurs_in(v, &r1) {
                return Err(SolveError::RowOccursCheck { var: v, row: r1 });
            }
            self.row_subst.insert(v, r1);
            return Ok(());
        }

        // General case: match abilities and unify tails
        self.unify_rows_general(r1, r2)
    }

    /// General row unification algorithm.
    ///
    /// ```text
    /// unify_row({A | ρ₁}, {A | ρ₂}) → unify_row(ρ₁, ρ₂)
    /// unify_row({A | ρ₁}, {B | ρ₂}) → fresh e; unify_row(ρ₁, {B | e}); unify_row(ρ₂, {A | e})
    /// ```
    fn unify_rows_general(
        &mut self,
        r1: EffectRow<'db>,
        r2: EffectRow<'db>,
    ) -> SolveResult<'db, ()> {
        // Find common abilities (for future use in duplicate detection)
        let _common: Vec<_> = r1
            .abilities()
            .intersection(r2.abilities())
            .cloned()
            .collect();

        // Find abilities only in r1
        let only_r1: Vec<_> = r1.abilities().difference(r2.abilities()).cloned().collect();

        // Find abilities only in r2
        let only_r2: Vec<_> = r2.abilities().difference(r1.abilities()).cloned().collect();

        match (r1.tail(), r2.tail()) {
            (None, None) => {
                // Both closed: must have exactly the same abilities
                if !only_r1.is_empty() || !only_r2.is_empty() {
                    // Mismatch - pick first different ability for error
                    let missing = only_r1.first().or(only_r2.first()).unwrap().clone();
                    return Err(SolveError::MissingAbility {
                        ability: missing,
                        row: if only_r1.is_empty() { r1 } else { r2 },
                    });
                }
                Ok(())
            }
            (Some(v1), None) => {
                // r1 has tail, r2 is closed
                // v1 must equal {only_r2}
                if self.row_occurs_in(v1, &EffectRow::concrete(only_r2.clone())) {
                    return Err(SolveError::RowOccursCheck {
                        var: v1,
                        row: EffectRow::concrete(only_r2),
                    });
                }
                self.row_subst.insert(v1, EffectRow::concrete(only_r2));
                Ok(())
            }
            (None, Some(v2)) => {
                // r2 has tail, r1 is closed
                // v2 must equal {only_r1}
                if self.row_occurs_in(v2, &EffectRow::concrete(only_r1.clone())) {
                    return Err(SolveError::RowOccursCheck {
                        var: v2,
                        row: EffectRow::concrete(only_r1),
                    });
                }
                self.row_subst.insert(v2, EffectRow::concrete(only_r1));
                Ok(())
            }
            (Some(v1), Some(v2)) => {
                // Both have tails
                // fresh e; v1 = {only_r2 | e}; v2 = {only_r1 | e}
                let fresh = self.fresh_row_var();

                let r1_tail = if only_r2.is_empty() {
                    EffectRow::var(fresh)
                } else {
                    EffectRow::with_tail(only_r2, fresh)
                };

                let r2_tail = if only_r1.is_empty() {
                    EffectRow::var(fresh)
                } else {
                    EffectRow::with_tail(only_r1, fresh)
                };

                self.row_subst.insert(v1, r1_tail);
                self.row_subst.insert(v2, r2_tail);
                Ok(())
            }
        }
    }

    /// Check if a row variable occurs in an effect row.
    fn row_occurs_in(&self, var: RowVar, row: &EffectRow<'db>) -> bool {
        let row = self.row_subst.apply(row);
        match row.tail() {
            Some(v) if v == var => true,
            Some(v) => {
                // Check if v maps to something containing var
                if let Some(mapped) = self.row_subst.get(v) {
                    self.row_occurs_in(var, mapped)
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// Check row subsumption: sub ⊆ sup.
    fn check_row_sub(
        &mut self,
        sub: &EffectRow<'db>,
        sup: &EffectRow<'db>,
    ) -> SolveResult<'db, ()> {
        let sub = self.row_subst.apply(sub);
        let sup = self.row_subst.apply(sup);

        // Every ability in sub must be in sup
        for ability in sub.abilities() {
            if !sup.contains(ability) && sup.tail().is_none() {
                return Err(SolveError::MissingAbility {
                    ability: ability.clone(),
                    row: sup,
                });
            }
        }

        // If sup has a tail, any additional abilities from sub can flow there
        // If sup is closed, sub must be a subset
        Ok(())
    }

    /// Check if an ability is in a row (possibly via row variable).
    fn check_ability_in(
        &mut self,
        ability: &AbilityRef<'db>,
        row_var: RowVar,
    ) -> SolveResult<'db, ()> {
        if let Some(row) = self.row_subst.get(row_var) {
            let row = row.clone();
            if row.contains(ability) {
                return Ok(());
            }
            if row.tail().is_some() {
                // Could be in the tail
                return Ok(());
            }
            return Err(SolveError::MissingAbility {
                ability: ability.clone(),
                row,
            });
        }

        // Row variable not yet bound - add ability to it
        let fresh = self.fresh_row_var();
        self.row_subst.insert(
            row_var,
            EffectRow::with_tail(std::iter::once(ability.clone()), fresh),
        );
        Ok(())
    }

    /// Get the type substitution.
    pub fn type_subst(&self) -> &TypeSubst<'db> {
        &self.type_subst
    }

    /// Get the row substitution.
    pub fn row_subst(&self) -> &RowSubst<'db> {
        &self.row_subst
    }

    /// Apply the current substitution to a type.
    pub fn apply_type(&self, ty: Type<'db>) -> Type<'db> {
        self.type_subst.apply(self.db, ty)
    }

    /// Apply the current substitution to an effect row.
    pub fn apply_row(&self, row: &EffectRow<'db>) -> EffectRow<'db> {
        self.row_subst.apply(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::DialectType;
    use trunk_ir::dialect::core;

    #[salsa_test]
    fn test_unify_same_type(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);

        let i64_ty = *core::I64::new(db);
        let result = solver.unify_types(i64_ty, i64_ty);
        assert!(result.is_ok());
    }

    #[salsa_test]
    fn test_unify_type_var(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);

        let var = ty::var_with_id(db, 0);
        let i64_ty = *core::I64::new(db);

        let result = solver.unify_types(var, i64_ty);
        assert!(result.is_ok());

        let resolved = solver.apply_type(var);
        assert_eq!(resolved, i64_ty);
    }

    #[salsa_test]
    fn test_unify_different_types(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);

        let i64_ty = *core::I64::new(db);
        let f64_ty = *core::F64::new(db);

        let result = solver.unify_types(i64_ty, f64_ty);
        assert!(matches!(result, Err(SolveError::TypeMismatch { .. })));
    }

    #[salsa_test]
    fn test_empty_row_unification(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);

        let r1 = EffectRow::empty();
        let r2 = EffectRow::empty();

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());
    }

    #[salsa_test]
    fn test_row_var_unification(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);

        let var = solver.fresh_row_var();
        let r1 = EffectRow::var(var);
        let r2 = EffectRow::empty();

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());

        // var should now be bound to empty
        let resolved = solver.apply_row(&EffectRow::var(var));
        assert!(resolved.is_empty());
    }

    #[salsa_test]
    fn test_function_effect_unification(db: &salsa::DatabaseImpl) {
        use trunk_ir::IdVec;

        let mut solver = TypeSolver::new(db);

        // Create two function types with the same effect row
        let effect = core::EffectRowType::empty(db).as_type();
        let func1 = core::Func::with_effect(db, IdVec::new(), *core::I64::new(db), Some(effect));
        let func2 = core::Func::with_effect(db, IdVec::new(), *core::I64::new(db), Some(effect));

        let result = solver.unify_types(*func1, *func2);
        assert!(result.is_ok(), "Same function types should unify");
    }

    #[salsa_test]
    fn test_function_effect_unification_with_row_var(db: &salsa::DatabaseImpl) {
        use trunk_ir::IdVec;

        let mut solver = TypeSolver::new(db);

        // Create a function with empty effect
        let empty_effect = core::EffectRowType::empty(db).as_type();
        let func_pure =
            core::Func::with_effect(db, IdVec::new(), *core::I64::new(db), Some(empty_effect));

        // Create a function with a row variable effect
        let row_var_id = 1;
        let row_var_effect = core::EffectRowType::var(db, row_var_id).as_type();
        let func_poly =
            core::Func::with_effect(db, IdVec::new(), *core::I64::new(db), Some(row_var_effect));

        // Unifying should bind the row variable to empty
        let result = solver.unify_types(*func_pure, *func_poly);
        assert!(
            result.is_ok(),
            "Pure function should unify with polymorphic function"
        );

        // Check that the row variable was bound to empty
        let row = EffectRow::var(RowVar(row_var_id));
        let resolved = solver.apply_row(&row);
        assert!(resolved.is_empty(), "Row variable should be bound to empty");
    }

    #[salsa_test]
    fn test_function_effect_unification_missing_effect(db: &salsa::DatabaseImpl) {
        use trunk_ir::IdVec;

        let mut solver = TypeSolver::new(db);

        // Create a function without explicit effect (pure)
        let func_no_effect = core::Func::new(db, IdVec::new(), *core::I64::new(db));

        // Create a function with empty effect
        let empty_effect = core::EffectRowType::empty(db).as_type();
        let func_empty =
            core::Func::with_effect(db, IdVec::new(), *core::I64::new(db), Some(empty_effect));

        // Both represent pure functions, should unify
        let result = solver.unify_types(*func_no_effect, *func_empty);
        assert!(
            result.is_ok(),
            "Function without effect should unify with empty effect"
        );
    }
}
