//! Type constraint solver.
//!
//! Solves type constraints using union-find based unification.

use std::collections::HashMap;

use crate::ast::{EffectRow, EffectVar, Type, TypeKind, TypeParam, UniVarId};

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
    /// Effect type argument arity mismatch.
    EffectArgArityMismatch {
        effect_name: trunk_ir::Symbol,
        expected: usize,
        found: usize,
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
                let row_applied = row_subst.apply(db, *effect);
                // Also apply type substitution to effect args (e.g., State(?a) → State(Int))
                let effects = row_applied.effects(db);
                let new_effects: Vec<_> = effects
                    .iter()
                    .map(|e| {
                        let new_args: Vec<_> = e
                            .args
                            .iter()
                            .map(|a| self.apply_with_rows(db, *a, row_subst))
                            .collect();
                        crate::ast::Effect {
                            name: e.name,
                            args: new_args,
                        }
                    })
                    .collect();
                let effect = if new_effects != *effects {
                    EffectRow::new(db, new_effects, row_applied.rest(db))
                } else {
                    row_applied
                };
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

    /// Generalize a type by replacing unresolved UniVars with BoundVars.
    ///
    /// After substitution, any remaining UniVar is unresolved (polymorphic).
    /// This method:
    /// 1. Collects unresolved UniVars in appearance order
    /// 2. Replaces each with `BoundVar { index }` in order
    ///
    /// Returns `(generalized_type, type_params)`.
    pub fn generalize(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
    ) -> (Type<'db>, Vec<TypeParam>) {
        // Pass 1: collect unresolved UniVars in appearance order
        let mut univars: Vec<UniVarId<'db>> = Vec::new();
        self.collect_unresolved_univars(db, ty, row_subst, &mut univars);

        if univars.is_empty() {
            return (ty, Vec::new());
        }

        // Build UniVar → BoundVar index mapping
        let var_to_index: HashMap<UniVarId<'db>, u32> = univars
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();

        // Pass 2: replace UniVars with BoundVars
        let generalized = self.replace_univars_with_bound(db, ty, row_subst, &var_to_index);

        // Build type params (anonymous — names not tracked through UniVar)
        let type_params: Vec<TypeParam> = univars.iter().map(|_| TypeParam::anonymous()).collect();

        (generalized, type_params)
    }

    /// Generalize a type and return the UniVar → BoundVar index mapping.
    ///
    /// This is used when you need to apply the same generalization mapping
    /// to multiple types (e.g., function signature and function body).
    pub fn generalize_with_mapping(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
    ) -> (Type<'db>, Vec<TypeParam>, HashMap<UniVarId<'db>, u32>) {
        // Pass 1: collect unresolved UniVars in appearance order
        let mut univars: Vec<UniVarId<'db>> = Vec::new();
        self.collect_unresolved_univars(db, ty, row_subst, &mut univars);

        if univars.is_empty() {
            return (ty, Vec::new(), HashMap::new());
        }

        // Build UniVar → BoundVar index mapping
        let var_to_index: HashMap<UniVarId<'db>, u32> = univars
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();

        // Pass 2: replace UniVars with BoundVars
        let generalized = self.replace_univars_with_bound(db, ty, row_subst, &var_to_index);

        // Build type params (anonymous — names not tracked through UniVar)
        let type_params: Vec<TypeParam> = univars.iter().map(|_| TypeParam::anonymous()).collect();

        (generalized, type_params, var_to_index)
    }

    /// Apply a generalization mapping to a type.
    ///
    /// This replaces UniVars in the type with BoundVars according to the given mapping.
    /// UniVars not in the mapping are left unchanged.
    pub fn apply_generalization(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Type<'db> {
        self.replace_univars_with_bound(db, ty, row_subst, var_to_index)
    }

    /// Collect unresolved UniVarIds from a type in appearance (left-to-right) order.
    ///
    /// Public wrapper for use by other modules.
    pub fn collect_univars_from_type(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        self.collect_unresolved_univars(db, ty, row_subst, out);
    }

    /// Collect unresolved UniVarIds from a type in appearance (left-to-right) order.
    fn collect_unresolved_univars(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        match ty.kind(db) {
            TypeKind::UniVar { id } => {
                // Follow substitution chain
                if let Some(subst_ty) = self.get(*id) {
                    self.collect_unresolved_univars(db, subst_ty, row_subst, out);
                } else if !out.contains(id) {
                    out.push(*id);
                }
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                for p in params {
                    self.collect_unresolved_univars(db, *p, row_subst, out);
                }
                self.collect_unresolved_univars(db, *result, row_subst, out);
                // Also collect from effect row type arguments
                let applied = row_subst.apply(db, *effect);
                for e in applied.effects(db) {
                    for a in &e.args {
                        self.collect_unresolved_univars(db, *a, row_subst, out);
                    }
                }
            }
            TypeKind::Named { args, .. } => {
                for a in args {
                    self.collect_unresolved_univars(db, *a, row_subst, out);
                }
            }
            TypeKind::Tuple(elems) => {
                for e in elems {
                    self.collect_unresolved_univars(db, *e, row_subst, out);
                }
            }
            TypeKind::App { ctor, args } => {
                self.collect_unresolved_univars(db, *ctor, row_subst, out);
                for a in args {
                    self.collect_unresolved_univars(db, *a, row_subst, out);
                }
            }
            _ => {}
        }
    }

    /// Replace unresolved UniVars with BoundVars according to the given mapping.
    fn replace_univars_with_bound(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Type<'db> {
        match ty.kind(db) {
            TypeKind::UniVar { id } => {
                // Check mapping first: if this UniVar should become a BoundVar,
                // do so immediately without following the substitution chain.
                // This is important because when unification binds UniVar(X) -> UniVar(Y),
                // we want to generalize X (which is in the mapping), not Y (which may not be).
                if let Some(&index) = var_to_index.get(id) {
                    Type::new(db, TypeKind::BoundVar { index })
                } else if let Some(subst_ty) = self.get(*id) {
                    self.replace_univars_with_bound(db, subst_ty, row_subst, var_to_index)
                } else {
                    ty
                }
            }
            TypeKind::Named { name, args } => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| self.replace_univars_with_bound(db, *a, row_subst, var_to_index))
                    .collect();
                Type::new(
                    db,
                    TypeKind::Named {
                        name: *name,
                        args: new_args,
                    },
                )
            }
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let new_params: Vec<_> = params
                    .iter()
                    .map(|p| self.replace_univars_with_bound(db, *p, row_subst, var_to_index))
                    .collect();
                let new_result =
                    self.replace_univars_with_bound(db, *result, row_subst, var_to_index);
                // Apply row substitution and generalize effect type args
                let applied_row = row_subst.apply(db, *effect);
                let new_effects: Vec<_> = applied_row
                    .effects(db)
                    .iter()
                    .map(|e| {
                        let new_args: Vec<_> = e
                            .args
                            .iter()
                            .map(|a| {
                                self.replace_univars_with_bound(db, *a, row_subst, var_to_index)
                            })
                            .collect();
                        crate::ast::Effect {
                            name: e.name,
                            args: new_args,
                        }
                    })
                    .collect();
                let new_effect = if new_effects != *applied_row.effects(db) {
                    EffectRow::new(db, new_effects, applied_row.rest(db))
                } else {
                    applied_row
                };
                Type::new(
                    db,
                    TypeKind::Func {
                        params: new_params,
                        result: new_result,
                        effect: new_effect,
                    },
                )
            }
            TypeKind::Tuple(elems) => {
                let new_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.replace_univars_with_bound(db, *e, row_subst, var_to_index))
                    .collect();
                Type::new(db, TypeKind::Tuple(new_elems))
            }
            TypeKind::App { ctor, args } => {
                let new_ctor = self.replace_univars_with_bound(db, *ctor, row_subst, var_to_index);
                let new_args: Vec<_> = args
                    .iter()
                    .map(|a| self.replace_univars_with_bound(db, *a, row_subst, var_to_index))
                    .collect();
                Type::new(
                    db,
                    TypeKind::App {
                        ctor: new_ctor,
                        args: new_args,
                    },
                )
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
        let constraints_vec = constraints.into_constraints();
        for constraint in constraints_vec.into_iter() {
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

            // BoundVar should never reach the solver — it must be instantiated first
            (TypeKind::BoundVar { .. }, _) | (_, TypeKind::BoundVar { .. }) => {
                debug_assert!(
                    false,
                    "BoundVar reached solver — should have been instantiated"
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
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                params.iter().any(|p| self.occurs_in(var, *p))
                    || self.occurs_in(var, *result)
                    || effect
                        .effects(self.db)
                        .iter()
                        .any(|e| e.args.iter().any(|a| self.occurs_in(var, *a)))
            }
            TypeKind::Tuple(elements) => elements.iter().any(|e| self.occurs_in(var, *e)),
            TypeKind::App { ctor, args } => {
                self.occurs_in(var, *ctor) || args.iter().any(|a| self.occurs_in(var, *a))
            }
            _ => false,
        }
    }

    /// Check if a row variable occurs in an effect row (for row occurs check).
    fn row_occurs_in(&self, var: EffectVar, row: EffectRow<'db>) -> bool {
        // Apply current substitution first
        let row = self.row_subst.apply(self.db, row);

        // Check if the row's rest is the same variable
        if row.rest(self.db) == Some(var) {
            return true;
        }

        // Check type variables inside effect args
        for effect in row.effects(self.db) {
            for arg in &effect.args {
                if self.row_occurs_in_type(var, *arg) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if a row variable occurs in a type.
    fn row_occurs_in_type(&self, var: EffectVar, ty: Type<'db>) -> bool {
        match ty.kind(self.db) {
            TypeKind::Func {
                params,
                result,
                effect,
            } => {
                let row = self.row_subst.apply(self.db, *effect);
                if row.rest(self.db) == Some(var) {
                    return true;
                }
                // Recursively check effect args
                for e in row.effects(self.db) {
                    for arg in &e.args {
                        if self.row_occurs_in_type(var, *arg) {
                            return true;
                        }
                    }
                }
                // Recursively check params and result
                if params.iter().any(|p| self.row_occurs_in_type(var, *p)) {
                    return true;
                }
                if self.row_occurs_in_type(var, *result) {
                    return true;
                }
                false
            }
            TypeKind::Named { args, .. } => args.iter().any(|a| self.row_occurs_in_type(var, *a)),
            TypeKind::Tuple(elems) => elems.iter().any(|e| self.row_occurs_in_type(var, *e)),
            TypeKind::App { ctor, args } => {
                self.row_occurs_in_type(var, *ctor)
                    || args.iter().any(|a| self.row_occurs_in_type(var, *a))
            }
            TypeKind::UniVar { id } => {
                if let Some(subst_ty) = self.type_subst.get(*id) {
                    self.row_occurs_in_type(var, subst_ty)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Unify two effect rows.
    ///
    /// Row unification handles several cases:
    /// 1. Both closed: effects must match exactly (as sets)
    /// 2. One open, one closed: bind the row variable to the difference
    /// 3. Same row variable: unify the effect lists
    /// 4. Different row variables: create a fresh variable for the common tail
    fn unify_rows(
        &mut self,
        r1: EffectRow<'db>,
        r2: EffectRow<'db>,
    ) -> Result<(), SolveError<'db>> {
        // Apply current row substitution first
        let r1 = self.row_subst.apply(self.db, r1);
        let r2 = self.row_subst.apply(self.db, r2);

        // Same row: done
        if r1 == r2 {
            return Ok(());
        }

        // If both are pure, they're equal
        if r1.is_pure(self.db) && r2.is_pure(self.db) {
            return Ok(());
        }

        let effects1 = r1.effects(self.db);
        let effects2 = r2.effects(self.db);
        let rest1 = r1.rest(self.db);
        let rest2 = r2.rest(self.db);

        match (rest1, rest2) {
            // Both closed: effects must match as sets
            (None, None) => self.unify_effects_as_sets(effects1, effects2, r1, r2),

            // r1 is open, r2 is closed: var1 = r2's effects minus r1's effects
            (Some(var1), None) => {
                // Row occurs check
                if self.row_occurs_in(var1, r2) {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }
                // Compute difference: effects in r2 but not in r1
                // First check that all effects in r1 have matches in r2
                let only_r2 = self.compute_effect_difference_with_unify(effects1, effects2)?;

                // r1's effects must all be in r2
                // (if any effect from r1 is in only_r2, it means no match was found)
                let matched_count = effects2.len() - only_r2.len();
                if matched_count != effects1.len() {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }

                // Bind var1 to remaining effects (closed)
                let remainder = EffectRow::new(self.db, only_r2, None);
                self.row_subst.insert(var1.id, remainder);
                Ok(())
            }

            // r1 is closed, r2 is open: var2 = r1's effects minus r2's effects
            (None, Some(var2)) => {
                // Pure subsumption: if r1 is pure (closed empty) and r2 has existing effects,
                // it means a pure function is being called from an effectful context.
                // This is always valid, and the caller's effect row should remain unchanged.
                //
                // Example: calling `identity: fn(Int) -> Int` (effect: {}) from a context
                // with effect `{State(Int) | var}` should not modify the caller's effect row.
                //
                // However, if r2 has no effects (just a bare variable `{|var}`), we should
                // bind the variable to the closed row as normal instantiation behavior.
                if r1.is_pure(self.db) && !effects2.is_empty() {
                    return Ok(());
                }

                // Row occurs check
                if self.row_occurs_in(var2, r1) {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }
                // Compute difference: effects in r1 but not in r2
                let only_r1 = self.compute_effect_difference_with_unify(effects2, effects1)?;

                // r2's effects must all be in r1
                let matched_count = effects1.len() - only_r1.len();
                if matched_count != effects2.len() {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }
                // Bind var2 to remaining effects (closed)
                let remainder = EffectRow::new(self.db, only_r1, None);
                self.row_subst.insert(var2.id, remainder);
                Ok(())
            }

            // Both open with the same variable: just unify the effect lists
            (Some(v1), Some(v2)) if v1 == v2 => {
                self.unify_effects_as_sets(effects1, effects2, r1, r2)
            }

            // Both open with different variables: create a fresh variable for the common tail
            // r1 = {A | v1}, r2 = {B | v2}
            // Unify: v1 = {B's not in A | v3}, v2 = {A's not in B | v3}
            (Some(v1), Some(v2)) => {
                // Row occurs check
                if self.row_occurs_in(v1, r2) || self.row_occurs_in(v2, r1) {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }

                let (only_r1, only_r2) =
                    self.compute_effect_split_with_unify(effects1, effects2)?;

                // Create a fresh row variable for the common tail
                let v3 = self.fresh_row_var();

                // v1 = {only_r2 | v3}
                let row_for_v1 = EffectRow::new(self.db, only_r2, Some(v3));
                self.row_subst.insert(v1.id, row_for_v1);

                // v2 = {only_r1 | v3}
                let row_for_v2 = EffectRow::new(self.db, only_r1, Some(v3));
                self.row_subst.insert(v2.id, row_for_v2);

                Ok(())
            }
        }
    }

    /// Check if two effect lists are equal as sets, unifying type arguments.
    ///
    /// Effects are matched by name first, then by arity, then args are unified.
    /// - Same name, different arity → EffectArgArityMismatch
    /// - Same name, same arity, args unify → matched
    /// - Same name, same arity, args don't unify → different abilities (RowMismatch)
    fn unify_effects_as_sets(
        &mut self,
        effects1: &[crate::ast::Effect<'db>],
        effects2: &[crate::ast::Effect<'db>],
        r1: EffectRow<'db>,
        r2: EffectRow<'db>,
    ) -> Result<(), SolveError<'db>> {
        if effects1.len() != effects2.len() {
            return Err(SolveError::RowMismatch {
                expected: r1,
                actual: r2,
            });
        }

        // For each effect in effects1, find a matching effect in effects2
        for e1 in effects1 {
            // Find effects with the same name
            let same_name: Vec<_> = effects2.iter().filter(|e2| e1.name == e2.name).collect();

            if same_name.is_empty() {
                return Err(SolveError::RowMismatch {
                    expected: r1,
                    actual: r2,
                });
            }

            // Check for arity mismatch
            for e2 in &same_name {
                if e1.args.len() != e2.args.len() {
                    return Err(SolveError::EffectArgArityMismatch {
                        effect_name: e1.name,
                        expected: e1.args.len(),
                        found: e2.args.len(),
                    });
                }
            }

            // Try to find a matching effect (args unify successfully)
            let mut found_match = false;
            for e2 in &same_name {
                // Try unifying args - if successful, we found a match
                let args_match = e1
                    .args
                    .iter()
                    .zip(e2.args.iter())
                    .all(|(a1, a2)| self.types_unifiable(*a1, *a2));

                if args_match {
                    // Actually perform the unification
                    for (a1, a2) in e1.args.iter().zip(e2.args.iter()) {
                        self.unify_types(*a1, *a2)?;
                    }
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                // Same name and arity, but args don't unify → different abilities
                return Err(SolveError::RowMismatch {
                    expected: r1,
                    actual: r2,
                });
            }
        }

        Ok(())
    }

    /// Check if two types can be unified without modifying substitution.
    ///
    /// This is a quick check that doesn't perform actual unification.
    fn types_unifiable(&self, t1: Type<'db>, t2: Type<'db>) -> bool {
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
            | (TypeKind::String, TypeKind::String)
            | (TypeKind::Bytes, TypeKind::Bytes)
            | (TypeKind::Rune, TypeKind::Rune)
            | (TypeKind::Nil, TypeKind::Nil) => true,
            (TypeKind::Named { name: n1, args: a1 }, TypeKind::Named { name: n2, args: a2 }) => {
                n1 == n2
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
            _ => false,
        }
    }

    /// Compute effect difference with unification support.
    ///
    /// Returns matched effects and unmatched effects from list2.
    fn compute_effect_difference_with_unify(
        &mut self,
        list1: &[crate::ast::Effect<'db>],
        list2: &[crate::ast::Effect<'db>],
    ) -> Result<Vec<crate::ast::Effect<'db>>, SolveError<'db>> {
        let mut only_list2 = Vec::new();

        for e2 in list2 {
            // Check for arity mismatch with same-named effects
            for e1 in list1.iter().filter(|e1| e1.name == e2.name) {
                if e1.args.len() != e2.args.len() {
                    return Err(SolveError::EffectArgArityMismatch {
                        effect_name: e1.name,
                        expected: e1.args.len(),
                        found: e2.args.len(),
                    });
                }
            }

            // Try to find a matching effect
            let mut found_match = false;
            for e1 in list1.iter().filter(|e1| e1.name == e2.name) {
                let args_match = e1
                    .args
                    .iter()
                    .zip(e2.args.iter())
                    .all(|(a1, a2)| self.types_unifiable(*a1, *a2));

                if args_match {
                    // Perform unification
                    for (a1, a2) in e1.args.iter().zip(e2.args.iter()) {
                        self.unify_types(*a1, *a2)?;
                    }
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                only_list2.push(e2.clone());
            }
        }

        Ok(only_list2)
    }

    /// Compute effect split with unification support.
    ///
    /// Returns (only_list1, only_list2) after matching and unifying.
    fn compute_effect_split_with_unify(
        &mut self,
        list1: &[crate::ast::Effect<'db>],
        list2: &[crate::ast::Effect<'db>],
    ) -> Result<(Vec<crate::ast::Effect<'db>>, Vec<crate::ast::Effect<'db>>), SolveError<'db>> {
        let mut only_list1 = Vec::new();
        let mut only_list2 = Vec::new();
        let mut matched_in_list2 = vec![false; list2.len()];

        // Find matches from list1's perspective
        for e1 in list1 {
            // Check for arity mismatch
            for e2 in list2.iter().filter(|e2| e2.name == e1.name) {
                if e1.args.len() != e2.args.len() {
                    return Err(SolveError::EffectArgArityMismatch {
                        effect_name: e1.name,
                        expected: e1.args.len(),
                        found: e2.args.len(),
                    });
                }
            }

            let mut found_match = false;
            for (i, e2) in list2.iter().enumerate() {
                if e1.name != e2.name || matched_in_list2[i] {
                    continue;
                }

                let args_match = e1
                    .args
                    .iter()
                    .zip(e2.args.iter())
                    .all(|(a1, a2)| self.types_unifiable(*a1, *a2));

                if args_match {
                    // Perform unification
                    for (a1, a2) in e1.args.iter().zip(e2.args.iter()) {
                        self.unify_types(*a1, *a2)?;
                    }
                    matched_in_list2[i] = true;
                    found_match = true;
                    break;
                }
            }

            if !found_match {
                only_list1.push(e1.clone());
            }
        }

        // Collect unmatched from list2
        for (i, e2) in list2.iter().enumerate() {
            if !matched_in_list2[i] {
                only_list2.push(e2.clone());
            }
        }

        Ok((only_list1, only_list2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Effect, EffectRow, UniVarSource};
    use trunk_ir::Symbol;

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
    fn test_occurs_check_in_effect_row() {
        // Unifying ?a with fn() ->{State(?a)} Int should fail the occurs check,
        // because ?a appears inside the effect row's type arguments.
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = fresh_var(&db, 0);
        let int_ty = Type::new(&db, TypeKind::Int);

        // Effect row: {State(?a)}
        let effect = EffectRow::new(
            &db,
            vec![Effect {
                name: trunk_ir::Symbol::new("State"),
                args: vec![var_ty],
            }],
            None,
        );

        // fn() ->{State(?a)} Int
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect,
            },
        );

        let result = solver.unify_types(var_ty, func_ty);
        assert!(
            matches!(result, Err(SolveError::OccursCheck { .. })),
            "Expected occurs check failure for ?a = fn() ->{{State(?a)}} Int"
        );
    }

    #[test]
    fn test_occurs_check_not_triggered_for_different_var_in_effect() {
        // Unifying ?a with fn() ->{State(?b)} Int should succeed,
        // because ?a does not appear in the effect row.
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_a = fresh_var(&db, 0);
        let var_b = fresh_var(&db, 1);
        let int_ty = Type::new(&db, TypeKind::Int);

        let effect = EffectRow::new(
            &db,
            vec![Effect {
                name: trunk_ir::Symbol::new("State"),
                args: vec![var_b],
            }],
            None,
        );

        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect,
            },
        );

        let result = solver.unify_types(var_a, func_ty);
        assert!(
            result.is_ok(),
            "Should not trigger occurs check when the var is different"
        );
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
    fn test_pure_callee_in_effectful_context() {
        // Test that calling a pure function from an effectful context succeeds
        // without modifying the caller's effect row.
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        // Create a pure effect row (callee's effect)
        let pure_effect = EffectRow::new(&db, vec![], None);

        // Create an effectful row with State effect (caller's context)
        let row_var = EffectVar { id: 100 };
        let state_effect = Effect {
            name: Symbol::new("State"),
            args: vec![Type::new(&db, TypeKind::Int)],
        };
        let effectful_row = EffectRow::new(&db, vec![state_effect], Some(row_var));

        // Unifying pure with effectful should succeed
        let result = solver.unify_rows(pure_effect, effectful_row);
        assert!(
            result.is_ok(),
            "Pure callee should be callable from effectful context"
        );

        // The row variable should NOT be bound - caller's effect stays unchanged
        let resolved = solver.row_subst.get(row_var.id);
        assert!(
            resolved.is_none(),
            "Caller's row variable should not be modified when calling pure function"
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

    #[test]
    fn test_type_subst_applies_to_effect_args() {
        // State(?a) where ?a = Int should become State(Int)
        let db = test_db();
        let mut type_subst = TypeSubst::new();

        let int_ty = Type::new(&db, TypeKind::Int);
        let var_ty = fresh_var(&db, 0);
        let var_id = match var_ty.kind(&db) {
            TypeKind::UniVar { id } => *id,
            _ => unreachable!(),
        };
        type_subst.insert(var_id, int_ty);

        // fn() ->{State(?a)} Int
        let effect = EffectRow::new(
            &db,
            vec![Effect {
                name: trunk_ir::Symbol::new("State"),
                args: vec![var_ty],
            }],
            None,
        );
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect,
            },
        );

        let row_subst = RowSubst::new();
        let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

        if let TypeKind::Func { effect, .. } = result.kind(&db) {
            let effects = effect.effects(&db);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].name, trunk_ir::Symbol::new("State"));
            assert_eq!(effects[0].args.len(), 1);
            assert_eq!(
                effects[0].args[0], int_ty,
                "Effect arg ?a should be substituted to Int"
            );
        } else {
            panic!("Expected Func type");
        }
    }

    #[test]
    fn test_type_subst_preserves_unchanged_effect_args() {
        // State(Int) with no relevant substitution should remain unchanged
        let db = test_db();
        let type_subst = TypeSubst::new();
        let int_ty = Type::new(&db, TypeKind::Int);

        let effect = EffectRow::new(
            &db,
            vec![Effect {
                name: trunk_ir::Symbol::new("State"),
                args: vec![int_ty],
            }],
            None,
        );
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect,
            },
        );

        let row_subst = RowSubst::new();
        let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

        if let TypeKind::Func {
            effect: result_effect,
            ..
        } = result.kind(&db)
        {
            let effects = result_effect.effects(&db);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].args[0], int_ty);
        } else {
            panic!("Expected Func type");
        }
    }

    #[test]
    #[should_panic(expected = "BoundVar reached solver")]
    fn test_bound_var_panics_in_debug() {
        // BoundVar should never reach the solver — it must be instantiated first.
        // In debug mode, this triggers a debug_assert panic.
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let bound_var = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int_ty = Type::new(&db, TypeKind::Int);

        let _ = solver.unify_types(bound_var, int_ty);
    }

    // =========================================================================
    // Generalization tests
    // =========================================================================

    #[test]
    fn test_generalize_no_univars() {
        // Concrete type (Int) → no type params, type unchanged
        let db = test_db();
        let subst = TypeSubst::new();
        let row_subst = RowSubst::new();
        let int_ty = Type::new(&db, TypeKind::Int);

        let (generalized, params) = subst.generalize(&db, int_ty, &row_subst);
        assert_eq!(generalized, int_ty);
        assert!(params.is_empty());
    }

    #[test]
    fn test_generalize_single_univar() {
        // fn(?a) -> ?a  →  fn(BoundVar(0)) -> BoundVar(0), 1 type param
        let db = test_db();
        let subst = TypeSubst::new();
        let row_subst = RowSubst::new();

        let var_ty = fresh_var(&db, 0);
        let effect = EffectRow::new(&db, vec![], None);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![var_ty],
                result: var_ty,
                effect,
            },
        );

        let (generalized, params) = subst.generalize(&db, func_ty, &row_subst);
        assert_eq!(params.len(), 1);

        if let TypeKind::Func {
            params: gen_params,
            result,
            ..
        } = generalized.kind(&db)
        {
            assert!(matches!(
                gen_params[0].kind(&db),
                TypeKind::BoundVar { index: 0 }
            ));
            assert!(matches!(result.kind(&db), TypeKind::BoundVar { index: 0 }));
        } else {
            panic!("Expected Func type");
        }
    }

    #[test]
    fn test_generalize_two_univars() {
        // fn(?a) -> ?b  →  fn(BoundVar(0)) -> BoundVar(1), 2 type params
        let db = test_db();
        let subst = TypeSubst::new();
        let row_subst = RowSubst::new();

        let var_a = fresh_var(&db, 0);
        let var_b = fresh_var(&db, 1);
        let effect = EffectRow::new(&db, vec![], None);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![var_a],
                result: var_b,
                effect,
            },
        );

        let (generalized, params) = subst.generalize(&db, func_ty, &row_subst);
        assert_eq!(params.len(), 2);

        if let TypeKind::Func {
            params: gen_params,
            result,
            ..
        } = generalized.kind(&db)
        {
            assert!(matches!(
                gen_params[0].kind(&db),
                TypeKind::BoundVar { index: 0 }
            ));
            assert!(matches!(result.kind(&db), TypeKind::BoundVar { index: 1 }));
        } else {
            panic!("Expected Func type");
        }
    }

    #[test]
    fn test_generalize_resolved_univar_not_generalized() {
        // ?a resolved to Int → after apply + generalize: no type params, no BoundVars
        let db = test_db();
        let mut subst = TypeSubst::new();
        let row_subst = RowSubst::new();

        let var_ty = fresh_var(&db, 0);
        let int_ty = Type::new(&db, TypeKind::Int);
        let var_id = match var_ty.kind(&db) {
            TypeKind::UniVar { id } => *id,
            _ => unreachable!(),
        };
        subst.insert(var_id, int_ty);

        let effect = EffectRow::new(&db, vec![], None);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![var_ty],
                result: var_ty,
                effect,
            },
        );

        // Apply substitution first (as done in Phase 4)
        let applied = subst.apply_with_rows(&db, func_ty, &row_subst);
        let (generalized, params) = subst.generalize(&db, applied, &row_subst);
        assert!(params.is_empty());

        if let TypeKind::Func {
            params: gen_params,
            result,
            ..
        } = generalized.kind(&db)
        {
            assert_eq!(gen_params[0], int_ty);
            assert_eq!(*result, int_ty);
        } else {
            panic!("Expected Func type");
        }
    }

    // =========================================================================
    // Advanced row unification tests
    // =========================================================================

    #[test]
    fn test_row_unification_with_effects() {
        // {Console} unifies with {Console}
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let console = Effect {
            name: trunk_ir::Symbol::new("Console"),
            args: vec![],
        };
        let r1 = EffectRow::new(&db, vec![console.clone()], None);
        let r2 = EffectRow::new(&db, vec![console], None);

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_row_unification_closed_rows_mismatch() {
        // {Console} does not unify with {IO}
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let console = Effect {
            name: trunk_ir::Symbol::new("Console"),
            args: vec![],
        };
        let io = Effect {
            name: trunk_ir::Symbol::new("IO"),
            args: vec![],
        };
        let r1 = EffectRow::new(&db, vec![console], None);
        let r2 = EffectRow::new(&db, vec![io], None);

        let result = solver.unify_rows(r1, r2);
        assert!(matches!(result, Err(SolveError::RowMismatch { .. })));
    }

    #[test]
    fn test_row_unification_open_row_binds_to_difference() {
        // {Console | e} unifies with {Console, IO}
        // Should bind e to {IO}
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let console = Effect {
            name: trunk_ir::Symbol::new("Console"),
            args: vec![],
        };
        let io = Effect {
            name: trunk_ir::Symbol::new("IO"),
            args: vec![],
        };
        let row_var = EffectVar { id: 50 };

        let r1 = EffectRow::new(&db, vec![console.clone()], Some(row_var));
        let r2 = EffectRow::new(&db, vec![console, io.clone()], None);

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());

        // e should be bound to {IO}
        let resolved = solver.row_subst.get(row_var.id).unwrap();
        let effects = resolved.effects(&db);
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].name, trunk_ir::Symbol::new("IO"));
        assert!(resolved.rest(&db).is_none()); // Closed
    }

    #[test]
    fn test_row_unification_two_open_rows() {
        // {Console | e1} unifies with {IO | e2}
        // Should create fresh e3:
        //   e1 = {IO | e3}
        //   e2 = {Console | e3}
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let console = Effect {
            name: trunk_ir::Symbol::new("Console"),
            args: vec![],
        };
        let io = Effect {
            name: trunk_ir::Symbol::new("IO"),
            args: vec![],
        };
        let e1 = EffectVar { id: 100 };
        let e2 = EffectVar { id: 200 };

        let r1 = EffectRow::new(&db, vec![console.clone()], Some(e1));
        let r2 = EffectRow::new(&db, vec![io.clone()], Some(e2));

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());

        // e1 should be bound to {IO | e3} for some fresh e3
        let e1_resolved = solver.row_subst.get(e1.id).unwrap();
        let e1_effects = e1_resolved.effects(&db);
        assert_eq!(e1_effects.len(), 1);
        assert_eq!(e1_effects[0].name, trunk_ir::Symbol::new("IO"));
        assert!(e1_resolved.rest(&db).is_some()); // Open with e3

        // e2 should be bound to {Console | e3}
        let e2_resolved = solver.row_subst.get(e2.id).unwrap();
        let e2_effects = e2_resolved.effects(&db);
        assert_eq!(e2_effects.len(), 1);
        assert_eq!(e2_effects[0].name, trunk_ir::Symbol::new("Console"));
        assert!(e2_resolved.rest(&db).is_some()); // Open with e3

        // Both should have the same fresh variable
        assert_eq!(e1_resolved.rest(&db), e2_resolved.rest(&db));
    }

    #[test]
    fn test_row_unification_unifies_type_args() {
        // {State(?a)} unifies with {State(Int)}
        // Should bind ?a to Int
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = fresh_var(&db, 0);
        let int_ty = Type::new(&db, TypeKind::Int);

        let state_var = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![var_ty],
        };
        let state_int = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![int_ty],
        };

        let r1 = EffectRow::new(&db, vec![state_var], None);
        let r2 = EffectRow::new(&db, vec![state_int], None);

        let result = solver.unify_rows(r1, r2);
        assert!(result.is_ok());

        // ?a should be bound to Int
        assert_eq!(solver.type_subst.apply(&db, var_ty), int_ty);
    }

    #[test]
    fn test_row_unification_same_var_different_effects_fails() {
        // {Console | e} and {IO | e} with the same e should fail
        // (because the concrete effects don't match)
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let console = Effect {
            name: trunk_ir::Symbol::new("Console"),
            args: vec![],
        };
        let io = Effect {
            name: trunk_ir::Symbol::new("IO"),
            args: vec![],
        };
        let row_var = EffectVar { id: 42 };

        let r1 = EffectRow::new(&db, vec![console], Some(row_var));
        let r2 = EffectRow::new(&db, vec![io], Some(row_var));

        let result = solver.unify_rows(r1, r2);
        assert!(matches!(result, Err(SolveError::RowMismatch { .. })));
    }

    #[test]
    fn test_different_effect_arity_returns_arity_mismatch() {
        // State(Int) and State() have different arity - this is an arity mismatch error
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);

        let state_with_arg = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![int_ty],
        };
        let state_no_arg = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![],
        };

        let r1 = EffectRow::new(&db, vec![state_with_arg], None);
        let r2 = EffectRow::new(&db, vec![state_no_arg], None);

        let result = solver.unify_rows(r1, r2);
        // Same ability name but different arity is an arity mismatch error
        assert!(
            matches!(
                result,
                Err(SolveError::EffectArgArityMismatch {
                    effect_name,
                    expected: 1,
                    found: 0,
                }) if effect_name == trunk_ir::Symbol::new("State")
            ),
            "Expected EffectArgArityMismatch error, got {:?}",
            result
        );
    }

    #[test]
    fn test_different_effect_arg_types_returns_row_mismatch() {
        // State(Int) and State(Bool) are different parameterized abilities
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        let state_int = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![int_ty],
        };
        let state_bool = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![bool_ty],
        };

        let r1 = EffectRow::new(&db, vec![state_int], None);
        let r2 = EffectRow::new(&db, vec![state_bool], None);

        let result = solver.unify_rows(r1, r2);
        // State(Int) and State(Bool) are distinct abilities, so this is a row mismatch
        assert!(
            matches!(result, Err(SolveError::RowMismatch { .. })),
            "Expected RowMismatch error, got {:?}",
            result
        );
    }

    #[test]
    fn test_same_effect_args_unifies_successfully() {
        // State(Int) and State(Int) are the same ability - should unify
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);

        let state_int1 = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![int_ty],
        };
        let state_int2 = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![int_ty],
        };

        let r1 = EffectRow::new(&db, vec![state_int1], None);
        let r2 = EffectRow::new(&db, vec![state_int2], None);

        let result = solver.unify_rows(r1, r2);
        assert!(
            result.is_ok(),
            "Same effects should unify, got {:?}",
            result
        );
    }

    // =========================================================================
    // Parameterized ability unification with type variables
    // =========================================================================

    #[test]
    fn test_effect_with_type_var_unifies_with_concrete() {
        // State(?a) and State(Int) should unify with ?a = Int
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = fresh_var(&db, 0);
        let int_ty = Type::new(&db, TypeKind::Int);

        let state_var = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![var_ty],
        };
        let state_int = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![int_ty],
        };

        let r1 = EffectRow::new(&db, vec![state_var], None);
        let r2 = EffectRow::new(&db, vec![state_int], None);

        let result = solver.unify_rows(r1, r2);
        assert!(
            result.is_ok(),
            "State(?a) should unify with State(Int), got {:?}",
            result
        );

        // Check that ?a was unified to Int
        let resolved = solver.type_subst.apply(&db, var_ty);
        assert_eq!(resolved, int_ty, "Type variable should be unified to Int");
    }

    #[test]
    fn test_effect_with_two_type_vars_unifies() {
        // State(?a) and State(?b) should unify with ?a = ?b
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_a = fresh_var(&db, 0);
        let var_b = fresh_var(&db, 1);

        let state_a = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![var_a],
        };
        let state_b = Effect {
            name: trunk_ir::Symbol::new("State"),
            args: vec![var_b],
        };

        let r1 = EffectRow::new(&db, vec![state_a], None);
        let r2 = EffectRow::new(&db, vec![state_b], None);

        let result = solver.unify_rows(r1, r2);
        assert!(
            result.is_ok(),
            "State(?a) should unify with State(?b), got {:?}",
            result
        );

        // Check that they are unified (both resolve to the same type)
        let resolved_a = solver.type_subst.apply(&db, var_a);
        let resolved_b = solver.type_subst.apply(&db, var_b);
        assert_eq!(resolved_a, resolved_b, "Type variables should be unified");
    }

    #[test]
    fn test_effect_mixed_type_var_and_concrete_unifies() {
        // Pair(?a, Int) and Pair(Bool, ?b) should unify with ?a = Bool, ?b = Int
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_a = fresh_var(&db, 0);
        let var_b = fresh_var(&db, 1);
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        let pair1 = Effect {
            name: trunk_ir::Symbol::new("Pair"),
            args: vec![var_a, int_ty],
        };
        let pair2 = Effect {
            name: trunk_ir::Symbol::new("Pair"),
            args: vec![bool_ty, var_b],
        };

        let r1 = EffectRow::new(&db, vec![pair1], None);
        let r2 = EffectRow::new(&db, vec![pair2], None);

        let result = solver.unify_rows(r1, r2);
        assert!(
            result.is_ok(),
            "Pair(?a, Int) should unify with Pair(Bool, ?b), got {:?}",
            result
        );

        // Check that ?a = Bool and ?b = Int
        assert_eq!(solver.type_subst.apply(&db, var_a), bool_ty);
        assert_eq!(solver.type_subst.apply(&db, var_b), int_ty);
    }

    #[test]
    fn test_types_unifiable_simple() {
        let db = test_db();
        let solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let var_ty = fresh_var(&db, 0);

        // Same types are unifiable
        assert!(solver.types_unifiable(int_ty, int_ty));

        // Different concrete types are not unifiable
        assert!(!solver.types_unifiable(int_ty, bool_ty));

        // Type variable is unifiable with any type
        assert!(solver.types_unifiable(var_ty, int_ty));
        assert!(solver.types_unifiable(int_ty, var_ty));

        // Two type variables are unifiable
        let var_ty2 = fresh_var(&db, 1);
        assert!(solver.types_unifiable(var_ty, var_ty2));
    }

    #[test]
    fn test_types_unifiable_func() {
        let db = test_db();
        let solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let effect = EffectRow::new(&db, vec![], None);

        // Same function types are unifiable
        let func1 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect,
            },
        );
        let func2 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect,
            },
        );
        assert!(solver.types_unifiable(func1, func2));

        // Different param types are not unifiable
        let func3 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bool_ty],
                result: int_ty,
                effect,
            },
        );
        assert!(!solver.types_unifiable(func1, func3));

        // Different result types are not unifiable
        let func4 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: bool_ty,
                effect,
            },
        );
        assert!(!solver.types_unifiable(func1, func4));

        // Function type with type variable in params is unifiable
        let var_ty = fresh_var(&db, 0);
        let func_with_var = Type::new(
            &db,
            TypeKind::Func {
                params: vec![var_ty],
                result: int_ty,
                effect,
            },
        );
        assert!(solver.types_unifiable(func1, func_with_var));
    }

    #[test]
    fn test_types_unifiable_app() {
        let db = test_db();
        let solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let list_ctor = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![],
            },
        );
        let option_ctor = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Option"),
                args: vec![],
            },
        );

        // Same App types are unifiable
        let app1 = Type::new(
            &db,
            TypeKind::App {
                ctor: list_ctor,
                args: vec![int_ty],
            },
        );
        let app2 = Type::new(
            &db,
            TypeKind::App {
                ctor: list_ctor,
                args: vec![int_ty],
            },
        );
        assert!(solver.types_unifiable(app1, app2));

        // Different constructor is not unifiable
        let app3 = Type::new(
            &db,
            TypeKind::App {
                ctor: option_ctor,
                args: vec![int_ty],
            },
        );
        assert!(!solver.types_unifiable(app1, app3));

        // Different arg types are not unifiable
        let app4 = Type::new(
            &db,
            TypeKind::App {
                ctor: list_ctor,
                args: vec![bool_ty],
            },
        );
        assert!(!solver.types_unifiable(app1, app4));

        // App with type variable in args is unifiable
        let var_ty = fresh_var(&db, 0);
        let app_with_var = Type::new(
            &db,
            TypeKind::App {
                ctor: list_ctor,
                args: vec![var_ty],
            },
        );
        assert!(solver.types_unifiable(app1, app_with_var));
    }

    // =========================================================================
    // row_occurs_in_type tests for params/result recursion
    // =========================================================================

    #[test]
    fn test_row_occurs_in_func_params() {
        // row var in function parameter should be detected
        let db = test_db();
        let solver = TypeSolver::new(&db);

        let row_var = EffectVar { id: 42 };
        let int_ty = Type::new(&db, TypeKind::Int);

        // fn(fn() ->{e} Int) -> Int where we check for e in outer func
        let inner_effect = EffectRow::new(&db, vec![], Some(row_var));
        let inner_func = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect: inner_effect,
            },
        );
        let outer_effect = EffectRow::new(&db, vec![], None);
        let outer_func = Type::new(
            &db,
            TypeKind::Func {
                params: vec![inner_func],
                result: int_ty,
                effect: outer_effect,
            },
        );

        assert!(
            solver.row_occurs_in_type(row_var, outer_func),
            "Row variable in param's effect should be detected"
        );
    }

    #[test]
    fn test_row_occurs_in_func_result() {
        // row var in function result should be detected
        let db = test_db();
        let solver = TypeSolver::new(&db);

        let row_var = EffectVar { id: 42 };
        let int_ty = Type::new(&db, TypeKind::Int);

        // fn() -> fn() ->{e} Int where we check for e in outer func
        let inner_effect = EffectRow::new(&db, vec![], Some(row_var));
        let inner_func = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect: inner_effect,
            },
        );
        let outer_effect = EffectRow::new(&db, vec![], None);
        let outer_func = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: inner_func,
                effect: outer_effect,
            },
        );

        assert!(
            solver.row_occurs_in_type(row_var, outer_func),
            "Row variable in result's effect should be detected"
        );
    }

    #[test]
    fn test_row_not_in_func_if_absent() {
        // row var not present should return false
        let db = test_db();
        let solver = TypeSolver::new(&db);

        let row_var = EffectVar { id: 42 };
        let other_var = EffectVar { id: 99 };
        let int_ty = Type::new(&db, TypeKind::Int);

        // fn() -> Int with empty effect
        let effect = EffectRow::new(&db, vec![], Some(other_var));
        let func = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect,
            },
        );

        assert!(
            !solver.row_occurs_in_type(row_var, func),
            "Row variable not present should not be detected"
        );
    }
}
