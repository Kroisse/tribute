//! Type constraint solver.
//!
//! Owns substitution storage, fresh-variable allocation, and the order in which
//! deferred producers and constraints are settled. Algorithm modules operate on
//! this same state; they do not introduce independent solver rounds.
//!
//! - `unify`: structural type equality and occurs checks
//! - `relations`: deferred common-result joins
//! - `rows`: effect equality, union, removal, and exact set matching
//! - `substitution`: substitution application and binder-aware generalization
//! - `error`: structured failures and their display

mod error;
mod relations;
mod rows;
mod substitution;
mod unify;

pub use error::{LocatedSolveError, SolveError};

use std::collections::HashMap;

use trunk_ir::smallvec::SmallVec;

use crate::ast::{
    Effect, EffectRow, EffectVar, NodeId, Type, TypeKind, TypeParam, UniVarId, UniVarSource,
    collect_effect_vars,
};

use super::constraint::{Constraint, ConstraintOrigin, ConstraintSet};

/// Apply a type-transforming function to all effect type arguments in an effect row.
///
/// This is the shared logic used by `apply_with_rows`, `replace_univars_with_bound`,
/// and `apply_type_subst_to_row`: map each effect's type args through `f`, then
/// rebuild the row only if something changed.
pub(super) fn map_effect_row_type_args<'db>(
    db: &'db dyn salsa::Database,
    row: EffectRow<'db>,
    mut f: impl FnMut(Type<'db>) -> Type<'db>,
) -> EffectRow<'db> {
    let effects = row.effects(db);
    let new_effects: Vec<_> = effects
        .iter()
        .map(|e| {
            let new_args: Vec<_> = e.args.iter().map(|a| f(*a)).collect();
            Effect {
                ability_id: e.ability_id,
                args: new_args,
            }
        })
        .collect();
    if new_effects != *effects {
        EffectRow::new(db, new_effects, row.rest(db))
    } else {
        row
    }
}

/// Type substitution: maps type variable IDs to types.
#[derive(Clone, Debug, Default)]
pub struct TypeSubst<'db> {
    map: HashMap<UniVarId<'db>, Type<'db>>,
}

/// Row substitution: maps row variable IDs to effect rows.
#[derive(Clone, Debug, Default)]
pub struct RowSubst<'db> {
    map: HashMap<u64, EffectRow<'db>>,
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
    /// Expression relations survive each equality/row and deferred-method round.
    pending_relations: Vec<Constraint<'db>>,
    pending_row_unions: Vec<(crate::ast::RowUnion<'db>, Option<ConstraintOrigin>)>,
    pending_row_removals: Vec<(
        crate::ast::RowRemoval<'db>,
        Option<super::constraint::ConstraintOrigin>,
    )>,
    /// Results whose producer signature has not yet been resolved.
    pending_producers: Vec<PendingProducer<'db>>,
}

struct PendingProducer<'db> {
    node_id: NodeId,
    result: Type<'db>,
    inputs: Vec<Type<'db>>,
}

impl<'db> TypeSolver<'db> {
    /// Create a new solver.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            type_subst: TypeSubst::new(),
            row_subst: RowSubst::new(),
            next_row_var: 0,
            pending_relations: Vec::new(),
            pending_row_unions: Vec::new(),
            pending_row_removals: Vec::new(),
            pending_producers: Vec::new(),
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

    pub(crate) fn next_row_var(&self) -> u64 {
        self.next_row_var
    }

    pub(crate) fn reserve_row_vars(&mut self, next: u64) {
        self.next_row_var = self.next_row_var.max(next);
    }

    pub fn defer_producer(&mut self, node_id: NodeId, result: Type<'db>, inputs: Vec<Type<'db>>) {
        if !self
            .pending_producers
            .iter()
            .any(|pending| pending.node_id == node_id)
        {
            self.pending_producers.push(PendingProducer {
                node_id,
                result,
                inputs,
            });
        }
    }

    pub fn resolve_producer(&mut self, node_id: NodeId) {
        self.pending_producers
            .retain(|pending| pending.node_id != node_id);
    }

    /// Variables in an unresolved relation cannot be quantified independently.
    pub fn pending_variables(&self) -> (Vec<UniVarId<'db>>, Vec<EffectVar>) {
        let mut types = Vec::new();
        for producer in &self.pending_producers {
            types.push(producer.result);
            types.extend(producer.inputs.iter().copied());
        }
        for relation in &self.pending_relations {
            match relation {
                Constraint::TypeCoerce(actual, expected, _) => types.extend([*actual, *expected]),
                Constraint::TypeJoin {
                    sources, result, ..
                } => {
                    types.push(*result);
                    types.extend(sources.iter().map(|(ty, _)| *ty));
                }
                _ => unreachable!("only expression relations are deferred"),
            }
        }
        let mut vars = Vec::new();
        let mut effects = Vec::new();
        for ty in types {
            let ty = self
                .type_subst
                .apply_with_rows(self.db, ty, &self.row_subst);
            self.type_subst
                .collect_univars_from_type(self.db, ty, &self.row_subst, &mut vars);
            for effect in collect_effect_vars(self.db, ty) {
                if !effects.contains(&effect) {
                    effects.push(effect);
                }
            }
        }
        self.expand_union_dependencies(&mut vars, &mut effects);
        (vars, effects)
    }

    /// At an inference boundary, equate remaining ordinary variables only when
    /// no unresolved producer or common-result relation can still supply Never.
    pub fn finalize_relations(&mut self) -> Result<(), LocatedSolveError<'db>> {
        self.settle_relations(true)
    }

    fn settle_relations(&mut self, finalize: bool) -> Result<(), LocatedSolveError<'db>> {
        let mut first_error = None;
        loop {
            let before = (
                self.type_subst.map.len(),
                self.row_subst.map.len(),
                self.pending_relations.len(),
                self.pending_row_unions.len() + self.pending_row_removals.len(),
            );
            let mut protected: Vec<_> = self
                .pending_producers
                .iter()
                .map(|producer| producer.result)
                .collect();
            protected.extend(
                self.pending_relations
                    .iter()
                    .filter_map(|relation| match relation {
                        Constraint::TypeJoin { result, .. } => Some(*result),
                        _ => None,
                    }),
            );
            let protected: Vec<_> = protected
                .into_iter()
                .map(|ty| self.type_subst.apply(self.db, ty))
                .collect();
            for relation in std::mem::take(&mut self.pending_relations) {
                let outcome = match &relation {
                    Constraint::TypeCoerce(actual, expected, origin) => {
                        let actual = self.type_subst.apply(self.db, *actual);
                        let expected = self.type_subst.apply(self.db, *expected);
                        if actual == expected || matches!(actual.kind(self.db), TypeKind::Never) {
                            Ok(true)
                        } else if matches!(actual.kind(self.db), TypeKind::UniVar { .. })
                            && (!finalize || protected.contains(&actual))
                        {
                            Ok(false)
                        } else {
                            self.unify_types(expected, actual)
                                .map(|()| true)
                                .map_err(|error| LocatedSolveError {
                                    error,
                                    origin: Some(*origin),
                                })
                        }
                    }
                    Constraint::TypeJoin {
                        sources,
                        result,
                        origin,
                        complete,
                    } => {
                        self.solve_join(sources, *result, *origin, *complete, finalize, &protected)
                    }
                    _ => unreachable!("only expression relations are deferred"),
                };
                match outcome {
                    Ok(true) => {}
                    Ok(false) => self.pending_relations.push(relation),
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                }
            }
            if let Err(error) = self.settle_row_unions() {
                first_error.get_or_insert(error);
            }
            let after = (
                self.type_subst.map.len(),
                self.row_subst.map.len(),
                self.pending_relations.len(),
                self.pending_row_unions.len() + self.pending_row_removals.len(),
            );
            if before == after {
                if finalize {
                    match self.resolve_join_cycle() {
                        Ok(true) => continue,
                        Ok(false) => {}
                        Err(error) => {
                            first_error.get_or_insert(error);
                        }
                    }
                }
                break;
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Generate a fresh type variable for post-solve instantiation.
    ///
    /// Uses `UniVarSource::Solver` to prevent aliasing with function-scoped UniVars.
    pub fn fresh_type_var(&mut self, db: &'db dyn salsa::Database) -> Type<'db> {
        let counter = self.next_row_var;
        self.next_row_var += 1;
        let id = UniVarId::new(db, UniVarSource::Solver { index: counter }, 0);
        Type::new(db, TypeKind::UniVar { id })
    }

    /// Generate a fresh row variable for post-solve instantiation.
    pub(super) fn fresh_row_var(&mut self) -> EffectVar {
        let id = self.next_row_var;
        self.next_row_var += 1;
        EffectVar { id }
    }

    pub(crate) fn reserve_effect_vars_in_type(&mut self, ty: Type<'db>) {
        for var in collect_effect_vars(self.db, ty) {
            self.reserve_effect_var(var);
        }
    }

    fn reserve_effect_var(&mut self, var: EffectVar) {
        let next = var
            .id
            .checked_add(1)
            .expect("cannot allocate an effect variable after u64::MAX");
        self.next_row_var = self.next_row_var.max(next);
    }

    pub(crate) fn reserve_effect_vars_in_row(&mut self, row: EffectRow<'db>) {
        if let Some(var) = row.rest(self.db) {
            self.reserve_effect_var(var);
        }
        for effect in row.effects(self.db) {
            for arg in &effect.args {
                self.reserve_effect_vars_in_type(*arg);
            }
        }
    }

    fn reserve_effect_vars_in_constraint(&mut self, constraint: &Constraint<'db>) {
        match constraint {
            Constraint::TypeEq(left, right)
            | Constraint::TypeEqAt(left, right, _)
            | Constraint::TypeCoerce(left, right, _) => {
                self.reserve_effect_vars_in_type(*left);
                self.reserve_effect_vars_in_type(*right);
            }
            Constraint::TypeJoin {
                sources, result, ..
            } => {
                self.reserve_effect_vars_in_type(*result);
                for (source, _) in sources {
                    self.reserve_effect_vars_in_type(*source);
                }
            }
            Constraint::RowEq(left, right) | Constraint::RowEqAt(left, right, _) => {
                self.reserve_effect_vars_in_row(*left);
                self.reserve_effect_vars_in_row(*right);
            }
            Constraint::RowRemoval(removal, _) => {
                for row in removal.rows() {
                    self.reserve_effect_vars_in_row(row);
                }
            }
            Constraint::RowUnion(union, _) => {
                for row in union.sources.iter().chain(std::iter::once(&union.result)) {
                    self.reserve_effect_vars_in_row(*row);
                }
            }
            Constraint::And(constraints) => {
                for constraint in constraints {
                    self.reserve_effect_vars_in_constraint(constraint);
                }
            }
        }
    }

    /// Solve a set of constraints.
    ///
    /// Processes all constraints even if some fail, so that as many type
    /// variables as possible are resolved.  Returns the first error (if any)
    /// after all constraints have been attempted.
    pub fn solve(&mut self, constraints: ConstraintSet<'db>) -> Result<(), SolveError<'db>> {
        self.solve_with_origin(constraints)
            .map_err(|error| error.error)
    }

    /// Solve constraints while retaining the source origin of the first error.
    pub fn solve_with_origin(
        &mut self,
        constraints: ConstraintSet<'db>,
    ) -> Result<(), LocatedSolveError<'db>> {
        for constraint in constraints.constraints() {
            self.reserve_effect_vars_in_constraint(constraint);
        }
        let constraints_vec = constraints.into_constraints();
        let mut first_error: Option<LocatedSolveError<'db>> = None;
        for constraint in constraints_vec.into_iter() {
            if let Err(e) = self.solve_constraint_with_origin(constraint) {
                first_error.get_or_insert(e);
            }
        }
        if let Err(error) = self.settle_relations(false) {
            first_error.get_or_insert(error);
        }
        match first_error {
            Some(mut error) => {
                // Relations may resolve type arguments after equality has already
                // reported a mismatch. Diagnose the final types, including nominal
                // identities whose arguments were initially inference variables.
                if let SolveError::TypeMismatch { expected, actual } = &mut error.error {
                    *expected =
                        self.type_subst
                            .apply_with_rows(self.db, *expected, &self.row_subst);
                    *actual = self
                        .type_subst
                        .apply_with_rows(self.db, *actual, &self.row_subst);
                }
                Err(error)
            }
            None => Ok(()),
        }
    }

    /// Solve a single constraint while preserving any attached source origin.
    fn solve_constraint_with_origin(
        &mut self,
        constraint: Constraint<'db>,
    ) -> Result<(), LocatedSolveError<'db>> {
        match constraint {
            Constraint::RowRemoval(removal, origin) => {
                if !self
                    .pending_row_removals
                    .iter()
                    .any(|(old, _)| *old == removal)
                {
                    self.pending_row_removals.push((removal, origin));
                }
                Ok(())
            }
            Constraint::RowUnion(union, origin) => {
                if !self
                    .pending_row_unions
                    .iter()
                    .any(|(existing, _)| *existing == union)
                {
                    self.pending_row_unions.push((union, origin));
                }
                Ok(())
            }
            relation @ (Constraint::TypeCoerce(..) | Constraint::TypeJoin { .. }) => {
                self.pending_relations.push(relation);
                Ok(())
            }
            Constraint::TypeEq(t1, t2) => {
                self.unify_types(t1, t2).map_err(|error| LocatedSolveError {
                    error,
                    origin: None,
                })
            }
            Constraint::RowEq(r1, r2) => {
                self.unify_rows(r1, r2).map_err(|error| LocatedSolveError {
                    error,
                    origin: None,
                })
            }
            Constraint::TypeEqAt(t1, t2, origin) => {
                self.unify_types(t1, t2).map_err(|error| LocatedSolveError {
                    error,
                    origin: Some(origin),
                })
            }
            Constraint::RowEqAt(r1, r2, origin) => {
                self.unify_rows(r1, r2).map_err(|error| LocatedSolveError {
                    error,
                    origin: Some(origin),
                })
            }
            Constraint::And(cs) => {
                let mut first_error: Option<LocatedSolveError<'db>> = None;
                for c in cs {
                    if let Err(e) = self.solve_constraint_with_origin(c) {
                        first_error.get_or_insert(e);
                    }
                }
                match first_error {
                    Some(e) => Err(e),
                    None => Ok(()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod row_union_tests;
